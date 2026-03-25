"""Generate dataset samples via SiliconFlow multimodal API.

Reads images from plain-photos/ (JPEG/PNG/WebP/BMP or HEIC/HEIF), writes
<stem>.(jpg|png|...) + .prompt.txt + .ans.txt + .think.txt (reasoning, may be empty)
under dataset/.
HEIC/HEIF are converted to JPEG for the API and for dataset files.
Install HEIC support: pip install pillow-heif
Optional progress bar: pip install tqdm

API format follows: https://docs.siliconflow.cn/cn/userguide/capabilities/multimodal-vision

Environment:
  export SILICONFLOW_API_KEY="your-key"

Usage:
  python generate_dataset_siliconflow.py --dry-run
  python generate_dataset_siliconflow.py [--variants-per-class N]
  (Each image: N variants per drink class * 3 classes; default N=1 -> 3 samples/image.)
  Skips samples when {stem}.ans.txt already exists with non-empty text (use --overwrite).
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
import shutil
import sys
import time
from typing import Iterable

try:
    from openai import APIStatusError, OpenAI, RateLimitError
except ImportError as exc:
    raise SystemExit("Please install: pip install openai") from exc

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Please install: pip install pillow") from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]


DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-235B-A22B-Thinking"

# HEIC/HEIF need pillow-heif: pip install pillow-heif
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

# Multiple user-goal phrasings per drink (robustness).
USER_GOAL_VARIANTS: dict[str, list[str]] = {
    "sprite": [
        'I want to pick up Sprite drink.',
        "Please help me grab the Sprite bottle.",
        "Get me the Sprite, I'm trying to reach it.",
        "I need to take the Sprite from the table.",
        "Could you guide me to the Sprite?",
        "I'm blind—help me get the Sprite drink.",
    ],
    "cola": [
        "I want to pick up Cola drink.",
        "Please help me grab the cola (bottle or can).",
        "Get me the cola—I want to hold it.",
        "I need to take the cola drink in front of me.",
        "Guide my hand to the cola.",
        "I'm blind—help me reach the cola.",
    ],
    "lemon_tea": [
        "I want to pick up lemon tea drink.",
        "Please help me grab the lemon tea bottle.",
        "Get me the lemon tea, I can't see it.",
        "I need to take the lemon tea from the scene.",
        "Could you guide me to the lemon tea?",
        "I'm blind—help me get the lemon tea drink.",
    ],
}

# Number of drink classes (sprite / cola / lemon_tea).
NUM_DRINK_CLASSES = len(USER_GOAL_VARIANTS)


def build_instruction_prompt(user_goal: str) -> str:
    """Full English prompt for the API text part."""
    return f"""You are a blind-assistance vision simulator. Given the input image and the user's goal, you must guide them to grasp the target object (a beverage) using ONLY short, spoken-style English commands.

User goal: "{user_goal}"

Visual reference — match the user goal to the correct bottle/can by typical packaging color (use this to disambiguate when several drinks appear):
- Sprite drink: green packaging (green bottle/label).
- Lemon tea drink: yellow packaging (yellow bottle/label).
- Cola drink: red packaging (red bottle/can or red-dominant label).

Always guide toward the drink that matches BOTH the user goal AND this packaging cue.

Honesty (critical):
- If the target drink is NOT clearly visible in the image, output ONLY: object missing. Do NOT invent move left/right/forward commands or pretend the drink is there.
- If the user's hand is NOT clearly visible, output ONLY: show your hand. Do NOT give directional guidance until the hand can be seen.
- If you are unsure, treat it as not visible — say object missing or show your hand; never guess or hallucinate positions.
- Movement commands are ONLY allowed when BOTH the hand AND the correct target drink are clearly visible and you are making a small correction. Otherwise refuse with the fixed phrases above.

Frame of reference (critical — hand-centric, not object-centric):
- Every **move** command tells the user how to **move their hand** from the hand's **current** pose toward the **target drink**. Directions are **relative to the hand** (and the user's body): e.g. "move left" means shift the **hand** left from where it is now to close the gap to the drink — NOT "the drink is left of something else" and NOT instructions as if moving the bottle, the table, or any other object.
- **Do not** phrase guidance as position of the drink **relative to unrelated items** (other bottles, edges, image center). **Do not** confuse camera / frame / scene axes with "which way the hand should go" unless they match the user's egocentric hand movement.
- Infer the needed hand motion from **hand vs. target drink** only: where the hand must go next, in **forward/back, up/down, left/right** from the hand's current location, to align with the correct drink.

Directions — consider **depth**, **vertical**, and **horizontal** separately (always as **hand** motion toward the target). **Do not default to left/right** when depth or height is clearly wrong.

Multi-axis output (critical):
- In **one** line, output **every** direction that still needs a **meaningful** correction: include depth (forward/back/closer), vertical (up/down), and horizontal (left/right) **as needed**.
- **Omit an axis** if the error on that axis is **too subtle** (barely visible, negligible, or would only be a micro-nudge). Do not mention that direction for this turn.
- Order parts consistently when combining: **depth first**, then **vertical**, then **horizontal** (e.g. `move forward and move up`, or `move back and move right`). Join with **and** (or a comma if very short).
- If only **one** axis needs a non-subtle fix, output that single move phrase (same as before).
- Still avoid left/right **only** when depth or vertical errors are clearly larger — but if several axes are all clearly off, say **all** of them in that one line.

Vocabulary (short fragments; optional "a little bit" / "slightly" on any part):
- Depth: move forward, move back, move closer
- Vertical: move up, move down
- Horizontal: move left, move right

Grab rule (critical):
- Output grab now or hold it now ONLY when the hand and the target drink appear **fully aligned / overlapping** in the image (contact-ready, same spot — not merely "close"). If alignment is incomplete, keep giving directional fixes; do NOT grab early.

Rules:
1. Output **one line** per turn, no paragraphs or bullet lists. It may be **one** move phrase **or** several joined with **and** when multiple axes need non-subtle fixes (see "Multi-axis output"). Typical length about 2–12 words. No explanations.
2. Allowed tokens: move forward, move back, move closer, move up, move down, move left, move right, grab now, hold it now, show your hand, object missing, done — plus optional "a little bit" / "slightly" on relevant parts.
3. If the user's hand is not visible, output: show your hand
4. If the target drink for this user goal is not visible (including wrong color/packaging for that drink), output: object missing
5. When (and only when) the hand and the correct drink are fully overlapped / contact-ready per "Grab rule", output: grab now or hold it now (pick one consistently) — **never** combine grab with move commands on the same line.
6. Directional fixes only when "Honesty" allows and alignment is not yet full. List every **non-subtle** axis; skip axes that are too fine to matter this turn. Every move phrase must be **hand-centric** (how to move the **hand** from its current position toward the target), never a description of where something else is relative to a third object.
7. When the user has clearly completed the grasp after grab (hand holding the drink), output exactly: done
8. Do NOT describe the whole scene in detail. Do NOT add safety lectures. Only output the command line (or done).

请迅速思考，不要想太多。

Now, look at the image and output the single best next command line for this moment."""


def _open_image_rgb(path: str) -> Image.Image:
    """Open image; HEIC/HEIF requires pillow-heif."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError as exc:
            raise SystemExit(
                "HEIC/HEIF images need: pip install pillow-heif"
            ) from exc
    im = Image.open(path)
    return im.convert("RGB")


def image_to_data_url(path: str) -> str:
    """Build data URL for the chat API. HEIC is converted to JPEG in memory."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".heic", ".heif"):
        im = _open_image_rgb(path)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".png":
        mime = "image/png"
    elif ext == ".webp":
        mime = "image/webp"
    else:
        mime = "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def write_dataset_image(src_path: str, dst_stem: str, out_dir: str) -> str:
    """Copy or convert source into out_dir. HEIC -> .jpg; others keep format."""
    ext = os.path.splitext(src_path)[1].lower()
    if ext in (".heic", ".heif"):
        out_path = os.path.join(out_dir, f"{dst_stem}.jpg")
        _open_image_rgb(src_path).save(out_path, "JPEG", quality=92)
        return out_path
    if ext in (".jpg", ".jpeg"):
        out_path = os.path.join(out_dir, f"{dst_stem}.jpg")
        shutil.copy2(src_path, out_path)
        return out_path
    if ext == ".png":
        out_path = os.path.join(out_dir, f"{dst_stem}.png")
        shutil.copy2(src_path, out_path)
        return out_path
    if ext == ".webp":
        out_path = os.path.join(out_dir, f"{dst_stem}.webp")
        shutil.copy2(src_path, out_path)
        return out_path
    if ext == ".bmp":
        out_path = os.path.join(out_dir, f"{dst_stem}.bmp")
        shutil.copy2(src_path, out_path)
        return out_path
    out_path = os.path.join(out_dir, f"{dst_stem}.jpg")
    _open_image_rgb(src_path).save(out_path, "JPEG", quality=92)
    return out_path


def sanitize_stem(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name)
    return s.strip("._") or "sample"


def normalize_answer(text: str) -> str:
    """Keep first line; strip any leaked thinking; strip quotes.

    Thinking must not appear in dataset labels: remove common CoT wrappers, then
    take text after the last `</think>` if present.
    """
    t = text.strip()
    # Final answer is after the last `</think>` if the model inlined thinking in content.
    think_end = "`</think>`"
    if think_end in t:
        t = t.split(think_end)[-1].strip()
    line = t.splitlines()[0] if t else ""
    line = line.strip().strip('"').strip("'")
    return line


def total_samples_per_image(variants_per_class: int) -> int:
    """Each image generates at most variants_per_class * NUM_DRINK_CLASSES samples."""
    return variants_per_class * NUM_DRINK_CLASSES


class _DummyPbar:
    """No-op when tqdm is not installed."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix_str(self, s: str = "", *, refresh: bool = True) -> None:
        pass

    def close(self) -> None:
        pass


def make_progress_bar(total: int):
    """Return a tqdm bar or a dummy."""
    if tqdm is not None:
        return tqdm(
            total=total,
            desc="SiliconFlow",
            unit="req",
            dynamic_ncols=True,
            file=sys.stderr,
        )
    return _DummyPbar()


def log_line(msg: str) -> None:
    if tqdm is not None:
        tqdm.write(msg)
    else:
        print(msg)


def log_warn(msg: str) -> None:
    if tqdm is not None:
        tqdm.write(msg, file=sys.stderr)
    else:
        print(msg, file=sys.stderr)


def iter_image_files(photo_dir: str) -> Iterable[str]:
    for fn in sorted(os.listdir(photo_dir)):
        p = os.path.join(photo_dir, fn)
        if os.path.isfile(p) and os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
            yield p


def is_sample_labeled(out_dir: str, stem: str) -> bool:
    """True if `{stem}.ans.txt` exists and has non-whitespace label text."""
    ans_path = os.path.join(out_dir, f"{stem}.ans.txt")
    if not os.path.isfile(ans_path):
        return False
    try:
        with open(ans_path, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    return bool(text.strip())


def _is_rate_limit_error(exc: Exception) -> bool:
    """True for HTTP 429, TPM limits, and similar provider throttling."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and getattr(exc, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    if "429" in msg or "too many requests" in msg:
        return True
    if "rate limit" in msg or "tpm" in msg or "tokens per min" in msg:
        return True
    return False


def call_vlm(
    client: OpenAI,
    model: str,
    data_url: str,
    text_prompt: str,
    temperature: float,
    max_tokens: int,
    enable_thinking: bool,
    max_retries: int,
    retry_base_seconds: float,
    retry_wait_cap: float,
) -> tuple[str, str]:
    """Call VLM with optional chain-of-thought.

    Returns:
        (raw_content, reasoning_text). `reasoning_text` comes from the API's
        reasoning_content when present; save it to `{stem}.think.txt` separately.
        `raw_content` is normalized for `{stem}.ans.txt` only.

    Retries with exponential backoff on 429 / TPM / rate limits.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url, "detail": "high"},
                            },
                            {"type": "text", "text": text_prompt},
                        ],
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": enable_thinking},
                },
            )
            msg = response.choices[0].message
            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None)
            if reasoning is None:
                reasoning_str = ""
            elif isinstance(reasoning, str):
                reasoning_str = reasoning.strip()
            else:
                reasoning_str = str(reasoning).strip()
            return (content, reasoning_str)
        except Exception as exc:
            if not _is_rate_limit_error(exc):
                raise
            last_exc = exc
            if attempt >= max_retries - 1:
                break
            wait = min(retry_base_seconds * (2**attempt), retry_wait_cap)
            log_warn(
                f"Rate limited (429/TPM), attempt {attempt + 1}/{max_retries}, "
                f"sleep {wait:.1f}s..."
            )
            time.sleep(wait)
    raise RuntimeError(
        f"API rate limited after {max_retries} attempts (429/TPM)."
    ) from last_exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate dataset via SiliconFlow multimodal chat API"
    )
    parser.add_argument(
        "--photo_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "plain-photos"),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset"),
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help=(
            "Disable model chain-of-thought in the API. "
            "Default: thinking ON; only the final answer is saved (no reasoning in .ans.txt)."
        ),
    )
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.35,
        help="Seconds between API calls to reduce rate limits.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=10,
        help="Max attempts per request when API returns 429 / TPM rate limits.",
    )
    parser.add_argument(
        "--retry-base-seconds",
        type=float,
        default=60.0,
        help="Initial backoff (seconds) after 429/TPM; default 60s (1 min), doubles each retry until cap.",
    )
    parser.add_argument(
        "--retry-wait-cap",
        type=float,
        default=120.0,
        help="Max sleep seconds between retries on rate limit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned stems only; no API calls and no writes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-call API even when {stem}.ans.txt already exists with content.",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        default=0,
        help="Process only first N images (0 = all).",
    )
    parser.add_argument(
        "-n",
        "--variants-per-class",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Max user-goal variants per drink class (default: 1). "
            f"Classes={NUM_DRINK_CLASSES}, so each image yields at most N * {NUM_DRINK_CLASSES} API calls."
        ),
    )
    args = parser.parse_args()

    if args.variants_per_class < 1:
        print("--variants-per-class / -n must be >= 1", file=sys.stderr)
        sys.exit(1)
    if args.max_retries < 1:
        print("--max-retries must be >= 1", file=sys.stderr)
        sys.exit(1)

    enable_thinking = not args.no_thinking

    api_key = os.environ.get("SILICONFLOW_API_KEY") or os.environ.get(
        "OPENAI_API_KEY"
    )
    if not args.dry_run and not api_key:
        print(
            "Set SILICONFLOW_API_KEY (or OPENAI_API_KEY) for API access.",
            file=sys.stderr,
        )
        sys.exit(1)

    photo_dir = os.path.abspath(args.photo_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    images = list(iter_image_files(photo_dir))
    if args.limit_images > 0:
        images = images[: args.limit_images]

    if not images:
        print(f"No images found under {photo_dir}", file=sys.stderr)
        sys.exit(1)

    client = None if args.dry_run else OpenAI(api_key=api_key, base_url=args.base_url)

    n_var = args.variants_per_class
    total_tasks = len(images) * total_samples_per_image(n_var)
    pbar = make_progress_bar(total_tasks)

    planned = 0
    written = 0
    skipped_labeled = 0
    try:
        for img_path in images:
            base = sanitize_stem(os.path.splitext(os.path.basename(img_path))[0])
            data_url = None if args.dry_run else image_to_data_url(img_path)

            for drink_key, goals in USER_GOAL_VARIANTS.items():
                for vi, user_goal in enumerate(goals[:n_var], start=1):
                    stem = f"{base}__{drink_key}__v{vi:02d}"
                    prompt_text = build_instruction_prompt(user_goal)

                    prompt_path = os.path.join(out_dir, f"{stem}.prompt.txt")
                    ans_path = os.path.join(out_dir, f"{stem}.ans.txt")
                    planned += 1

                    already = (not args.overwrite) and is_sample_labeled(
                        out_dir, stem
                    )

                    if args.dry_run:
                        tag = "SKIP exists" if already else "would run"
                        log_line(
                            f"[dry-run] {tag} {stem} -> {user_goal[:50]}..."
                        )
                        pbar.update(1)
                        pbar.set_postfix_str(stem[:56], refresh=False)
                        continue

                    if already:
                        skipped_labeled += 1
                        log_line(f"skip (already labeled): {stem}")
                        pbar.update(1)
                        pbar.set_postfix_str(stem[:56], refresh=False)
                        continue

                    assert client is not None and data_url is not None
                    pbar.set_postfix_str(stem[:56], refresh=False)
                    raw_answer, reasoning_text = call_vlm(
                        client,
                        args.model,
                        data_url,
                        prompt_text,
                        args.temperature,
                        args.max_tokens,
                        enable_thinking=enable_thinking,
                        max_retries=args.max_retries,
                        retry_base_seconds=args.retry_base_seconds,
                        retry_wait_cap=args.retry_wait_cap,
                    )
                    answer = normalize_answer(raw_answer)
                    if not answer:
                        log_warn(f"[warn] empty answer for {stem}, skipping write")
                        pbar.update(1)
                        time.sleep(args.sleep)
                        continue

                    think_path = os.path.join(out_dir, f"{stem}.think.txt")
                    write_dataset_image(img_path, stem, out_dir)
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write(prompt_text)
                    with open(ans_path, "w", encoding="utf-8") as f:
                        f.write(answer)
                    with open(think_path, "w", encoding="utf-8") as f:
                        f.write(reasoning_text)

                    written += 1
                    log_line(f"OK {stem} -> {answer}")
                    pbar.update(1)
                    time.sleep(args.sleep)
    finally:
        pbar.close()

    if args.dry_run:
        log_line(
            f"Dry-run: {planned} sample slots ({len(images)} images). "
            f"Use --overwrite to ignore existing .ans.txt."
        )
    else:
        log_line(
            f"Done. wrote={written} skipped_labeled={skipped_labeled} "
            f"slots={planned} images={len(images)}."
        )


if __name__ == "__main__":
    main()
