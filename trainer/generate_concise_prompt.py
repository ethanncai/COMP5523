"""Write concise training prompts for dataset stems that encode a drink class.

Scans a folder for images; for each ``<stem>.jpg`` (or similar), if the stem
matches ``...__<sprite|cola|lemon_tea>__vNN``, builds a short prompt from
``concise_prompt_vocab`` and writes a text file next to the image.

Default output: ``<stem>.prompt.concise.txt``. Use ``--replace-main`` to
overwrite ``<stem>.prompt.txt`` for LoRA training with the concise version.
"""

from __future__ import annotations

import argparse
import os
import sys

from concise_prompt_vocab import (
    CONCISE_PROMPT_HEADER,
    build_user_goal_line,
    parse_drink_key_from_stem,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_image_stems(data_dir: str) -> list[str]:
    """Return sorted unique stems that have an image extension under data_dir."""
    data_dir = os.path.abspath(data_dir)
    stems: set[str] = set()
    for name in os.listdir(data_dir):
        _, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        stem, _ = os.path.splitext(name)
        stems.add(stem)
    return sorted(stems)


def build_full_concise_prompt(stem: str) -> tuple[str, str] | None:
    """Return (drink_key, full_prompt_text) or None if stem is not encoded."""
    drink_key = parse_drink_key_from_stem(stem)
    if drink_key is None:
        return None
    goal = build_user_goal_line(stem)
    if goal is None:
        return None
    text = f'{CONCISE_PROMPT_HEADER}"{goal}"'
    return drink_key, text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate concise .prompt files from dataset image stems.",
    )
    parser.add_argument(
        "dataset",
        type=str,
        help="Directory containing <stem>.jpg and optional <stem>.prompt.txt",
    )
    parser.add_argument(
        "--replace-main",
        action="store_true",
        help="Write to <stem>.prompt.txt instead of <stem>.prompt.concise.txt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not write files",
    )
    args = parser.parse_args()

    data_dir = os.path.abspath(args.dataset)
    if not os.path.isdir(data_dir):
        print(f"Not a directory: {data_dir}", file=sys.stderr)
        sys.exit(1)

    stems = iter_image_stems(data_dir)
    if not stems:
        print(f"No images found under {data_dir}", file=sys.stderr)
        sys.exit(1)

    suffix = ".prompt.txt" if args.replace_main else ".prompt.concise.txt"
    written = 0
    skipped = 0

    for stem in stems:
        built = build_full_concise_prompt(stem)
        if built is None:
            skipped += 1
            if args.dry_run:
                print(f"[skip] {stem} (no __<drink>__vNN pattern)")
            continue
        drink_key, body = built
        out_path = os.path.join(data_dir, f"{stem}{suffix}")
        if args.dry_run:
            print(f"[write] {out_path} class={drink_key}")
            written += 1
            continue
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(body)
        written += 1
        print(f"Wrote {out_path} ({drink_key})")

    print(
        f"Done. written={written} skipped_no_pattern={skipped} dir={data_dir}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
