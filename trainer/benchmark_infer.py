"""Benchmark repeated VLM inference latency (same as ``infer.py``, timed).

Defaults match ``memo.txt``: LoRA under ``smolvlm-lora-out``, cola image + prompt.

Run from anywhere::

    python benchmark_infer.py

Or from ``trainer/`` with memo-relative paths (overrides use your paths)::

    python benchmark_infer.py --repeats 50

Progress bars require ``pip install tqdm`` (warmup + benchmark loops). Use
``--no-progress`` to disable.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from typing import Any

import torch
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

_TRAINER_DIR = os.path.dirname(os.path.abspath(__file__))
if _TRAINER_DIR not in sys.path:
    sys.path.insert(0, _TRAINER_DIR)

from infer import infer_one, load_model_and_processor, load_prompt


def _sync_device(device: torch.device) -> None:
    """Wait for GPU work to finish so timing includes kernel time."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_one_infer(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> float:
    """Return wall time for one forward (seconds), with device sync."""
    _sync_device(device)
    t0 = time.perf_counter()
    infer_one(
        model,
        processor,
        device,
        image,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    _sync_device(device)
    return time.perf_counter() - t0


def main() -> None:
    _root = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description="Benchmark SmolVLM inference latency (repeat N times, report stats).",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=os.path.join(_root, "smolvlm-lora-out"),
        help="PEFT LoRA directory (default: trainer/smolvlm-lora-out)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(_root, "dataset", "IMG_7007__cola__v01.jpg"),
        help="Input image (default: memo dataset/IMG_7007__cola__v01.jpg)",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=os.path.join(_root, "cola_prompt_example.txt"),
        help="Prompt text file (default: trainer/cola_prompt_example.txt)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=50,
        help="Number of timed inference runs (default 50).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Untimed warmup runs before measuring (default 2).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cuda", "mps", "cpu"),
        help="Force device; default: auto.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar (if tqdm is installed).",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    image_path = os.path.abspath(args.image)
    prompt_path = os.path.abspath(args.prompt_file)
    adapter_path = os.path.abspath(args.adapter_path)

    if not os.path.isfile(image_path):
        raise SystemExit(f"Image not found: {image_path}")
    if not os.path.isfile(prompt_path):
        raise SystemExit(f"Prompt file not found: {prompt_path}")
    if not os.path.isdir(adapter_path):
        raise SystemExit(f"Adapter directory not found: {adapter_path}")

    prompt_text = load_prompt(prompt_path, None)
    pil = Image.open(image_path).convert("RGB")

    device_str = args.device or None
    model, processor, device = load_model_and_processor(
        args.model_id,
        adapter_path=adapter_path,
        device=device_str,
    )

    use_pbar = tqdm is not None and not args.no_progress
    warmup_iter = (
        tqdm(
            range(args.warmup),
            desc="Warmup",
            unit="run",
            leave=False,
            dynamic_ncols=True,
        )
        if use_pbar and args.warmup > 0
        else range(args.warmup)
    )
    for _ in warmup_iter:
        _time_one_infer(
            model,
            processor,
            device,
            pil,
            prompt_text,
            args.max_new_tokens,
        )

    times_s: list[float] = []
    bench_iter = (
        tqdm(
            range(args.repeats),
            desc="Benchmark",
            unit="run",
            dynamic_ncols=True,
        )
        if use_pbar
        else range(args.repeats)
    )
    for _ in bench_iter:
        dt = _time_one_infer(
            model,
            processor,
            device,
            pil,
            prompt_text,
            args.max_new_tokens,
        )
        times_s.append(dt)
        if use_pbar:
            bench_iter.set_postfix_str(f"last={dt * 1000:.1f}ms", refresh=True)

    times_ms = [t * 1000.0 for t in times_s]
    mean_ms = statistics.mean(times_ms)
    stdev_ms = statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0

    print(
        f"Device: {device} | repeats={args.repeats} warmup={args.warmup} "
        f"max_new_tokens={args.max_new_tokens}"
    )
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt_path}")
    print(
        f"Mean: {mean_ms:.2f} ms (per run)\n"
        f"Std:  {stdev_ms:.2f} ms\n"
        f"Min:  {min(times_ms):.2f} ms\n"
        f"Max:  {max(times_ms):.2f} ms\n"
        f"Median: {statistics.median(times_ms):.2f} ms"
    )


if __name__ == "__main__":
    main()
