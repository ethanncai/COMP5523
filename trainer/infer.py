"""Load SmolVLM + optional LoRA adapter and run a single image + prompt."""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_prompt(path: str | None, inline: str | None) -> str:
    if inline is not None:
        return inline.strip()
    if path is None:
        raise ValueError("Provide --prompt or --prompt_file")
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def extract_assistant(response: str) -> str:
    marker = "Assistant:"
    idx = response.find(marker)
    if idx == -1:
        return response.strip()
    return response[idx + len(marker) :].strip()


def load_model_and_processor(
    model_id: str,
    adapter_path: str | None = None,
    device: str | None = None,
) -> tuple[Any, Any, torch.device]:
    """Load base weights (local path or Hub id), optionally apply LoRA, eval mode.

    Args:
        model_id: Hugging Face model id or local directory with base weights.
        adapter_path: Directory with PEFT LoRA adapter; ``None`` for base-only.
        device: ``"cuda"`` / ``"mps"`` / ``"cpu"``; auto if ``None``.

    Returns:
        ``(model, processor, torch.device)``.
    """
    device_str = device if device is not None else pick_device()
    dev = torch.device(device_str)
    dtype = pick_dtype(device_str)

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=dtype,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model = model.to(dev)
    model.eval()
    return model, processor, dev


def infer_one(
    model: Any,
    processor: Any,
    device: torch.device,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
) -> tuple[str, str]:
    """Run one VLM forward pass. Returns ``(assistant_text, raw_decoded)``."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt.strip()},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True).strip()
    batch = processor(
        text=[text],
        images=[[image]],
        return_tensors="pt",
        padding=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=max_new_tokens)

    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return extract_assistant(raw), raw


def main() -> None:
    parser = argparse.ArgumentParser(description="SmolVLM inference with optional LoRA")
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help=(
            "Optional: directory with PEFT LoRA adapter (train.py --output_dir). "
            "If omitted, inference uses only base weights from --model_id."
        ),
    )
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    model, processor, device = load_model_and_processor(
        args.model_id,
        adapter_path=args.adapter_path,
    )

    image = Image.open(args.image).convert("RGB")
    user_text = load_prompt(args.prompt_file, args.prompt)

    assistant, _ = infer_one(
        model,
        processor,
        device,
        image,
        user_text,
        max_new_tokens=args.max_new_tokens,
    )
    print(assistant)


if __name__ == "__main__":
    main()
