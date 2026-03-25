"""Load SmolVLM + optional LoRA adapter and run a single image + prompt."""

from __future__ import annotations

import argparse
import os

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

    device_str = pick_device()
    device = torch.device(device_str)
    dtype = pick_dtype(device_str)

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        dtype=dtype,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    user_text = load_prompt(args.prompt_file, args.prompt)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    batch = processor(
        text=[text],
        images=[[image]],
        return_tensors="pt",
        padding=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=args.max_new_tokens)

    raw = processor.batch_decode(generated, skip_special_tokens=True)[0]
    print(extract_assistant(raw))


if __name__ == "__main__":
    main()
