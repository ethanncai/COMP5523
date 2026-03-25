"""LoRA fine-tuning for SmolVLM (e.g. SmolVLM-256M-Instruct).

Expects dataset layout under --data_dir:
  <stem>.<image_ext>              e.g. IMG_6841__cola__v01.jpg
  <stem>.ans.txt                  assistant answer (plain text)

For ``--prompt-variant concise`` (default), user prompts are generated on the
fly from the stem's encoded drink class — no sidecar prompt files needed.

Use ``--prompt-variant full`` to read ``<stem>.prompt.txt`` from disk instead.
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Any

import torch
from concise_prompt_vocab import (
    CONCISE_PROMPT_HEADER,
    build_random_user_goal_line,
    parse_drink_key_from_stem,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _build_random_concise_prompt(drink_key: str, rng: random.Random | None = None) -> str:
    """Build a concise prompt with randomly sampled template and drink alias."""
    goal = build_random_user_goal_line(drink_key, rng=rng)
    return f'{CONCISE_PROMPT_HEADER}"{goal}"'


def load_samples(data_dir: str, prompt_variant: str = "concise") -> list[dict[str, Any]]:
    """Load paired image + answer files (and optionally prompt files).

    For ``concise`` variant, each sample carries a ``drink_key`` so the
    prompt can be randomly generated at collation time — giving every epoch
    a different phrasing for the same image.

    Args:
        data_dir: Directory containing images and sidecar text files.
        prompt_variant: ``concise`` stores ``drink_key`` for on-the-fly random
            prompt generation; ``full`` reads ``<stem>.prompt.txt``.
    """
    data_dir = os.path.abspath(data_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    samples: list[dict[str, Any]] = []

    for name in os.listdir(data_dir):
        stem, ext = os.path.splitext(name)
        if ext.lower() not in image_exts:
            continue
        ans_path = os.path.join(data_dir, f"{stem}.ans.txt")
        img_path = os.path.join(data_dir, name)
        if not os.path.isfile(ans_path):
            raise FileNotFoundError(f"Missing {stem}.ans.txt for image {name}.")

        sample: dict[str, Any] = {"answer": "", "image_path": img_path}

        if prompt_variant == "concise":
            drink_key = parse_drink_key_from_stem(stem)
            if drink_key is None:
                raise ValueError(
                    f"Stem {stem!r} does not match __<drink>__vNN pattern; "
                    "cannot determine drink class for concise prompt."
                )
            sample["drink_key"] = drink_key
        elif prompt_variant == "full":
            prompt_path = os.path.join(data_dir, f"{stem}.prompt.txt")
            if not os.path.isfile(prompt_path):
                raise FileNotFoundError(
                    f"Missing {stem}.prompt.txt for image {name}."
                )
            with open(prompt_path, encoding="utf-8") as f:
                sample["question"] = f.read().strip()
        else:
            raise ValueError(f"Unknown prompt variant: {prompt_variant!r}")

        with open(ans_path, encoding="utf-8") as f:
            sample["answer"] = f.read().strip()
        samples.append(sample)

    if not samples:
        raise ValueError(f"No valid samples found in {data_dir}")
    return samples


def get_image_token_id(tokenizer: Any) -> int:
    """Resolve <image> id (GPT2Tokenizer lacks additional_special_tokens_ids)."""
    token_id = tokenizer.convert_tokens_to_ids("<image>")
    unk = getattr(tokenizer, "unk_token_id", None)
    if token_id is None or (unk is not None and token_id == unk):
        raise ValueError('Tokenizer must define the "<image>" special token.')
    return int(token_id)


def build_lora_config(
    r: int,
    lora_alpha: int,
    lora_dropout: float,
) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        inference_mode=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA / QLoRA training for SmolVLM")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "dataset"),
        help="Folder with <stem>.jpg + <stem>.ans.txt (concise prompts generated on the fly)",
    )
    parser.add_argument(
        "--prompt-variant",
        type=str,
        choices=("concise", "full"),
        default="concise",
        help="concise: generate prompt from stem (no file needed); full: read <stem>.prompt.txt",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="HuggingFaceTB/SmolVLM-256M-Instruct",
    )
    parser.add_argument("--output_dir", type=str, default="./smolvlm-lora-out")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--qlora", action="store_true", help="4-bit QLoRA (needs bitsandbytes)")
    args = parser.parse_args()

    if args.qlora and not torch.cuda.is_available():
        raise RuntimeError("QLoRA (--qlora) requires CUDA and bitsandbytes.")

    dtype = pick_dtype()
    processor = AutoProcessor.from_pretrained(args.model_id)

    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model)
    elif torch.cuda.is_available():
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            dtype=dtype,
            device_map="auto",
        )
    else:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_id,
            dtype=dtype,
        ).to(dev)

    lora_config = build_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    samples = load_samples(args.data_dir, prompt_variant=args.prompt_variant)
    train_dataset = Dataset.from_list(samples)

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = tokenizer.pad_token_id
    image_token_id = get_image_token_id(tokenizer)

    prompt_variant = args.prompt_variant

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        texts_full: list[str] = []
        images: list[list[Image.Image]] = []
        prompt_lens: list[int] = []

        for ex in examples:
            img = Image.open(ex["image_path"]).convert("RGB")
            if prompt_variant == "concise":
                question = _build_random_concise_prompt(ex["drink_key"])
            else:
                question = ex["question"]
            user_msgs = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ]
            full_msgs = user_msgs + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": ex["answer"]}],
                }
            ]
            text_prompt = processor.apply_chat_template(
                user_msgs, add_generation_prompt=True
            )
            text_full = processor.apply_chat_template(
                full_msgs, add_generation_prompt=False
            )
            inp_p = processor(
                text=text_prompt.strip(),
                images=[img],
                return_tensors="pt",
            )
            pl = int(inp_p["input_ids"].shape[1])
            prompt_lens.append(pl)
            texts_full.append(text_full.strip())
            images.append([img])

        batch = processor(
            text=texts_full,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        labels = batch["input_ids"].clone()
        for i, pl in enumerate(prompt_lens):
            labels[i, :pl] = -100
        if pad_id is not None:
            labels[labels == pad_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    use_mps = bool(torch.backends.mps.is_available() and not torch.cuda.is_available())
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=(not use_bf16 and torch.cuda.is_available()) or use_mps,
        optim="adamw_torch",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        dataloader_pin_memory=not use_mps,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved adapter and processor to {args.output_dir}")


if __name__ == "__main__":
    main()
