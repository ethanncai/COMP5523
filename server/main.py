"""HTTP API: POST image + prompt, return VLM text (non-streaming).

Run from repository root, with required base model and LoRA paths::

    python -m server.main --model-path /path/to/base --adapter-path /path/to/lora

Or set env and run uvicorn (``--model-path`` can be a Hub id, e.g.
``HuggingFaceTB/SmolVLM-256M-Instruct``)::

    export SMOL_MODEL_PATH=/path/to/base
    export SMOL_ADAPTER_PATH=/path/to/lora
    uvicorn server.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Allow ``import trainer`` when running as script or from non-root cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import uvicorn

from trainer.infer import infer_one, load_model_and_processor

# Populated in lifespan; read by route handlers.
_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.environ.get("SMOL_MODEL_PATH")
    adapter_path = os.environ.get("SMOL_ADAPTER_PATH")
    device = os.environ.get("SMOL_DEVICE") or None
    if not model_path or not adapter_path:
        raise RuntimeError(
            "SMOL_MODEL_PATH and SMOL_ADAPTER_PATH must be set before app startup."
        )
    model, processor, dev = load_model_and_processor(
        model_path,
        adapter_path=adapter_path,
        device=device,
    )
    _state["model"] = model
    _state["processor"] = processor
    _state["device"] = dev
    yield
    _state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="SmolVLM Inference",
    description="POST image + prompt, return assistant text (no streaming).",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/infer")
def infer(
    prompt: str = Form(..., description="User text prompt"),
    image: UploadFile = File(..., description="Input image"),
    max_new_tokens: int = Form(512),
) -> JSONResponse:
    """Run VLM on one image and return the decoded assistant reply."""
    if max_new_tokens < 1 or max_new_tokens > 8192:
        raise HTTPException(status_code=400, detail="max_new_tokens must be in [1, 8192]")

    raw_bytes = image.file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        pil = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    model = _state.get("model")
    processor = _state.get("processor")
    device = _state.get("device")
    if model is None or processor is None or device is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        assistant, raw = infer_one(
            model,
            processor,
            device,
            pil,
            prompt,
            max_new_tokens=max_new_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return JSONResponse(
        {
            "text": assistant,
            "raw": raw,
        }
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLM FastAPI server")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Base model: Hugging Face id or local directory (trainer infer --model_id).",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="PEFT LoRA adapter directory (required).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=("cuda", "mps", "cpu"),
        help="Force device; default: auto (omit this flag).",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def _normalize_model_path(model_path: str) -> str:
    """Use absolute path for local dirs; keep Hub ids unchanged."""
    if os.path.isdir(model_path) or os.path.isfile(model_path):
        return os.path.abspath(model_path)
    if os.path.exists(model_path):
        return os.path.abspath(model_path)
    return model_path


def main() -> None:
    args = _parse_args()
    os.environ["SMOL_MODEL_PATH"] = _normalize_model_path(args.model_path)
    os.environ["SMOL_ADAPTER_PATH"] = os.path.abspath(args.adapter_path)
    if args.device:
        os.environ["SMOL_DEVICE"] = args.device

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        factory=False,
        reload=False,
    )


if __name__ == "__main__":
    main()
