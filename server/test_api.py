"""Integration test client for the FastAPI inference server.

Requires a **running** server (see ``server/API.md``). Example::

    python server/test_api.py --image /path/to/photo.jpg

Environment: install ``requests`` (listed in ``requirements.txt``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call GET /health and POST /infer on a SmolVLM inference server.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="Server root URL (no trailing slash).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Image file path for POST /infer. Required unless --health-only.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe briefly what you see and one short spoken command.",
        help="Text prompt for /infer.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Form field max_new_tokens (default 512).",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only request GET /health; skip /infer.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds (default 300).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_url.rstrip("/")
    timeout = args.timeout

    health_url = f"{base}/health"
    try:
        r = requests.get(health_url, timeout=min(timeout, 30.0))
    except requests.RequestException as exc:
        print(f"GET {health_url} failed: {exc}", file=sys.stderr)
        return 1

    if r.status_code != 200:
        print(f"GET {health_url} -> {r.status_code}: {r.text}", file=sys.stderr)
        return 1

    try:
        health_body: dict[str, Any] = r.json()
    except json.JSONDecodeError:
        print(f"GET {health_url}: invalid JSON: {r.text!r}", file=sys.stderr)
        return 1

    print("GET /health OK:", json.dumps(health_body, ensure_ascii=False))

    if args.health_only:
        return 0

    if not args.image:
        print("--image is required unless --health-only", file=sys.stderr)
        return 2

    infer_url = f"{base}/infer"
    try:
        with open(args.image, "rb") as f:
            files = {
                "image": (
                    os.path.basename(args.image),
                    f,
                    "application/octet-stream",
                )
            }
            data = {
                "prompt": args.prompt,
                "max_new_tokens": str(args.max_new_tokens),
            }
            r2 = requests.post(infer_url, files=files, data=data, timeout=timeout)
    except OSError as exc:
        print(f"Cannot read image {args.image!r}: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"POST {infer_url} failed: {exc}", file=sys.stderr)
        return 1

    print(f"POST /infer -> HTTP {r2.status_code}")
    try:
        body = r2.json()
    except json.JSONDecodeError:
        print(r2.text, file=sys.stderr)
        return 1

    print(json.dumps(body, ensure_ascii=False, indent=2))
    if r2.status_code != 200:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
