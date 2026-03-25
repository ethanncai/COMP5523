#!/usr/bin/env python3
"""Convert images under a directory to JPEG in place (delete originals).

Supports PNG, WebP, BMP, TIFF, HEIC/HEIF, etc. Existing .jpg files are left
unchanged. Files named *.jpeg are renamed to *.jpg without re-encoding.

HEIC/HEIF requires: pip install pillow-heif

Usage:
  python convert_images_to_jpg.py --dir ./plain-photos
  python convert_images_to_jpg.py --dir ./dataset --recursive --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit("Please install: pip install pillow") from exc

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

# Extensions we convert to .jpg (lossy save from RGB).
CONVERT_EXTS = frozenset(
    {".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
)
JPEG_EXTS = frozenset({".jpg", ".jpeg"})


def iter_image_files(root: Path, recursive: bool) -> list[Path]:
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file()]
    else:
        paths = [p for p in root.iterdir() if p.is_file()]
    return sorted(paths, key=lambda p: str(p).lower())


def convert_one(
    path: Path,
    quality: int,
    dry_run: bool,
) -> tuple[str, str | None]:
    """Process one file. Returns (status, message). status: skip|dry|ok|err."""
    suffix = path.suffix.lower()

    if suffix == ".jpg":
        return ("skip", f"already JPEG: {path}")

    if suffix == ".jpeg":
        dest = path.with_suffix(".jpg")
        if dest.exists() and dest.resolve() != path.resolve():
            return ("err", f"refuses overwrite: {dest}")
        if dry_run:
            return ("dry", f"{path} -> {dest} (rename)")
        path.rename(dest)
        return ("ok", f"renamed {path.name} -> {dest.name}")

    if suffix not in CONVERT_EXTS:
        return ("skip", None)

    dest = path.with_suffix(".jpg")
    if dest.exists() and dest.resolve() != path.resolve():
        return ("err", f"refuses overwrite: {dest} exists")

    if dry_run:
        return ("dry", f"{path} -> {dest}")

    try:
        image = Image.open(path)
        rgb = image.convert("RGB")
        rgb.save(dest, "JPEG", quality=quality, optimize=True)
    except OSError as exc:
        return ("err", f"{path}: {exc}")

    path.unlink()
    return ("ok", f"{path.name} -> {dest.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert images to JPEG in place (non-JPEG originals removed)."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="Root directory to scan (default: current directory).",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Include subdirectories.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=92,
        help="JPEG quality 1-95 (default: 92).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without changing files.",
    )
    args = parser.parse_args()

    if not 1 <= args.quality <= 95:
        print("--quality must be between 1 and 95", file=sys.stderr)
        sys.exit(1)

    root = args.dir.resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    counts = {"ok": 0, "dry": 0, "skip": 0, "err": 0}

    for path in iter_image_files(root, args.recursive):
        status, msg = convert_one(path, args.quality, args.dry_run)
        if status == "skip" and msg is None:
            continue
        if status == "skip":
            counts["skip"] += 1
            continue
        if status == "err":
            counts["err"] += 1
            print(f"ERROR: {msg}", file=sys.stderr)
            continue
        if status == "dry":
            counts["dry"] += 1
            print(msg)
            continue
        counts["ok"] += 1
        print(msg)

    print(
        f"Done. converted={counts['ok']} dry_run={counts['dry']} "
        f"skipped={counts['skip']} errors={counts['err']}"
    )


if __name__ == "__main__":
    main()
