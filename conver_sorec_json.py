#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Optional


def build_filename(image_root: str, image_name: str) -> str:
    return os.path.normpath(os.path.join(image_root, image_name))


def convert_item(item: Dict[str, Any], image_root: str) -> Optional[Dict[str, str]]:
    image_name = item.get("image")
    if not image_name:
        return None

    caption = item.get("ref")
    if caption is None:
        caption = item.get("caption")
    if caption is None:
        return None

    return {
        "filename": build_filename(image_root, image_name),
        "caption": caption,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert list-JSON with keys (image, ref, ...) into list-JSON with keys (filename, caption)."
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSON file (a list of dicts)")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file (a list of dicts)")
    parser.add_argument(
        "--image_root",
        default="/home/3/um02143/workspace/mmGDINO/data/SODA-D/rawData/Images",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=4,
        help="Indent size for output JSON (default: 4). Use 0 for minified.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list (e.g., [ {...}, {...} ])")

    out: List[Dict[str, str]] = []
    skipped = 0

    for item in data:
        if not isinstance(item, dict):
            skipped += 1
            continue
        converted = convert_item(item, args.image_root)
        if converted is None:
            skipped += 1
            continue
        out.append(converted)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    indent = None if args.indent == 0 else args.indent
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=indent)

    print(f"done: in={len(data)}, out={len(out)}, skipped={skipped}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
