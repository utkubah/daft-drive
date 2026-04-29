from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter


IMG_EXTS = {".jpg", ".jpeg", ".png"}
JSON_EXTS = {".json"}


def parse_args():
    p = argparse.ArgumentParser(description="Discover BDD100K dataset structure")
    p.add_argument("--root", type=str, required=True, help="Path to extracted BDD100K root")
    p.add_argument("--max_examples", type=int, default=10, help="How many sample paths to print")
    return p.parse_args()


def list_top_levels(root: Path):
    print("\n== TOP LEVEL ==")
    for p in sorted(root.iterdir()):
        kind = "DIR " if p.is_dir() else "FILE"
        print(f"{kind:4s}  {p.name}")


def walk_summary(root: Path, max_depth: int = 2):
    print("\n== DIRECTORY SUMMARY (up to depth 2) ==")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        depth = len(rel.parts)
        if depth > max_depth:
            continue
        prefix = "  " * (depth - 1)
        marker = "/" if p.is_dir() else ""
        print(f"{prefix}{rel}{marker}")


def collect_files(root: Path):
    images = []
    jsons = []
    other = Counter()

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in IMG_EXTS:
            images.append(p)
        elif ext in JSON_EXTS:
            jsons.append(p)
        else:
            other[ext] += 1

    return images, jsons, other


def try_preview_json(json_paths: list[Path], max_examples: int = 5):
    print("\n== JSON PREVIEW ==")
    shown = 0
    for p in json_paths[:max_examples]:
        print(f"\nFILE: {p}")
        try:
            with open(p, "r") as f:
                obj = json.load(f)

            if isinstance(obj, dict):
                print("type: dict")
                print("keys:", list(obj.keys())[:20])

                # common metadata hints
                for key in ["attributes", "labels", "name", "frames", "objects", "split"]:
                    if key in obj:
                        value = obj[key]
                        if isinstance(value, dict):
                            print(f"{key}: dict keys -> {list(value.keys())[:20]}")
                        elif isinstance(value, list):
                            print(f"{key}: list len -> {len(value)}")
                        else:
                            print(f"{key}: {value}")

            elif isinstance(obj, list):
                print("type: list")
                print("len:", len(obj))
                if len(obj) > 0 and isinstance(obj[0], dict):
                    print("first item keys:", list(obj[0].keys())[:20])
            else:
                print("type:", type(obj).__name__)

            shown += 1

        except Exception as e:
            print("FAILED TO READ:", e)

    if shown == 0:
        print("No JSON files previewed.")


def print_examples(title: str, paths: list[Path], max_examples: int):
    print(f"\n== {title} ({len(paths)}) ==")
    for p in paths[:max_examples]:
        print(p)


def main():
    args = parse_args()
    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    print(f"Dataset root: {root}")

    list_top_levels(root)
    walk_summary(root, max_depth=2)

    images, jsons, other = collect_files(root)

    print_examples("IMAGE FILES", images, args.max_examples)
    print_examples("JSON FILES", jsons, args.max_examples)

    print("\n== OTHER EXTENSIONS ==")
    if other:
        for ext, count in other.most_common(20):
            print(f"{ext or '<no_ext>'}: {count}")
    else:
        print("None")

    try_preview_json(jsons, max_examples=min(5, args.max_examples))


if __name__ == "__main__":
    main()