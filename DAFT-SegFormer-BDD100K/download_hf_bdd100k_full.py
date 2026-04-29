from __future__ import annotations

from pathlib import Path
from collections import Counter
from huggingface_hub import snapshot_download


REPO_ID = "dgural/bdd100k"
LOCAL_DIR = Path.home() / "datasets" / "hf_bdd100k_full_repo"


def main():
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading dataset repo...")
    print("repo_id   :", REPO_ID)
    print("local_dir :", LOCAL_DIR)

    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
        max_workers=4,
        resume_download=True,
    )

    print("\nDownload finished.")
    print("saved_to:", path)

    root = Path(path)

    print("\n=== TOP LEVEL ===")
    for p in sorted(root.iterdir()):
        kind = "DIR " if p.is_dir() else "FILE"
        print(kind, p.name)

    print("\n=== EXTENSION COUNTS ===")
    ext_counter = Counter(
        p.suffix.lower() for p in root.rglob("*") if p.is_file()
    )
    for ext, count in ext_counter.most_common():
        print(repr(ext), count)

    print("\n=== IMPORTANT PATHS ===")
    candidates = [
        "train",
        "val",
        "validation",
        "test",
        "fields",
        "train/data",
        "train/fields",
        "train/fields/drivable",
    ]
    for rel in candidates:
        p = root / rel
        print(f"{rel}: {'exists' if p.exists() else 'missing'}")

    print("\n=== SAMPLE JPG FILES ===")
    jpgs = list(root.rglob("*.jpg"))[:20]
    for p in jpgs:
        print(p)

    print("\n=== SAMPLE PNG FILES ===")
    pngs = list(root.rglob("*.png"))[:20]
    for p in pngs:
        print(p)

    print("\n=== SAMPLE JSON FILES ===")
    jsons = list(root.rglob("*.json"))[:20]
    for p in jsons:
        print(p)

    print("\nDone.")


if __name__ == "__main__":
    main()