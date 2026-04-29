from __future__ import annotations

import json
from pathlib import Path
import csv

import fiftyone as fo
import fiftyone.utils.huggingface as fouh


DATASET_NAME = "bdd100k_hf_probe_50"
HUB_NAME = "dgural/bdd100k"
SPLIT = "train"
MAX_SAMPLES = 50
OUT_DIR = Path("analysis/hf_bdd100k_probe")


def safe_to_dict(obj):
    try:
        return obj.to_dict()
    except Exception:
        return str(obj)


def summarize_label_obj(name, obj):
    print(f"\n--- FIELD: {name} ---")
    if obj is None:
        print("value: None")
        return

    print("python type:", type(obj).__name__)

    d = safe_to_dict(obj)
    if isinstance(d, dict):
        print("top-level keys:", list(d.keys())[:20])

        if "label" in d:
            print("label:", d["label"])

        if "mask_path" in d:
            print("mask_path:", d["mask_path"])

        if "mask" in d:
            mask = d["mask"]
            if mask is None:
                print("mask: None")
            else:
                print("mask: present (inline)")

        if "detections" in d and isinstance(d["detections"], list):
            print("num detections:", len(d["detections"]))

        if "polylines" in d and isinstance(d["polylines"], list):
            print("num polylines:", len(d["polylines"]))

        if "points" in d:
            try:
                print("points len:", len(d["points"]))
            except Exception:
                print("points: present")
    else:
        print("value:", d)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading small HF sample...")
    dataset = fouh.load_from_hub(
        HUB_NAME,
        split=SPLIT,
        max_samples=MAX_SAMPLES,
        name=DATASET_NAME,
    )

    print("\n=== DATASET BASIC INFO ===")
    print("dataset name:", dataset.name)
    print("num samples:", len(dataset))

    schema = dataset.get_field_schema()
    print("\n=== FIELD SCHEMA ===")
    for k, v in schema.items():
        print(f"{k}: {type(v).__name__}")

    candidate_fields = [
        "weather",
        "timeofday",
        "scene",
        "detections",
        "polylines",
        "drivable",
    ]

    summary_rows = []
    first_examples = {}

    for field in candidate_fields:
        present = 0
        first_obj = None

        for sample in dataset:
            value = sample[field] if field in sample else None
            if value is not None:
                present += 1
                if first_obj is None:
                    first_obj = value
                    first_examples[field] = {
                        "filepath": sample.filepath,
                        "object": safe_to_dict(value),
                    }

        summary_rows.append({
            "field": field,
            "present_count": present,
            "fraction_present": present / max(len(dataset), 1),
        })

        print(f"\n=== FIELD COVERAGE: {field} ===")
        print("present_count:", present, "/", len(dataset))

        summarize_label_obj(field, first_obj)

    with open(OUT_DIR / "field_examples.json", "w") as f:
        json.dump(first_examples, f, indent=2)

    with open(OUT_DIR / "field_coverage.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    manifest = []
    for sample in dataset:
        row = {
            "filepath": sample.filepath,
            "image_name": Path(sample.filepath).name,
        }
        for field in candidate_fields:
            row[f"has_{field}"] = sample[field] is not None if field in sample else False

            if field in ["weather", "timeofday", "scene"] and sample[field] is not None:
                try:
                    row[field] = sample[field].label
                except Exception:
                    row[field] = str(sample[field])
            else:
                row[field] = None

        manifest.append(row)

    with open(OUT_DIR / "sample_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    print("\nSaved:")
    print("-", OUT_DIR / "field_examples.json")
    print("-", OUT_DIR / "field_coverage.json")
    print("-", OUT_DIR / "sample_manifest.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()
