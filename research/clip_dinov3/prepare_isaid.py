"""Build a manifest for the published OVRSISBench iSAID train/val files."""

import argparse
import json
from pathlib import Path


CLASSES = [
    "ship", "storage tank", "baseball diamond", "tennis court", "basketball court",
    "ground track field", "bridge", "large vehicle", "small vehicle", "helicopter",
    "swimming pool", "roundabout", "soccer ball field", "plane", "harbor",
]
SPLIT_SIZES = {"train": 18076, "val": 6363}


def split_rows(root, split, count):
    folder = root / "iSAID_split" / split
    images = {p.name: p for p in (folder / "images").glob("*.png") if p.is_file()}
    masks = {p.name: p for p in (folder / "D2masks").glob("*.png") if p.is_file()}
    if images.keys() != masks.keys() or len(images) != count:
        raise ValueError(f"{split}: expected {count} matched PNG pairs, found "
                         f"{len(images)} images and {len(masks)} masks in {folder}")
    return [
        {"sample_id": images[name].stem, "file_name": str(images[name]),
         "sem_seg_file_name": str(masks[name])}
        for name in sorted(images)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    rows = {split: split_rows(root, split, count) for split, count in SPLIT_SIZES.items()}
    train_ids = {row["sample_id"] for row in rows["train"]}
    if train_ids.intersection(row["sample_id"] for row in rows["val"]):
        raise ValueError("The published train and val split must use disjoint sample IDs")
    manifest = {
        "version": 1, "ignore_label": 255, "train_classes": "classes.json",
        "test_classes": "classes.json", "seen_class_ids": list(range(len(CLASSES))),
        "split_mode": "OVSISBench published train/val", **rows,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, contents in (("classes.json", CLASSES), ("manifest.json", manifest)):
        with (args.output / name).open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(contents, indent=2) + "\n")


if __name__ == "__main__":
    main()
