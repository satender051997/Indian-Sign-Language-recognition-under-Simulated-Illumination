"""
Prepare the uploaded ISL video dataset for the lighting robustness experiment.

The dataset in this workspace is already grouped as:

    dataset/
      Adjectives/
        1. loud/
          MVI_5177.MOV
          ...
      Animals/
        ...

This script discovers every leaf folder that contains videos, treats that folder
as one sign class, and writes reproducible train/val/test manifests. By default
it does not copy the videos; the experiment reads the original files through the
manifest. Use --materialize if you also want a copied folder tree.

Example:
    python prepare_dataset.py --source_dir dataset --output_dir isl_lighting_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".mpeg", ".mpg", ".m4v"}
DEFAULT_IGNORED_DIRS = {"zips", "__macosx", ".git", "isl_lighting_dataset"}


def sanitise_class_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "class"


def class_name_for_video(source_dir: Path, video_path: Path) -> str:
    class_dir = video_path.parent
    rel_parts = class_dir.relative_to(source_dir).parts
    if len(rel_parts) >= 2:
        raw = f"{rel_parts[-2]}__{rel_parts[-1]}"
    else:
        raw = rel_parts[-1]
    return sanitise_class_name(raw)


def is_ignored(path: Path, source_dir: Path, ignored_dirs: set[str]) -> bool:
    try:
        rel_parts = path.relative_to(source_dir).parts
    except ValueError:
        return False
    return any(part.lower() in ignored_dirs for part in rel_parts)


def discover_videos(source_dir: Path, ignored_dirs: set[str]) -> dict[str, list[Path]]:
    class_to_videos: dict[str, list[Path]] = defaultdict(list)
    for path in source_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if is_ignored(path, source_dir, ignored_dirs):
            continue
        label = class_name_for_video(source_dir, path)
        class_to_videos[label].append(path.resolve())

    return {label: sorted(paths) for label, paths in sorted(class_to_videos.items())}


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    shuffled = items[:]
    random.shuffle(shuffled)
    n = len(shuffled)

    if n == 1:
        return {"train": shuffled, "val": [], "test": []}
    if n == 2:
        return {"train": shuffled[:1], "val": [], "test": shuffled[1:]}
    if n == 3:
        return {"train": shuffled[:1], "val": shuffled[1:2], "test": shuffled[2:]}

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_path", "label", "label_id", "class_name", "split"],
        )
        writer.writeheader()
        writer.writerows(rows)


def materialize_videos(output_dir: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        src = Path(row["video_path"])
        dst = (
            output_dir
            / "videos"
            / row["split"]
            / row["class_name"]
            / src.name
        )
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)


def prepare(
    source_dir: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_videos_per_class: int,
    max_classes: int | None,
    ignored_dirs: set[str],
    materialize: bool,
) -> None:
    random.seed(seed)
    source_path = Path(source_dir).resolve()
    output_path = Path(output_dir).resolve()

    class_to_videos = discover_videos(source_path, ignored_dirs)
    class_to_videos = {
        label: paths
        for label, paths in class_to_videos.items()
        if len(paths) >= min_videos_per_class
    }
    if max_classes:
        class_to_videos = dict(list(class_to_videos.items())[:max_classes])

    if not class_to_videos:
        raise RuntimeError(f"No video classes found in {source_path}")

    class_names = sorted(class_to_videos)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    split_rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    for class_name in class_names:
        split_map = split_items(class_to_videos[class_name], train_ratio, val_ratio)
        for split, videos in split_map.items():
            for video_path in videos:
                split_rows[split].append(
                    {
                        "video_path": str(video_path),
                        "label": class_name,
                        "label_id": str(class_to_idx[class_name]),
                        "class_name": class_name,
                        "split": split,
                    }
                )

    output_path.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, str]] = []
    for split in ["train", "val", "test"]:
        rows = split_rows[split]
        rows.sort(key=lambda r: (r["label_id"], r["video_path"]))
        write_manifest(output_path / "manifests" / f"{split}.csv", rows)
        all_rows.extend(rows)
    write_manifest(output_path / "manifest.csv", all_rows)

    metadata = {
        "source_dir": str(source_path),
        "num_classes": len(class_names),
        "num_videos": len(all_rows),
        "splits": {split: len(rows) for split, rows in split_rows.items()},
        "class_to_idx": class_to_idx,
    }
    with (output_path / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2)
    with (output_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if materialize:
        materialize_videos(output_path, all_rows)

    print("\nDataset preparation complete")
    print(f"  Source       : {source_path}")
    print(f"  Output       : {output_path}")
    print(f"  Classes      : {metadata['num_classes']}")
    print(f"  Videos       : {metadata['num_videos']}")
    print(f"  Train / Val / Test: {metadata['splits']['train']} / "
          f"{metadata['splits']['val']} / {metadata['splits']['test']}")
    print(f"  Manifest     : {output_path / 'manifest.csv'}")
    print("\nNext:")
    print(f"  python isl_lighting_experiment.py --data_root \"{output_path}\"")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ISL lighting experiment manifests.")
    parser.add_argument("--source_dir", default="dataset", help="Root folder containing the uploaded videos.")
    parser.add_argument("--output_dir", default="isl_lighting_dataset", help="Prepared dataset output folder.")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_videos_per_class", type=int, default=2)
    parser.add_argument("--max_classes", type=int, default=None, help="Optional smaller subset for pilot runs.")
    parser.add_argument("--ignore_dir", action="append", default=[], help="Directory name to ignore; can repeat.")
    parser.add_argument("--materialize", action="store_true", help="Copy videos into output_dir/videos.")
    args = parser.parse_args()

    ignored_dirs = {d.lower() for d in DEFAULT_IGNORED_DIRS}
    ignored_dirs.update(d.lower() for d in args.ignore_dir)

    prepare(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_videos_per_class=args.min_videos_per_class,
        max_classes=args.max_classes,
        ignored_dirs=ignored_dirs,
        materialize=args.materialize,
    )


if __name__ == "__main__":
    main()
