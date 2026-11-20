"""
Find out which source videos are NOT in the manifest, and why.

Run it from the folder that contains prepare_dataset.py and isl_lighting_dataset/:
    python check_excluded_videos.py

It uses the same discovery rules as prepare_dataset.py (it imports them), reads the
source folder recorded in isl_lighting_dataset/metadata.json, and reports:
  * every video file in the source folder that is missing from the manifest, with the reason
    (ignored folder, class with fewer than 2 videos, or other);
  * every video in the manifest that OpenCV cannot read (these were NOT excluded);
  * non-video files, in case some videos have an unexpected extension.
Results are printed and saved to isl_lighting_dataset/excluded_videos_report.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from prepare_dataset import DEFAULT_IGNORED_DIRS, VIDEO_EXTENSIONS, class_name_for_video, is_ignored


def readable(path: Path) -> tuple[bool, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return False, 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    return bool(ok) and n > 0, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="isl_lighting_dataset")
    ap.add_argument("--source_dir", default=None, help="Defaults to the source_dir saved in metadata.json")
    ap.add_argument("--min_videos_per_class", type=int, default=2, help="Same default as prepare_dataset.py")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    meta = json.loads((data_root / "metadata.json").read_text(encoding="utf-8"))
    source = Path(args.source_dir or meta["source_dir"]).resolve()
    print(f"Source folder : {source}")
    print(f"Manifest      : {data_root / 'manifest.csv'}  ({meta['num_videos']} videos, {meta['num_classes']} classes)\n")

    with (data_root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    in_manifest = {str(Path(r["video_path"]).resolve()).lower() for r in manifest}

    ignored = {d.lower() for d in DEFAULT_IGNORED_DIRS}
    all_videos, other_files = [], Counter()
    for p in source.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in VIDEO_EXTENSIONS:
            all_videos.append(p.resolve())
        else:
            other_files[p.suffix.lower() or "(no extension)"] += 1

    # class sizes, counted exactly as prepare_dataset.py does (ignored folders excluded)
    class_videos = defaultdict(list)
    for p in all_videos:
        if not is_ignored(p, source, ignored):
            class_videos[class_name_for_video(source, p)].append(p)

    rows = []
    for p in sorted(all_videos):
        if str(p).lower() in in_manifest:
            continue
        if is_ignored(p, source, ignored):
            reason = "inside an ignored folder (e.g. __MACOSX, zips)"
        else:
            cls = class_name_for_video(source, p)
            n = len(class_videos[cls])
            reason = (f"class '{cls}' has only {n} video(s) (< {args.min_videos_per_class})"
                      if n < args.min_videos_per_class else "other (check manually)")
        ok, frames = readable(p)
        rows.append({"status": "NOT in manifest", "video_path": str(p), "reason": reason,
                     "readable": ok, "frame_count": frames})

    print("Checking that every manifest video can be read (this takes a few minutes)...")
    for r in manifest:
        ok, frames = readable(Path(r["video_path"]))
        if not ok:
            rows.append({"status": "IN manifest but unreadable", "video_path": r["video_path"],
                         "reason": "OpenCV cannot read it or reports 0 frames", "readable": ok, "frame_count": frames})

    not_in = [r for r in rows if r["status"] == "NOT in manifest"]
    bad_in = [r for r in rows if r["status"] != "NOT in manifest"]
    hidden = [p for p in all_videos if p.name.startswith("._")]

    print("\n================ RESULT ================")
    print(f"Video files found in source folder : {len(all_videos)}")
    print(f"Videos in manifest                 : {len(manifest)}")
    print(f"Video files NOT in manifest        : {len(not_in)}")
    for reason, n in Counter(r["reason"] for r in not_in).most_common():
        print(f"    {n:4d}  {reason}")
    print(f"Manifest videos that are unreadable: {len(bad_in)}")
    if hidden:
        print(f"Hidden '._' macOS files counted as videos: {len(hidden)}")
    if other_files:
        print("Other (non-video) files by extension: " + ", ".join(f"{k}: {v}" for k, v in other_files.most_common(10)))
    print("\nFor reference, the published INCLUDE dataset has 4,287 videos.")

    out = data_root / "excluded_videos_report.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["status", "video_path", "reason", "readable", "frame_count"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}  <- send this file (and the RESULT block above)")


if __name__ == "__main__":
    main()
