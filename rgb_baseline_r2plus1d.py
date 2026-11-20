"""
RGB baseline for the ISL lighting-robustness paper: R(2+1)D-18 pretrained on Kinetics-400.

Uses your existing manifests (isl_lighting_dataset/manifests/*.csv), the same lighting
conditions (x0.35, x1.0, x1.65 with clipping) and the same protocol as the skeleton model:
train on normal light only, select the checkpoint by validation accuracy, then test the same
checkpoint on normal, low-light and overexposed test videos.

Why this replaces isl_lighting_experiment.py as the RGB baseline
  * R(2+1)D-18 is a standard, widely cited video network, built for exactly this input
    (16 frames at 112x112), so reviewers cannot call it a weak or home-made baseline.
  * Frames are decoded once and cached, so an epoch takes minutes instead of ~40 minutes.
  * Backbone and new classifier get separate learning rates, with a warm-up, so the
    pretrained features are not destroyed early in training.

Usage
    python rgb_baseline_r2plus1d.py --mode cache      # decode + cache frames once
    python rgb_baseline_r2plus1d.py --mode train
    python rgb_baseline_r2plus1d.py --mode evaluate
    python rgb_baseline_r2plus1d.py --mode all         # all three in order

Smoke test (a few videos, 1 epoch):
    python rgb_baseline_r2plus1d.py --mode all --quick_run
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18

LIGHTING = {"normal": "Normal Light", "low_light": "Low Light", "overexposed": "Overexposed"}
RESIZE_HW = (128, 171)          # Kinetics preprocessing: resize, then 112x112 crop
CROP = 112
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


# ------------------------------------------------------------------ data
def load_manifest(data_root: Path, split: str, limit: int | None = None) -> list[dict]:
    with (data_root / "manifests" / f"{split}.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit else rows


def apply_lighting(frame, lighting: str):
    """Identical to isl_skeleton_lighting_experiment.apply_lighting."""
    if lighting == "normal":
        return frame
    if lighting == "low_light":
        return cv2.convertScaleAbs(frame, alpha=0.35, beta=0)
    if lighting == "overexposed":
        return cv2.convertScaleAbs(frame, alpha=1.65, beta=0)
    raise ValueError(lighting)


def safe_stem(video_path: str) -> str:
    p = Path(video_path)
    return f"{p.parent.name}_{p.stem}".replace(" ", "_").replace(".", "_")


def cache_path(args, split: str, lighting: str, row: dict) -> Path:
    return (Path(args.output_dir) / "frame_cache" / lighting / split / row["class_name"]
            / f"{safe_stem(row['video_path'])}.npy")


def decode_video(video_path: str, frames: int, lighting: str) -> np.ndarray:
    """Same frame sampling as your pipelines: np.linspace over the video, lighting applied
    to the full-resolution frame, then BGR->RGB and resize. Returns uint8 (T, 128, 171, 3)."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(0, n - 1), frames).astype(int) if n > 0 else np.arange(frames)
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(apply_lighting(frame, lighting), cv2.COLOR_BGR2RGB)
        out.append(cv2.resize(frame, (RESIZE_HW[1], RESIZE_HW[0]), interpolation=cv2.INTER_AREA))
    cap.release()
    if not out:
        out = [np.zeros((*RESIZE_HW, 3), np.uint8)]
    while len(out) < frames:
        out.append(out[-1].copy())
    return np.stack(out[:frames])


def build_cache(args) -> None:
    jobs = [("train", "normal"), ("val", "normal"),
            ("test", "normal"), ("test", "low_light"), ("test", "overexposed")]
    for split, lighting in jobs:
        rows = load_manifest(Path(args.data_root), split, args.limit(split))
        todo = [r for r in rows if not cache_path(args, split, lighting, r).exists()]
        print(f"cache {split}/{lighting}: {len(rows) - len(todo)} cached, {len(todo)} to decode")
        t0, bright = time.time(), []
        for k, r in enumerate(todo, 1):
            arr = decode_video(r["video_path"], args.frames, lighting)
            p = cache_path(args, split, lighting, r)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.save(p, arr)
            bright.append(float(arr.mean()))
            if k % 200 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} ({time.time() - t0:.0f}s)")
        if bright:
            print(f"  mean pixel value {np.mean(bright):.1f}")


class FrameCache(Dataset):
    def __init__(self, args, split: str, lighting: str, train: bool):
        self.rows = load_manifest(Path(args.data_root), split, args.limit(split))
        self.paths = [cache_path(args, split, lighting, r) for r in self.rows]
        missing = [p for p in self.paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} cached clips missing for {split}/{lighting}. Run --mode cache.")
        self.labels = [int(r["label_id"]) for r in self.rows]
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        v = np.load(self.paths[i])                                   # (T, 128, 171, 3) uint8
        H, W = v.shape[1:3]
        if self.train:                                               # same random crop for all frames
            y0, x0 = random.randint(0, H - CROP), random.randint(0, W - CROP)
        else:
            y0, x0 = (H - CROP) // 2, (W - CROP) // 2
        v = v[:, y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255.0
        v = (v - MEAN) / STD
        # No horizontal flip: it would swap the signer's dominant hand.
        return torch.from_numpy(v).permute(3, 0, 1, 2).contiguous(), torch.tensor(self.labels[i])


# ------------------------------------------------------------------ model / training
def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    model = r2plus1d_18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def class_weights(rows: list[dict], num_classes: int, device) -> torch.Tensor:
    """Same formula as the skeleton model."""
    counts = np.ones(num_classes, dtype=np.float32)
    for r in rows:
        counts[int(r["label_id"])] += 1
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        ps += logits.argmax(1).cpu().tolist()
        ys += y.tolist()
    return {"accuracy": accuracy_score(ys, ps),
            "precision": precision_score(ys, ps, average="macro", zero_division=0),
            "recall": recall_score(ys, ps, average="macro", zero_division=0),
            "f1": f1_score(ys, ps, average="macro", zero_division=0),
            "labels": ys, "preds": ps}


def num_classes_of(args) -> int:
    with (Path(args.data_root) / "metadata.json").open(encoding="utf-8") as f:
        return int(json.load(f)["num_classes"])


def train(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nc = num_classes_of(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = FrameCache(args, "train", "normal", train=True)
    val_ds = FrameCache(args, "val", "normal", train=False)
    g = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, generator=g, num_workers=args.num_workers,
                          pin_memory=device.type == "cuda", drop_last=len(train_ds) > args.batch_size)
    val_dl = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(nc, not args.no_pretrained).to(device)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr_backbone},
                             {"params": head, "lr": args.lr_head}], weight_decay=args.weight_decay)
    total = args.epochs * len(train_dl)
    warm = max(1, args.warmup_epochs * len(train_dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    crit = nn.CrossEntropyLoss(weight=class_weights(train_ds.rows, nc, device), label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    print(f"R(2+1)D-18 | pretrained={not args.no_pretrained} | device={device} | classes={nc} | "
          f"train={len(train_ds)} val={len(val_ds)} | epochs={args.epochs}")
    hist_path = out / "epoch_metrics.csv"
    if hist_path.exists():
        hist_path.unlink()
    best, chance = -1.0, 1.0 / nc
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += float(loss) * y.size(0)
            correct += int((logits.argmax(1) == y).sum())
            seen += y.size(0)
        val = evaluate(model, val_dl, device)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, "train_accuracy": correct / seen,
               "val_accuracy": val["accuracy"], "val_f1": val["f1"], "seconds": time.time() - t0}
        with hist_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if f.tell() == 0:
                w.writeheader()
            w.writerow(row)
        print(f"epoch {epoch:3d}: loss={row['train_loss']:.3f} train_acc={row['train_accuracy']:.3f} "
              f"val_acc={val['accuracy']:.3f} val_f1={val['f1']:.3f} ({row['seconds']:.0f}s)", flush=True)
        if epoch == 3 and row["train_accuracy"] < 5 * chance and not args.quick_run:
            print("  WARNING: training accuracy is still near chance after 3 epochs. "
                  "Check that the pretrained weights loaded and that the cache looks right.")
        if val["accuracy"] > best:
            best = val["accuracy"]
            torch.save({"model": model.state_dict(), "num_classes": nc, "epoch": epoch,
                        "best_val_accuracy": best,
                        "args": {k: v for k, v in vars(args).items() if not callable(v)}}, out / "best_r2plus1d.pth")
    print(f"best val accuracy {best:.4f}")


def mcnemar_p(a: np.ndarray, b: np.ndarray) -> float:
    n10, n01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = n10 + n01
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)


def evaluate_lighting(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    ck = torch.load(out / "best_r2plus1d.pth", map_location=device)
    model = build_model(ck["num_classes"], pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    res = {}
    for lt in LIGHTING:
        dl = DataLoader(FrameCache(args, "test", lt, train=False), args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
        res[lt] = evaluate(model, dl, device)
        with (out / f"predictions_{lt}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true_label_id", "pred_label_id"])
            w.writerows(zip(res[lt]["labels"], res[lt]["preds"]))
        print(f"{LIGHTING[lt]}: acc={100 * res[lt]['accuracy']:.2f}% macro-F1={res[lt]['f1']:.3f}")

    pre = "from scratch" if ck.get("args", {}).get("no_pretrained") else "Kinetics-400 pretrained"
    table = [{"Model": f"R(2+1)D-18 RGB ({pre})",
              **{f"{LIGHTING[lt]} acc (%)": f"{100 * res[lt]['accuracy']:.2f}" for lt in LIGHTING},
              **{f"{LIGHTING[lt]} F1": f"{res[lt]['f1']:.3f}" for lt in LIGHTING}}]

    # Matched comparison with the skeleton model, if its predictions are available
    sk_dir = Path(args.skeleton_dir)
    sk = {lt: sk_dir / f"predictions_{lt}.csv" for lt in LIGHTING}
    compare, head_to_head = [], []
    if all(p.exists() for p in sk.values()) and not args.quick_run:
        def load(p):
            with p.open(newline="", encoding="utf-8") as f:
                r = list(csv.DictReader(f))
            return np.array([int(x["true_label_id"]) for x in r]), np.array([int(x["pred_label_id"]) for x in r])
        sk_res = {}
        for lt in LIGHTING:
            y, p = load(sk[lt])
            if not np.array_equal(y, np.array(res[lt]["labels"])):
                print("  skeleton predictions are in a different clip order; skipped the matched comparison")
                sk_res = {}
                break
            sk_res[lt] = p == y
        if sk_res:
            table.append({"Model": "Skeleton Transformer (MediaPipe Holistic)",
                          **{f"{LIGHTING[lt]} acc (%)": f"{100 * sk_res[lt].mean():.2f}" for lt in LIGHTING},
                          **{f"{LIGHTING[lt]} F1": "" for lt in LIGHTING}})
            rgb_ok = {lt: np.array(res[lt]["preds"]) == np.array(res[lt]["labels"]) for lt in LIGHTING}
            for name, ok in [("R(2+1)D-18 RGB", rgb_ok), ("Skeleton Transformer", sk_res)]:
                n = ok["normal"].mean()
                row = {"Model": name}
                for lt in ("low_light", "overexposed"):
                    d = n - ok[lt].mean()
                    row[f"{LIGHTING[lt]} drop (pp)"] = f"{100 * d:.2f}"
                    row[f"{LIGHTING[lt]} relative drop (%)"] = f"{100 * d / n:.1f}" if n > 0 else "-"
                    row[f"McNemar p, normal vs {LIGHTING[lt]}"] = f"{mcnemar_p(ok['normal'], ok[lt]):.3g}"
                compare.append(row)
            for lt in LIGHTING:
                head_to_head.append({"Condition": LIGHTING[lt],
                                     "RGB acc (%)": f"{100 * rgb_ok[lt].mean():.2f}",
                                     "Skeleton acc (%)": f"{100 * sk_res[lt].mean():.2f}",
                                     "Clips only RGB got right": int(np.sum(rgb_ok[lt] & ~sk_res[lt])),
                                     "Clips only skeleton got right": int(np.sum(~rgb_ok[lt] & sk_res[lt])),
                                     "McNemar p": f"{mcnemar_p(rgb_ok[lt], sk_res[lt]):.3g}"})

    def md(rows):
        cols = list(rows[0])
        return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] +
                         ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows])
    text = "## RGB baseline under the three lighting conditions\n\n" + md(table)
    if compare:
        text += "\n\n## Accuracy drop relative to normal light (same test clips)\n\n" + md(compare)
        text += "\n\n## RGB vs skeleton, clip by clip, within each condition\n\n" + md(head_to_head)
    (out / "rgb_vs_skeleton_summary.md").write_text(text + "\n", encoding="utf-8")
    with (out / "rgb_all_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({lt: {k: v for k, v in m.items() if k not in ("labels", "preds")} for lt, m in res.items()}, f, indent=2)
    print("\n" + text + f"\n\nSaved {out / 'rgb_vs_skeleton_summary.md'}  <- send this file back")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["cache", "train", "evaluate", "all"], default="all")
    ap.add_argument("--data_root", default="isl_lighting_dataset")
    ap.add_argument("--output_dir", default="results_rgb_r2plus1d")
    ap.add_argument("--skeleton_dir", default="results_skeleton_lighting",
                    help="Folder with the skeleton model's predictions_*.csv, for the matched comparison.")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr_backbone", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--warmup_epochs", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true", help="Train from scratch (not recommended).")
    ap.add_argument("--quick_run", action="store_true", help="Smoke test: 24 train/12 eval videos, 1 epoch.")
    args = ap.parse_args()
    if args.quick_run:
        args.epochs = 1
        args.output_dir = args.output_dir + "_quick"
    args.limit = (lambda split: (24 if split == "train" else 12)) if args.quick_run else (lambda split: None)
    if args.mode in ("cache", "all"):
        build_cache(args)
    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("evaluate", "all"):
        evaluate_lighting(args)


if __name__ == "__main__":
    main()
