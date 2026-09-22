"""
Skeleton-based Transformer lighting robustness experiment for ISL videos.

This is the correct experiment for the stated claim:
    Skeleton-based recognition should be more robust to lighting than RGB
    appearance recognition, but landmark extraction can still degrade when
    illumination changes.

Pipeline:
    1. Apply lighting condition to video frames.
    2. Extract pose + hand landmarks with MediaPipe Holistic.
    3. Train a Transformer on normal-light skeleton sequences.
    4. Evaluate the same checkpoint on normal, low-light, and overexposed
       skeleton sequences.

Conference table:
    Model | Normal Light | Low Light | Overexposed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than 50% of the number of samples.*",
)
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype.*")


LIGHTING = {
    "normal": "Normal Light",
    "low_light": "Low Light",
    "overexposed": "Overexposed",
}

POSE_IDS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
HAND_IDS = list(range(21))
NUM_JOINTS = len(POSE_IDS) + 2 * len(HAND_IDS)
COORD_DIM = 4  # x, y, z, visibility/presence


@dataclass
class Config:
    data_root: str = "isl_lighting_dataset"
    output_dir: str = "results_skeleton_lighting"
    cache_dir: str = "skeleton_cache"
    model_name: str = "Skeleton Transformer"
    seed: int = 42
    frames: int = 32
    batch_size: int = 32
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    d_model: int = 192
    nhead: int = 8
    num_layers: int = 4
    dropout: float = 0.2
    num_workers: int = 0
    progress_every: int = 25
    max_train_samples: int | None = None
    max_eval_samples: int | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest(data_root: Path, split: str) -> list[dict[str, str]]:
    with (data_root / "manifests" / f"{split}.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_num_classes(data_root: Path) -> int:
    with (data_root / "metadata.json").open(encoding="utf-8") as f:
        return int(json.load(f)["num_classes"])


def apply_lighting(frame, lighting: str):
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


def skeleton_path(cfg: Config, split: str, lighting: str, row: dict[str, str]) -> Path:
    return (
        Path(cfg.output_dir)
        / cfg.cache_dir
        / lighting
        / split
        / row["class_name"]
        / f"{safe_stem(row['video_path'])}.npz"
    )


def landmarks_to_array(results) -> tuple[np.ndarray, np.ndarray]:
    joints = np.zeros((NUM_JOINTS, COORD_DIM), dtype=np.float32)
    mask = np.zeros((NUM_JOINTS,), dtype=np.float32)
    idx = 0

    if results.pose_landmarks:
        for jid in POSE_IDS:
            lm = results.pose_landmarks.landmark[jid]
            joints[idx] = [lm.x, lm.y, lm.z, lm.visibility]
            mask[idx] = 1.0
            idx += 1
    else:
        idx += len(POSE_IDS)

    for hand_landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_landmarks:
            for jid in HAND_IDS:
                lm = hand_landmarks.landmark[jid]
                joints[idx] = [lm.x, lm.y, lm.z, 1.0]
                mask[idx] = 1.0
                idx += 1
        else:
            idx += len(HAND_IDS)

    return joints, mask


def normalize_sequence(seq: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Center around shoulder midpoint when available, otherwise all visible joints.
    coords = seq[..., :3].copy()
    for t in range(coords.shape[0]):
        visible = mask[t] > 0
        if visible.any():
            center = coords[t, visible].mean(axis=0, keepdims=True)
            coords[t] -= center
            scale = np.linalg.norm(coords[t, visible, :2], axis=1).max()
            if scale > 1e-6:
                coords[t] /= scale
    return np.concatenate([coords, seq[..., 3:4]], axis=-1).astype(np.float32)


def extract_video_skeleton(video_path: str, lighting: str, frames: int) -> tuple[np.ndarray, np.ndarray, dict]:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames > 0:
        indices = np.linspace(0, max(0, total_frames - 1), frames).astype(int)
    else:
        indices = np.arange(frames)

    seq = np.zeros((frames, NUM_JOINTS, COORD_DIM), dtype=np.float32)
    mask = np.zeros((frames, NUM_JOINTS), dtype=np.float32)
    brightness = []

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    try:
        for out_idx, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = apply_lighting(frame, lighting)
            brightness.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            seq[out_idx], mask[out_idx] = landmarks_to_array(results)
    finally:
        holistic.close()
        cap.release()

    seq = normalize_sequence(seq, mask)
    stats = {
        "frame_detection_rate": float((mask.sum(axis=1) > 0).mean()),
        "joint_detection_rate": float(mask.mean()),
        "mean_brightness": float(np.mean(brightness)) if brightness else 0.0,
    }
    return seq, mask, stats


def extract_split(cfg: Config, split: str, lighting: str, limit: int | None = None) -> None:
    rows = load_manifest(Path(cfg.data_root), split)
    if limit:
        rows = rows[:limit]
    stats_rows = []

    cached = 0
    for row in tqdm(rows, desc=f"extract {split} {lighting}", dynamic_ncols=True):
        out = skeleton_path(cfg, split, lighting, row)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            cached += 1
            continue
        seq, mask, stats = extract_video_skeleton(row["video_path"], lighting, cfg.frames)
        np.savez_compressed(out, skeleton=seq, mask=mask, label_id=int(row["label_id"]))
        stats_rows.append({"split": split, "lighting": lighting, "class_name": row["class_name"], **stats})

    if stats_rows:
        stats_path = Path(cfg.output_dir) / f"extraction_stats_{split}_{lighting}.csv"
        with stats_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
            writer.writeheader()
            writer.writerows(stats_rows)
        summarize_extraction_stats(stats_path)
    elif cached:
        print(f"extract {split} {lighting}: {cached} cached skeleton files already available")


def summarize_extraction_stats(stats_path: Path) -> None:
    rows = []
    with stats_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    frame_rate = np.mean([float(r["frame_detection_rate"]) for r in rows])
    joint_rate = np.mean([float(r["joint_detection_rate"]) for r in rows])
    brightness = np.mean([float(r["mean_brightness"]) for r in rows])
    print(
        f"  {stats_path.name}: frame_detect={frame_rate:.3f} "
        f"joint_detect={joint_rate:.3f} brightness={brightness:.1f}"
    )


def extract_required(cfg: Config, full_test_lighting: bool = True) -> None:
    extract_split(cfg, "train", "normal", cfg.max_train_samples)
    extract_split(cfg, "val", "normal", cfg.max_eval_samples)
    test_lightings = ["normal", "low_light", "overexposed"] if full_test_lighting else ["normal"]
    for lighting in test_lightings:
        extract_split(cfg, "test", lighting, cfg.max_eval_samples)


class SkeletonDataset(Dataset):
    def __init__(self, cfg: Config, split: str, lighting: str) -> None:
        rows = load_manifest(Path(cfg.data_root), split)
        if split == "train" and cfg.max_train_samples:
            rows = rows[: cfg.max_train_samples]
        if split != "train" and cfg.max_eval_samples:
            rows = rows[: cfg.max_eval_samples]
        self.samples = [(skeleton_path(cfg, split, lighting, row), int(row["label_id"])) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        if not path.exists():
            raise FileNotFoundError(f"Missing skeleton cache: {path}\nRun --mode extract first.")
        data = np.load(path)
        x = torch.from_numpy(data["skeleton"].astype(np.float32)).flatten(1)
        return x, torch.tensor(label, dtype=torch.long)


def count_missing_cache(cfg: Config, split: str, lighting: str) -> tuple[int, int]:
    rows = load_manifest(Path(cfg.data_root), split)
    if split == "train" and cfg.max_train_samples:
        rows = rows[: cfg.max_train_samples]
    if split != "train" and cfg.max_eval_samples:
        rows = rows[: cfg.max_eval_samples]
    missing = sum(1 for row in rows if not skeleton_path(cfg, split, lighting, row).exists())
    return missing, len(rows)


def ensure_cache(cfg: Config, requirements: list[tuple[str, str]]) -> None:
    for split, lighting in requirements:
        missing, total = count_missing_cache(cfg, split, lighting)
        if missing:
            print(
                f"Missing {missing}/{total} skeleton files for split='{split}', lighting='{lighting}'. "
                "Extracting now..."
            )
            limit = cfg.max_train_samples if split == "train" else cfg.max_eval_samples
            extract_split(cfg, split, lighting, limit)
        else:
            print(f"Skeleton cache ready: split='{split}', lighting='{lighting}' ({total} files)")


class SkeletonTransformer(nn.Module):
    def __init__(self, num_classes: int, cfg: Config) -> None:
        super().__init__()
        in_dim = NUM_JOINTS * COORD_DIM
        self.input_proj = nn.Sequential(nn.Linear(in_dim, cfg.d_model), nn.LayerNorm(cfg.d_model), nn.GELU())
        self.cls = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos = nn.Parameter(torch.zeros(1, cfg.frames + 1, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.num_layers)
        self.head = nn.Sequential(nn.LayerNorm(cfg.d_model), nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, num_classes))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        b = x.size(0)
        x = self.input_proj(x)
        cls = self.cls.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, : x.size(1) + 1]
        return self.head(self.encoder(x)[:, 0])


def make_loader(cfg: Config, split: str, lighting: str, shuffle: bool):
    return DataLoader(
        SkeletonDataset(cfg, split, lighting),
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def class_weights(cfg: Config, num_classes: int, device):
    rows = load_manifest(Path(cfg.data_root), "train")
    counts = np.ones(num_classes, dtype=np.float32)
    for row in rows:
        counts[int(row["label_id"])] += 1
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)


def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    ensure_cache(cfg, [("train", "normal"), ("val", "normal")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = load_num_classes(Path(cfg.data_root))
    model = SkeletonTransformer(num_classes, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights(cfg, num_classes, device), label_smoothing=0.05)
    train_loader = make_loader(cfg, "train", "normal", True)
    val_loader = make_loader(cfg, "val", "normal", False)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_skeleton_transformer.pth"
    history_path = out_dir / "skeleton_epoch_metrics.csv"
    if history_path.exists():
        history_path.unlink()
    best = -1.0

    print(f"Training Skeleton Transformer on normal-light skeletons with device={device}")
    print(f"Classes={num_classes} | train batches={len(train_loader)} | val batches={len(val_loader)}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = correct = total = 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch:03d}/{cfg.epochs} train", dynamic_ncols=True)
        for batch_idx, (x, y) in enumerate(iterator, start=1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            correct += int((logits.argmax(1) == y).sum())
            total += int(y.size(0))
            if batch_idx == 1 or batch_idx % cfg.progress_every == 0 or batch_idx == len(train_loader):
                iterator.set_postfix(loss=f"{total_loss / total:.4f}", acc=f"{correct / total:.4f}")

        val = evaluate(model, val_loader, device, "val normal")
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "train_accuracy": correct / max(1, total),
            "val_accuracy": val["accuracy"],
            "val_f1": val["f1"],
        }
        with history_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"Epoch {epoch:03d}: loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} val_acc={row['val_accuracy']:.4f} val_f1={row['val_f1']:.4f}"
        )
        if val["accuracy"] > best:
            best = val["accuracy"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "num_classes": num_classes}, best_path)
            print(f"  saved {best_path}")


@torch.no_grad()
def evaluate(model, loader, device, desc: str):
    model.eval()
    preds, labels = [], []
    for x, y in tqdm(loader, desc=desc, dynamic_ncols=True):
        x = x.to(device)
        logits = model(x)
        preds.extend(logits.argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "preds": preds,
        "labels": labels,
    }


def evaluate_lighting(cfg: Config) -> None:
    ensure_cache(cfg, [("test", "normal"), ("test", "low_light"), ("test", "overexposed")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(cfg.output_dir) / "best_skeleton_transformer.pth", map_location=device)
    model = SkeletonTransformer(ckpt["num_classes"], cfg).to(device)
    model.load_state_dict(ckpt["model"])

    out_dir = Path(cfg.output_dir)
    results = {}
    for lighting in ["normal", "low_light", "overexposed"]:
        loader = make_loader(cfg, "test", lighting, False)
        results[lighting] = evaluate(model, loader, device, f"test {LIGHTING[lighting]}")
        print(f"{LIGHTING[lighting]}: {results[lighting]['accuracy'] * 100:.2f}%")

        pred_path = out_dir / f"predictions_{lighting}.csv"
        with pred_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true_label_id", "pred_label_id"])
            writer.writerows(zip(results[lighting]["labels"], results[lighting]["preds"]))
        print(f"  saved {pred_path}")

    row = {
        "Model": cfg.model_name,
        "Normal Light": f"{results['normal']['accuracy'] * 100:.2f}",
        "Low Light": f"{results['low_light']['accuracy'] * 100:.2f}",
        "Overexposed": f"{results['overexposed']['accuracy'] * 100:.2f}",
    }
    for name, ext in [("skeleton_conference_results_table", "csv")]:
        with (out_dir / f"{name}.{ext}").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
    with (out_dir / "skeleton_conference_results_table.md").open("w", encoding="utf-8") as f:
        f.write("| Model | Normal Light | Low Light | Overexposed |\n")
        f.write("|---|---:|---:|---:|\n")
        f.write(f"| {row['Model']} | {row['Normal Light']}% | {row['Low Light']}% | {row['Overexposed']}% |\n")
    metrics_only = {
        lighting: {k: v for k, v in metrics.items() if k not in ("preds", "labels")}
        for lighting, metrics in results.items()
    }
    with (out_dir / "skeleton_all_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_only, f, indent=2)

    print("\nConference Format Result Table")
    print("| Model | Normal Light | Low Light | Overexposed |")
    print("|---|---:|---:|---:|")
    print(f"| {row['Model']} | {row['Normal Light']}% | {row['Low Light']}% | {row['Overexposed']}% |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["extract", "train", "evaluate", "all"], default="all")
    parser.add_argument("--data_root", default="isl_lighting_dataset")
    parser.add_argument("--output_dir", default="results_skeleton_lighting")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--progress_every", type=int, default=25)
    parser.add_argument("--quick_run", action="store_true")
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        output_dir=args.output_dir,
        frames=args.frames,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        progress_every=args.progress_every,
        max_train_samples=128 if args.quick_run else None,
        max_eval_samples=64 if args.quick_run else None,
    )

    if args.mode in {"extract", "all"}:
        extract_required(cfg)
    if args.mode in {"train", "all"}:
        train(cfg)
    if args.mode in {"evaluate", "all"}:
        evaluate_lighting(cfg)


if __name__ == "__main__":
    main()
