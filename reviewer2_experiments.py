"""
Additional experiments requested by Reviewer 2 (novelty / technical contribution).

Put this file in the same folder as isl_skeleton_lighting_experiment.py and run it
from that folder. It reuses your manifests, your skeleton cache and your trained
checkpoint, so nothing already computed has to be recomputed.

Experiments
-----------
E1  bodypart   Joint completeness for pose / left hand / right hand under each
               lighting condition, joint retention relative to normal light,
               and accuracy versus hand completeness per test clip.
               (No training, no extraction. Minutes.)

E2  stats      McNemar's exact test and paired bootstrap 95% CIs on the
               predictions you already have. (No training. Seconds.)

E3  train      Trains one model variant over several seeds and evaluates every
               checkpoint on all test lighting conditions. Variants:
                 baseline     your model and input, unchanged
                 renorm       presence-consistent normalisation: centre on the
                              shoulder midpoint, scale by shoulder width, and
                              set missing joints to exactly zero
                 renorm_drop  renorm + hand-dropout training augmentation
                 robust       renorm_drop + frame-level attention masking of
                              frames with no detected landmarks
               (Uses the cached normal-light training skeletons only.)

E4  enhance    Re-extracts the test videos with an image fix applied after the
               lighting change and before MediaPipe:
                 low_light_clahe     CLAHE on the L channel (Lab colour space)
                 low_light_restored  ideal brightness restoration (x 1/0.35),
                                     an upper bound on what any brightness
                                     correction can recover
                 overexposed_clahe, normal_clahe   controls
               then evaluates your trained checkpoint on them.

E5  sweep      Re-extracts the test videos at a range of brightness scales and
               evaluates your checkpoint (and optionally E3 checkpoints) at each,
               giving an accuracy-versus-brightness curve.

E6  summary    Collects every result into results_reviewer2/summary.md.

Typical order
-------------
    python reviewer2_experiments.py          # quick: bodypart + stats + summary (minutes)
    python reviewer2_experiments.py all      # everything below, in order (many hours)

or one at a time:
    python reviewer2_experiments.py bodypart
    python reviewer2_experiments.py stats
    python reviewer2_experiments.py train --variant baseline --seeds 42 43 44 45 46
    python reviewer2_experiments.py train --variant robust   --seeds 42 43 44 45 46
    python reviewer2_experiments.py train --variant renorm   --seeds 42 43 44 45 46
    python reviewer2_experiments.py train --variant renorm_drop --seeds 42 43 44 45 46
    python reviewer2_experiments.py enhance
    python reviewer2_experiments.py sweep
    python reviewer2_experiments.py summary

Add --epochs 2 --seeds 42 to any train command for a fast smoke test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

import isl_skeleton_lighting_experiment as exp

POSE = slice(0, len(exp.POSE_IDS))                                  # 9 joints
LEFT = slice(len(exp.POSE_IDS), len(exp.POSE_IDS) + 21)             # 21 joints
RIGHT = slice(len(exp.POSE_IDS) + 21, exp.NUM_JOINTS)               # 21 joints
HANDS = slice(len(exp.POSE_IDS), exp.NUM_JOINTS)                    # 42 joints
L_SHOULDER, R_SHOULDER = exp.POSE_IDS.index(11), exp.POSE_IDS.index(12)
BASE_LIGHTINGS = ["normal", "low_light", "overexposed"]
VARIANTS = ["baseline", "renorm", "renorm_drop", "robust"]


# --------------------------------------------------------------------------- utils
def make_cfg(args) -> exp.Config:
    return exp.Config(data_root=args.data_root, output_dir=args.output_dir)


def r2_dir(args) -> Path:
    p = Path(args.r2_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_npz(cfg, split, lighting, row):
    path = exp.skeleton_path(cfg, split, lighting, row)
    if not path.exists():
        raise SystemExit(
            f"Missing skeleton cache: {path}\n"
            f"Run the extraction for split='{split}', lighting='{lighting}' first."
        )
    d = np.load(path)
    return d["skeleton"].astype(np.float32), d["mask"].astype(np.float32)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  saved {path}")


def md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return ""
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels, preds = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["true_label_id"]))
            preds.append(int(row["pred_label_id"]))
    return np.array(labels), np.array(preds)


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    """b = A right & B wrong, c = A wrong & B right; exact two-sided binomial p-value."""
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# --------------------------------------------------------------------------- E1
def cmd_bodypart(args) -> None:
    cfg = make_cfg(args)
    rows = exp.load_manifest(Path(cfg.data_root), "test")
    out = r2_dir(args)
    masks = {lt: np.stack([load_npz(cfg, "test", lt, r)[1] for r in rows]) for lt in BASE_LIGHTINGS}
    n_clips, T, J = masks["normal"].shape
    print(f"E1: {n_clips} test clips x {T} frames x {J} joints")

    parts = {"Pose (9)": POSE, "Left hand (21)": LEFT, "Right hand (21)": RIGHT,
             "Both hands (42)": HANDS, "All joints (51)": slice(0, J)}

    # completeness per part and condition
    comp_rows = []
    for name, sl in parts.items():
        row = {"Joints": name}
        for lt in BASE_LIGHTINGS:
            row[exp.LIGHTING[lt]] = f"{100 * masks[lt][:, :, sl].mean():.1f}"
        row["Low - Normal (pp)"] = f"{100 * (masks['low_light'][:, :, sl].mean() - masks['normal'][:, :, sl].mean()):+.1f}"
        row["Over - Normal (pp)"] = f"{100 * (masks['overexposed'][:, :, sl].mean() - masks['normal'][:, :, sl].mean()):+.1f}"
        comp_rows.append(row)

    # retention: of joints detected under normal light, share still detected
    ret_rows = []
    for name, sl in parts.items():
        n = masks["normal"][:, :, sl]
        row = {"Joints": name}
        for lt in ["low_light", "overexposed"]:
            m = masks[lt][:, :, sl]
            row[f"Retained under {exp.LIGHTING[lt]} (%)"] = f"{100 * (m * n).sum() / max(1, n.sum()):.1f}"
            row[f"Newly detected under {exp.LIGHTING[lt]} (%)"] = f"{100 * (m * (1 - n)).sum() / max(1, n.sum()):.1f}"
        ret_rows.append(row)

    # frame-level: pose found / at least one hand found
    frame_rows = []
    for lt in BASE_LIGHTINGS:
        m = masks[lt]
        frame_rows.append({
            "Condition": exp.LIGHTING[lt],
            "Frames with any landmark (%)": f"{100 * (m.sum(2) > 0).mean():.1f}",
            "Frames with pose (%)": f"{100 * (m[:, :, POSE].sum(2) > 0).mean():.1f}",
            "Frames with >=1 hand (%)": f"{100 * (m[:, :, HANDS].sum(2) > 0).mean():.1f}",
            "Frames with both hands (%)": f"{100 * ((m[:, :, LEFT].sum(2) > 0) & (m[:, :, RIGHT].sum(2) > 0)).mean():.1f}",
        })

    write_csv(out / "E1_completeness_by_bodypart.csv", comp_rows)
    write_csv(out / "E1_retention_vs_normal.csv", ret_rows)
    write_csv(out / "E1_frame_level_by_bodypart.csv", frame_rows)

    # per-clip accuracy vs hand completeness (needs predictions from your evaluate run)
    md = ["## E1. Landmark completeness by body part (test split, 630 clips)\n",
          md_table(comp_rows), "\n", md_table(ret_rows), "\n", md_table(frame_rows)]
    pred_paths = {lt: Path(cfg.output_dir) / f"predictions_{lt}.csv" for lt in BASE_LIGHTINGS}
    if all(p.exists() for p in pred_paths.values()):
        correct = {}
        for lt, p in pred_paths.items():
            y, yhat = load_predictions(p)
            if len(y) != n_clips:
                raise SystemExit(f"{p} has {len(y)} rows but the test manifest has {n_clips}.")
            labels = np.array([int(r["label_id"]) for r in rows])
            if not np.array_equal(labels, y):
                raise SystemExit(f"Row order of {p} does not match the test manifest.")
            correct[lt] = yhat == y

        hand_c = {lt: masks[lt][:, :, HANDS].mean(axis=(1, 2)) for lt in BASE_LIGHTINGS}
        bins = [0.0, 0.25, 0.5, 0.75, 1.0001]
        labels_b = ["0-25%", "25-50%", "50-75%", "75-100%"]
        bin_rows = []
        for lt in BASE_LIGHTINGS:
            for lo, hi, lab in zip(bins[:-1], bins[1:], labels_b):
                sel = (hand_c[lt] >= lo) & (hand_c[lt] < hi)
                bin_rows.append({"Condition": exp.LIGHTING[lt], "Hand completeness": lab,
                                 "Clips": int(sel.sum()),
                                 "Accuracy (%)": f"{100 * correct[lt][sel].mean():.1f}" if sel.any() else "-"})
        write_csv(out / "E1_accuracy_by_hand_completeness.csv", bin_rows)

        # clips that flip from correct (normal) to wrong (low light)
        flip = correct["normal"] & ~correct["low_light"]
        kept = correct["normal"] & correct["low_light"]
        drop_h = hand_c["normal"] - hand_c["low_light"]
        pose_c = {lt: masks[lt][:, :, POSE].mean(axis=(1, 2)) for lt in BASE_LIGHTINGS}
        drop_p = pose_c["normal"] - pose_c["low_light"]
        flip_rows = [
            {"Group": "Correct in normal, wrong in low light", "Clips": int(flip.sum()),
             "Mean hand-completeness loss (pp)": f"{100 * drop_h[flip].mean():.1f}" if flip.any() else "-",
             "Mean pose-completeness loss (pp)": f"{100 * drop_p[flip].mean():.1f}" if flip.any() else "-"},
            {"Group": "Correct in both", "Clips": int(kept.sum()),
             "Mean hand-completeness loss (pp)": f"{100 * drop_h[kept].mean():.1f}" if kept.any() else "-",
             "Mean pose-completeness loss (pp)": f"{100 * drop_p[kept].mean():.1f}" if kept.any() else "-"},
        ]
        try:
            from scipy.stats import mannwhitneyu
            if flip.any() and kept.any():
                p = mannwhitneyu(drop_h[flip], drop_h[kept], alternative="greater").pvalue
                flip_rows[0]["Mann-Whitney p (hand loss larger)"] = fmt_p(p)
                flip_rows[1]["Mann-Whitney p (hand loss larger)"] = ""
        except ImportError:
            pass
        write_csv(out / "E1_flipped_clips.csv", flip_rows)

        per_clip = []
        for i, r in enumerate(rows):
            rec = {"video_path": r["video_path"], "class_name": r["class_name"]}
            for lt in BASE_LIGHTINGS:
                rec[f"hand_jc_{lt}"] = round(float(hand_c[lt][i]), 4)
                rec[f"pose_jc_{lt}"] = round(float(pose_c[lt][i]), 4)
                rec[f"correct_{lt}"] = int(correct[lt][i])
            per_clip.append(rec)
        write_csv(out / "E1_per_clip.csv", per_clip)
        md += ["\n", md_table(bin_rows), "\n", md_table(flip_rows)]
    else:
        print("  (predictions_*.csv not found in output_dir: skipped accuracy-vs-completeness part)")

    (out / "E1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


# --------------------------------------------------------------------------- E2
def cmd_stats(args) -> None:
    cfg = make_cfg(args)
    out = r2_dir(args)
    rng = np.random.default_rng(0)
    preds = {lt: load_predictions(Path(cfg.output_dir) / f"predictions_{lt}.csv") for lt in BASE_LIGHTINGS}
    y = preds["normal"][0]
    correct = {lt: preds[lt][1] == preds[lt][0] for lt in BASE_LIGHTINGS}
    n = len(y)
    B = args.bootstrap
    idx = rng.integers(0, n, size=(B, n))
    boot = {lt: correct[lt][idx].mean(axis=1) for lt in BASE_LIGHTINGS}

    acc_rows = []
    for lt in BASE_LIGHTINGS:
        lo, hi = np.percentile(boot[lt], [2.5, 97.5])
        acc_rows.append({"Condition": exp.LIGHTING[lt], "Accuracy (%)": f"{100 * correct[lt].mean():.2f}",
                         "95% CI (%)": f"[{100 * lo:.2f}, {100 * hi:.2f}]"})
    test_rows = []
    for lt in ["low_light", "overexposed"]:
        d = boot["normal"] - boot[lt]
        lo, hi = np.percentile(d, [2.5, 97.5])
        b, c, p = mcnemar_exact(correct["normal"], correct[lt])
        test_rows.append({
            "Comparison": f"Normal vs {exp.LIGHTING[lt]}",
            "Accuracy drop (pp)": f"{100 * (correct['normal'].mean() - correct[lt].mean()):.2f}",
            "95% CI of drop (pp)": f"[{100 * lo:.2f}, {100 * hi:.2f}]",
            "Correct->wrong": b, "Wrong->correct": c, "McNemar exact p": fmt_p(p),
        })
    b, c, p = mcnemar_exact(correct["overexposed"], correct["low_light"])
    test_rows.append({"Comparison": "Overexposed vs Low Light",
                      "Accuracy drop (pp)": f"{100 * (correct['overexposed'].mean() - correct['low_light'].mean()):.2f}",
                      "95% CI of drop (pp)": "", "Correct->wrong": b, "Wrong->correct": c,
                      "McNemar exact p": fmt_p(p)})
    write_csv(out / "E2_accuracy_ci.csv", acc_rows)
    write_csv(out / "E2_mcnemar.csv", test_rows)
    md = [f"## E2. Significance of the accuracy drops ({n} paired test clips, {B} bootstrap resamples)\n",
          md_table(acc_rows), "\n", md_table(test_rows)]
    (out / "E2_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


# --------------------------------------------------------------------------- E3
def renormalize(skel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Presence-consistent normalisation from the cached sequence.

    The cache stores coordinates already centred on the mean of visible joints and
    divided by the largest visible distance. That is a translation plus a uniform
    scale of the visible joints, so re-normalising them here gives exactly what
    normalising the raw MediaPipe output would give. Missing joints, which the
    original centring moved away from zero, are set back to exactly zero.
    """
    coords = skel[..., :3].copy()
    flag = skel[..., 3:4].copy()
    vis = mask > 0
    for t in range(coords.shape[0]):
        v = vis[t]
        if not v.any():
            coords[t] = 0.0
            continue
        if v[L_SHOULDER] and v[R_SHOULDER]:
            centre = 0.5 * (coords[t, L_SHOULDER] + coords[t, R_SHOULDER])
            scale = float(np.linalg.norm(coords[t, L_SHOULDER, :2] - coords[t, R_SHOULDER, :2]))
        else:
            centre = coords[t, v].mean(axis=0)
            scale = float(np.linalg.norm(coords[t, v, :2] - centre[:2], axis=1).max())
        coords[t] -= centre
        if scale > 1e-6:
            coords[t] /= scale
        coords[t, ~v] = 0.0
    flag[~vis] = 0.0
    return np.concatenate([coords, flag], axis=-1).astype(np.float32)


def hand_dropout(x: np.ndarray, mask: np.ndarray, rng: random.Random, max_rate: float):
    """Training augmentation: drop each hand in random frames (coords and flag to 0)."""
    rate = rng.uniform(0.0, max_rate)
    x, mask = x.copy(), mask.copy()
    for t in range(x.shape[0]):
        for sl in (LEFT, RIGHT):
            if rng.random() < rate:
                x[t, sl] = 0.0
                mask[t, sl] = 0.0
    return x, mask


class CachedSkeletons:
    """Torch dataset over your cache; the input representation depends on the variant."""

    def __init__(self, cfg, split, lighting, variant, train=False, max_drop=0.5, seed=0):
        import torch  # noqa: F401
        self.rows = exp.load_manifest(Path(cfg.data_root), split)
        self.cfg, self.split, self.lighting = cfg, split, lighting
        self.variant, self.train, self.max_drop = variant, train, max_drop
        self.rng = random.Random(seed)
        self.cache = [load_npz(cfg, split, lighting, r) for r in self.rows]
        if variant != "baseline":
            self.cache = [(renormalize(s, m), m) for s, m in self.cache]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        import torch
        x, m = self.cache[i]
        if self.train and self.variant in ("renorm_drop", "robust") and self.max_drop > 0:
            x, m = hand_dropout(x, m, self.rng, self.max_drop)
        pad = m.sum(axis=1) == 0  # frames with no landmark at all
        return (torch.from_numpy(x).flatten(1), torch.from_numpy(pad),
                torch.tensor(int(self.rows[i]["label_id"]), dtype=torch.long))


def build_model(num_classes, cfg, variant):
    import torch

    class MaskedSkeletonTransformer(exp.SkeletonTransformer):
        """Same layers as yours; frames without landmarks are masked out of attention."""

        def forward(self, x, frame_pad=None):
            b = x.size(0)
            x = self.input_proj(x)
            x = torch.cat([self.cls.expand(b, -1, -1), x], dim=1) + self.pos[:, : x.size(1) + 1]
            mask = None
            if frame_pad is not None:
                mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), frame_pad], dim=1)
            return self.head(self.encoder(x, src_key_padding_mask=mask)[:, 0])

    if variant == "robust":
        return MaskedSkeletonTransformer(num_classes, cfg)
    return exp.SkeletonTransformer(num_classes, cfg)


def forward(model, x, pad, variant):
    return model(x, pad) if variant == "robust" else model(x)


def run_eval(model, loader, device, variant):
    import torch
    from sklearn.metrics import f1_score
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, pad, y in loader:
            logits = forward(model, x.to(device), pad.to(device), variant)
            ps += logits.argmax(1).cpu().tolist()
            ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)
    return {"accuracy": float((ys == ps).mean()),
            "macro_f1": float(f1_score(ys, ps, average="macro", zero_division=0)),
            "labels": ys, "preds": ps}


def available_lightings(cfg, extra: list[str]) -> list[str]:
    rows = exp.load_manifest(Path(cfg.data_root), "test")
    out = []
    for lt in BASE_LIGHTINGS + extra:
        if all(exp.skeleton_path(cfg, "test", lt, r).exists() for r in rows):
            out.append(lt)
        elif lt in BASE_LIGHTINGS:
            raise SystemExit(f"Test cache for '{lt}' is incomplete. Run your extraction first.")
        else:
            print(f"  (skipping '{lt}': test cache not found; run 'enhance' or 'sweep' first)")
    return out


def cmd_train(args) -> None:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    cfg = make_cfg(args)
    cfg.epochs = args.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = exp.load_num_classes(Path(cfg.data_root))
    out = r2_dir(args) / "E3_models" / args.variant
    out.mkdir(parents=True, exist_ok=True)
    lightings = available_lightings(cfg, args.extra_lightings)

    print(f"E3: variant={args.variant} seeds={args.seeds} epochs={cfg.epochs} device={device}")
    eval_sets = {lt: CachedSkeletons(cfg, "test", lt, args.variant) for lt in lightings}
    val_set = CachedSkeletons(cfg, "val", "normal", args.variant)

    for seed in args.seeds:
        exp.set_seed(seed)
        train_set = CachedSkeletons(cfg, "train", "normal", args.variant, train=True,
                                    max_drop=args.max_drop, seed=seed)
        g = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, generator=g)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
        model = build_model(num_classes, cfg, args.variant).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        crit = nn.CrossEntropyLoss(weight=exp.class_weights(cfg, num_classes, device), label_smoothing=0.05)
        best, best_state, t0 = -1.0, None, time.time()
        for epoch in range(1, cfg.epochs + 1):
            model.train()
            for x, pad, y in train_loader:
                x, pad, y = x.to(device), pad.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(forward(model, x, pad, args.variant), y)
                loss.backward()
                opt.step()
            val = run_eval(model, val_loader, device, args.variant)
            if val["accuracy"] > best:
                best = val["accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
                print(f"  seed {seed} epoch {epoch:3d}/{cfg.epochs} val_acc={val['accuracy']:.4f} "
                      f"best={best:.4f} ({time.time() - t0:.0f}s)")
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "variant": args.variant, "seed": seed,
                    "num_classes": num_classes, "best_val_accuracy": best}, out / f"seed{seed}.pth")

        result = {"variant": args.variant, "seed": seed, "best_val_accuracy": best}
        for lt in lightings:
            loader = DataLoader(eval_sets[lt], batch_size=cfg.batch_size, shuffle=False)
            m = run_eval(model, loader, device, args.variant)
            result[f"acc_{lt}"] = m["accuracy"]
            result[f"f1_{lt}"] = m["macro_f1"]
            with (out / f"seed{seed}_predictions_{lt}.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["true_label_id", "pred_label_id"])
                w.writerows(zip(m["labels"].tolist(), m["preds"].tolist()))
        (out / f"seed{seed}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("  " + " | ".join(f"{exp.LIGHTING.get(lt, lt)}={100 * result[f'acc_{lt}']:.2f}%" for lt in lightings))
    print(f"Done. Run 'summary' to aggregate seeds. Models in {out}")


# --------------------------------------------------------------------------- E4 / E5
_original_apply_lighting = exp.apply_lighting


def _clahe(frame):
    import cv2
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def extended_apply_lighting(frame, lighting: str):
    import cv2
    if lighting.endswith("_clahe"):
        return _clahe(extended_apply_lighting(frame, lighting[: -len("_clahe")]))
    if lighting == "low_light_restored":
        low = _original_apply_lighting(frame, "low_light")
        return cv2.convertScaleAbs(low, alpha=1 / 0.35, beta=0)
    if lighting.startswith("scale_"):
        return cv2.convertScaleAbs(frame, alpha=float(lighting[len("scale_"):]), beta=0)
    return _original_apply_lighting(frame, lighting)


exp.apply_lighting = extended_apply_lighting  # used by exp.extract_video_skeleton


def scale_name(s: float) -> str:
    if abs(s - 1.0) < 1e-9:
        return "normal"
    if abs(s - 0.35) < 1e-9:
        return "low_light"
    if abs(s - 1.65) < 1e-9:
        return "overexposed"
    return f"scale_{s:g}"


def extraction_summary(cfg, lighting) -> dict:
    """DR / JC / pose JC / hand JC from the cache, and brightness from your stats CSV."""
    rows = exp.load_manifest(Path(cfg.data_root), "test")
    masks = np.stack([load_npz(cfg, "test", lighting, r)[1] for r in rows])
    stats = Path(cfg.output_dir) / f"extraction_stats_test_{lighting}.csv"
    bright = ""
    if stats.exists():
        with stats.open(newline="", encoding="utf-8") as f:
            vals = [float(r["mean_brightness"]) for r in csv.DictReader(f)]
        bright = f"{np.mean(vals):.1f}" if vals else ""
    return {"Mean brightness": bright,
            "Frame detection (%)": f"{100 * (masks.sum(2) > 0).mean():.1f}",
            "Joint completeness (%)": f"{100 * masks.mean():.1f}",
            "Pose completeness (%)": f"{100 * masks[:, :, POSE].mean():.1f}",
            "Hand completeness (%)": f"{100 * masks[:, :, HANDS].mean():.1f}"}


def eval_original_checkpoint(cfg, lighting, device):
    import torch
    from torch.utils.data import DataLoader
    ckpt = torch.load(Path(cfg.output_dir) / "best_skeleton_transformer.pth", map_location=device)
    model = exp.SkeletonTransformer(ckpt["num_classes"], cfg).to(device)
    model.load_state_dict(ckpt["model"])
    ds = CachedSkeletons(cfg, "test", lighting, "baseline")
    return run_eval(model, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False), device, "baseline")


def eval_r2_models(args, cfg, lighting, device) -> dict:
    """Mean accuracy of every trained E3 variant on one lighting (if any exist)."""
    import torch
    from torch.utils.data import DataLoader
    res = {}
    root = Path(args.r2_dir) / "E3_models"
    for variant in VARIANTS:
        paths = sorted((root / variant).glob("seed*.pth")) if root.exists() else []
        if not paths:
            continue
        ds = CachedSkeletons(cfg, "test", lighting, variant)
        accs = []
        for p in paths:
            ck = torch.load(p, map_location=device)
            m = build_model(ck["num_classes"], cfg, variant).to(device)
            m.load_state_dict(ck["model"])
            accs.append(run_eval(m, DataLoader(ds, batch_size=cfg.batch_size, shuffle=False), device, variant)["accuracy"])
        res[variant] = (float(np.mean(accs)), float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, len(accs))
    return res


def cmd_enhance(args) -> None:
    import torch
    cfg = make_cfg(args)
    out = r2_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for lt in ["normal", "normal_clahe", "low_light", "low_light_clahe", "low_light_restored",
               "overexposed", "overexposed_clahe"]:
        if lt not in BASE_LIGHTINGS:
            exp.extract_split(cfg, "test", lt)
        m = eval_original_checkpoint(cfg, lt, device)
        row = {"Test input": lt, **extraction_summary(cfg, lt),
               "Accuracy (%)": f"{100 * m['accuracy']:.2f}", "Macro-F1": f"{m['macro_f1']:.3f}"}
        for v, (mu, sd, k) in eval_r2_models(args, cfg, lt, device).items():
            row[f"{v} acc (%)"] = f"{100 * mu:.2f} ± {100 * sd:.2f} (n={k})"
        rows.append(row)
        print(f"  {lt}: acc={100 * m['accuracy']:.2f}%  {row['Joint completeness (%)']}% JC")
    write_csv(out / "E4_enhancement.csv", rows)
    md = ["## E4. Image enhancement before MediaPipe (your trained checkpoint)\n", md_table(rows)]
    (out / "E4_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))


def cmd_sweep(args) -> None:
    import torch
    cfg = make_cfg(args)
    out = r2_dir(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for s in args.scales:
        lt = scale_name(s)
        if lt not in BASE_LIGHTINGS:
            exp.extract_split(cfg, "test", lt)
        m = eval_original_checkpoint(cfg, lt, device)
        row = {"Brightness scale": f"{s:g}", **extraction_summary(cfg, lt),
               "Accuracy (%)": f"{100 * m['accuracy']:.2f}", "Macro-F1": f"{m['macro_f1']:.3f}"}
        for v, (mu, sd, k) in eval_r2_models(args, cfg, lt, device).items():
            row[f"{v} acc (%)"] = f"{100 * mu:.2f} ± {100 * sd:.2f} (n={k})"
        rows.append(row)
        print(f"  scale {s:g}: acc={row['Accuracy (%)']}%  JC={row['Joint completeness (%)']}%  "
              f"hand JC={row['Hand completeness (%)']}%")
    write_csv(out / "E5_brightness_sweep.csv", rows)
    md = ["## E5. Brightness sweep (your trained checkpoint)\n", md_table(rows)]
    (out / "E5_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [float(r["Brightness scale"]) for r in rows]
        fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
        ax1.plot(xs, [float(r["Accuracy (%)"]) for r in rows], "o-", color="#1f4e79", label="Top-1 accuracy")
        ax1.plot(xs, [float(r["Hand completeness (%)"]) for r in rows], "s--", color="#c55a11", label="Hand completeness")
        ax1.plot(xs, [float(r["Pose completeness (%)"]) for r in rows], "^:", color="#548235", label="Pose completeness")
        ax1.axvline(1.0, color="grey", lw=0.8, ls="--")
        ax1.set_xlabel("Brightness scale factor (1.0 = original video)")
        ax1.set_ylabel("Percent")
        ax1.set_ylim(0, 103)
        ax1.legend(loc="lower right", frameon=False, fontsize=8)
        ax1.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / "E5_brightness_sweep.png", dpi=600)
        print(f"  saved {out / 'E5_brightness_sweep.png'}")
    except ImportError:
        pass


# --------------------------------------------------------------------------- E6
def cmd_summary(args) -> None:
    out = r2_dir(args)
    md = ["# Reviewer 2 experiments: summary\n"]
    for name in ["E1_summary.md", "E2_summary.md"]:
        if (out / name).exists():
            md.append((out / name).read_text(encoding="utf-8"))

    root = out / "E3_models"
    agg_rows, per_seed = [], {}
    if root.exists():
        for variant in VARIANTS:
            files = sorted((root / variant).glob("seed*_metrics.json"))
            if not files:
                continue
            res = [json.loads(f.read_text(encoding="utf-8")) for f in files]
            per_seed[variant] = res
            lts = [k[4:] for k in res[0] if k.startswith("acc_") and all(k in r for r in res)]
            row = {"Variant": variant, "Seeds": len(res)}
            for lt in lts:
                a = np.array([r[f"acc_{lt}"] for r in res]) * 100
                row[f"{exp.LIGHTING.get(lt, lt)} acc (%)"] = f"{a.mean():.2f} ± {a.std(ddof=1) if len(a) > 1 else 0:.2f}"
            for lt in lts:
                f1 = np.array([r[f"f1_{lt}"] for r in res])
                row[f"{exp.LIGHTING.get(lt, lt)} F1"] = f"{f1.mean():.3f} ± {f1.std(ddof=1) if len(f1) > 1 else 0:.3f}"
            agg_rows.append(row)
    if agg_rows:
        md += ["## E3. Mitigation and ablation (mean ± SD over seeds)\n", md_table(agg_rows)]
        write_csv(out / "E3_summary.csv", agg_rows)
        if "baseline" in per_seed:
            b = per_seed["baseline"]
            gap = np.mean([r["acc_normal"] - r["acc_low_light"] for r in b])
            rec_rows = []
            for v, res in per_seed.items():
                if v == "baseline":
                    continue
                gain = np.mean([r["acc_low_light"] for r in res]) - np.mean([r["acc_low_light"] for r in b])
                rec_rows.append({"Variant": v,
                                 "Low-light gain over baseline (pp)": f"{100 * gain:+.2f}",
                                 "Share of baseline low-light gap recovered (%)": f"{100 * gain / gap:.1f}" if gap > 0 else "-"})
            # paired McNemar on matching seeds, low light
            seeds_b = {r["seed"]: r for r in b}
            for row, (v, res) in zip(rec_rows, [(v, r) for v, r in per_seed.items() if v != "baseline"]):
                ps = []
                for r in res:
                    if r["seed"] not in seeds_b:
                        continue
                    pb = load_predictions(root / "baseline" / f"seed{r['seed']}_predictions_low_light.csv")
                    pv = load_predictions(root / v / f"seed{r['seed']}_predictions_low_light.csv")
                    ps.append(mcnemar_exact(pv[1] == pv[0], pb[1] == pb[0])[2])
                row["McNemar p per seed (low light)"] = ", ".join(fmt_p(p) for p in ps) if ps else "-"
            if rec_rows:
                md += ["\n", md_table(rec_rows)]
                write_csv(out / "E3_gap_recovered.csv", rec_rows)
    for name in ["E4_summary.md", "E5_summary.md"]:
        if (out / name).exists():
            md.append((out / name).read_text(encoding="utf-8"))
    (out / "summary.md").write_text("\n\n".join(md) + "\n", encoding="utf-8")
    print("\n\n".join(md))
    print(f"\nSaved {out / 'summary.md'}  <- send this file back for the manuscript update")


# --------------------------------------------------------------------------- main
# If you start this file with a Run button (Spyder, VS Code, PyCharm, Jupyter), no command is
# passed. Then this one is used. "quick" takes minutes; "all" runs every experiment (many hours).
COMMAND = "quick"


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.append(COMMAND)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="quick",
                    choices=["quick", "all", "bodypart", "stats", "train", "enhance", "sweep", "summary"],
                    help="quick (default) = bodypart + stats + summary; all = every experiment in order.")
    ap.add_argument("--data_root", default="isl_lighting_dataset")
    ap.add_argument("--output_dir", default="results_skeleton_lighting",
                    help="Your original results folder (holds skeleton_cache, predictions, checkpoint).")
    ap.add_argument("--r2_dir", default="results_reviewer2", help="Where the new results are written.")
    ap.add_argument("--variant", choices=VARIANTS, default="robust")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--max_drop", type=float, default=0.5, help="Max per-frame hand-dropout rate.")
    ap.add_argument("--extra_lightings", nargs="*", default=[],
                    help="Extra cached test lightings to evaluate in 'train', e.g. low_light_clahe scale_0.25")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 1.65, 2.0])
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()
    cmds = {"bodypart": cmd_bodypart, "stats": cmd_stats, "train": cmd_train,
            "enhance": cmd_enhance, "sweep": cmd_sweep, "summary": cmd_summary}
    if args.command == "quick":
        print("No command given: running the quick set (bodypart, stats, summary).\n")
        for name in ["bodypart", "stats", "summary"]:
            print(f"\n========== {name} ==========")
            cmds[name](args)
        print("\nNext, for the mitigation results, run (or set COMMAND below and press Run):\n"
              "  python reviewer2_experiments.py all")
    elif args.command == "all":
        steps = ["bodypart", "stats"] + [f"train:{v}" for v in VARIANTS] + ["enhance", "sweep", "summary"]
        for step in steps:
            name, _, variant = step.partition(":")
            print(f"\n========== {step} ==========")
            if variant:
                args.variant = variant
            cmds[name](args)
    else:
        cmds[args.command](args)


if __name__ == "__main__":
    main()
