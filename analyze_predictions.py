import csv
from pathlib import Path
from sklearn.metrics import classification_report

OUTPUT_DIR = Path("results_skeleton_lighting")
DATA_ROOT = Path("isl_lighting_dataset")

id_to_name = {}
with (DATA_ROOT / "manifests" / "test.csv").open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        id_to_name[int(row["label_id"])] = row["class_name"]

def load_preds(lighting):
    labels, preds = [], []
    with (OUTPUT_DIR / f"predictions_{lighting}.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["true_label_id"]))
            preds.append(int(row["pred_label_id"]))
    return labels, preds

for lighting in ["normal", "low_light", "overexposed"]:
    labels, preds = load_preds(lighting)
    class_names = [id_to_name[i] for i in sorted(id_to_name)]
    report = classification_report(
        labels, preds,
        labels=sorted(id_to_name),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    out_path = OUTPUT_DIR / f"classwise_report_{lighting}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class_name", "precision", "recall", "f1-score", "support"])
        for cname in class_names:
            r = report[cname]
            writer.writerow([cname, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1-score']:.3f}", int(r["support"])])
    print(f"Saved {out_path}")

import pandas as pd
normal_df = pd.read_csv(OUTPUT_DIR / "classwise_report_normal.csv").set_index("class_name")
low_df = pd.read_csv(OUTPUT_DIR / "classwise_report_low_light.csv").set_index("class_name")
drop = (normal_df["f1-score"] - low_df["f1-score"]).sort_values(ascending=False)
print("\nTop 10 classes most degraded by low light (F1 drop, Normal minus Low Light):")
print(drop.head(10))
drop.to_csv(OUTPUT_DIR / "classwise_f1_drop_normal_vs_lowlight.csv", header=["f1_drop"])
print(f"\nSaved {OUTPUT_DIR / 'classwise_f1_drop_normal_vs_lowlight.csv'}")