# Indian Sign Language Recognition under Simulated Illumination

Code, data splits, cached skeletons and result tables for the paper:

> **Effect of Simulated Illumination on Skeleton-Based Indian Sign Language Recognition: Landmark Completeness as the Bottleneck**
> Satender Singh and Priyanka Mathur, Manipal University Jaipur, India.
> *Manuscript under review.*

## About

Skeleton-based sign language recognition is often assumed to be less sensitive to lighting than RGB-based
recognition, because it classifies landmark coordinates rather than pixels. This repository tests that
assumption for Indian Sign Language (ISL) on the INCLUDE dataset.

Each of the 630 test clips is rendered under three lighting conditions: normal light, low light
(brightness scale 0.35) and overexposure (brightness scale 1.65). MediaPipe Holistic then extracts 51 pose and hand
landmarks per frame, and a temporal skeleton Transformer classifies the sequence. The model is trained under
normal light only.

**Main findings**

- MediaPipe still finds a skeleton in at least 99.6% of frames under every condition, but joint completeness
  falls from 82.0% to 75.4% under low light, almost entirely through lost hand landmarks.
- Skeleton Transformer accuracy falls from 89.52% to 74.44% under low light (McNemar p < 0.001) and to
  86.03% under overexposure.
- A matched R(2+1)D-18 RGB baseline loses only 2.22 points under the same low light, so in this setting the
  skeleton pipeline is less robust than RGB.
- A presence-consistent variant (shoulder-anchored normalization, missing joints kept at zero, hand-landmark
  dropout during training, empty-frame masking) raises low-light accuracy from 73.24% to 88.35% over five
  seeds, without extra data or changes to MediaPipe.

| Model | Normal | Low light | Overexposed |
|---|---|---|---|
| Skeleton Transformer (single run, seed 42) | 89.52 | 74.44 | 86.03 |
| Skeleton Transformer (5 seeds, mean ± sd) | 88.00 ± 0.86 | 73.24 ± 1.30 | 85.59 ± 1.51 |
| Presence-consistent variant (5 seeds, mean ± sd) | 93.71 ± 0.18 | 88.35 ± 0.74 | 92.38 ± 0.56 |
| R(2+1)D-18 RGB baseline | 95.56 | 93.33 | 92.22 |

Test accuracy (%) on the same 630 clips.

## Repository contents

| Path | What it contains |
|---|---|
| `prepare_dataset.py` | Builds the manifest and the train / val / test split (70 / 15 / 15, seed 42) |
| `check_excluded_videos.py` | Lists every source video left out of the manifest, with the reason |
| `isl_skeleton_lighting_experiment.py` | Main pipeline: lighting simulation, MediaPipe extraction, Transformer training and evaluation |
| `analyze_predictions.py` | Per-class precision, recall and F1 from the saved predictions |
| `reviewer2_experiments.py` | Hand-vs-pose analysis, statistical tests, presence-consistent variant and ablations |
| `rgb_baseline_r2plus1d.py` | Matched R(2+1)D-18 RGB baseline |
| `make_figure4_hd.py` | Recreates Figure 4 (per-video joint detection box plot) |
| `isl_lighting_dataset/` | `manifests/` (train, val, test CSV files), `class_to_idx.json`, `metadata.json`, `excluded_videos_report.csv` |
| `results_skeleton_lighting/` | Extraction statistics, predictions, training log, trained checkpoint |
| `results_reviewer2/` | Results of the additional analyses and the five-seed runs |
| `results_rgb_r2plus1d/` | RGB baseline results |
| `skeleton_cache.zip` | Cached MediaPipe skeletons (32 × 51 × 4 per clip) for every split and lighting condition |

## Which script produces which result

| Paper | Script | Command |
|---|---|---|
| Table 1, Figure 1 | `prepare_dataset.py` | `python prepare_dataset.py --source_dir <INCLUDE folder>` |
| Section 2.2 (excluded videos) | `check_excluded_videos.py` | `python check_excluded_videos.py` |
| Tables 4 and 5, Figure 3 | `isl_skeleton_lighting_experiment.py` | `python isl_skeleton_lighting_experiment.py --mode all` |
| Table 6 | `analyze_predictions.py` | `python analyze_predictions.py` |
| Tables 7 and 8 | `reviewer2_experiments.py` | `python reviewer2_experiments.py bodypart` |
| Table 9 | `reviewer2_experiments.py` | `python reviewer2_experiments.py stats` |
| Table 10 | `reviewer2_experiments.py` | `python reviewer2_experiments.py train --variant <baseline / renorm / renorm_drop / robust>` |
| Table 11 | `rgb_baseline_r2plus1d.py` | `python rgb_baseline_r2plus1d.py --mode all` |
| Figure 4 | `make_figure4_hd.py` | `python make_figure4_hd.py` |

`python reviewer2_experiments.py all` runs every additional analysis in order and writes
`results_reviewer2/summary.md`.

## How to run

1. **Install the requirements** (Python 3.10 or later):
   ```
   pip install -r requirements.txt
   ```
   The main packages are `mediapipe`, `opencv-python`, `torch`, `torchvision`, `numpy`, `scikit-learn`,
   `scipy`, `pandas`, `matplotlib` and `tqdm`. The code uses MediaPipe's `solutions.holistic` API.

2. **Get the INCLUDE videos.** The videos are not included in this repository. Download them from the
   original source ([INCLUDE paper](https://doi.org/10.1145/3394171.3413528),
   [Hugging Face](https://huggingface.co/datasets/ai4bharat/INCLUDE)).

3. **Prepare the split.** Run `prepare_dataset.py` as shown in the table above. It writes the manifests to
   `isl_lighting_dataset/manifests/`.

4. **Skip skeleton extraction (optional).** Unzip `skeleton_cache.zip` into `results_skeleton_lighting/`, so
   that the files sit in `results_skeleton_lighting/skeleton_cache/`. The scripts reuse cached skeletons, so
   MediaPipe does not need to be run again.

5. **Run the experiments** with the commands in the table above. All scripts use relative paths and should
   be run from the repository root.

## Dataset notes

- The local copy of INCLUDE used here contains 4,275 videos in 269 class folders after manifest preparation.
  Seven signs appear under two class labels (folders ending in `_extra`), so the 269 classes represent 262
  distinct signs. Nine single-video folders were excluded because a class needs at least two videos to be
  split. `excluded_videos_report.csv` lists every excluded file and the reason.
- The split is a random video-level split, not a signer-independent split.
- Lighting is simulated with `cv2.convertScaleAbs` (brightness scale 0.35 and 1.65, clipped at 255) before
  landmark extraction. It is not a recording made in real low light.


## Contact

Satender Singh: Satender.2410200006@muj.manipal.edu
