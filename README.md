Lighting Robustness of Skeleton-Based Transformer Recognition for Indian Sign Language

This repository contains the code, cached skeleton outputs, and result tables accompanying our study of photometric robustness in a skeleton-based Transformer pipeline for Indian Sign Language (ISL) recognition, evaluated on the INCLUDE dataset.

We simulate three lighting regimes — normal light, low light (brightness scale 0.35), and overexposed (brightness scale 1.65) — apply MediaPipe Holistic to extract 51-joint pose-and-hand landmark sequences (x, y, z, visibility/presence), and train a Spatial-Temporal Transformer classifier on normal-light skeletons only. We report two complementary robustness metrics — frame-level skeleton detection rate and per-joint completeness — alongside downstream classification accuracy, showing that skeleton extraction remains stable across lighting conditions while recognition accuracy is disproportionately affected under low light.

Contents:

Deterministic photometric perturbation and MediaPipe extraction scripts
Manifest-based train/val/test split (3,028 / 617 / 630 videos, 269 classes)
Cached skeleton sequences for all three lighting conditions
Skeleton Transformer training and evaluation code
Result tables (extraction robustness, classification accuracy, class-wise performance)

Released for reproducibility and extension in support of our manuscript, "Lighting Robustness of Skeleton-Based Transformer Recognition for Indian Sign Language under Simulated Illumination."
