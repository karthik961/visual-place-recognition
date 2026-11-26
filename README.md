# Robust Visual Place Recognition

CS 682 Computer Vision Project — George Mason University

## Project Overview

This project investigates visual place recognition under
changes in camera viewpoint and illumination.

The objective is to compare deep visual features with
geometric feature matching to improve image retrieval.

## Dataset

Gardens Point Walking Dataset:
- Day Left: 200 images
- Day Right: 200 images
- Night Right: 200 images

Day Right is the reference sequence. The other two
sequences are used as queries.

Dataset:
https://huggingface.co/datasets/medwa126/GardensPointWalking

## Phase 1 — Dataset Validation

- Downloaded and validated 600 images.
- Verified image resolutions and sequence integrity.
- Visually inspected approximate frame correspondence.

## Phase 2 — ResNet18 Baseline

- Extracted 512-dimensional ResNet18 embeddings.
- Used cosine similarity for image retrieval.
- Evaluated retrieval with a +/-2 frame tolerance.

### Baseline Results

| Metric | Daytime | Nighttime |
|--------|---------|-----------|
| Recall@1 | 79% | 50% |
| Recall@5 | 95% | 78% |
| Recall@10 | 99% | 85% |

These results use approximate frame-index correspondence
rather than precise GPS ground truth.

## Phase 3 — Completed

Implemented ORB feature matching and RANSAC-based
geometric verification for top-10 image reranking.

Compared three retrieval approaches:
- ResNet18 baseline
- Fixed-weight hybrid
- Confidence-gated adaptive hybrid

### Results

| Method | Daytime R@1 | Nighttime R@1 |
|---|---:|---:|
| ResNet18 | 79.0% | 50.0% |
| Fixed Hybrid | 35.5% | 25.0% |
| Corrected Adaptive Hybrid | 84.5% | 52.0% |

The corrected adaptive approach improved Recall@1
by 5.5 percentage points during daytime and
2 percentage points at night.

These are exploratory results on Gardens Point;
parameters were not selected using a fully
independent evaluation dataset.

### Running Phase 3

Install dependencies:

    pip install -r requirements_phase3.txt

Run scoring tests:

    python test_scoring.py

Run corrected hybrid retrieval:

    python run_phase3_corrected.py


## Technologies

Python, PyTorch, torchvision, OpenCV, NumPy, Pillow

## Setup

Install the dependencies:

    pip install -r requirements.txt
    pip install -r requirements_phase2.txt

Download and validate the dataset:

    python download_data.py
    python validate_dataset.py

Run the ResNet18 baseline:

    python run_retrieval.py

Generated results are saved in the results/ directory.
