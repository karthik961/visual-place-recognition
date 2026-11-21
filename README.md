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

## Phase 3 — Planned

- ORB feature extraction and matching
- RANSAC geometric verification
- Hybrid image retrieval and reranking
- Comparison against the ResNet18 baseline

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
