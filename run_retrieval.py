"""Phase 2: ResNet18 baseline retrieval on Gardens Point Walking.

Run from the project root:
    python run_retrieval.py

Outputs: results/phase2_metrics.json and results/phase2_examples.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.deep_features import build_model, embed_images, get_device

ROOT = Path(__file__).resolve().parent
SEQUENCES = ('day_right', 'day_left', 'night_right')


def image_paths(data_dir: Path, sequence: str) -> list[Path]:
    directory = data_dir / sequence
    paths = sorted(directory.glob('Image*.jpg'))
    expected = [directory / f'Image{i:03d}.jpg' for i in range(200)]
    if paths != expected:
        missing = [p.name for p in expected if not p.is_file()]
        raise FileNotFoundError(f'{sequence}: expected 200 numbered images. Missing: {missing[:8]}')
    return paths


def match_scores(query_features: np.ndarray, reference_features: np.ndarray) -> np.ndarray:
    # Both sets of features are L2-normalized; dot product = cosine similarity.
    return query_features @ reference_features.T


def retrieval_metrics(scores: np.ndarray, tolerance: int = 2) -> dict:
    # In this dataset, index i is an approximate corresponding location in
    # every traversal. A +-2 index tolerance is an *explicit evaluation rule*.
    ranked = np.argsort(-scores, axis=1)
    query_indices = np.arange(scores.shape[0])[:, None]
    correct = np.abs(ranked - query_indices) <= tolerance
    return {
        'recall_at_1': round(float(correct[:, :1].any(axis=1).mean()), 4),
        'recall_at_5': round(float(correct[:, :5].any(axis=1).mean()), 4),
        'recall_at_10': round(float(correct[:, :10].any(axis=1).mean()), 4),
        'queries': int(scores.shape[0]),
        'tolerance_frames': tolerance,
    }


def save_examples(scores_by_sequence: dict, image_map: dict, out_path: Path) -> None:
    sample_indices = (30, 90, 150)
    sequences = ('day_left', 'night_right')
    thumb_w, thumb_h = 250, 150
    text_h, left_w, margin = 35, 115, 12
    width = left_w + 4 * thumb_w + 5 * margin
    height = len(sequences) * len(sample_indices) * (thumb_h + text_h + margin) + 42
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), 'Query                  Top-1                     Top-2                     Top-3', fill='black')
    y = 42
    for sequence in sequences:
        scores = scores_by_sequence[sequence]
        for query_index in sample_indices:
            draw.text((12, y + 8), f'{sequence}\nImage{query_index:03d}', fill='black')
            ranked = np.argsort(-scores[query_index])[:3]
            items = [(image_map[sequence][query_index], 'QUERY')] + [
                (image_map['day_right'][int(i)], f'RANK {rank} / Image{int(i):03d}')
                for rank, i in enumerate(ranked, 1)
            ]
            for j, (path, label) in enumerate(items):
                with Image.open(path) as source:
                    im = ImageOps.exif_transpose(source).convert('RGB')
                    im.thumbnail((thumb_w-8, thumb_h-8))
                x = left_w + j * (thumb_w + margin)
                canvas.paste(im, (x + (thumb_w - im.width)//2, y + (thumb_h - im.height)//2))
                draw.text((x+4, y+thumb_h+4), label, fill='black')
            y += thumb_h + text_h + margin
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data'/'GardensPoint')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--tolerance', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.tolerance < 0:
        parser.error('batch-size must be positive and tolerance must be nonnegative')
    image_map = {seq: image_paths(args.data_dir, seq) for seq in SEQUENCES}
    device = get_device()
    print(f'Device: {device}')
    print('Loading ImageNet-pretrained ResNet18 (downloads weights on first run)...')
    model, transform = build_model(device)
    features = {}
    for sequence in SEQUENCES:
        print(f'Extracting features: {sequence} ({len(image_map[sequence])} images)')
        features[sequence] = embed_images(image_map[sequence], model, transform,
                                          device, batch_size=args.batch_size)
        print(f'  shape: {features[sequence].shape}')

    out_dir = ROOT/'results'
    out_dir.mkdir(parents=True, exist_ok=True)
    # This saved cache is useful for Phase 3 geometric reranking.
    np.savez_compressed(out_dir/'phase2_embeddings.npz', **features)
    scores_by_sequence = {}
    metrics = {}
    for sequence in ('day_left','night_right'):
        scores = match_scores(features[sequence], features['day_right'])
        scores_by_sequence[sequence] = scores
        metrics[sequence] = retrieval_metrics(scores, args.tolerance)
        first = np.argsort(-scores[50])[:5]
        print(f'\n{sequence} -> day_right')
        print(f'  Recall@1:  {metrics[sequence]["recall_at_1"]:.3f}')
        print(f'  Recall@5:  {metrics[sequence]["recall_at_5"]:.3f}')
        print(f'  Recall@10: {metrics[sequence]["recall_at_10"]:.3f}')
        print(f'  Example Image050 top-5: {[int(x) for x in first]}')

    (out_dir/'phase2_metrics.json').write_text(json.dumps({
        'model': 'ImageNet-pretrained ResNet18, penultimate layer (512D)',
        'reference': 'day_right',
        'evaluation_note': 'Same-index correspondence is approximate; metrics depend on frame alignment.',
        'experiments': metrics,
    }, indent=2), encoding='utf-8')
    save_examples(scores_by_sequence, image_map, out_dir/'phase2_examples.png')
    print('\nSaved results/phase2_metrics.json')
    print('Saved results/phase2_embeddings.npz')
    print('Saved results/phase2_examples.png')


if __name__ == '__main__':
    main()
