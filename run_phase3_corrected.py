"""Phase 3 corrected-scoring experiment: ResNet18 top-10 with ORB + RANSAC reranking.

Run from the existing Phase 2 directory after copying this file and src/orb_geometry.py:
    python run_phase3.py --data-dir ../cs682_vpr_phase1/data/GardensPoint

Uses saved results/phase2_embeddings.npz, without loading or rerunning ResNet18.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from src.orb_geometry import extract_local_features, verify_pair

ROOT = Path(__file__).resolve().parent
SEQUENCES = ('day_right', 'day_left', 'night_right')
QUERY_SEQUENCES = ('day_left', 'night_right')


def image_paths(data_dir: Path, sequence: str) -> list[Path]:
    folder = data_dir / sequence
    expected = [folder / f'Image{i:03d}.jpg' for i in range(200)]
    missing = [p.name for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'{sequence}: missing images: {missing[:8]}')
    return expected


def recall_metrics(rankings: np.ndarray, tolerance: int) -> dict:
    correct = np.abs(rankings - np.arange(len(rankings))[:, None]) <= tolerance
    return {f'recall_at_{k}': round(float(correct[:, :k].any(axis=1).mean()), 4)
            for k in (1, 5, 10)}


def fuse_ranking(candidates: np.ndarray, cosines: np.ndarray, geom_stats: list[dict],
                 alpha: float, min_inliers: int, min_ratio: float, adaptive: bool,
                 geom_bonus: float = 0.05, inlier_scale: float = 30.0) -> np.ndarray:
    """Rerank only the original top-10 candidate images.

    Fixed hybrid preserves the original weighted-average experiment.
    Corrected adaptive hybrid adds a small positive geometry bonus ONLY when
    the pair passes both inlier-count and inlier-ratio thresholds. Otherwise
    its score remains the original deep score (no reliability penalty).

    The geometry bonus and thresholds are experimental parameters; do not
    report tuning on the same evaluation set as unbiased test performance.
    """
    deep = np.clip((cosines[candidates] + 1.0) / 2.0, 0.0, 1.0)
    geometric = np.clip(np.array([s['inlier_ratio'] for s in geom_stats], dtype=np.float64), 0.0, 1.0)
    if adaptive:
        inliers = np.array([s['inliers'] for s in geom_stats], dtype=np.float64)
        reliable = (inliers >= min_inliers) & (geometric >= min_ratio)
        confidence = np.minimum(inliers / inlier_scale, 1.0) * geometric
        fused = deep + geom_bonus * np.where(reliable, confidence, 0.0)
    else:
        fused = alpha * deep + (1 - alpha) * geometric
    # Stable tie handling maintains original deep ranking for equal fused scores.
    return candidates[np.argsort(-fused, kind='stable')]


def make_examples(rows, image_map, output: Path, tolerance: int):
    wanted = [r for r in rows if r['query_index'] in (30, 90, 150)]
    # Query + deep top1 + fixed top1 + adaptive top1 = four pictures per row.
    w, h, gap, label_h, left = 224, 126, 12, 54, 130
    canvas = Image.new('RGB', (left + 4*(w+gap)+12, 40+len(wanted)*(h+label_h)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), 'Query                    Deep top-1                Fixed hybrid top-1           Adaptive hybrid top-1', fill='black')
    for i, row in enumerate(wanted):
        y = 40 + i*(h+label_h)
        seq, idx = row['sequence'], row['query_index']
        draw.text((8, y+5), f'{seq}\nImage{idx:03d}', fill='black')
        indices = [None, row['deep_top1'], row['fixed_top1'], row['adaptive_top1']]
        for j, ref_idx in enumerate(indices):
            path = image_map[seq][idx] if j == 0 else image_map['day_right'][ref_idx]
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((w, h))
                x = left + j*(w+gap)
                canvas.paste(im, (x+(w-im.width)//2, y+(h-im.height)//2))
            label = 'QUERY' if ref_idx is None else f'Image{ref_idx:03d} | {"OK" if abs(idx-ref_idx) <= tolerance else "MISS"}'
            draw.text((x+4, y+h+4), label, fill='black')
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=ROOT/'data'/'GardensPoint')
    p.add_argument('--embeddings', type=Path, default=ROOT/'results'/'phase2_embeddings.npz')
    p.add_argument('--output-dir', type=Path, default=ROOT/'results')
    p.add_argument('--top-k', type=int, default=10)
    p.add_argument('--alpha', type=float, default=0.7,
                   help='Weight on deep score in the original fixed hybrid method')
    p.add_argument('--geom-bonus', type=float, default=0.05,
                   help='Max geometry bonus for corrected adaptive reranking')
    p.add_argument('--inlier-scale', type=float, default=30.0,
                   help='Inlier count at which the confidence multiplier saturates')
    p.add_argument('--min-inliers', type=int, default=8)
    p.add_argument('--min-ratio', type=float, default=0.2)
    p.add_argument('--tolerance', type=int, default=2)
    p.add_argument('--orb-features', type=int, default=1000)
    p.add_argument('--max-width', type=int, default=640)
    args = p.parse_args()
    if args.top_k != 10:
        p.error('Use --top-k 10 to keep Recall@1, @5, @10 comparable.')
    if not 0 <= args.alpha <= 1 or not 0 <= args.min_ratio <= 1:
        p.error('alpha and min-ratio must be between 0 and 1')
    if not 0 <= args.geom_bonus <= 1 or args.inlier_scale <= 0:
        p.error('geom-bonus must be in [0,1] and inlier-scale must be positive')
    if args.min_inliers < 4 or args.tolerance < 0 or args.orb_features < 4 or args.max_width < 32:
        p.error('Invalid ORB / RANSAC / tolerance parameters')
    image_map = {seq: image_paths(args.data_dir, seq) for seq in SEQUENCES}
    if not args.embeddings.is_file():
        p.error(f'Embedding cache not found: {args.embeddings}. Run Phase 2 first.')
    with np.load(args.embeddings) as cache:
        if set(SEQUENCES) - set(cache.files):
            p.error('Embedding file is missing required sequence keys')
        features = {seq: np.asarray(cache[seq], dtype=np.float32) for seq in SEQUENCES}
    for seq, embedding in features.items():
        if embedding.shape != (200, 512) or not np.isfinite(embedding).all():
            p.error(f'{seq}: embeddings must have finite shape (200,512), got {embedding.shape}')
        norms = np.linalg.norm(embedding, axis=1)
        if not np.allclose(norms, 1, atol=1e-3):
            p.error(f'{seq}: embeddings are not L2-normalized')

    print('Extracting ORB descriptors once per image (no ResNet rerun)...')
    cv2.setRNGSeed(0)  # Improve repeatability of OpenCV RANSAC
    orb = cv2.ORB_create(nfeatures=args.orb_features)
    local = {}
    for seq in SEQUENCES:
        t = time.perf_counter()
        local[seq] = [extract_local_features(path, orb, args.max_width)
                      for path in image_map[seq]]
        print(f'  {seq}: {len(local[seq])} images, {time.perf_counter()-t:.1f} seconds')
    print('Computing geometric verification for top-10 candidates...')

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summary = {
        'dataset': 'Gardens Point Walking',
        'ground_truth': f'approximate same-index correspondence +/-{args.tolerance} frames',
        'reference': 'day_right',
        'settings': vars(args) | {
            'data_dir': str(args.data_dir),
            # Avoid embedding local absolute home directory paths in public results.
            'embeddings': str(args.embeddings.relative_to(ROOT))
            if args.embeddings.is_relative_to(ROOT) else args.embeddings.name,
            'output_dir': str(args.output_dir.relative_to(ROOT))
            if args.output_dir.is_relative_to(ROOT) else args.output_dir.name,
        },
        'experiments': {},
        'interpretation_note': 'This is evaluation on the same dataset, not an unbiased tuning/test split; freeze parameters before future testing.'
    }
    for seq in QUERY_SEQUENCES:
        scores = features[seq] @ features['day_right'].T
        deep_rank = np.argsort(-scores, axis=1)[:, :args.top_k]
        fixed_rank = np.empty_like(deep_rank)
        adaptive_rank = np.empty_like(deep_rank)
        successes = 0
        comparisons = 0
        geometric_time = 0.0
        for qi in range(200):
            candidates = deep_rank[qi]
            t = time.perf_counter()
            stats = [verify_pair(local[seq][qi], local['day_right'][int(ri)])
                     for ri in candidates]
            geometric_time += time.perf_counter() - t
            successes += sum(s['inliers'] >= args.min_inliers and s['inlier_ratio'] >= args.min_ratio
                             for s in stats)
            comparisons += len(stats)
            fixed_rank[qi] = fuse_ranking(candidates, scores[qi], stats, args.alpha,
                                          args.min_inliers, args.min_ratio, adaptive=False)
            adaptive_rank[qi] = fuse_ranking(candidates, scores[qi], stats, args.alpha,
                                             args.min_inliers, args.min_ratio, adaptive=True,
                                             geom_bonus=args.geom_bonus,
                                             inlier_scale=args.inlier_scale)
            all_rows.append({'sequence':seq, 'query_index':qi,
                             'deep_top1': int(deep_rank[qi, 0]),
                             'fixed_top1': int(fixed_rank[qi, 0]),
                             'adaptive_top1': int(adaptive_rank[qi, 0]),
                             'deep_top5': ' '.join(map(str, deep_rank[qi, :5])),
                             'fixed_top5': ' '.join(map(str, fixed_rank[qi, :5])),
                             'adaptive_top5': ' '.join(map(str, adaptive_rank[qi, :5]))})
            if qi in (49, 99, 149, 199):
                print(f'  {seq}: {qi+1}/200 queries')
        metrics = {'deep_baseline': recall_metrics(deep_rank, args.tolerance),
                   'fixed_hybrid': recall_metrics(fixed_rank, args.tolerance),
                   'adaptive_hybrid': recall_metrics(adaptive_rank, args.tolerance),
                   'geometric_verification_seconds': round(geometric_time, 3),
                   'geometric_seconds_per_query': round(geometric_time/200, 4),
                   'reliable_pairs_fraction': round(successes / comparisons, 4)}
        summary['experiments'][seq] = metrics
        print(f'\n{seq} -> day_right')
        for method in ('deep_baseline', 'fixed_hybrid', 'adaptive_hybrid'):
            m = metrics[method]
            print(f'  {method:16s} R@1 {m["recall_at_1"]:.3f} | R@5 {m["recall_at_5"]:.3f} | R@10 {m["recall_at_10"]:.3f}')
        print(f'  ORB/RANSAC: {geometric_time:.1f}s over 200 queries')
        if metrics['deep_baseline']['recall_at_10'] != metrics['fixed_hybrid']['recall_at_10'] or metrics['deep_baseline']['recall_at_10'] != metrics['adaptive_hybrid']['recall_at_10']:
            raise RuntimeError('Unexpected change in Recall@10: reranking must preserve top-10 candidates')
    (output/'phase3_corrected_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (output/'phase3_corrected_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    make_examples(all_rows, image_map, output/'phase3_corrected_examples.png', args.tolerance)
    print('\nSaved results/phase3_corrected_metrics.json, phase3_corrected_predictions.csv, phase3_corrected_examples.png')


if __name__ == '__main__':
    main()
