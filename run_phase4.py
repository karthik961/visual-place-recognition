"""CS 682 Phase 4: sensitivity, retrospective evaluation, and retrieval failures.

Run from the main visual-place-recognition repository (next to run_phase3_corrected.py).

Reuses Phase 2 embeddings and the exact Phase 3 ORB/RANSAC implementation.
Geometric results are computed only once and cached locally for parameter sweeps.
No ResNet model is loaded and no Phase 3 output files are overwritten.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from run_phase3_corrected import fuse_ranking
from src.orb_geometry import extract_local_features, verify_pair

ROOT = Path(__file__).resolve().parent
QUERY_SEQUENCES = ("day_left", "night_right")
ALL_SEQUENCES = ("day_right",) + QUERY_SEQUENCES
TOP_K = 10
BASELINE_BONUS = 0.05
BASELINE_INLIERS = 8
BASELINE_RATIO = 0.2
INLIER_SCALE = 30.0
FIXED_ALPHA = 0.7
BONUS_GRID = (0.0, 0.01, 0.025, 0.05, 0.075, 0.10)
INLIER_GRID = (4, 8, 12, 20)
RATIO_GRID = (0.1, 0.2, 0.3)


def image_paths(data_dir: Path, seq: str) -> list[Path]:
    folder = data_dir / seq
    expected = [folder / f"Image{i:03d}.jpg" for i in range(200)]
    missing = [str(p) for p in expected if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing images in {seq}, first missing: {missing[0]}")
    return expected


def load_embeddings(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Embedding cache not found: {path}. Run 'python run_retrieval.py' first.")
    with np.load(path) as inp:
        missing = set(ALL_SEQUENCES) - set(inp.files)
        if missing:
            raise ValueError(f"Embeddings missing sequences: {sorted(missing)}")
        features = {seq: np.asarray(inp[seq], dtype=np.float32) for seq in ALL_SEQUENCES}
    for seq, arr in features.items():
        if arr.shape != (200, 512) or not np.isfinite(arr).all():
            raise ValueError(f"{seq}: expected finite (200,512) embeddings, got {arr.shape}")
        if not np.allclose(np.linalg.norm(arr, axis=1), 1, atol=1e-3):
            raise ValueError(f"{seq}: embeddings are not L2-normalized")
    return features


def candidate_data(features: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for seq in QUERY_SEQUENCES:
        scores = features[seq] @ features["day_right"].T
        candidates = np.argsort(-scores, axis=1)[:, :TOP_K]
        deep = np.clip((np.take_along_axis(scores, candidates, axis=1) + 1) / 2, 0, 1)
        result[seq] = {"cosines": scores, "candidates": candidates, "deep": deep}
    return result


def cache_fingerprint(embeddings: Path, image_map: dict, orb_features: int, max_width: int) -> str:
    """Invalidate local geometry cache after input image, model cache, or parameter changes."""
    hash_ = hashlib.sha256()
    hash_.update(b"phase4-geometry-v1|orb-ratio-0.75|ransac-3.0")
    hash_.update(f"|orb={orb_features}|width={max_width}|cv2={cv2.__version__}".encode())
    for path in [embeddings, *(p for seq in ALL_SEQUENCES for p in image_map[seq])]:
        stat = path.stat()
        hash_.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return hash_.hexdigest()


def geometry_stats(
    image_map: dict, data: dict, output: Path, embeddings: Path,
    orb_features: int, max_width: int, force: bool,
) -> tuple[dict, dict]:
    cache = output / "phase4_geometry_cache.npz"
    fingerprint = cache_fingerprint(embeddings, image_map, orb_features, max_width)
    if cache.exists() and not force:
        try:
            with np.load(cache, allow_pickle=False) as saved:
                if str(saved["fingerprint"].item()) == fingerprint:
                    stats = {}
                    for seq in QUERY_SEQUENCES:
                        stats[seq] = {
                            key: saved[f"{seq}_{key}"].copy()
                            for key in ("inliers", "inlier_ratio", "good_matches")
                        }
                        if any(x.shape != (200, TOP_K) for x in stats[seq].values()):
                            raise ValueError("Invalid geometry cache array shape")
                    print(f"Reusing cached ORB/RANSAC results: {cache}")
                    return stats, {"loaded_from_cache": True, "verification_seconds": None}
        except (OSError, ValueError, KeyError, TypeError) as ex:
            print(f"Ignoring stale/unreadable geometry cache: {ex}")

    cv2.setRNGSeed(0)  # Same seed as Phase 3 corrected.
    orb = cv2.ORB_create(nfeatures=orb_features)
    print("Extracting ORB features once per image...")
    local = {}
    for seq in ALL_SEQUENCES:
        start = time.perf_counter()
        local[seq] = [extract_local_features(p, orb, max_width) for p in image_map[seq]]
        print(f"  {seq}: {len(local[seq])} images in {time.perf_counter() - start:.2f}s")

    stats = {}
    timing = {}
    for seq in QUERY_SEQUENCES:
        mats = {
            "inliers": np.zeros((200, TOP_K), dtype=np.int32),
            "inlier_ratio": np.zeros((200, TOP_K), dtype=np.float64),
            "good_matches": np.zeros((200, TOP_K), dtype=np.int32),
        }
        start = time.perf_counter()
        for i, row in enumerate(data[seq]["candidates"]):
            for j, ref_i in enumerate(row):
                result = verify_pair(local[seq][i], local["day_right"][int(ref_i)])
                for field in mats:
                    mats[field][i, j] = result[field]
            if (i + 1) % 50 == 0:
                print(f"  {seq}: {i + 1}/200 queries")
        timing[seq] = round(time.perf_counter() - start, 3)
        print(f"  {seq}: geometric verification {timing[seq]:.2f}s")
        stats[seq] = mats

    packed = {"fingerprint": np.array(fingerprint)}
    for seq in QUERY_SEQUENCES:
        for field in stats[seq]:
            packed[f"{seq}_{field}"] = stats[seq][field]
    np.savez_compressed(cache, **packed)
    print(f"Saved local cache (excluded from Git): {cache}")
    return stats, {"loaded_from_cache": False, "verification_seconds": timing}


def adaptive_ranking(candidates: np.ndarray, deep: np.ndarray, stats: dict,
                     bonus: float, min_inliers: int, min_ratio: float,
                     inlier_scale: float = INLIER_SCALE) -> np.ndarray:
    """Vectorized equivalent of Phase 3 corrected fuse_ranking(adaptive=True)."""
    if inlier_scale <= 0 or not 0 <= bonus <= 1 or not 0 <= min_ratio <= 1:
        raise ValueError("Invalid adaptive fusion parameters")
    ratios = np.clip(stats["inlier_ratio"], 0, 1)
    inliers = stats["inliers"]
    reliable = (inliers >= min_inliers) & (ratios >= min_ratio)
    confidence = np.minimum(inliers / inlier_scale, 1.0) * ratios
    scores = deep + bonus * np.where(reliable, confidence, 0)
    order = np.argsort(-scores, axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


def fixed_ranking(candidates: np.ndarray, deep: np.ndarray, stats: dict,
                  alpha: float = FIXED_ALPHA) -> np.ndarray:
    fused = alpha * deep + (1 - alpha) * np.clip(stats["inlier_ratio"], 0, 1)
    return np.take_along_axis(candidates, np.argsort(-fused, axis=1, kind="stable"), axis=1)


def recall(ranking: np.ndarray, indices: np.ndarray, tolerance: int = 2) -> dict:
    chosen = ranking[indices]
    correct = np.abs(chosen - indices[:, None]) <= tolerance
    return {f"recall_at_{k}": round(float(correct[:, :k].any(axis=1).mean()), 4)
            for k in (1, 5, 10)}


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def choose_parameters(grid_rows: list[dict]) -> dict:
    """Select a single config across both conditions using DEVELOPMENT queries only."""
    grouped = {}
    for row in grid_rows:
        if row["subset"] != "development":
            continue
        key = (row["geom_bonus"], row["min_inliers"], row["min_ratio"])
        grouped.setdefault(key, []).append(row)
    if not grouped or any(len(v) != 2 for v in grouped.values()):
        raise ValueError("Expected exactly two development sequences per parameter setting")
    def quality(item):
        (bonus, inliers, ratio), rows = item
        r1 = np.mean([float(r["recall_at_1"]) for r in rows])
        r5 = np.mean([float(r["recall_at_5"]) for r in rows])
        # Deterministic tie-breaker prefers Phase 3 reference values.
        return (r1, r5, -abs(bonus - BASELINE_BONUS),
                -abs(inliers - BASELINE_INLIERS), -abs(ratio - BASELINE_RATIO))
    best, rows = max(grouped.items(), key=quality)
    return {
        "geom_bonus": best[0], "min_inliers": best[1], "min_ratio": best[2],
        "development_mean_recall_at_1": round(float(np.mean([r["recall_at_1"] for r in rows])), 4),
    }


def classify_top1(deep_index: int, adaptive_index: int, query_index: int,
                  tolerance: int) -> str:
    d = abs(deep_index - query_index) <= tolerance
    a = abs(adaptive_index - query_index) <= tolerance
    if a and not d:
        return "improved"
    if d and not a:
        return "worsened"
    if a and d:
        return "both_correct"
    return "both_wrong"


def visualize_cases(rows: list[dict], image_map: dict, output: Path, tolerance: int) -> None:
    """Show selected representative examples of each category for each condition."""
    chosen = []
    for seq in QUERY_SEQUENCES:
        for label in ("improved", "worsened", "both_wrong", "both_correct"):
            sample = next((r for r in rows if r["sequence"] == seq and r["category"] == label), None)
            if sample is not None:
                chosen.append(sample)
    if not chosen:
        raise ValueError("No failure-case examples could be selected")
    thumb_w, thumb_h, text_h, start_x, gap = 250, 143, 35, 150, 14
    row_h = thumb_h + text_h + 13
    canvas = Image.new("RGB", (start_x + 3 * (thumb_w + gap) + 16,
                               43 + len(chosen) * row_h + 8), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((start_x, 15), "QUERY", fill="black")
    draw.text((start_x + thumb_w + gap, 15), "DEEP TOP-1", fill="black")
    draw.text((start_x + 2 * (thumb_w + gap), 15), "ADAPTIVE TOP-1", fill="black")
    for n, r in enumerate(chosen):
        y = 43 + n * row_h
        seq, qi = r["sequence"], int(r["query_index"])
        draw.text((7, y + 6), f"{seq}\nImage{qi:03d}\n{r['category'].upper()}", fill="black")
        items = [
            (image_map[seq][qi], "QUERY"),
            (image_map["day_right"][int(r["deep_top1"])],
             f"Image{int(r['deep_top1']):03d}  {'OK' if abs(int(r['deep_top1'])-qi)<=tolerance else 'MISS'}"),
            (image_map["day_right"][int(r["adaptive_top1"])],
             f"Image{int(r['adaptive_top1']):03d}  {'OK' if abs(int(r['adaptive_top1'])-qi)<=tolerance else 'MISS'}"),
        ]
        for j, (path, caption) in enumerate(items):
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
                x = start_x + j * (thumb_w + gap)
                canvas.paste(im, (x + (thumb_w - im.width) // 2,
                                  y + (thumb_h - im.height) // 2))
            draw.text((x + 3, y + thumb_h + 5), caption, fill="black")
    canvas.save(output)


def plot_comparison(full_metrics: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("deep_baseline", "fixed_hybrid", "adaptive_phase3", "adaptive_selected")
    labels = ("ResNet18", "Fixed hybrid", "Adaptive (0.05)", "Selected on dev")
    metric_names = ("recall_at_1", "recall_at_5", "recall_at_10")
    axis_labels = [f"{s.replace('_', ' ').title()}\nR@{k.rsplit('_', 1)[-1]}"
                   for s in QUERY_SEQUENCES for k in metric_names]
    x = np.arange(len(axis_labels))
    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    for i, (method, label) in enumerate(zip(methods, labels)):
        values = [100 * full_metrics[s][method][metric]
                  for s in QUERY_SEQUENCES for metric in metric_names]
        ax.bar(x + (i - 1.5) * 0.18, values, width=0.18, label=label)
    ax.set_xticks(x, axis_labels)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Gardens Point — Phase 4 full-dataset retrieval comparison")
    ax.legend(loc="upper center", ncol=4)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def plot_bonus_sweep(rows: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for seq in QUERY_SEQUENCES:
        chosen = sorted((r for r in rows if r["subset"] == "development" and
                         r["sequence"] == seq and r["min_inliers"] == BASELINE_INLIERS and
                         r["min_ratio"] == BASELINE_RATIO), key=lambda z: z["geom_bonus"])
        ax.plot([r["geom_bonus"] for r in chosen],
                [100 * r["recall_at_1"] for r in chosen], marker="o",
                label=f"{seq} (development)")
    ax.set_xlabel("Geometric bonus")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("Parameter sensitivity: bonus, with 8 inliers and ratio 0.2")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict:
    if args.tolerance < 0 or args.orb_features < 4 or args.max_width < 32:
        raise ValueError("Invalid tolerance or ORB parameters")
    image_map = {seq: image_paths(args.data_dir, seq) for seq in ALL_SEQUENCES}
    features = load_embeddings(args.embeddings)
    data = candidate_data(features)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    geometric, geom_timing = geometry_stats(
        image_map, data, args.output_dir, args.embeddings,
        args.orb_features, args.max_width, args.recompute_geometry)

    # Retrospective contiguous query split. Every reference image remains eligible.
    # The earlier project already examined ALL 400 query results; this is not blind testing.
    subsets = {
        "development": np.arange(0, 100, dtype=np.int64),
        "retrospective_evaluation": np.arange(100, 200, dtype=np.int64),
        "all_queries": np.arange(200, dtype=np.int64),
    }
    rows = []
    configs = list(itertools.product(BONUS_GRID, INLIER_GRID, RATIO_GRID))
    print(f"Evaluating {len(configs)} configurations without rerunning ORB/RANSAC...")
    for bonus, min_inliers, min_ratio in configs:
        for seq in QUERY_SEQUENCES:
            item = data[seq]
            ranks = adaptive_ranking(item["candidates"], item["deep"], geometric[seq],
                                     bonus, min_inliers, min_ratio)
            for split, indices in subsets.items():
                rows.append({
                    "sequence": seq, "subset": split,
                    "geom_bonus": bonus, "min_inliers": min_inliers,
                    "min_ratio": min_ratio,
                    **recall(ranks, indices, args.tolerance),
                })

    grid_file = args.output_dir / "phase4_parameter_sensitivity.csv"
    write_csv(grid_file,
              ["sequence", "subset", "geom_bonus", "min_inliers", "min_ratio",
               "recall_at_1", "recall_at_5", "recall_at_10"], rows)
    selected = choose_parameters(rows)
    print(f"Selected on development only: bonus={selected['geom_bonus']}, "
          f"inliers>={selected['min_inliers']}, ratio>={selected['min_ratio']}")

    methods = {}
    full_metrics = {}
    all_cases = []
    case_counts = {}
    for seq in QUERY_SEQUENCES:
        item = data[seq]
        stats = geometric[seq]
        methods[seq] = {
            "deep_baseline": item["candidates"],
            "fixed_hybrid": fixed_ranking(item["candidates"], item["deep"], stats),
            "adaptive_phase3": adaptive_ranking(item["candidates"], item["deep"], stats,
                                                BASELINE_BONUS, BASELINE_INLIERS, BASELINE_RATIO),
            "adaptive_selected": adaptive_ranking(item["candidates"], item["deep"], stats,
                                                  selected["geom_bonus"], selected["min_inliers"],
                                                  selected["min_ratio"]),
        }
        full_metrics[seq] = {method: recall(rank, subsets["all_queries"], args.tolerance)
                             for method, rank in methods[seq].items()}
        for i in range(200):
            deep_index = int(methods[seq]["deep_baseline"][i, 0])
            # Compare the unchanged Phase 3 corrected method to the original baseline.
            adaptive_index = int(methods[seq]["adaptive_phase3"][i, 0])
            category = classify_top1(deep_index, adaptive_index, i, args.tolerance)
            all_cases.append({
                "sequence": seq, "query_index": i, "category": category,
                "deep_top1": deep_index, "adaptive_top1": adaptive_index,
                "selected_top1": int(methods[seq]["adaptive_selected"][i, 0]),
            })
        case_counts[seq] = {kind: sum(r["sequence"] == seq and r["category"] == kind
                                      for r in all_cases)
                            for kind in ("improved", "worsened", "both_correct", "both_wrong")}

    # Verify vectorized results exactly match actual Phase 3 fuse_ranking on sample queries.
    for seq in QUERY_SEQUENCES:
        for i in (0, 25, 50, 90, 150, 199):
            stats = [{"inliers": int(geometric[seq]["inliers"][i, j]),
                      "inlier_ratio": float(geometric[seq]["inlier_ratio"][i, j])}
                     for j in range(TOP_K)]
            expected = fuse_ranking(data[seq]["candidates"][i], data[seq]["cosines"][i],
                                    stats, FIXED_ALPHA, BASELINE_INLIERS, BASELINE_RATIO,
                                    adaptive=True, geom_bonus=BASELINE_BONUS,
                                    inlier_scale=INLIER_SCALE)
            if not np.array_equal(expected, methods[seq]["adaptive_phase3"][i]):
                raise AssertionError(f"Phase 3 equivalence check failed: {seq} query {i}")

    cases_file = args.output_dir / "phase4_failure_cases.csv"
    write_csv(cases_file, ["sequence", "query_index", "category", "deep_top1",
                           "adaptive_top1", "selected_top1"], all_cases)
    plot_comparison(full_metrics, args.output_dir / "phase4_comparison.png")
    plot_bonus_sweep(rows, args.output_dir / "phase4_bonus_sensitivity.png")
    visualize_cases(all_cases, image_map, args.output_dir / "phase4_failure_cases.png", args.tolerance)

    summary = {
        "dataset": "Gardens Point Walking",
        "reference": "day_right",
        "evaluation": "Approximate same-index +/-2 frame tolerance, not GPS ground truth",
        "important_limitations": [
            "All 400 query results influenced earlier Phase 3 development; this is not a blinded test.",
            "Development query indices 0-99 and retrospective evaluation indices 100-199.",
            "Both splits traverse the same route and share the full 200-image reference gallery.",
            "Do not interpret these results as generalization to an independent dataset.",
        ],
        "settings": {
            "top_k": TOP_K, "tolerance": args.tolerance,
            "orb_features": args.orb_features, "max_width": args.max_width,
            "inlier_scale": INLIER_SCALE, "fixed_alpha": FIXED_ALPHA,
            "phase3_reference": {"geom_bonus": BASELINE_BONUS,
                                 "min_inliers": BASELINE_INLIERS, "min_ratio": BASELINE_RATIO},
            "parameters_tested": len(configs),
        },
        "selected_on_development": selected,
        "geometry_cache": geom_timing,
        "full_dataset_metrics": full_metrics,
        "metrics_by_subset": {
            split: {
                seq: {name: recall(rank, indices, args.tolerance)
                      for name, rank in methods[seq].items()}
                for seq in QUERY_SEQUENCES}
            for split, indices in subsets.items() if split != "all_queries"
        },
        "phase3_default_top1_transitions": case_counts,
        "artifacts": [grid_file.name, cases_file.name,
                      "phase4_comparison.png", "phase4_bonus_sensitivity.png",
                      "phase4_failure_cases.png"],
    }
    summary_path = args.output_dir / "phase4_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nFULL-DATASET Recall@1 (exploratory):")
    for seq in QUERY_SEQUENCES:
        m = full_metrics[seq]
        print(f"  {seq}: Deep {m['deep_baseline']['recall_at_1']:.3f}, "
              f"Fixed {m['fixed_hybrid']['recall_at_1']:.3f}, "
              f"Phase3 Adaptive {m['adaptive_phase3']['recall_at_1']:.3f}, "
              f"Dev-selected {m['adaptive_selected']['recall_at_1']:.3f}")
        print(f"    Phase3 correction transitions: {case_counts[seq]}")
    print(f"\nSaved Phase 4 artifacts to {args.output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "GardensPoint")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "results" / "phase2_embeddings.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument("--orb-features", type=int, default=1000)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--recompute-geometry", action="store_true",
                        help="Ignore cached ORB/RANSAC results and recalculate matches")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
