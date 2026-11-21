"""Verify data integrity and generate a three-sequence comparison contact sheet.

Run from the project root:
    python validate_dataset.py
Optional:
    python validate_dataset.py --data-dir /path/to/GardensPoint
"""
import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

SEQUENCES = ("day_left", "day_right", "night_right")
EXPECTED_COUNT = 200
SAMPLE_INDICES = (0, 40, 80, 120, 160, 199)


def check_sequence(directory: Path) -> dict:
    """Check file count, exact expected names, decoding, and image sizes."""
    expected = {f"Image{i:03d}.jpg" for i in range(EXPECTED_COUNT)}
    files = list(directory.glob("*.jpg")) if directory.is_dir() else []
    names = {p.name for p in files}
    corrupted = []
    sizes = Counter()

    for path in sorted(files):
        try:
            with Image.open(path) as image:
                image.load()  # Forces pixel decoding, not just header inspection
                sizes[f"{image.width}x{image.height}"] += 1
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            corrupted.append({"file": path.name, "error": str(exc)})

    return {
        "directory_exists": directory.is_dir(),
        "count": len(files),
        "expected_count": EXPECTED_COUNT,
        "missing": sorted(expected - names),
        "unexpected": sorted(names - expected),
        "corrupted": corrupted,
        "image_dimensions": dict(sizes),
        "passed": (len(files) == EXPECTED_COUNT and names == expected and not corrupted),
    }


def make_contact_sheet(data_dir: Path, destination: Path) -> None:
    """Show frame-index triplets side by side. Visual verification is still required."""
    cell_w, cell_h = 350, 220
    header_h, row_label_h = 44, 26
    width = cell_w * 3
    height = header_h + len(SAMPLE_INDICES) * (cell_h + row_label_h)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    for j, sequence in enumerate(SEQUENCES):
        x = j * cell_w
        draw.rectangle((x, 0, x + cell_w - 1, header_h - 1), fill="#253858")
        draw.text((x + 12, 13), sequence.replace("_", " ").upper(), fill="white")

    for i, frame in enumerate(SAMPLE_INDICES):
        top = header_h + i * (cell_h + row_label_h)
        for j, sequence in enumerate(SEQUENCES):
            path = data_dir / sequence / f"Image{frame:03d}.jpg"
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((cell_w - 8, cell_h - 8))
                x = j * cell_w + (cell_w - image.width) // 2
                y = top + (cell_h - image.height) // 2
                sheet.paste(image, (x, y))
            draw.text((j * cell_w + 12, top + cell_h + 5), f"Image{frame:03d}.jpg", fill="#202020")

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "GardensPoint")
    parser.add_argument("--output-dir", type=Path, default=root / "results")
    args = parser.parse_args()

    checks = {}
    for sequence in SEQUENCES:
        result = check_sequence(args.data_dir / sequence)
        checks[sequence] = result
        status = "PASS" if result["passed"] else "FAIL"
        print(f"{status} | {sequence}: {result['count']}/{EXPECTED_COUNT} images")
        print(f"       resolutions: {result['image_dimensions']}")
        if result["missing"]:
            print(f"       missing (first 10): {result['missing'][:10]}")
        if result["unexpected"]:
            print(f"       unexpected (first 10): {result['unexpected'][:10]}")
        if result["corrupted"]:
            print(f"       unreadable (first 3): {result['corrupted'][:3]}")

    all_good = all(result["passed"] for result in checks.values())
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "medwa126/GardensPointWalking",
        "data_dir": str(args.data_dir.resolve()),
        "all_sequences_passed": all_good,
        "sequences": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "dataset_validation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved validation report: {report_path}")

    if not all_good:
        print("Validation failed. Inspect missing/unreadable files, then rerun the download.")
        raise SystemExit(1)

    montage = args.output_dir / "dataset_comparison.png"
    make_contact_sheet(args.data_dir, montage)
    print(f"Saved comparison image: {montage}")
    print("\nSUCCESS: All 600 images were found and decoded.")
    print("IMPORTANT: Open the comparison image and visually inspect frame alignment.")
    print("Identical frame indices are approximate position matches, not pixel-level ground truth.")


if __name__ == "__main__":
    main()
