"""Download the three Gardens Point Walking image sequences from Hugging Face.

Run from the project root:
    python download_data.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "GardensPoint"
SEQUENCES = ("day_left", "day_right", "night_right")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading to: {DATA_DIR}")
    snapshot_download(
        repo_id="medwa126/GardensPointWalking",
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=[f"{sequence}/*.jpg" for sequence in SEQUENCES],
        max_workers=4,
    )
    for sequence in SEQUENCES:
        directory = DATA_DIR / sequence
        count = len(list(directory.glob("*.jpg"))) if directory.exists() else 0
        print(f"  {sequence}: {count} JPEG files")
    print("\nRun: python validate_dataset.py")


if __name__ == "__main__":
    main()
