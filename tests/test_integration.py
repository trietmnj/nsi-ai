"""Integration test: run the real FFE pipeline on a folder of images.

Run with:
    uv run pytest tests/test_integration.py -v -s
    uv run pytest tests/test_integration.py -v -s --image-dir data/tech-ref/

Default image directory: data/gsv/
Model weights are downloaded on first run into tmp/models/.
"""

import pytest
from pathlib import Path

from nsi_ai.ffe.pipeline import FFEPipeline

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _image_ids(directory: Path) -> set[str]:
    return {p.stem for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}


def test_ffe_on_sample_images(image_dir):
    """FFH-only pass on sample images (classifier skipped)."""
    if not image_dir.exists():
        pytest.skip(f"{image_dir} not present")

    pipeline = FFEPipeline(image_dir=image_dir, work_dir="tmp", skip_classifier=True)
    df = pipeline.run()

    print("\n", df.to_string(index=False))

    assert set(df["id"]) == _image_ids(image_dir)
    assert "ffh_ft" in df.columns
    detected = df["ffh_ft"].dropna()
    print(f"\nDoor detected in {len(detected)}/{len(df)} images")


def test_ffe_raised_buildings(image_dir):
    """Full pipeline (classifier + FFH) on sample images.

    The classifier should return elevated=1 for raised structures; low
    elevated_prob across the board suggests the images don't show elevated houses.
    """
    if not image_dir.exists():
        pytest.skip(f"{image_dir} not present")

    pipeline = FFEPipeline(image_dir=image_dir, work_dir="tmp", skip_classifier=False)
    df = pipeline.run()

    print("\n", df.to_string(index=False))

    assert set(df["id"]) == _image_ids(image_dir)
    assert "elevated" in df.columns
    assert "elevated_prob" in df.columns
    assert "ffh_ft" in df.columns

    n_elevated = (df["elevated"] == 1).sum()
    detected = df["ffh_ft"].dropna()
    print(f"\nClassified elevated: {n_elevated}/{len(df)}")
    print(f"Door detected:       {len(detected)}/{len(df)}")
