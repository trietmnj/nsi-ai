"""Integration test: run the real FFE pipeline on sample GSV images in data/gsv/.

Run with:
    uv run pytest tests/test_integration.py -v -s

Model weights are downloaded on first run into tmp/models/.
"""

import pytest
import pandas as pd
from pathlib import Path

from nsi_ai.ffe.pipeline import FFEPipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gsv"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def _image_ids(directory: Path) -> set[str]:
    return {p.stem for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/gsv not present")
def test_ffe_on_sample_images():
    """FFH-only pass on sample images (classifier skipped)."""
    pipeline = FFEPipeline(
        image_dir=DATA_DIR,
        work_dir="tmp",
        skip_classifier=True,
    )
    df = pipeline.run()

    print("\n", df.to_string(index=False))

    assert set(df["id"]) == _image_ids(DATA_DIR)
    assert "ffh_ft" in df.columns
    detected = df["ffh_ft"].dropna()
    print(f"\nDoor detected in {len(detected)}/{len(df)} images")


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/gsv not present")
def test_ffe_raised_buildings():
    """Full pipeline (classifier + FFH) on the same sample images.

    Checks that the classifier runs and produces elevated/elevated_prob for each
    image. The classifier should return elevated=1 for raised structures; low
    elevated_prob across the board suggests the images don't show elevated houses.
    """
    pipeline = FFEPipeline(
        image_dir=DATA_DIR,
        work_dir="tmp",
        skip_classifier=False,
    )
    df = pipeline.run()

    print("\n", df.to_string(index=False))

    assert set(df["id"]) == _image_ids(DATA_DIR)
    assert "elevated" in df.columns
    assert "elevated_prob" in df.columns
    assert "ffh_ft" in df.columns

    n_elevated = (df["elevated"] == 1).sum()
    detected = df["ffh_ft"].dropna()
    print(f"\nClassified elevated: {n_elevated}/{len(df)}")
    print(f"Door detected:       {len(detected)}/{len(df)}")
