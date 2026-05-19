"""Integration test: run the real FFE pipeline on sample GSV images in data/gsv/.

Run with:
    uv run pytest tests/test_integration.py -v -s

Model weights are downloaded on first run into tmp/checkpoints/.
skip_classifier=True skips the ResNet50 binary filter so only FFHPredictorKlepac runs.
"""

import pytest
import pandas as pd
from pathlib import Path

from nsi_ai.ffe.pipeline import FFEPipeline

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "gsv"


@pytest.mark.skipif(not DATA_DIR.exists(), reason="data/gsv not present")
def test_ffe_on_sample_images():
    pipeline = FFEPipeline(
        image_dir=DATA_DIR,
        work_dir="tmp",
        skip_classifier=True,
    )
    df = pipeline.run()

    print("\n", df.to_string(index=False))

    ids = set(p.stem for p in DATA_DIR.glob("*.jpg"))
    assert set(df["id"]) == ids
    assert "ffh_ft" in df.columns
    detected = df["ffh_ft"].dropna()
    print(f"\nDoor detected in {len(detected)}/{len(df)} images")
