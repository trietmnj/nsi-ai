"""Smoke tests for FFEPipeline that don't require model weights or a GSV key."""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

from nsi_ai.ffe.pipeline import FFEPipeline


@pytest.fixture
def fake_image_dir(tmp_path):
    for bid in ["a", "b"]:
        (tmp_path / f"{bid}.jpg").write_bytes(b"fake")
    (tmp_path / "c.png").write_bytes(b"fake")
    return tmp_path


def test_pipeline_raises_on_empty_dir(tmp_path):
    pipeline = FFEPipeline(image_dir=tmp_path, work_dir=tmp_path)
    with pytest.raises(ValueError, match="No images found"):
        pipeline.run()


def test_pipeline_discovers_images(fake_image_dir):
    pipeline = FFEPipeline(image_dir=fake_image_dir, work_dir=fake_image_dir, skip_classifier=True)

    fake_ffh = {"a": 1.5, "b": None, "c": 2.0}

    with patch(
        "brails.processors.ffh_predictor_klepac.ffh_predictor_klepac.FFHPredictorKlepac"
    ) as MockFFH:
        mock_instance = MagicMock()
        mock_instance.predict.return_value = fake_ffh
        MockFFH.return_value = mock_instance

        df = pipeline.run()

    assert set(df["id"]) == {"a", "b", "c"}
    assert pd.isna(df.loc[df["id"] == "b", "ffh_ft"].iloc[0])
    assert df.loc[df["id"] == "a", "ffh_ft"].iloc[0] == pytest.approx(1.5)
