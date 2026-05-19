"""FFE estimation pipeline combining foundation classification and FFH prediction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from brails.types.image_set import ImageSet


class FFEPipeline:
    """
    Estimate First Floor Elevation (FFE) from street-view images.

    Two-stage pipeline:
      1. FoundationElevationClassifier — binary filter (elevated vs not).
      2. FFHPredictorKlepac — geometric FFH in feet for each building.

    The classifier result is included alongside the measured FFH so callers
    can apply their own thresholding or imputation logic for None predictions.

    Args:
        image_dir: Directory containing per-building JPG images named <id>.jpg.
        work_dir:  Scratch directory for model weights and intermediate files.
        skip_classifier: Skip the binary classifier pre-filter and run FFH on
                         every image. Useful when all buildings are suspected
                         elevated.
    """

    def __init__(
        self,
        image_dir: str | Path,
        work_dir: str | Path = "tmp",
        skip_classifier: bool = False,
    ):
        self.image_dir = Path(image_dir)
        self.work_dir = Path(work_dir)
        self.skip_classifier = skip_classifier

        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run(self, ids: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Run the full pipeline for images in image_dir.

        Args:
            ids: Subset of building IDs to process. Defaults to all images
                 found in image_dir.

        Returns:
            DataFrame with columns:
              id, elevated (0/1/None), elevated_prob, ffh_ft
        """
        if ids is None:
            ids = [
                p.stem
                for p in sorted(self.image_dir.iterdir())
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]

        if not ids:
            raise ValueError(f"No images found in {self.image_dir}")

        image_set = self._build_image_set(ids)

        elevated_map: dict[str, Optional[int]] = {i: None for i in ids}
        elevated_prob_map: dict[str, Optional[float]] = {i: None for i in ids}

        if not self.skip_classifier:
            elevated_map, elevated_prob_map = self._run_classifier(image_set, ids)

        ffh_map = self._run_ffh_predictor(image_set, ids)

        rows = []
        for bid in ids:
            rows.append(
                {
                    "id": bid,
                    "elevated": elevated_map.get(bid),
                    "elevated_prob": elevated_prob_map.get(bid),
                    "ffh_ft": ffh_map.get(bid),
                }
            )

        return pd.DataFrame(rows)

    def _build_image_set(self, ids: list[str]) -> ImageSet:
        images = {}
        for bid in ids:
            for ext in (".jpg", ".jpeg", ".png"):
                img_path = self.image_dir / f"{bid}{ext}"
                if img_path.exists():
                    images[bid] = type("_Img", (), {"filename": f"{bid}{ext}"})()
                    break
        image_set = ImageSet()
        image_set.dir_path = str(self.image_dir)
        image_set.images = images
        return image_set

    def _run_classifier(
        self, image_set: ImageSet, ids: list[str]
    ) -> tuple[dict[str, Optional[int]], dict[str, Optional[float]]]:
        from brails.processors.foundation_classifier.foundation_classifier import (
            FoundationElevationClassifier,
        )

        clf = FoundationElevationClassifier(input_data={"workDir": str(self.work_dir)})
        # returns {key: 'Elevated' | 'Non-elevated'}
        raw: dict[str, str] = clf.predict(image_set)

        elevated: dict[str, Optional[int]] = {}
        prob: dict[str, Optional[float]] = {}

        for bid, label in raw.items():
            elevated[bid] = 1 if label == "Elevated" else 0
            prob[bid] = None  # classifier doesn't expose confidence in its return value

        return elevated, prob

    def _run_ffh_predictor(
        self, image_set: ImageSet, ids: list[str]
    ) -> dict[str, Optional[float]]:
        from brails.processors.ffh_predictor_klepac.ffh_predictor_klepac import (
            FFHPredictorKlepac,
        )

        predictor = FFHPredictorKlepac()
        return predictor.predict(image_set)
