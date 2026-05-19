# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`nsi-ai` estimates First Floor Elevation (FFE) of buildings from Google Street View images. It is part of a broader NSI (National Structure Inventory) toolchain at `../nsi-*`. The upstream ML framework is BRAILS++, cloned at `../BrailsPlusPlus`.

## Setup

```bash
# detectron2 must be installed from source before the rest
pip install "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. A Google Street View Static API key is needed to fetch images.

## Commands

```bash
pytest tests/                        # run all tests
pytest tests/test_pipeline.py::test_pipeline_discovers_images  # single test
ruff check nsi_ai/                   # lint
ruff format nsi_ai/                  # format (line length 100)
```

Run the full pipeline:
```bash
python scripts/run_ffe.py --locations buildings.csv --api-key KEY --output results.csv
```
`buildings.csv` requires columns `id, lat, lon`.

## Architecture

The pipeline has two sequential stages, both imported lazily inside methods to avoid loading heavy model weights at import time:

**Stage 1 — `FoundationElevationClassifier` (BRAILS++)**: ResNet50 binary classifier. Returns `elevated` (0/1) and a confidence probability. Can be skipped via `skip_classifier=True` when all buildings are known to be elevated.

**Stage 2 — `FFHPredictorKlepac` (BRAILS++)**: Detectron2 object detector that finds house and door bounding boxes, then uses standard door height (80 in) as a pixel-to-feet scale reference to compute FFH. Returns `None` when no door is detected inside the house box — this is common and callers must handle it.

`FFEPipeline.run()` returns a DataFrame: `id, elevated, elevated_prob, ffh_ft`.

**Image convention**: images are named `<id>.jpg` and live in a flat directory. `_build_image_set()` constructs BRAILS++ `ImageSet` objects using an inline anonymous class for image entries — this matches the BRAILS++ `ImageSet` API without requiring a full BRAILS++ import.

**GSV fetching** (`nsi_ai/ffe/gsv.py`): checks metadata endpoint first to skip locations with no Street View coverage, then downloads 640×640 images. Returns `None` for uncovered locations; callers track coverage rate before invoking the pipeline.

## Key constraints

- `ffh_ft` is `None` for any building where Detectron2 cannot detect a door inside the house bounding box. Plan for substantial `None` rates in practice (garages, obstructed views, oblique angles).
- The 80-inch door assumption is hardcoded in `FFHPredictorKlepac._calculate_ffh()` inside BRAILS++. Non-standard doors and perspective distortion are the main sources of error.
- Model weights are downloaded on first use into `work_dir/checkpoints/`. Default `work_dir` is `tmp/`.
