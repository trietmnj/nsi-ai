# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

`nsi-ai` estimates First Floor Elevation (FFE) of buildings from Google Street View images. It is part of a broader NSI (National Structure Inventory) toolchain at `../nsi-*`. The upstream ML framework is BRAILS++, cloned at `../BrailsPlusPlus`.

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/). `detectron2` builds from source and requires `torch` to already be present, so install in two steps:

```bash
uv venv
uv pip install torch torchvision
uv sync --extra dev
```

`pyproject.toml` sets `torch<2.3` because torch dropped x86_64 macOS wheels in 2.3. `[tool.uv] no-build-isolation-package = ["detectron2"]` tells uv not to isolate detectron2's build so it can find the already-installed torch.

Requires Python ≥ 3.10. A Google Street View Static API key is needed to fetch images.

## Commands

```bash
uv run pytest tests/                        # run all tests
uv run pytest tests/test_integration.py -v -s  # integration test on data/gsv/ samples
uv run ruff check nsi_ai/                   # lint
uv run ruff format nsi_ai/                  # format (line length 100)
```

Run the full pipeline:
```bash
uv run python scripts/run_ffe.py --locations buildings.csv --api-key KEY --output results.csv
```
`buildings.csv` requires columns `id, lat, lon`.

Sample GSV images for local testing live in `data/gsv/` (a.jpg–d.jpg). The integration test runs the real model on them with no API key needed.

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
- Model weights are downloaded on first use into `work_dir/models/`. Default `work_dir` is `tmp/`.

## Known third-party issues

**BRAILS 4.2.0 — `FFHPredictorKlepac.predict()` missing return**: the method builds a `predictions` dict but never returns it, so it always returns `None`. Fixed by appending `return predictions` to the installed file at `.venv/lib/python3.11/site-packages/brails/processors/ffh_predictor_klepac/ffh_predictor_klepac.py`. Must be re-applied after any `uv sync` that upgrades BRAILS.

**CPU-only machines**: BRAILS model weights are serialized with CUDA tensors. `tests/conftest.py` applies two session-scoped patches to make the test suite work without a GPU: `torch.load` defaults to `map_location="cpu"`, and `detectron2.engine.defaults.DefaultPredictor.__call__` remaps the pickled CUDA device to CPU before inference.
