# nsi-ai

First Floor Elevation (FFE) estimation from Google Street View images, built on [BRAILS++](https://github.com/NHERI-SimCenter/BrailsPlusPlus).

## Approach

Two-stage pipeline:

1. **`FoundationElevationClassifier`** — ResNet50 binary classifier that flags whether a building is elevated (on piles, piers, or posts). F1 ≈ 72%.
2. **`FFHPredictorKlepac`** — Detectron2 model that detects the house and door bounding boxes, then uses standard door height (80 in) as a scale reference to compute First Floor Height in feet.

Returns `None` for any building where a door is not detected within the house bounding box (no door visible, obstructed view, etc.).

## Setup

```bash
cd nsi-ai
python -m venv .venv && source .venv/bin/activate

# detectron2 must be installed from source
pip install "detectron2 @ git+https://github.com/facebookresearch/detectron2.git"
pip install -e ".[dev]"
```

You need a [Google Street View Static API key](https://developers.google.com/maps/documentation/streetview/get-api-key).

## Usage

Prepare a CSV with `id,lat,lon` columns, then:

```bash
python scripts/run_ffe.py \
    --locations buildings.csv \
    --api-key YOUR_KEY \
    --output results.csv \
    --limit 100
```

Output CSV columns: `id`, `elevated` (0/1), `elevated_prob`, `ffh_ft`.

`ffh_ft` is `None` when FFH could not be computed (no visible door).

## Tests

```bash
pytest tests/
```

## Limitations

- `ffh_ft` assumes standard 80-inch interior door height — non-standard doors, garage doors, and perspective distortion reduce accuracy.
- Coverage depends on Google Street View availability and image angle.
- For ground-truth validation, cross-reference with USGS 3DEP LiDAR where available.
