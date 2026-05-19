#!/usr/bin/env python3
"""
Run the FFE pipeline on a set of buildings.

Usage:
    python scripts/run_ffe.py \
        --locations buildings.csv \
        --api-key YOUR_GOOGLE_API_KEY \
        --output results.csv

buildings.csv must have columns: id, lat, lon
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nsi_ai.ffe.gsv import fetch_images
from nsi_ai.ffe.pipeline import FFEPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--locations", required=True, help="CSV with id,lat,lon columns")
    p.add_argument("--api-key", required=True, help="Google Street View API key")
    p.add_argument("--output", default="ffe_results.csv")
    p.add_argument("--image-dir", default="tmp/images")
    p.add_argument("--work-dir", default="tmp")
    p.add_argument("--skip-classifier", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Process only first N buildings")
    return p.parse_args()


def main():
    args = parse_args()

    locs = pd.read_csv(args.locations)
    if args.limit:
        locs = locs.head(args.limit)
    locations = locs[["id", "lat", "lon"]].to_dict("records")

    print(f"Fetching GSV images for {len(locations)} buildings...")
    gsv_results = fetch_images(locations, args.api_key, args.image_dir)

    covered = [lid for lid, path in gsv_results.items() if path is not None]
    no_coverage = [lid for lid, path in gsv_results.items() if path is None]
    print(f"  Coverage: {len(covered)} / {len(locations)} buildings")
    if no_coverage:
        print(f"  No GSV coverage: {no_coverage[:5]}{'...' if len(no_coverage) > 5 else ''}")

    if not covered:
        print("No images to process.")
        return

    pipeline = FFEPipeline(
        image_dir=args.image_dir,
        work_dir=args.work_dir,
        skip_classifier=args.skip_classifier,
    )
    results = pipeline.run(ids=covered)

    results.to_csv(args.output, index=False)
    print(f"\nResults written to {args.output}")
    print(results.describe())


if __name__ == "__main__":
    main()
