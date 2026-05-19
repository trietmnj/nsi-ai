"""Google Street View image fetching for a list of (lat, lon) points."""

import os
import time
from pathlib import Path
from typing import Optional

import requests


GSV_METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"
GSV_IMAGE_URL = "https://maps.googleapis.com/maps/api/streetview"

DEFAULT_SIZE = "640x640"
DEFAULT_FOV = 90
DEFAULT_PITCH = 0


def _metadata(lat: float, lon: float, api_key: str) -> dict:
    r = requests.get(
        GSV_METADATA_URL,
        params={"location": f"{lat},{lon}", "key": api_key},
        timeout=10,
    )
    r.raise_for_status()
    return r.json()


def fetch_images(
    locations: list[dict],
    api_key: str,
    out_dir: str | Path,
    size: str = DEFAULT_SIZE,
    fov: int = DEFAULT_FOV,
    pitch: int = DEFAULT_PITCH,
    heading: Optional[int] = None,
    delay: float = 0.05,
) -> dict[str, Optional[str]]:
    """
    Download one GSV image per location into out_dir.

    locations: list of dicts with at least 'id', 'lat', 'lon' keys.
    Returns a mapping of id -> saved file path (None if no GSV coverage).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Optional[str]] = {}

    for loc in locations:
        lid = str(loc["id"])
        lat, lon = loc["lat"], loc["lon"]

        meta = _metadata(lat, lon, api_key)
        if meta.get("status") != "OK":
            results[lid] = None
            continue

        params: dict = {
            "location": f"{lat},{lon}",
            "size": size,
            "fov": fov,
            "pitch": pitch,
            "key": api_key,
        }
        if heading is not None:
            params["heading"] = heading

        r = requests.get(GSV_IMAGE_URL, params=params, timeout=15)
        r.raise_for_status()

        img_path = out_dir / f"{lid}.jpg"
        img_path.write_bytes(r.content)
        results[lid] = str(img_path)

        time.sleep(delay)

    return results
