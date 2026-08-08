"""
risk_engine/watch_object.py
----------------------------
Ad-hoc conjunction check for a single user-specified object.

Unlike conjunction_check.py (which screens a curated list of protected
station-class assets against known debris fields), this script lets
anyone check ANY public NORAD-catalogued object against the same debris
fields — no satellite-owner/operator registration required, unlike
official Conjunction Data Messages (CDMs) from Space-Track.

This is the practical demonstration of this project's real-world
access gap: official CDM access requires registered operator status.
This works from public CelesTrak data alone.

Usage:
    python risk_engine/watch_object.py --norad-id 25544 --name "ISS (ZARYA)"
    python risk_engine/watch_object.py --norad-id 20580 --name "HST"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
import numpy as np
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

CACHE = Path(__file__).parent.parent / "cache"
DEBRIS_FILES = [
    CACHE / "cosmos-2251-debris.json",
    CACHE / "iridium-33-debris.json",
]
BASE_URL = "https://celestrak.org/NORAD/elements/gp.php"

THRESHOLD_CRITICAL_KM = 5.0
THRESHOLD_WATCH_KM    = 25.0
WINDOW_DAYS  = 7
STEP_MINUTES = 1


def fetch_object(norad_id: int) -> dict:
    """Fetch a single object's OMM record directly from CelesTrak by NORAD ID."""
    resp = httpx.get(BASE_URL, params={"CATNR": norad_id, "FORMAT": "json"}, timeout=20.0)
    resp.raise_for_status()
    records = resp.json()
    if not records:
        raise ValueError(f"No object found for NORAD ID {norad_id}. "
                          f"Check the ID is correct and currently catalogued.")
    return records[0]


def load_debris() -> list[tuple[str, Satrec]]:
    out: list[tuple[str, Satrec]] = []
    for f in DEBRIS_FILES:
        with open(f, "r", encoding="utf-8") as fh:
            for rec in json.load(fh):
                sat = Satrec()
                omm.initialize(sat, rec)
                out.append((rec["OBJECT_NAME"], sat))
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Check conjunction risk for any public NORAD-catalogued object, "
                    "no operator registration required."
    )
    p.add_argument("--norad-id", required=True, type=int,
                   help="NORAD catalog ID of the object to check (e.g. 25544 for ISS)")
    p.add_argument("--name", default=None,
                   help="Display name override (defaults to CelesTrak's OBJECT_NAME)")
    args = p.parse_args()

    print(f"\nFetching object {args.norad_id} from CelesTrak...")
    rec = fetch_object(args.norad_id)
    display_name = args.name or rec["OBJECT_NAME"]
    print(f"  Found: {display_name} ({rec['OBJECT_ID']})")

    watch_sat = Satrec()
    omm.initialize(watch_sat, rec)

    print("Loading debris fields (Cosmos 2251 + Iridium 33)...")
    debris_list = load_debris()
    print(f"  {len(debris_list)} debris objects loaded")

    total_minutes = WINDOW_DAYS * 24 * 60
    steps    = np.arange(0, total_minutes + 1, STEP_MINUTES, dtype=np.float64)
    jd_fracs = steps / 1440.0

    # SGP4 epoch handling matches conjunction_check.py's convention
    jd_start, fr_start = watch_sat.jdsatepoch, watch_sat.jdsatepochF

    print(f"\nPropagating {WINDOW_DAYS}-day window at {STEP_MINUTES}-min resolution "
          f"({len(steps)} steps)...")

    watch_arr  = SatrecArray([watch_sat])
    debris_sats = [s for _, s in debris_list]
    debris_arr  = SatrecArray(debris_sats)

    jd = jd_start + jd_fracs
    fr = np.full_like(jd, fr_start)

    _, watch_pos, _  = watch_arr.sgp4(jd, fr)
    _, debris_pos, _ = debris_arr.sgp4(jd, fr)

    watch_pos  = watch_pos[0]           # (N_t, 3)
    diffs      = debris_pos - watch_pos  # (N_debris, N_t, 3)
    dists      = np.sqrt(np.nansum(diffs ** 2, axis=2))  # (N_debris, N_t)

    min_per_debris = np.nanmin(dists, axis=1)
    min_idx        = np.nanargmin(min_per_debris)
    global_min      = float(min_per_debris[min_idx])
    global_debris   = debris_list[min_idx][0]

    critical_hits = int(np.sum(np.nanmin(dists, axis=1) < THRESHOLD_CRITICAL_KM))
    watch_hits    = int(np.sum(
        (np.nanmin(dists, axis=1) >= THRESHOLD_CRITICAL_KM) &
        (np.nanmin(dists, axis=1) < THRESHOLD_WATCH_KM)
    ))

    print("\n" + "=" * 60)
    print(f"WATCH-OBJECT CONJUNCTION CHECK — {display_name}")
    print("=" * 60)
    print(f"Debris objects screened : {len(debris_list)}")
    print(f"Window                  : {WINDOW_DAYS} days, {STEP_MINUTES}-min resolution")
    print(f"CRITICAL (< {THRESHOLD_CRITICAL_KM:.0f} km)     : {critical_hits}")
    print(f"WATCH    (< {THRESHOLD_WATCH_KM:.0f} km)     : {watch_hits}")
    print(f"Closest approach found  : {global_min:.3f} km  (vs {global_debris})")
    print("=" * 60)
    print("\nNote: this check requires no satellite-operator registration —")
    print("unlike official CDMs, it works from public CelesTrak data alone.")


if __name__ == "__main__":
    main()