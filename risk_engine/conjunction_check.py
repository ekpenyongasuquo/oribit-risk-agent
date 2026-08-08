"""
Conjunction check: Cosmos-2251 + Iridium-33 debris vs crewed/active stations.

Strategy
--------
Full 7-day window at 1-minute steps using vectorised NumPy distance
evaluation. Every satellite is propagated once into an (N_steps, 3)
position array; pair minimum distances are computed with array operations,
cutting per-step Python overhead to near-zero.

Alert tiers
-----------
  CRITICAL : min distance < THRESHOLD_CRITICAL_KM   (< 5 km)
  WATCH    : min distance < THRESHOLD_WATCH_KM       (< 25 km)
"""

import json
import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
from sgp4 import omm
from sgp4.api import Satrec, SatrecArray

CACHE              = Path(__file__).parent.parent / "cache"
DEBRIS_FILES       = [
    CACHE / "cosmos-2251-debris.json",
    CACHE / "iridium-33-debris.json",
]
STATIONS_FILE      = CACHE / "stations.json"

# ── alert thresholds ──────────────────────────────────────────────────────────
THRESHOLD_CRITICAL_KM = 5.0
THRESHOLD_WATCH_KM    = 25.0

# ── propagation parameters ────────────────────────────────────────────────────
WINDOW_DAYS     = 7
STEP_MINUTES    = 1

# ── protected assets: station modules + docked/en-route crew/cargo vehicles ───
PROTECTED_STATIONS = {
    "ISS (ZARYA)",
    "POISK",
    "ISS (NAUKA)",
    "CSS (TIANHE)",
    "CSS (WENTIAN)",
    "CSS (MENGTIAN)",
    "CREW DRAGON 12",
    "PROGRESS-MS 33",
    "PROGRESS-MS 34",
    "CYGNUS NG-24",
    "TIANZHOU-10",
    "SHENZHOU-23 (SZ-23)",
    "SOYUZ-MS 29",
}


def load_satrecs(path: Path, allowed: set[str] | None = None) -> list[tuple[str, Satrec]]:
    """Load OMM JSON and return (name, Satrec) pairs.

    If *allowed* is given, only records whose OBJECT_NAME is in that set
    are loaded; everything else is silently skipped.
    """
    records = json.loads(path.read_text())
    result = []
    for rec in records:
        name = rec.get("OBJECT_NAME", "")
        if allowed is not None and name not in allowed:
            continue
        sat = Satrec()
        try:
            omm.initialize(sat, rec)
        except Exception as exc:
            print(f"  [warn] skipping {rec.get('OBJECT_NAME', '?')}: {exc}",
                  file=sys.stderr)
            continue
        result.append((name, sat))
    return result


def propagate_batch(
    satrecs: list[Satrec],
    jd_whole: float,
    jd_fracs: np.ndarray,
) -> np.ndarray:
    """
    Propagate a list of Satrec objects over a time grid in one vectorised call.

    Returns an (N_objects, N_steps, 3) float64 array.
    Invalid (error) positions are replaced with NaN.
    """
    sat_array = SatrecArray(satrecs)
    # sgp4 batch API: jd_whole is scalar broadcast; jd_fracs shape (N_steps,)
    e, r, _ = sat_array.sgp4(
        np.full(len(jd_fracs), jd_whole),
        jd_fracs,
    )
    # e shape: (N_objects, N_steps); r shape: (N_objects, N_steps, 3)
    r = r.astype(np.float64)
    r[e != 0] = np.nan          # mask propagation errors
    return r                    # (N_objects, N_steps, 3)


def distance_km(a: tuple, b: tuple) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def scan_debris_group(
    debris_list: list[tuple[str, Satrec]],
    station_list: list[tuple[str, Satrec]],
    station_pos: np.ndarray,
    jd_now: float,
    jd_fracs: np.ndarray,
    steps: np.ndarray,
    now_utc: datetime,
) -> list[dict]:
    """Propagate one debris group and return all watch-threshold hits."""
    debris_sats = [s for _, s in debris_list]
    debris_pos  = propagate_batch(debris_sats, jd_now, jd_fracs)

    results: list[dict] = []
    for d_idx, (d_name, _) in enumerate(debris_list):
        d_pos = debris_pos[d_idx]                          # (N_steps, 3)
        diff  = station_pos - d_pos[np.newaxis, :, :]     # (N_stations, N_steps, 3)
        dist  = np.sqrt(np.nansum(diff ** 2, axis=2))     # (N_stations, N_steps)

        min_dist_per_station = np.nanmin(dist, axis=1)    # (N_stations,)
        min_idx_per_station  = np.nanargmin(dist, axis=1) # (N_stations,)

        for s_idx, (s_name, _) in enumerate(station_list):
            min_d = float(min_dist_per_station[s_idx])
            if min_d < THRESHOLD_WATCH_KM:
                t_min = float(steps[min_idx_per_station[s_idx]])
                tca   = now_utc + timedelta(minutes=t_min)
                results.append({
                    "debris":   d_name,
                    "station":  s_name,
                    "min_dist": min_d,
                    "tca":      tca,
                })
    return results


def main() -> None:
    # ── load debris groups ────────────────────────────────────────────────────
    debris_groups: list[tuple[str, list[tuple[str, Satrec]]]] = []
    total_debris = 0
    for f in DEBRIS_FILES:
        group = load_satrecs(f)
        label = f.stem                          # e.g. "cosmos-2251-debris"
        debris_groups.append((label, group))
        print(f"  {len(group):>4} objects loaded from {f.name}")
        total_debris += len(group)
    print(f"  ---- {total_debris} total debris objects\n")

    # ── load stations ─────────────────────────────────────────────────────────
    print("Loading station objects ...")
    station_list = load_satrecs(STATIONS_FILE, allowed=PROTECTED_STATIONS)
    print(f"  {len(station_list)} station objects loaded "
          f"(filtered to {len(PROTECTED_STATIONS)} protected assets)\n")

    # ── time grid (shared across all groups) ──────────────────────────────────
    now_utc       = datetime.now(timezone.utc)
    jd_now        = 2440587.5 + now_utc.timestamp() / 86400.0
    total_minutes = WINDOW_DAYS * 24 * 60
    steps         = np.arange(0, total_minutes + 1, STEP_MINUTES, dtype=np.float64)
    jd_fracs      = steps / 1440.0
    n_steps       = len(steps)
    n_pairs       = total_debris * len(station_list)

    print(f"Window  : {now_utc.strftime('%Y-%m-%dT%H:%MZ')} + {WINDOW_DAYS} days  "
          f"({n_steps:,} steps x {STEP_MINUTES}-min)")
    print(f"Pairs   : {total_debris} debris x {len(station_list)} stations "
          f"= {n_pairs:,}\n")

    # ── propagate stations once (shared) ──────────────────────────────────────
    print("Propagating station objects ...")
    station_sats = [s for _, s in station_list]
    station_pos  = propagate_batch(station_sats, jd_now, jd_fracs)
    print("  Done.\n")

    # ── scan each debris group ────────────────────────────────────────────────
    results: list[dict] = []
    for label, debris_list in debris_groups:
        print(f"Propagating + scanning {label} ({len(debris_list)} objects) ...")
        hits = scan_debris_group(
            debris_list, station_list, station_pos,
            jd_now, jd_fracs, steps, now_utc,
        )
        results.extend(hits)
        print(f"  Done — {len(hits)} hit(s) within {THRESHOLD_WATCH_KM:.0f} km.\n")

    # ── classify ──────────────────────────────────────────────────────────────
    critical = sorted(
        [r for r in results if r["min_dist"] < THRESHOLD_CRITICAL_KM],
        key=lambda x: x["min_dist"],
    )
    watch = sorted(
        [r for r in results if THRESHOLD_CRITICAL_KM <= r["min_dist"] < THRESHOLD_WATCH_KM],
        key=lambda x: x["min_dist"],
    )

    # ── summary ───────────────────────────────────────────────────────────────
    HDR = f"{'Debris':<35} {'Station':<22} {'Min dist (km)':>13}  {'TCA (UTC)':>19}"
    SEP = "-" * 95

    print("=" * 95)
    print("CONJUNCTION CHECK SUMMARY  (Cosmos-2251 + Iridium-33 debris)")
    print("=" * 95)
    print(f"Window            : {now_utc.strftime('%Y-%m-%dT%H:%MZ')} + {WINDOW_DAYS} days")
    print(f"Debris groups     : {len(debris_groups)}  "
          f"({', '.join(f'{lbl} [{len(g)}]' for lbl, g in debris_groups)})")
    print(f"Pairs checked     : {n_pairs:,}")
    print(f"CRITICAL (< {THRESHOLD_CRITICAL_KM:.0f} km) : {len(critical)}")
    print(f"WATCH    (< {THRESHOLD_WATCH_KM:.0f} km) : {len(watch)}")
    print("=" * 95)

    if critical:
        print(f"\n*** CRITICAL --- {len(critical)} pair(s) ***")
        print(HDR)
        print(SEP)
        for p in critical:
            print(f"{p['debris']:<35} {p['station']:<22} {p['min_dist']:>13.3f}  "
                  f"{p['tca'].strftime('%Y-%m-%dT%H:%MZ'):>19}")
    else:
        print("\n  No CRITICAL conjunctions in this window.")

    if watch:
        print(f"\n--- WATCH --- {len(watch)} pair(s)")
        print(HDR)
        print(SEP)
        for p in watch:
            print(f"{p['debris']:<35} {p['station']:<22} {p['min_dist']:>13.3f}  "
                  f"{p['tca'].strftime('%Y-%m-%dT%H:%MZ'):>19}")
    else:
        print("\n  No WATCH conjunctions in this window.")

    print("=" * 95)


if __name__ == "__main__":
    main()
