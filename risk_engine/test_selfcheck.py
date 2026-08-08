"""
Self-test for conjunction_check.py.

Three checks:
  1. Position sanity  – pick the first debris and first station, propagate at
                        t=0 (step=0 min) and confirm the ECI radius is in the
                        LEO band (6,371 + 200 km … 6,371 + 2,000 km).
  2. Zero self-distance – compute the distance between the first debris object
                          and itself at the same timestamp; must be exactly 0.
  3. Global minimum   – run the full propagation window over all pairs and
                        report the closest actual approach, regardless of
                        threshold.
"""

import math
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

# allow running from any cwd
sys.path.insert(0, str(Path(__file__).parent))
from conjunction_check import (
    load_satrecs,
    propagate_batch,
    distance_km,
    DEBRIS_FILES,
    STATIONS_FILE,
    WINDOW_DAYS,
    STEP_MINUTES,
)

EARTH_RADIUS_KM = 6_371.0
LEO_MIN_KM = EARTH_RADIUS_KM + 200.0    # ~6 571 km
LEO_MAX_KM = EARTH_RADIUS_KM + 2_000.0  # ~8 371 km


def _radius(pos: tuple[float, float, float]) -> float:
    """Distance from Earth's centre in km."""
    return math.sqrt(pos[0] ** 2 + pos[1] ** 2 + pos[2] ** 2)


# ── shared setup ─────────────────────────────────────────────────────────────

debris_list: list = []
for _f in DEBRIS_FILES:
    debris_list.extend(load_satrecs(_f))
station_list = load_satrecs(STATIONS_FILE)

if not debris_list:
    sys.exit("ERROR: no debris objects loaded — check cache/cosmos-2251-debris.json "
             "and cache/iridium-33-debris.json")
if not station_list:
    sys.exit("ERROR: no station objects loaded — check cache/stations.json")

now_utc       = datetime.now(timezone.utc)
jd_now        = 2440587.5 + now_utc.timestamp() / 86400.0
total_minutes = WINDOW_DAYS * 24 * 60
steps         = np.arange(0, total_minutes + 1, STEP_MINUTES, dtype=np.float64)
jd_fracs      = steps / 1440.0


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 - ECI position sanity at t = 0
# ---------------------------------------------------------------------------
print("=" * 60)
print("CHECK 1 - ECI position sanity at t = 0")
print("=" * 60)

d_name, d_sat = debris_list[0]
s_name, s_sat = station_list[0]

# propagate_batch returns (N_objects, N_steps, 3); pick object 0, step 0
d_pos0 = tuple(propagate_batch([d_sat], jd_now, np.array([0.0]))[0, 0])
s_pos0 = tuple(propagate_batch([s_sat], jd_now, np.array([0.0]))[0, 0])

for label, pos in [(f"Debris  : {d_name}", d_pos0),
                   (f"Station : {s_name}", s_pos0)]:
    if any(math.isnan(v) for v in pos):
        print(f"  {label}  ->  propagation ERROR at t=0")
        continue
    r = _radius(pos)
    alt = r - EARTH_RADIUS_KM
    ok  = LEO_MIN_KM <= r <= LEO_MAX_KM
    status = "OK" if ok else "FAIL - outside LEO band!"
    print(f"  {label}")
    print(f"    ECI (km) : x={pos[0]:+.3f}  y={pos[1]:+.3f}  z={pos[2]:+.3f}")
    print(f"    |r|      : {r:.3f} km  (altitude ~{alt:.1f} km)  [{status}]")

print()

# ---------------------------------------------------------------------------
# CHECK 2 - Zero self-distance
# ---------------------------------------------------------------------------
print("=" * 60)
print("CHECK 2 - Self-distance (same object, same time)")
print("=" * 60)

# Propagate the first debris object at step 0 twice and compare.
pos_a = tuple(propagate_batch([d_sat], jd_now, np.array([0.0]))[0, 0])
pos_b = tuple(propagate_batch([d_sat], jd_now, np.array([0.0]))[0, 0])

if any(math.isnan(v) for v in pos_a) or any(math.isnan(v) for v in pos_b):
    print("  FAIL - propagation error prevented the check")
else:
    self_dist = distance_km(pos_a, pos_b)
    status = "OK" if self_dist == 0.0 else f"FAIL - got {self_dist}"
    print(f"  distance({d_name}, {d_name}) at t=0 = {self_dist}  [{status}]")

print()

# ---------------------------------------------------------------------------
# CHECK 3 - Global minimum across all pairs in the full window
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"CHECK 3 - Global minimum approach distance (all pairs, {WINDOW_DAYS * 24} h)")
print("=" * 60)
print(f"  {len(debris_list)} debris x {len(station_list)} stations "
      f"= {len(debris_list) * len(station_list):,} pairs  "
      f"({len(steps)} steps each) ...")

# Use propagate_batch for both groups — mirrors the production code path.
station_sats = [s for _, s in station_list]
station_pos  = propagate_batch(station_sats, jd_now, jd_fracs)  # (N_s, N_t, 3)

debris_sats = [s for _, s in debris_list]
debris_pos  = propagate_batch(debris_sats, jd_now, jd_fracs)    # (N_d, N_t, 3)

global_min  = float("inf")
global_pair = ("?", "?")
global_step = 0

for d_idx, (d_name_i, _) in enumerate(debris_list):
    for s_idx, (s_name_j, _) in enumerate(station_list):
        diff = debris_pos[d_idx] - station_pos[s_idx]           # (N_t, 3)
        dist = np.sqrt(np.nansum(diff ** 2, axis=1))            # (N_t,)
        min_i = int(np.nanargmin(dist))
        if dist[min_i] < global_min:
            global_min  = float(dist[min_i])
            global_pair = (d_name_i, s_name_j)
            global_step = min_i

if global_min == float("inf"):
    print("  Could not compute any distance (all propagation errors?).")
else:
    tca = now_utc + timedelta(minutes=steps[global_step])
    print(f"  Closest approach : {global_min:.3f} km")
    print(f"  Pair             : '{global_pair[0]}' vs '{global_pair[1]}'")
    print(f"  Time of closest  : {tca.strftime('%Y-%m-%dT%H:%MZ')} "
          f"(t + {steps[global_step]} min)")

print()
