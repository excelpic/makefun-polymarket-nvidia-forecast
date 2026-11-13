#!/usr/bin/env python3
"""Forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, minute by minute,
4 hours ahead, as five quantiles (5/25/50/75/95%) per market per minute.

Model, per market, using only the case's own 72h of minute prices:
  * work in logit space (prices clamped to [0.0005, 0.9995]);
  * quantile bands come from the market's own empirical distribution of
    h-minute logit changes, pooled over sample horizons and standardised by
    h**0.42 (these step/jump series widen slower than a random walk's
    sqrt(h)); the five pooled quantiles, centred on the pooled median,
    give the band offsets at every horizon;
  * the median path starts at the last price and mean-reverts gently
    toward the mean logit of the last 720 minutes (time constant 1080 min),
    which prices in the snap-back these markets show after a spike;
  * offsets are scaled by 0.85 (the raw empirical bands run slightly wide)
    and the 90% band is floored at 0.008 wide in price space so a market
    that has never moved still carries a minimal honest spread.
Deterministic: no randomness, no clock, no network. Standard library only.
"""

import glob
import json
import math
import os

EPS = 0.0005
TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
SAMPLE_H = (1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 90, 120, 180, 240)
BETA = 0.42        # horizon scaling of standardised changes
WIDTH = 0.85       # global band width multiplier
MR_TAU = 1080.0    # mean-reversion time constant, minutes
MU_WIN = 720       # window for the mean-reversion target, minutes
FLOOR_W = 0.008    # minimum width of the 90% band, price units


def logit(p):
    p = min(1.0 - EPS, max(EPS, p))
    return math.log(p / (1.0 - p))


def sigmoid(z):
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def quantile(sorted_vals, tau):
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 >= n:
        return sorted_vals[-1]
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[lo + 1] * frac


def forecast_market(prices, horizon):
    if not prices:
        return [[0.5] * len(TAUS) for _ in range(horizon)]
    series = [logit(p) for p in prices]
    n = len(series)

    pool = []
    for h in SAMPLE_H:
        if h >= n:
            continue
        scale = h ** BETA
        for i in range(h, n):
            pool.append((series[i] - series[i - h]) / scale)
    if pool:
        pool.sort()
        qs = [quantile(pool, t) for t in TAUS]
        offs = [q - qs[2] for q in qs]
    else:
        offs = [0.0] * len(TAUS)

    x = series[-1]
    win = series[-MU_WIN:]
    mu = sum(win) / len(win)
    d0 = x - mu

    scales = [h ** BETA for h in range(horizon + 1)]
    rows = []
    for h in range(1, horizon + 1):
        scale = scales[h]
        med = x - d0 * (1.0 - math.exp(-h / MR_TAU))
        row = [sigmoid(med + off * scale * WIDTH) for off in offs]
        if row[4] - row[0] < FLOOR_W:
            need = (FLOOR_W - (row[4] - row[0])) / 2.0
            row[0] = max(0.0, row[0] - need)
            row[4] = min(1.0, row[4] + need)
        row = [round(min(1.0, max(0.0, v)), 6) for v in row]
        row.sort()
        rows.append(row)
    return rows


def run_case(path_in, path_out):
    with open(path_in, "r", encoding="utf-8") as f:
        case = json.load(f)
    horizon = int(case["horizon_minutes"])
    forecasts = []
    for market in case["markets"]:
        rows = forecast_market(market["prices"], horizon)
        forecasts.append({"id": market["id"], "q": rows})
    with open(path_out, "w", encoding="utf-8") as f:
        json.dump({"forecasts": forecasts}, f, separators=(",", ":"))


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for path_in in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        case_id = os.path.basename(path_in)[:-3]
        run_case(path_in, os.path.join(out_dir, case_id + ".out"))


if __name__ == "__main__":
    main()
