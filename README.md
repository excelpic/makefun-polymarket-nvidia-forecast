# Polymarket NVIDIA CEO — minute-by-minute quantile forecaster

Entry for the make.fun bounty `polymarket-nvidia-ceo-forecast-01`.

`main.py` (Python 3.13, standard library only) reads one case at a time from
`/input/<id>.in`: every market of Polymarket's "Next CEO of NVIDIA?" event with
its Yes price at each minute of the 72 hours up to the checkpoint. It writes
`/output/<id>.out` with, for each market and each of the 240 minutes after the
checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of that minute's price.

## Model

Per market, using only the case's own series:

- Work in logit space (prices clamped at 0.0005) so bands respect the (0, 1) bounds.
- Quantile bands come from the empirical distribution of historical h-minute
  logit changes, pooled over 14 sample horizons and standardised by h^0.42 —
  these step-like series widen more slowly than sqrt(h).
- Band width adapts to the market's current volatility regime: the mean
  absolute 1-minute logit move over the last 180 minutes, relative to its
  full-window average, scales every width by the square root of that ratio
  (clamped to [0.625, 1.6]) — quiet markets get narrower bands, markets that
  have woken up get wider ones.
- Widths taper gently with horizon (a factor h^-0.04), and the outer (5/95%)
  offsets are scaled by 0.80 against 0.85 for the inner ones.
- The median path mean-reverts gently toward the trimmed mean logit of the
  last 720 minutes (time constant 1080 minutes), pricing in spike snap-back;
  trimming keeps one spike inside the window from dragging the target.
- The 90% band is floored at 0.012 wide in price space for markets that never
  moved, results are clamped to [0, 1], rounded to 6 decimals and sorted
  monotone per row.

Fully deterministic: two runs on the same case write byte-identical output.
Runs in about 0.3 s per case, far under the 8 s cap.
