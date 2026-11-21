[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/triyatmojo09-coder/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/triyatmojo09-coder/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecast

A forecaster for make.fun's **gta6-trophy-rarity-forecast-01** bounty: for every
PlayStation 5 trophy of Grand Theft Auto VI, it predicts five quantiles
(5/25/50/75/95%) of the trophy's earn rate on Exophase at the case's target,
scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3, standard library only) reads each case in the input
directory and reasons from that case's recorded history alone:

- **Refresh points.** Exophase republishes its counts in steps, so the read
  history is compressed to the moments the page actually changed.
- **Earners ÷ players.** An earn rate is earners over tracked players, and in
  week one the two move differently: the tracked crowd keeps growing (diluting
  rates) while a trophy's earners grow and saturate. Both are projected
  separately on a ln(1 + hours since release) axis with damped least-squares
  slopes fitted on the last 36 hours, and the median rate is their ratio at the
  target.
- **Damped trend blend.** The decomposition is blended with a damped
  logit-space trend of the rate itself, so fast risers still move.
- **Story structure.** Medians respect the game's own logic: Chapter 1 ≥
  Chapter 2 ≥ Chapter 3 ≥ Chapter 4 ≥ final mission, 100% completion no higher
  than the rarest other trophy, platinum no higher than that.
- **Honest spread.** Each trophy's sigma combines a random-walk term in the
  horizon, a counting term from its earners (a rate held by a handful of
  players can double on one refresh) and the trend fit's residual. A history
  in which nothing moved at all is treated as a stale page awaiting a refresh:
  medians shade down and the spread widens.

The program is deterministic: same case in, same forecast out, byte for byte.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

with case files (`*.in`) in `input/`. In the grader's sandbox the program is
run as `python3 /bot/main.py` with the case at `/input` and forecasts written
to `/output`.
