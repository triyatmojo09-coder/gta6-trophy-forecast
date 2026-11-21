"""Forecast GTA VI PS5 trophy earn rates (Exophase) at a case's target.

Entry for make.fun's gta6-trophy-rarity-forecast-01. Reads every <id>.in case in the
input directory and writes <id>.out with five quantiles (5/25/50/75/95%) of each
trophy's earn rate at the target, scored by pinball loss on the logit scale.

Model, per trophy, from the case's own history only:

  * Exophase refreshes its counts in steps, so the history is first compressed to
    its refresh points (reads where the players tracked or any rate moved).
  * An earn rate is earners / players tracked. The two parts move differently in
    week one: the tracked crowd keeps growing (and dilutes rates), while a
    trophy's earners grow and then saturate. Both are projected separately on a
    ln(1 + hours since release) axis with damped least-squares slopes, and the
    median rate is earners / players at the target.
  * That decomposition is blended with a damped logit-space trend of the rate
    itself, so fast risers the decomposition under-reads still move.
  * Story structure is enforced on the medians: Chapter 1 >= Chapter 2 >=
    Chapter 3 >= Chapter 4 >= final mission, 100% completion no higher than the
    rarest other trophy, platinum no higher than 100% completion.
  * The spread combines a random-walk term in the horizon, a counting term from
    the trophy's earners (a rate held by a handful of players can double on a
    refresh) and the trend fit's residual, so rare trophies get honestly wide
    quantiles and common ones tighter ones.
  * A history in which nothing at all moved is a stale page awaiting a refresh:
    medians shade down (a refresh grows the tracked crowd faster than most
    trophies' earners) and the spread widens.

Standard library only, deterministic, no randomness anywhere.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Trophy ids with structural roles in GTA VI's list (position ids from the case).
PLATINUM_ID = 1
INTRO_ID = 2  # "Complete Gone Fishing": earned by effectively every starter.
CHAPTER_IDS = [3, 4, 5, 6, 7]  # Chapter 1..4, then the final mission.
COMPLETION_ID = 14  # Attain 100% Completion: needs every other trophy.
GRIND_IDS = {15, 25, 30, 31, 32}  # Long collections and challenge grinds.

# Week-one ceilings for the earners share of tracked players, by trophy kind.
CEILING_DEFAULT = 0.60
CEILING_GRIND = 0.25
CEILING_COMPLETION = 0.02

# Median earners at the target for a trophy nobody has earned yet, as a share of
# the projected tracked players: by Nov 26 a handful of the tracked crowd will
# hold almost anything, but only a sliver hold the long grinds.
PRIOR_DEFAULT = 0.0008
PRIOR_GRIND = 0.0004
PRIOR_COMPLETION = 0.00025
PRIOR_PLATINUM = 0.00015


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text: str) -> float:
    """ISO 8601 UTC (fractional seconds allowed) to epoch seconds."""
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def hours_since(anchor: float, text: str) -> float:
    return (parse_time(text) - anchor) / 3600.0


def wls_slope(points):
    """Weighted least squares of y on x; returns (slope, residual_std).

    points: list of (x, y, w). Falls back to (0, None) when the fit is
    underdetermined or all x coincide.
    """
    n = len(points)
    if n < 2:
        return 0.0, None
    sw = sum(w for _, _, w in points)
    mx = sum(w * x for x, _, w in points) / sw
    my = sum(w * y for _, y, w in points) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in points)
    if sxx <= 1e-12:
        return 0.0, None
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in points)
    slope = sxy / sxx
    intercept = my - slope * mx
    if n > 2:
        var = sum(w * (y - (intercept + slope * x)) ** 2 for x, y, w in points) / sw
        resid = math.sqrt(max(var, 0.0))
    else:
        resid = None
    return slope, resid


def refresh_points(history):
    """Compress the read history to the moments anything actually changed.

    Exophase republishes its counts in steps; between refreshes the reads are
    copies. Keeping one point per distinct page keeps the fits honest.
    """
    points = []
    last_key = None
    for read in history:
        key = (read["players_tracked"], tuple(read["rates"]))
        if key != last_key:
            points.append(read)
            last_key = key
    return points


def ceiling_for(trophy_id: int) -> float:
    if trophy_id in (PLATINUM_ID, COMPLETION_ID):
        return CEILING_COMPLETION
    if trophy_id in GRIND_IDS:
        return CEILING_GRIND
    return CEILING_DEFAULT


def prior_for(trophy_id: int) -> float:
    if trophy_id == PLATINUM_ID:
        return PRIOR_PLATINUM
    if trophy_id == COMPLETION_ID:
        return PRIOR_COMPLETION
    if trophy_id in GRIND_IDS:
        return PRIOR_GRIND
    return PRIOR_DEFAULT


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n = len(trophies)

    anchor = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target_h = hours_since(anchor, case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    points = refresh_points(history)
    times = [hours_since(anchor, p["at"]) for p in points]
    players = [float(p["players_tracked"]) for p in points]
    last = history[-1]
    t_last = times[-1] if times else hours_since(anchor, last["at"])
    p_last = float(last["players_tracked"])
    x_last = math.log1p(max(t_last, 0.0))
    x_tgt = math.log1p(max(target_h, 0.0))

    # Fits run on the recent regime: the last 36 hours of refresh points (but
    # never fewer than the last four), because launch-night dynamics are not
    # the dynamics that carry the next four days.
    if len(points) > 4:
        cut = t_last - 36.0
        start = 0
        while start < len(points) - 4 and times[start] < cut:
            start += 1
        fit_idx = list(range(start, len(points)))
    else:
        fit_idx = list(range(len(points)))

    # A page in which nothing moved at all is stale: a refresh is pending.
    flat = len(points) <= 1

    # --- Project the tracked crowd -------------------------------------
    if len(fit_idx) >= 2 and players[fit_idx[0]] > 0:
        pts = [
            (math.log1p(max(times[j], 0.0)), math.log(players[j]), 1.0 + times[j] / max(times[-1], 1.0))
            for j in fit_idx
            if players[j] > 0
        ]
        slope_p, _ = wls_slope(pts)
    else:
        slope_p = 0.0
    slope_p = min(max(slope_p * 0.85, 0.0), 3.0)
    p_tgt = p_last * math.exp(slope_p * (x_tgt - x_last))
    p_tgt = min(max(p_tgt, p_last * 1.01), p_last * 9.0)

    medians = [0.0] * n
    sigmas = [0.0] * n
    walk = 0.5 * math.sqrt(max(horizon_days, 0.0))

    for i, trophy in enumerate(trophies):
        tid = trophy["id"]
        rates = [p["rates"][i] for p in points]
        r_last = last["rates"][i]
        r_last = 0.0 if r_last is None else float(r_last)
        e_last = r_last / 100.0 * p_last

        if r_last >= 99.9:
            # Earned by (nearly) every tracked player and stays that way.
            medians[i] = 99.97
            sigmas[i] = 0.05
            continue

        # --- Earners projection ---------------------------------------
        if e_last > 0:
            pts = []
            for j in fit_idx:
                r = rates[j]
                if r is None:
                    continue
                e = float(r) / 100.0 * players[j]
                t = times[j]
                pts.append(
                    (math.log1p(max(t, 0.0)), math.log(e + 0.5), 1.0 + t / max(times[-1], 1.0))
                )
            slope_e, _ = wls_slope(pts)
            slope_e = min(max(slope_e * 0.7, 0.0), 4.0)
            e_tgt = (e_last + 0.5) * math.exp(slope_e * (x_tgt - x_last)) - 0.5
            e_tgt = max(e_tgt, e_last * 0.98)
        else:
            e_tgt = p_tgt * prior_for(tid)
        e_tgt = min(e_tgt, ceiling_for(tid) * p_tgt)
        rate_decomp = 100.0 * e_tgt / p_tgt if p_tgt > 0 else r_last

        # --- Damped logit trend of the rate itself ---------------------
        if r_last > 0:
            pts = []
            for j in fit_idx:
                r = rates[j]
                if r is None or not (0.0 < float(r) < 100.0):
                    continue
                t = times[j]
                pts.append(
                    (math.log1p(max(t, 0.0)), logit(float(r)), 1.0 + t / max(times[-1], 1.0))
                )
            slope_l, resid = wls_slope(pts)
            slope_l = min(max(slope_l * 0.55, -2.0), 2.5)
            trend_logit = logit(r_last) + slope_l * (x_tgt - x_last)
            med_logit = 0.55 * logit(max(rate_decomp, 0.01)) + 0.45 * trend_logit
        else:
            resid = None
            med_logit = logit(max(rate_decomp, 0.01))

        # --- Spread -----------------------------------------------------
        count_term = min(1.2 / math.sqrt(max(e_last, 1.0)), 1.2)
        resid_term = 0.6 * resid if resid is not None else 0.3
        sigma = math.sqrt(walk * walk + count_term * count_term + resid_term * resid_term) + 0.15
        if e_last <= 0:
            sigma = max(sigma, 1.2)
        sigma = min(max(sigma, 0.3), 2.4)

        if flat:
            # Stale page: the pending refresh grows the crowd faster than
            # most earners, so shade down and widen honestly.
            med_logit -= 1.1
            sigma = 1.15

        medians[i] = inv_logit(med_logit)
        sigmas[i] = sigma

    # --- Story structure on the medians --------------------------------
    idx = {t["id"]: i for i, t in enumerate(trophies)}
    chain = [idx[t] for t in CHAPTER_IDS if t in idx]
    # Pool adjacent violators: chapter rates cannot rise along the story.
    stack = []
    for j in chain:
        stack.append([medians[j], 1])
        while len(stack) >= 2 and stack[-2][0] < stack[-1][0]:
            top = stack.pop()
            prev = stack.pop()
            total = prev[0] * prev[1] + top[0] * top[1]
            count = prev[1] + top[1]
            stack.append([total / count, count])
    adjusted = []
    for value, count in stack:
        adjusted.extend([value] * count)
    for j, value in zip(chain, adjusted):
        medians[j] = value
    if COMPLETION_ID in idx:
        others = [medians[i] for i in range(n) if trophies[i]["id"] != COMPLETION_ID]
        if others:
            medians[idx[COMPLETION_ID]] = min(medians[idx[COMPLETION_ID]], min(others))
    if PLATINUM_ID in idx and COMPLETION_ID in idx:
        medians[idx[PLATINUM_ID]] = min(
            medians[idx[PLATINUM_ID]], medians[idx[COMPLETION_ID]]
        )

    # --- Quantiles ------------------------------------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        med_logit = logit(medians[i])
        sigma = sigmas[i]
        q = []
        for z in Z:
            v = inv_logit(med_logit + z * sigma)
            q.append(round4(min(100.0, max(0.0, v))))
        for k in range(1, len(q)):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
