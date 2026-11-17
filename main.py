"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's launch-day Twitch
viewers, hour by hour, from the case's own comparable launches.

Model (chosen by leave-one-out cross-validation over the comparable launches,
where it scores a mean log pinball loss of about 0.39 against 0.93 for the
persistence baseline):

  For every past launch in the case's `comparables`, take its release-day
  curve hour by hour and form the log-ratio to its own hour-1 viewers. For the
  target, each comparable is weighted by similarity: same unlock pattern
  (rolling midnight local / single global unlock / early-access contaminated,
  read from the unlock notes), closeness of its hour-1 audience size, and
  closeness of its pre-release tail (mean viewers in the 6 hours before the
  release day, relative to hour 1). The five quantiles for hour h are then

      q_tau(h) = p1 * exp(mu_h + z_tau * sd_h)

  with mu_h, sd_h the weighted mean and standard deviation of the log-ratios
  at hour h. If a case carries no usable comparable, the entry falls back to
  the persistence forecast (every hour at hour 1's level, spread widening like
  a random walk's).

Runs on the Python 3.13 standard library only, no network, deterministic.
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
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

# Cross-validated parameters (leave-one-out over 37 comparable launches).
W_P1 = 0.6        # kernel width for log10 of hour-1 viewers
W_PRE = 1.6       # kernel width for log of the pre-release tail ratio
W_MISMATCH = 0.2  # weight when the unlock class clearly differs
W_UNKNOWN = 0.7   # weight when either unlock class is unreadable
SD_FLOOR = 0.22
SD_CAP = 1.6


def parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def unlock_class(text):
    t = (text or "").lower()
    if "early access" in t or "early-access" in t:
        return "early"
    if ("single global" in t or "simultaneous global" in t
            or "single simultaneous" in t or "simultaneous across" in t):
        return "global"
    if "midnight" in t:
        return "rolling"
    return "unknown"


def day_curve(day_start, hourly):
    """Hour index 1..24 of the release day -> viewers, plus the pre-tail ratio."""
    start = parse(day_start)
    curve, pre = {}, []
    for pt in hourly:
        off = (parse(pt["from"]) - start).total_seconds() / 3600.0
        if off != int(off):
            continue
        off = int(off)
        if 0 <= off < 24:
            curve[off + 1] = max(1, pt["viewers"])
        elif -6 <= off < 0:
            pre.append(max(1, pt["viewers"]))
    pre_ratio = (sum(pre) / len(pre)) / curve[1] if pre and 1 in curve else None
    return curve, pre_ratio


def launches_from(case):
    out = []
    for comp in case.get("comparables") or []:
        curve, pre_ratio = day_curve(comp["day_start"], comp["hourly"])
        if 1 not in curve or len(curve) < 20:
            continue  # not a launch day (e.g. a category's pre-release history)
        out.append({"cls": unlock_class(comp.get("unlock")),
                    "curve": curve, "p1": curve[1], "pre_ratio": pre_ratio})
    return out


def target_features(case):
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError("case has no hour-1 print")
    p1 = max(1, first["viewers"])
    start = parse(case["day_start"])
    pre = []
    for pt in case.get("history_hourly") or []:
        off = (parse(pt["from"]) - start).total_seconds() / 3600.0
        if off == int(off) and -6 <= off < 0:
            pre.append(max(1, pt["viewers"]))
    pre_ratio = (sum(pre) / len(pre)) / p1 if pre else None
    return p1, pre_ratio, unlock_class(case.get("notes"))


def weight(target, t):
    p1, pre_ratio, cls = target
    if t["cls"] == cls:
        w = 1.0
    elif "unknown" in (t["cls"], cls):
        w = W_UNKNOWN
    else:
        w = W_MISMATCH
    d = abs(math.log10(t["p1"]) - math.log10(p1))
    w *= math.exp(-0.5 * (d / W_P1) ** 2)
    if pre_ratio and t["pre_ratio"]:
        dp = abs(math.log(max(t["pre_ratio"], 1e-3)) - math.log(max(pre_ratio, 1e-3)))
        w *= math.exp(-0.5 * (dp / W_PRE) ** 2)
    return w


def round2(x):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError(f"case asks for levels {case['levels']}, not {LEVELS}")
    target = target_features(case)
    p1 = target[0]
    train = launches_from(case)
    hours = []
    if not train:
        for row in case["forecast"]:
            h = row["hour"]
            q = [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]
            hours.append({"hour": h, "q": q})
        return {"hours": hours}
    weighted = [(t, weight(target, t)) for t in train]
    for row in case["forecast"]:
        h = row["hour"]
        pairs = [(math.log(t["curve"][h] / t["p1"]), w)
                 for t, w in weighted if h in t["curve"]]
        sw = sum(w for _, w in pairs)
        mu = sum(v * w for v, w in pairs) / sw
        var = sum(w * (v - mu) ** 2 for v, w in pairs) / sw
        sd = min(max(math.sqrt(var), SD_FLOOR), SD_CAP)
        q = [round2(min(p1 * math.exp(mu + z * sd), 1e9)) for z in Z]
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
