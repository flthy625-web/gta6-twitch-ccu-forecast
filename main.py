"""gta6-twitch-ccu-forecast-01 entry: forecast GTA VI's launch-day Twitch
viewers, hour by hour, from the case's own comparable launches.

Model (v4; leave-one-out cross-validation over the comparable launches in
the practice cases, mean log pinball loss: 0.368 over all launches, 0.242
on rolling-midnight launches, 0.171 on large rolling-midnight launches —
v3 scored 0.370 / 0.251 / 0.184, v2 0.377 / 0.295 / 0.202, v1 0.392 /
0.315 / 0.231):

  For every past launch in the case's `comparables`, take its release-day
  curve hour by hour and form the log-ratio to its own hour-1 viewers. Each
  comparable is weighted by similarity to the target: same unlock pattern
  (rolling midnight local / single global unlock / early-access
  contaminated, read from the unlock notes), closeness of hour-1 audience
  size, closeness of the pre-release tail (mean viewers in the 6 hours
  before the release day, relative to hour 1), franchise-name overlap with
  the target game, and platform overlap.

  For rolling-midnight targets there is one more similarity signal, the
  strongest of the set: the audience's diurnal fingerprint — the mean
  log-viewer profile by UTC hour of day over the week before the release
  day (offsets -168..-25), which captures the audience's timezone mix.
  Two launches whose pre-release audiences keep the same daily rhythm
  tend to trace the same release-day shape, so comparables are weighted
  by the cosine similarity of their fingerprint to the target's. The
  same fingerprint is computed for live channels (the streamer supply's
  daily rhythm) and applied as a second, wider kernel.

  Per hour, a weighted ridge regression of the log-ratios on the size gap
  (log10 of the comparable's hour-1 viewers minus the target's) removes
  the size trend, so the intercept is the size-adjusted centre for a
  target of exactly this size; the spread is the regression's weighted
  residual standard deviation (scaled by 0.85, or 1.0 in the fingerprint
  regime, as cross-validation prefers). In the fingerprint regime the
  centre is then blended 60/40 with a second fit over only the closest
  fingerprint matches (cosine above 0.8), which sharpens the shape for
  large launches. The five quantiles are

      q_tau(h) = p1 * exp(mu_h + z_tau * sd_h)

  If a case carries no usable comparable, the entry falls back to the
  persistence forecast (every hour at hour 1's level, spread widening like
  a random walk's).

Runs on the Python 3.13 standard library only, no network, deterministic.
"""

import json
import math
import os
import re
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
W_MISMATCH = 0.1  # weight when the unlock class clearly differs
W_UNKNOWN = 0.6   # weight when either unlock class is unreadable
BETA_NAME = 1.0   # weight boost per unit of franchise-name Jaccard overlap
BETA_PLAT = 0.5   # weight boost per unit of platform Jaccard overlap
W_FINGER = 0.3    # kernel width for 1 - cosine(viewer diurnal fingerprints)
W_FINGER_CH = 0.5  # kernel width for 1 - cosine(channel diurnal fingerprints)
SUBSET_COS = 0.8  # fingerprint cosine above which a comparable is "close"
SUBSET_BLEND = 0.4  # weight of the close-subset fit in the blended centre
RIDGE = 0.05      # ridge penalty on the size-trend slope
SD_SCALE = 0.85
SD_SCALE_FINGER = 1.0  # spread scale in the fingerprint regime
SD_FLOOR = 0.22
SD_CAP = 1.6

STOP = {"the", "of", "and", "a"}


def parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def unlock_class(text):
    t = (text or "").lower()
    # Negations first: "no early access" describes a clean launch, not an
    # early-access one (the Wolverine case notes say exactly that).
    t = t.replace("no early access", "").replace("no early-access", "")
    t = t.replace("without early access", "").replace("without early-access", "")
    if "early access" in t or "early-access" in t:
        return "early"
    if ("single global" in t or "simultaneous global" in t
            or "single simultaneous" in t or "simultaneous across" in t):
        return "global"
    if "midnight" in t:
        return "rolling"
    return "unknown"


def name_tokens(name):
    raw = "".join(c if c.isalnum() else " " for c in (name or "").lower()).split()
    return {t for t in raw if t not in STOP and len(t) >= 2}


def name_sim(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def platform_set(text):
    """Coarse platform families from strings like 'PC, PS4/PS5, Xbox One/Series'."""
    t = (text or "").lower()
    out = set()
    if "ps5" in t or "playstation 5" in t:
        out.add("ps5")
    if "ps4" in t or "playstation 4" in t:
        out.add("ps4")
    if "xbox" in t:
        out.add("xbox")
    if "pc" in t or "windows" in t:
        out.add("pc")
    if "switch" in t or "nintendo" in t:
        out.add("switch")
    return out


def target_platforms(notes):
    """The case notes name the target's platforms in prose ('on PS5 only',
    'on PS4, Xbox One (console only; ...)'). Best effort; empty if unreadable."""
    m = re.search(r"\bon\s+([^.;()]+)", notes or "")
    return platform_set(m.group(1)) if m else set()


def diurnal_fingerprint(day_start, hourly, field="viewers"):
    """Mean log of `field` by UTC hour of day over offsets -168..-25, centred."""
    start = parse(day_start)
    by_hour = {}
    for pt in hourly:
        off = (parse(pt["from"]) - start).total_seconds() / 3600.0
        if off != int(off):
            continue
        off = int(off)
        if -168 <= off <= -25 and pt.get(field):
            by_hour.setdefault(off % 24, []).append(math.log(max(1, pt[field])))
    if len(by_hour) < 18:
        return None
    vec = []
    for hod in range(24):
        vals = by_hour.get(hod)
        vec.append(sum(vals) / len(vals) if vals else 0.0)
    mean = sum(vec) / 24
    return [v - mean for v in vec]


def cosine(a, b):
    if a is None or b is None:
        return None
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    if da <= 0 or db <= 0:
        return None
    return sum(x * y for x, y in zip(a, b)) / (da * db)


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
                    "curve": curve, "p1": curve[1], "pre_ratio": pre_ratio,
                    "tokens": name_tokens(comp.get("name")),
                    "plats": platform_set(comp.get("platforms")),
                    "finger": diurnal_fingerprint(comp["day_start"], comp["hourly"]),
                    "finger_ch": diurnal_fingerprint(comp["day_start"], comp["hourly"],
                                                     field="channels")})
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
    notes = case.get("notes")
    return {"p1": p1, "pre_ratio": pre_ratio, "cls": unlock_class(notes),
            "tokens": name_tokens((case.get("game") or {}).get("name")),
            "plats": target_platforms(notes),
            "finger": diurnal_fingerprint(case["day_start"],
                                          case.get("history_hourly") or []),
            "finger_ch": diurnal_fingerprint(case["day_start"],
                                             case.get("history_hourly") or [],
                                             field="channels")}


def weight(target, t, use_finger):
    if t["cls"] == target["cls"]:
        w = 1.0
    elif "unknown" in (t["cls"], target["cls"]):
        w = W_UNKNOWN
    else:
        w = W_MISMATCH
    d = abs(math.log10(t["p1"]) - math.log10(target["p1"]))
    w *= math.exp(-0.5 * (d / W_P1) ** 2)
    if target["pre_ratio"] and t["pre_ratio"]:
        dp = abs(math.log(max(t["pre_ratio"], 1e-3)) - math.log(max(target["pre_ratio"], 1e-3)))
        w *= math.exp(-0.5 * (dp / W_PRE) ** 2)
    w *= math.exp(BETA_NAME * name_sim(target["tokens"], t["tokens"]))
    if target["plats"] and t["plats"]:
        jac = len(target["plats"] & t["plats"]) / len(target["plats"] | t["plats"])
        w *= math.exp(BETA_PLAT * jac)
    if use_finger:
        cs = cosine(target["finger"], t["finger"])
        if cs is not None:
            w *= math.exp(-(1.0 - cs) / W_FINGER)
        cs_ch = cosine(target["finger_ch"], t["finger_ch"])
        if cs_ch is not None:
            w *= math.exp(-(1.0 - cs_ch) / W_FINGER_CH)
    return w


def loclin(rows):
    """rows: [(x, y, w)] with x centred on the target -> (intercept, resid sd)."""
    sw = sum(w for _, _, w in rows)
    if sw <= 0:
        return 0.0, 0.5
    mx = sum(x * w for x, _, w in rows) / sw
    my = sum(y * w for _, y, w in rows) / sw
    sxx = sum(w * (x - mx) ** 2 for x, _, w in rows) + RIDGE
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in rows)
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in rows) / sw
    return a, math.sqrt(max(var, 0.0))


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
    p1 = target["p1"]
    train = launches_from(case)
    hours = []
    if not train:
        for row in case["forecast"]:
            h = row["hour"]
            q = [round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1))) for z in Z]
            hours.append({"hour": h, "q": q})
        return {"hours": hours}
    use_finger = target["cls"] == "rolling" and target["finger"] is not None
    sd_scale = SD_SCALE_FINGER if use_finger else SD_SCALE
    weighted = [(t, weight(target, t, use_finger)) for t in train]
    tx = math.log10(p1)
    for row in case["forecast"]:
        h = row["hour"]
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in weighted if h in t["curve"] and w > 0]
        if rows:
            mu, sd = loclin(rows)
        else:
            mu, sd = 0.0, 0.5
        if use_finger and rows:
            close = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                     for t, w in weighted
                     if h in t["curve"] and w > 0
                     and (cosine(target["finger"], t["finger"]) or 0.0) > SUBSET_COS]
            if len(close) >= 4:
                mu2, sd2 = loclin(close)
                mu = (1 - SUBSET_BLEND) * mu + SUBSET_BLEND * mu2
                sd = (1 - SUBSET_BLEND) * sd + SUBSET_BLEND * sd2
        sd = min(max(sd * sd_scale, SD_FLOOR), SD_CAP)
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
