"""Leave-one-out cross-validation over the comparable launches in a practice case.

Each comparable with a full release-day curve takes a turn as the target: its
hours 2..24 are forecast from the other launches' curves, and scored with the
contest's pinball loss on the log scale. Methods compared:

  M0  house persistence baseline (the kit starter's formula)
  M1  unweighted empirical quantiles of the per-hour log-ratio to hour 1
  M2  unweighted lognormal fit of the per-hour log-ratio
  M3  weighted lognormal: unlock-class match x kernels on log p1 and pre-ratio
  M4  M3 with the mean/sd curves smoothed across neighbouring hours
  M5  weighted lognormal with feature kernels only (no unlock text)
"""
import json
import math
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]

def parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def load_launches(path):
    case = json.load(open(path))
    launches = []
    for comp in case["comparables"]:
        start = parse(comp["day_start"])
        curve, pre = {}, {}
        for pt in comp["hourly"]:
            off = (parse(pt["from"]) - start).total_seconds() / 3600.0
            if off == int(off):
                off = int(off)
                if 0 <= off < 24:
                    curve[off + 1] = max(1, pt["viewers"])
                elif -24 <= off < 0:
                    pre[off] = max(1, pt["viewers"])
        if 1 not in curve or len(curve) < 20:
            continue  # not a launch day (e.g. the GTA VI pre-release category)
        pre6 = [pre[o] for o in range(-6, 0) if o in pre]
        launches.append({
            "name": comp["name"],
            "unlock": comp["unlock"],
            "curve": curve,
            "p1": curve[1],
            "pre_ratio": (sum(pre6) / len(pre6)) / curve[1] if pre6 else None,
        })
    return launches

def unlock_class(text):
    t = text.lower()
    if "early access" in t or "early-access" in t:
        return "early"
    if "single global" in t or "simultaneous global" in t or "single simultaneous" in t or "simultaneous across" in t:
        return "global"
    if "midnight local" in t or "midnight" in t and "rolling" in t:
        return "rolling"
    if "midnight" in t:
        return "rolling"
    return "unknown"

def pinball(y, q, tau):
    d = math.log(y) - math.log(q)
    return 2.0 * max(tau * d, (tau - 1.0) * d)

def score(forecast, curve):
    """forecast: {hour: [5 quantiles]}; curve: actual viewers by hour."""
    tot, n = 0.0, 0
    for h in range(2, 25):
        if h not in curve or h not in forecast:
            continue
        for tau, q in zip(LEVELS, forecast[h]):
            tot += pinball(curve[h], max(q, 1e-9), tau)
            n += 1
    return tot / n if n else None

def logratios(train, h):
    return [math.log(t["curve"][h] / t["p1"]) for t in train if h in t["curve"]]

def quantile(sorted_vals, tau):
    if not sorted_vals:
        return 0.0
    pos = tau * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)

def wmean_sd(pairs):
    """pairs: [(value, weight)] -> weighted mean, sd."""
    sw = sum(w for _, w in pairs)
    if sw <= 0:
        return 0.0, 0.5
    mu = sum(v * w for v, w in pairs) / sw
    var = sum(w * (v - mu) ** 2 for v, w in pairs) / sw
    return mu, math.sqrt(max(var, 0.0))

def weights_for(target, train, use_text):
    ws = []
    tcls = unlock_class(target["unlock"])
    for t in train:
        w = 1.0
        if use_text:
            c = unlock_class(t["unlock"])
            w *= 1.0 if c == tcls else (0.55 if "unknown" in (c, tcls) else 0.3)
        d = abs(math.log10(t["p1"]) - math.log10(target["p1"]))
        w *= math.exp(-0.5 * (d / 0.9) ** 2)
        if target["pre_ratio"] and t["pre_ratio"]:
            dp = abs(math.log(max(t["pre_ratio"], 1e-3)) - math.log(max(target["pre_ratio"], 1e-3)))
            w *= math.exp(-0.5 * (dp / 1.1) ** 2)
        ws.append(w)
    return ws

def forecast_m0(target, train):
    p1 = target["p1"]
    return {h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in range(2, 25)}

def forecast_m1(target, train):
    out = {}
    for h in range(2, 25):
        vals = sorted(logratios(train, h))
        out[h] = [target["p1"] * math.exp(quantile(vals, tau)) for tau in LEVELS]
    return out

def forecast_lognormal(target, train, use_text, smooth, unweighted=False):
    mus, sds = {}, {}
    ws = [1.0] * len(train) if unweighted else weights_for(target, train, use_text)
    for h in range(2, 25):
        pairs = [(math.log(t["curve"][h] / t["p1"]), w)
                 for t, w in zip(train, ws) if h in t["curve"]]
        mu, sd = wmean_sd(pairs)
        mus[h], sds[h] = mu, sd
    if smooth:
        for src in (mus, sds):
            sm = {}
            for h in range(2, 25):
                vals = [src[x] for x in (h - 1, h, h + 1) if x in src]
                sm[h] = sum(vals) / len(vals)
            src.update(sm)
    out = {}
    for h in range(2, 25):
        sd = min(max(sds[h], 0.22), 1.6)
        out[h] = [target["p1"] * math.exp(mus[h] + z * sd) for z in Z]
    return out

def main(path):
    launches = load_launches(path)
    print(f"{len(launches)} launches with a usable release-day curve")
    methods = {
        "M0 baseline": forecast_m0,
        "M1 empirical": forecast_m1,
        "M2 lognormal": lambda t, tr: forecast_lognormal(t, tr, False, False, unweighted=True),
        "M3 weighted+text": lambda t, tr: forecast_lognormal(t, tr, True, False),
        "M4 weighted+text+smooth": lambda t, tr: forecast_lognormal(t, tr, True, True),
        "M5 weighted feats": lambda t, tr: forecast_lognormal(t, tr, False, False),
    }
    # M5 differs from M2 only through weights: patch weights_for call via flag
    results = {m: [] for m in methods}
    rolling_results = {m: [] for m in methods}
    for i, target in enumerate(launches):
        train = launches[:i] + launches[i + 1:]
        for name, fn in methods.items():
            if name == "M5 weighted feats":
                # weighted by feature kernels but no text class
                fc = forecast_lognormal_feats(target, train)
            else:
                fc = fn(target, train)
            s = score(fc, target["curve"])
            if s is not None:
                results[name].append(s)
                if unlock_class(target["unlock"]) == "rolling":
                    rolling_results[name].append(s)
    print(f"\n{'method':28s} {'mean loss (all)':>16s} {'mean loss (rolling)':>20s} {'n roll':>7s}")
    for name in methods:
        allm = sum(results[name]) / len(results[name])
        rr = rolling_results[name]
        rm = sum(rr) / len(rr) if rr else float("nan")
        print(f"{name:28s} {allm:16.4f} {rm:20.4f} {len(rr):7d}")

def forecast_lognormal_feats(target, train):
    mus, sds = {}, {}
    ws = []
    for t in train:
        w = 1.0
        d = abs(math.log10(t["p1"]) - math.log10(target["p1"]))
        w *= math.exp(-0.5 * (d / 0.9) ** 2)
        if target["pre_ratio"] and t["pre_ratio"]:
            dp = abs(math.log(max(t["pre_ratio"], 1e-3)) - math.log(max(target["pre_ratio"], 1e-3)))
            w *= math.exp(-0.5 * (dp / 1.1) ** 2)
        ws.append(w)
    out = {}
    for h in range(2, 25):
        pairs = [(math.log(t["curve"][h] / t["p1"]), w)
                 for t, w in zip(train, ws) if h in t["curve"]]
        mu, sd = wmean_sd(pairs)
        sd = min(max(sd, 0.22), 1.6)
        out[h] = [target["p1"] * math.exp(mu + z * sd) for z in Z]
    return out

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in")
