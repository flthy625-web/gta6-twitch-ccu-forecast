"""V4 experiments on top of v3 (fingerprint loclin):
(A) channels diurnal fingerprint as a second similarity kernel;
(B) residual-quantile output: replace normal z with weighted empirical
    quantiles of the loclin residuals (captures skew / heavy tails).
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import unlock_class, parse, LEVELS, Z  # noqa: E402
from cv2 import wls_intercept  # noqa: E402
from cv4 import wquantile, evaluate  # noqa: E402
import cv6  # noqa: E402


def finger_channels(hourly, day_start):
    start = parse(day_start)
    by_hour = {}
    for pt in hourly:
        off = (parse(pt["from"]) - start).total_seconds() / 3600.0
        if off != int(off):
            continue
        off = int(off)
        if -168 <= off <= -25 and pt.get("channels"):
            by_hour.setdefault(off % 24, []).append(math.log(max(1, pt["channels"])))
    if len(by_hour) < 18:
        return None
    vec = [sum(by_hour[h]) / len(by_hour[h]) if h in by_hour else 0.0 for h in range(24)]
    mean = sum(vec) / 24
    return [v - mean for v in vec]


def enrich9(case_path):
    case = json.load(open(case_path))
    launches = cv6.enrich6(case_path)
    raw = {c["name"]: c for c in case["comparables"]}
    for L in launches:
        L["finger_ch"] = finger_channels(raw[L["name"]]["hourly"], raw[L["name"]]["day_start"])
    return launches


def forecast_v4(target, train, P):
    ws = cv6.weights6(target, train, P)
    if P.get("w_finger_ch"):
        out = []
        for t, w in zip(train, ws):
            cs = cv6.cosine(target.get("finger_ch"), t.get("finger_ch"))
            if cs is not None:
                w *= math.exp(-(1.0 - cs) / P["w_finger_ch"])
            out.append(w)
        ws = out
    tx = math.log10(target["p1"])
    # first pass: loclin per hour, collect standardized residuals
    mus, sds, resid = {}, {}, []
    for h in range(2, 25):
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in zip(train, ws) if h in t["curve"] and w > 0]
        if not rows:
            mus[h], sds[h] = 0.0, 0.5
            continue
        mu, sd = wls_intercept([r[0] for r in rows], [r[1] for r in rows],
                               [r[2] for r in rows], P["ridge"])
        mus[h], sds[h] = mu, sd
        if P.get("resid_q") and sd > 1e-6:
            sw = sum(r[2] for r in rows)
            mx = sum(r[0] * r[2] for r in rows) / sw
            my = sum(r[1] * r[2] for r in rows) / sw
            sxx = sum(r[2] * (r[0] - mx) ** 2 for r in rows) + P["ridge"]
            sxy = sum(r[2] * (r[0] - mx) * (r[1] - my) for r in rows)
            b = sxy / sxx if sxx > 0 else 0.0
            a = my - b * mx
            for x, y, w in rows:
                resid.append(((y - (a + b * x)) / sd, w))
    use_finger = unlock_class(target["unlock"]) == "rolling" and target.get("finger") is not None
    scale = 1.0 if use_finger else P["sd_scale"]
    out = {}
    for h in range(2, 25):
        sd = min(max(sds[h] * scale, P["sd_floor"]), P["sd_cap"])
        if P.get("resid_q") and len(resid) >= 40:
            qs = [wquantile(resid, tau) for tau in LEVELS]
            out[h] = [target["p1"] * math.exp(mus[h] + q * sd) for q in qs]
        else:
            out[h] = [target["p1"] * math.exp(mus[h] + z * sd) for z in Z]
    return out


BASE = dict(cv6.BASE, w_finger=0.25, w_finger_ch=0.0, resid_q=False)

CONFIGS = [
    ("v3 (finger only)", {}),
    ("+ channels finger .25", {"w_finger_ch": 0.25}),
    ("+ channels finger .4", {"w_finger_ch": 0.4}),
    ("+ resid quantiles", {"resid_q": True}),
    ("+ ch finger .25 + residQ", {"w_finger_ch": 0.25, "resid_q": True}),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    launches = enrich9(path)
    # target-side channels fingerprint for CV targets comes from their own entry
    print(f"{'config':30s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for name, over in CONFIGS:
        P = dict(BASE, **over)
        def fn(t, tr, P=P):
            if unlock_class(t["unlock"]) != "rolling":
                return cv6.forecast_v3(t, tr, dict(P, w_finger=0.0))
            return forecast_v4(t, tr, P)
        res, cov = evaluate(launches, fn)
        print(f"{name:30s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


if __name__ == "__main__":
    main()
