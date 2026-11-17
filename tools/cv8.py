"""V3 wave 3: fingerprint variants (window, ramp shape), top-k blending under
fingerprint weights, and per-hour weight switching."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import unlock_class, parse, LEVELS, Z  # noqa: E402
from cv4 import evaluate  # noqa: E402
import cv6  # noqa: E402


def fingerprint2(hourly, day_start, lo, hi, mode):
    start = parse(day_start)
    pts = {}
    for pt in hourly:
        off = (parse(pt["from"]) - start).total_seconds() / 3600.0
        if off == int(off) and lo <= off <= hi:
            pts[int(off)] = math.log(max(1, pt["viewers"]))
    if mode == "diurnal":
        by_hour = {}
        for off, v in pts.items():
            by_hour.setdefault(off % 24, []).append(v)
        if len(by_hour) < 18:
            return None
        vec = [sum(by_hour[h]) / len(by_hour[h]) if h in by_hour else 0.0 for h in range(24)]
    else:  # ramp: raw hourly log series over the window, needs near-complete data
        offs = list(range(lo, hi + 1))
        if sum(1 for o in offs if o in pts) < 0.8 * len(offs):
            return None
        vec = [pts.get(o, 0.0) for o in offs]
        # fill gaps by neighbour average
        for i, o in enumerate(offs):
            if o not in pts:
                prev = next((pts[offs[j]] for j in range(i - 1, -1, -1) if offs[j] in pts), 0.0)
                nxt = next((pts[offs[j]] for j in range(i + 1, len(offs)) if offs[j] in pts), prev)
                vec[i] = 0.5 * (prev + nxt)
    mean = sum(vec) / len(vec)
    return [v - mean for v in vec]


def enrich8(case_path, lo, hi, mode):
    case = json.load(open(case_path))
    launches = cv6.enrich6(case_path)
    raw = {c["name"]: c for c in case["comparables"]}
    for L in launches:
        c = raw[L["name"]]
        L["finger"] = fingerprint2(c["hourly"], c["day_start"], lo, hi, mode)
    return launches


def forecast_finger_topk(target, train, P):
    """loclin with finger weights + blend with top-3 finger neighbours' shape."""
    ws = cv6.weights6(target, train, P)
    e1 = cv6.loclin_forecast(target, train, ws, P)
    order = sorted(range(len(train)), key=lambda i: -ws[i])[:3]
    b = P.get("blend_topk", 0.0)
    out = {}
    for h in range(2, 25):
        mu, sd = e1[h]
        if b:
            vals = [math.log(train[i]["curve"][h] / train[i]["p1"])
                    for i in order if h in train[i]["curve"]]
            if vals:
                shape = sum(vals) / len(vals)
                spread = math.sqrt(sum((v - shape) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else sd
                mu = (1 - b) * mu + b * shape
                sd = (1 - b) * sd + b * max(spread, 0.15)
        out[h] = [target["p1"] * math.exp(mu + z * sd) for z in Z]
    return out


def run_configs(launches, label, configs):
    print(f"--- {label}")
    print(f"{'config':34s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for name, over, fn in configs:
        P = dict(cv6.BASE, **over)
        if unlock_only := over.get("_rollonly"):
            def fn2(t, tr, P=P, fn=fn):
                if unlock_class(t["unlock"]) != "rolling":
                    return cv6.forecast_v3(t, tr, dict(P, w_finger=0.0))
                return fn(t, tr, P)
            res, cov = evaluate(launches, fn2)
        else:
            res, cov = evaluate(launches, lambda t, tr, P=P, fn=fn: fn(t, tr, P))
        print(f"{name:34s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    base = {"w_finger": 0.25, "sd_scale": 1.0, "_rollonly": True}
    for label, lo, hi, mode in [
        ("diurnal -168..-25 (current)", -168, -25, "diurnal"),
        ("diurnal -72..-25", -72, -25, "diurnal"),
        ("diurnal -336..-25", -336, -25, "diurnal"),
        ("ramp -48..-1", -48, -1, "ramp"),
    ]:
        launches = enrich8(path, lo, hi, mode)
        run_configs(launches, label, [
            ("finger loclin", base, cv6.forecast_v3),
            ("finger + top3 blend .35", dict(base, blend_topk=0.35), forecast_finger_topk),
            ("finger + top3 blend .55", dict(base, blend_topk=0.55), forecast_finger_topk),
        ])


if __name__ == "__main__":
    main()
