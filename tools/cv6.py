"""V3 experiments: unlock-time features, diurnal-fingerprint similarity,
and expert stacking (v2 loclin + top-k shape median + same-class loclin).
"""
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import load_launches, unlock_class, pinball, parse, LEVELS, Z  # noqa: E402
from cv2 import wls_intercept  # noqa: E402
from cv3 import enrich  # noqa: E402
from cv4 import wquantile, evaluate  # noqa: E402
import cv3  # noqa: E402


def unlock_times(text, day_start):
    """(first_off, us_wave): hours of the earliest dated unlock relative to
    day_start, and the US Eastern wave hour (UTC hour of day)."""
    t = text or ""
    start = parse(day_start)
    offs = []
    for m in re.finditer(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})Z", t):
        y, mo, d, hh, mm = (int(x) for x in m.groups())
        from datetime import datetime, timezone
        dt = datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)
        offs.append((dt - start).total_seconds() / 3600.0)
    first_off = min(offs) if offs else None
    us = None
    m = re.search(r"Eastern[^;.]*?=\s*(\d{1,2}):(\d{2})Z", t)
    if m:
        us = int(m.group(1)) + int(m.group(2)) / 60.0
    return first_off, us


def diurnal_fingerprint(hourly, day_start):
    """Mean log viewers by UTC hour-of-day over offsets -168..-25, centred."""
    start = parse(day_start)
    by_hour = {}
    for pt in hourly:
        off = (parse(pt["from"]) - start).total_seconds() / 3600.0
        if off != int(off):
            continue
        off = int(off)
        if -168 <= off <= -25:
            hod = (off % 24)
            by_hour.setdefault(hod, []).append(math.log(max(1, pt["viewers"])))
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


def enrich6(case_path):
    case = json.load(open(case_path))
    launches = enrich(case_path)
    raw = {c["name"]: c for c in case["comparables"]}
    for L in launches:
        c = raw[L["name"]]
        L["first_off"], L["us_wave"] = unlock_times(c.get("unlock"), c["day_start"])
        cls = unlock_class(c.get("unlock"))
        if L["first_off"] is None:
            L["first_off"] = -13.0 if cls == "rolling" else 0.0
        if L["us_wave"] is None:
            L["us_wave"] = 5.0 if cls == "rolling" else (L["first_off"] % 24)
        L["finger"] = diurnal_fingerprint(c["hourly"], c["day_start"])
    return launches


def weights6(target, train, P):
    ws = cv3.make_weights(target, train, P)
    out = []
    for t, w in zip(train, ws):
        if P.get("w_us"):
            du = abs((t["us_wave"] or 5.0) - (target["us_wave"] or 5.0))
            du = min(du, 24 - du)
            w *= math.exp(-0.5 * (du / P["w_us"]) ** 2)
        if P.get("w_first"):
            df = abs((t["first_off"] or 0.0) - (target["first_off"] or 0.0))
            w *= math.exp(-0.5 * (df / P["w_first"]) ** 2)
        if P.get("w_finger"):
            cs = cosine(target.get("finger"), t.get("finger"))
            if cs is not None:
                w *= math.exp(-(1.0 - cs) / P["w_finger"])
        out.append(w)
    return out


def loclin_forecast(target, train, ws, P, sd_scale=None):
    tx = math.log10(target["p1"])
    out = {}
    for h in range(2, 25):
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in zip(train, ws) if h in t["curve"] and w > 0]
        if rows:
            mu, sd = wls_intercept([r[0] for r in rows], [r[1] for r in rows],
                                   [r[2] for r in rows], P["ridge"])
        else:
            mu, sd = 0.0, 0.5
        s = min(max(sd * (sd_scale or P["sd_scale"]), P["sd_floor"]), P["sd_cap"])
        out[h] = (mu, s)
    return out


def topk_forecast(target, train, ws, k=5):
    order = sorted(range(len(train)), key=lambda i: -ws[i])[:k]
    top = [(train[i], ws[i]) for i in order]
    out = {}
    for h in range(2, 25):
        pairs = [(math.log(t["curve"][h] / t["p1"]), w) for t, w in top if h in t["curve"]]
        if not pairs:
            out[h] = (0.0, 0.5)
            continue
        med = wquantile(pairs, 0.5)
        mad = wquantile([(abs(v - med), w) for v, w in pairs], 0.5)
        out[h] = (med, max(1.4826 * mad, 0.18))
    return out


def to_quantiles(target, mus_sd):
    return {h: [target["p1"] * math.exp(mu + z * sd) for z in Z]
            for h, (mu, sd) in mus_sd.items()}


def forecast_v3(target, train, P):
    ws = weights6(target, train, P)
    e1 = loclin_forecast(target, train, ws, P)
    if P.get("blend_topk"):
        e2 = topk_forecast(target, train, ws, k=P.get("topk", 5))
        b = P["blend_topk"]
        mix = {h: ((1 - b) * e1[h][0] + b * e2[h][0],
                   (1 - b) * e1[h][1] + b * e2[h][1]) for h in e1}
        return to_quantiles(target, mix)
    if P.get("blend_class"):
        tcls = unlock_class(target["unlock"])
        sub = [(t, w) for t, w in zip(train, ws) if unlock_class(t["unlock"]) == tcls]
        if len(sub) >= 6:
            e3 = loclin_forecast(target, [t for t, _ in sub], [w for _, w in sub], P)
            b = P["blend_class"]
            mix = {h: ((1 - b) * e1[h][0] + b * e3[h][0],
                       (1 - b) * e1[h][1] + b * e3[h][1]) for h in e1}
            return to_quantiles(target, mix)
    return to_quantiles(target, e1)


BASE = dict(w_p1=0.6, w_pre=1.6, w_mismatch=0.1, w_unknown=0.6,
            sd_floor=0.22, sd_cap=1.6, sd_scale=0.85,
            loclin=True, ridge=0.05, beta_name=1.0, beta_plat=0.5,
            w_ch=0.0, w_growth=0.0, knn=0,
            w_us=0.0, w_first=0.0, w_finger=0.0, blend_topk=0.0, blend_class=0.0)

CONFIGS = [
    ("v2 champion", {}),
    ("+ us wave 2.0", {"w_us": 2.0}),
    ("+ us wave 1.2", {"w_us": 1.2}),
    ("+ first off 6", {"w_first": 6.0}),
    ("+ finger .25", {"w_finger": 0.25}),
    ("+ finger .12", {"w_finger": 0.12}),
    ("+ us2 + finger.25", {"w_us": 2.0, "w_finger": 0.25}),
    ("+ topk blend .3", {"blend_topk": 0.3, "topk": 5}),
    ("+ topk blend .5", {"blend_topk": 0.5, "topk": 5}),
    ("+ class blend .35", {"blend_class": 0.35}),
    ("+ finger.25 + topk.3", {"w_finger": 0.25, "blend_topk": 0.3, "topk": 5}),
    ("+ us2 + finger.25 + topk.3", {"w_us": 2.0, "w_finger": 0.25, "blend_topk": 0.3, "topk": 5}),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    launches = enrich6(path)
    got_f = sum(1 for L in launches if L.get("finger"))
    print(f"{len(launches)} launches, fingerprints for {got_f}")
    print(f"{'config':30s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for name, over in CONFIGS:
        P = dict(BASE, **over)
        res, cov = evaluate(launches, lambda t, tr, P=P: forecast_v3(t, tr, P))
        print(f"{name:30s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


if __name__ == "__main__":
    main()
