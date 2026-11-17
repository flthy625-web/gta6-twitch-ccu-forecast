"""V4 wave 2: (A) two-regressor loclin (size gap + first-unlock gap);
(B) high-similarity subset refit blended with the full loclin."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import unlock_class, LEVELS, Z  # noqa: E402
from cv2 import wls_intercept  # noqa: E402
from cv4 import evaluate  # noqa: E402
import cv6  # noqa: E402
import cv9  # noqa: E402


def wls2(rows, ridge):
    """rows: [(x1, x2, y, w)] -> (a, resid_sd) for y ~ a + b1 x1 + b2 x2."""
    sw = sum(r[3] for r in rows)
    if sw <= 0:
        return 0.0, 0.5
    m1 = sum(r[0] * r[3] for r in rows) / sw
    m2 = sum(r[1] * r[3] for r in rows) / sw
    my = sum(r[2] * r[3] for r in rows) / sw
    s11 = sum(r[3] * (r[0] - m1) ** 2 for r in rows) + ridge
    s22 = sum(r[3] * (r[1] - m2) ** 2 for r in rows) + ridge
    s12 = sum(r[3] * (r[0] - m1) * (r[1] - m2) for r in rows)
    s1y = sum(r[3] * (r[0] - m1) * (r[2] - my) for r in rows)
    s2y = sum(r[3] * (r[1] - m2) * (r[2] - my) for r in rows)
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        b1 = b2 = 0.0
    else:
        b1 = (s1y * s22 - s2y * s12) / det
        b2 = (s2y * s11 - s1y * s12) / det
    a = my - b1 * m1 - b2 * m2
    var = sum(r[3] * (r[2] - (a + b1 * r[0] + b2 * r[1])) ** 2 for r in rows) / sw
    return a, math.sqrt(max(var, 0.0))


def forecast_v5(target, train, P):
    ws = cv6.weights6(target, train, P)
    out = []
    for t, w in zip(train, ws):
        cs = cv6.cosine(target.get("finger_ch"), t.get("finger_ch"))
        if cs is not None and P.get("w_finger_ch"):
            w *= math.exp(-(1.0 - cs) / P["w_finger_ch"])
        out.append(w)
    ws = out
    tx = math.log10(target["p1"])
    tfo = target.get("first_off") or 0.0
    res = {}
    for h in range(2, 25):
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in zip(train, ws) if h in t["curve"] and w > 0]
        if not rows:
            res[h] = (0.0, 0.5)
            continue
        if P.get("loclin2"):
            rows2 = [(r[0], (t.get("first_off") or 0.0) - tfo, r[1], r[2])
                     for r, (t, w) in zip(rows, [(t, w) for t, w in zip(train, ws) if h in t["curve"] and w > 0])]
            mu, sd = wls2(rows2, P["ridge"])
        else:
            mu, sd = wls_intercept([r[0] for r in rows], [r[1] for r in rows],
                                   [r[2] for r in rows], P["ridge"])
        if P.get("subset_blend"):
            cosines = [cv6.cosine(target.get("finger"), t.get("finger")) for t in train]
            sub = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                   for (t, w), cs in zip(zip(train, ws), cosines)
                   if h in t["curve"] and w > 0 and cs is not None and cs > P.get("subset_cos", 0.9)]
            if len(sub) >= 4:
                mu2, sd2 = wls_intercept([r[0] for r in sub], [r[1] for r in sub],
                                         [r[2] for r in sub], P["ridge"])
                b = P["subset_blend"]
                mu = (1 - b) * mu + b * mu2
                sd = (1 - b) * sd + b * sd2
        s = min(max(sd * 1.0, P["sd_floor"]), P["sd_cap"])
        res[h] = (mu, s)
    return {h: [target["p1"] * math.exp(mu + z * sd) for z in Z] for h, (mu, sd) in res.items()}


BASE = dict(cv9.BASE, w_finger=0.3, w_finger_ch=0.5, loclin2=False, subset_blend=0.0)

CONFIGS = [
    ("v4 (ch finger)", {}),
    ("+ loclin2 unlock", {"loclin2": True}),
    ("+ subset blend .4", {"subset_blend": 0.4, "subset_cos": 0.9}),
    ("+ subset blend .4 cos.8", {"subset_blend": 0.4, "subset_cos": 0.8}),
    ("+ loclin2 + subset .4", {"loclin2": True, "subset_blend": 0.4, "subset_cos": 0.9}),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    launches = cv9.enrich9(path)
    print(f"{'config':28s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for name, over in CONFIGS:
        P = dict(BASE, **over)
        def fn(t, tr, P=P):
            if unlock_class(t["unlock"]) != "rolling":
                return cv6.forecast_v3(t, tr, dict(P, w_finger=0.0))
            return forecast_v5(t, tr, P)
        res, cov = evaluate(launches, fn)
        print(f"{name:28s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


if __name__ == "__main__":
    main()
