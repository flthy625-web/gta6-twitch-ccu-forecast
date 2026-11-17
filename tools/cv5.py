"""V2 wave 4: sharpen the champion (loclin lognormal) — weight power,
best-analogue blending, weighted-median mu, block-wise sd scales, pre width.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import unlock_class, pinball, LEVELS, Z  # noqa: E402
from cv2 import wls_intercept  # noqa: E402
from cv3 import enrich, make_weights  # noqa: E402
from cv4 import wquantile, evaluate  # noqa: E402


def forecast_v5(target, train, P):
    ws = make_weights(target, train, P)
    if P.get("wpower"):
        ws = [w ** P["wpower"] for w in ws]
    tx = math.log10(target["p1"])
    best = None
    if P.get("blend") and train:
        bi = max(range(len(train)), key=lambda i: ws[i])
        best = train[bi]
    out = {}
    for h in range(2, 25):
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in zip(train, ws) if h in t["curve"] and w > 0]
        if not rows:
            mu, sd = 0.0, 0.5
        else:
            mu, sd = wls_intercept([r[0] for r in rows], [r[1] for r in rows],
                                   [r[2] for r in rows], P["ridge"])
            if P.get("median_mu"):
                sw = sum(r[2] for r in rows)
                mx = sum(r[0] * r[2] for r in rows) / sw
                my = sum(r[1] * r[2] for r in rows) / sw
                sxx = sum(r[2] * (r[0] - mx) ** 2 for r in rows) + P["ridge"]
                sxy = sum(r[2] * (r[0] - mx) * (r[1] - my) for r in rows)
                b = sxy / sxx if sxx > 0 else 0.0
                mu = wquantile([(r[1] - b * r[0], r[2]) for r in rows], 0.5)
        if best is not None and h in best["curve"]:
            mu = (1 - P["blend"]) * mu + P["blend"] * math.log(best["curve"][h] / best["p1"])
        scale = P["sd_early"] if h <= 8 else P["sd_late"]
        sd = min(max(sd * scale, P["sd_floor"]), P["sd_cap"])
        out[h] = [target["p1"] * math.exp(mu + z * sd) for z in Z]
    return out


BASE = dict(w_p1=0.6, w_pre=1.6, w_mismatch=0.1, w_unknown=0.6,
            sd_floor=0.22, sd_cap=1.6, sd_scale=0.85,
            loclin=True, ridge=0.05, beta_name=1.0, beta_plat=0.5,
            w_ch=0.0, w_growth=0.0, knn=0,
            wpower=0.0, blend=0.0, median_mu=False, sd_early=0.85, sd_late=0.85)

CONFIGS = [
    ("champion", {}),
    ("wpower 1.5", {"wpower": 1.5}),
    ("wpower 2", {"wpower": 2.0}),
    ("blend .2", {"blend": 0.2}),
    ("blend .35", {"blend": 0.35}),
    ("median mu", {"median_mu": True}),
    ("sd early .8 late .9", {"sd_early": 0.8, "sd_late": 0.9}),
    ("sd early .9 late .8", {"sd_early": 0.9, "sd_late": 0.8}),
    ("w_pre 1.0", {"w_pre": 1.0}),
    ("w_pre 0.7", {"w_pre": 0.7}),
    ("wpower1.5 + blend.2", {"wpower": 1.5, "blend": 0.2}),
    ("wpower1.5 + sd .8/.9", {"wpower": 1.5, "sd_early": 0.8, "sd_late": 0.9}),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    launches = enrich(path)
    print(f"{len(launches)} launches")
    print(f"{'config':28s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for name, over in CONFIGS:
        P = dict(BASE, **over)
        res, cov = evaluate(launches, lambda t, tr, P=P: forecast_v5(t, tr, P))
        print(f"{name:28s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


if __name__ == "__main__":
    main()
