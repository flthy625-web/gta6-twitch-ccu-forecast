"""Coarse grid search for the weighted-lognormal forecaster, scored by the same
leave-one-out CV as cv.py. Prefers plateaus: reports the best combos on all
targets and on rolling-midnight targets separately."""
import itertools
import json
import math
import sys

sys.path.insert(0, "tools")
from cv import LEVELS, Z, load_launches, unlock_class, score  # noqa: E402

def make_forecast(target, train, w_p1, w_pre, w_mismatch, w_unknown, sd_floor, sd_scale, weekday_w):
    tcls = unlock_class(target["unlock"])
    pairs_by_h = {h: [] for h in range(2, 25)}
    twd = target["wd"]
    for t in train:
        c = unlock_class(t["unlock"])
        if c == tcls:
            w = 1.0
        elif "unknown" in (c, tcls):
            w = w_unknown
        else:
            w = w_mismatch
        d = abs(math.log10(t["p1"]) - math.log10(target["p1"]))
        w *= math.exp(-0.5 * (d / w_p1) ** 2)
        if target["pre_ratio"] and t["pre_ratio"]:
            dp = abs(math.log(max(t["pre_ratio"], 1e-3)) - math.log(max(target["pre_ratio"], 1e-3)))
            w *= math.exp(-0.5 * (dp / w_pre) ** 2)
        if weekday_w and t["wd"] != twd:
            w *= weekday_w
        for h in range(2, 25):
            if h in t["curve"]:
                pairs_by_h[h].append((math.log(t["curve"][h] / t["p1"]), w))
    out = {}
    for h in range(2, 25):
        pairs = pairs_by_h[h]
        sw = sum(w for _, w in pairs)
        mu = sum(v * w for v, w in pairs) / sw
        var = sum(w * (v - mu) ** 2 for v, w in pairs) / sw
        sd = min(max(math.sqrt(var) * sd_scale, sd_floor), 1.6)
        out[h] = [target["p1"] * math.exp(mu + z * sd) for z in Z]
    return out

def main(path):
    launches = load_launches(path)
    from datetime import datetime
    for L in launches:
        pass
    # weekday needs day_start: reload with it
    case = json.load(open(path))
    wd = {}
    for comp in case["comparables"]:
        wd[comp["name"]] = datetime.strptime(comp["day_start"], "%Y-%m-%dT%H:%M:%SZ").weekday()
    for L in launches:
        L["wd"] = wd[L["name"]]
    grid = list(itertools.product(
        [0.6, 0.9, 1.3],        # w_p1
        [0.7, 1.1, 1.6],        # w_pre
        [0.2, 0.3, 0.45],       # class mismatch
        [0.45, 0.55, 0.7],      # unknown class
        [0.18, 0.22, 0.28],     # sd floor
        [0.9, 1.0, 1.1],        # sd scale
        [None, 0.6],            # weekday mismatch weight
    ))
    print(f"{len(grid)} combos x {len(launches)} folds")
    rows = []
    for gi, g in enumerate(grid):
        tot, n, rtot, rn = 0.0, 0, 0.0, 0
        for i, target in enumerate(launches):
            train = launches[:i] + launches[i + 1:]
            s = score(make_forecast(target, train, *g), target["curve"])
            if s is not None:
                tot += s; n += 1
                if unlock_class(target["unlock"]) == "rolling":
                    rtot += s; rn += 1
        rows.append((tot / n, rtot / rn, g))
        if gi % 400 == 0:
            print(f"  ...{gi}/{len(grid)}", flush=True)
    rows.sort(key=lambda r: r[0])
    print("\nbest by ALL targets:")
    for a, r, g in rows[:6]:
        print(f"  all={a:.4f} rolling={r:.4f}  w_p1={g[0]} w_pre={g[1]} mismatch={g[2]} unknown={g[3]} sd_floor={g[4]} sd_scale={g[5]} weekday={g[6]}")
    rows.sort(key=lambda r: r[1])
    print("\nbest by ROLLING targets:")
    for a, r, g in rows[:6]:
        print(f"  all={a:.4f} rolling={r:.4f}  w_p1={g[0]} w_pre={g[1]} mismatch={g[2]} unknown={g[3]} sd_floor={g[4]} sd_scale={g[5]} weekday={g[6]}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in")
