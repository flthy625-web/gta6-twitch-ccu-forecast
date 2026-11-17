"""V2 cross-validation: parametrized forecaster, LOO over the comparable pool.

Views: all targets, rolling-midnight targets, and the practice-like subset
(rolling + hour-1 >= 50k). Also reports 90% coverage (nominal 0.90).
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import load_launches, unlock_class, pinball, LEVELS, Z  # noqa: E402

STOP = {"the", "of", "and", "a"}


def tokens(name):
    raw = "".join(c if c.isalnum() else " " for c in name.lower()).split()
    return {t for t in raw if t not in STOP and len(t) >= 2}


def name_sim(a, b):
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def platforms_set(text):
    return {p.strip().lower() for p in (text or "").replace("/", ",").split(",") if p.strip()}


def make_weights(target, train, P):
    tcls = unlock_class(target["unlock"])
    tname = target.get("name", "")
    tplat = platforms_set(target.get("platforms", ""))
    ws = []
    for t in train:
        c = unlock_class(t["unlock"])
        if c == tcls:
            w = 1.0
        elif "unknown" in (c, tcls):
            w = P["w_unknown"]
        else:
            w = P["w_mismatch"]
        d = abs(math.log10(t["p1"]) - math.log10(target["p1"]))
        w *= math.exp(-0.5 * (d / P["w_p1"]) ** 2)
        if target["pre_ratio"] and t["pre_ratio"]:
            dp = abs(math.log(max(t["pre_ratio"], 1e-3)) - math.log(max(target["pre_ratio"], 1e-3)))
            w *= math.exp(-0.5 * (dp / P["w_pre"]) ** 2)
        if P["beta_name"] and tname:
            w *= math.exp(P["beta_name"] * name_sim(tname, t.get("name", "")))
        if P["beta_plat"] and tplat:
            sp = platforms_set(t.get("platforms", ""))
            if sp:
                jac = len(tplat & sp) / len(tplat | sp)
                w *= math.exp(P["beta_plat"] * jac)
        ws.append(w)
    return ws


def wls_intercept(xs, ys, ws, ridge):
    """Weighted ridge fit y ~ a + b.x (x already centred on the target);
    returns (a, resid_sd) where resid uses fitted values."""
    n = len(ys)
    if n == 0:
        return 0.0, 0.5
    sw = sum(ws)
    if sw <= 0:
        return sum(ys) / n, 0.5
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws)) + ridge
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx if sxx > 0 else 0.0
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / sw
    return a + b * (0.0 - 0.0) if False else (my + b * (0.0 - mx)), math.sqrt(max(var, 0.0))


def forecast_v2(target, train, P):
    ws = make_weights(target, train, P)
    tx = math.log10(target["p1"])
    mus, sds = {}, {}
    for h in range(2, 25):
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in zip(train, ws) if h in t["curve"]]
        if not rows:
            mus[h], sds[h] = 0.0, 0.5
            continue
        if P["loclin"]:
            mu, sd = wls_intercept([r[0] for r in rows], [r[1] for r in rows],
                                   [r[2] for r in rows], P["ridge"])
        else:
            sw = sum(r[2] for r in rows)
            mu = sum(r[1] * r[2] for r in rows) / sw
            sd = math.sqrt(sum(r[2] * (r[1] - mu) ** 2 for r in rows) / sw)
        mus[h], sds[h] = mu, sd
    if P["smooth"]:
        for src in (mus, sds):
            sm = {}
            for h in range(2, 25):
                vals = [src[x] for x in (h - 1, h, h + 1) if x in src]
                sm[h] = sum(vals) / len(vals)
            src.update(sm)
    out = {}
    for h in range(2, 25):
        sd = min(max(sds[h] * P["sd_scale"], P["sd_floor"]), P["sd_cap"])
        out[h] = [target["p1"] * math.exp(mus[h] + z * sd) for z in Z]
    return out


def evaluate(launches, P):
    groups = {"all": [], "rolling": [], "bigroll": []}
    cov = []
    for i, target in enumerate(launches):
        train = launches[:i] + launches[i + 1:]
        fc = forecast_v2(target, train, P)
        tot, n, hits, tot_pts = 0.0, 0, 0, 0
        for h in range(2, 25):
            if h not in target["curve"] or h not in fc:
                continue
            y = target["curve"][h]
            for tau, q in zip(LEVELS, fc[h]):
                tot += pinball(y, max(q, 1e-9), tau)
                n += 1
            if fc[h][0] <= y <= fc[h][4]:
                hits += 1
            tot_pts += 1
        if not n:
            continue
        loss = tot / n
        groups["all"].append(loss)
        if unlock_class(target["unlock"]) == "rolling":
            groups["rolling"].append(loss)
            if target["p1"] >= 50000:
                groups["bigroll"].append(loss)
        cov.append(hits / tot_pts if tot_pts else 0.0)
    return {k: (sum(v) / len(v) if v else float("nan"), len(v)) for k, v in groups.items()}, \
        (sum(cov) / len(cov) if cov else float("nan"))


BASE = dict(w_p1=0.6, w_pre=1.6, w_mismatch=0.2, w_unknown=0.7,
            sd_floor=0.22, sd_cap=1.6, sd_scale=1.0,
            loclin=False, ridge=0.05, smooth=False, beta_name=0.0, beta_plat=0.0)

CONFIGS = [
    ("v1 (as submitted)", {}),
    ("sd_scale .8", {"sd_scale": 0.8}),
    ("sd_scale .7", {"sd_scale": 0.7}),
    ("sd .8 floor .15", {"sd_scale": 0.8, "sd_floor": 0.15}),
    ("loclin", {"loclin": True}),
    ("loclin + sd .85", {"loclin": True, "sd_scale": 0.85}),
    ("loclin + sd .8", {"loclin": True, "sd_scale": 0.8}),
    ("loclin + sd .8 + smooth", {"loclin": True, "sd_scale": 0.8, "smooth": True}),
    ("loclin + sd .8 + name 1.0", {"loclin": True, "sd_scale": 0.8, "beta_name": 1.0}),
    ("loclin + sd .8 + name 2.0", {"loclin": True, "sd_scale": 0.8, "beta_name": 2.0}),
    ("loclin + sd .8 + plat 0.7", {"loclin": True, "sd_scale": 0.8, "beta_plat": 0.7}),
    ("loclin + sd .8 + name1 + plat.7", {"loclin": True, "sd_scale": 0.8, "beta_name": 1.0, "beta_plat": 0.7}),
    ("loclin ridge .2 + sd .8", {"loclin": True, "sd_scale": 0.8, "ridge": 0.2}),
    ("loclin ridge .01 + sd .8", {"loclin": True, "sd_scale": 0.8, "ridge": 0.01}),
    ("wider p1 kernel .9 + loclin + sd .8", {"loclin": True, "sd_scale": 0.8, "w_p1": 0.9}),
    ("narrow p1 .45 + loclin + sd .8", {"loclin": True, "sd_scale": 0.8, "w_p1": 0.45}),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    case = json.load(open(path))
    launches = load_launches(path)
    # attach platforms for the platform-similarity feature
    plats = {c["name"]: c.get("platforms", "") for c in case["comparables"]}
    for L in launches:
        L["platforms"] = plats.get(L["name"], "")
    print(f"{len(launches)} launches")
    print(f"{'config':38s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'n_br':>4s} {'cov90':>6s}")
    for name, over in CONFIGS:
        P = dict(BASE, **over)
        res, cov = evaluate(launches, P)
        print(f"{name:38s} {res['all'][0]:7.4f} {res['rolling'][0]:8.4f} "
              f"{res['bigroll'][0]:8.4f} {res['bigroll'][1]:4d} {cov:6.3f}")


if __name__ == "__main__":
    main()
