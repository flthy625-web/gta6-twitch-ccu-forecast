"""V2 wave 2: harder class separation, kNN restriction, extra kernels
(channels at hour 1, pre-release growth), per-hour bias offsets for
rolling targets (nested: offsets from other targets' LOO residuals only).
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import load_launches, unlock_class, pinball, LEVELS, Z  # noqa: E402
from cv2 import tokens, name_sim, platforms_set, wls_intercept  # noqa: E402


def enrich(case_path):
    case = json.load(open(case_path))
    launches = load_launches(case_path)
    extra = {}
    from cv import parse
    for comp in case["comparables"]:
        start = parse(comp["day_start"])
        ch1, pre = None, {}
        for pt in comp["hourly"]:
            off = (parse(pt["from"]) - start).total_seconds() / 3600.0
            if off == int(off):
                off = int(off)
                if off == 0:
                    ch1 = pt.get("channels")
                if -30 <= off < 0:
                    pre[off] = max(1, pt["viewers"])
        last6 = [pre[o] for o in range(-6, 0) if o in pre]
        prev = [pre[o] for o in range(-30, -6) if o in pre]
        growth = (sum(last6) / len(last6)) / (sum(prev) / len(prev)) if last6 and prev else None
        extra[comp["name"]] = {"platforms": comp.get("platforms", ""),
                               "channels1": ch1, "growth": growth}
    for L in launches:
        L.update(extra.get(L["name"], {}))
    return launches


def make_weights(target, train, P):
    tcls = unlock_class(target["unlock"])
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
        if P.get("beta_name"):
            w *= math.exp(P["beta_name"] * name_sim(target.get("name", ""), t.get("name", "")))
        if P.get("beta_plat") and tplat:
            sp = platforms_set(t.get("platforms", ""))
            if sp:
                w *= math.exp(P["beta_plat"] * len(tplat & sp) / len(tplat | sp))
        if P.get("w_ch") and target.get("channels1") and t.get("channels1"):
            dc = abs(math.log10(t["channels1"]) - math.log10(target["channels1"]))
            w *= math.exp(-0.5 * (dc / P["w_ch"]) ** 2)
        if P.get("w_growth") and target.get("growth") and t.get("growth"):
            dg = abs(math.log(max(t["growth"], 1e-2)) - math.log(max(target["growth"], 1e-2)))
            w *= math.exp(-0.5 * (dg / P["w_growth"]) ** 2)
        ws.append(w)
    if P.get("knn"):
        order = sorted(range(len(ws)), key=lambda i: -ws[i])
        keep = set(order[:P["knn"]])
        ws = [w if i in keep else 0.0 for i, w in enumerate(ws)]
    return ws


def forecast_v3(target, train, P, bias=None):
    ws = make_weights(target, train, P)
    tx = math.log10(target["p1"])
    out = {}
    for h in range(2, 25):
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in zip(train, ws) if h in t["curve"] and w > 0]
        if not rows:
            mu, sd = 0.0, 0.5
        elif P["loclin"]:
            mu, sd = wls_intercept([r[0] for r in rows], [r[1] for r in rows],
                                   [r[2] for r in rows], P["ridge"])
        else:
            sw = sum(r[2] for r in rows)
            mu = sum(r[1] * r[2] for r in rows) / sw
            sd = math.sqrt(sum(r[2] * (r[1] - mu) ** 2 for r in rows) / sw)
        if bias:
            mu += bias.get(h, 0.0)
        sd = min(max(sd * P["sd_scale"], P["sd_floor"]), P["sd_cap"])
        out[h] = [target["p1"] * math.exp(mu + z * sd) for z in Z]
    return out


def run(launches, P, use_bias=False):
    # first pass: raw forecasts + residuals per target
    raw = {}
    for i, target in enumerate(launches):
        train = launches[:i] + launches[i + 1:]
        raw[i] = forecast_v3(target, train, P)
    biases = {}
    if use_bias:
        # per-hour mean log residual over OTHER rolling targets
        for i, target in enumerate(launches):
            if unlock_class(target["unlock"]) != "rolling":
                continue
            b = {}
            for h in range(2, 25):
                res = []
                for j, other in enumerate(launches):
                    if j == i or unlock_class(other["unlock"]) != "rolling" or h not in other["curve"]:
                        continue
                    med = raw[j][h][2]
                    res.append(math.log(other["curve"][h]) - math.log(med))
                if len(res) >= 5:
                    b[h] = sum(res) / len(res)
            biases[i] = b
    groups = {"all": [], "rolling": [], "bigroll": []}
    covs = []
    for i, target in enumerate(launches):
        fc = raw[i]
        if use_bias and i in biases:
            train = launches[:i] + launches[i + 1:]
            fc = forecast_v3(target, train, P, bias=biases[i])
        tot, n, hits, pts = 0.0, 0, 0, 0
        for h in range(2, 25):
            if h not in target["curve"] or h not in fc:
                continue
            y = target["curve"][h]
            for tau, q in zip(LEVELS, fc[h]):
                tot += pinball(y, max(q, 1e-9), tau)
                n += 1
            hits += 1 if fc[h][0] <= y <= fc[h][4] else 0
            pts += 1
        if not n:
            continue
        loss = tot / n
        groups["all"].append(loss)
        if unlock_class(target["unlock"]) == "rolling":
            groups["rolling"].append(loss)
            if target["p1"] >= 50000:
                groups["bigroll"].append(loss)
        covs.append(hits / pts if pts else 0.0)
    return {k: (sum(v) / len(v) if v else float("nan")) for k, v in groups.items()}, \
        (sum(covs) / len(covs) if covs else float("nan"))


BASE = dict(w_p1=0.6, w_pre=1.6, w_mismatch=0.2, w_unknown=0.7,
            sd_floor=0.22, sd_cap=1.6, sd_scale=0.85,
            loclin=True, ridge=0.05, beta_name=0.0, beta_plat=0.0,
            w_ch=0.0, w_growth=0.0, knn=0)

CONFIGS = [
    ("base loclin sd.85", {}, False),
    ("mismatch .1", {"w_mismatch": 0.1}, False),
    ("mismatch .05 unknown .5", {"w_mismatch": 0.05, "w_unknown": 0.5}, False),
    ("knn 12", {"knn": 12}, False),
    ("knn 8", {"knn": 8}, False),
    ("channels kernel .8", {"w_ch": 0.8}, False),
    ("growth kernel 1.0", {"w_growth": 1.0}, False),
    ("ch .8 + growth 1.0", {"w_ch": 0.8, "w_growth": 1.0}, False),
    ("plat .5 + name 1", {"beta_plat": 0.5, "beta_name": 1.0}, False),
    ("bias rolling", {}, True),
    ("mismatch .1 + knn12 + bias", {"w_mismatch": 0.1, "knn": 12}, True),
    ("mismatch .1 + ch.8 + growth1 + bias", {"w_mismatch": 0.1, "w_ch": 0.8, "w_growth": 1.0}, True),
    ("everything", {"w_mismatch": 0.1, "w_ch": 0.8, "w_growth": 1.0, "beta_plat": 0.5, "beta_name": 1.0, "knn": 14}, True),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    launches = enrich(path)
    print(f"{len(launches)} launches")
    print(f"{'config':36s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for name, over, ub in CONFIGS:
        P = dict(BASE, **over)
        res, cov = run(launches, P, use_bias=ub)
        print(f"{name:36s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


if __name__ == "__main__":
    main()
