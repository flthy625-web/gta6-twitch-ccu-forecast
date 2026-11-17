"""V2 wave 3: (A) per-hour global ridge regression on launch features;
(B) local-linear-adjusted weighted empirical quantiles.
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import load_launches, unlock_class, pinball, LEVELS, Z  # noqa: E402
from cv2 import wls_intercept  # noqa: E402
from cv3 import enrich, make_weights  # noqa: E402


def features(L):
    cls = unlock_class(L["unlock"])
    return [
        1.0,
        math.log10(L["p1"]) - 5.0,
        math.log(max(L["pre_ratio"] or 0.01, 1e-3)),
        math.log(max(L.get("growth") or 1.0, 1e-2)),
        1.0 if cls == "global" else 0.0,
        1.0 if cls == "early" else 0.0,
        math.log10(max(L.get("channels1") or 100, 1)) - 3.0,
    ]


def solve(A, b):
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                f = M[r][col]
                M[r] = [x - f * y for x, y in zip(M[r], M[col])]
    return [M[i][n] for i in range(n)]


def ridge_fit(X, y, lam):
    d = len(X[0])
    A = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for xi, yi in zip(X, y):
        for i in range(d):
            b[i] += xi[i] * yi
            for j in range(d):
                A[i][j] += xi[i] * xj if False else xi[i] * xi[j]
    for i in range(1, d):  # don't penalise the intercept
        A[i][i] += lam
    return solve(A, b)


def forecast_ridge(target, train, lam, sd_scale, sd_floor=0.15):
    Xt = [features(t) for t in train]
    ft = features(target)
    out = {}
    sds = {}
    for h in range(2, 25):
        ys = [math.log(t["curve"][h] / t["p1"]) for t in train if h in t["curve"]]
        Xs = [x for t, x in zip(train, Xt) if h in t["curve"]]
        if len(ys) < 10:
            mu, sd = (sum(ys) / len(ys) if ys else 0.0), 0.5
        else:
            beta = ridge_fit(Xs, ys, lam)
            mu = sum(b * f for b, f in zip(beta, ft))
            res = [y - sum(b * f for b, f in zip(beta, x)) for x, y in zip(Xs, ys)]
            sd = math.sqrt(sum(r * r for r in res) / max(len(res) - len(ft), 1))
        sds[h] = sd
        out[h] = mu
    gsd = sum(sds.values()) / len(sds)
    res_out = {}
    for h in range(2, 25):
        sd = min(max((0.5 * sds[h] + 0.5 * gsd) * sd_scale, sd_floor), 1.6)
        res_out[h] = [target["p1"] * math.exp(out[h] + z * sd) for z in Z]
    return res_out


def wquantile(pairs, tau):
    """pairs: [(value, weight)] -> weighted empirical quantile."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return pairs[len(pairs) // 2][0] if pairs else 0.0
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc / tot >= tau:
            if i == 0:
                return v
            pv, pw = pairs[i - 1]
            prev_acc = acc - w
            frac = (tau * tot - prev_acc) / w if w > 0 else 0.0
            return pv + (v - pv) * min(max(frac, 0.0), 1.0)
    return pairs[-1][0]


BASE3 = dict(w_p1=0.6, w_pre=1.6, w_mismatch=0.1, w_unknown=0.6,
             sd_floor=0.22, sd_cap=1.6, sd_scale=0.85,
             loclin=True, ridge=0.05, beta_name=1.0, beta_plat=0.5,
             w_ch=0.0, w_growth=0.0, knn=0)


def forecast_adj_emp(target, train, P, sd_scale):
    ws = make_weights(target, train, P)
    tx = math.log10(target["p1"])
    out = {}
    for h in range(2, 25):
        rows = [(math.log10(t["p1"]) - tx, math.log(t["curve"][h] / t["p1"]), w)
                for t, w in zip(train, ws) if h in t["curve"] and w > 0]
        if not rows:
            out[h] = [target["p1"]] * 5
            continue
        mu, _sd = wls_intercept([r[0] for r in rows], [r[1] for r in rows],
                                [r[2] for r in rows], P["ridge"])
        # slope from the same fit: recompute quickly
        sw = sum(r[2] for r in rows)
        mx = sum(r[0] * r[2] for r in rows) / sw
        my = sum(r[1] * r[2] for r in rows) / sw
        sxx = sum(r[2] * (r[0] - mx) ** 2 for r in rows) + P["ridge"]
        sxy = sum(r[2] * (r[0] - mx) * (r[1] - my) for r in rows)
        b = sxy / sxx if sxx > 0 else 0.0
        adj = [(r[1] - b * r[0], r[2]) for r in rows]
        qs = [wquantile(adj, tau) for tau in LEVELS]
        # widen/narrow around the median by sd_scale in log space
        med = qs[2]
        qs = [med + (q - med) * sd_scale for q in qs]
        out[h] = [target["p1"] * math.exp(q) for q in qs]
    return out


def evaluate(launches, fn):
    groups = {"all": [], "rolling": [], "bigroll": []}
    covs = []
    for i, target in enumerate(launches):
        train = launches[:i] + launches[i + 1:]
        fc = fn(target, train)
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    launches = enrich(path)
    print(f"{len(launches)} launches")
    print(f"{'config':34s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for lam in (3.0, 10.0, 30.0):
        for sds in (0.85, 1.0):
            res, cov = evaluate(launches, lambda t, tr, lam=lam, sds=sds: forecast_ridge(t, tr, lam, sds))
            print(f"{'ridge lam=' + str(lam) + ' sd=' + str(sds):34s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")
    for sds in (0.9, 1.0, 1.1):
        res, cov = evaluate(launches, lambda t, tr, sds=sds: forecast_adj_emp(t, tr, BASE3, sds))
        print(f"{'adj-empirical scale=' + str(sds):34s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


if __name__ == "__main__":
    main()
