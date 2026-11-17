"""V3 wave 2: tune the diurnal-fingerprint kernel (width, sd recalibration,
rolling-only gating, fingerprint window)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv import unlock_class  # noqa: E402
from cv4 import evaluate  # noqa: E402
import cv6  # noqa: E402


def forecast_gated(target, train, P):
    if P.get("finger_rolling_only") and unlock_class(target["unlock"]) != "rolling":
        P = dict(P, w_finger=0.0)
    return cv6.forecast_v3(target, train, P)


CONFIGS = []
for wf in (0.18, 0.25, 0.35, 0.5):
    for sds in (0.85, 1.0, 1.15):
        CONFIGS.append((f"finger {wf} sd {sds}", {"w_finger": wf, "sd_scale": sds}, False))
CONFIGS += [
    ("finger .25 sd 1.0 rollonly", {"w_finger": 0.25, "sd_scale": 1.0, "finger_rolling_only": True}, True),
    ("finger .35 sd 1.0 rollonly", {"w_finger": 0.35, "sd_scale": 1.0, "finger_rolling_only": True}, True),
    ("finger .25 sd 1.15 rollonly", {"w_finger": 0.25, "sd_scale": 1.15, "finger_rolling_only": True}, True),
]


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "../gta6-twitch-ccu-forecast-01-kit/input/p00.in"
    launches = cv6.enrich6(path)
    print(f"{'config':32s} {'all':>7s} {'rolling':>8s} {'bigroll':>8s} {'cov90':>6s}")
    for name, over, gated in CONFIGS:
        P = dict(cv6.BASE, **over)
        fn = (lambda t, tr, P=P: forecast_gated(t, tr, P)) if gated else (lambda t, tr, P=P: cv6.forecast_v3(t, tr, P))
        res, cov = evaluate(launches, fn)
        print(f"{name:32s} {res['all']:7.4f} {res['rolling']:8.4f} {res['bigroll']:8.4f} {cov:6.3f}")


if __name__ == "__main__":
    main()
