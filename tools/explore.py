"""Explore the comparables inside a practice case: for each past launch, its
release-day hour-1 viewers, peak hour, and how complete its release day is."""
import json
import sys
from datetime import datetime, timedelta, timezone

def parse(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)

def day_curve(comp):
    """Map hour index 1..24 of the release day (UTC) to viewers."""
    start = parse(comp["day_start"])
    by_hour = {}
    for pt in comp["hourly"]:
        t = parse(pt["from"])
        delta = (t - start).total_seconds()
        if 0 <= delta < 24 * 3600:
            by_hour[int(delta // 3600) + 1] = pt["viewers"]
    return by_hour

def main(path):
    case = json.load(open(path))
    print(f"case {case['id']}: {case['game']['name']}, {len(case['comparables'])} comparables")
    for comp in case["comparables"]:
        curve = day_curve(comp)
        p1 = curve.get(1)
        have = len(curve)
        if p1:
            peak_h = max(curve, key=curve.get)
            print(f"{comp['name'][:44]:46s} p1={p1:>8} hours={have:>2} peak h{peak_h}={curve[peak_h]:>8} ratio={curve[peak_h]/p1:5.2f}")
        else:
            print(f"{comp['name'][:44]:46s} p1=  MISSING hours={have:>2}")

if __name__ == "__main__":
    main(sys.argv[1])
