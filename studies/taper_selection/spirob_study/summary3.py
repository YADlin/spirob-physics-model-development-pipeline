"""Tables for part 3 (results/position/summary_part3.json)."""
from __future__ import annotations

import json
import statistics as stats
from collections import defaultdict

from . import actuation as A

R = A.OUT
MODES = ("force", "position-stiff", "capped-stiff", "admittance-mjlab", "position-mjlab")


def load(group: str) -> list[dict]:
    merged = R / "wrap" / f"{group}.json"
    if merged.exists():
        return json.loads(merged.read_text())
    out = []
    for f in sorted((R / "wrap" / group).glob("*.json")):     # per-body files (while a sweep runs)
        out += json.loads(f.read_text())
    return out


def cell(rs: list[dict]) -> dict:
    ok = [r for r in rs if not r.get("infeasible") and "error" not in r]
    def n(k): return sum(bool(r.get(k)) for r in ok)
    sq = [r.get("sustained_squeeze_N", 0.0) for r in ok]
    touched = [r for r in ok if r.get("sustained_squeeze_N", 0.0) > 0.5]
    snaps = [r["snap_mm"] for r in ok if r.get("snap_mm") is not None]
    return dict(n=len(ok), errors=sum("error" in r for r in rs), infeasible=sum(bool(r.get("infeasible")) for r in rs),
                in_curl=n("in_curl"), wrap=n("wrap"), secured=n("secured"), grasp=n("grasp"),
                crush_risk=n("crush_risk"), dropped=n("dropped"), unstable=n("unstable"),
                carry_max_mm=max([r.get("carry_mm", 0.0) for r in ok] or [0.0]),
                squeeze_median_N=stats.median([r["sustained_squeeze_N"] for r in touched]) if touched else 0.0,
                squeeze_max_N=max(sq or [0.0]),
                peak_T_max_N=max([max(r.get("peak_T_N", [0, 0])) for r in ok] or [0.0]),
                push_median_N=stats.median([r.get("peak_push_N", 0.0) for r in touched]) if touched else 0.0,
                snap_median_mm=stats.median(snaps) if snaps else None,
                fling=sum(1 for s in snaps if s > 300.0),
                best_wrap=max([r.get("wrap_deg", 0.0) for r in ok] or [0.0]))


def table() -> dict:
    out = {}
    for group in ("main", "nofric", "mjlab", "dtcheck", "hybrid2"):
        g = defaultdict(list)
        for r in load(group):
            g[(r["design"], r["mode"], "holder" if r["anchored"] else "pedestal", r["mu"])].append(r)
        out[group] = [dict(design=k[0], mode=k[1], egg=k[2], mu=k[3], **cell(v)) for k, v in sorted(g.items())]
    (R / "summary_part3.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    t = table()
    for grp, rows in t.items():
        for r in rows:
            print(grp, r)
