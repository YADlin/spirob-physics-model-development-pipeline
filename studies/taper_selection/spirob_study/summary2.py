"""Part-2 summary table: every held-egg case near success, both measures."""
from __future__ import annotations

import json

from . import designs as D

R = D.STUDY / "results"


def table() -> list[dict]:
    res = json.loads((R / "wrap" / "carry.json").read_text())
    out = []
    for name in [b.name for b in (*D.length_grid(),)] + ["authors_45cm"]:
        for dmm in (15, 25, 35, 45):
            s = [r for r in res if r["design"] == name and round(r["d_mm"]) == dmm]
            if not s:
                continue
            out.append(dict(design=name, d_mm=dmm, cases=len(s),
                            wrap=sum(r["wrap"] for r in s),
                            in_curl=sum(r["in_curl"] for r in s),
                            either=sum(r["wrap"] or r["in_curl"] for r in s),
                            secured=sum(bool(r.get("secured")) for r in s),
                            grasp=sum(r["grasp"] for r in s),
                            best_wrap=max(r["wrap_deg"] for r in s), best_curl=max(r["curl_deg"] for r in s),
                            max_carry_mm=max(r["carry_mm"] for r in s),
                            push_N=[min(r["peak_push_N"] for r in s if r["wrap"] or r["in_curl"]),
                                    max(r["peak_push_N"] for r in s if r["wrap"] or r["in_curl"])]
                            if any(r["wrap"] or r["in_curl"] for r in s) else None,
                            snap_mm=[min(r["snap_mm"] for r in s if r.get("snap_mm") is not None),
                                     max(r["snap_mm"] for r in s if r.get("snap_mm") is not None)]
                            if any(r.get("snap_mm") is not None for r in s) else None))
    (R / "summary_part2.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    for r in table():
        print(r)
