> Engineering history. For current commands, parameters and supported workflows, use the [main README](../../README.md).

# Understanding the test summary

`passed`, `skipped` and `xfailed` have different meanings. A skipped test was
not run. An expected failure was run and failed a documented expectation.
Neither should be counted as a passing physical validation.

## The 19 skips in the previous delivered suite

| Count | Test group | Reason |
|---:|---|---|
| 16 | `test_collision_workflows.py` face/corner contact probes | The fixture also supplies a three-cable model. These particular assertions require the legacy two-cable box/cylinder compound, so the three-cable cases are intentionally skipped. |
| 3 | `test_convex_collision.py` flat-patch benchmark | The benchmark compares the old constant-thickness rectangular compound against the same rectangular CAD with a hull. The other three section/profile combinations are intentionally excluded from this specific comparison. |

These 19 skips do not indicate missing packages in the validated environment.
Other tests exercise the excluded section/profile combinations. The new
n-cable suite independently tests 3, 4 and 6 cables; it does not rely on the
skipped two-cable compound assertions.

There are also dependency-based skips elsewhere in the project. If your local
count/reasons differ, inspect them: missing CADQuery, MuJoCo or trimesh can
prevent relevant checks. Install the supplied development requirements first.

## The one expected failure: F-08

Test: `tests/test_conventions.py::test_tendon_offset_from_surface_is_constant`.
It is marked `xfail(strict=True)` to track an explicitly deferred routing
decision. The implementation deliberately moves the distal cable site inward:

```text
r1 = max(r_original - inward_shift, 1e-6)
r2 = max(r_original - inward_shift - dz*tan(phi/2), 1e-6)
```

The test expects equal endpoint radii under a constant-offset routing rule;
the existing implementation does not satisfy that expectation. With the
current root parameters, the cable-0 radius reduction within a link ranges
from approximately **0.1601 to 0.6478 mm**. The new two-cable thickness taper
along Y does not eliminate this radial cable-placement rule in the XZ plane.

Despite its name, this test compares the nominal cable-0 endpoint radii. It
does not measure signed clearance from the entire 3D CAD surface. Choosing
the physically correct routing therefore requires comparing the intended
cable guides/holes and actual hardware, not just making this assertion pass.

**Practical assessment:** this is not a newly introduced collision failure
or a build blocker. It is a known modelling discrepancy worth addressing
before quantitative hardware matching, stiffness identification or cable-force
validation, because tendon routing determines cable lengths and moment arms.
It is not evidence that the current routing is physically accurate. It also
does not establish the cause of the earlier arena-memory crashes.

This update preserves existing routing and marks no new expected failures.
Changing it would change dynamics and would need its own validation. The
strict marker causes an unexpected pass to fail the suite until the marker
is deliberately removed, so it cannot quietly become stale after a fix.

## Inspect your own results

```bash
uv pip install -r requirements-dev.txt
uv run python -m pytest -q -rxXs
```

The final lines identify every skip and expected failure. Ordinary failures
remain failures and should be investigated. The validated run's full output
is included in the delivery package under `verification/`.
