The recovered consolidation exports a smooth two-cable slab instead of preserving
SpiRob segment gaps, and its STEP declares millimetres for numeric metre geometry.
Its GUI, splitter and array paths also have untested failure cases.

This PR includes the three recovered feature commits (`f389d26`, `0f93648`,
`a980a61`) above main `55cc5f1`, then repairs them on a separate review branch.

- Restore canonical sloping segment faces; export fabrication STEP/STL in mm.
- Add explicit finite flexure and optional routed cable channels; validate one
  connected fabrication solid and a closed oriented STL after file re-import.
- Stage complete builds and compile MJCF before replacing outputs; preserve paths
  with spaces and the selected Python interpreter; restore `--nlobe` alias.
- Replicate sensor/contact/equality references, preserve shared world references,
  resolve assets on array relocation, and compile before writing.
- Correct translated-solid splitting, invalid cut handling, fit reporting and
  missing mesh dependencies.
- Correct GUI thread handling, save/build gating and stale exports; add exported
  CAD preview, splitter controls, output path and array controls.
- Document geometry/units, calibrated-physics limits, exact provenance and tests.
- Correct the partial base surface: flat mounting face and nominal joint-facing
  slope, preserving width and backbone endpoints; propagate through CSV/STL/CAD.
- Add a body-frame inspection viewer and audit the authored mesh transforms.
  MuJoCo's principal-axis mesh frames explain the reported axis changes near
  the tip; the actual complete-link surfaces were already aligned.

Final follow-up suite: **210 passed, 1 existing xfailed**. Full 2/3/4-cable
builds compile in MuJoCo. In the follow-up, all 20 non-base simulation meshes
remain byte-identical to the preceding repair in two-/three-cable comparisons.
The base surface, mass/inertia and cable coordinates intentionally change.
Body poses, names and joint stiffness/damping are retained. Compiled vertex
checks confirm mesh placement; drilled two-/three-cable CAD re-imports valid.
The earlier six-copy two-cable array check compiled with 127
bodies, 126 joints and 12 tendons/actuators. STEP/STL splitting and volume checks
pass, including a translated solid and drilled two-cable mesh.

Desktop window execution is still unverified: the environment has Tk but no
display server. GUI command/validation tests are not a substitute for desktop
acceptance. Fabrication geometry needs a physical design/print review; no mass,
stiffness, damping or trained-policy equivalence is claimed. Split parts do not
include assembly joints. Do not merge solely on the passing test count.

See docs/CONSOLIDATION_REPAIR.md and docs/LINK_ORIENTATION_REPAIR.md, including
the before/after base outline and the distinction between Geom and Body frames.
