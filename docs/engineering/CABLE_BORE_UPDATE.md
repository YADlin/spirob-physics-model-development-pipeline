# Straight fabrication bores and trimmed IGES export

The prior fabrication exporter swept a circle around every MJCF routing point
with round transitions. Exact-length designs include a partial base, and its
routing anchor does not lie on the continuation of the complete-link route.
Round transitions introduce small revolution surfaces near that kink. Cutter
extensions were one entire robot length, leaving remote analytic surface origins.

The manufacturing route now uses the first and last routing anchors to define a
single straight axis, extrapolated to the base and tip planes. Each bore is cut
with one cylinder, extending only a diameter-dependent clearance past the ends.
The MJCF route is not silently changed: changing it affects tendon lengths,
moment arms and calibration. CAD reports record the difference in millimetres.

Validation now checks a full-length 99.9%-diameter probe in STEP and IGES, and
33 centre/ring rays through each STL bore up to 95% of its diameter. STL tolerance
is capped at 0.5% of the bore diameter. Disconnected STL components are rejected.
IGES uses mode 1 (trimmed BRep) and validates face count, solid count and all six
bounding coordinates on reimport, not just the three bounding-box extents.

The website displays the same axes, true elliptical XY bore intersections, and
3D face openings. The sampled 3D display omits internal bore-wall shading and is
not a substitute for the checked fabrication exports. Brown marks are bores;
blue marks are the independent simulation routes. Hole visibility does not
depend on the simulation-route checkbox. End view is oriented consistently with
the XY cross-section; coincident base/tip labels are suppressed.

## Uploaded-file diagnosis

The inspected STEP was one valid solid; its STL was one closed connected volume.
The IGES contained 298 trimmed faces and no face outside the same model bounds.
There were no detached spherical solids. Six small revolution transition patches
occurred around Z = 9.99 and 18.27 mm. Two base-hole cylinder surface origins were
at Z = -227.109 mm, despite their trimmed faces being within the robot.
Displaying untrimmed support geometry can therefore explain the reported remote
cylinders, but this depends on the importing CAD application.

Sampled centre lines of the base-region cylindrical segments were open in the
supplied STEP. The physical print blockage is not conclusively diagnosed from
these files: it must not be reported as a proven pair of solid spheres. The
uploaded params.json and slicer project were not supplied. Regenerate from the
original JSON to reproduce the exact design with the new bore construction.
