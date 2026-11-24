"""Taper-selection study for the two-cable SpiRob (see ../README.md)."""
import os

# Off-screen MuJoCo rendering (software EGL) for the animations; must be set
# before mujoco is imported. Harmless when nothing is rendered.
for _k, _v in (("MUJOCO_GL", "egl"), ("PYOPENGL_PLATFORM", "egl"),
               ("EGL_PLATFORM", "surfaceless"), ("LIBGL_ALWAYS_SOFTWARE", "1")):
    os.environ.setdefault(_k, _v)
