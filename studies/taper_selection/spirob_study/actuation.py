"""Cable actuation modes for part 3: force, length (position) and hybrids.

Why this module exists
----------------------
Parts 1 and 2 drove the cables with *force* (MuJoCo ``motor``): the controller
chooses the tension and the cable length follows. The RL task in spirob-mjlab
and the hardware drive the cables with *length* (MuJoCo ``position``): the
controller chooses how much cable is reeled in and the tension follows. This
module gives every experiment the same four ways to drive a cable:

``force``      tension commanded directly (parts 1 and 2).
``position``   cable-length servo, F = kp (L_cmd - L) - kv dL/dt, pull only,
               limited at F_MAX = 1000 N (the spirob-mjlab limit). With a large
               kp this is an inextensible cable on a position-controlled spool.
``capped``     hybrid 1: the same length servo with its pull limited to the
               grip force F_cap. In free space it is a length servo (shape is
               set by the commanded length); in contact it saturates and
               becomes a force source (grip force is set by F_cap). MuJoCo and
               mjlab both support it natively (``forcerange``); on hardware it
               is a motor current limit.
``admittance`` hybrid 2: a length servo with an outer force loop that runs at
               the 100 Hz control rate of spirob-mjlab. When the measured
               tension exceeds F_ref, the commanded length is paid out until
               it falls back to F_ref:
                   e <- max(0, e + dt_c (T - F_ref) / b),   L_cmd = L_prog + e
               with b = kp * tau_a. Needs only a tension (motor current or
               load cell) measurement, which a position-controlled hardware
               spool can provide.

Programmes are written once, as cable *tensions* (the force programmes of
part 2), and translated for the length modes through each body's free-curl
curve L(F): the length a cable reaches when pulled with tension F in free
space (no contact, no friction). Without contact and friction the two
translations give the same shapes, so every difference in the results comes
from contact, friction or servo dynamics, not from a different programme.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import mujoco
import numpy as np

from . import model as M, sim
from .designs import STUDY

OUT = STUDY / "results" / "position"
F_MAX = 1000.0            # N, spirob-mjlab forcerange
KP_MJLAB, KV_MJLAB = 2500.0, 20.0
KP_STIFF = 1.0e5                     # N/m, "inextensible": < 0.5 mm stretch at the grip force of every body
RATE = 0.1                           # m/s, spirob-mjlab command rate limit (1 mm per 10 ms step)
STRETCH_FRAC = 0.01                  # "inextensible": stretch <= 1 % of the stroke at the grip force
CONTROL_DT = 0.01                    # s, spirob-mjlab control step
SLACK_FRAC = 0.02                    # passive cable is paid out 2 % of the stroke beyond its free length
MODES = ("force", "position", "capped", "admittance")


@dataclass
class Stroke:
    """Free-curl curve of one body (cable_1 pulled, cable_0 passive)."""
    design: str
    f_pack: float            # N, analytic tension at which the base joint locks
    rest: float              # m, cable length of the straight body
    full: float              # m, pulled-cable length at the grip force 1.25 f_pack
    far_full: float          # m, passive-cable length at that point
    F: list                  # N, tension samples (increasing)
    L_near: list             # m, pulled-cable length at each tension
    L_far: list              # m, passive-cable length at each tension
    lock_L: list             # m, pulled-cable length at which each joint reaches its stop
    lock_F: list             # N, tension at that moment

    @property
    def stroke(self) -> float:
        return self.rest - self.full

    @property
    def f_grip(self) -> float:
        return 1.25 * self.f_pack

    def L_pull(self, F: float) -> float:
        """Length that tension F produces in free space (saturates at full)."""
        if F <= 0.0:
            return self.rest
        return float(np.interp(min(F, self.f_grip), self.F, self.L_near))

    def L_passive(self, L_near: float) -> float:
        """Free length of the other cable when this one is at L_near."""
        x = np.asarray(self.L_near)[::-1]
        return float(np.interp(L_near, x, np.asarray(self.L_far)[::-1]))

    def kp_inextensible(self) -> float:
        return max(KP_MJLAB, self.f_grip / (STRETCH_FRAC * self.stroke))

    def ctrlrange(self) -> tuple[float, float]:
        """Commandable cable length: from fully curled toward this cable to
        fully paid out when curled the other way (plus the slack margin)."""
        return (self.full, self.far_full + SLACK_FRAC * self.stroke)


def measure_stroke(design, t_ramp: float = 8.0) -> Stroke:
    sm = M.make(design, tag=f"stroke_{os.getpid()}")
    fp = sim.f_pack_estimate(sm)
    m, d = sm.load()
    mujoco.mj_forward(m, d)
    rest = float(d.ten_length[1])
    log = sim.ramp_single_cable(sm, 1, 1.5 * fp, t_ramp=t_ramp, t_hold=0.3)
    F, ten, q = log["F"], log["ten"], log["q"]
    sel = F <= 1.25 * fp + 1e-9
    Fs, Ln, Lf = F[sel], np.minimum.accumulate(ten[sel, 1]), np.maximum.accumulate(ten[sel, 0])
    dth = math.radians(design.dtheta_deg)
    lock_L, lock_F = [], []
    for j in range(q.shape[1]):
        hit = np.nonzero(np.abs(q[:, j]) >= 0.95 * dth)[0]
        k = hit[0] if len(hit) else len(F) - 1
        lock_L.append(float(ten[k, 1])); lock_F.append(float(F[k]))
    # thin the samples (monotone, ~200 points)
    idx = np.unique(np.linspace(0, len(Fs) - 1, 200).astype(int))
    return Stroke(design.name, fp, rest, float(Ln[-1]), float(Lf[-1]), Fs[idx].tolist(),
                  Ln[idx].tolist(), Lf[idx].tolist(), lock_L, lock_F)


_CACHE: dict[str, Stroke] = {}


def stroke(design) -> Stroke:
    if design.name in _CACHE:
        return _CACHE[design.name]
    path = OUT / "strokes" / f"{design.name}.json"
    if path.exists():
        s = Stroke(**json.loads(path.read_text()))
    else:
        s = measure_stroke(design)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(s)))
    _CACHE[design.name] = s
    return s


def kv_for(kp: float) -> float:
    """Servo damping with the same damping ratio as spirob-mjlab's (kv ~ sqrt(kp))."""
    return KV_MJLAB * math.sqrt(kp / KP_MJLAB)


# ------------------------------------------------------------------ controller
@dataclass
class Gains:
    mode: str = "force"
    kp: float | None = None          # None: KP_STIFF
    f_cap: float | None = None       # capped / admittance: grip force (None: 1.25 f_pack)
    tau_a: float = CONTROL_DT        # admittance time constant (one control step: deadbeat)
    per_cable: bool = False          # hybrids v2: each cable gets its own (length, force limit) command
    margin: float = 0.2              # v2: force limit = programme tension + margin * f_pack
    freeze_release: bool = False     # v2: hold both cable lengths while the holder lets go

    def __post_init__(self):
        if self.mode == "admittance" and self.kp is None:
            self.kp = KP_MJLAB           # a software force loop needs a compliant servo (see part-3 notes)

    def label(self) -> str:
        if self.mode == "force":
            return "force"
        k = "mjlab" if self.kp == KP_MJLAB else "stiff"
        v = "2" if self.per_cable else ""
        fz = "-freeze" if self.freeze_release else ""
        return f"{self.mode}{v}-{k}{fz}"


def build(design, gains: Gains, *, objects=None, tag="act", timestep: float = 2e-4) -> tuple[M.StudyModel, Stroke, float]:
    """Model with the actuators that ``gains`` asks for. Returns (model, stroke, kp)."""
    st = stroke(design)
    if gains.mode == "force":
        return M.make(design, f_max=1e3, tag=tag, objects=objects, timestep=timestep), st, 0.0
    kp = gains.kp or KP_STIFF
    f_lim = F_MAX if gains.mode in ("position", "admittance") else (gains.f_cap or st.f_grip)
    lo, hi = st.ctrlrange()
    sm = M.make(design, f_max=f_lim, tag=tag, objects=objects, actuation="position",
                kp=kp, kv=kv_for(kp), ctrlrange=(lo, hi), timestep=timestep)
    return sm, st, kp


class Driver:
    """Turns a tension programme (T0, T1) into actuator commands for one mode.

    Call ``step(d, tension)`` once per physics step, before mj_step. It writes
    d.ctrl (and, for the force mode, returns the tensions that the friction
    model should apply). ``log`` keeps commanded and actual cable length and
    the tension at the base for each cable.
    """

    def __init__(self, m, st: Stroke, gains: Gains, kp: float):
        self.m, self.st, self.g, self.kp = m, st, gains, kp
        self.f_ref = gains.f_cap or st.f_grip
        self.e = np.zeros(2)
        self.cmd = np.array([st.rest, st.rest])
        self.base = self.cmd.copy()
        self.next_ctrl = 0.0
        self.n_ctrl = max(1, int(round(CONTROL_DT / m.opt.timestep)))
        self.k = 0
        self.tension = np.zeros(2)
        self.T_meas = np.zeros(2)                   # what a 100 Hz load cell reads
        self.override = None                        # frozen cable lengths (v2, release)
        self._acc, self._nacc = np.zeros(2), 0

    def prime(self, d) -> None:
        """Start with the cables at rest length (no t = 0 transient)."""
        if self.g.mode != "force":
            d.ctrl[:2] = self.cmd
            mujoco.mj_forward(self.m, d)

    def length_targets(self, T) -> np.ndarray:
        """Free-space lengths equivalent to the tension pair T (mu = 0)."""
        # Without friction the two cables have equal and opposite moment arms,
        # so the free shape depends only on the net tension T1 - T0; the
        # pulled cable goes to L(net) and the other is paid out to its free
        # length plus a small slack (co-contraction has no length equivalent).
        st = self.st
        net = T[1] - T[0]
        pulled = 1 if net >= 0 else 0
        out = np.zeros(2)
        out[pulled] = st.L_pull(abs(net))
        out[1 - pulled] = st.L_passive(out[pulled]) + SLACK_FRAC * st.stroke
        return out

    def length_targets_per_cable(self, T) -> np.ndarray:
        """v2: each pulled cable aims for the length its own tension would give
        in free space; a cable with no tension is paid out (slack). Two pulled
        cables then usually cannot both reach their targets, and the force
        limits decide the tensions, as in force control (co-contraction)."""
        st = self.st
        out = np.zeros(2)
        on = T > 1e-9
        if on.all():
            out[:] = [st.L_pull(T[0]), st.L_pull(T[1])]
        elif on.any():
            c = int(np.argmax(on))
            out[c] = st.L_pull(T[c])
            out[1 - c] = st.L_passive(out[c]) + SLACK_FRAC * st.stroke
        else:
            out[:] = st.rest + SLACK_FRAC * st.stroke
        return out

    def freeze(self, d) -> None:
        self.override = np.array([d.ten_length[0], d.ten_length[1]])

    def unfreeze(self) -> None:
        self.override = None

    def step(self, d, T) -> np.ndarray:
        T = np.asarray(T, float)
        mode = self.g.mode
        if mode == "force":
            d.ctrl[:] = 0.0
            self.tension[:] = T
            self.k += 1
            return T
        self._acc += np.maximum(-d.actuator_force[:2], 0.0)
        self._nacc += 1
        if self.k % self.n_ctrl == 0:               # 100 Hz command update
            if self._nacc:
                self.T_meas = self._acc / self._nacc    # tension averaged over the control step
            self._acc[:] = 0.0; self._nacc = 0
            v2 = self.g.per_cable
            tgt = self.length_targets_per_cable(T) if v2 else self.length_targets(T)
            lo, hi = self.st.ctrlrange()
            step = RATE * CONTROL_DT                # spirob-mjlab rate limit on the programme
            if self.override is not None:
                self.base = np.clip(self.override, lo, hi)
            else:
                self.base = np.clip(self.base + np.clip(np.clip(tgt, lo, hi) - self.base, -step, step), lo, hi)
            f_ref = (T + self.g.margin * self.st.f_pack) if v2 else np.full(2, self.f_ref)
            if self.override is not None and v2:
                f_ref = np.full(2, self.st.f_grip + self.g.margin * self.st.f_pack)
            if mode == "capped" and v2:
                self.m.actuator_forcerange[:2, 0] = -f_ref      # per-cable force limit
            if mode == "admittance":
                # force guard below the rate limit: pays cable out as soon as
                # the measured tension exceeds F_ref, takes it back when it falls
                b = self.kp * self.g.tau_a
                self.e = np.maximum(0.0, self.e + CONTROL_DT * (self.T_meas - f_ref) / b)
            self.cmd = np.clip(self.base + self.e, lo, hi)
        d.ctrl[:2] = self.cmd
        self.tension[:] = self.T_meas
        self.k += 1
        return self.tension

    def snapshot(self, d) -> tuple:
        """(commanded L0, L1, actual L0, L1, base tension T0, T1)."""
        ten = self.tension if self.g.mode != "force" else self.tension
        if self.g.mode == "force":
            return (np.nan, np.nan, float(d.ten_length[0]), float(d.ten_length[1]), float(ten[0]), float(ten[1]))
        return (float(self.cmd[0]), float(self.cmd[1]), float(d.ten_length[0]), float(d.ten_length[1]),
                float(self.T_meas[0]), float(self.T_meas[1]))
