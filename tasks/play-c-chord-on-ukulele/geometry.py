"""Dimensions of the soprano ukulele and its four strings, the string model (a taut string discretised into nodes),
and the maths shared by make_assets.py, task.py, demo.py and the tests. Pure Python (NumPy).

Frames
- Station: x forward, y to the robot's left, z up (m); the table top is at ``TABLE``.
- U, the ukulele's frame: origin on the neck's centre line, in the fingerboard's top surface, at the nut's
  fingerboard-side face; z up, y from the body toward the headstock (a string's distance from the nut, ``s``, runs
  along -y: ``y_U = -s``), x across the neck from the G side (-x) to the A side (+x). The ukulele lies on its back in
  a cradle with its neck to the robot's right, so U is the station frame turned half a turn about z: ``s`` runs
  toward the robot's left (+y), the G string is farthest from the robot and the A string nearest, and a downstroke
  (G to A) moves toward the robot.

The string model
- Each string is a chain of nodes between the nut (s = 0) and the saddle (s = SCALE). Node i sits at ``S_NODES[i]``
  and can move across the neck (x) and up and down (z). In the sim each node is a link carrying a short capsule of
  the string (its share of the string's length), and each link is joined to the previous one (the first to the
  nut) by two prismatic joints: its displacement *relative to the previous node*. A spring of stiffness T / h on
  each relative joint (T the string's tension, h the span) and a spring of stiffness T / h_last pulling the last node
  back to its line (the span to the saddle; applied by the task after every physics step) make the discretised
  taut-string equations exactly: the restoring force on node i is T (w_i - w_{i-1}) / h_i + T (w_i - w_{i+1}) / h_{i+1}.
- So the load on every node (the force that whatever touches the string puts on it) is read from the string's
  shape: ``loads()``. That is how the scorer sees the fret, the finger and anything muting a string.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from yam_arena import station

    TABLE = float(station.TABLE_TOP)
except Exception:  # pragma: no cover - make_assets.py may run without yam_arena
    TABLE = 0.7475

# -- the instrument (U frame, metres) ----------------------------------------------------------------------------
SCALE = 0.346                   # soprano scale length (13.6 in), nut to saddle
N_FRETS = 15                    # 12 frets to the body, 15 on the fingerboard


def fret_s(n: int) -> float:
    """Distance of fret ``n``'s centre from the nut (equal temperament)."""
    return SCALE * (1.0 - 2.0 ** (-n / 12.0))


FRETS = np.array([fret_s(n) for n in range(1, N_FRETS + 1)])
FRET_R = 0.0010                 # fret wire: a 2.0 mm wide, 1.0 mm tall crown (capsule centred in the board's face)
FB_END = 0.205                  # the fingerboard ends 4.5 mm past the 15th fret
FB_T = 0.005                    # fingerboard (rosewood) thickness
FB_W = (0.035, 0.043)           # fingerboard width at the nut and at its end


def fb_width(s: float) -> float:
    return FB_W[0] + (FB_W[1] - FB_W[0]) * s / FB_END


NECK_DEPTH = 0.015              # neck below the fingerboard (20 mm total at the nut)
NUT = (0.006, 0.0016)           # bone nut: 6 mm along the neck, top 1.6 mm above the fingerboard
HEAD = (-0.105, -0.006, 0.050, 0.014)   # headstock: s from, s to, width, thickness
BODY_JOIN = fret_s(12)          # the neck meets the body at the 12th fret
TOP_Z = -FB_T                   # the soundboard's top face (the fingerboard lies on it over the body)
BODY_DEPTH = 0.060
UPPER_BOUT = (BODY_JOIN + 0.0675, 0.0675)   # (centre s, radius): 135 mm upper bout
LOWER_BOUT = (0.342, 0.081)                 # 162 mm lower bout; the body is 250 mm long
BODY_END = LOWER_BOUT[0] + LOWER_BOUT[1]
SOUNDHOLE = (0.2405, 0.025)                 # 50 mm soundhole, centred in the upper bout
BRIDGE = (0.339, 0.353, 0.058, 0.007)       # rosewood bridge: s from, s to, width, height above the top
SADDLE_W = 0.0025                           # bone saddle, centred on s = SCALE

# String heights: the string's underside above the fret crowns' plane, linear from the nut to the saddle:
# 0.25 mm at the nut, 0.50 mm at the 1st fret, 0.97 mm at the 3rd, 2.5 mm at the 12th, 4.75 mm at the saddle.
ACTION_NUT, ACTION_SADDLE = 0.00025, 0.00475

# -- the strings (fluorocarbon, standard GCEA, re-entrant G) ----------------------------------------------------------
# name, diameter (m), tension (N) at pitch on a 346 mm scale, x at the nut and at the saddle (from the centre line),
# open note, note at the 3rd fret
STRINGS = (
    ("G", 0.00066, 34.0, -0.0135, -0.021, "G4", "A#4"),
    ("C", 0.00084, 31.0, -0.0045, -0.007, "C4", "D#4"),
    ("E", 0.00074, 30.0, 0.0045, 0.007, "E4", "G4"),
    ("A", 0.00061, 36.0, 0.0135, 0.021, "A4", "C5"),
)
NAMES = tuple(st[0] for st in STRINGS)
FRETTED = "A"                   # the C chord: A string at the 3rd fret, G, C, E open
CHORD_FRET = 3
CHORDS = {"C": {"G": 0, "C": 0, "E": 0, "A": 3},      # the fret each string is stopped at (0: open)
          "Am": {"G": 2, "C": 0, "E": 0, "A": 0}}
SEQUENCE = ("C",) * 4 + ("Am",) * 4                  # four downstrokes on C, then four on Am


def pitch_hz(name: str, fret: int) -> float:
    """Equal-tempered pitch of string ``name`` stopped at ``fret`` (A4 = 440 Hz)."""
    open_hz = {"G": 392.00, "C": 261.63, "E": 329.63, "A": 440.00}[name]
    return open_hz * 2.0 ** (fret / 12.0)


def speaking_length(fret: int) -> float:
    """The vibrating length from the stopping point (nut or fret crown) to the saddle (m)."""
    return SCALE - (fret_s(fret) if fret else 0.0)

# Nodes along each string (m from the nut): close together where a finger presses (between frets 1 and 2 for the Am
# chord's G string, between frets 2 and 3 for the C chord's A string), one on every fret crown up to the 5th, and
# over the strumming zone (the end of the fingerboard and the soundhole).
S_NODES = np.array([0.008, fret_s(1), 0.0255, 0.0295, 0.0335, fret_s(2), 0.0405, 0.0440, 0.0470, 0.0505, fret_s(3),
                    0.0632,
                    fret_s(4), fret_s(5), 0.105, 0.130, 0.155, 0.180, 0.200, 0.212, 0.222, 0.232, 0.245, 0.265,
                    0.290, 0.315])
N_NODES = len(S_NODES)
SPANS = np.diff(np.r_[0.0, S_NODES, SCALE])     # h_1 .. h_{N+1}: nut-node1, ..., nodeN-saddle
I_FRET3 = int(np.argmin(np.abs(S_NODES - fret_s(CHORD_FRET))))
I_FRETS = {n: int(np.argmin(np.abs(S_NODES - fret_s(n)))) for n in range(1, 6)}
CAP_END_GAP = 0.0015            # the first and last capsules stop 1.5 mm short of the nut and the saddle
CAP_END_GAP_SADDLE = 0.004


def node_extent(i: int) -> tuple[float, float]:
    """The stretch of string node ``i``'s capsule covers (s from, s to): half way to each neighbour."""
    lo = CAP_END_GAP if i == 0 else 0.5 * (S_NODES[i - 1] + S_NODES[i])
    hi = SCALE - CAP_END_GAP_SADDLE if i == N_NODES - 1 else 0.5 * (S_NODES[i] + S_NODES[i + 1])
    return float(lo), float(hi)


def string_spec(name: str):
    for st in STRINGS:
        if st[0] == name:
            return st
    raise KeyError(name)


def string_x(name: str, s) -> np.ndarray:
    _, _, _, x0, x1, _, _ = string_spec(name)
    return x0 + (x1 - x0) * np.asarray(s, float) / SCALE


def string_z(name: str, s) -> np.ndarray:
    """Height of the string's centre above the fingerboard face (U frame z) at rest."""
    d = string_spec(name)[1]
    return FRET_R + ACTION_NUT + (ACTION_SADDLE - ACTION_NUT) * np.asarray(s, float) / SCALE + d / 2


def node_rest(name: str) -> np.ndarray:
    """Rest positions of a string's nodes, U frame (N x 3)."""
    return np.column_stack([string_x(name, S_NODES), -S_NODES, string_z(name, S_NODES)])


def gap_over_crown(name: str, s: float) -> float:
    """At rest, the string's underside above the fret crowns' plane at ``s``."""
    return float(ACTION_NUT + (ACTION_SADDLE - ACTION_NUT) * s / SCALE)


# -- string mechanics ------------------------------------------------------------------------------------------------
LINK_MASS = 0.001               # kg per link (two links per node); the real string is ~0.3 g: see the fidelity notes
LAST_MASS = 0.040               # kg: the last node (31 mm from the saddle) is heavy, so the spring closing the string
                                # onto the saddle (T / 31 mm, applied explicitly) is well inside the stable range
DAMP_TAU = 0.002                # s: each joint's damping is this times its stiffness (a plucked string swings
                                # through its line and rings on, dying away over about half a second)
TENDON_DAMP = 2.0               # N s/m on the closing spring (the last span's damping)
ARMATURE = 0.01                 # kg on every joint: an added joint-space inertia that keeps the stiff, light chain
                                # stable in the solver (without it the strings drift in their low modes)

JOINT_LIMIT = 0.02              # m: each relative joint's travel


def joint_stiffness(name: str) -> np.ndarray:
    """N/m on each node's two relative joints: T / h_i (h_i the span from the previous node, or the nut)."""
    return string_spec(name)[2] / SPANS[:-1]


def tendon_stiffness(name: str) -> float:
    """N/m of the spring closing the string onto the saddle, on the last node's displacement (the sum of the joints):
    T / h_last, the span to the saddle."""
    return string_spec(name)[2] / SPANS[-1]


def closing_force(name: str, w_last, wdot_last) -> np.ndarray:
    """The last span's pull on the last node (U frame, N): -(T / h_last) w - c w'."""
    return -(tendon_stiffness(name) * np.asarray(w_last, float) + TENDON_DAMP * np.asarray(wdot_last, float))


def stiffness_matrix(name: str) -> np.ndarray:
    """K (N/m), N x N, of the discretised string: the elastic force on the nodes is -K w (same along x and z)."""
    t = string_spec(name)[2]
    k = t / SPANS
    K = np.diag(k[:-1] + k[1:])
    K -= np.diag(k[1:-1], 1) + np.diag(k[1:-1], -1)
    return K


_K = {nm: stiffness_matrix(nm) for nm in NAMES}


def absolute(q) -> np.ndarray:
    """Node displacements from the relative joint positions (cumulative sum from the nut)."""
    return np.cumsum(np.asarray(q, float), axis=0)


def mass_matrix(name: str, axis: int) -> np.ndarray:
    """M (kg), N x N, of the nodes' displacements along ``axis`` (0 across, 1 up): the links' masses (across, both
    links of a node move; up, only the node's) plus each relative joint's armature, a (w_i - w_(i-1))^2 / 2."""
    m = np.full(N_NODES, LINK_MASS * (2 if axis == 0 else 1))
    m[-1] = LAST_MASS + (LINK_MASS if axis == 0 else 0.0)
    D = np.eye(N_NODES) - np.eye(N_NODES, k=-1)
    return np.diag(m) + ARMATURE * D.T @ D


_M = {}


def loads(name: str, w, wdot=None, wddot=None, closing=None) -> np.ndarray:
    """The external load on each node (N; N x 2 for across and up, or N for one axis): the string's equation of
    motion read backwards from its measured shape ``w`` (node displacements from rest, m), their rates ``wdot`` (the
    joints' damping) and accelerations ``wddot`` (inertia). ``closing``: the saddle-end force actually applied on the
    last node this step (default: the one the current state gives). What is left is what touches the string."""
    w = np.asarray(w, float)
    kt = tendon_stiffness(name)
    Kj = _K[name].copy()
    Kj[-1, -1] -= kt
    out = Kj @ w
    wd = np.zeros_like(w) if wdot is None else np.asarray(wdot, float)
    out = out + DAMP_TAU * (Kj @ wd)
    fc = closing_force(name, w[-1], wd[-1]) if closing is None else np.asarray(closing, float)
    out[-1] = out[-1] - fc
    if wddot is not None:
        a = np.asarray(wddot, float)
        if a.ndim == 1:
            out = out + _mass(name, 0) @ a
        else:
            out = out + np.column_stack([_mass(name, k) @ a[:, k] for k in range(a.shape[1])])
    return out


def _mass(name: str, axis: int) -> np.ndarray:
    if (name, axis) not in _M:
        _M[(name, axis)] = mass_matrix(name, axis)
    return _M[(name, axis)]


def solve_shape(name: str, fixed: dict) -> np.ndarray:
    """The string's quasi-static shape (one axis) with nodes ``fixed`` {index: displacement} held and every other
    node free: a piecewise-straight string through the held points."""
    K = _K[name]
    idx = sorted(fixed)
    free = [i for i in range(N_NODES) if i not in fixed]
    w = np.zeros(N_NODES)
    for i in idx:
        w[i] = fixed[i]
    if free:
        Kff = K[np.ix_(free, free)]
        Kfc = K[np.ix_(free, idx)]
        w[free] = np.linalg.solve(Kff, -Kfc @ w[idx])
    return w


def fretted_z(name: str = FRETTED, finger: dict | None = None, fret: int = CHORD_FRET) -> np.ndarray:
    """The z shape of a string held down on fret ``fret``'s crown: the node on the crown sits at crown height,
    ``finger`` {node: displacement} gives the finger's nodes."""
    i = I_FRETS[fret]
    fixed = {i: -gap_over_crown(name, S_NODES[i])}
    fixed.update(finger or {})
    return solve_shape(name, fixed)


# -- rotations and small helpers -------------------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(r, float)
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    q = np.array(q, float)
    return tuple(float(v) for v in q / np.linalg.norm(q))


# -- placement in the station ------------------------------------------------------------------------------------------
X0 = 0.36                       # the neck's centre line
Y_NUT = -0.16                   # the nut (the headstock to the robot's right, toward the right arm)
CRADLE_Z = 0.030                # the body's back rests on the cradle's pads, 30 mm above the table
U0 = np.array([X0, Y_NUT, TABLE + CRADLE_Z + BODY_DEPTH - TOP_Z])   # fingerboard face 95 mm above the table
ROT_U = rot_z(math.pi)          # U's axes in the station frame (half a turn about z)
U_QUAT_XYZW = (0.0, 0.0, 1.0, 0.0)


def u_to_station(p) -> np.ndarray:
    return U0 + ROT_U @ np.asarray(p, float)


def s_to_y(s: float) -> float:
    return float(Y_NUT + s)


# -- instrument parts (U frame) for make_assets.py and the clearance tests ---------------------------------------------
def neck_boxes():
    """(name, centre, size): fingerboard, neck, heel, headstock, nut, bridge, saddle (boxes; the fingerboard as
    tapering slices)."""
    out = []
    n = 8
    for k in range(n):
        s0, s1 = FB_END * k / n, FB_END * (k + 1) / n
        w = fb_width(0.5 * (s0 + s1))
        out.append((f"fingerboard{k}", (0.0, -(s0 + s1) / 2, -FB_T / 2), (w, s1 - s0, FB_T)))
    nk = BODY_JOIN + 0.004
    out.append(("neck", (0.0, -nk / 2, -FB_T - NECK_DEPTH / 2), (0.034, nk, NECK_DEPTH)))
    out.append(("heel", (0.0, -(BODY_JOIN - 0.012), TOP_Z - 0.030), (0.034, 0.024, 0.060)))
    hs0, hs1, hw, ht = HEAD
    out.append(("headstock", (0.0, -(hs0 + hs1) / 2, -ht / 2 - 0.002), (hw, hs1 - hs0, ht)))
    out.append(("nut", (0.0, NUT[0] / 2, (NUT[1] - FB_T) / 2), (0.036, NUT[0], NUT[1] + FB_T)))
    b0, b1, bw, bh = BRIDGE
    out.append(("bridge", (0.0, -(b0 + b1) / 2, TOP_Z + bh / 2), (bw, b1 - b0, bh)))
    sad_top = FRET_R + ACTION_SADDLE - 0.0002          # 0.2 mm under the strings' undersides (they sit in slots)
    out.append(("saddle", (0.0, -SCALE, (TOP_Z + bh + sad_top) / 2), (0.046, SADDLE_W, sad_top - TOP_Z - bh)))
    return out


def body_cylinders():
    """(name, centre, radius, height): the two bouts (z-axis cylinders) forming the body's figure of eight."""
    zc = TOP_Z - BODY_DEPTH / 2
    return [("upper_bout", (0.0, -UPPER_BOUT[0], zc), UPPER_BOUT[1], BODY_DEPTH),
            ("lower_bout", (0.0, -LOWER_BOUT[0], zc), LOWER_BOUT[1], BODY_DEPTH)]


def fret_capsules():
    """(name, centre, half-length, radius): each fret wire, across the fingerboard (x)."""
    return [(f"fret{n}", (0.0, -fret_s(n), 0.0), fb_width(fret_s(n)) / 2 - FRET_R, FRET_R)
            for n in range(1, N_FRETS + 1)]


def neck_rest_box():
    """The neck rest: a padded block under the headstock, from the table to the headstock's underside."""
    zt = -(U0[2] - TABLE)
    under = -0.002 - HEAD[3]
    return ("neck_rest", (0.0, 0.040, (zt + under) / 2), (0.06, 0.04, under - zt))


def cradle_parts():
    zt = -(U0[2] - TABLE)
    return [("cradle", (0.0, -LOWER_BOUT[0], zt + CRADLE_Z / 2), (0.12, 0.10, CRADLE_Z)),
            ("cradle_front", (0.0, -UPPER_BOUT[0], zt + CRADLE_Z / 2), (0.08, 0.05, CRADLE_Z)),
            neck_rest_box()]


# -- the robot's fingertip (for planning and clearance checks) ---------------------------------------------------------
# Each YAM fingertip ends in a nub 4 mm thick (along the closing axis), 15.4 mm wide and 13 mm long, its end rounded.
# Planning models the nub's end as a capsule along the TCP's y: centre (+-NUB_X, 0, NUB_Z) in the TCP frame (the
# open gripper; + for the right-hand tip), half length NUB_HALF, radius NUB_R.
NUB_X = 0.049
NUB_Z = 0.0128 - 0.0020
NUB_HALF = 0.0057
NUB_R = 0.0020
OPEN_TIP_X = (0.047, 0.0509)    # the open gripper's tips span this far either side of the TCP (along its x)


def nub_capsule(tcp, rot, side: int):
    """(end a, end b, radius) of the nub end of the tip on ``side`` (+1: the tip at the TCP's +x), station frame."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    c = tcp + rot @ np.array([side * NUB_X, 0.0, NUB_Z])
    a = rot @ np.array([0.0, NUB_HALF, 0.0])
    return c - a, c + a, NUB_R


def seg_seg_dist(p1, q1, p2, q2) -> float:
    """Shortest distance between segments p1-q1 and p2-q2."""
    p1, q1, p2, q2 = (np.asarray(v, float) for v in (p1, q1, p2, q2))
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-15 and e < 1e-15:
        return float(np.linalg.norm(r))
    if a < 1e-15:
        s, t = 0.0, float(np.clip(f / e, 0, 1))
    else:
        c = d1 @ r
        if e < 1e-15:
            s, t = float(np.clip(-c / a, 0, 1)), 0.0
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = float(np.clip((b * f - c * e) / den, 0, 1)) if den > 1e-15 else 0.0
            t = (b * s + f) / e
            if t < 0:
                s, t = float(np.clip(-c / a, 0, 1)), 0.0
            elif t > 1:
                s, t = float(np.clip((b - c) / a, 0, 1)), 1.0
    return float(np.linalg.norm(p1 + d1 * s - (p2 + d2 * t)))


def node_segment(name: str, i: int, w=(0.0, 0.0)):
    """Node ``i``'s capsule axis (two ends), station frame, displaced by ``w`` (x, z)."""
    lo, hi = node_extent(i)
    ends = []
    for s in (lo, hi):
        ends.append(u_to_station([string_x(name, s) + w[0], -s, string_z(name, s) + w[1]]))
    return ends[0], ends[1]
