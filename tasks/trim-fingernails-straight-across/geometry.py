"""Dimensions of the resident's hand and the lever nail clipper, the clipper's mechanism (lever, cam heel, sprung jaw),
the nail-cutting model, and the box maths shared by make_assets.py, task.py, demo.py and the tests. Pure Python (NumPy).

Frames
- Station: x forward, y to the robot's left, z up (m); the table top is at ``TABLE``.
- Hand: the hand articulation's origin (``HAND0``), axes parallel to the station's.
- F, a finger's frame (its link): origin on the knuckle (MCP joint) axis, x along the finger toward the tip, y across it
  (toward the thumb), z up through the nail. The knuckle joint turns F about its y axis; + is flexion (tip down).
- N, a nail's frame: origin on the nail's mid-plane, centred across it, on the line where the nail leaves the skin
  under it (the hyponychium, the "quick"). Axes as F: u (x) toward the free edge's tip, v (y) across, w (z) up. The
  white free edge is the nail beyond u = 0; its length per bin is the "white edge" the task scores.
- C, the clipper's frame (its body link): origin on the underside of the lower arm at its rear end, centred across it;
  x along the clipper toward the jaws, y across, z up through the lever.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from yam_arena import station

    TABLE = float(station.TABLE_TOP)
except Exception:  # pragma: no cover - make_assets.py may run without yam_arena
    TABLE = 0.7475

MM = 1e-3


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
    q = np.array(q)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return tuple(float(v) for v in q)


def quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = quat_wxyz(r)
    return (x, y, z, w)


def rot_from_xyzw(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- the hand (hand frame = station axes, origin HAND0) ---------------------------------------------------------------
# A full-size adult right hand lies palm down on a foam hand rest, reaching in from the robot's far left with the
# fingers spread and pointing toward the robot's right arm. Its index and middle fingers have long nails (3-4 mm of
# white free edge); the ring and little fingers and the thumb have been done already.
Z_F = TABLE + 0.108                     # the fingers' axes: 10.8 cm above the table
HAND0 = np.array([0.325, 0.066, Z_F])   # hand origin: the middle finger's knuckle (MCP joint) centre

MIDDLE_DEG = 210.0                      # direction the middle finger points (deg from +x toward +y)
SPREAD_DEG = 20.0                       # fingers spread 20 degrees apart


def _dir(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([math.cos(a), math.sin(a), 0.0])


_XM = _dir(MIDDLE_DEG)                  # middle finger: along it, toward the tip
_YM = np.array([-_XM[1], _XM[0], 0.0])  # across it, toward the thumb

# segment lengths knuckle -> PIP, PIP -> DIP, DIP -> hyponychium (m); radii of the three segments (m); nail width (m);
# free edge at the start: length at the nail's centre and at its corners (m)
FINGERS = {
    "index": dict(mcp=HAND0 + 0.022 * _YM - 0.002 * _XM, deg=MIDDLE_DEG + SPREAD_DEG,
                  seg=(0.044, 0.025, 0.019), r=(0.0090, 0.0080, 0.0068), W=0.0105, L0c=3.7 * MM, L0e=2.9 * MM),
    "middle": dict(mcp=HAND0.copy(), deg=MIDDLE_DEG,
                   seg=(0.048, 0.029, 0.020), r=(0.0092, 0.0082, 0.0070), W=0.0115, L0c=4.1 * MM, L0e=3.2 * MM),
}
NAILS = ("index", "middle")
# fingers already done (rigid in the palm link): (name, knuckle, direction deg, length to tip, radius)
STATIC_FINGERS = (
    ("ring", HAND0 - 0.020 * _YM - 0.002 * _XM, MIDDLE_DEG - SPREAD_DEG, 0.090, 0.0085),
    ("little", HAND0 - 0.038 * _YM - 0.012 * _XM, MIDDLE_DEG - 2 * SPREAD_DEG, 0.072, 0.0075),
)
THUMB = (HAND0 + 0.030 * _YM - 0.045 * _XM + np.array([0.0, 0.0, -0.006]), MIDDLE_DEG + 55.0, 0.062, 0.0095)
PALM = dict(back=0.088, width=0.086, thick=0.028, centre_y=-0.0085)   # behind the knuckles, along _XM / _YM
FOREARM = dict(length=0.13, radius=0.024, z=0.008)   # its underside 0.5 mm above the rest

N_BINS = 8                              # each nail's free edge as 8 strips across its width
NAIL_T = 0.5 * MM                       # nail plate thickness
BIN_T = 0.45 * MM                       # the free-edge strips (a hair thinner, so a cut one hides inside the plate)
NAIL_ATTACHED = 0.012                   # the nail plate over the nail bed, behind the hyponychium

FINGER_K = 0.40                         # N m/rad: a relaxed knuckle (0.7 N at the nail turns the finger 10 degrees)
FINGER_D = 0.004                        # N m s/rad
FINGER_LIMITS = (-20.0, 45.0)           # deg (extension, flexion)
FINGER_MASS = 0.030
BIN_MASS = 0.002
BIN_K = 1.0e5                           # N/m: the strips' slide drive (positions them after a cut)
BIN_D = 50.0


def finger_rot(name: str) -> np.ndarray:
    """F's axes at rest (station frame): columns x (to the tip), y (to the thumb), z (up)."""
    return rot_z(math.radians(FINGERS[name]["deg"]))


def finger_origin(name: str) -> np.ndarray:
    return np.asarray(FINGERS[name]["mcp"], float)


def hyp_x(name: str) -> float:
    """Distance from the knuckle to the hyponychium along the finger (m)."""
    return float(sum(FINGERS[name]["seg"]))


def nail_in_f(name: str) -> np.ndarray:
    """N's origin in F."""
    f = FINGERS[name]
    return np.array([hyp_x(name), 0.0, f["r"][2] + NAIL_T / 2])


def bin_v(name: str) -> np.ndarray:
    """Bin centres across the nail (v, m)."""
    w = FINGERS[name]["W"]
    return -w / 2 + (np.arange(N_BINS) + 0.5) * w / N_BINS


def bin_width(name: str) -> float:
    return FINGERS[name]["W"] / N_BINS


def free_edge0(name: str) -> np.ndarray:
    """White edge per bin at the start (m): a rounded tip, longest in the middle."""
    f = FINGERS[name]
    t = 2 * bin_v(name) / f["W"]
    return f["L0e"] + (f["L0c"] - f["L0e"]) * (1 - t ** 2)


def nail_pose(name: str, f_pos, f_rot):
    """(origin, rotation) of N in the station frame for the finger link at (f_pos, f_rot)."""
    f_rot = np.asarray(f_rot, float)
    return np.asarray(f_pos, float) + f_rot @ nail_in_f(name), f_rot


def finger_pose_at(name: str, flex: float):
    """F's pose (station) with the knuckle flexed by ``flex`` rad."""
    r = finger_rot(name) @ rot_y(flex)
    return finger_origin(name), r


def pinch_point(name: str) -> np.ndarray:
    """Middle of the finger's middle segment (station, at rest): where the other hand holds it."""
    f = FINGERS[name]
    return finger_origin(name) + finger_rot(name) @ np.array([f["seg"][0] + 0.65 * f["seg"][1], 0.0, 0.0])


def finger_capsules(name: str):
    """The finger's segments as capsules in F: (name, start x, end x, radius)."""
    f = FINGERS[name]
    a, b, c = f["seg"]
    rp, rm, rd = f["r"]
    return [("proximal", -0.004, a, rp), ("middle", a, a + b, rm), ("distal", a + b, hyp_x(name) + 0.001 - rd, rd)]


def finger_boxes_f(name: str):
    """Nail bed and attached nail plate as boxes in F: (name, centre, size)."""
    f = FINGERS[name]
    h, rd, w = hyp_x(name), f["r"][2], f["W"]
    return [("bed", np.array([h - NAIL_ATTACHED / 2, 0.0, rd / 2]), np.array([NAIL_ATTACHED, w, rd])),
            ("plate", np.array([h - NAIL_ATTACHED / 2, 0.0, rd + NAIL_T / 2]), np.array([NAIL_ATTACHED, w, NAIL_T]))]


def tissue_dist(name: str, p_f) -> np.ndarray:
    """Signed depth of points (F frame) inside the finger's soft tissue near the tip (the distal segment's capsule and
    the nail bed under the attached plate): > 0 inside."""
    p = np.atleast_2d(p_f)
    _, x0, x1, r = finger_capsules(name)[2]
    t = np.clip(p[:, 0], x0, x1)
    d_cap = r - np.linalg.norm(p - np.c_[t, np.zeros(len(p)), np.zeros(len(p))], axis=1)
    c, s = finger_boxes_f(name)[0][1:]
    d_box = np.min(s / 2 - np.abs(p - c), axis=1)
    return np.maximum(d_cap, d_box)


# -- the hand rest (fixed shape) --------------------------------------------------------------------------------------
REST_FRONT = 0.006                      # its front edge 6 mm behind the knuckles: the fingers reach out past it
REST_SIZE = (0.20, 0.13, Z_F - 0.0165 - TABLE)  # along the hand, across, height (top 0.5 mm under the palm)


def rest_pose():
    c = HAND0 - (REST_FRONT + REST_SIZE[0] / 2) * _XM + PALM["centre_y"] * _YM
    c = np.array([c[0], c[1], TABLE + REST_SIZE[2] / 2])
    return c, rot_z(math.radians(MIDDLE_DEG))


# A rolled towel lies across the hand rest's front, under the fingers' middle segments (the tips stay free): the hand
# holding a finger down presses it onto the towel, so it can't flex down further or lift.
TOWEL_S = (0.040, 0.072)                # along the middle finger from its knuckle (m)
TOWEL_W = 0.100                         # across the hand (m)
TOWEL_TOP = Z_F - 0.0087                # 0.5 mm under the middle finger's middle segment


def towel_pose():
    """(centre, rotation, size) of the towel roll (a fixed box): from the table up to TOWEL_TOP."""
    s0, s1 = TOWEL_S
    c = HAND0 + (s0 + s1) / 2 * _XM + 0.016 * _YM
    h = TOWEL_TOP - TABLE
    return np.array([c[0], c[1], TABLE + h / 2]), rot_z(math.radians(MIDDLE_DEG)), np.array([s1 - s0, TOWEL_W, h])


def held_flex(name: str) -> float:
    """Knuckle angle (rad, + flexes) with the finger pressed down onto the towel at its middle segment."""
    p = pinch_point(name)
    r = FINGERS[name]["r"][1]
    s = float(np.linalg.norm((p - finger_origin(name))[:2]))
    return math.asin((Z_F - r - TOWEL_TOP) / s)


# -- the clipper (C frame) --------------------------------------------------------------------------------------------
# A standard lever fingernail clipper (58 mm, 14 mm wide, 25 g): two spring-steel arms riveted at the rear, curved
# jaws at the front, a pin through both arms near the front and a lever on the pin's head, already swung round over
# the arms into cutting position. Pressing the lever's end down rocks it on the pin's head; a heel under the lever,
# behind the pin, pushes the upper arm down and the jaws close. The upper arm is sprung open.
ARM_W = 0.014                           # arms' width
LOW_T = 0.0015                          # lower arm (body) thickness
E_X = 0.0580                            # cutting edges' centre (C x): the front of the clipper
JAW_HW = 0.0065                         # half width of the cutting edges (13 mm)
SAG = 0.0006                            # the edges are curved: their ends 0.6 mm further forward than the middle
JAW_X0 = 0.0490                         # the jaws' blocks from here forward
Z_LE = 0.0020                           # the lower edge's height (top of the lower jaw)
X_H, Z_H = 0.0045, 0.0060               # rivet: the upper arm's hinge axis (along C y), at the spacer's top
UP_T = 0.0015                           # upper arm thickness: z 6.0-7.5 mm at rest; its edge is its front underside
GAP0 = Z_H - Z_LE                       # 4.0 mm: the jaws' opening at rest
P_X = 0.0440                            # the pin (14 mm behind the edges); the lever rocks on its head
LEVER_L, LEVER_T, LEVER_W = 0.036, 0.0015, 0.011
PHI0 = math.radians(20.0)               # lever angle at rest (its end raised)
HEEL = np.array([-0.0080, 0.0, -0.0045])   # heel centre in the lever's own frame (unrotated): 8 mm behind the pin
HEEL_R = 0.0012
LEVER_END = np.array([-LEVER_L, 0.0, LEVER_T / 2])   # the lever end's top corner, lever frame (unrotated)

BODY_MASS, UPPER_MASS, LEVER_MASS = 0.012, 0.007, 0.006
CLIPPER_MASS = BODY_MASS + UPPER_MASS + LEVER_MASS
JAW_TAU0 = 0.12                         # N m: the upper arm's spring preload at rest (against its stop)
JAW_K = 11.0                            # N m/rad
LEVER_D = 0.0005


def _heel_z_offset(beta: float) -> float:
    return float((rot_y(beta) @ HEEL)[2])


# The lever's frame turns about +y by beta from C (so its end, at -x, rises with beta). Its pivot height is set so
# that at rest the heel rests on the upper arm.
Z_P = Z_H + UP_T + HEEL_R - _heel_z_offset(PHI0)


def lever_rot(beta: float) -> np.ndarray:
    """Lever frame -> C at lever angle ``beta`` (rad, end up)."""
    return rot_y(beta)


def heel_centre(beta: float) -> np.ndarray:
    return np.array([P_X, 0.0, Z_P]) + lever_rot(beta) @ HEEL


def lever_end(beta: float) -> np.ndarray:
    return np.array([P_X, 0.0, Z_P]) + lever_rot(beta) @ LEVER_END


def upper_rot(theta: float) -> np.ndarray:
    """Upper arm frame -> C at jaw angle ``theta`` (rad, + closes: the front goes down)."""
    return rot_y(theta)


def _heel_clear(beta: float, theta: float) -> float:
    n = upper_rot(theta) @ np.array([0.0, 0.0, 1.0])
    return float(n @ (heel_centre(beta) - np.array([X_H, 0.0, Z_H])) - UP_T - HEEL_R)


def heel_theta(beta: float) -> float:
    """Jaw angle the heel pushes the upper arm to at lever angle ``beta`` (0 if the heel is clear of it): the cam's
    exact geometry, from which the simulated coupling's ratio is taken."""
    if _heel_clear(beta, 0.0) >= 0:
        return 0.0
    lo, hi = 0.0, 0.3
    for _ in range(60):
        mid = (lo + hi) / 2
        if _heel_clear(beta, mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


THETA_SHUT = math.asin(GAP0 / (E_X - X_H))   # the edges meet


def _heel_beta(theta: float) -> float:
    lo, hi = -0.5, PHI0
    for _ in range(60):
        mid = (lo + hi) / 2
        if heel_theta(mid) > theta:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# The cam, as simulated: a mimic (gear) joint ties the upper arm's hinge to the lever's: jaw = CAM_GEAR x (how far
# the lever is pressed). The ratio is the heel's, over the stroke from rest to shut (it is within 0.1 degree of the
# exact cam everywhere on the stroke: see the tests).
CAM_GEAR = THETA_SHUT / (PHI0 - _heel_beta(THETA_SHUT))


def theta_from_beta(beta: float) -> float:
    """Jaw angle for lever angle ``beta`` (the simulated cam)."""
    return min(max(CAM_GEAR * (PHI0 - beta), 0.0), THETA_SHUT)


def beta_from_theta(theta: float) -> float:
    return PHI0 - theta / CAM_GEAR


BETA_SHUT = beta_from_theta(THETA_SHUT)
LEVER_MIN = BETA_SHUT - PHI0 + 0.002    # the lever's joint stop (rad, from rest): the jaws a hair from shut


def x_edge(y) -> np.ndarray:
    """C x of the cutting edges across the jaws (curved: the ends reach further forward)."""
    return E_X + SAG * (np.asarray(y, float) / JAW_HW) ** 2


N_EDGE = 27
EDGE_Y = np.linspace(-JAW_HW, JAW_HW, N_EDGE)


def lower_edge_c() -> np.ndarray:
    return np.c_[x_edge(EDGE_Y), EDGE_Y, np.full(N_EDGE, Z_LE)]


def upper_edge_c(theta: float) -> np.ndarray:
    local = np.c_[x_edge(EDGE_Y) - X_H, EDGE_Y, np.zeros(N_EDGE)]
    return np.array([X_H, 0.0, Z_H]) + local @ upper_rot(theta).T


def jaw_gap(theta: float) -> float:
    """Opening between the edges at their centre (m), from the jaw angle."""
    u = upper_edge_c(theta)[N_EDGE // 2]
    return float(u[2] - Z_LE)


def lever_height(beta: float) -> float:
    """Height of the lever end's top above the body's underside (C z): the pad gap when held lever-to-body."""
    return float(lever_end(beta)[2])


def jaw_state_from_gap(gap: float):
    """(theta, beta) for a jaw opening ``gap`` (m)."""
    theta = math.asin(max(0.0, GAP0 - gap) / (E_X - X_H))
    return theta, beta_from_theta(theta)


def lever_force(theta: float) -> float:
    """Force at the lever's end (N) to hold the jaws at ``theta`` against the upper arm's spring (lever ratio from the
    geometry, at the heel and the end)."""
    tau = JAW_TAU0 + JAW_K * max(theta, 0.0)
    beta = beta_from_theta(theta)
    h = heel_centre(beta)
    heel_force = tau / (h[0] - X_H)
    e = lever_end(beta)
    return heel_force * (h[0] - P_X) / (e[0] - P_X)


# The arms' spring, felt where the hand feels it, at the lever: the lever joint's spring gives the torque the arms'
# spring (JAW_TAU0, JAW_K) gives through the cam (LEVER_TAU0 at rest, rising to the shut jaws' value).
JAW_D = 0.002


def _lever_tau(theta: float) -> float:
    beta = beta_from_theta(theta)
    return lever_force(theta) * float(P_X - lever_end(beta)[0])


LEVER_TAU0 = _lever_tau(0.0)
LEVER_K_EQ = (_lever_tau(THETA_SHUT) - LEVER_TAU0) / (PHI0 - BETA_SHUT)
LEVER_PRE = LEVER_TAU0 / LEVER_K_EQ     # the lever's spring target above its rest stop (a preload)
F_CUT = 6.0                             # N at the lever's pivot (about the squeeze on the lever) to cut through a nail


# -- holding and squeezing: the gripper's pads on the lever's end and the body's underside ------------------------------
X_T = 0.0180                            # the TCP's position along the clipper (C x) when held
GRIP_K = 5000.0                         # N/m per finger (yam_arena GRIPPER stiffness)
FULL_OPEN = 2 * 0.04695                 # pad gap at grip 1.0 (m)
HOLD_GAP = 2.5 * MM                     # jaws held 2.5 mm open while placing them
SHUT_PRESS = 4.0 * MM                   # squeeze: command the pads this far past the shut lever


def pad_gap(gap: float) -> float:
    """Pad gap (m) with the jaws ``gap`` open."""
    return lever_height(jaw_state_from_gap(gap)[1])


def grip_for_gap(gap: float) -> float:
    """Gripper command (1 open .. 0 closed) that holds the jaws ``gap`` open: the pads' targets sit inside the
    clipper by the spring force over their stiffness."""
    theta, _ = jaw_state_from_gap(gap)
    f = lever_force(theta)
    return (pad_gap(gap) - 2 * f / GRIP_K) / FULL_OPEN


def squeeze_grip() -> float:
    return (pad_gap(0.0) - SHUT_PRESS) / FULL_OPEN


def tcp_in_c(pg: float) -> np.ndarray:
    """The TCP in C for a pad gap ``pg``: midway between the body's underside and the lever's end."""
    return np.array([X_T, 0.0, pg / 2])


def tcp_rot_from_c(r_c) -> np.ndarray:
    """TCP axes from C's: the fingers close along C z, the TCP's y is -C y, it points along C x."""
    r_c = np.asarray(r_c, float)
    return np.column_stack([r_c[:, 2], -r_c[:, 1], r_c[:, 0]])


def c_rot_from_tcp(r_t) -> np.ndarray:
    r_t = np.asarray(r_t, float)
    return np.column_stack([r_t[:, 2], -r_t[:, 1], r_t[:, 0]])


def clipper_boxes_c(theta: float = 0.0, beta: float = PHI0):
    """The clipper as boxes (centre, rotation, half sizes) in C: arms and jaws, the upper arm, the lever."""
    out = [(np.array([E_X / 2 + SAG / 2, 0.0, Z_LE / 2]), np.eye(3), np.array([E_X / 2 + SAG / 2, ARM_W / 2, Z_LE / 2])),
           (np.array([(X_H + 0.0035) / 2, 0.0, (LOW_T + Z_H) / 2]), np.eye(3),
            np.array([0.0035, 0.006, (Z_H - LOW_T) / 2]))]
    ru = upper_rot(theta)
    cu = np.array([X_H, 0.0, Z_H]) + ru @ np.array([(E_X - X_H) / 2 - 0.002, 0.0, UP_T / 2])
    out.append((cu, ru, np.array([(E_X - X_H) / 2 + 0.0023, ARM_W / 2, UP_T / 2])))
    rl = lever_rot(beta)
    cl = np.array([P_X, 0.0, Z_P]) + rl @ np.array([-(LEVER_L - 0.003) / 2, 0.0, 0.0])
    out.append((cl, rl, np.array([(LEVER_L + 0.003) / 2, LEVER_W / 2, LEVER_T / 2])))
    return out


def clipper_boxes(pos, rot, theta: float = 0.0, beta: float = PHI0):
    rot = np.asarray(rot, float)
    return [(np.asarray(pos, float) + rot @ c, rot @ r, h) for c, r, h in clipper_boxes_c(theta, beta)]


# -- the caddy (fixed shapes) and the clipper's start -----------------------------------------------------------------
# A moulded nail-care caddy on the table in front of the right arm. The clipper stands in its slot, jaws down, the
# lever toward the far side (+x).
CADDY_C = np.array([0.205, -0.215])      # caddy centre (x, y)
CADDY_SIZE = (0.080, 0.120, 0.029)      # x, y, height
SLOT = (0.0085, 0.0165, 0.009)          # slot: along x (the clipper's thickness), along y (its width), depth
SLOT_C = np.array([0.205, -0.215])      # slot centre (x, y)
SLOT_FLOOR = TABLE + CADDY_SIZE[2] - SLOT[2]
R_STAND = np.column_stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # C axes standing: x down, z to +x
CLIPPER_START = np.array([SLOT_C[0] - SLOT[0] / 2 + 0.0005, SLOT_C[1], SLOT_FLOOR + 0.0005 + E_X + SAG])


def caddy_boxes():
    """The caddy as fixed boxes (name, centre, size): a base under the slot's floor and a rim round the slot."""
    cx, cy = CADDY_C
    sx, sy, h = CADDY_SIZE
    base_h = SLOT_FLOOR - TABLE
    out = [("base", np.array([cx, cy, TABLE + base_h / 2]), np.array([sx, sy, base_h]))]
    x0, x1 = SLOT_C[0] - SLOT[0] / 2, SLOT_C[0] + SLOT[0] / 2
    y0, y1 = SLOT_C[1] - SLOT[1] / 2, SLOT_C[1] + SLOT[1] / 2
    z = SLOT_FLOOR + SLOT[2] / 2
    out.append(("rim_back", np.array([(cx - sx / 2 + x0) / 2, cy, z]), np.array([x0 - (cx - sx / 2), sy, SLOT[2]])))
    out.append(("rim_front", np.array([(x1 + cx + sx / 2) / 2, cy, z]), np.array([cx + sx / 2 - x1, sy, SLOT[2]])))
    out.append(("rim_right", np.array([(x0 + x1) / 2, (cy - sy / 2 + y0) / 2, z]), np.array([SLOT[0], y0 - (cy - sy / 2), SLOT[2]])))
    out.append(("rim_left", np.array([(x0 + x1) / 2, (y1 + cy + sy / 2) / 2, z]), np.array([SLOT[0], cy + sy / 2 - y1, SLOT[2]])))
    return out


# -- the cut model ----------------------------------------------------------------------------------------------------
BAND = (1.0 * MM, 2.0 * MM)             # white edge to leave in every bin
STRAIGHT = 0.75 * MM                    # straight across: longest minus shortest bin at most this
NICK_U = 0.5 * MM                       # an edge closing within this of the hyponychium (or behind it) nicks the skin
CUT_GAP = NAIL_T + 0.3 * MM             # a closure: the jaws at most this open ...
CUT_STEPS = 2                           # ... for this many control steps in a row
REARM_GAP = 1.5 * MM                    # the jaws must open this far before the next closure counts
PINCH_GAP = 2.0 * MM                    # jaws this far shut with tissue between the edges: they pinch the skin
PINCH_DEPTH = 0.3 * MM
STEADY_DEG = 10.0                       # a finger turned more than this from rest at its knuckle was not held steady


def edge_points_world(c_pos, c_rot, theta: float) -> np.ndarray:
    """The cutting line (midway between the lower and upper edges) in the station frame."""
    mid = (lower_edge_c() + upper_edge_c(theta)) / 2
    return np.asarray(c_pos, float) + mid @ np.asarray(c_rot, float).T


def edge_segments_world(c_pos, c_rot, theta: float):
    lo = np.asarray(c_pos, float) + lower_edge_c() @ np.asarray(c_rot, float).T
    up = np.asarray(c_pos, float) + upper_edge_c(theta) @ np.asarray(c_rot, float).T
    return lo, up


def cut_line_in_nail(name: str, pts_world, f_pos, f_rot) -> np.ndarray:
    """Cutting-line points in the nail's frame N (u, v, w)."""
    n0, nr = nail_pose(name, f_pos, f_rot)
    return (np.asarray(pts_world, float) - n0) @ nr


def closure(name: str, uvw: np.ndarray, length: np.ndarray):
    """One closure of the jaws on this nail. ``uvw``: the cutting line in N when the jaws shut (where their edges
    meet); ``length``: white edge per bin (m). Returns (new lengths, cut mask, nicked, info). The shut jaws' blades
    fill the space from Z_LE (2 mm, the lower jaw) below that line to UP_T (1.5 mm, the upper jaw) above it: where the
    nail's mid-plane lies inside them, the closing jaws went through the nail, and the line cuts it. A bin is severed
    where the line crosses it between the hyponychium and the bin's tip; the line within NICK_U of the hyponychium (or
    behind it) over the nail or 0.5 mm past its sides nicks the skin."""
    uvw = np.asarray(uvw, float)
    order = np.argsort(uvw[:, 1])
    u, v, w = uvw[order, 0], uvw[order, 1], uvw[order, 2]
    half = FINGERS[name]["W"] / 2
    in_plane = (w - Z_LE <= 0.0) & (w + UP_T >= 0.0)
    nick = bool(np.any(in_plane & (np.abs(v) <= half + 0.5 * MM) & (u < NICK_U)))
    new = np.array(length, float)
    cut = np.zeros(N_BINS, bool)
    for b, vb in enumerate(bin_v(name)):
        if vb < v[0] or vb > v[-1]:
            continue
        k = int(np.argmin(np.abs(v - vb)))
        ub = float(np.interp(vb, v, u))
        if not in_plane[k] or ub >= new[b] or ub <= 0.0:
            continue
        new[b] = ub
        cut[b] = True
    return new, cut, nick, {"u_mid": float(np.interp(0.0, v, u)) if v[0] <= 0 <= v[-1] else None,
                            "w_mid": float(np.interp(0.0, v, w)) if v[0] <= 0 <= v[-1] else None}


def pinches_skin(name: str, lo_world, up_world, f_pos, f_rot) -> bool:
    """Tissue between the edges: any point on the segments from the lower to the upper edge PINCH_DEPTH inside the
    fingertip's soft tissue."""
    f_rot = np.asarray(f_rot, float)
    pts = []
    for s in np.linspace(0.0, 1.0, 7):
        pts.append(np.asarray(lo_world) * (1 - s) + np.asarray(up_world) * s)
    p = (np.vstack(pts) - np.asarray(f_pos, float)) @ f_rot
    return bool(np.any(tissue_dist(name, p) >= PINCH_DEPTH))


# -- box maths -------------------------------------------------------------------------------------------------------
def box_points(c, r, h, n: int = 5) -> np.ndarray:
    u = np.linspace(-1, 1, n)
    grid = np.array([[a, b, d] for a in u for b in u for d in u])
    grid = grid[np.abs(grid).max(axis=1) == 1.0]
    return np.asarray(c) + (grid * np.asarray(h)) @ np.asarray(r).T


def point_box_dist(points, c, r, h) -> np.ndarray:
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    d = np.maximum(np.abs(local) - np.asarray(h), 0.0)
    return np.linalg.norm(d, axis=-1)


def boxes_gap(a, b, n: int = 7) -> float:
    """Approximate gap between two boxes (centre, rotation, half sizes); 0 if they touch or overlap."""
    return float(min(point_box_dist(box_points(*a, n), *b).min(), point_box_dist(box_points(*b, n), *a).min()))


def point_capsule_dist(points, a, b, radius) -> np.ndarray:
    """Distance from points to a capsule (segment a-b, radius); 0 inside."""
    p = np.atleast_2d(points)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.maximum(np.linalg.norm(p - (a + np.outer(t, ab)), axis=1) - radius, 0.0)


def finger_capsules_world(name: str, f_pos, f_rot):
    """The finger's capsules (a, b, radius) in the station frame."""
    f_pos, f_rot = np.asarray(f_pos, float), np.asarray(f_rot, float)
    out = []
    for _, x0, x1, r in finger_capsules(name):
        out.append((f_pos + f_rot @ np.array([x0, 0.0, 0.0]), f_pos + f_rot @ np.array([x1, 0.0, 0.0]), r))
    return out


def static_capsules_world():
    """The hand's fixed parts as capsules (a, b, radius), station frame: ring, little finger, thumb, forearm."""
    out = []
    for _, mcp, deg, length, r in STATIC_FINGERS:
        out.append((np.asarray(mcp, float), np.asarray(mcp, float) + length * _dir(deg), r))
    base, deg, length, r = THUMB
    out.append((np.asarray(base, float), np.asarray(base, float) + length * _dir(deg), r))
    back = HAND0 - (PALM["back"] - 0.01) * _XM + PALM["centre_y"] * _YM + np.array([0.0, 0.0, FOREARM["z"]])
    out.append((back, back - FOREARM["length"] * _XM, FOREARM["radius"]))
    return out


def palm_box():
    c = HAND0 - (PALM["back"] / 2 - 0.006) * _XM + PALM["centre_y"] * _YM + np.array([0.0, 0.0, -0.002])
    return c, rot_z(math.radians(MIDDLE_DEG)), np.array([PALM["back"] / 2, PALM["width"] / 2, PALM["thick"] / 2])
