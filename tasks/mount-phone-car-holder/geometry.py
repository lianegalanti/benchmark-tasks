"""Dimensions, layout and box maths for the phone-holder task. Pure NumPy; shared by make_assets.py, task.py, demo.py
and the tests.

Station frame: x forward, y to the robot's left, z up (m). The holder frame H is the station frame shifted to
``HOLDER``: its origin is the centre of the cradle's back-plate front face, at the height of the feet's tops (where
the phone's bottom edge rests). The cradle faces the robot, so a mounted phone lies at x < 0 in H, its back toward +x.

The holder (a one-touch, gear-linked two-arm cradle on an air-vent clip) is a fixed-base articulation:
- root ``body``: back plate, two feet, the housing behind the plate and the vent clip;
- ``arm_left`` / ``arm_right``: prismatic along y, 0 = closed (pads 58 mm apart) to ``ARM_STOP`` (pads 94 mm apart);
  a spring pulls each toward closed; ``arm_right`` mimics ``arm_left`` (the rack and pinion gearing them together);
- ``trigger``: prismatic along +x (pressed into the housing), 0 = rest (pad standing 6 mm proud of the plate) to
  ``TRIG_STOP``; a spring pushes it back out. It carries the two catches inside the housing.

Each arm has a tail inside the housing. Pulled open, the tail's outer face rides up a catch's 45-degree chamfer,
pushing the trigger in, and once past, the catch springs back out behind it: the arms stay open (cocked). Pressing the
trigger in by ``RELEASE`` or more moves the catches out of the tails' way and the springs close the arms.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import station

TABLE = float(station.TABLE_TOP)
MM = 0.001

# -- the phone (iPhone 15-class) ------------------------------------------------------------------------------------
PHONE_L, PHONE_W, PHONE_T = 0.1476, 0.0716, 0.0078     # m: height, width, thickness
PHONE_MASS = 0.171                                       # kg
PHONE_HALF = np.array([PHONE_T / 2, PHONE_W / 2, PHONE_L / 2])   # phone frame: x thickness (+x = back), y width,
#                                                                 z long axis (+z = top); upright in the cradle it is
#                                                                 the holder frame

# -- the holder (H frame, m) ----------------------------------------------------------------------------------------
HOLDER = np.array([0.27, 0.0, TABLE + 0.030])   # H origin in the station frame
CLOSED_HALF = 0.029          # pad faces' half span with the arms closed (58 mm)
ARM_STOP = 0.018             # arm travel to the opening stop (94 mm)
LATCHED = 0.016              # arm travel when the catch holds it (90 mm)
PHONE_D = PHONE_W / 2 - CLOSED_HALF       # arm travel when the pads grip the phone's edges (6.8 mm)
ARM_RATE = 80.0              # N/m, each arm's spring
ARM_PRELOAD = 0.8            # N, each arm's spring force when closed
ARM_TARGET = -ARM_PRELOAD / ARM_RATE      # spring rest point (beyond the closed stop)
ARM_DAMPING = 1.5            # N s/m
ARM_MASS = 0.03              # kg each
TRIG_PROUD = 0.006           # trigger pad stands this far proud of the back plate at rest
TRIG_STOP = 0.006            # trigger travel to its stop (pad flush with the plate)
TRIG_RATE = 300.0            # N/m
TRIG_PRELOAD = 1.2           # N at rest
TRIG_TARGET = -TRIG_PRELOAD / TRIG_RATE
TRIG_DAMPING = 0.5
TRIG_MASS = 0.01
OVERLAP = 0.003              # catch overlaps the tail by this much (along x) at rest ...
RELEASE = OVERLAP            # ... so the trigger releases the arms once pressed this far
BODY_MASS = 0.20             # kg, the holder head (fixed in the sim)

# root boxes (lo, hi) in H, with a material key
PLATE = ((0.0, -0.032, -0.010), (0.006, 0.032, 0.092))
FEET = [((-0.018, 0.010, -0.005), (0.0, 0.030, 0.0)), ((-0.018, -0.030, -0.005), (0.0, -0.010, 0.0))]
HOUSING = ((0.006, -0.044, 0.010), (0.040, 0.044, 0.080))
CLIP = ((0.040, -0.010, 0.030), (0.045, 0.010, 0.050))
DASH_FACE_X = 0.045          # the dashboard's face, behind the vent clip (H frame)

# arm (left; the right arm is its mirror in y). Boxes in the arm link's frame = H at arm travel 0.
JAW_Z = (0.055, 0.085)
T0 = 0.012                   # tail's inner face at travel 0 (y)
TAIL_W = 0.0025
TAIL_X = (0.016, 0.024)
LATCH_Z = (0.038, 0.050)
ARM_BOXES = [
    ("pad", (-0.012, CLOSED_HALF, JAW_Z[0]), (0.0, CLOSED_HALF + 0.002, JAW_Z[1])),          # silicone pad
    ("lip", (-0.016, CLOSED_HALF - 0.003, JAW_Z[0]), (-0.012, CLOSED_HALF + 0.002, JAW_Z[1])),
    ("jaw", (-0.016, CLOSED_HALF + 0.002, JAW_Z[0]), (0.008, CLOSED_HALF + 0.010, JAW_Z[1])),
    ("bar", (0.008, T0, 0.056), (0.024, CLOSED_HALF + 0.010, 0.064)),
    ("tail", (TAIL_X[0], T0, 0.036), (TAIL_X[1], T0 + TAIL_W, 0.064)),
]

# catch (left; mirrored for the right), on the trigger link (frame = H at trigger travel 0)
C_OUT = T0 + LATCHED                     # catch's outer (latching) face: the tail rests on it when cocked
C_W = 0.005
C_IN = C_OUT - C_W
C_X = (TAIL_X[1] - OVERLAP, TAIL_X[1] - OVERLAP + 0.006)   # x 21..27 mm
CHAMFER = 0.004                          # 45-degree chamfer on the catch's inner, low-x corner


def catch_boxes(sign: float = 1.0):
    """The catch as (name, centre, size, yaw) boxes in the trigger frame; ``sign`` -1 mirrors it to the right."""
    x0, x1 = C_X
    out = [("catch_back", ((x0 + CHAMFER + x1) / 2, (C_IN + C_OUT) / 2, sum(LATCH_Z) / 2),
            (x1 - x0 - CHAMFER, C_W, LATCH_Z[1] - LATCH_Z[0]), 0.0),
           ("catch_face", ((x0 + x0 + CHAMFER) / 2, (C_IN + CHAMFER + C_OUT) / 2, sum(LATCH_Z) / 2),
            (CHAMFER, C_OUT - C_IN - CHAMFER, LATCH_Z[1] - LATCH_Z[0]), 0.0)]
    # the chamfer: a slab whose outer face runs from (x0 + CHAMFER, C_IN) to (x0, C_IN + CHAMFER)
    t = 0.0015
    mid = np.array([x0 + CHAMFER / 2, C_IN + CHAMFER / 2])
    inward = np.array([1.0, 1.0]) / math.sqrt(2)       # (x, y): from the face into the catch
    c = mid + inward * t / 2
    out.append(("catch_chamfer", (float(c[0]), float(c[1]), sum(LATCH_Z) / 2), (CHAMFER * math.sqrt(2), t,
                                                                  LATCH_Z[1] - LATCH_Z[0]), math.radians(-45.0)))
    if sign < 0:
        out = [(n, (c[0], -c[1], c[2]), s, -yaw) for n, c, s, yaw in out]
    return out


TRIGGER_BOXES = [
    ("pad", (-TRIG_PROUD, -0.010, 0.060), (0.0, 0.010, 0.080)),       # at the phone's mid-height
    ("stem", (0.0, -0.005, 0.038), (C_X[0] + CHAMFER, 0.005, 0.080)),
    ("crossbar", (C_X[0] + CHAMFER, -C_OUT, 0.038), (C_X[1], C_OUT, 0.050)),
]

# -- the dashboard section and the phone's start --------------------------------------------------------------------
DASH_X = float(HOLDER[0] + DASH_FACE_X)    # its front face (station x)
DASH_H = 0.09
DASH_D = 0.15
DASH_W = 0.60
LEAN = math.radians(18.0)    # the phone leans back this far from vertical against the dash's top front edge
PHONE_Y0 = -0.16


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


PHONE_R0 = rot_y(LEAN)       # phone frame -> station at the start: its top leans toward +x


def phone_start(gap: float = 0.0003) -> np.ndarray:
    """Centre of the leaning phone: its lowest edge ``gap`` above the table and its back ``gap`` off the dash edge."""
    r = PHONE_R0
    # lowest point: back (+x local) bottom (-z local) edge
    low = r @ np.array([PHONE_T / 2, 0.0, -PHONE_L / 2])
    cz = TABLE + gap - low[2]
    n = r @ np.array([1.0, 0.0, 0.0])      # back-face normal
    edge = np.array([DASH_X, 0.0, TABLE + DASH_H])
    # back face plane: (p - c) . n = T/2 (+gap) through the edge
    cx = edge[0] - ((PHONE_T / 2 + gap) - (edge[2] - cz) * n[2]) / n[0]
    return np.array([cx, PHONE_Y0, cz])


def to_station(p) -> np.ndarray:
    return np.asarray(p, float) + HOLDER


def quat_xyzw(r: np.ndarray) -> tuple:
    from yam_arena.ik import matrix_to_quat_xyzw

    return tuple(float(v) for v in matrix_to_quat_xyzw(r))


# -- box maths ------------------------------------------------------------------------------------------------------
def box_points(c, r, h, n: int = 5) -> np.ndarray:
    """Points on a box's surface grid (centre, rotation, half sizes)."""
    u = np.linspace(-1, 1, n)
    g = np.array([[a, b, d] for a in u for b in u for d in u])
    g = g[np.abs(g).max(axis=1) == 1.0]
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def point_box_dist(points, c, r, h) -> np.ndarray:
    """Distance from points to a solid box (0 inside)."""
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    d = np.maximum(np.abs(local) - np.asarray(h), 0.0)
    return np.linalg.norm(d, axis=-1)


def boxes_gap(a, b, n: int = 7) -> float:
    """Approximate gap between two boxes (sampled; 0 if they touch or overlap)."""
    return float(min(point_box_dist(box_points(*a, n), *b).min(), point_box_dist(box_points(*b, n), *a).min()))


def lohi(lo, hi, offset=(0.0, 0.0, 0.0)):
    """A (centre, rotation, half sizes) box from corners, shifted by ``offset``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return ((lo + hi) / 2 + np.asarray(offset, float), np.eye(3), (hi - lo) / 2)


def phone_box(pos, rot):
    return (np.asarray(pos, float), np.asarray(rot, float), PHONE_HALF)


def jaw_boxes(travel_left: float, travel_right: float):
    """The two jaws (pad, lip and jaw body together) in H at the given arm travels, as boxes."""
    out = []
    for sign, d in ((1.0, travel_left), (-1.0, travel_right)):
        lo = np.array([-0.016, CLOSED_HALF - 0.003 + d, JAW_Z[0]])
        hi = np.array([0.008, CLOSED_HALF + 0.010 + d, JAW_Z[1]])
        if sign < 0:
            lo, hi = np.array([lo[0], -hi[1], lo[2]]), np.array([hi[0], -lo[1], hi[2]])
        out.append(lohi(lo, hi))
    return out


def tail_y(d: float) -> tuple[float, float]:
    """The left tail's inner and outer faces (y) at arm travel ``d``."""
    return T0 + d, T0 + d + TAIL_W


def catch_blocks(d: float, q_trig: float) -> bool:
    """Whether the left catch stops the left tail moving inward at arm travel ``d`` and trigger travel ``q_trig``:
    the tail is outside the catch (cocked) and the catch still overlaps it along x."""
    t_in, _ = tail_y(d)
    return bool(t_in >= C_OUT - 1e-6 and C_X[0] + q_trig < TAIL_X[1])


def chamfer_x(y: float, q_trig: float) -> float:
    """The catch chamfer's x at height y (inner faces between C_IN and C_IN + CHAMFER) with the trigger at q."""
    return C_X[0] + CHAMFER - (y - C_IN) + q_trig
