"""Dimensions, layout and box maths for the phone-holder task. Pure NumPy; shared by make_assets.py, task.py, demo.py
and the tests.

Station frame: x forward, y to the robot's left, z up (m). The holder frame H is the station frame shifted to
``HOLDER``: its origin is the centre of the cradle's back-plate front face, at the height of the feet's tops (where
the phone's bottom edge rests). The cradle faces the robot, so a mounted phone lies at x < 0 in H, its back toward +x.

The holder (a spring-clamp, gear-linked two-arm cradle on an air-vent clip) is a fixed-base articulation:
- root ``body``: back plate, two feet, the housing behind the plate and the vent clip;
- ``arm_left`` / ``arm_right``: prismatic along y, 0 = closed (pads 58 mm apart) to ``ARM_STOP`` (pads 94 mm apart);
  a preloaded spring pulls each toward closed at every travel; ``arm_right`` mimics ``arm_left`` (the rack and
  pinion gearing them together). Each arm has a pull tab on its front, outside the phone's path.

There is no catch, latch or detent: with nothing holding them, the arms always spring back shut. Something (a hand
on a pull tab, or the phone's own edge) has to hold them open while the phone goes in; let go, they close on it.
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
PHONE_D = PHONE_W / 2 - CLOSED_HALF       # arm travel when the pads grip the phone's edges (6.8 mm)
ADMIT_D = PHONE_W / 2 - (CLOSED_HALF - 0.003)   # arm travel at which the lips clear the phone's edges (9.8 mm)
ARM_RATE = 120.0             # N/m, each arm's spring
ARM_PRELOAD = 1.2            # N, each arm's spring force when closed
ARM_TARGET = -ARM_PRELOAD / ARM_RATE      # spring rest point (beyond the closed stop: it always pulls shut)
ARM_DAMPING = 1.5            # N s/m
ARM_MASS = 0.03              # kg each
BODY_MASS = 0.20             # kg, the holder head (fixed in the sim)

# root boxes (lo, hi) in H, with a material key
PLATE = ((0.0, -0.032, -0.010), (0.006, 0.032, 0.092))
FEET = [((-0.018, 0.010, -0.005), (0.0, 0.030, 0.0)), ((-0.018, -0.030, -0.005), (0.0, -0.010, 0.0))]
HOUSING = ((0.006, -0.044, 0.010), (0.040, 0.044, 0.080))
CLIP = ((0.040, -0.010, 0.030), (0.045, 0.010, 0.050))
DASH_FACE_X = 0.045          # the dashboard's face, behind the vent clip (H frame)

# arm (left; the right arm is its mirror in y). Boxes in the arm link's frame = H at arm travel 0.
JAW_Z = (0.055, 0.085)
TAB_X = (-0.040, -0.016)     # the pull tab: a 4 mm fin on the jaw's front, 24 mm deep, 30 mm tall
TAB_Y = (CLOSED_HALF + 0.006, CLOSED_HALF + 0.010)
ARM_BOXES = [
    ("pad", (-0.012, CLOSED_HALF, JAW_Z[0]), (0.0, CLOSED_HALF + 0.002, JAW_Z[1])),          # silicone pad
    ("lip", (-0.016, CLOSED_HALF - 0.003, JAW_Z[0]), (-0.012, CLOSED_HALF + 0.002, JAW_Z[1])),
    ("jaw", (-0.016, CLOSED_HALF + 0.002, JAW_Z[0]), (0.008, CLOSED_HALF + 0.010, JAW_Z[1])),
    ("tab", (TAB_X[0], TAB_Y[0], JAW_Z[0]), (TAB_X[1], TAB_Y[1], JAW_Z[1])),                  # pull tab
    ("bar", (0.008, 0.012, 0.056), (0.024, CLOSED_HALF + 0.010, 0.064)),                      # rack (in the housing)
]


def spring_force(d: float) -> float:
    """Each arm's spring force (N, toward closed) at travel ``d``."""
    return ARM_RATE * (d - ARM_TARGET)


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


def _arm_box(lo, hi, sign: float, d: float):
    lo, hi = np.array(lo, float) + [0.0, d, 0.0], np.array(hi, float) + [0.0, d, 0.0]
    if sign < 0:
        lo, hi = np.array([lo[0], -hi[1], lo[2]]), np.array([hi[0], -lo[1], hi[2]])
    return lohi(lo, hi)


def jaw_boxes(travel_left: float, travel_right: float):
    """The two jaws (pad, lip and jaw body together) in H at the given arm travels, as boxes."""
    lo, hi = (-0.016, CLOSED_HALF - 0.003, JAW_Z[0]), (0.008, CLOSED_HALF + 0.010, JAW_Z[1])
    return [_arm_box(lo, hi, 1.0, travel_left), _arm_box(lo, hi, -1.0, travel_right)]


def pad_boxes(travel_left: float, travel_right: float):
    """The two silicone pads (the faces that meet the phone's edges) in H at the given arm travels, as boxes."""
    lo, hi = ARM_BOXES[0][1], ARM_BOXES[0][2]
    return [_arm_box(lo, hi, 1.0, travel_left), _arm_box(lo, hi, -1.0, travel_right)]


def tab_boxes(travel_left: float, travel_right: float):
    """The two pull tabs in H at the given arm travels, as boxes."""
    lo, hi = (TAB_X[0], TAB_Y[0], JAW_Z[0]), (TAB_X[1], TAB_Y[1], JAW_Z[1])
    return [_arm_box(lo, hi, 1.0, travel_left), _arm_box(lo, hi, -1.0, travel_right)]
