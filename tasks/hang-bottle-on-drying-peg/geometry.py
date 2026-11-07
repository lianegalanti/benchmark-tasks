"""Dimensions and geometry for the drying-peg task. NumPy only, no Isaac Sim.

Station frame: x forward from the arms, y to the robot's left, z up, metres. The bottle's own frame has its origin on
its axis halfway along the tube and +z pointing out of its open mouth, so "mouth down" is +z pointing down.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # yam_arena.task_file.TABLE_TOP["station"]

# -- the bottle: a straight wide-neck baby bottle without its teat and cap -------------------------------------------
LENGTH = 0.12          # outside, closed bottom to mouth
R_OUT = 0.030          # outer radius (to the flat of each of the 16 wall staves)
WALL = 0.003
R_IN = R_OUT - WALL    # 2.7 cm: the bore the peg goes into
BOTTOM_T = 0.004       # closed bottom's thickness
COLLAR = 0.015         # the coloured screw-collar band at the mouth
STAVES = 16
MASS = 0.07            # kg
STAVE_W = 2 * R_OUT * math.tan(math.pi / STAVES)

MOUTH_Z = LENGTH / 2                 # mouth plane, bottle frame
INNER_BOTTOM_Z = -LENGTH / 2 + BOTTOM_T  # inside face of the closed bottom, bottle frame


def _com_z() -> float:
    """Centre of mass along the axis (bottle frame), for a uniform-density wall and bottom."""
    wall = 2 * math.pi * (R_OUT - WALL / 2) * WALL * LENGTH
    bottom = math.pi * R_IN**2 * BOTTOM_T
    return (wall * 0.0 + bottom * (-LENGTH / 2 + BOTTOM_T / 2)) / (wall + bottom)


COM_Z = _com_z()  # about -7 mm: the closed end is the heavier one

# -- the drying rack: a flat base with two upright pegs ---------------------------------------------------------------
RACK_XY = (0.36, 0.14)
BASE = (0.09, 0.22, 0.012)           # x, y, thickness
BASE_TOP = TOP + BASE[2]
PEG_R = 0.006                        # 12 mm pegs with rounded tips
PEG_TIP = TOP + 0.15                 # tip height
PEGS = ((RACK_XY[0], RACK_XY[1] + 0.06), (RACK_XY[0], RACK_XY[1] - 0.06))  # 12 cm apart along y
DEMO_PEG = 0                         # the demo hangs the bottle on the left peg (nearer the left arm)

# Hanging, the bottle's inside bottom rests on a peg tip: its mouth is then this high above the rack base.
HUNG_MOUTH_Z = PEG_TIP - (MOUTH_Z - INNER_BOTTOM_Z)
HUNG_RIM_GAP = HUNG_MOUTH_Z - BASE_TOP

# -- the towel and the bottle's start, lying on its side with its mouth toward the robot -----------------------------
TOWEL_XY = (0.21, 0.30)
TOWEL = (0.20, 0.12, 0.004)
TOWEL_TOP = TOP + TOWEL[2]
BOTTLE_START = (0.21, 0.30, TOWEL_TOP + R_OUT + 0.001)   # 1 mm drop onto the towel
# +z of the bottle (its mouth) along world -x: a turn of -90 degrees about y. One stave lies flat on the towel, and
# two stand vertical at the sides, where the fingers close.
BOTTLE_START_XYZW = (0.0, -math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4))

# -- scoring thresholds -----------------------------------------------------------------------------------------------
INSIDE_MIN = 0.03                    # a peg's tip at least this far in past the mouth plane
MOUTH_DOWN = math.radians(30.0)      # bottle axis within this of straight down
RIM_CLEAR = 0.005                    # hanging: the bottle's lowest point this far above the rack base's top
LIFTED = 0.03                        # lifted: its lowest point this far above the table top
FALLEN = TOP - 0.05                  # its centre this far below the table top: it went off the table
GRIPPER_CLEAR = R_OUT + 0.05         # let go: each gripper's centre line this far from the bottle's axis
HOLD_OPEN = 2 * R_OUT + 0.008        # a gripper holds the bottle when its fingers are closer than this ...
HOLD_R = R_OUT + 0.02                # ... and its grasp point this close to the axis


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mouth_dir(R: np.ndarray) -> np.ndarray:
    return R[:, 2]


def point(pos, R, z: float) -> np.ndarray:
    """The point on the bottle's axis at bottle-frame height ``z``, station frame."""
    return np.asarray(pos, float) + R[:, 2] * z


def peg_depth(pos, R, peg_xy) -> float:
    """How far (m) the peg's tip is in past the bottle's mouth plane while inside the bore; -1 when it isn't inside
    (outside the bore's radius or past the bottom)."""
    if mouth_dir(R)[2] >= 0.0:  # the peg rises from below: only a mouth facing down can take it
        return -1.0
    tip = np.array([peg_xy[0], peg_xy[1], PEG_TIP])
    local = R.T @ (tip - np.asarray(pos, float))
    if math.hypot(local[0], local[1]) >= R_IN or local[2] > MOUTH_Z or local[2] < INNER_BOTTOM_Z - 0.003:
        return -1.0
    return float(MOUTH_Z - local[2])


def tilt_from_down(R: np.ndarray) -> float:
    """Angle (rad) between the bottle's mouth direction and straight down."""
    return float(math.acos(max(-1.0, min(1.0, -mouth_dir(R)[2]))))


def lowest_z(pos, R, n: int = 24) -> float:
    """Height of the bottle's lowest point: the lower of its two rims."""
    a = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
    ring = R[:, 0][None, :] * np.cos(a)[:, None] + R[:, 1][None, :] * np.sin(a)[:, None]
    zs = [point(pos, R, z)[2] + R_OUT * ring[:, 2] for z in (MOUTH_Z, -LENGTH / 2)]
    return float(min(z.min() for z in zs))


def seg_dist(a0, a1, b0, b1) -> float:
    """Shortest distance between segments a0-a1 and b0-b1 (sampled finely; exact enough for 1 mm checks)."""
    s = np.linspace(0.0, 1.0, 41)
    pa = np.asarray(a0, float) + np.outer(s, np.asarray(a1, float) - np.asarray(a0, float))
    pb = np.asarray(b0, float) + np.outer(s, np.asarray(b1, float) - np.asarray(b0, float))
    return float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2).min())


def axis_segment(pos, R):
    return point(pos, R, -LENGTH / 2), point(pos, R, MOUTH_Z)
