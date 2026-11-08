"""Dimensions, layout and geometric tests for the parcel-this-side-up task. NumPy only, no Isaac Sim.

Station frame: x forward from the arms, y to the robot's left, z up, metres. The parcel's own frame has its origin at
the box's centre and +z pointing to its "up" end (where its THIS SIDE UP arrows point and its red top panel is), so
"the right way up" is +z pointing up.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # yam_arena.task_file.TABLE_TOP["station"]

# -- the parcel: a closed corrugated-cardboard box with ISO 780 "this way up" arrows ----------------------------------
LENGTH = 0.14          # along its up axis (body z)
WIDTH = 0.07           # square cross-section (body x and y)
SIZE = (WIDTH, WIDTH, LENGTH)
HALF = np.array(SIZE) / 2
MASS = 0.30            # kg, evenly spread (the contents are packed tight)

# Printed marks (visual only, 0.5 mm proud of the faces, no collision). On each of the four long faces a white label
# with one bold black arrow pointing to the up end; the up end's square face is a red panel.
DECAL = 0.0005
LABEL = (0.056, 0.110)                 # across the face, along the up axis
LABEL_Z = 0.0                          # label centre along the up axis
SHAFT = (0.018, 0.055, -0.045)         # arrow shaft: width, length, the end nearest the bottom (body z)
HEAD = (0.046, 0.035)                  # arrowhead: base width, height; its base starts where the shaft ends
ARROW_TIP_Z = SHAFT[2] + SHAFT[1] + HEAD[1]  # 4.5 cm above the centre, 2.5 cm short of the up end

# -- the layout -------------------------------------------------------------------------------------------------------
# The parcel lies on one long face, its up axis along y, the up end toward the robot's right (-y): it sits in front of
# the left arm and its arrows, seen from above, point across the table toward the right arm.
PARCEL_XY = (0.28, 0.16)
PARCEL_START = (PARCEL_XY[0], PARCEL_XY[1], TOP + WIDTH / 2 + 0.001)   # 1 mm drop onto the table
PARCEL_START_XYZW = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # +90 degrees about x: body z -> -y

# The green pallet mark: a square of green floor-marking tape on the table.
MARK_XY = (0.30, -0.04)
MARK = (0.12, 0.12, 0.002)             # x, y, thickness
MARK_TOP = TOP + MARK[2]

# -- scoring thresholds -----------------------------------------------------------------------------------------------
ON_MARK = 0.03                         # the parcel's centre, seen from above, within this of the mark's centre
RESTING = 0.01                         # its lowest point within this of the mark's (or table's) top
UPRIGHT = math.radians(15.0)           # its up axis within this of straight up (or, upside down, straight down)
LIFTED = 0.03                          # its lowest point this far above the table top: lifted
FALLEN = TOP - 0.05                    # its centre this far below the table top: it went off the table
CLEAR = 0.02                           # let go: every gripper's centre line at least this far from the parcel
HOLD_IN = 0.015                        # a gripper holds the parcel when its grasp point is within this of the box ...
HOLD_OPEN = WIDTH + 0.008              # ... and its fingers are closer than this


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def up_dir(R: np.ndarray) -> np.ndarray:
    """The parcel's up axis (where its arrows point), station frame."""
    return np.asarray(R)[:, 2]


def tilt(R: np.ndarray) -> float:
    """Angle (rad) between the parcel's up axis and straight up: 0 the right way up, pi upside down."""
    return float(math.acos(max(-1.0, min(1.0, up_dir(R)[2]))))


def corners(pos, R) -> np.ndarray:
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return (signs * HALF) @ np.asarray(R).T + np.asarray(pos, float)


def lowest_z(pos, R) -> float:
    return float(corners(pos, R)[:, 2].min())


def point(pos, R, local) -> np.ndarray:
    """A point given in the parcel's frame, in the station frame."""
    return np.asarray(pos, float) + np.asarray(R) @ np.asarray(local, float)


def box_distance(points, pos, R) -> np.ndarray:
    """Signed distance (m) of each point from the parcel's surface: negative inside."""
    local = (np.atleast_2d(points) - np.asarray(pos, float)) @ np.asarray(R)
    q = np.abs(local) - HALF
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    return outside + inside


def segment_distance(a, b, pos, R, n: int = 25) -> float:
    """Shortest distance (m) from the segment a-b to the parcel (sampled; exact enough for 1 mm checks)."""
    s = np.linspace(0.0, 1.0, n)[:, None]
    pts = np.asarray(a, float) + s * (np.asarray(b, float) - np.asarray(a, float))
    return float(box_distance(pts, pos, R).min())


def classify(pos, R) -> dict:
    """Where a parcel posed at ``pos``/``R`` (centre, rotation; station frame) is, by geometry alone."""
    pos = np.asarray(pos, float)
    low = lowest_z(pos, R)
    t = tilt(R)
    upright_right = t <= UPRIGHT
    upright_any = upright_right or t >= math.pi - UPRIGHT
    centre_off = math.hypot(pos[0] - MARK_XY[0], pos[1] - MARK_XY[1])
    on_mark = bool(centre_off <= ON_MARK and low <= MARK_TOP + RESTING)
    return {"low": low, "tilt": t, "centre_off": centre_off, "on_mark": on_mark,
            "upright_right": bool(upright_right), "upright_any": bool(upright_any),
            "lifted": bool(low >= TOP + LIFTED), "fallen": bool(pos[2] < FALLEN)}
