"""Geometry of the stow-knife-blade-away scene: the knife, the cutting board and the open drawer with its knife tray,
in the station frame (x forward, away from the robot; y to the robot's left; z up; metres). Pure NumPy, so the task
file, the asset script and the tests can import it before Isaac Sim starts.

Knife frame: the origin is on the knife's flat underside, at the bolster (where the handle meets the blade), on the
handle's centre line. +x runs toward the blade's tip, +z up out of the face it lies on. The knife lies on its side,
so its handle and blade undersides are one flat face.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                    # yam_arena.task_file.TABLE_TOP["station"]

# -- the knife (knife frame) -----------------------------------------------------------------------------------------
HANDLE_LEN, HANDLE_W, HANDLE_T = 0.100, 0.024, 0.022   # wooden handle: length, width (in the blade's plane), thickness
BLADE_LEN, BLADE_W, BLADE_T = 0.100, 0.028, 0.004      # full-width part of the blade; 4 mm thick for stable contact
TIP_LEN, TIP_W = 0.020, 0.016                          # the narrower point, along the spine
SPINE_Y = -HANDLE_W / 2                                # the blade's spine is flush with one side of the handle
KNIFE_LEN = HANDLE_LEN + BLADE_LEN + TIP_LEN           # 22 cm
KNIFE_MASS = 0.09

# name: (centre, full size), knife frame
KNIFE_PARTS = {
    "handle": ((-HANDLE_LEN / 2, 0.0, HANDLE_T / 2), (HANDLE_LEN, HANDLE_W, HANDLE_T)),
    "blade": ((BLADE_LEN / 2, SPINE_Y + BLADE_W / 2, BLADE_T / 2), (BLADE_LEN, BLADE_W, BLADE_T)),
    "tip": ((BLADE_LEN + TIP_LEN / 2, SPINE_Y + TIP_W / 2, BLADE_T / 2), (TIP_LEN, TIP_W, BLADE_T)),
}
BUTT_X, TIP_X = -HANDLE_LEN, BLADE_LEN + TIP_LEN

# Where a gripper takes the handle: 3.5 cm behind the bolster, the tool point 5 mm under the handle's top face, so
# the pad tips stop 4 mm above the surface the knife lies on.
GRIP = np.array([-0.035, 0.0, HANDLE_T - 0.005])

# -- the cutting board (fixed) ---------------------------------------------------------------------------------------
BOARD_LO, BOARD_HI = (0.12, -0.17), (0.42, 0.13)       # x, y
BOARD_T = 0.015
BOARD_TOP = TOP + BOARD_T

# -- the drawer and its knife tray (fixed); its front, with the pull, faces the robot --------------------------------
FLOOR_T = 0.010
FLOOR = TOP + FLOOR_T                                  # the tray floor's top
WALL_TOP = TOP + 0.050                                 # drawer walls, from the table
DIVIDER_H = 0.015                                      # dividers, above the tray floor
FRONT_IN, BACK_IN = 0.180, 0.470                       # inner faces of the front panel and the back wall (x)
FRONT_T, WALL_T = 0.020, 0.012                         # front panel, other walls
SLOT_W, DIVIDER_W = 0.088, 0.010
RIGHT_IN = 0.170                                       # the right side wall's inner face (y)
SLOTS = tuple((RIGHT_IN + i * (SLOT_W + DIVIDER_W), RIGHT_IN + i * (SLOT_W + DIVIDER_W) + SLOT_W) for i in range(3))
LEFT_IN = SLOTS[-1][1]
EMPTY = SLOTS[0]                                       # the knife's slot: the one nearest the board
TRAY_LO, TRAY_HI = (FRONT_IN, RIGHT_IN), (BACK_IN, LEFT_IN)

# -- poses -------------------------------------------------------------------------------------------------------------


def rz(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def origin_for_grip(grip_xy, yaw: float, floor: float) -> np.ndarray:
    """The knife origin that puts its grip point over ``grip_xy`` with the knife lying flat on ``floor``, its
    blade pointing along ``yaw``."""
    off = rz(yaw) @ GRIP
    return np.array([grip_xy[0] - off[0], grip_xy[1] - off[1], floor])


# The start: on the board, blade toward the robot (-x), handle away from it, gripped in front of the right arm.
START_YAW = math.pi
START_GRIP_XY = (0.30, -0.03)
START = origin_for_grip(START_GRIP_XY, START_YAW, BOARD_TOP + 0.001)

# Where the demo's right arm lays the knife down, turned a quarter turn (blade toward the robot's right).
MID_YAW = -math.pi / 2
MID_GRIP_XY = (0.28, 0.04)

# Stowed: in the empty slot, handle toward the drawer front (the robot), blade toward the back, centred along x.
GOAL_YAW = 0.0
GOAL_ORIGIN_X = (FRONT_IN + BACK_IN) / 2 - (BUTT_X + TIP_X) / 2
GOAL_GRIP_XY = (GOAL_ORIGIN_X + GRIP[0], (EMPTY[0] + EMPTY[1]) / 2)


def yaw_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, float(math.sin(yaw / 2)), float(math.cos(yaw / 2)))


def quat_to_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(q_xyzw, float) / np.linalg.norm(q_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def knife_corners(pos, rot) -> np.ndarray:
    """The 24 corners of the knife's three boxes (world frame) for a knife origin at ``pos`` turned by ``rot``."""
    out = []
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    for centre, size in KNIFE_PARTS.values():
        local = np.asarray(centre) + signs * np.asarray(size) / 2
        out.append(local @ np.asarray(rot).T + np.asarray(pos))
    return np.vstack(out)


def knife_axis(pos, rot, n: int = 12) -> np.ndarray:
    """Points along the knife's centre line, butt to tip, at half the handle's thickness (world frame)."""
    xs = np.linspace(BUTT_X, TIP_X, n)
    local = np.stack([xs, np.zeros(n), np.full(n, HANDLE_T / 2)], axis=1)
    return local @ np.asarray(rot).T + np.asarray(pos)
