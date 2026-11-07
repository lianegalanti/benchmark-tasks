"""Geometry of the reclose-vaccine-cooler scene on the station table, the end-state tests, and the poses the demo uses.

Station frame: x forward from the line through the arm bases, y left, z up, in metres (z absolute, the table top is
TABLE). Pure NumPy, so the task file and the tests can import it before Isaac Sim starts.

The cooler is an open box (floor and four walls) that stands still. Its lid is one rigid body: a plate that rests on
the walls' top edges (the rim), a skirt that hangs 2 cm down outside the walls with an 8 mm gap all round (so a lid
that rests flat on the rim is also centred to within 8 mm), and a handle bar on top. The lid's origin is the centre of
the plate's underside, so a seated lid has its origin on the rim plane, over the cooler's centre.
"""

from __future__ import annotations

import math

import numpy as np

TABLE = 0.7475                                   # yam_arena.task_file.TABLE_TOP["station"]

# -- cooler (fixed) --------------------------------------------------------------------------------------------------
COOLER_XY = (0.25, 0.0)
COOLER_HALF = (0.08, 0.10)                       # outer half sizes, x and y
WALL = 0.012                                     # wall and floor thickness
COOLER_H = 0.09                                  # outer height
RIM = TABLE + COOLER_H                           # the walls' top edges
FLOOR_TOP = TABLE + WALL
INNER_HALF = (COOLER_HALF[0] - WALL, COOLER_HALF[1] - WALL)

# -- cartons (loose) -------------------------------------------------------------------------------------------------
CARTON = (0.035, 0.045, 0.07)                    # x, y, height (standing)
CARTON_MASS = 0.08
CARTON_XY = {"orange_carton": (COOLER_XY[0], COOLER_XY[1] - 0.04),
             "white_carton": (COOLER_XY[0], COOLER_XY[1] + 0.04)}

# -- lid (loose, one rigid body) -------------------------------------------------------------------------------------
PLATE_T = 0.012
SKIRT_T, SKIRT_H, SKIRT_GAP = 0.006, 0.02, 0.008
SKIRT_IN = (COOLER_HALF[0] + SKIRT_GAP, COOLER_HALF[1] + SKIRT_GAP)   # inner half sizes of the skirt
LID_HALF = (SKIRT_IN[0] + SKIRT_T, SKIRT_IN[1] + SKIRT_T)             # outer half sizes of plate and skirt
HANDLE = (0.08, 0.025, 0.035)                    # bar on the plate's centre, long along the lid's x
LID_MASS = 0.2

# -- tray (fixed) and where the demo puts the lid down -----------------------------------------------------------------
TRAY_XY = (0.22, -0.31)
TRAY = (0.12, 0.12, 0.008)
TRAY_TOP = TABLE + TRAY[2]
LID_ASIDE_XY = (0.29, 0.27)                      # lid centre while it lies on the table, skirt down

# -- grasps (tool point heights, absolute z) ---------------------------------------------------------------------------
PAD_TIP = 0.013                                  # yam_arena.constants.PAD_TIP_PAST_TCP
GRIPPER_OPEN = 0.094
HANDLE_TCP_DZ = HANDLE[2] - 0.012                # tool point above the plate's top: fingertips 1 cm above the plate
CARTON_TCP_DOWN = 0.012                          # tool point below a standing carton's top: tips 2.5 cm down
CARTON_OPEN = 0.07                               # finger opening around a carton (pads 1.75 cm clear of it)
CARTON_YAW = math.pi / 2                         # fingers close along x, across the carton's 3.5 cm side

# -- end-state tests ---------------------------------------------------------------------------------------------------
SEAT_Z = 0.004                                   # m: plate underside within this of the rim
SEAT_XY = 0.012                                  # m: lid centre within this of the cooler's centre
SEAT_TILT = math.radians(3.0)
OFF_TABLE = 0.05                                 # m: a centre this far below the table top has fallen off it


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt(quat_xyzw) -> float:
    """Angle (rad) between the body's z axis and the vertical."""
    return math.acos(max(-1.0, min(1.0, rotation(quat_xyzw)[2, 2])))


def yaw(quat_xyzw) -> float:
    r = rotation(quat_xyzw)
    return math.atan2(r[1, 0], r[0, 0])


def yaw_xyzw(a: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def lid_seated(pos, quat_xyzw) -> bool:
    """The lid rests flat on the rim: plate underside within 4 mm of the rim, tilted under 3 degrees, centre within
    12 mm of the cooler's (the skirt already keeps it within 8 mm)."""
    return bool(abs(pos[2] - RIM) < SEAT_Z and tilt(quat_xyzw) < SEAT_TILT
                and math.hypot(pos[0] - COOLER_XY[0], pos[1] - COOLER_XY[1]) < SEAT_XY)


def handle_centre(pos, quat_xyzw) -> np.ndarray:
    return np.asarray(pos, float) + rotation(quat_xyzw) @ np.array([0.0, 0.0, PLATE_T + HANDLE[2] / 2])


def in_cooler(pos) -> bool:
    """A carton's centre inside the cooler's walls and below its rim."""
    return bool(abs(pos[0] - COOLER_XY[0]) < INNER_HALF[0] and abs(pos[1] - COOLER_XY[1]) < INNER_HALF[1]
                and FLOOR_TOP < pos[2] < RIM)


def on_tray(pos) -> bool:
    """A carton's centre over the tray and no higher than a carton standing on it (5 mm slack): standing, or lying on
    any side."""
    return bool(abs(pos[0] - TRAY_XY[0]) < TRAY[0] / 2 and abs(pos[1] - TRAY_XY[1]) < TRAY[1] / 2
                and TRAY_TOP < pos[2] < TRAY_TOP + CARTON[2] / 2 + 0.005)


def off_table(pos) -> bool:
    return bool(pos[2] < TABLE - OFF_TABLE)


# -- demo poses (tool point positions, absolute) -----------------------------------------------------------------------
def handle_grasp(lid_pos, lid_quat) -> np.ndarray:
    """Tool point gripping the handle of a lid at this pose."""
    return np.asarray(lid_pos, float) + rotation(lid_quat) @ np.array([0.0, 0.0, PLATE_T + HANDLE_TCP_DZ])


SEATED_GRASP = np.array([*COOLER_XY, RIM + PLATE_T + HANDLE_TCP_DZ])
LID_LIFT = 0.04                                  # lift off the rim: skirt bottom 2 cm above it
ASIDE_GRASP = np.array([*LID_ASIDE_XY, TABLE + SKIRT_H + PLATE_T + HANDLE_TCP_DZ])
LID_CARRY_Z = SEATED_GRASP[2] + LID_LIFT


def carton_grasp(xy) -> np.ndarray:
    return np.array([xy[0], xy[1], FLOOR_TOP + CARTON[2] - CARTON_TCP_DOWN])


CARTON_CARRY_Z = RIM + 0.02 + CARTON[2] - CARTON_TCP_DOWN          # carton bottom 2 cm above the rim
TRAY_PLACE = np.array([*TRAY_XY, TRAY_TOP + CARTON[2] - CARTON_TCP_DOWN + 0.003])
