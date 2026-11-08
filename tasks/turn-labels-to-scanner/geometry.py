"""Dimensions, layout and geometric tests for the turn-labels-to-scanner task. NumPy only, no Isaac Sim.

Station frame: x forward from the arms, y to the robot's left, z up, metres. A carton's own frame has its origin at
the box's centre, +z up and its shipping label on its +x face; so with yaw 0 (and upright) the label faces +x, away
from the robot, toward the scanner on the far side of the line.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # yam_arena.task_file.TABLE_TOP["station"]
TABLE_LEFT_Y = 0.65  # the station table's left edge (yam_arena.station: TABLE_CENTER y + TABLE_SIZE y / 2)

# -- the cartons: small closed corrugated-cardboard boxes, a shipping label on one side ------------------------------
SIZE = (0.06, 0.06, 0.08)   # x, y (the square cross-section), z (height)
HALF = np.array(SIZE) / 2
MASS = 0.15                 # kg, evenly spread (the contents are packed tight)
DECAL = 0.0005              # printed marks: this thick, proud of the faces, visual only
LABEL = (0.05, 0.05)        # the white label on the +x face: width (across, y), height (z)
BAND = 0.012                # the orange header band above the label on the side face ...
BAND_TOP = 0.015            # ... folded over the top edge, this far onto the top face
BAR_MARGIN = 0.006          # barcode bars stop this far short of the label's top and bottom
# barcode bars: (left edge y, width) on the label, carton frame
BARS = ((-0.020, 0.003), (-0.015, 0.0015), (-0.011, 0.004), (-0.004, 0.0015), (0.000, 0.003),
        (0.006, 0.0015), (0.010, 0.004), (0.017, 0.002))

# -- the conveyor line -----------------------------------------------------------------------------------------------
LINE_X = 0.34               # the line's centre, metres in front of the arm bases
BED = (0.12, 1.24, 0.015)   # the fixed track bed: width (x), length (y), height; 3 cm inside each table edge
CARRIER = (0.10, 0.10, 0.015)
CARRIER_Z = TOP + BED[2] + 0.002 + CARRIER[2] / 2   # carriers ride 2 mm above the bed
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
CARTON_Z = CARRIER_TOP + SIZE[2] / 2                # a carton's centre, standing on its carrier
SPEED = 0.04                # m/s, toward +y (from the right arm toward the left arm)
RAMP_S = 0.5                # the line speeds up from rest over this long at the start of the trial
SPACING = 0.22              # between carriers: one every SPACING / SPEED = 5.5 s
FIRST_Y = -0.12             # the first carrier's centre at the start
TURN_Y = TABLE_LEFT_Y + CARRIER[1] / 2 + 0.02       # past the table's edge, carriers turn down out of the way
DROP, DROP_SPEED = 0.30, 0.15

# Each carton's yaw at the start (rad): its label faces the robot's right (upstream), the robot, and the robot's
# left (downstream). None faces the scanner: they need a quarter turn one way, a half turn, a quarter turn the other.
START_YAWS = (-math.pi / 2, math.pi, math.pi / 2)
NAMES = ("carton_1", "carton_2", "carton_3")        # in arrival order (carton_1 leads)
STARTS = {n: FIRST_Y - i * SPACING for i, n in enumerate(NAMES)}

# -- the scanner gate ------------------------------------------------------------------------------------------------
GATE_Y = 0.50               # the scanner reads a carton as its carrier's centre passes this line
POST = (0.03, 0.03, 0.20)   # two dark grey posts either side of the line
POST_X = 0.115              # posts' centres this far either side of the line's centre (inner faces 10 cm out)
BEAM = (2 * POST_X + POST[0], 0.03, 0.03)          # the crossbar on top of the posts
SCANNER = (0.02, 0.07, 0.06)                        # the black scanner head on the far post, facing the line
SCANNER_Z = CARTON_Z        # its window centred on the cartons' height
WINDOW = (0.002, 0.05, 0.02)                        # red window on its face toward the line

# -- scoring thresholds ----------------------------------------------------------------------------------------------
LABEL_TOL = math.radians(20.0)    # read: the label's facing within this of straight at the scanner (+x)
UPRIGHT = math.radians(10.0)      # the carton's up axis within this of straight up
ON_CARRIER_XY = 0.03              # its centre within this (each of x and y) of its carrier's centre ...
SEAT = 0.01                       # ... and its underside within this of the carrier's top
HAND_CLEAR = 0.08                 # both grippers' grasp points at least this far from the carton's centre
TABLE_TOUCH = 0.01                # a carton whose lowest corner comes this close to the table top is off its carrier


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def lowest_z(pos, R) -> float:
    """The carton's lowest corner (it is a box), station frame."""
    return float(pos[2] - np.abs(R[2]) @ HALF)


def label_error(R) -> float:
    """Radians between the label's facing (the carton's +x axis, seen from above) and straight at the scanner (+x).
    pi if the label faces straight up or down."""
    n = R[:, 0]
    if math.hypot(n[0], n[1]) < 1e-6:
        return math.pi
    return abs(math.atan2(n[1], n[0]))


def tilt(R) -> float:
    """Radians between the carton's up axis and straight up (pi when upside down)."""
    return math.acos(max(-1.0, min(1.0, float(R[2, 2]))))


def check(pos, R, carrier_pos, grasp_points) -> dict:
    """The scanner's read of one carton as its carrier's centre passes the gate. ``read`` is True when it stands
    upright on its own carrier, untouched by the hands, its label within LABEL_TOL of facing the scanner."""
    pos, carrier_pos = np.asarray(pos, float), np.asarray(carrier_pos, float)
    off = pos[:2] - carrier_pos[:2]
    seated = abs(lowest_z(pos, R) - (carrier_pos[2] + CARRIER[2] / 2)) <= SEAT
    on = bool(np.all(np.abs(off) <= ON_CARRIER_XY) and seated)
    upright = tilt(R) <= UPRIGHT
    clear = all(np.linalg.norm(np.asarray(p, float) - pos) >= HAND_CLEAR for p in grasp_points)
    err = label_error(R)
    return dict(on_carrier=on, upright=bool(upright), hands_clear=bool(clear), label_deg=math.degrees(err),
                read=bool(on and upright and clear and err <= LABEL_TOL))
