"""On the station table, a grey shuttle plate (12 x 12 x 2 cm) rides on two dark rails running left to right, 34 cm
in front of the arms. It carries an orange aluminium block (5 x 5 x 3 cm). The shuttle never stops for the robot:
it waits 2.5 s at its right end, slides 30 cm to its left end in 2.5 s, waits there 2.5 s, slides back in 2.5 s,
and repeats (a 10 s cycle, starting at the right end). At the left end the block is in front of the left arm.
A blue open tray (14 x 14 cm, 4 cm walls) stands on the table in front of the left arm, nearer the arms.

The robot must take the block off the shuttle and put it in the tray, without the shuttle stopping.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Take the orange block off the moving shuttle and put it in the blue tray." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, dwell_shuttle  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "best"  # partial credit for the furthest the block got; success scores 1.0

# -- the shuttle ---------------------------------------------------------------------------------------------------
LINE_X = 0.34  # the shuttle's centre line, metres in front of the arm bases
RIGHT_Y, LEFT_Y = -0.12, 0.18  # its two stops (y is to the robot's left)
DWELL_S, TRAVEL_S = 2.5, 2.5
CYCLE_S = 2 * (DWELL_S + TRAVEL_S)
RAIL_H = 0.015
PLATE = (0.12, 0.12, 0.02)
PLATE_Z = TOP + RAIL_H + 0.002 + PLATE[2] / 2  # 2 mm above the rails: the plate rides, it doesn't rub
PLATE_TOP = PLATE_Z + PLATE[2] / 2
PATH = dwell_shuttle((LINE_X, RIGHT_Y, PLATE_Z), (LINE_X, LEFT_Y, PLATE_Z), TRAVEL_S, DWELL_S)
AT_LEFT = (DWELL_S + TRAVEL_S, 2 * DWELL_S + TRAVEL_S)  # when, in each cycle, the shuttle waits at its left end

SHUTTLE = DrivenBox("shuttle", pos=(LINE_X, RIGHT_Y, PLATE_Z), size=PLATE, color=(0.62, 0.64, 0.66), path=PATH)
RAILS = [
    Box(f"rail_{side}", pos=(LINE_X + dx, (RIGHT_Y + LEFT_Y) / 2, TOP + RAIL_H / 2),
        size=(0.015, LEFT_Y - RIGHT_Y + PLATE[1] + 0.04, RAIL_H), color=(0.18, 0.18, 0.2))
    for side, dx in (("near", -0.04), ("far", 0.04))
]

# -- the block -----------------------------------------------------------------------------------------------------
PART = (0.05, 0.05, 0.03)
BLOCK = Box("block", pos=(LINE_X, RIGHT_Y, PLATE_TOP + PART[2] / 2), size=PART, mass=0.2, color=(0.95, 0.45, 0.1))

# -- the tray ------------------------------------------------------------------------------------------------------
TRAY_XY = (0.17, 0.36)
TRAY_IN = 0.12  # inner width and depth
WALL, FLOOR, WALL_H = 0.01, 0.01, 0.04
TRAY_FLOOR_TOP = TOP + FLOOR
BLUE = (0.15, 0.32, 0.8)


def _tray() -> list[Box]:
    x, y = TRAY_XY
    out = TRAY_IN / 2 + WALL / 2
    span = TRAY_IN + 2 * WALL
    z = TOP + WALL_H / 2
    return [
        Box("tray_floor", pos=(x, y, TOP + FLOOR / 2), size=(span, span, FLOOR), color=BLUE),
        Box("tray_wall_near", pos=(x - out, y, z), size=(WALL, span, WALL_H), color=BLUE),
        Box("tray_wall_far", pos=(x + out, y, z), size=(WALL, span, WALL_H), color=BLUE),
        Box("tray_wall_right", pos=(x, y - out, z), size=(TRAY_IN, WALL, WALL_H), color=BLUE),
        Box("tray_wall_left", pos=(x, y + out, z), size=(TRAY_IN, WALL, WALL_H), color=BLUE),
    ]


OBJECTS = [*RAILS, SHUTTLE, BLOCK, *_tray()]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_M = 0.03  # the block's underside this far above the shuttle's top counts as taken off it


def _over_tray(block) -> bool:
    """The block's centre is over the tray's inside (between the walls)."""
    dx, dy = block.pos[0] - TRAY_XY[0], block.pos[1] - TRAY_XY[1]
    return bool(abs(dx) < TRAY_IN / 2 and abs(dy) < TRAY_IN / 2)


def success(objects):
    """In the tray: centre between the walls, underside within 1 cm of the tray floor, and still."""
    block = objects["block"]
    return bool(_over_tray(block) and block.low[2] < TRAY_FLOOR_TOP + 0.01 and block.resting())


def distance(objects):
    """Metres from the block's centre to where it sits on the middle of the tray floor."""
    goal = np.array([*TRAY_XY, TRAY_FLOOR_TOP + PART[2] / 2])
    return float(np.linalg.norm(objects["block"].pos - goal))


def score(objects):
    """Partial credit, kept at its best over the trial: 0.5 once the block is over the tray's inside below its rim
    plus 5 cm (carried there); 0.25 once its underside is 3 cm or more above the shuttle's top (taken off)."""
    block = objects["block"]
    if _over_tray(block) and block.low[2] < TOP + WALL_H + 0.05:
        return 0.5
    if block.low[2] >= PLATE_TOP + LIFT_M:
        return 0.25
    return 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
TRAVEL_Z = TOP + 0.13
GRASP_Z = BLOCK.pos[2] - 0.003  # TCP a little below the block's centre
PRE_Z = GRASP_Z + 0.06  # waiting height: the pad tips clear the arriving block by about 3 cm


def demo():
    """Left arm: wait above the shuttle's left stop, and grab the block during the first wait there (5.0 to 7.5 s):
    down, close and up in 1.9 s, clear before the shuttle leaves. Then carry it to the tray and let go."""
    x, y = LINE_X, LEFT_Y
    tx, ty = TRAY_XY
    place_z = TRAY_FLOOR_TOP + PART[2] / 2 + 0.008
    start = AT_LEFT[0] + 0.15  # the block has stopped
    plan = [
        Waypoint(2.0, (x, y, TRAVEL_Z), FINGER_OPEN, "above the left stop"),
        Waypoint(1.5, (x, y, PRE_Z), FINGER_OPEN, "wait height"),
    ]
    plan.append(Waypoint(start - sum(w.duration for w in plan), (x, y, PRE_Z), FINGER_OPEN, "wait for the block"))
    plan += [
        Waypoint(0.6, (x, y, GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.6, (x, y, GRASP_Z), FINGER_CLOSED, "close"),
        Waypoint(0.7, (x, y, TRAVEL_Z), FINGER_CLOSED, "lift"),
        Waypoint(1.8, (tx, ty, TRAVEL_Z), FINGER_CLOSED, "carry"),
        Waypoint(1.2, (tx, ty, place_z), FINGER_CLOSED, "lower"),
        Waypoint(0.6, (tx, ty, place_z), FINGER_OPEN, "release"),
        Waypoint(1.0, (tx, ty, TRAVEL_Z), FINGER_OPEN, "retreat"),
    ]
    return {"left": plan}
