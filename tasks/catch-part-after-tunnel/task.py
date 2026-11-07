"""On the station table, a grey carrier plate (12 x 12 x 2 cm) rides on two dark rails running left to right, 32 cm
in front of the arms. It carries an orange aluminium block (5 x 5 x 3 cm). Halfway along, the rails run through a
dark grey tunnel (24 cm long, open at both ends) whose roof hides the block from above; the roof is only 2.3 cm above
the block, too low to reach in. The carrier never stops for the robot: it waits 2 s at its right end (in front of
the right arm), runs 44 cm to its left end at 10 cm/s through the tunnel, waits there 2 s, runs back through the
tunnel, and repeats (a 13.8 s cycle, starting at the right end). The block comes out of the tunnel's left end and
stops in front of the left arm. A green pad (10 x 10 cm, 5 mm thick) lies on the table in front of the left arm,
nearer the arms.

The robot must take the block off the carrier after it comes out of the tunnel and put it on the green pad.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Take the orange block off the carrier when it comes out of the tunnel and put it on the green pad." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, cruise_shuttle, cruise_travel_s  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "best"  # partial credit for the furthest the block got; success scores 1.0

# -- the carrier ---------------------------------------------------------------------------------------------------
LINE_X = 0.32  # the carrier's centre line, metres in front of the arm bases
RIGHT_Y, LEFT_Y = -0.20, 0.24  # its two stops (y is to the robot's left); the left one is in front of the left arm
SPEED, RAMP_S = 0.10, 0.5  # cruising speed (m/s), and the time it takes to speed up or slow down
DWELL_RIGHT_S, DWELL_LEFT_S = 2.0, 2.0
TRAVEL_S = cruise_travel_s(LEFT_Y - RIGHT_Y, SPEED, RAMP_S)  # 4.9 s each way
CYCLE_S = DWELL_RIGHT_S + DWELL_LEFT_S + 2 * TRAVEL_S
RAIL_H = 0.015
PLATE = (0.12, 0.12, 0.02)
PLATE_Z = TOP + RAIL_H + 0.002 + PLATE[2] / 2  # 2 mm above the rails: the plate rides, it doesn't rub
PLATE_TOP = PLATE_Z + PLATE[2] / 2
PATH = cruise_shuttle((LINE_X, RIGHT_Y, PLATE_Z), (LINE_X, LEFT_Y, PLATE_Z), SPEED, RAMP_S, DWELL_RIGHT_S, DWELL_LEFT_S)
AT_LEFT = (DWELL_RIGHT_S + TRAVEL_S, DWELL_RIGHT_S + TRAVEL_S + DWELL_LEFT_S)  # when, in each cycle, it waits left

CARRIER = DrivenBox("carrier", pos=(LINE_X, RIGHT_Y, PLATE_Z), size=PLATE, color=(0.62, 0.64, 0.66), path=PATH)
RAILS = [
    Box(f"rail_{side}", pos=(LINE_X + dx, (RIGHT_Y + LEFT_Y) / 2, TOP + RAIL_H / 2),
        size=(0.015, LEFT_Y - RIGHT_Y + PLATE[1] + 0.04, RAIL_H), color=(0.18, 0.18, 0.2))
    for side, dx in (("near", -0.04), ("far", 0.04))
]

# -- the block -----------------------------------------------------------------------------------------------------
PART = (0.05, 0.05, 0.03)
BLOCK = Box("block", pos=(LINE_X, RIGHT_Y, PLATE_TOP + PART[2] / 2), size=PART, mass=0.2, color=(0.95, 0.45, 0.1))
BLOCK_TOP = BLOCK.pos[2] + PART[2] / 2

# -- the tunnel ----------------------------------------------------------------------------------------------------
TUNNEL_Y = (-0.12, 0.12)  # its right and left (exit) ends
TUNNEL_IN = 0.15  # inside width across the rails: 1.5 cm of side clearance to the carrier plate
TUNNEL_WALL = 0.01
ROOF_LOW = TOP + 0.09  # the roof's underside: 2.3 cm above the block, too low for fingers to reach in
ROOF_TOP = ROOF_LOW + TUNNEL_WALL
TUNNEL_OUT = TUNNEL_IN + 2 * TUNNEL_WALL  # outside width, walls included
TUNNEL_LEN = TUNNEL_Y[1] - TUNNEL_Y[0]
GREY = (0.3, 0.31, 0.33)


def _tunnel() -> list[Box]:
    mid_y = (TUNNEL_Y[0] + TUNNEL_Y[1]) / 2
    out = TUNNEL_IN / 2 + TUNNEL_WALL / 2
    wall_h = ROOF_LOW - TOP
    return [
        Box("tunnel_roof", pos=(LINE_X, mid_y, ROOF_LOW + TUNNEL_WALL / 2), size=(TUNNEL_OUT, TUNNEL_LEN, TUNNEL_WALL),
            color=GREY),
        Box("tunnel_wall_near", pos=(LINE_X - out, mid_y, TOP + wall_h / 2), size=(TUNNEL_WALL, TUNNEL_LEN, wall_h),
            color=GREY),
        Box("tunnel_wall_far", pos=(LINE_X + out, mid_y, TOP + wall_h / 2), size=(TUNNEL_WALL, TUNNEL_LEN, wall_h),
            color=GREY),
    ]


# -- the pad -------------------------------------------------------------------------------------------------------
PAD_XY = (0.15, 0.33)
PAD_SIDE, PAD_H = 0.10, 0.005
PAD_TOP = TOP + PAD_H
PAD = Box("pad", pos=(*PAD_XY, TOP + PAD_H / 2), size=(PAD_SIDE, PAD_SIDE, PAD_H), color=(0.2, 0.7, 0.3))

OBJECTS = [*RAILS, CARRIER, BLOCK, *_tunnel(), PAD]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_M = 0.03  # the block's underside this far above the carrier's top counts as taken off it


def _over_pad(block) -> bool:
    """The block's centre is over the pad."""
    dx, dy = block.pos[0] - PAD_XY[0], block.pos[1] - PAD_XY[1]
    return bool(abs(dx) < PAD_SIDE / 2 and abs(dy) < PAD_SIDE / 2)


def success(objects):
    """On the pad: centre over it, underside within 1 cm of its top, and still."""
    block = objects["block"]
    return bool(_over_pad(block) and block.low[2] < PAD_TOP + 0.01 and block.resting())


def distance(objects):
    """Metres from the block's centre to where it sits on the middle of the pad."""
    goal = np.array([*PAD_XY, PAD_TOP + PART[2] / 2])
    return float(np.linalg.norm(objects["block"].pos - goal))


def score(objects):
    """Partial credit, kept at its best over the trial: 0.5 once the block's centre is over the pad with its underside
    less than 5 cm above the pad's top (carried there); 0.25 once its underside is 3 cm or more above the carrier's
    top (taken off)."""
    block = objects["block"]
    if _over_pad(block) and block.low[2] < PAD_TOP + 0.05:
        return 0.5
    if block.low[2] >= PLATE_TOP + LIFT_M:
        return 0.25
    return 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
TRAVEL_Z = TOP + 0.13
GRASP_Z = BLOCK.pos[2]  # TCP at the block's centre: the pad tips stop 2 mm above the carrier's top
PRE_Z = GRASP_Z + 0.06  # waiting height: the pad tips clear the arriving block by about 3 cm


def demo():
    """Left arm: wait above the carrier's left stop, where the block comes to rest 1.2 s after leaving the tunnel,
    and grab it during the first wait there (6.9 to 8.9 s): down, close and up in 1.9 s, clear before the carrier
    heads back into the tunnel. Then carry it to the pad and let go."""
    x, y = LINE_X, LEFT_Y
    px, py = PAD_XY
    place_z = PAD_TOP + PART[2] / 2 + 0.008
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
        Waypoint(1.5, (px, py, TRAVEL_Z), FINGER_CLOSED, "carry"),
        Waypoint(1.2, (px, py, place_z), FINGER_CLOSED, "lower"),
        Waypoint(0.6, (px, py, place_z), FINGER_OPEN, "release"),
        Waypoint(1.0, (px, py, TRAVEL_Z), FINGER_OPEN, "retreat"),
    ]
    return {"left": plan}
