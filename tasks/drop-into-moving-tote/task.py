"""On the station table, an orange aluminium block (5 x 5 x 3 cm) lies 20 cm in front of the left arm. A blue open
tote (14 x 14 cm inside, 6 cm tall) glides back and forth across the front of both arms, 34 cm out, never pausing:
it starts at rest 20 cm toward the right arm from the midpoint between the bases, runs 50 cm to its left end (in
front of the left arm) at 10 cm/s, easing in and out of each end over 0.8 s, turns straight back, and repeats
(5.8 s each way, an 11.6 s cycle).

The robot must put the block in the moving tote, without the tote stopping.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Put the orange block in the blue tote. The tote keeps moving and won't stop for you." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, back_and_forth, offset_path  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "best"  # partial credit for the furthest the block got; success scores 1.0

# -- the tote ------------------------------------------------------------------------------------------------------
LINE_X = 0.34  # the tote's centre line, metres in front of the arm bases
RIGHT_Y, LEFT_Y = -0.20, 0.30  # its two ends (y is to the robot's left)
SPEED, RAMP_S = 0.10, 0.8  # cruise speed (m/s) and the time to ease in or out of each end
LEG_S = (LEFT_Y - RIGHT_Y) / SPEED + RAMP_S  # one way: 5.8 s
CYCLE_S = 2 * LEG_S
TOTE_IN = 0.14  # inner width and depth
WALL, FLOOR, WALL_H, GAP = 0.01, 0.01, 0.06, 0.001  # the 1 mm gaps keep the tote's driven parts from touching
TOTE_LOW = TOP + 0.002  # it glides 2 mm above the table: driven, it doesn't rub
FLOOR_TOP = TOTE_LOW + FLOOR
RIM = TOTE_LOW + WALL_H
FLOOR_Z = TOTE_LOW + FLOOR / 2
PATH = back_and_forth((LINE_X, RIGHT_Y, FLOOR_Z), (LINE_X, LEFT_Y, FLOOR_Z), SPEED, RAMP_S)  # the floor's centre
BLUE = (0.15, 0.32, 0.8)


def _tote() -> list[DrivenBox]:
    """Floor and four walls, each a driven box following PATH with its own offset (all move as one)."""
    out = TOTE_IN / 2 + GAP + WALL / 2
    span = TOTE_IN + 2 * (GAP + WALL)
    dz = WALL_H / 2 - FLOOR / 2  # a wall's centre above the floor's
    parts = [
        ("tote_floor", (0.0, 0.0, 0.0), (TOTE_IN, TOTE_IN, FLOOR)),
        ("tote_wall_near", (-out, 0.0, dz), (WALL, span, WALL_H)),
        ("tote_wall_far", (out, 0.0, dz), (WALL, span, WALL_H)),
        ("tote_wall_right", (0.0, -out, dz), (TOTE_IN, WALL, WALL_H)),
        ("tote_wall_left", (0.0, out, dz), (TOTE_IN, WALL, WALL_H)),
    ]
    boxes = []
    for name, off, size in parts:
        path = offset_path(PATH, off)
        boxes.append(DrivenBox(name, pos=tuple(path(0.0)[0]), size=size, color=BLUE, path=path))
    return boxes


TOTE = _tote()

# -- the block -----------------------------------------------------------------------------------------------------
PART = (0.05, 0.05, 0.03)
BLOCK_XY = (0.20, 0.305)
BLOCK = Box("block", pos=(*BLOCK_XY, TOP + PART[2] / 2), size=PART, mass=0.2, color=(0.95, 0.45, 0.1))

OBJECTS = [BLOCK, *TOTE]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_M = 0.03  # the block's underside this far above the table counts as picked up
ABOVE_RIM_M = 0.05


def _in_tote(block, tote) -> bool:
    """The block's centre is between the tote's walls, wherever the tote is now."""
    dx, dy = block.pos[0] - tote.pos[0], block.pos[1] - tote.pos[1]
    return bool(abs(dx) < TOTE_IN / 2 and abs(dy) < TOTE_IN / 2)


def success(objects):
    """In the tote: centre between the walls, underside within 1 cm of the tote floor, and riding with the tote
    (velocity within 1 cm/s of the tote's, barely turning)."""
    block, tote = objects["block"], objects["tote_floor"]
    riding = np.linalg.norm(block.lin_vel - tote.lin_vel) < 0.01 and np.linalg.norm(block.ang_vel) < 0.5
    return bool(_in_tote(block, tote) and block.low[2] < tote.high[2] + 0.01 and riding)


def distance(objects):
    """Metres from the block's centre to where it would sit on the middle of the tote floor, wherever the tote is."""
    tote = objects["tote_floor"]
    goal = np.array([tote.pos[0], tote.pos[1], tote.high[2] + PART[2] / 2])
    return float(np.linalg.norm(objects["block"].pos - goal))


def score(objects):
    """Partial credit, kept at its best over the trial: 0.5 once the block's centre is between the tote's walls with
    its underside less than 5 cm above the rim (brought to the tote); 0.25 once its underside is 3 cm or more above
    the table (picked up)."""
    block, tote = objects["block"], objects["tote_floor"]
    if _in_tote(block, tote) and block.low[2] < RIM + ABOVE_RIM_M:
        return 0.5
    if block.low[2] >= TOP + LIFT_M:
        return 0.25
    return 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
TRAVEL_Z = TOP + 0.14  # the held block's underside clears the rim by about 5 cm
GRASP_Z = BLOCK.pos[2] + 0.002  # TCP a little above the block's centre: the pad tips stay 4 mm off the table
RELEASE_Z = RIM + 0.02 + PART[2] / 2 + 0.002  # the held block's underside 2 cm above the rim
MEET_S = LEG_S + 0.2  # the tote has just turned at its left end and heads right
STEP_S = 0.2  # following waypoints: the TCP is over the tote at the end of each


def _over_tote(t: float, z: float) -> tuple[float, float, float]:
    x, y, _ = PATH(t)[0]
    return (float(x), float(y), z)


def demo():
    """Left arm: pick the block, carry it over the tote just after it turns at its left end (6.0 s), then follow the
    tote in 0.2 s steps: lower to 2 cm above the rim, let go (7.0 to 7.4 s), rise, and retreat."""
    bx, by = BLOCK_XY
    plan = [
        Waypoint(1.5, (bx, by, TRAVEL_Z), FINGER_OPEN, "above the block"),
        Waypoint(1.0, (bx, by, GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.6, (bx, by, GRASP_Z), FINGER_CLOSED, "close"),
        Waypoint(0.8, (bx, by, TRAVEL_Z), FINGER_CLOSED, "lift"),
    ]
    plan.append(Waypoint(MEET_S - sum(w.duration for w in plan), _over_tote(MEET_S, TRAVEL_Z), FINGER_CLOSED, "carry"))
    t = MEET_S
    follow = [("lower", FINGER_CLOSED, TRAVEL_Z + (RELEASE_Z - TRAVEL_Z) * k / 5) for k in range(1, 6)]
    follow += [("release", FINGER_OPEN, RELEASE_Z)] * 2
    follow += [("rise", FINGER_OPEN, RELEASE_Z + (TRAVEL_Z - RELEASE_Z) * k / 3) for k in range(1, 4)]
    for label, finger, z in follow:
        t += STEP_S
        plan.append(Waypoint(STEP_S, _over_tote(t, z), finger, label))
    plan.append(Waypoint(1.0, (bx, by, TRAVEL_Z), FINGER_OPEN, "retreat"))
    return {"left": plan}
