"""On the station table lies a grey ball-transfer deck (35 x 78 x 3 cm), parallel to the line through the arm bases,
from 7 to 42 cm in front of them, running from 62 cm to the right of the midpoint between the bases to 16 cm to its
left, where it ends in a 3 cm drop to the table. Its top is almost frictionless (friction 0.01). A green square
(16 x 16 cm) is marked on it, its centre 17 cm in front of the bases and 6 cm toward the left arm.

A brown cardboard carton (10 x 10 x 6 cm, 150 g) starts on the deck's right end, 31 cm in front of the bases, and is
already sliding toward the left at 0.42 m/s when the episode starts. Untouched, it passes beside the green square
1.5 s later and slides off the deck's left end after 2.3 s.

The robot must stop the carton before it slides off and push it into the green square, so that it ends at rest
standing upright, wholly inside it. The carton is wider than the open gripper; it is meant to be pushed.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Stop the sliding carton before it falls off the deck and push it into the green square." ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import LaunchedBox, Marker, SlickBox  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # where the carton is when the trial ends: a carton stopped and then pushed off scores 0

# -- the deck ------------------------------------------------------------------------------------------------------
DECK_H = 0.03
DECK_TOP = TOP + DECK_H
DECK_X = (0.07, 0.42)  # near and far edges, metres in front of the arm bases
DECK_Y = (-0.62, 0.16)  # right and left ends (y is to the robot's left); the left end is the open drop
MU = 0.01  # the deck's friction, static and dynamic: ball transfers under a light carton
GREY, GREEN = (0.55, 0.57, 0.6), (0.25, 0.7, 0.35)

DECK = SlickBox("deck", pos=(sum(DECK_X) / 2, sum(DECK_Y) / 2, TOP + DECK_H / 2),
                size=(DECK_X[1] - DECK_X[0], DECK_Y[1] - DECK_Y[0], DECK_H), color=GREY, friction=(MU, MU))

# -- the green square: paint on the deck, seen but not touched ------------------------------------------------------
# (one deck with a visual-only square, not a deck tiled around the square: in Isaac Sim the carton, pushed across
# the seam between two flush tiles, caught on the next tile's edge and tipped over it)
ZONE_XY = (0.17, 0.06)
ZONE = 0.16  # side
ZONE_LO = np.array(ZONE_XY) - ZONE / 2
ZONE_HI = np.array(ZONE_XY) + ZONE / 2
PAINT = 0.001  # the square's thickness above the deck top (it has no collider)
SQUARE = Marker("goal_square", pos=(*ZONE_XY, DECK_TOP + PAINT / 2), size=(ZONE, ZONE, PAINT), color=GREEN)

# -- the carton ----------------------------------------------------------------------------------------------------
CARTON = (0.10, 0.10, 0.06)
LANE_X = 0.31  # the line it slides along
START_Y = -0.54
V0 = 0.42  # m/s, toward the left (+y)
CARTON_SPEC = LaunchedBox("carton", pos=(LANE_X, START_Y, DECK_TOP + CARTON[2] / 2), size=CARTON, mass=0.15,
                          color=(0.62, 0.45, 0.28), velocity=(0.0, V0, 0.0), friction=(0.5, 0.4))

OBJECTS = [DECK, SQUARE, CARTON_SPEC]

# -- scoring -------------------------------------------------------------------------------------------------------
MAX_TILT_DEG = 15.0


def _on_deck(c) -> bool:
    """Lying on the deck: centre over it, underside within 1 cm of its top (not fallen off, not held up)."""
    over = DECK_X[0] < c.pos[0] < DECK_X[1] and DECK_Y[0] < c.pos[1] < DECK_Y[1]
    return bool(over and abs(c.low[2] - DECK_TOP) < 0.01)


def _upright(c) -> bool:
    """Its top face points within 15 degrees of straight up."""
    x, y, _, _ = c.quat_xyzw
    return bool(1 - 2 * (x * x + y * y) > math.cos(math.radians(MAX_TILT_DEG)))


def _centre_in_square(c) -> bool:
    return bool(np.all(c.pos[:2] > ZONE_LO) and np.all(c.pos[:2] < ZONE_HI))


def _wholly_in_square(c) -> bool:
    """Its whole footprint (the bounding box seen from above) inside the green square."""
    return bool(np.all(c.low[:2] >= ZONE_LO) and np.all(c.high[:2] <= ZONE_HI))


def success(objects):
    """Upright on the deck, its footprint wholly inside the green square, and still."""
    c = objects["carton"]
    return bool(_on_deck(c) and _upright(c) and _wholly_in_square(c) and c.resting())


def distance(objects):
    """Metres from the carton's centre to the centre of the green square (horizontally)."""
    return float(np.linalg.norm(objects["carton"].pos[:2] - np.array(ZONE_XY)))


def score(objects):
    """Partial credit at the end of the trial: 0.5 if the carton stands upright on the deck with its centre inside the
    green square (but sticks out of it, or is still moving); 0.25 if it stands upright on the deck anywhere else;
    0 if it fell off the deck, tipped over or is held up."""
    c = objects["carton"]
    if not (_on_deck(c) and _upright(c)):
        return 0.0
    return 0.5 if _centre_in_square(c) else 0.25


# -- the scripted demo ---------------------------------------------------------------------------------------------
FINGER_HALF = 0.018  # from the fingers' centre line to where they touch the carton (measured in Isaac Sim)
STOP_Y = ZONE_XY[1]  # where the demo stops the carton: in line with the square
TIP_Z = DECK_TOP + PAD_TIP_PAST_TCP + 0.006  # TCP height with the pad tips 6 mm above the deck
CLEAR_Z = DECK_TOP + CARTON[2] + PAD_TIP_PAST_TCP + 0.02  # pad tips 2 cm above the carton's top
BLOCK = (LANE_X, STOP_Y + CARTON[1] / 2 + FINGER_HALF, TIP_Z)  # closed fingers standing in the carton's path
PUSH_FROM_X = LANE_X + CARTON[0] / 2 + FINGER_HALF + 0.017  # 1.7 cm beyond the carton's far side
PUSH_TO_X = ZONE_XY[0] + CARTON[0] / 2 + FINGER_HALF  # leaves the carton's centre on the square's centre line
TO_BLOCK_S, PUSH_S = 1.1, 4.0


def arrival_s() -> float:
    """When the carton, sliding untouched, reaches the demo's fingers (its leading face at their face)."""
    gap = STOP_Y - START_Y
    a = MU * 9.81
    return (V0 - math.sqrt(V0 * V0 - 2 * a * gap)) / a


def demo():
    """Left arm: stand the closed fingers in the carton's path, in line with the green square, well before it arrives
    (fingers closing along y, so the carton meets a finger's flat side), and hold them there while it runs into them
    and stops. Then lift over it, turn the fingers a quarter turn, come down behind its far side and push it slowly
    toward the arms (slower to stop than friction alone would stop it, so it doesn't run on) onto the square."""
    bx, by, bz = BLOCK
    y = STOP_Y
    quarter = -math.pi / 2  # (the other way round runs joint 6 into its stop)
    plan = [Waypoint(TO_BLOCK_S, BLOCK, FINGER_CLOSED, "stop in its path")]
    hold = arrival_s() + 0.8 - TO_BLOCK_S
    plan += [
        Waypoint(hold, BLOCK, FINGER_CLOSED, "hold while it runs in"),
        Waypoint(0.6, (bx, by, CLEAR_Z), FINGER_CLOSED, "lift over it"),
        Waypoint(1.0, (PUSH_FROM_X, y, CLEAR_Z), FINGER_CLOSED, "behind its far side", quarter),
        Waypoint(0.8, (PUSH_FROM_X, y, TIP_Z), FINGER_CLOSED, "down", quarter),
        Waypoint(PUSH_S, (PUSH_TO_X, y, TIP_Z), FINGER_CLOSED, "push onto the square", quarter),
        Waypoint(0.4, (PUSH_TO_X + 0.01, y, TIP_Z), FINGER_CLOSED, "back off", quarter),
        Waypoint(0.8, (PUSH_TO_X + 0.01, y, CLEAR_Z), FINGER_CLOSED, "up and away", quarter),
    ]
    return {"left": plan}

