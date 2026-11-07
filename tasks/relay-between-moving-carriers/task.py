"""Two shuttle carriers, one on each side of the station, run back and forth toward and away from the arms on their
own rails, and neither ever stops for the robot. Carrier A (grey, 12 x 12 x 2 cm), outside the right arm, carries an
orange aluminium block (5 x 5 x 3 cm): it waits 2 s at its far stop, comes 16 cm toward the arms in 2.5 s, waits 2 s
at its near stop, goes back, and repeats (a 9 s cycle). Carrier B (grey, with a green nest: four 5 mm lips around a
7 x 7 cm pocket), outside the left arm, runs the same way on a 12 s cycle (3 s waits, 3 s moves). A fixed yellow
handover pad stands on the table halfway between the arms.

Neither arm reaches both carriers, so the block has to go from the right arm to the left arm: the right arm takes it
off A, sets it on the pad, and the left arm takes it from there and puts it in B's nest while B waits.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Move the orange block from the right carrier into the green nest on the left carrier." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, dwell_shuttle  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "best"  # partial credit for the furthest the block got; success scores 1.0

# -- the two carriers ----------------------------------------------------------------------------------------------
# WHY: the lines sit 48 cm out from the midline, 17.5 cm outside each arm's base. In any wrist orientation an arm's TCP
# reaches at most about 0.78 m across the table from its own base (sampled, see the tests); the far line is 0.82 m or
# more away, so neither arm reaches the other's carrier and the relay is forced.
A_Y, B_Y = -0.48, 0.48  # each carrier's centre line (y is to the robot's left)
NEAR_X, FAR_X = 0.24, 0.40  # both carriers' stops, metres in front of the arm bases
A_DWELL_S, A_TRAVEL_S = 2.0, 2.5  # carrier A: a 9 s cycle
B_DWELL_S, B_TRAVEL_S = 3.0, 3.0  # carrier B: a 12 s cycle
A_CYCLE_S = 2 * (A_DWELL_S + A_TRAVEL_S)
B_CYCLE_S = 2 * (B_DWELL_S + B_TRAVEL_S)
RAIL_H = 0.015
PLATE = (0.12, 0.12, 0.02)
PLATE_Z = TOP + RAIL_H + 0.002 + PLATE[2] / 2  # 2 mm above the rails: the plates ride, they don't rub
PLATE_TOP = PLATE_Z + PLATE[2] / 2
GREY, RAIL_GREY = (0.62, 0.64, 0.66), (0.18, 0.18, 0.2)


def _carrier_path(y: float, dz: float, dwell_s: float, travel_s: float, dx: float = 0.0, dy: float = 0.0):
    """Both carriers start at their far stop: wait there, come in to the near stop, wait, go back out."""
    return dwell_shuttle((FAR_X + dx, y + dy, PLATE_Z + dz), (NEAR_X + dx, y + dy, PLATE_Z + dz), travel_s, dwell_s)


PATH_A = _carrier_path(A_Y, 0.0, A_DWELL_S, A_TRAVEL_S)
PATH_B = _carrier_path(B_Y, 0.0, B_DWELL_S, B_TRAVEL_S)
A_AT_NEAR = (A_DWELL_S + A_TRAVEL_S, 2 * A_DWELL_S + A_TRAVEL_S)  # when, in each cycle, A waits at its near stop
B_AT_NEAR = (B_DWELL_S + B_TRAVEL_S, 2 * B_DWELL_S + B_TRAVEL_S)

CARRIER_A = DrivenBox("carrier_a", pos=(FAR_X, A_Y, PLATE_Z), size=PLATE, color=GREY, path=PATH_A)
CARRIER_B = DrivenBox("carrier_b", pos=(FAR_X, B_Y, PLATE_Z), size=PLATE, color=GREY, path=PATH_B)


def _rails(tag: str, y: float) -> list[Box]:
    length = FAR_X - NEAR_X + PLATE[0] + 0.04
    return [
        Box(f"rail_{tag}_{side}", pos=((NEAR_X + FAR_X) / 2, y + dy, TOP + RAIL_H / 2), size=(length, 0.015, RAIL_H),
            color=RAIL_GREY)
        for side, dy in (("in", 0.04 if y < 0 else -0.04), ("out", -0.04 if y < 0 else 0.04))
    ]


# -- carrier B's nest: four low lips riding on B ---------------------------------------------------------------------
NEST_IN = 0.07  # the pocket's inner width and depth
LIP_W, LIP_H = 0.01, 0.005
LIP_GAP = 0.001  # the lips ride 1 mm above B's top, like B above its rails: driven boxes must not touch
GREEN = (0.15, 0.6, 0.25)


def _nest() -> list[DrivenBox]:
    out = NEST_IN / 2 + LIP_W / 2
    span = NEST_IN + 2 * LIP_W
    dz = PLATE[2] / 2 + LIP_GAP + LIP_H / 2
    lips = (("near", -out, 0.0, (LIP_W, span, LIP_H)), ("far", out, 0.0, (LIP_W, span, LIP_H)),
            ("right", 0.0, -out, (NEST_IN, LIP_W, LIP_H)), ("left", 0.0, out, (NEST_IN, LIP_W, LIP_H)))
    return [
        DrivenBox(f"nest_lip_{side}", pos=(FAR_X + dx, B_Y + dy, PLATE_Z + dz), size=size, color=GREEN,
                  path=_carrier_path(B_Y, dz, B_DWELL_S, B_TRAVEL_S, dx, dy))
        for side, dx, dy, size in lips
    ]


NEST = _nest()

# -- the handover pad ----------------------------------------------------------------------------------------------
PAD_XY = (0.30, 0.0)
PAD = (0.10, 0.10, 0.02)
PAD_TOP = TOP + PAD[2]
HANDOVER_PAD = Box("handover_pad", pos=(*PAD_XY, TOP + PAD[2] / 2), size=PAD, color=(0.9, 0.75, 0.15))

# -- the block -----------------------------------------------------------------------------------------------------
PART = (0.05, 0.05, 0.03)
BLOCK = Box("block", pos=(FAR_X, A_Y, PLATE_TOP + PART[2] / 2), size=PART, mass=0.2, color=(0.95, 0.45, 0.1))

OBJECTS = [*_rails("a", A_Y), *_rails("b", B_Y), CARRIER_A, CARRIER_B, *NEST, HANDOVER_PAD, BLOCK]

# -- scoring -------------------------------------------------------------------------------------------------------
NEST_TOL = (NEST_IN - PART[0]) / 2  # 1 cm: the block's centre this close to the nest's centre lies between the lips
LIFT_M = 0.03  # the block's underside this far above a surface counts as lifted off it
RIGHT_BASE_XY = (0.0, -0.305)  # yam_arena's station: the right arm's base
RIGHT_OUT_OF_REACH = 0.81  # m across the table from the right arm's base: 3 cm past where its TCP reaches (the tests)


def _in_nest(block, carrier) -> bool:
    """Between the lips (centre within 1 cm of B's centre in x and y), underside within 1 cm of B's top."""
    dx, dy = block.pos[0] - carrier.pos[0], block.pos[1] - carrier.pos[1]
    gap = block.low[2] - carrier.high[2]
    return bool(abs(dx) <= NEST_TOL and abs(dy) <= NEST_TOL and -0.005 < gap < 0.01)


def success(objects):
    """In B's nest and riding with it: moving within 1 cm/s of B's velocity, turning under 0.5 rad/s."""
    block, carrier = objects["block"], objects["carrier_b"]
    rel = np.linalg.norm(np.asarray(block.lin_vel) - np.asarray(carrier.lin_vel))
    return bool(_in_nest(block, carrier) and rel < 0.01 and np.linalg.norm(block.ang_vel) < 0.5)


def distance(objects):
    """Metres from the block's centre to where it sits in the middle of B's nest, wherever B is now."""
    carrier = objects["carrier_b"]
    goal = np.array([carrier.pos[0], carrier.pos[1], carrier.high[2] + PART[2] / 2])
    return float(np.linalg.norm(objects["block"].pos - goal))


def _out_of_right_reach(block) -> bool:
    return bool(np.hypot(block.pos[0] - RIGHT_BASE_XY[0], block.pos[1] - RIGHT_BASE_XY[1]) >= RIGHT_OUT_OF_REACH)


def score(objects):
    """Partial credit, kept at its best over the trial: 0.5 once the block's centre is 81 cm or more across the table
    from the right arm's base, beyond that arm's reach, with its underside 3 cm or more above the table (the left arm
    carries it); 0.25 once its underside is 3 cm or more above carrier A's top (taken off A)."""
    block = objects["block"]
    if _out_of_right_reach(block) and block.low[2] >= TOP + LIFT_M:
        return 0.5
    if block.low[2] >= PLATE_TOP + LIFT_M:
        return 0.25
    return 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
TRAVEL_Z = TOP + 0.13
# WHY: commanded fully closed, both fingers push at their 50 N effort limit and balance each other whatever their
# position, so the held block slides freely along the closing axis (world y): in Isaac Sim it drifted 2 to 3 cm off the
# TCP while the left arm lifted it and landed on a lip of the nest. Fingers commanded 4 cm apart instead squeeze the
# 5 cm block with about 25 N each, short of the limit, and their stiffness keeps it centred between them.
GRIP = -(PART[1] - 0.01) / 2  # finger joint target: pads 4 cm apart, 5 mm inside each face of the block
A_GRASP_Z = BLOCK.pos[2] - 0.002  # TCP a little below the block's centre: pad tips 1 mm above A's top
PRE_Z = A_GRASP_Z + 0.06  # waiting height: the pad tips clear the arriving block by about 3 cm
SET_GAP = 0.005  # the block's underside this far above the surface it is set on when the fingers open
PAD_SET_Z = PAD_TOP + PART[2] / 2 + SET_GAP + 0.002  # the right arm's TCP when it lets go on the pad
PAD_GRASP_Z = PAD_TOP + PART[2] / 2 + 0.006  # the left arm grips high: pad tips 8 mm above the block's underside
NEST_SET_Z = PLATE_TOP + PART[2] / 2 + SET_GAP + 0.006  # pad tips 13 mm above B's top, 7 mm above the lips
RIGHT_PARK = (0.24, -0.28, TRAVEL_Z)  # over the right arm's own side, clear of the pad
LEFT_PARK = (0.24, 0.24, TRAVEL_Z)
B_PLACE_CYCLE = 1  # the left arm sets the block in during B's second wait at its near stop (18 to 21 s)


def _timed(plan: list[Waypoint], until: float, pos, finger, label: str) -> Waypoint:
    """A wait that ends at episode time ``until``."""
    return Waypoint(until - sum(w.duration for w in plan), pos, finger, label)


def demo():
    """Right arm: wait above A's near stop, grab the block during A's first wait there (4.5 to 6.5 s), lifting from
    5.65 s, set it on the pad and go back to its own side. Left arm: wait on its side until the right arm has gone,
    take the block from the pad, wait above B's near stop, and lower it into the nest during B's wait there from 18 s,
    slowly over the lips, letting go at 19.7 s and clear before B leaves at 21 s. The TCPs stay at least 24 cm apart throughout."""
    ax, ay = NEAR_X, A_Y
    px, py = PAD_XY
    right = [
        Waypoint(2.0, (ax, ay, TRAVEL_Z), FINGER_OPEN, "above A's near stop"),
        Waypoint(1.5, (ax, ay, PRE_Z), FINGER_OPEN, "wait height"),
    ]
    right.append(_timed(right, A_AT_NEAR[0] + 0.15, (ax, ay, PRE_Z), FINGER_OPEN, "wait for the block"))
    right += [
        Waypoint(0.5, (ax, ay, A_GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.5, (ax, ay, A_GRASP_Z), GRIP, "close"),
        Waypoint(0.7, (ax, ay, TRAVEL_Z), GRIP, "lift"),
        Waypoint(2.0, (px, py, TRAVEL_Z), GRIP, "carry to the pad"),
        Waypoint(1.2, (px, py, PAD_SET_Z), GRIP, "lower"),
        Waypoint(0.6, (px, py, PAD_SET_Z), FINGER_OPEN, "release"),
        Waypoint(0.8, (px, py, TRAVEL_Z), FINGER_OPEN, "up"),
        Waypoint(1.5, RIGHT_PARK, FINGER_OPEN, "back to its side"),
    ]
    right_gone = sum(w.duration for w in right[:-1]) + 0.5  # the right arm is well on its way back

    bx, by = NEAR_X, B_Y
    b_stop = B_AT_NEAR[0] + B_PLACE_CYCLE * B_CYCLE_S
    left = [Waypoint(2.0, LEFT_PARK, FINGER_OPEN, "to its side of the pad")]
    left.append(_timed(left, right_gone, LEFT_PARK, FINGER_OPEN, "wait for the right arm to clear the pad"))
    left += [
        Waypoint(1.2, (px, py, TRAVEL_Z), FINGER_OPEN, "above the pad"),
        Waypoint(1.0, (px, py, PAD_GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.6, (px, py, PAD_GRASP_Z), GRIP, "close"),
        Waypoint(0.8, (px, py, TRAVEL_Z), GRIP, "lift"),
        Waypoint(1.8, (bx, by, TRAVEL_Z), GRIP, "carry above B's near stop"),
    ]
    left.append(_timed(left, b_stop, (bx, by, TRAVEL_Z), GRIP, "wait for B to stop"))
    # WHY slow, with a pause: the arm's joints are soft (PD), and a quick 7 cm drop left the block 7 mm short of the
    # nest's centre along the line when it reached the lips, enough to catch the near lip with 1 cm of clearance.
    left += [
        Waypoint(0.6, (bx, by, NEST_SET_Z + 0.02), GRIP, "down to just above the lips"),
        Waypoint(0.8, (bx, by, NEST_SET_Z), GRIP, "lower into the nest"),
        Waypoint(0.3, (bx, by, NEST_SET_Z), GRIP, "settle"),
        Waypoint(0.5, (bx, by, NEST_SET_Z), FINGER_OPEN, "release"),
        Waypoint(0.7, (bx, by, TRAVEL_Z), FINGER_OPEN, "up"),
    ]
    return {"left": left, "right": right}
