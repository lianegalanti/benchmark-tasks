"""On the station table, between the arms and 25 cm in front of them, stands a laboratory rocker: a black
rubber-matted platform (18 cm deep, 44 cm long from left to right, 1.5 cm thick) on a grey drive base. The platform
never stops for the robot: it rocks about its centre line (which runs away from the arms, 6 cm above the table)
like a see-saw, 10 degrees each way, one full rock every 4 s. It starts level, its left end rising first.

Three square plastic sample bottles (4 x 4 x 6 cm, capped) stand on the mat in a row along the platform's length:
orange at the left end, green in the middle, purple at the right end, 17 cm apart. The mat holds them, so they ride
with the platform: the end ones rise and fall 3 cm each way and lean 10 degrees with it. Two grey open bins
(16 x 16 cm, 3.5 cm walls) stand on the table beside the rocker's ends, one in front of each arm.

The robot must take the three bottles off the rocking platform and leave them standing upright in the bins (either
bin), let go, without knocking any off the platform or dropping one on the table: a bottle that leaves the platform
other than in a gripper's closed fingers, or touches the table, scores the whole trial 0, even if it is put in a bin
afterwards.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Take the three bottles off the rocking platform and leave them standing upright in the grey bins. ..." ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, on_reset, on_step, quat_matrix, rocking  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # score() combines each bottle's state now with what the monitor has seen (see _STATE)
INSTRUCTION = ("Take the three bottles off the rocking platform and leave them standing upright in the grey bins. The "
               "platform keeps rocking and won't stop for you; don't knock any bottle off it or drop one on the table.")

# -- the rocker ----------------------------------------------------------------------------------------------------
ROCKER_X = 0.25  # the platform's centre, metres in front of the arm bases, on the midline between them
PLATE = (0.18, 0.44, 0.015)  # depth (x), length (y, left to right), thickness
PIVOT_Z = TOP + 0.06  # the rocking axis runs along x through the platform's centre
PLATE_TOP = PIVOT_Z + PLATE[2] / 2  # the mat's top when level
AMPLITUDE = math.radians(10.0)  # tilt each way; a positive tilt raises the left (+y) end
PERIOD_S = 4.0  # one full rock (15 a minute); the left end is highest at 1 s, 5 s, 9 s ..., the right at 3 s, 7 s ...
PATH = rocking((ROCKER_X, 0.0, PIVOT_Z), (1.0, 0.0, 0.0), AMPLITUDE, PERIOD_S)
MAT_FRICTION = (1.0, 0.9)  # rubber mat (static, dynamic): far above tan(10 deg) = 0.18, so nothing slides on it
ROCKER = DrivenBox("rocker", pos=(ROCKER_X, 0.0, PIVOT_Z), size=PLATE, color=(0.1, 0.1, 0.11), path=PATH,
                   friction=MAT_FRICTION)
# The drive base and the pivot block under the platform's middle; fixed, never touched by the tilted platform
# (test_task.py checks the gaps).
BASE = Box("rocker_base", pos=(ROCKER_X, 0.0, TOP + 0.015), size=(0.12, 0.10, 0.03), color=(0.55, 0.57, 0.6))
PIVOT = Box("rocker_pivot", pos=(ROCKER_X, 0.0, TOP + 0.036), size=(0.10, 0.02, 0.012),
            color=(0.35, 0.36, 0.38))


def tilt(t: float) -> float:
    """The platform's tilt (rad, positive raises its left end) at episode time ``t``."""
    return float(AMPLITUDE * math.sin(2 * math.pi * t / PERIOD_S))


def on_platform(local, t: float) -> np.ndarray:
    """Where a point fixed to the platform is at time ``t``: ``local`` is (x, y, z) in the level platform's frame,
    from its centre (z up from the mat's top)."""
    a = tilt(t)
    x, y, z = local
    z = z + PLATE[2] / 2
    return np.array([ROCKER_X + x, y * math.cos(a) - z * math.sin(a), PIVOT_Z + y * math.sin(a) + z * math.cos(a)])


# -- the bottles ---------------------------------------------------------------------------------------------------
BOTTLE = (0.04, 0.04, 0.06)
BOTTLE_MASS = 0.08
SPACING = 0.17
BOTTLE_Y = {"orange_bottle": SPACING, "green_bottle": 0.0, "purple_bottle": -SPACING}
COLORS = {"orange_bottle": (0.95, 0.5, 0.1), "green_bottle": (0.2, 0.7, 0.3), "purple_bottle": (0.55, 0.3, 0.75)}
BOTTLES = [Box(n, pos=(ROCKER_X, y, PLATE_TOP + BOTTLE[2] / 2), size=BOTTLE, mass=BOTTLE_MASS, color=COLORS[n])
           for n, y in BOTTLE_Y.items()]
NAMES = tuple(BOTTLE_Y)

# -- the bins ------------------------------------------------------------------------------------------------------
BIN_XY = {"left": (0.22, 0.39), "right": (0.22, -0.39)}
BIN_IN = 0.14  # inner width and depth
WALL, FLOOR, WALL_H = 0.01, 0.01, 0.035
BIN_FLOOR_TOP = TOP + FLOOR
GREY = (0.6, 0.6, 0.62)


def _bin(side: str) -> list[Box]:
    x, y = BIN_XY[side]
    out = BIN_IN / 2 + WALL / 2
    span = BIN_IN + 2 * WALL
    z = TOP + WALL_H / 2
    return [
        Box(f"bin_{side}_floor", pos=(x, y, TOP + FLOOR / 2), size=(span, span, FLOOR), color=GREY),
        Box(f"bin_{side}_wall_near", pos=(x - out, y, z), size=(WALL, span, WALL_H), color=GREY),
        Box(f"bin_{side}_wall_far", pos=(x + out, y, z), size=(WALL, span, WALL_H), color=GREY),
        Box(f"bin_{side}_wall_right", pos=(x, y - out, z), size=(BIN_IN, WALL, WALL_H), color=GREY),
        Box(f"bin_{side}_wall_left", pos=(x, y + out, z), size=(BIN_IN, WALL, WALL_H), color=GREY),
    ]


OBJECTS = [BASE, PIVOT, ROCKER, *BOTTLES, *_bin("left"), *_bin("right")]

# -- scoring -------------------------------------------------------------------------------------------------------
LIFT_M = 0.03  # a bottle's underside this far above the mat beneath it counts as taken off the platform
UPRIGHT_DEG = 20.0  # standing: its axis within this of vertical (it tips over past 34 degrees)
# A gripper holds a bottle only when its fingers are stopped by it: their opening between HOLD_OPEN (the bottle's
# 4 cm less 5 mm of squeeze) and its tool point within HOLD_AXIS of the bottle's axis, between its ends. A fully closed
# gripper pressed against a bottle's side (opening near 0, tool point 2 cm or more off the axis) does not hold it.
HOLD_OPEN = (BOTTLE[0] - 0.005, BOTTLE[0] + 0.012)  # 3.5 to 5.2 cm
HOLD_AXIS = 0.01
GRIPPER_CLEAR = 0.08  # let go: each gripper's centre line this far from the bottle's axis (the open fingers, 4.7 cm
# off that line, then stay 3 cm or more clear of the bottle's corners)
TABLE_TOUCH = 0.005  # a bottle whose underside comes this close to the table outside the bins has touched it
PAD_TIP = 0.013  # the fingertips past the tool point (yam_arena.constants.PAD_TIP_PAST_TCP)
WRIST_BACK = 0.134  # the gripper's wrist, back from the tool point along its axis


def mat_height(rocker, x: float, y: float) -> float:
    """Height of the mat's top surface (extended as a plane) above the point (x, y), from the platform's state."""
    n = quat_matrix(rocker.quat_xyzw)[:, 2]
    p = np.asarray(rocker.pos, float) + n * PLATE[2] / 2
    return float(p[2] - (n[0] * (x - p[0]) + n[1] * (y - p[1])) / n[2])


_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.asarray(BOTTLE) / 2


def taken_off(bottle, rocker) -> bool:
    """Every corner of the bottle is LIFT_M or more above the mat (extended as a plane) directly beneath it."""
    corners = np.asarray(bottle.pos, float) + _CORNERS @ quat_matrix(bottle.quat_xyzw).T
    return all(c[2] >= mat_height(rocker, c[0], c[1]) + LIFT_M for c in corners)


def over_platform(bottle, rocker) -> bool:
    """The bottle's centre is within the platform's outline (seen square to the platform)."""
    local = quat_matrix(rocker.quat_xyzw).T @ (np.asarray(bottle.pos, float) - np.asarray(rocker.pos, float))
    return bool(abs(local[0]) <= PLATE[0] / 2 and abs(local[1]) <= PLATE[1] / 2)


def in_bin(bottle) -> bool:
    """The bottle's centre is over the inside of a bin (between its walls)."""
    return any(abs(bottle.pos[0] - x) < BIN_IN / 2 and abs(bottle.pos[1] - y) < BIN_IN / 2
               for x, y in BIN_XY.values())


def upright(bottle) -> bool:
    return bool(quat_matrix(bottle.quat_xyzw)[2, 2] >= math.cos(math.radians(UPRIGHT_DEG)))


def standing_in_bin(bottle) -> bool:
    """Centre between a bin's walls, underside within 1 cm of its floor, axis within 20 degrees of vertical."""
    return bool(in_bin(bottle) and bottle.low[2] < BIN_FLOOR_TOP + 0.01 and upright(bottle))


def on_table(bottle) -> bool:
    """Down on the table outside the bins: its underside within 5 mm of the table top, its centre outside the bins."""
    return bool(not in_bin(bottle) and bottle.low[2] < TOP + TABLE_TOUCH)


def _seg_dist(a0, a1, b0, b1) -> float:
    """Shortest distance between segments a0-a1 and b0-b1 (sampled finely; exact to well under 1 mm here)."""
    s = np.linspace(0.0, 1.0, 41)
    pa = np.asarray(a0, float) + np.outer(s, np.asarray(a1, float) - np.asarray(a0, float))
    pb = np.asarray(b0, float) + np.outer(s, np.asarray(b1, float) - np.asarray(b0, float))
    return float(np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2).min())


def _axis(bottle):
    """The bottle's centre line, base to top."""
    up = quat_matrix(bottle.quat_xyzw)[:, 2] * BOTTLE[2] / 2
    return np.asarray(bottle.pos, float) - up, np.asarray(bottle.pos, float) + up


def holds(gripper, bottle) -> bool:
    """A gripper (tool point, finger opening, wrist point, fingertip point) is closed on the bottle."""
    tcp, opening, _, _ = gripper
    if not HOLD_OPEN[0] <= opening <= HOLD_OPEN[1]:
        return False
    axis = quat_matrix(bottle.quat_xyzw)[:, 2]
    d = np.asarray(tcp, float) - np.asarray(bottle.pos, float)
    along = float(d @ axis)
    return bool(abs(along) <= BOTTLE[2] / 2 and np.linalg.norm(d - along * axis) <= HOLD_AXIS)


def clear_of(gripper, bottle) -> bool:
    """The gripper's centre line (wrist to fingertips) is GRIPPER_CLEAR or more from the bottle's axis."""
    _, _, wrist, tip = gripper
    return _seg_dist(wrist, tip, *_axis(bottle)) >= GRIPPER_CLEAR


_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(lifted={n: False for n in NAMES}, over={n: True for n in NAMES}, clear={n: True for n in NAMES},
                  knocked_off=False, why="")


reset_state()


def update(objects, grippers) -> None:
    """One control step of the monitor, from the objects' states and the two grippers' (tool point, finger opening,
    wrist point, fingertip point): remember which bottles have been taken off the platform, whether each is clear of
    both grippers now, and latch knocked_off for the rest of the trial if a bottle leaves the platform's outline
    while no gripper holds it, or touches the table outside the bins."""
    rocker = objects["rocker"]
    for n in NAMES:
        b = objects[n]
        if taken_off(b, rocker):
            _STATE["lifted"][n] = True
        over = over_platform(b, rocker)
        if _STATE["over"][n] and not over and not any(holds(g, b) for g in grippers):
            _STATE.update(knocked_off=True, why=_STATE["why"] or f"{n} left the platform without being held")
        if on_table(b):
            _STATE.update(knocked_off=True, why=_STATE["why"] or f"{n} touched the table")
        _STATE["over"][n] = over
        _STATE["clear"][n] = all(clear_of(g, b) for g in grippers)


def grasp_point(tcp, approach, tip_left=None, tip_right=None) -> np.ndarray:
    """Midway between the fingers: the two fingers' link origins sit symmetrically about the gripper's centre line,
    so their midpoint gives where the fingers are centred across, even when one has closed further than the other
    (the tool point alone assumes they move together); along the approach axis it is the tool point's depth."""
    tcp = np.asarray(tcp, float)
    if tip_left is None or tip_right is None:
        return tcp
    off = (np.asarray(tip_left, float) + np.asarray(tip_right, float)) / 2 - tcp
    return tcp + off - (off @ approach) * np.asarray(approach, float)


def _grippers(environment):
    from motionkit import arm_joints, robot_links
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    links = robot_links(environment)
    out = []
    for side in ("left", "right"):
        q = arm_joints(environment, side, ARM_JOINTS)
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        opening = float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())
        wrist = links[side].get("gripper", tcp - rot[:, 2] * WRIST_BACK)
        out.append((grasp_point(tcp, rot[:, 2], links[side].get("tip_left"), links[side].get("tip_right")), opening,
                    np.asarray(wrist, float), tcp + rot[:, 2] * PAD_TIP))
    return out


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    update(object_states(environment, ["rocker", *NAMES]), _grippers(environment))


def placed(name: str, bottle) -> bool:
    """Full credit: taken off the platform, standing in a bin, and let go (both grippers clear of it)."""
    return bool(_STATE["lifted"][name] and standing_in_bin(bottle) and _STATE["clear"][name])


def credit(name: str, bottle) -> float:
    """One bottle's credit (the trial scores 0 instead if a bottle was knocked off): 0 if it was never taken off the
    platform; 1.0 standing in a bin and let go; 0.5 with its centre over a bin's inside below the rim plus 5 cm
    (lying or leaning in a bin, or standing there still held); otherwise 0.25 (held, or put back on the platform)."""
    if not _STATE["lifted"][name]:
        return 0.0
    if placed(name, bottle):
        return 1.0
    if in_bin(bottle) and bottle.low[2] < TOP + WALL_H + 0.05:
        return 0.5
    return 0.25


def success(objects):
    """No bottle knocked off, and all three standing in the bins, let go, and still."""
    return bool(not _STATE["knocked_off"] and all(placed(n, objects[n]) and objects[n].resting() for n in NAMES))


def distance(objects):
    """Metres, summed over the bottles not yet standing in a bin, from each to its nearer bin's floor centre."""
    total = 0.0
    for n in NAMES:
        b = objects[n]
        if standing_in_bin(b):
            continue
        total += min(float(np.linalg.norm(b.pos - np.array([x, y, BIN_FLOOR_TOP + BOTTLE[2] / 2])))
                     for x, y in BIN_XY.values())
    return total


def score(objects):
    """0 if a bottle was knocked off at any moment; otherwise the mean of the three bottles' credits (see credit())."""
    if _STATE["knocked_off"]:
        return 0.0
    return float(np.mean([credit(n, objects[n]) for n in NAMES]))


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Each bottle is gripped where it will be at a turn of the rock (when the platform stands still for a moment), its
# end high: the fingers close across the bottle's front and back faces (along x), which stay vertical however the
# platform tilts, 4 cm up its side. The gripper is in place 0.5 s before the turn, closes over the turn, and lifts
# 0.25 s after it, as the mat falls away below the bottle. Left arm: orange bottle (turn at 5 s), then the green one
# (13 s), both to the left bin; right arm: purple bottle (turn at 7 s) to the right bin.
GRIP_UP = 0.04  # the tool point this far up the bottle's axis from its base
TRAVEL_Z = TOP + 0.18  # a held bottle's underside then hangs 13.7 cm above the table, 3 cm over the platform's top
# at its highest; the reach ends near here over the green bottle (test_task.py checks the clearances)
LEFT_YAW, RIGHT_YAW = -math.pi / 2, math.pi / 2  # fingers closing along x
PRE_UP = 0.06  # waiting this far above the grip point: the pad tips stay 2.7 cm over the bottle's top at the turn
PLACE_GAP = 0.01  # a held bottle's lowest corner this far above the bin floor when let go
SPOTS = {"orange_bottle": ("left", 0.035), "green_bottle": ("left", -0.035), "purple_bottle": ("right", 0.0)}
TURNS = {"orange_bottle": 5.0, "green_bottle": 13.0, "purple_bottle": 7.0}  # when each is gripped (end high)


def grip_point(name: str) -> np.ndarray:
    """The tool point that grips ``name`` at its turn."""
    return on_platform((0.0, BOTTLE_Y[name], GRIP_UP), TURNS[name])


def place_point(name: str) -> np.ndarray:
    """The tool point that holds ``name`` (leaning as it was gripped) with its lowest corner PLACE_GAP above the bin
    floor, at its spot in the bin."""
    side, dy = SPOTS[name]
    x, y = BIN_XY[side]
    a = abs(tilt(TURNS[name]))
    low = GRIP_UP * math.cos(a) + BOTTLE[1] / 2 * math.sin(a)  # tool point to the lowest corner, vertically
    return np.array([x, y + dy, BIN_FLOOR_TOP + PLACE_GAP + low])


def _timed(stops, yaw: float) -> list[Waypoint]:
    """Waypoints from (time reached, position, finger, label) stops, timed from the start of the episode."""
    out, t = [], 0.0
    for at, pos, finger, label in stops:
        out.append(Waypoint(at - t, tuple(float(v) for v in pos), finger, label, yaw))
        t = at
    return out


def _fetch(name: str, start: float, reach_s: float = 1.5) -> list:
    """Stops that grip ``name`` at its turn and stand it in its bin; ``start`` is when the arm leaves for it, and
    it is over the bottle ``reach_s`` later."""
    g, p, turn = grip_point(name), place_point(name), TURNS[name]
    up = lambda q, z: (q[0], q[1], z)  # noqa: E731
    return [
        (start + reach_s, up(g, TRAVEL_Z), FINGER_OPEN, f"over the {name}"),
        (turn - 1.2, g + [0, 0, PRE_UP], FINGER_OPEN, "wait height"),
        (turn - 1.0, g + [0, 0, PRE_UP], FINGER_OPEN, "wait for the turn"),
        (turn - 0.5, g, FINGER_OPEN, "descend"),
        (turn + 0.1, g, FINGER_CLOSED, "close"),
        (turn + 0.25, g, FINGER_CLOSED, "hold"),
        (turn + 0.95, up(g, TRAVEL_Z), FINGER_CLOSED, "lift"),
        (turn + 3.0, up(p, TRAVEL_Z), FINGER_CLOSED, "carry"),
        (turn + 4.2, p, FINGER_CLOSED, "lower"),
        (turn + 4.8, p, FINGER_OPEN, "release"),
        (turn + 5.6, up(p, TRAVEL_Z), FINGER_OPEN, "retreat"),
    ]


def demo():
    left = _fetch("orange_bottle", 0.5)
    left += _fetch("green_bottle", left[-1][0], 1.0)
    right = _fetch("purple_bottle", 1.0)
    return {"left": _timed(left, LEFT_YAW), "right": _timed(right, RIGHT_YAW)}
