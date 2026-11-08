"""On the station table, a conveyor runs straight across the table from the robot's right to its left, 34 cm in front
of the arm bases. Four grey steel carrier plates (14 cm across the belt, 9 cm along it) ride on it, each with one
painted wooden cube (4 cm) in its middle. From the front of the line: blue, red, blue, red. At the left end of the
line, 42 cm left of the midpoint between the arms, the conveyor goes into a grey sheet-steel tunnel that is too low
to reach into: whatever is still on the belt there goes on to the next station. A red bin stands in front of the
right arm and a blue bin in front of the left arm, between the arms and the belt.

The belt's speed changes abruptly, on a fixed schedule: it runs at 4 cm/s from the start, stops dead at 9 s for 5.5 s
(as when the line downstream jams), restarts at 8 cm/s at 14.5 s to clear the backlog, drops back to 4 cm/s at 19.5 s,
and stops at 28.5 s, when every carrier is inside the tunnel. While the belt is stopped, a pusher bar behind the belt
(on its far side, 8 cm left of the midpoint) shoves the first blue cube 4 cm across its carrier toward the robot and
pulls back.

The robot must put each red cube in the red bin and each blue cube in the blue bin before it goes into the tunnel,
and put nothing in the wrong bin.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Sort the cubes on the belt: red cubes in the red bin, blue cubes in the blue bin. ..." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, belt_path, speed_schedule, stroke  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import FLOOR_Z  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # the score is what lies in the bins when the trial ends


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the belt's speed ------------------------------------------------------------------------------------------------
# +y (to the robot's left) is the direction of flow.
CRUISE, SURGE = 0.04, 0.08  # m/s
T_STOP, T_GO, T_SLOW, T_END = 9.0, 14.5, 19.5, 28.5
RAMP_S = 0.3  # every change of speed takes 0.3 s: abrupt, but without a jerk the cubes would slide on
SCHEDULE = ((0.0, CRUISE), (T_STOP, 0.0), (T_GO, SURGE), (T_SLOW, CRUISE), (T_END, 0.0))
RUN = speed_schedule(SCHEDULE, RAMP_S)
STOPPED = (T_STOP + RAMP_S, T_GO)  # the belt stands still


# -- the line ---------------------------------------------------------------------------------------------------------
LINE_X = 0.34  # the belt's centre line, metres in front of the arm bases
END_Y = 0.42  # the tunnel's mouth: the end of the line the robot works on
RIGHT_END_Y = -0.665  # the conveyor's right end, 1.5 cm past the table's right edge
TUNNEL_END_Y = 0.90  # the tunnel's closed far end, 25 cm past the table's left edge
RAIL_H, RAIL_W, RAIL_DX = 0.015, 0.015, 0.04  # two steel rails, centred 4 cm either side of the line
CARRIER = (0.14, 0.09, 0.015)  # across the belt (x), along it (y), thick
CARRIER_Z = TOP + RAIL_H + 0.002 + CARRIER[2] / 2  # 2 mm above the rails: the carriers ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
CUBE = 0.04
CUBE_MASS = 0.05
CUBE_Z = CARRIER_TOP + CUBE / 2
CUBE_TOP = CARRIER_TOP + CUBE
GREY_STEEL = (0.62, 0.64, 0.66)
DARK = (0.18, 0.18, 0.2)
RED = (0.82, 0.12, 0.1)
BLUE = (0.15, 0.32, 0.8)

# The cubes from the front of the line back: name, colour, where its carrier starts along the line.
# Nose to tail 11 cm apart (2 cm between carriers), as close as they go, so the tunnel that takes them all is short.
PARTS = (("blue_1", "blue", -0.28), ("red_1", "red", -0.39), ("blue_2", "blue", -0.50), ("red_2", "red", -0.61))
COLOUR = {name: colour for name, colour, _ in PARTS}
START_Y = {name: y for name, _, y in PARTS}
NAMES = tuple(COLOUR)


def carrier_path(name: str):
    return belt_path((LINE_X, START_Y[name], CARRIER_Z), (0.0, 1.0, 0.0), RUN)


def part_y(name: str, t: float) -> float:
    """Where a cube riding on its carrier is along the line at ``t``."""
    return float(START_Y[name] + RUN(t)[0])


CARRIERS = [
    DrivenBox(f"carrier_{n}", pos=(LINE_X, START_Y[n], CARRIER_Z), size=CARRIER, color=GREY_STEEL, path=carrier_path(n))
    for n in NAMES
]
CUBES = [
    Box(n, pos=(LINE_X, START_Y[n], CUBE_Z), size=(CUBE, CUBE, CUBE), mass=CUBE_MASS, color=RED if c == "red" else BLUE)
    for n, c, _ in PARTS
]
RAILS = [
    Box(f"rail_{side}", pos=_floats((LINE_X + dx, (RIGHT_END_Y + TUNNEL_END_Y) / 2, TOP + RAIL_H / 2)),
        size=_floats((RAIL_W, TUNNEL_END_Y - RIGHT_END_Y, RAIL_H)), color=DARK)
    for side, dx in (("near", -RAIL_DX), ("far", RAIL_DX))
]

# -- the pusher ------------------------------------------------------------------------------------------------------
# A flat steel bar behind the belt, sliding on a guide block: during the stop it shoves blue_1 4 cm across its carrier
# toward the robot, low down (its face spans 3 to 13 mm above the carrier, well under the height that would tip a
# 4 cm cube), and pulls back. It moves only while the belt stands still.
PUSH_Y = 0.08  # the bar's centre line along the belt, where blue_1 stands during the stop
PUSHED = "blue_1"
NUDGE = 0.04  # how far the cube is shoved toward the robot (-x)
BAR = (0.22, 0.06, 0.01)  # along its stroke (x), across (y), thick
BAR_Z = CARRIER_TOP + 0.003 + BAR[2] / 2
BAR_REST_FACE = LINE_X + CARRIER[0] / 2 + 0.015  # its pushing face at rest, 1.5 cm past the carriers' far edge
BAR_OUT_FACE = LINE_X + CUBE / 2 - NUDGE  # fully out: where it leaves the cube's far face
PUSH_T0, PUSH_OUT_S, PUSH_HOLD_S, PUSH_BACK_S = 9.4, 0.8, 0.1, 0.6
PUSH_DONE = PUSH_T0 + PUSH_OUT_S + PUSH_HOLD_S + PUSH_BACK_S  # 10.9 s: back at rest, 3.6 s before the belt restarts
BAR_PATH = stroke((BAR_REST_FACE + BAR[0] / 2, PUSH_Y, BAR_Z), (BAR_OUT_FACE + BAR[0] / 2, PUSH_Y, BAR_Z), PUSH_T0,
                  PUSH_OUT_S, PUSH_HOLD_S, PUSH_BACK_S)
PUSHER = DrivenBox("pusher_bar", pos=_floats(BAR_PATH(0.0)[0]), size=BAR, color=GREY_STEEL, path=BAR_PATH)
GUIDE = (0.20, 0.08)  # the guide block's footprint: x from 10 cm to 30 cm past the line, 8 cm wide
GUIDE_TOP = BAR_Z - BAR[2] / 2 - 0.002
GUIDE_BLOCK = Box("pusher_guide", pos=_floats((LINE_X + 0.10 + GUIDE[0] / 2, PUSH_Y, (TOP + GUIDE_TOP) / 2)),
                  size=_floats((GUIDE[0], GUIDE[1], GUIDE_TOP - TOP)), color=DARK)

# -- the tunnel ------------------------------------------------------------------------------------------------------
# Sheet steel over the line from END_Y on: a roof 2 cm above the cubes' tops on two side walls that stand on the
# table (and, past its edge, hang down to the rails' level), closed at the far end and held up by a post.
WALL_IN = CARRIER[0] / 2 + 0.0125  # the side walls' inner faces, 1.25 cm outside the carriers
SHEET = 0.01
ROOF_T = 0.008
ROOF_LOW = CUBE_TOP + 0.02
ROOF_TOP = ROOF_LOW + ROOF_T
TUNNEL_OUT = WALL_IN + SHEET
TUNNEL_LEN = TUNNEL_END_Y - END_Y


def _tunnel() -> list[Box]:
    yc = (END_Y + TUNNEL_END_Y) / 2
    wall_h = ROOF_LOW - TOP
    return [
        Box("tunnel_roof", pos=_floats((LINE_X, yc, ROOF_LOW + ROOF_T / 2)),
            size=_floats((2 * TUNNEL_OUT, TUNNEL_LEN, ROOF_T)), color=GREY_STEEL),
        Box("tunnel_near", pos=_floats((LINE_X - WALL_IN - SHEET / 2, yc, TOP + wall_h / 2)),
            size=_floats((SHEET, TUNNEL_LEN, wall_h)), color=GREY_STEEL),
        Box("tunnel_far", pos=_floats((LINE_X + WALL_IN + SHEET / 2, yc, TOP + wall_h / 2)),
            size=_floats((SHEET, TUNNEL_LEN, wall_h)), color=GREY_STEEL),
        Box("tunnel_end", pos=_floats((LINE_X, TUNNEL_END_Y + SHEET / 2, TOP + wall_h / 2)),
            size=_floats((2 * WALL_IN, SHEET, wall_h)), color=GREY_STEEL),
        # past the table's left edge the tunnel's far end stands on a post to the floor
        Box("post_left", pos=_floats((LINE_X, TUNNEL_END_Y - 0.03, (TOP + FLOOR_Z) / 2)),
            size=_floats((2 * TUNNEL_OUT, 0.04, TOP - FLOOR_Z)), color=DARK),
    ]


# -- the bins --------------------------------------------------------------------------------------------------------
BIN_X = 0.14  # far walls 5 cm from the carriers, near walls 2.5 cm in front of the arm bases
BIN_Y = {"red": -0.30, "blue": 0.30}  # in front of the right and the left arm
BIN_IN = (0.14, 0.14)  # inside
BIN_WALL, BIN_FLOOR, BIN_H = 0.01, 0.01, 0.05
BIN_FLOOR_TOP = TOP + BIN_FLOOR
BIN_RIM = TOP + BIN_H


def _bin(colour: str) -> list[Box]:
    x, y = BIN_X, BIN_Y[colour]
    ix, iy = BIN_IN
    ox, oy = ix + 2 * BIN_WALL, iy + 2 * BIN_WALL
    z = TOP + BIN_H / 2
    c = RED if colour == "red" else BLUE
    n = f"{colour}_bin"
    return [
        Box(f"{n}_floor", pos=_floats((x, y, TOP + BIN_FLOOR / 2)), size=_floats((ox, oy, BIN_FLOOR)), color=c),
        Box(f"{n}_near", pos=_floats((x - ix / 2 - BIN_WALL / 2, y, z)), size=_floats((BIN_WALL, oy, BIN_H)), color=c),
        Box(f"{n}_far", pos=_floats((x + ix / 2 + BIN_WALL / 2, y, z)), size=_floats((BIN_WALL, oy, BIN_H)), color=c),
        Box(f"{n}_right", pos=_floats((x, y - iy / 2 - BIN_WALL / 2, z)), size=_floats((ix, BIN_WALL, BIN_H)), color=c),
        Box(f"{n}_left", pos=_floats((x, y + iy / 2 + BIN_WALL / 2, z)), size=_floats((ix, BIN_WALL, BIN_H)), color=c),
    ]


OBJECTS = [*RAILS, *CARRIERS, *CUBES, PUSHER, GUIDE_BLOCK, *_tunnel(), *_bin("red"), *_bin("blue")]

# -- scoring ---------------------------------------------------------------------------------------------------------


def in_bin(cube, colour: str) -> bool:
    """In a bin: the cube's centre between the bin's inner walls and below the top of its walls, and the cube still."""
    dx, dy = cube.pos[0] - BIN_X, cube.pos[1] - BIN_Y[colour]
    inside = abs(dx) < BIN_IN[0] / 2 and abs(dy) < BIN_IN[1] / 2
    return bool(inside and cube.pos[2] < BIN_RIM and cube.resting())


def tally(objects) -> tuple[int, int]:
    """(cubes in the bin of their own colour, cubes in the other bin)."""
    right = sum(in_bin(objects[n], COLOUR[n]) for n in NAMES)
    wrong = sum(in_bin(objects[n], "blue" if COLOUR[n] == "red" else "red") for n in NAMES)
    return right, wrong


def score(objects):
    """0.25 for each cube in the bin of its colour, minus 0.25 for each cube in the other bin, never below 0. A cube
    anywhere else (on the belt, gone into the tunnel, on the table or the floor, in a gripper) counts 0."""
    right, wrong = tally(objects)
    return max(0.0, 0.25 * right - 0.25 * wrong)


def success(objects):
    """Every cube in the bin of its own colour."""
    return tally(objects) == (len(NAMES), 0)


def distance(objects):
    """Metres from the first cube (front of the line first) not yet in its bin to the middle of that bin's floor."""
    for n in NAMES:
        cube = objects[n]
        if not in_bin(cube, COLOUR[n]):
            goal = np.array([BIN_X, BIN_Y[COLOUR[n]], BIN_FLOOR_TOP + CUBE / 2])
            return float(np.linalg.norm(cube.pos - goal))
    return 0.0


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop, timed against the belt's schedule. Each arm keeps to its own half and its own colour: the right arm takes
# the red cubes (red_1 while the belt cruises, red_2 while it stands still) and the left arm the blue ones (blue_1
# after the pusher has shoved it, from where it now is, and blue_2 after the surge). Fingers close across the belt
# (yaw -90 degrees), so a cube's place along the belt only has to be right to within a centimetre or so, while a
# cube shoved 4 cm across the belt is missed unless the grasp aims where it is now.
DT = 0.05  # one control step at 20 Hz
YAW = float(-np.pi / 2)
GRASP_Z = CUBE_Z + 0.004  # TCP a little above the cube's centre: the pad tips stop 1.1 cm above the carrier
PRE_Z = CUBE_TOP + PAD_TIP_PAST_TCP + 0.03  # over the belt: the open pad tips 3 cm above the cubes' tops
OFF_Z = GRASP_Z + 0.04  # lifted: the cube's underside 4 cm above its carrier
TRAVEL_Z = TOP + 0.15  # carrying: a held cube's underside 7.6 cm above the table, 5 cm over the bins' rims
PLACE_Z = BIN_FLOOR_TOP + CUBE / 2 + 0.004 + 0.008  # let go with the cube 8 mm above the bin floor
DESCEND_S, CLOSE_S, LIFT_S = 0.6, 0.5, 0.5
RISE_S, ACROSS_S, LOWER_S, RELEASE_S, UP_S = 0.5, 1.5, 0.8, 0.5, 0.5
PLACE_DY = {"red_1": -0.035, "red_2": 0.035, "blue_1": -0.035, "blue_2": 0.035}  # two cubes side by side in a bin
# When each arm starts down onto each cube (the TCP is over it then, at PRE_Z).
MEET = {"red_1": 2.5, "red_2": 9.8, "blue_1": 11.9, "blue_2": 19.9}
ARM = {"red_1": "right", "red_2": "right", "blue_1": "left", "blue_2": "left"}


def grasp_x(name: str) -> float:
    """Where the cube is across the belt when the demo grasps it: shoved toward the robot, for the pushed cube."""
    return LINE_X - NUDGE if name == PUSHED else LINE_X


def _track(name: str, t0: float, seconds: float, z0: float, z1: float, f0: float, f1: float, label: str):
    """Waypoints one control step apart that follow cube ``name`` along the belt from ``t0``, while the TCP goes from
    ``z0`` to ``z1`` and the fingers from ``f0`` to ``f1`` (both smoothly)."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        y = part_y(name, t0 + k * DT)
        out.append(Waypoint(DT, (grasp_x(name), y, z0 + s * (z1 - z0)), f0 + s * (f1 - f0), label, YAW))
    return out


def _take(name: str, t: float) -> tuple[list[Waypoint], float]:
    """From time ``t``: go over cube ``name`` where it will be at its meeting time, follow it down, close, lift it off
    its carrier, carry it to its bin and let go there. Returns the waypoints and the time they end."""
    meet = MEET[name]
    span = round((meet - t) / DT) * DT
    assert span >= 1.2, f"{name}: only {span:.2f} s to get over the belt"
    x, y = grasp_x(name), part_y(name, meet)
    if span >= 3.0:  # time to spare: go over the meeting point high, wait there, then come down to the belt
        go = min(2.0, span - 1.0)
        plan = [
            Waypoint(go, (x, y, TRAVEL_Z), FINGER_OPEN, f"high over {name}", YAW),
            Waypoint(span - go - 1.0, (x, y, TRAVEL_Z), FINGER_OPEN, f"wait for {name}", YAW),
            Waypoint(1.0, (x, y, PRE_Z), FINGER_OPEN, f"over {name}", YAW),
        ]
    else:
        plan = [Waypoint(span, (x, y, PRE_Z), FINGER_OPEN, f"over {name}", YAW)]
    t = meet
    plan += _track(name, t, DESCEND_S, PRE_Z, GRASP_Z, FINGER_OPEN, FINGER_OPEN, f"follow {name} down")
    t += DESCEND_S
    plan += _track(name, t, CLOSE_S, GRASP_Z, GRASP_Z, FINGER_OPEN, FINGER_CLOSED, f"close on {name}")
    t += CLOSE_S
    plan += _track(name, t, LIFT_S, GRASP_Z, OFF_Z, FINGER_CLOSED, FINGER_CLOSED, f"lift {name} off")
    t += LIFT_S
    bx, by = BIN_X, BIN_Y[COLOUR[name]] + PLACE_DY[name]
    plan += [
        Waypoint(RISE_S, (x, part_y(name, t), TRAVEL_Z), FINGER_CLOSED, f"rise with {name}", YAW),
        Waypoint(ACROSS_S, (bx, by, TRAVEL_Z), FINGER_CLOSED, f"to the {COLOUR[name]} bin", YAW),
        Waypoint(LOWER_S, (bx, by, PLACE_Z), FINGER_CLOSED, f"lower {name}", YAW),
        Waypoint(RELEASE_S, (bx, by, PLACE_Z), FINGER_OPEN, f"let go of {name}", YAW),
        Waypoint(UP_S, (bx, by, TRAVEL_Z), FINGER_OPEN, f"up from {name}", YAW),
    ]
    return plan, t + RISE_S + ACROSS_S + LOWER_S + RELEASE_S + UP_S


def demo():
    """Right arm: red_1 at 2.5 s while the belt cruises, red_2 at 9.8 s while it stands still. Left arm: blue_1 at
    11.9 s, 1 s after the pusher is back at rest, aiming 4 cm nearer than the line it rode on, and lifted off 1 s
    before the belt restarts; blue_2 at 19.9 s, once the belt is back at cruising speed. Each is 4 cm off its carrier
    1.6 s after its meeting time; the last is let go at 24.3 s."""
    plans = {}
    for side in ("right", "left"):
        plan, t = [], 0.0
        for name in sorted((n for n in NAMES if ARM[n] == side), key=MEET.get):
            take, t = _take(name, t)
            plan += take
        plans[side] = plan
    return plans
