"""A guarded indexing conveyor runs straight across the station table, 32 cm in front of the arm bases, and carries
parts from the robot's right to its left. Its grey sheet-steel cover hides the whole line except a 10 cm open
window, centred 10 cm left of the midpoint between the arms. Eight steel carriers (6.5 x 6.5 x 1.5 cm) ride on it
nose to tail, 9 cm apart, each with one bare aluminium cube (4 x 4 x 4 cm) on top; the eight cubes are
identical. The line indexes: every 4 s it moves all carriers 9 cm in 1.2 s, so each carrier in turn stops in the
window for 2.8 s and then goes on under the cover, out of reach. The window starts empty; the first cube arrives
4.2 s into the trial, the eighth at 32.2 s, and after that the line stops with the window empty. A blue open QC tray
(17 x 17 cm, 4 cm walls) stands on the table in front of the left arm, nearer the arms than the conveyor.

The robot must take every fourth part off the line into the QC tray, counting the first part to arrive as part 1 (so
parts 4 and 8), and let all the others ride past.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Take every fourth part off the line ..." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, index_line  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import FLOOR_Z  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # the score is what is in the tray and off the line when the trial ends

# -- the line's motion ---------------------------------------------------------------------------------------------
LINE_X = 0.32  # the conveyor's centre line, metres in front of the arm bases
WINDOW_Y = 0.10  # the window's centre, 10 cm left of the midpoint between the arms; +y (left) is the flow direction
PITCH = 0.09  # carrier spacing, and how far the line moves per index
N_PARTS = 8
SAMPLES = (4, 8)  # every fourth part, counting the first to arrive as part 1
FIRST_S = 3.0  # the first index starts 3 s into the trial
TRAVEL_S, CYCLE_S = 1.2, 4.0  # each index moves 9 cm in 1.2 s, once every 4 s: a 2.8 s stop in the window
N_INDEX = N_PARTS + 1  # one more index after part 8's stop, so the line ends with the window empty


def at_window(k: int) -> tuple[float, float]:
    """When part ``k`` (1..8) stands still in the window: after index k ends, until index k + 1 starts."""
    return FIRST_S + (k - 1) * CYCLE_S + TRAVEL_S, FIRST_S + k * CYCLE_S


# -- the conveyor --------------------------------------------------------------------------------------------------
RAIL_H, RAIL_W, RAIL_DX = 0.015, 0.015, 0.02  # two steel rails, centred 2 cm either side of the line
HALF_LEN = 0.78  # the conveyor runs 78 cm either way from the window: 3 cm past the table's right edge, 23 cm past its left
CARRIER = (0.065, 0.065, 0.015)
CARRIER_Z = TOP + RAIL_H + 0.002 + CARRIER[2] / 2  # 2 mm above the rails: the carriers ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
CUBE = 0.04
CUBE_Z = CARRIER_TOP + CUBE / 2
CUBE_TOP = CARRIER_TOP + CUBE
GREY_STEEL = (0.62, 0.64, 0.66)
DARK = (0.18, 0.18, 0.2)
ALUMINIUM = (0.80, 0.81, 0.83)


def carrier_start(k: int) -> tuple[float, float, float]:
    """Carrier ``k`` (1..8) at the start: k pitches upstream (to the robot's right) of the window."""
    return (LINE_X, WINDOW_Y - k * PITCH, CARRIER_Z)


def carrier_path(k: int):
    return index_line(carrier_start(k), (0.0, PITCH, 0.0), FIRST_S, TRAVEL_S, CYCLE_S, N_INDEX)


CARRIERS = [
    DrivenBox(f"carrier_{k}", pos=carrier_start(k), size=CARRIER, color=GREY_STEEL, path=carrier_path(k))
    for k in range(1, N_PARTS + 1)
]
PARTS = [
    Box(f"part_{k}", pos=(LINE_X, WINDOW_Y - k * PITCH, CUBE_Z), size=(CUBE, CUBE, CUBE), mass=0.17, color=ALUMINIUM)
    for k in range(1, N_PARTS + 1)
]
RAILS = [
    Box(f"rail_{side}", pos=(LINE_X + dx, WINDOW_Y, TOP + RAIL_H / 2), size=(RAIL_W, 2 * HALF_LEN, RAIL_H), color=DARK)
    for side, dx in (("near", -RAIL_DX), ("far", RAIL_DX))
]

# -- the cover -----------------------------------------------------------------------------------------------------
# Two sheet-steel tunnels over the line, one each side of the window: a roof 2 cm above the cubes' tops on two side
# walls that stand on the table, closed at the outer end. Between them the window is open from the table up.
WINDOW_HALF = 0.05  # the window is 10 cm long, along the line
WALL_IN = 0.045  # the side walls' inner faces, this far either side of the line (1.25 cm outside the carriers)
SHEET = 0.01  # wall thickness
ROOF_T = 0.008
ROOF_LOW = CUBE_TOP + 0.02
ROOF_TOP = ROOF_LOW + ROOF_T
COVER_OUT = WALL_IN + SHEET  # the cover's outer faces, either side of the line
COVER_LEN = HALF_LEN - WINDOW_HALF


def _cover() -> list[Box]:
    out = []
    wall_h = ROOF_LOW - TOP
    for side, sign in (("right", -1), ("left", 1)):
        yc = WINDOW_Y + sign * (WINDOW_HALF + COVER_LEN / 2)
        out += [
            Box(f"cover_{side}_roof", pos=(LINE_X, yc, ROOF_LOW + ROOF_T / 2), size=(2 * COVER_OUT, COVER_LEN, ROOF_T),
                color=GREY_STEEL),
            Box(f"cover_{side}_near", pos=(LINE_X - WALL_IN - SHEET / 2, yc, TOP + wall_h / 2),
                size=(SHEET, COVER_LEN, wall_h), color=GREY_STEEL),
            Box(f"cover_{side}_far", pos=(LINE_X + WALL_IN + SHEET / 2, yc, TOP + wall_h / 2),
                size=(SHEET, COVER_LEN, wall_h), color=GREY_STEEL),
            Box(f"cover_{side}_end", pos=(LINE_X, WINDOW_Y + sign * (HALF_LEN + SHEET / 2), TOP + wall_h / 2),
                size=(2 * WALL_IN, SHEET, wall_h), color=GREY_STEEL),
        ]
    # The conveyor's end stand: past the table's left edge, the line stands on a post to the floor.
    out.append(Box("stand_left", pos=(LINE_X, WINDOW_Y + HALF_LEN - 0.04, (TOP + FLOOR_Z) / 2),
                   size=(2 * COVER_OUT, 0.04, TOP - FLOOR_Z), color=DARK))
    return out


# -- the QC tray ---------------------------------------------------------------------------------------------------
TRAY_XY = (0.16, 0.33)
TRAY_IN = 0.15  # inner width and depth
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


OBJECTS = [*RAILS, *CARRIERS, *PARTS, *_cover(), *_tray()]

# -- scoring -------------------------------------------------------------------------------------------------------
# Parts are told apart by name (on a real line, by a number engraved under each cube), so a part counts as itself
# whenever and however it was taken: the count is in the parts, not in when they were picked.


def in_tray(part) -> bool:
    """Lying in the tray: centre between the walls, underside within 1 cm of the tray floor, and still."""
    dx, dy = part.pos[0] - TRAY_XY[0], part.pos[1] - TRAY_XY[1]
    inside = abs(dx) < TRAY_IN / 2 and abs(dy) < TRAY_IN / 2
    return bool(inside and part.low[2] < TRAY_FLOOR_TOP + 0.01 and part.resting())


def on_a_carrier(part, carriers) -> bool:
    """Riding on the line: its centre over some carrier's top, its underside within 1 cm of that top."""
    for c in carriers:
        over = abs(part.pos[0] - c.pos[0]) < CARRIER[0] / 2 and abs(part.pos[1] - c.pos[1]) < CARRIER[1] / 2
        if over and abs(part.low[2] - c.high[2]) < 0.01:
            return True
    return False


def tally(objects) -> tuple[int, int]:
    """(samples lying in the tray, other parts not riding on a carrier)."""
    carriers = [objects[f"carrier_{k}"] for k in range(1, N_PARTS + 1)]
    right = sum(in_tray(objects[f"part_{k}"]) for k in SAMPLES)
    wrong = sum(not on_a_carrier(objects[f"part_{k}"], carriers) for k in range(1, N_PARTS + 1) if k not in SAMPLES)
    return right, wrong


def score(objects):
    """0.5 for each of parts 4 and 8 lying in the tray, minus 0.25 for each other part taken off the line (anywhere
    but on a carrier: in the tray, on the table, in a gripper), and never below 0."""
    right, wrong = tally(objects)
    return max(0.0, 0.5 * right - 0.25 * wrong)


def success(objects):
    """Parts 4 and 8 lying in the tray, and every other part still riding on the line."""
    return tally(objects) == (len(SAMPLES), 0)


def distance(objects):
    """Metres from the next sample still missing from the tray to the middle of the tray floor (0 when both are in)."""
    goal = np.array([*TRAY_XY, TRAY_FLOOR_TOP + CUBE / 2])
    for k in SAMPLES:
        part = objects[f"part_{k}"]
        if not in_tray(part):
            return float(np.linalg.norm(part.pos - goal))
    return 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Open loop, timed against the line: the left arm waits over the window, fingers closing across the line (yaw 90
# degrees) so they stay clear of the cover's mouths, and takes parts 4 and 8 during their stops.
YAW = float(-np.pi / 2)
TRAVEL_Z = TOP + 0.17  # TCP height for moving about: a held cube's underside is 4.5 cm above the cover's roof
GRASP_Z = CUBE_Z + 0.004  # TCP a little above the cube's centre: the pad tips stop 1.5 cm above the carrier
WAIT_Z = CUBE_TOP + PAD_TIP_PAST_TCP + 0.045  # waiting over the window, the pad tips are 4.5 cm above passing cubes
PLACE_Y = {4: TRAY_XY[1] - 0.04, 8: TRAY_XY[1] + 0.04}  # side by side in the tray, 4 cm apart
PLACE_Z = TRAY_FLOOR_TOP + CUBE / 2 + 0.008


def _take(k: int, t_now: float) -> tuple[list[Waypoint], float]:
    """Wait over the window until part ``k`` has stopped, take it during its stop and lay it in the tray."""
    x, y = LINE_X, WINDOW_Y
    tx, ty = TRAY_XY[0], PLACE_Y[k]
    stop, _ = at_window(k)
    plan = [Waypoint(stop + 0.2 - t_now, (x, y, WAIT_Z), FINGER_OPEN, f"wait for part {k}", YAW)]
    plan += [
        Waypoint(0.6, (x, y, GRASP_Z), FINGER_OPEN, f"descend to part {k}", YAW),
        Waypoint(0.6, (x, y, GRASP_Z), FINGER_CLOSED, f"close on part {k}", YAW),
        Waypoint(0.8, (x, y, TRAVEL_Z), FINGER_CLOSED, f"lift part {k}", YAW),
        Waypoint(1.8, (tx, ty, TRAVEL_Z), FINGER_CLOSED, "carry", YAW),
        Waypoint(1.2, (tx, ty, PLACE_Z), FINGER_CLOSED, "lower", YAW),
        Waypoint(0.6, (tx, ty, PLACE_Z), FINGER_OPEN, "release", YAW),
        Waypoint(0.8, (tx, ty, TRAVEL_Z), FINGER_OPEN, "retreat", YAW),
    ]
    return plan, t_now + sum(w.duration for w in plan)


def demo():
    """Left arm: over the window by 2.5 s, let parts 1 to 3 pass under it, take part 4 in its stop (16.2 to 19.0 s)
    and lay it in the tray, come back, let parts 5 to 7 pass, take part 8 (32.2 to 35.0 s) and lay it beside part 4."""
    plan = [
        Waypoint(1.5, (LINE_X, WINDOW_Y, TRAVEL_Z), FINGER_OPEN, "over the window", YAW),
        Waypoint(1.0, (LINE_X, WINDOW_Y, WAIT_Z), FINGER_OPEN, "wait height", YAW),
    ]
    t = sum(w.duration for w in plan)
    for k in SAMPLES:
        if k != SAMPLES[0]:
            plan.append(Waypoint(1.8, (LINE_X, WINDOW_Y, TRAVEL_Z), FINGER_OPEN, "back over the window", YAW))
            plan.append(Waypoint(1.0, (LINE_X, WINDOW_Y, WAIT_Z), FINGER_OPEN, "wait height", YAW))
            t += 2.8
        take, t = _take(k, t)
        plan += take
    return {"left": plan}
