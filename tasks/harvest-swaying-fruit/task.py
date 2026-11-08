"""On the station table stand two mock tomato plants, 18 cm apart, about 30 cm in front of the arms: each is a green
stem ending in a small square calyx cup, and in each cup sits a tomato (a 6 cm ball, 150 g). The one on the right
(in front of the right arm's side) is ripe red, the one on the left is still green. Both plants keep swaying, as in
wind or on a shaken greenhouse trellis: each cup draws a figure of eight, 7 cm across and 4 cm deep, every 4.8 s,
never stopping, and the two are a quarter of a sway apart. A brown basket stands on the table at the right.

The robot must pick the ripe tomato, following it as it sways, and put it in the basket, leaving the green one in
its cup.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Pick the ripe tomato and put it in the basket. ..." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, lissajous, offset_path, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Sphere  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # score() reads the monitor's flags (kept over the whole trial) and the end state

# -- the fruit and the calyx cups -----------------------------------------------------------------------------------
R = 0.03  # fruit radius: a 6 cm tomato
FRUIT_MASS = 0.15
SEAT_HALF = 0.015  # the cup's square opening is 3 cm across: the fruit rests on its four top inner edges
WALL, WALL_H, FLOOR, GAP = 0.01, 0.02, 0.01, 0.001  # GAP: 1 mm between the driven parts of one plant
SEAT_RISE = float(np.sqrt(R**2 - SEAT_HALF**2))  # 2.6 cm: the seated fruit's centre above the rim
FRUIT_Z = TOP + 0.11  # the seated fruit's centre
RIM = FRUIT_Z - SEAT_RISE
FLOOR_Z = RIM - WALL_H - GAP - FLOOR / 2
STEM = 0.012  # square stem, 1.2 cm
STEM_LOW = TOP + 0.005  # the stem ends 5 mm above the table: carried from below, it never rubs

# -- the sway -------------------------------------------------------------------------------------------------------
SWAY_S = 4.8  # one figure of eight
AMP_X, AMP_Y = 0.02, 0.035  # 4 cm deep (toward and away from the arms), 7 cm across
RAMP_S = 2.0  # the sway grows to full size over the first 2 s
CENTRES = {"ripe": (0.30, -0.09), "green": (0.30, 0.09)}
PHASES = {"ripe": 0.0, "green": SWAY_S / 4}  # the green plant is a quarter of a sway ahead
PATHS = {k: lissajous((*CENTRES[k], FRUIT_Z), AMP_X, AMP_Y, SWAY_S, phase_s=PHASES[k], ramp_s=RAMP_S)
         for k in CENTRES}
STEM_GREEN = (0.25, 0.5, 0.2)
RED = (0.82, 0.12, 0.08)
UNRIPE = (0.45, 0.72, 0.25)


def seat(which: str, t: float) -> np.ndarray:
    """Where the seated fruit's centre is at episode time ``t``."""
    return PATHS[which](t)[0]


def _plant(which: str) -> list[DrivenBox]:
    """Stem, cup floor and four cup walls, each a driven box following the fruit's seat with its own offset."""
    out = SEAT_HALF + WALL / 2
    span = 2 * (SEAT_HALF + WALL)
    wall_dz = RIM - WALL_H / 2 - FRUIT_Z  # each part's centre relative to the seated fruit's
    floor_dz = FLOOR_Z - FRUIT_Z
    stem_high = FLOOR_Z - FLOOR / 2 - GAP
    stem_dz = (STEM_LOW + stem_high) / 2 - FRUIT_Z
    parts = [
        ("stem", (0.0, 0.0, stem_dz), (STEM, STEM, stem_high - STEM_LOW)),
        ("cup_floor", (0.0, 0.0, floor_dz), (span, span, FLOOR)),
        ("cup_wall_near", (-out, 0.0, wall_dz), (WALL, span, WALL_H)),
        ("cup_wall_far", (out, 0.0, wall_dz), (WALL, span, WALL_H)),
        ("cup_wall_right", (0.0, -out, wall_dz), (2 * (SEAT_HALF - GAP), WALL, WALL_H)),
        ("cup_wall_left", (0.0, out, wall_dz), (2 * (SEAT_HALF - GAP), WALL, WALL_H)),
    ]
    boxes = []
    for name, off, size in parts:
        path = offset_path(PATHS[which], off)
        pos = tuple(float(v) for v in path(0.0)[0])
        boxes.append(DrivenBox(f"{which}_{name}", pos=pos, size=tuple(float(v) for v in size), color=STEM_GREEN,
                               path=path))
    return boxes


PLANTS = {k: _plant(k) for k in CENTRES}
FRUIT = {
    "ripe": Sphere("ripe_tomato", pos=tuple(float(v) for v in seat("ripe", 0.0)), radius=R, mass=FRUIT_MASS,
                   color=RED),
    "green": Sphere("green_tomato", pos=tuple(float(v) for v in seat("green", 0.0)), radius=R, mass=FRUIT_MASS,
                    color=UNRIPE),
}

# -- the basket -----------------------------------------------------------------------------------------------------
BASKET_XY = (0.17, -0.38)
BASKET_IN = 0.14  # inner width and depth
B_WALL, B_FLOOR, B_WALL_H = 0.01, 0.01, 0.05
BASKET_FLOOR_TOP = TOP + B_FLOOR
BROWN = (0.55, 0.38, 0.2)


def _basket() -> list[Box]:
    x, y = BASKET_XY
    out = BASKET_IN / 2 + B_WALL / 2
    span = BASKET_IN + 2 * B_WALL
    z = TOP + B_WALL_H / 2
    return [
        Box("basket_floor", pos=(x, y, TOP + B_FLOOR / 2), size=(span, span, B_FLOOR), color=BROWN),
        Box("basket_wall_near", pos=(x - out, y, z), size=(B_WALL, span, B_WALL_H), color=BROWN),
        Box("basket_wall_far", pos=(x + out, y, z), size=(B_WALL, span, B_WALL_H), color=BROWN),
        Box("basket_wall_right", pos=(x, y - out, z), size=(BASKET_IN, B_WALL, B_WALL_H), color=BROWN),
        Box("basket_wall_left", pos=(x, y + out, z), size=(BASKET_IN, B_WALL, B_WALL_H), color=BROWN),
    ]


OBJECTS = [*PLANTS["ripe"], *PLANTS["green"], FRUIT["ripe"], FRUIT["green"], *_basket()]

# -- the monitor ----------------------------------------------------------------------------------------------------
# Every control step the monitor compares each fruit with its own cup (read from the cup floor's measured position,
# so it moves with the sway). A fruit is out of its cup once its centre is more than 1.5 cm sideways from the cup's
# axis or more than 1.5 cm above or below where it sits seated; the ripe fruit is picked once its centre is 3 cm or
# more above its seated height.
OUT_M = 0.015
LIFT_M = 0.03
FLOOR_TO_SEAT = FRUIT_Z - FLOOR_Z  # the seated fruit's centre above the cup floor's centre

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.update(ripe_lifted=False, green_out=False)


reset_state()


def offset_from_seat(fruit, cup_floor) -> tuple[float, float]:
    """(sideways distance from the cup's axis, height above the seated position), in m."""
    d = np.asarray(fruit.pos, float) - np.asarray(cup_floor.pos, float)
    return float(np.hypot(d[0], d[1])), float(d[2] - FLOOR_TO_SEAT)


def in_cup(fruit, cup_floor) -> bool:
    side, up = offset_from_seat(fruit, cup_floor)
    return bool(side <= OUT_M and abs(up) <= OUT_M)


def update(objects) -> None:
    """One control step of the monitor, over the measured object states."""
    _, up = offset_from_seat(objects["ripe_tomato"], objects["ripe_cup_floor"])
    if up >= LIFT_M:
        _STATE["ripe_lifted"] = True
    if not in_cup(objects["green_tomato"], objects["green_cup_floor"]):
        _STATE["green_out"] = True


MONITORED = ("ripe_tomato", "green_tomato", "ripe_cup_floor", "green_cup_floor")


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    update(object_states(environment, list(MONITORED)))


# -- scoring --------------------------------------------------------------------------------------------------------


def in_basket(fruit) -> bool:
    """Resting in the basket: centre between the walls, underside within 1 cm of the floor, still."""
    dx, dy = fruit.pos[0] - BASKET_XY[0], fruit.pos[1] - BASKET_XY[1]
    inside = abs(dx) < BASKET_IN / 2 and abs(dy) < BASKET_IN / 2
    return bool(inside and fruit.low[2] < BASKET_FLOOR_TOP + 0.01 and fruit.resting())


def success(objects):
    """The ripe fruit rests in the basket, and the green one is in its cup and never left it."""
    green_ok = not _STATE["green_out"] and in_cup(objects["green_tomato"], objects["green_cup_floor"])
    return bool(in_basket(objects["ripe_tomato"]) and green_ok)


def distance(objects):
    """Metres from the ripe fruit's centre to where it would rest in the middle of the basket."""
    goal = np.array([*BASKET_XY, BASKET_FLOOR_TOP + R])
    return float(np.linalg.norm(objects["ripe_tomato"].pos - goal))


def score(objects):
    """Partial credit at the end of the trial (success scores 1.0): 0.5 if the ripe fruit rests in the basket but the
    green one was out of its cup at some point; 0.25 if the ripe fruit was ever picked (3 cm above its seat)."""
    if in_basket(objects["ripe_tomato"]):
        return 0.5
    return 0.25 if _STATE["ripe_lifted"] else 0.0


# -- the scripted demo ----------------------------------------------------------------------------------------------
GRIP_ABOVE_CENTRE = 0.003  # TCP this far above the fruit's centre: pad tips 1.0 cm below it, 1.6 cm above the rim
HOVER_Z = TOP + 0.18  # TCP; the open pad tips 2.7 cm above the seated fruit's top
GRASP_Z = FRUIT_Z + GRIP_ABOVE_CENTRE
LIFT_Z = TOP + 0.18  # the held fruit lifted 6.7 cm: its underside above where its top was
FOLLOW_STEP = 0.05  # s, one control step at 20 Hz: a waypoint on the sway every step
LEAD_S = 0.05  # aim one control step ahead, where the fruit will be when the arm gets there
MEET_S = 2.5  # over the ripe fruit at hover height
DESCEND_S = (3.0, 4.4)
CLOSE_S = (4.6, 5.2)
LIFT_S = (5.4, 6.4)
LEAVE_S = 6.4
PLACE_Z = BASKET_FLOOR_TOP + R + 0.01 + GRIP_ABOVE_CENTRE  # the fruit's underside 1 cm above the basket floor
PARK = (0.17, -0.30, TOP + 0.18)


def _ease(a: float, b: float, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return a + (b - a) * (1 - np.cos(np.pi * u)) / 2


def follow_z(t: float) -> float:
    """The TCP height while following the ripe fruit."""
    if t < LIFT_S[0]:
        return _ease(HOVER_Z, GRASP_Z, (t - DESCEND_S[0]) / (DESCEND_S[1] - DESCEND_S[0]))
    return _ease(GRASP_Z, LIFT_Z, (t - LIFT_S[0]) / (LIFT_S[1] - LIFT_S[0]))


def follow_finger(t: float) -> float:
    return _ease(FINGER_OPEN, FINGER_CLOSED, (t - CLOSE_S[0]) / (CLOSE_S[1] - CLOSE_S[0]))


def _over_fruit(t: float, z: float) -> tuple[float, float, float]:
    x, y, _ = seat("ripe", t + LEAD_S)
    return (float(x), float(y), float(z))


def demo():
    """Right arm: over the ripe fruit by 2.5 s, then a waypoint on its sway every control step: descend around it
    (3.0 to 4.4 s), close (4.6 to 5.2 s), lift it 6.7 cm (5.4 to 6.4 s), all while following. Then carry it to the
    basket, lower it to 1 cm above the floor, let go and park."""
    bx, by = BASKET_XY
    plan = [Waypoint(MEET_S, _over_fruit(MEET_S, HOVER_Z), FINGER_OPEN, "over the ripe fruit")]
    n = round((LEAVE_S - MEET_S) / FOLLOW_STEP)
    for k in range(1, n + 1):
        t = MEET_S + k * FOLLOW_STEP
        label = "follow" if t < DESCEND_S[0] else "descend" if t <= DESCEND_S[1] + 1e-9 else "close" \
            if t <= CLOSE_S[1] + 1e-9 else "hold" if t < LIFT_S[0] - 1e-9 else "lift"
        plan.append(Waypoint(FOLLOW_STEP, _over_fruit(t, follow_z(t)), float(follow_finger(t)), label))
    plan += [
        Waypoint(1.8, (bx, by, LIFT_Z), FINGER_CLOSED, "carry"),
        Waypoint(1.2, (bx, by, PLACE_Z), FINGER_CLOSED, "lower"),
        Waypoint(0.6, (bx, by, PLACE_Z), FINGER_OPEN, "release"),
        Waypoint(1.0, PARK, FINGER_OPEN, "park"),
    ]
    return {"right": plan}
