"""On the station table, in front of the right arm, stands a grey steel-topped bench (23 x 22 cm, 8 cm tall). Beyond
its far edge, across a 1.5 cm gap, is the yellow deck (18 x 18 x 2 cm) of a small automatic lift, with a dark mast
behind it. The lift never stops for the robot: it serves three levels on a fixed 18 s cycle, waiting 3 s at each
stop and taking 1.5 s between stops: low (5.4 cm below the bench top), level with the bench (4 mm below its top),
high (4.6 cm above it), level again, and back to low. It starts at its low stop.

A brown cardboard carton (12 x 12 x 6 cm, 300 g) stands on the bench, its far side 3 cm from the bench's far edge.
It is wider than the open gripper (9.4 cm), so it is slid, not carried.

The robot must slide the carton off the bench onto the lift's deck while the deck waits level with the bench, so it
ends standing on the deck. Pushed while the deck is low, the carton tips over the bench edge into the gap; pushed
while the deck is high, it runs into the deck's side; left half across when the deck moves, it is levered over.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Slide the carton onto the lift's deck while the deck is level with the bench. The lift keeps cycling and won't stop for you." ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import motionkit  # noqa: E402
import numpy as np  # noqa: E402
from motionkit import DrivenBox, SlickBox, dwell_stops, dwell_windows  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": a carton that tips after reaching a milestone scores 0, which "best" can't express. The best milestone is
# kept by the MONITOR below instead, and cleared at every episode start (motionkit.on_reset).
SCORE_OVER = "end"
RUN_STEPS = 1200  # 60 s at 20 Hz


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


LINE_Y = -0.26  # the push line, metres to the robot's left: 4.5 cm inside the right arm's base
STEEL = (0.4, 0.3)  # static, dynamic friction of the bench top and the deck; "min" combined, so they rule

# -- the bench -----------------------------------------------------------------------------------------------------
BENCH_X = (0.10, 0.33)  # near and far edges, metres in front of the arm bases
BENCH_W, BENCH_H = 0.22, 0.08
BENCH_TOP = TOP + BENCH_H
EDGE_X = BENCH_X[1]
BENCH = SlickBox("bench", pos=_floats((sum(BENCH_X) / 2, LINE_Y, TOP + BENCH_H / 2)),
                 size=_floats((BENCH_X[1] - BENCH_X[0], BENCH_W, BENCH_H)), color=(0.55, 0.56, 0.58), friction=STEEL)

# -- the lift ------------------------------------------------------------------------------------------------------
GAP = 0.015  # between the bench's far edge and the deck
DECK = (0.18, 0.18, 0.02)
DECK_X = (EDGE_X + GAP, EDGE_X + GAP + DECK[0])
DECK_CX = sum(DECK_X) / 2
STEP = 0.004  # level: the deck's top this far below the bench top, so the carton's leading edge drops onto it
RISE = 0.05  # the low and high stops, below and above level
LEVEL_TOP = BENCH_TOP - STEP
LOW_TOP, HIGH_TOP = LEVEL_TOP - RISE, LEVEL_TOP + RISE
DWELL_S, TRAVEL_S = 3.0, 1.5


def _deck_centre(top: float) -> tuple[float, float, float]:
    return _floats((DECK_CX, LINE_Y, top - DECK[2] / 2))


STOPS = [_deck_centre(LOW_TOP), _deck_centre(LEVEL_TOP), _deck_centre(HIGH_TOP), _deck_centre(LEVEL_TOP)]
LEVEL_STOPS = (1, 3)
CYCLE_S = len(STOPS) * (DWELL_S + TRAVEL_S)
PATH = dwell_stops(STOPS, TRAVEL_S, DWELL_S)
RUN_S = RUN_STEPS / 20


def level_windows(until: float = RUN_S) -> list[tuple[float, float]]:
    """Every (start, end) of a wait level with the bench, in episode time, up to ``until``."""
    out = [w for k in LEVEL_STOPS for w in dwell_windows(STOPS, TRAVEL_S, DWELL_S, k, until)]
    return sorted(out)


LIFT = DrivenBox("lift_deck", pos=STOPS[0], size=DECK, color=(0.95, 0.75, 0.1), path=PATH, friction=STEEL,
                 combine="min")
MAST_GAP, MAST_T, MAST_H = 0.01, 0.03, 0.22
MAST = Box("lift_mast", pos=_floats((DECK_X[1] + MAST_GAP + MAST_T / 2, LINE_Y, TOP + MAST_H / 2)),
           size=_floats((MAST_T, DECK[1], MAST_H)), color=(0.2, 0.2, 0.22))

# -- the carton ----------------------------------------------------------------------------------------------------
CARTON = (0.12, 0.12, 0.06)
CARTON_MASS = 0.3
START_X = EDGE_X - 0.03 - CARTON[0] / 2  # its far side 3 cm from the bench's far edge
START = np.array([START_X, LINE_Y, BENCH_TOP + CARTON[2] / 2])
CARTON_SPEC = Box("carton", pos=_floats(START), size=CARTON, mass=CARTON_MASS, color=(0.62, 0.45, 0.28))

OBJECTS = [BENCH, LIFT, MAST, CARTON_SPEC]

# -- scoring -------------------------------------------------------------------------------------------------------
UPRIGHT_DEG = 15.0  # tilted more than this at any time: tipped, 0 for the rest of the trial
SEAT_M = 0.01  # its underside within this of a surface: standing on it
OVERLAP_M = 0.02  # this much of its underside over the deck: partly on it
MOVED_M = 0.05  # its centre this far toward the lift: moved
REL_SPEED = 0.02  # m/s: at rest relative to the deck


def _rot(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def bottom_corners(c) -> np.ndarray:
    """The carton's four bottom corners (world frame), from its centre and orientation."""
    hx, hy, hz = (s / 2 for s in CARTON)
    local = np.array([[sx * hx, sy * hy, -hz] for sx in (-1, 1) for sy in (-1, 1)])
    return np.asarray(c.pos, float) + local @ _rot(c.quat_xyzw).T


def tilt_deg(c) -> float:
    """Angle between the carton's up axis and straight up."""
    return float(np.degrees(np.arccos(np.clip(_rot(c.quat_xyzw)[2, 2], -1.0, 1.0))))


def _deck_top(deck) -> float:
    return float(deck.pos[2] + DECK[2] / 2)


def _inside_deck(xy, deck) -> bool:
    lo = np.asarray(deck.pos[:2]) - np.array(DECK[:2]) / 2
    hi = np.asarray(deck.pos[:2]) + np.array(DECK[:2]) / 2
    return bool(np.all(xy >= lo) and np.all(xy <= hi))


def fully_on_deck(c, deck) -> bool:
    """Upright, all four bottom corners over the deck, its underside within 1 cm of the deck's top."""
    corners = bottom_corners(c)
    return bool(tilt_deg(c) <= UPRIGHT_DEG and all(_inside_deck(p[:2], deck) for p in corners)
                and abs(corners[:, 2].min() - _deck_top(deck)) < SEAT_M)


def partly_on_deck(c, deck) -> bool:
    """Upright, at least 2 cm of its underside over the deck (along the push), its underside within 1 cm above the
    deck's top, and its centre within the deck's width."""
    corners = bottom_corners(c)
    over = corners[:, 0].max() - deck.pos[0] + DECK[0] / 2 >= OVERLAP_M
    beside = abs(c.pos[1] - deck.pos[1]) <= DECK[1] / 2
    return bool(tilt_deg(c) <= UPRIGHT_DEG and over and beside
                and abs(corners[:, 2].min() - _deck_top(deck)) < SEAT_M)


def moved_on_bench(c) -> bool:
    """Upright on the bench, its centre over it and at least 5 cm toward the lift from where it started."""
    corners = bottom_corners(c)
    over = BENCH_X[0] < c.pos[0] < EDGE_X and abs(c.pos[1] - LINE_Y) < BENCH_W / 2
    return bool(tilt_deg(c) <= UPRIGHT_DEG and over and abs(corners[:, 2].min() - BENCH_TOP) < SEAT_M
                and c.pos[0] - START[0] >= MOVED_M)


def at_rest_on(c, deck) -> bool:
    return bool(np.linalg.norm(np.asarray(c.lin_vel) - np.asarray(deck.lin_vel)) < REL_SPEED
                and np.linalg.norm(c.ang_vel) < 0.5)


class Monitor:
    """What the carton has done so far this episode: whether it has tipped or fallen, and the best milestone."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.spoiled, self.best = False, 0.0

    def update(self, c, deck) -> float:
        """Take one control step's carton and deck states; return the score now."""
        if tilt_deg(c) > UPRIGHT_DEG or bottom_corners(c)[:, 2].min() < LOW_TOP - SEAT_M:
            self.spoiled = True  # tipped, or fallen below the lowest surface it could stand on (table, gap)
        now = 0.0
        if partly_on_deck(c, deck):  # includes wholly on it but not (yet) at rest; success scores 1.0
            now = 0.5
        elif moved_on_bench(c):
            now = 0.25
        self.best = max(self.best, now)
        return 0.0 if self.spoiled else self.best


MONITOR = Monitor()
motionkit.on_reset(MONITOR.reset)


def success(objects):
    """Standing on the deck, all of its underside over the deck, at rest relative to it, and never tipped."""
    c, deck = objects["carton"], objects["lift_deck"]
    return bool(not MONITOR.spoiled and fully_on_deck(c, deck) and at_rest_on(c, deck))


def distance(objects):
    """Metres (horizontally) from the carton's centre to the deck's centre."""
    return float(np.linalg.norm(objects["carton"].pos[:2] - objects["lift_deck"].pos[:2]))


def score(objects):
    """0 once the carton has tipped more than 15 degrees or fallen (its underside 1 cm below the deck's low stop), for
    the rest of the trial. Otherwise the best reached so far: 0.5 once it stood upright partly on the deck (2 cm or
    more of its underside over it); 0.25 once it stood upright on the bench at least 5 cm toward the lift from its
    start. Success (wholly on the deck, at rest on it, for 1 s) scores 1.0."""
    return MONITOR.update(objects["carton"], objects["lift_deck"])


# -- the scripted demo ---------------------------------------------------------------------------------------------
# The right arm pushes the carton's near side with its closed fingers, turned a quarter turn so they close along the
# push (as in the sliding-carton task, where the contact sat 1.8 cm ahead of the fingers' centre line in Isaac Sim).
FINGER_HALF = 0.018
YAW = math.pi / 2
PUSH_Z = BENCH_TOP + PAD_TIP_PAST_TCP + 0.006  # pad tips 6 mm above the bench top (1 cm above the level deck)
CLEAR_Z = BENCH_TOP + CARTON[2] + PAD_TIP_PAST_TCP + 0.03  # pad tips 3 cm above the carton's top
FINGER_GAP = 0.015  # between the fingers and the carton's near side before the push
PUSH_FROM_X = START[0] - CARTON[0] / 2 - FINGER_GAP - FINGER_HALF
END_CX = DECK_CX + 0.005  # where the push leaves the carton's centre: 5 mm past the deck's centre
PUSH_TO_X = END_CX - CARTON[0] / 2 - FINGER_HALF
BACK_X = EDGE_X - 0.03  # backed off: the fingers' front 1.2 cm short of the bench edge
WINDOW = level_windows()[0]  # the first wait level with the bench: 4.5 to 7.5 s
PUSH_AT = WINDOW[0] + 0.1  # the deck has stopped
PUSH_S, SETTLE_S, BACK_S = 2.0, 0.2, 0.5


def demo():
    """Right arm: fingers down behind the carton by 3.2 s while the deck is still on its way up from the low stop, wait,
    push it 22 cm across onto the deck from 4.6 to 6.6 s while the deck waits level (4.5 to 7.5 s), back off past the
    bench edge by 7.3 s, before the deck leaves for its high stop, and lift clear."""
    y = LINE_Y
    plan = [
        Waypoint(2.0, (PUSH_FROM_X, y, CLEAR_Z), FINGER_CLOSED, "above, behind the carton", YAW),
        Waypoint(1.2, (PUSH_FROM_X, y, PUSH_Z), FINGER_CLOSED, "down behind it", YAW),
    ]
    plan.append(Waypoint(PUSH_AT - sum(w.duration for w in plan), (PUSH_FROM_X, y, PUSH_Z), FINGER_CLOSED,
                         "wait for the deck to be level", YAW))
    plan += [
        Waypoint(PUSH_S, (PUSH_TO_X, y, PUSH_Z), FINGER_CLOSED, "push onto the deck", YAW),
        Waypoint(SETTLE_S, (PUSH_TO_X, y, PUSH_Z), FINGER_CLOSED, "settle", YAW),
        Waypoint(BACK_S, (BACK_X, y, PUSH_Z), FINGER_CLOSED, "back off the deck", YAW),
        Waypoint(0.8, (BACK_X, y, CLEAR_Z), FINGER_CLOSED, "lift clear", YAW),
    ]
    return {"right": plan}
