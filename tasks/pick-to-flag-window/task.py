"""On the station table, a row of four shallow grey bins stands 33 cm in front of the arms, two in front of each arm.
Each bin holds one orange wooden cube (4 cm). Behind the row, a dark screen (10 cm tall) runs across the table, and
behind it each bin has a green signal flag (10 x 8 cm plate), hidden behind the screen while it is down. A grey tote
stands nearer the arms, on the midline. The bins are numbered 1 to 4 from the robot's left.

Like a pick-to-light rack, the flags call for parts: a flag pops up above the screen in 0.3 s, stays up for 4 s and
goes back down. The order is fixed: bin 3's flag rises at 2 s, bin 1's at 4 s, bin 4's at 10 s and bin 2's at
12 s, each once (so bins 3 and 1, and bins 4 and 2, are up together for about 2 s). The robot must take a part from
a bin only while that bin's flag is up, and drop it in the tote.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Each bin has a green flag behind it that pops up for a few seconds when its part is \\
           wanted. Take the part from a bin only while that bin's flag is up, and drop it in the grey tote. ..." ...
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, on_reset, on_step, pop_up  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # score() latches each bin's pick outcome in the monitor; where the parts lie counts at the end

# -- the bins and the parts ----------------------------------------------------------------------------------------
BIN_X = 0.33  # the row's centre line, metres in front of the arm bases
BIN_Y = (0.40, 0.21, -0.21, -0.40)  # bins 1 to 4, from the robot's left (y is to the left); arm bases at y +-0.305
ARM_OF = ("left", "left", "right", "right")
BIN_IN = 0.15  # inner width and depth: open fingers (94 mm, plus the finger pads) stay 1.5 cm or more inside the lip
BIN_WALL, BIN_FLOOR, BIN_H = 0.01, 0.01, 0.015  # a 5 mm lip above the floor: the fingers stay 1.5 cm above it
BIN_FLOOR_TOP = TOP + BIN_FLOOR
GREY = (0.55, 0.57, 0.6)
PART = 0.04  # cube edge
PARTS = [
    Box(f"part_{i + 1}", pos=(BIN_X, y, BIN_FLOOR_TOP + PART / 2), size=(PART, PART, PART), mass=0.04,
        color=(0.95, 0.45, 0.1))
    for i, y in enumerate(BIN_Y)
]


def _open_box(name, x, y, inner, wall, floor, height, color) -> list[Box]:
    """A square open box (bin or tote) standing on the table: ``inner`` is its inside (x, y) size."""
    ix, iy = inner
    sx, sy = ix + 2 * wall, iy + 2 * wall
    z = TOP + height / 2
    return [
        Box(f"{name}_floor", pos=(x, y, TOP + floor / 2), size=(sx, sy, floor), color=color),
        Box(f"{name}_wall_near", pos=(x - ix / 2 - wall / 2, y, z), size=(wall, sy, height), color=color),
        Box(f"{name}_wall_far", pos=(x + ix / 2 + wall / 2, y, z), size=(wall, sy, height), color=color),
        Box(f"{name}_wall_right", pos=(x, y - iy / 2 - wall / 2, z), size=(ix, wall, height), color=color),
        Box(f"{name}_wall_left", pos=(x, y + iy / 2 + wall / 2, z), size=(ix, wall, height), color=color),
    ]


BINS = [b for i, y in enumerate(BIN_Y) for b in _open_box(f"bin_{i + 1}", BIN_X, y, (BIN_IN, BIN_IN), BIN_WALL,
                                                          BIN_FLOOR, BIN_H, GREY)]

# -- the tote ------------------------------------------------------------------------------------------------------
TOTE_XY = (0.15, 0.0)
TOTE_IN = (0.12, 0.43)  # inside depth (x) and length (y): room for four cubes set side by side, 5.5 cm apart
TOTE_WALL, TOTE_FLOOR, TOTE_H = 0.01, 0.01, 0.05
TOTE_FLOOR_TOP = TOP + TOTE_FLOOR
TOTE = _open_box("tote", *TOTE_XY, TOTE_IN, TOTE_WALL, TOTE_FLOOR, TOTE_H, (0.25, 0.27, 0.3))

# -- the screen and the flags --------------------------------------------------------------------------------------
SCREEN_X = 0.445  # its near face 2.5 cm behind the bins' far walls
SCREEN = (0.01, 0.98, 0.10)  # depth, length (across all four bins), height
SCREEN_TOP = TOP + SCREEN[2]
SCREEN_BOX = Box("screen", pos=(SCREEN_X, 0.0, TOP + SCREEN[2] / 2), size=SCREEN, color=(0.12, 0.12, 0.14))
FLAG = (0.01, 0.10, 0.08)  # depth, width, height
FLAG_X = SCREEN_X + SCREEN[0] / 2 + 0.01 + FLAG[0] / 2  # 1 cm behind the screen, never touching it
FLAG_DOWN_Z = TOP + 0.012 + FLAG[2] / 2  # down: 1.2 cm above the table, its top 0.8 cm below the screen's
FLAG_RISE = 0.08  # up: its top 7.2 cm above the screen's
RAISED_M = 0.05  # a flag counts as up while its top is 5 cm or more above the screen's top
MOVE_S, UP_S = 0.3, 4.0  # it rises in 0.3 s, stays fully up 4 s and drops in 0.3 s
RISE_AT = (4.0, 12.0, 2.0, 10.0)  # when each bin's flag starts rising (bins 1 to 4), s from the start; once each
FLAG_PATHS = [pop_up((FLAG_X, y, FLAG_DOWN_Z), (FLAG_X, y, FLAG_DOWN_Z + FLAG_RISE), t, MOVE_S, UP_S)
              for y, t in zip(BIN_Y, RISE_AT, strict=True)]
FLAGS = [DrivenBox(f"flag_{i + 1}", pos=(FLAG_X, y, FLAG_DOWN_Z), size=FLAG, color=(0.1, 0.85, 0.25), path=path)
         for i, (y, path) in enumerate(zip(BIN_Y, FLAG_PATHS, strict=True))]

OBJECTS = [*BINS, *PARTS, *TOTE, SCREEN_BOX, *FLAGS]

# -- the monitor: when each part left its bin, and whether its flag was up then ---------------------------------------
LIFT_M = 0.03  # a part has left its bin once its underside is 3 cm or more above the bin floor ...
# ... or its centre is outside the bin's inside (pushed or knocked out)
TAKEN, PUSHED_OUT, WRONG = "taken", "pushed_out", "wrong"
GOOD_IN_TOTE, GOOD_ELSEWHERE, WRONG_PICK = 1.0, 0.5, -0.5  # points per bin; the score is their sum over 4, in 0..1


def flag_up(flag) -> bool:
    """The flag's top is 5 cm or more above the screen's top."""
    return bool(flag.high[2] >= SCREEN_TOP + RAISED_M)


def in_bin(i: int, part) -> bool:
    """Part ``i`` (0-based) still in its bin: centre inside the bin's walls, underside under 3 cm above its floor."""
    dx, dy = part.pos[0] - BIN_X, part.pos[1] - BIN_Y[i]
    return bool(abs(dx) < BIN_IN / 2 and abs(dy) < BIN_IN / 2 and part.low[2] < BIN_FLOOR_TOP + LIFT_M)


_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.update(left=[None] * len(BIN_Y))  # per bin: None (still in), TAKEN, PUSHED_OUT or WRONG


reset_state()


def update(parts, flags) -> None:
    """One control step of the monitor: latch how each part left its bin, the first step it is out.
    ``parts`` and ``flags`` are the four parts' and four flags' states, bins 1 to 4."""
    for i, (part, flag) in enumerate(zip(parts, flags, strict=True)):
        if _STATE["left"][i] is not None or in_bin(i, part):
            continue
        if not flag_up(flag):
            _STATE["left"][i] = WRONG
        elif part.low[2] >= BIN_FLOOR_TOP + LIFT_M:
            _STATE["left"][i] = TAKEN
        else:
            _STATE["left"][i] = PUSHED_OUT  # slid over the lip, not lifted: no credit, and nothing left to pick


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    names = [p.name for p in PARTS] + [f.name for f in FLAGS]
    st = object_states(environment, names)
    update([st[p.name] for p in PARTS], [st[f.name] for f in FLAGS])


def outcomes() -> list[str | None]:
    return list(_STATE["left"])


# -- scoring -------------------------------------------------------------------------------------------------------


def in_tote(part) -> bool:
    """Lying in the tote: centre inside its walls, underside within 1 cm of its floor, and still."""
    dx, dy = part.pos[0] - TOTE_XY[0], part.pos[1] - TOTE_XY[1]
    return bool(abs(dx) < TOTE_IN[0] / 2 and abs(dy) < TOTE_IN[1] / 2 and part.low[2] < TOTE_FLOOR_TOP + 0.01
                and part.resting())


def _points(objects) -> list[float]:
    out = []
    for i, part in enumerate(PARTS):
        how = _STATE["left"][i]
        if how == WRONG:
            out.append(WRONG_PICK)
        elif how == TAKEN:
            out.append(GOOD_IN_TOTE if in_tote(objects[part.name]) else GOOD_ELSEWHERE)
        else:
            out.append(0.0)
    return out


def success(objects):
    """Every part was taken while its own flag was up, and all four lie still in the tote."""
    return bool(all(p == GOOD_IN_TOTE for p in _points(objects)))


def distance(objects):
    """Metres, summed over the parts not yet in the tote, from each to the middle of the tote's floor."""
    goal = np.array([*TOTE_XY, TOTE_FLOOR_TOP + PART / 2])
    return float(sum(np.linalg.norm(objects[p.name].pos - goal) for p in PARTS if not in_tote(objects[p.name])))


def score(objects):
    """Per bin: 1 if its part was taken (lifted out) while its flag was up and now lies in the tote; 0.5 if taken
    while its flag was up but not in the tote; -0.5 if the part left its bin in any way while its flag was down
    (a wrong pick); 0 otherwise (never taken, or pushed out over the lip while its flag was up). The score is the sum
    over the four bins divided by 4, clipped to 0..1."""
    out = float(np.clip(sum(_points(objects)) / len(PARTS), 0.0, 1.0))
    if DEBUG:
        _debug(objects, out)
    return out


DEBUG = os.environ.get("PICK_DEBUG", "") not in ("", "0")
_DEBUG = {"calls": 0, "last": None}


def _debug(objects, out) -> None:
    """PICK_DEBUG=1: print each bin's outcome and every cube's position (cm, station frame) whenever an outcome
    changes, and once a second (every 20 scored steps), so the end of a run shows where everything ended up."""
    _DEBUG["calls"] += 1
    key = tuple(_STATE["left"])
    if key == _DEBUG["last"] and _DEBUG["calls"] % 20:
        return
    _DEBUG["last"] = key
    from motionkit import clock

    parts = []
    for i, p in enumerate(PARTS):
        st = objects[p.name]
        x, y, z = (100 * st.pos).round(1)
        parts.append(f"bin{i + 1}={_STATE['left'][i]} ({x:.1f},{y:.1f},{100 * (st.low[2] - TOP):.1f}) "
                     f"tote={in_tote(st)}")
    print(f"[pick-debug] t={clock():.2f} score={out:.3f} " + " | ".join(parts), flush=True)


# -- the scripted demo ---------------------------------------------------------------------------------------------
HOVER_Z = TOP + 0.16  # TCP height for waiting and carrying: a held cube's underside 12 cm above the table
PRE_Z = TOP + 0.10  # above a bin, ready to go down: the open pad tips 4.5 cm over the cube's top
GRASP_Z = BIN_FLOOR_TOP + PART / 2 + 0.013  # TCP 1.3 cm above the cube's centre: pad tips 1.5 cm over the walls
# Each cube has its own spot along the tote, bins 1 to 4: each arm fills its own half. Spots are 9.5 cm apart (cubes
# 5.5 cm apart), so the open fingers (94 mm apart) clear a cube already set down by 1.8 cm, and the outer cubes stand
# 5.25 cm from the end walls. The two arms visit the tote at overlapping times (bins 3 and 1, then 4 and 2), so each
# such pair goes to spots 19 cm apart: bin 3 outer right with bin 1 inner left, bin 4 inner right with bin 2 outer left.
SPOT_Y = (0.0475, 0.1425, -0.1425, -0.0475)
SET_Z = TOTE_FLOOR_TOP + 0.01 + PART / 2 + 0.013  # let go with the cube's underside 1 cm above the tote floor
REACT_S = 0.3  # the demo starts each pick 0.3 s after its flag counts as up, as a reactive policy would


def raised_from(i: int) -> float:
    """When flag ``i`` (0-based) first counts as up: its top RAISED_M above the screen (on the rising cosine)."""
    s = (SCREEN_TOP + RAISED_M - (FLAG_DOWN_Z + FLAG[2] / 2)) / FLAG_RISE
    return RISE_AT[i] + MOVE_S * float(np.arccos(1 - 2 * s)) / np.pi


def raised_until(i: int) -> float:
    """When flag ``i`` (0-based) stops counting as up, on the way down."""
    return 2 * RISE_AT[i] + 2 * MOVE_S + UP_S - raised_from(i)


def _neutral(side: str) -> tuple[float, float, float]:
    """Where an arm waits: above the row, between its two bins."""
    ys = [y for y, s in zip(BIN_Y, ARM_OF, strict=True) if s == side]
    return (BIN_X, float(np.mean(ys)), HOVER_Z)


def _pick(i: int, side: str) -> list[Waypoint]:
    x, y = BIN_X, BIN_Y[i]
    tx, ty = TOTE_XY[0], SPOT_Y[i]
    n = f"bin {i + 1}"
    return [
        Waypoint(0.7, (x, y, PRE_Z), FINGER_OPEN, f"over {n}"),
        Waypoint(0.5, (x, y, GRASP_Z), FINGER_OPEN, f"descend into {n}"),
        Waypoint(0.6, (x, y, GRASP_Z), FINGER_CLOSED, f"close on {n}"),
        Waypoint(0.7, (x, y, HOVER_Z), FINGER_CLOSED, f"lift from {n}"),
        Waypoint(1.2, (tx, ty, HOVER_Z), FINGER_CLOSED, f"carry {n}"),
        Waypoint(1.2, (tx, ty, SET_Z), FINGER_CLOSED, f"lower {n}"),
        Waypoint(0.3, (tx, ty, SET_Z), FINGER_CLOSED, f"settle {n}"),
        Waypoint(0.6, (tx, ty, SET_Z), FINGER_OPEN, f"drop {n}"),
        Waypoint(0.7, (tx, ty, HOVER_Z), FINGER_OPEN, f"up from {n}"),
        Waypoint(1.0, _neutral(side), FINGER_OPEN, f"back from {n}"),
    ]


def demo():
    """Each arm waits above its two bins and, 0.3 s after one of its flags comes up, picks that bin's cube (lifted
    out about 2.4 s after the flag counts as up, of the 4.2 s it stays up), sets it down on its own spot in the tote
    (let go 1 cm above the floor, fingers open before rising) and goes back to wait. Nothing is started before
    its flag is up, so the plan only needs what a camera shows."""
    plans = {}
    for side in ("left", "right"):
        plan = [Waypoint(1.5, _neutral(side), FINGER_OPEN, "to the waiting point")]
        for i in sorted((i for i in range(len(BIN_Y)) if ARM_OF[i] == side), key=raised_from):
            start = raised_from(i) + REACT_S
            now = sum(w.duration for w in plan)
            plan.append(Waypoint(start - now, _neutral(side), FINGER_OPEN, f"wait for flag {i + 1}"))
            plan += _pick(i, side)
        plans[side] = plan
    return plans
