"""On the station table, a conveyor line runs across the table from right to left (toward the left arm), 38 cm in front
of the arms: a black track bed (10 cm wide, 1.5 cm tall) with three flat grey carrier plates (8 x 8 x 1.5 cm) riding
on it, 28 cm apart. Each carrier holds one painted wooden cube (5 cm): red in front (12 cm left of the midpoint
between the arms), then yellow, then blue. The line starts with the trial and never stops for the robot: it reaches
4 cm/s within 0.5 s, so a new block comes past the left arm every 7 s. A red end line is taped across the table on
both sides of the track, 48 cm left of the midpoint; past it the line carries the blocks out of reach and off the
end of the table. A green square pad (8 x 8 x 2 cm) is fixed to the table between the line and the left arm.

The robot must build a tower of the three blocks on the green pad, each block standing squarely on the one before,
taking each block off the line before it passes the end line. A block is lost once its centre passes the end line
or it touches the table; lost blocks don't count, even if they end up on the tower.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Stack the three blocks into a tower on the green pad as they arrive on the conveyor. ..." ...
"""

from __future__ import annotations

import itertools
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, belt_run, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import TABLE_CENTER, TABLE_SIZE  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the tower is scored as it stands at the end of the trial (a tower knocked over scores what is left of it).
# Lost blocks are tracked over the whole trial by the monitor below, cleared at every episode start.
SCORE_OVER = "end"
TABLE_LEFT_Y = TABLE_CENTER[1] + TABLE_SIZE[1] / 2  # 0.65: the table's left edge (y is to the robot's left)


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the conveyor line ---------------------------------------------------------------------------------------------
LINE_X = 0.38  # the line's centre, metres in front of the arm bases
BED = (0.10, 1.24, 0.015)  # the fixed track bed: width (x), length (y), height; it ends 3 cm inside each table edge
CARRIER = (0.08, 0.08, 0.015)
CARRIER_Z = TOP + BED[2] + 0.002 + CARRIER[2] / 2  # 2 mm above the bed: carriers ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
SPEED = 0.04  # m/s, toward +y (the left arm)
RAMP_S = 0.5  # the line speeds up from rest over this long at the start of the trial
SPACING = 0.28  # between carriers: one every SPACING / SPEED = 7 s
FIRST_Y = 0.12  # the first carrier's centre at the start
END_Y = 0.48  # the end line: a block whose centre passes it is lost
TURN_Y = TABLE_LEFT_Y + CARRIER[1] / 2 + 0.02  # 0.71: past the table's edge, carriers turn down out of the way
DROP, DROP_SPEED = 0.30, 0.15
BED_BOX = Box("track_bed", pos=(LINE_X, 0.0, TOP + BED[2] / 2), size=BED, color=(0.08, 0.08, 0.09))

# -- the blocks ----------------------------------------------------------------------------------------------------
CUBE = 0.05
CUBE_MASS = 0.08  # painted wood, about 640 kg/m^3
COLORS = {"red": (0.85, 0.15, 0.12), "yellow": (0.95, 0.78, 0.1), "blue": (0.15, 0.32, 0.8)}
NAMES = tuple(f"block_{c}" for c in COLORS)  # in arrival order
STARTS = {n: FIRST_Y - i * SPACING for i, n in enumerate(NAMES)}  # each block's starting y
PATHS = {n: belt_run((LINE_X, y, CARRIER_Z), SPEED, RAMP_S, TURN_Y, DROP, DROP_SPEED) for n, y in STARTS.items()}
CARRIERS = [
    DrivenBox(f"carrier_{c}", pos=_floats((LINE_X, STARTS[n], CARRIER_Z)), size=CARRIER, color=(0.5, 0.52, 0.55),
              path=PATHS[n])
    for c, n in zip(COLORS, NAMES, strict=True)
]
BLOCKS = [
    Box(n, pos=_floats((LINE_X, STARTS[n], CARRIER_TOP + CUBE / 2 + 0.001)), size=(CUBE, CUBE, CUBE), mass=CUBE_MASS,
        color=COLORS[c])
    for c, n in zip(COLORS, NAMES, strict=True)
]


def carrier_y(name: str, t: float) -> float:
    """Where the carrier of block ``name`` is along the line (its centre's y) at episode time ``t``."""
    return float(PATHS[name](t)[0][1])


def arrival(name: str, y: float) -> float:
    """When the carrier of block ``name`` reaches ``y`` (after the ramp; the line runs at constant speed then)."""
    return float((y - STARTS[name]) / SPEED + RAMP_S / 2)


# -- the end line (red tape on the table, either side of the track) --------------------------------------------------
TAPE = (0.06, 0.012, 0.002)
TAPES = [
    Box(f"end_tape_{side}", pos=(LINE_X + sign * (BED[0] / 2 + 0.005 + TAPE[0] / 2), END_Y, TOP + TAPE[2] / 2),
        size=TAPE, color=(0.9, 0.05, 0.05))
    for side, sign in (("near", -1), ("far", 1))
]

# -- the pad -------------------------------------------------------------------------------------------------------
PAD_XY = (0.20, 0.30)  # in front of the left arm (base at y 0.305), between it and the line
PAD = (0.08, 0.08, 0.02)
PAD_TOP = TOP + PAD[2]
PAD_BOX = Box("pad", pos=(*PAD_XY, TOP + PAD[2] / 2), size=PAD, color=(0.2, 0.6, 0.3))

OBJECTS = [BED_BOX, *TAPES, PAD_BOX, *CARRIERS, *BLOCKS]

# -- the lost-block monitor ----------------------------------------------------------------------------------------
TABLE_TOUCH = 0.01  # a block's underside less than this above the table top: it has touched (been set on) the table
SEAT = 0.01  # a block's underside within this of the surface below it: resting on it
ALIGN = 0.015  # full score: each block's centre within this (horizontally) of the centre of the one below
TILT_DEG = 10.0  # a tower block stands upright: its axes within this of square to the table
CLEAR_M = 0.08  # full score: both grippers' grasp points at least this far from every tower block's centre

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.update(lost={}, tcps=None)


reset_state()


def update(blocks: dict, tcps) -> None:
    """One control step of the monitor: mark blocks lost (for good), and keep the grippers' grasp points."""
    for name in NAMES:
        b = blocks[name]
        if name in _STATE["lost"]:
            continue
        if b.pos[1] >= END_Y:
            _STATE["lost"][name] = "passed the end line"
        elif b.low[2] < TOP + TABLE_TOUCH:
            _STATE["lost"][name] = "touched the table"
    _STATE["tcps"] = None if tcps is None else [np.asarray(p, float) for p in tcps]


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.scene import object_states

    tcps = [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0] for side in ("left", "right")]
    update(object_states(environment, list(NAMES)), tcps)


def lost() -> dict[str, str]:
    """The blocks lost so far, with why."""
    return dict(_STATE["lost"])


# -- the tower -----------------------------------------------------------------------------------------------------


def _upright(b) -> bool:
    """The cube's up axis is within TILT_DEG of vertical (a cube has no wrong way up)."""
    x, y, z, w = (float(v) for v in b.quat_xyzw)
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    return bool(np.max(np.abs(rot[2])) >= math.cos(math.radians(TILT_DEG)))


def _on_pad(b) -> bool:
    """Resting on the pad: centre over the pad, underside within SEAT of its top, upright."""
    dx, dy = b.pos[0] - PAD_XY[0], b.pos[1] - PAD_XY[1]
    return bool(abs(dx) <= PAD[0] / 2 and abs(dy) <= PAD[1] / 2 and abs(b.low[2] - PAD_TOP) <= SEAT and _upright(b))


def _on(b, below) -> bool:
    """Standing on ``below``: centre over its top face, underside within SEAT of its top, upright."""
    dx, dy = b.pos[0] - below.pos[0], b.pos[1] - below.pos[1]
    return bool(abs(dx) <= CUBE / 2 and abs(dy) <= CUBE / 2 and abs(b.low[2] - below.high[2]) <= SEAT and _upright(b))


def tower(objects) -> list[str]:
    """The tallest stack of blocks not lost, standing on the pad, bottom first."""
    names = [n for n in NAMES if n not in _STATE["lost"]]
    best: list[str] = []
    for order in itertools.permutations(names):
        stack: list[str] = []
        for n in order:
            b = objects[n]
            if (not stack and _on_pad(b)) or (stack and _on(b, objects[stack[-1]])):
                stack.append(n)
            else:
                break
        if len(stack) > len(best):
            best = stack
    return best


def _aligned(objects, stack) -> bool:
    return all(np.hypot(*(objects[a].pos[:2] - objects[b].pos[:2])) <= ALIGN for b, a in zip(stack, stack[1:]))


def _clear(objects, stack) -> bool:
    tcps = _STATE["tcps"]
    if tcps is None:
        return False
    return all(np.linalg.norm(p - objects[n].pos) >= CLEAR_M for p in tcps for n in stack)


# -- scoring -------------------------------------------------------------------------------------------------------


def success(objects):
    """All three blocks (none lost) stand in a tower on the pad, each centre within 1.5 cm of the one below, still,
    with both grippers' grasp points 8 cm or more from every block's centre (let go)."""
    stack = tower(objects)
    return bool(len(stack) == 3 and _aligned(objects, stack) and all(objects[n].resting() for n in stack)
                and _clear(objects, stack))


def distance(objects):
    """Metres from the tower's top block to where the next one would sit (0 once all three are stacked)."""
    stack = tower(objects)
    if len(stack) == 3:
        return 0.0
    goal = np.array([*PAD_XY, PAD_TOP + (len(stack) + 0.5) * CUBE])
    left = [n for n in NAMES if n not in stack and n not in _STATE["lost"]]
    if not left:
        return float("inf")
    return float(min(np.linalg.norm(objects[n].pos - goal) for n in left))


def score(objects):
    """The tower at the end of the trial (success scores 1.0): 0.75 if all three blocks stand in it, 0.5 if two,
    0.25 if one block stands on the pad, otherwise 0. Lost blocks (passed the end line, or touched the table) never
    count."""
    return 0.25 * len(tower(objects))


# -- the scripted demo ---------------------------------------------------------------------------------------------
# The left arm takes each block off its moving carrier while moving along with it (the waypoints follow the
# carrier, one per control step), with the fingers closing across the line (along x), so a lag along the line only
# shifts the grip along the block's face. It sets each block on the tower and is back over the line before the next.
DT = 1 / 20
GRIP_YAW = math.pi / 2  # the fingers close along x, across the line
MEET_Y = 0.22  # where the gripper meets each block over the line
GRASP_Z = CARRIER_TOP + CUBE / 2 + 0.004  # TCP 4 mm above the block's centre: pad tips 1.6 cm over the carrier
PRE_Z = GRASP_Z + 0.06  # over the line: the open pad tips 2.6 cm above the blocks' tops
OFF_Z = GRASP_Z + 0.04  # lifted: the block's underside 4 cm above its carrier
DESCEND_S, CLOSE_S, LIFT_S = 0.6, 0.5, 0.5
RISE_S, ACROSS_S, LOWER_S, SETTLE_S, RELEASE_S, UP_S = 0.5, 1.0, 0.8, 0.3, 0.5, 0.5
DROP_M = 0.003  # each block is let go this far above its seat


def place_z(level: int) -> float:
    """TCP height to let go of a block at ``level`` (0 on the pad)."""
    return PAD_TOP + level * CUBE + CUBE / 2 + 0.004 + DROP_M


def carry_z(level: int) -> float:
    """TCP height to carry a block across to the tower: its underside 3 cm above where it will sit."""
    return place_z(level) + 0.03


def _track(name: str, t0: float, seconds: float, z0: float, z1: float, f0: float, f1: float, label: str):
    """Waypoints one control step apart that follow block ``name``'s carrier along the line from ``t0``, while the TCP
    goes from ``z0`` to ``z1`` and the fingers from ``f0`` to ``f1`` (both smoothly)."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        y = carrier_y(name, t0 + k * DT)
        out.append(Waypoint(DT, (LINE_X, y, z0 + s * (z1 - z0)), f0 + s * (f1 - f0), label, GRIP_YAW))
    return out


def demo():
    """Left arm, for each block in turn: be over the line where the block will be at its meeting time (2.75 s, then
    every 7 s), follow it down, close, lift it 4 cm off the carrier while still following it, rise, go across to the
    tower, lower it onto the pad or the block below, let go 3 mm above its seat and go back up. After the third
    block it goes up 6 cm and 10 cm away toward the line, and waits."""
    px, py = PAD_XY
    plan: list[Waypoint] = []
    t = 0.0
    for level, name in enumerate(NAMES):
        meet = arrival(name, MEET_Y)
        span = round((meet - t) / DT) * DT
        assert span >= 1.2, f"{name}: only {span:.2f} s to get over the line"
        plan.append(Waypoint(span, (LINE_X, MEET_Y, PRE_Z), FINGER_OPEN, f"over the line for {name}", GRIP_YAW))
        t += span
        plan += _track(name, t, DESCEND_S, PRE_Z, GRASP_Z, FINGER_OPEN, FINGER_OPEN, f"follow {name} down")
        t += DESCEND_S
        plan += _track(name, t, CLOSE_S, GRASP_Z, GRASP_Z, FINGER_OPEN, FINGER_CLOSED, f"close on {name}")
        t += CLOSE_S
        plan += _track(name, t, LIFT_S, GRASP_Z, OFF_Z, FINGER_CLOSED, FINGER_CLOSED, f"lift {name} off")
        t += LIFT_S
        y_off = carrier_y(name, t)
        top = max(carry_z(level), OFF_Z)
        place = place_z(level)
        plan += [
            Waypoint(RISE_S, (LINE_X, y_off, top), FINGER_CLOSED, f"rise with {name}", GRIP_YAW),
            Waypoint(ACROSS_S, (px, py, top), FINGER_CLOSED, "across to the tower", GRIP_YAW),
            Waypoint(LOWER_S, (px, py, place), FINGER_CLOSED, "lower", GRIP_YAW),
            Waypoint(SETTLE_S, (px, py, place), FINGER_CLOSED, "settle", GRIP_YAW),
            Waypoint(RELEASE_S, (px, py, place), FINGER_OPEN, "release", GRIP_YAW),
        ]
        t += RISE_S + ACROSS_S + LOWER_S + SETTLE_S + RELEASE_S
        if level < len(NAMES) - 1:
            plan.append(Waypoint(UP_S, (px, py, place + 0.08), FINGER_OPEN, "up off the tower", GRIP_YAW))
            t += UP_S
    top = place_z(len(NAMES) - 1) + 0.06  # straight up 6 cm (higher is out of reach over the pad), then away
    plan += [
        Waypoint(UP_S, (px, py, top), FINGER_OPEN, "up off the tower", GRIP_YAW),
        Waypoint(1.0, (px + 0.10, py, top), FINGER_OPEN, "away from the tower", GRIP_YAW),
        Waypoint(2.0, (px + 0.10, py, top), FINGER_OPEN, "wait", GRIP_YAW),
    ]
    return {"left": plan}
