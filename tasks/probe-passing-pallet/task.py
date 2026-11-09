"""On the station table, a conveyor line runs across the table from right to left (toward the left arm), 32 cm in front
of the arms: a black track bed (14 cm wide, 1.5 cm tall) with two grey aluminium pallets (12 x 12 x 3 cm) riding on
it, 30 cm apart. Each pallet carries a gold test pad (3 x 3 cm, 4 mm thick) in its middle. The line starts with the
trial and never stops for the robot: it reaches 4 cm/s within 0.5 s, so a pallet comes past the left arm every
7.5 s. A red end line is taped across the table on both sides of the track, 40 cm left of the midpoint. A test probe
(a black square handle, 2.6 x 2.6 x 10 cm, with a thin brass tip, 0.8 x 0.8 x 4 cm, under it) stands tip down in a
grey stand (four fixed wooden blocks around a 1.6 cm square well, 4.5 cm tall) between the line and the left arm.

The robot must test each pallet as it passes: hold the probe's tip on the pallet's gold pad for 2 s without a break
before the pallet reaches the end line. Since a pallet moves 8 cm in 2 s and the pad is only 3 cm across, the probe
has to travel with the pallet; held still, it stays on a pad for at most 0.75 s.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Test both pallets as they pass. ..." ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, UsdBody, belt_run, clock, offset_path, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import TABLE_CENTER, TABLE_SIZE  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# The monitor below keeps each pallet's longest unbroken contact over the whole trial (cleared at every episode
# start), so the score is the same at the end as at its best.
SCORE_OVER = "end"
TABLE_LEFT_Y = TABLE_CENTER[1] + TABLE_SIZE[1] / 2  # 0.65: the table's left edge (y is to the robot's left)


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the conveyor line ---------------------------------------------------------------------------------------------
LINE_X = 0.32  # the line's centre, metres in front of the arm bases
BED = (0.14, 1.24, 0.015)  # the fixed track bed: width (x), length (y), height; it ends 3 cm inside each table edge
PALLET = (0.12, 0.12, 0.03)
PALLET_Z = TOP + BED[2] + 0.002 + PALLET[2] / 2  # 2 mm above the bed: pallets ride, they don't rub
PALLET_TOP = PALLET_Z + PALLET[2] / 2
PAD = (0.03, 0.03, 0.004)  # the gold test pad, in the middle of each pallet's top
PAD_GAP = 0.001  # the pad floats 1 mm over its pallet so the two driven boxes never touch (glued on a real pallet)
PAD_Z = PALLET_TOP + PAD_GAP + PAD[2] / 2
PAD_TOP = PAD_Z + PAD[2] / 2
SPEED = 0.04  # m/s, toward +y (the left arm)
RAMP_S = 0.5  # the line speeds up from rest over this long at the start of the trial
SPACING = 0.30  # between pallets: one every SPACING / SPEED = 7.5 s
FIRST_Y = -0.20  # the first pallet's centre at the start
END_Y = 0.40  # the end line: contact counts only while the pallet's centre is short of it
TURN_Y = TABLE_LEFT_Y + PALLET[1] / 2 + 0.02  # 0.73: past the table's edge, pallets turn down out of the way
DROP, DROP_SPEED = 0.30, 0.15
BED_BOX = Box("track_bed", pos=(LINE_X, 0.0, TOP + BED[2] / 2), size=BED, color=(0.08, 0.08, 0.09))

NAMES = ("first", "second")  # the pallets, in arrival order
STARTS = {n: FIRST_Y - i * SPACING for i, n in enumerate(NAMES)}  # each pallet's starting y
PATHS = {n: belt_run((LINE_X, y, PALLET_Z), SPEED, RAMP_S, TURN_Y, DROP, DROP_SPEED) for n, y in STARTS.items()}
PAD_PATHS = {n: offset_path(PATHS[n], (0.0, 0.0, PAD_Z - PALLET_Z)) for n in NAMES}
PALLETS = [DrivenBox(f"pallet_{n}", pos=_floats(PATHS[n](0.0)[0]), size=PALLET, color=(0.55, 0.57, 0.6), path=PATHS[n])
           for n in NAMES]
PADS = [DrivenBox(f"pad_{n}", pos=_floats(PAD_PATHS[n](0.0)[0]), size=PAD, color=(0.85, 0.68, 0.2),
                  path=PAD_PATHS[n])
        for n in NAMES]


def pallet_y(name: str, t: float) -> float:
    """Where pallet ``name`` is along the line (its centre's y) at episode time ``t``."""
    return float(PATHS[name](t)[0][1])


def arrival(name: str, y: float) -> float:
    """When pallet ``name`` reaches ``y`` (after the ramp; the line runs at constant speed then)."""
    return float((y - STARTS[name]) / SPEED + RAMP_S / 2)


# -- the end line (red tape on the table, either side of the track) --------------------------------------------------
TAPE = (0.06, 0.012, 0.002)
TAPES = [
    Box(f"end_tape_{side}", pos=(LINE_X + sign * (BED[0] / 2 + 0.005 + TAPE[0] / 2), END_Y, TOP + TAPE[2] / 2),
        size=TAPE, color=(0.9, 0.05, 0.05))
    for side, sign in (("near", -1), ("far", 1))
]

# -- the probe and its stand ---------------------------------------------------------------------------------------
HANDLE = (0.026, 0.026, 0.10)  # black plastic handle; the probe's origin is its centre
TIP = (0.008, 0.008, 0.04)  # brass tip under the handle, centred
PROBE_MASS = 0.06
TIP_END = -(HANDLE[2] / 2 + TIP[2])  # the tip's end, below the probe's origin (along its axis)
GRIP_UP = HANDLE[2] / 2 - 0.03  # where it is held: 3 cm below the handle's top, above its origin
STAND_XY = (0.18, 0.22)  # between the line and the left arm (base at y 0.305)
WELL, WALL, STAND_H = 0.016, 0.01, 0.045  # the stand's square well, its walls' thickness and height
PROBE_Z0 = TOP + STAND_H + HANDLE[2] / 2 + 0.0005  # the handle rests on the wall tops, the tip in the well
PROBE = UsdBody("probe", usd="assets/probe.usda", pos=_floats((*STAND_XY, PROBE_Z0)),
                size=(HANDLE[0], HANDLE[1], HANDLE[2] + TIP[2]), mass=PROBE_MASS, friction=(0.8, 0.7),
                torsional_patch=0.01)


def _stand():
    sx, sy = STAND_XY
    out = WELL / 2 + WALL / 2
    span = WELL + 2 * WALL
    z = TOP + STAND_H / 2
    walls = [("near", (-out, 0.0), (WALL, span)), ("far", (out, 0.0), (WALL, span)),
             ("right", (0.0, -out), (WELL, WALL)), ("left", (0.0, out), (WELL, WALL))]
    return [Box(f"stand_{side}", pos=_floats((sx + dx, sy + dy, z)), size=(*size, STAND_H), color=(0.55, 0.5, 0.45))
            for side, (dx, dy), size in walls]


STAND = _stand()

OBJECTS = [BED_BOX, *TAPES, *STAND, *PALLETS, *PADS, PROBE]

# -- the test monitor ----------------------------------------------------------------------------------------------
ABOVE_M = 0.004  # "on the pad": the tip's end at most this far above the pad's top (a spring test pin's travel) ...
INTO_M = 0.005  # ... and at most this far below it (pressed in: the sim lets contacts sink a little)
TILT_DEG = 20.0  # the probe stands tip down: its axis within this of vertical
HELD_M = 0.03  # held: a gripper's grasp point within this of the handle's grip point
TEST_S = 2.0  # full credit per pallet: this long on its pad without a break
LEVELS = ((2.0, 1.0), (1.5, 0.75), (1.0, 0.5))  # unbroken seconds on the pad -> credit; any contact earns 0.25

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.update(best={n: None for n in NAMES}, since={n: None for n in NAMES})


reset_state()


def _rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tip_end(probe) -> np.ndarray:
    """The probe tip's end (the middle of its bottom face)."""
    return np.asarray(probe.pos, float) + _rotation(probe.quat_xyzw) @ np.array([0.0, 0.0, TIP_END])


def held(probe, tcps) -> bool:
    """A gripper's grasp point is at the handle's grip point (within HELD_M)."""
    if tcps is None:
        return False
    grip = np.asarray(probe.pos, float) + _rotation(probe.quat_xyzw) @ np.array([0.0, 0.0, GRIP_UP])
    return any(np.linalg.norm(np.asarray(p, float) - grip) <= HELD_M for p in tcps)


def on_pad(probe, pad, tcps) -> bool:
    """The held, upright probe's tip end is inside the pad's outline and within ABOVE_M above / INTO_M below its top,
    with the pad's centre short of the end line."""
    if pad.pos[1] >= END_Y:
        return False
    if _rotation(probe.quat_xyzw)[2, 2] < math.cos(math.radians(TILT_DEG)):
        return False
    tip = tip_end(probe)
    dx, dy = tip[0] - pad.pos[0], tip[1] - pad.pos[1]
    dz = tip[2] - pad.high[2]
    return bool(abs(dx) <= PAD[0] / 2 and abs(dy) <= PAD[1] / 2 and -INTO_M <= dz <= ABOVE_M and held(probe, tcps))


def update(t: float, objects: dict, tcps) -> None:
    """One control step of the monitor at episode time ``t``: for each pallet, the longest unbroken time so far with
    the probe on its pad (None if never)."""
    probe = objects["probe"]
    for n in NAMES:
        if on_pad(probe, objects[f"pad_{n}"], tcps):
            if _STATE["since"][n] is None:
                _STATE["since"][n] = t
            run = t - _STATE["since"][n]
            _STATE["best"][n] = max(run, _STATE["best"][n] or 0.0)
        else:
            _STATE["since"][n] = None


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.scene import object_states

    tcps = [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0] for side in ("left", "right")]
    update(clock(), object_states(environment, ["probe", *(f"pad_{n}" for n in NAMES)]), tcps)


def tested() -> dict:
    """Each pallet's longest unbroken time on its pad so far (s), or None."""
    return dict(_STATE["best"])


def credit(best: float | None) -> float:
    """One pallet's credit from its longest unbroken time on its pad."""
    if best is None:
        return 0.0
    for need, value in LEVELS:
        if best >= need - 1e-9:
            return value
    return 0.25


# -- scoring -------------------------------------------------------------------------------------------------------


def success(objects):
    """Both pallets had the held probe's tip on their pad for 2 s without a break, before the end line."""
    return all(credit(b) == 1.0 for b in _STATE["best"].values())


def score(objects):
    """The mean of the two pallets' credits: 1.0 for 2 s unbroken on the pad, 0.75 for 1.5 s, 0.5 for 1 s, 0.25 for
    any contact, 0 for none."""
    return float(np.mean([credit(_STATE["best"][n]) for n in NAMES]))


# -- the scripted demo ---------------------------------------------------------------------------------------------
# The left arm takes the probe by its handle (fingers closing along x), carries it over the line, and for each
# pallet comes down onto the pad while following the pallet (one waypoint per control step), stays on it 2.8 s,
# and rises off it, still following, well before the end line.
DT = 1 / 20
GRIP_YAW = -math.pi / 2  # the fingers close along x, across the line (this way round the wrist reaches inward)
TCP_ABOVE_TIP = GRIP_UP - TIP_END  # 0.11: the grasp point above the tip's end, while held upright
GRASP_Z = PROBE_Z0 + GRIP_UP  # grasp point at the handle's grip point, probe in its stand
PRE_Z = GRASP_Z + 0.06  # above the stand, open
HOVER_Z = PAD_TOP + 0.03 + TCP_ABOVE_TIP  # carrying: the tip's end 3 cm above the pads
PRESS_M = 0.001  # on the pad, the tip's end is aimed 1 mm into it (light contact)
TOUCH_Z = PAD_TOP - PRESS_M + TCP_ABOVE_TIP
MEET_Y = 0.10  # where the probe meets each pallet's pad
DESCEND_S, DWELL_S, RISE_S = 0.6, 2.8, 0.5


def _track(name: str, t0: float, seconds: float, z0: float, z1: float, label: str):
    """Waypoints one control step apart over pallet ``name``'s pad from ``t0``, while the grasp point goes smoothly
    from ``z0`` to ``z1``."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        y = pallet_y(name, t0 + k * DT)
        out.append(Waypoint(DT, (LINE_X, y, z0 + s * (z1 - z0)), FINGER_CLOSED, label, GRIP_YAW))
    return out


def demo():
    """Left arm: take the probe from its stand, then for each pallet be over the line where its pad will be at the
    meeting time (7.75 s, 15.25 s), follow it down onto the pad, stay on it 2.8 s, rise off still following, and
    finally lift the probe clear of the line and wait."""
    sx, sy = STAND_XY
    plan = [
        Waypoint(2.0, (sx, sy, PRE_Z), FINGER_OPEN, "above the stand", GRIP_YAW),
        Waypoint(1.0, (sx, sy, GRASP_Z), FINGER_OPEN, "down to the handle", GRIP_YAW),
        Waypoint(0.5, (sx, sy, GRASP_Z), FINGER_CLOSED, "close on the handle", GRIP_YAW),
        Waypoint(1.0, (sx, sy, HOVER_Z), FINGER_CLOSED, "lift the probe out", GRIP_YAW),
    ]
    t = sum(w.duration for w in plan)
    for name in NAMES:
        meet = arrival(name, MEET_Y)
        span = round((meet - t) / DT) * DT
        assert span >= 1.2, f"{name}: only {span:.2f} s to get over the line"
        plan.append(Waypoint(span, (LINE_X, MEET_Y, HOVER_Z), FINGER_CLOSED, f"over the line for {name}", GRIP_YAW))
        t += span
        plan += _track(name, t, DESCEND_S, HOVER_Z, TOUCH_Z, f"down onto {name}")
        t += DESCEND_S
        plan += _track(name, t, DWELL_S, TOUCH_Z, TOUCH_Z, f"test {name}")
        t += DWELL_S
        plan += _track(name, t, RISE_S, TOUCH_Z, HOVER_Z, f"up off {name}")
        t += RISE_S
    y_end = pallet_y(NAMES[-1], t)
    plan += [
        Waypoint(1.0, (LINE_X - 0.10, y_end, HOVER_Z + 0.03), FINGER_CLOSED, "away from the line", GRIP_YAW),
        Waypoint(2.0, (LINE_X - 0.10, y_end, HOVER_Z + 0.03), FINGER_CLOSED, "wait", GRIP_YAW),
    ]
    return {"left": plan}
