"""On the station table, a conveyor line runs across the table from right to left (toward the left arm), 34 cm in front
of the arms: a black track bed (12 cm wide, 1 cm tall) with three grey carrier plates (10 x 10 x 1 cm) riding on it,
32 cm apart. Each carrier holds an open white plastic tub, square, 8 cm across and 5 cm tall. The line starts with
the trial and never stops for the robot: it reaches 3 cm/s within 0.5 s, so a new tub comes past the left arm every
10.7 s. A red end line is taped across the table on both sides of the track, 46 cm left of the midpoint between the
arms; past it the line carries the tubs out of reach and off the end of the table. Three white lids lie flat on the
table in a row in front of the left arm, between it and the line: each a square plate (10.8 cm) with a skirt
(1.5 cm deep) round its edge, which fits over a tub's rim with 1 cm to spare on every side, and a blue grip fin
(8 cm long front to back, 2.4 cm thick, 5.5 cm tall) standing along its middle.

The robot must put a lid on each tub while the tub moves, seated flat on its rim and let go, before the tub passes
the end line. Any lid fits any tub.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Put a lid on each of the three tubs as they go by on the conveyor. ..." ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as g  # noqa: E402
import numpy as np  # noqa: E402
from motionkit import DrivenBox, UsdBody, belt_run, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import TABLE_CENTER, TABLE_SIZE  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the score is the tubs lidded when each passed the end line, which the monitor below latches at that moment
# (cleared at every episode start); all three have passed it 33 s into the trial, well inside the run limit.
SCORE_OVER = "end"
TABLE_LEFT_Y = TABLE_CENTER[1] + TABLE_SIZE[1] / 2  # 0.65: the table's left edge (y is to the robot's left)


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the conveyor line ---------------------------------------------------------------------------------------------
LINE_X = 0.34  # the line's centre, metres in front of the arm bases
BED = (0.12, 1.24, 0.01)  # the fixed track bed: width (x), length (y), height; it ends 3 cm inside each table edge
CARRIER = (0.10, 0.10, 0.01)
CARRIER_Z = TOP + BED[2] + 0.002 + CARRIER[2] / 2  # 2 mm above the bed: carriers ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
WALL_Z = CARRIER_TOP + g.WALL_GAP + g.TUB_H / 2  # the tub walls' centres
RIM_Z = CARRIER_TOP + g.WALL_GAP + g.TUB_H  # the tubs' rim: TOP + 0.073
SPEED = 0.03  # m/s, toward +y (the left arm)
RAMP_S = 0.5  # the line speeds up from rest over this long at the start of the trial
SPACING = 0.32  # between carriers: one every SPACING / SPEED = 10.7 s
FIRST_Y = 0.075  # the first carrier's centre at the start
END_Y = 0.46  # the end line: a tub is judged when its centre reaches it
TURN_Y = TABLE_LEFT_Y + CARRIER[1] / 2 + 0.02  # 0.72: past the table's edge, carriers turn down out of the way
DROP, DROP_SPEED = 0.30, 0.15
BED_BOX = Box("track_bed", pos=(LINE_X, 0.0, TOP + BED[2] / 2), size=BED, color=(0.08, 0.08, 0.09))

TUBS = ("tub_1", "tub_2", "tub_3")  # in arrival order
STARTS = {n: FIRST_Y - i * SPACING for i, n in enumerate(TUBS)}  # each carrier's starting y
PATHS = {n: belt_run((LINE_X, y, CARRIER_Z), SPEED, RAMP_S, TURN_Y, DROP, DROP_SPEED) for n, y in STARTS.items()}


def _offset(path, d):
    """``path`` shifted by ``d``: a tub wall moves exactly with its carrier."""
    d = np.asarray(d, float)
    return lambda t: (path(t)[0] + d, path(t)[1])


CARRIERS = [
    DrivenBox(f"carrier_{n}", pos=_floats((LINE_X, STARTS[n], CARRIER_Z)), size=CARRIER, color=(0.5, 0.52, 0.55),
              path=PATHS[n])
    for n in TUBS
]
WALLS = [
    DrivenBox(f"{n}_{side}", pos=_floats((LINE_X + dx, STARTS[n] + dy, WALL_Z)), size=size, color=(0.95, 0.95, 0.95),
              path=_offset(PATHS[n], (dx, dy, WALL_Z - CARRIER_Z)))
    for n in TUBS for side, (dx, dy), size in g.tub_walls()
]


def carrier_y(name: str, t: float) -> float:
    """Where tub ``name``'s carrier is along the line (its centre's y) at episode time ``t``."""
    return float(PATHS[name](t)[0][1])


def arrival(name: str, y: float) -> float:
    """When tub ``name``'s carrier reaches ``y`` (after the ramp; the line runs at constant speed then)."""
    return float((y - STARTS[name]) / SPEED + RAMP_S / 2)


# -- the end line (red tape on the table, either side of the track) --------------------------------------------------
TAPE = (0.06, 0.012, 0.002)
TAPES = [
    Box(f"end_tape_{side}", pos=(LINE_X + sign * (BED[0] / 2 + 0.005 + TAPE[0] / 2), END_Y, TOP + TAPE[2] / 2),
        size=TAPE, color=(0.9, 0.05, 0.05))
    for side, sign in (("near", -1), ("far", 1))
]

# -- the lids ------------------------------------------------------------------------------------------------------
LIDS = ("lid_a", "lid_b", "lid_c")
LID_X = 0.16  # the row of lids, between the left arm and the line
LID_YS = {"lid_a": 0.14, "lid_b": 0.27, "lid_c": 0.40}
LID_REST_Z = TOP - g.SKIRT_BOTTOM  # a lid's origin (its plate's centre), lying on the table on its skirt
LID_OBJS = [
    UsdBody(n, pos=_floats((LID_X, y, LID_REST_Z)), size=(g.LID, g.LID, g.TAB_TOP - g.SKIRT_BOTTOM), mass=g.LID_MASS,
            usd="assets/lid.usda", friction=(1.2, 1.0), torsional_patch=0.015)  # the station's grippy friction
    for n, y in LID_YS.items()
]

OBJECTS = [BED_BOX, *TAPES, *CARRIERS, *WALLS, *LID_OBJS]

# -- the lidding monitor -------------------------------------------------------------------------------------------
CENTRE_M = 0.015  # a seated lid's centre within this (horizontally) of the tub's axis
SEAT_M = 0.005  # its plate's underside within this of the rim
TILT_DEG = 5.0  # its plate within this of level, right way up
LET_GO_M = 0.05  # both grippers' grasp points at least this far from the grip fin's centre
RIDE_MPS = 0.02  # the lid moving with the tub: velocity within this of the tub's

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(judged={}, tcps=None, step=0, printed=set())


reset_state()


def _up(lid) -> np.ndarray:
    """The lid's up axis (world frame)."""
    x, y, z, w = (float(v) for v in lid.quat_xyzw)
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def seated(lid, tub) -> bool:
    """``lid`` sits on ``tub`` (its carrier's state): centre within 1.5 cm of the tub's axis, plate's underside within
    5 mm of the rim, level within 5 degrees and right way up, moving with the tub."""
    up = _up(lid)
    if up[2] < math.cos(math.radians(TILT_DEG)):
        return False
    rim = tub.pos[2] + (RIM_Z - CARRIER_Z)
    under = lid.pos + up * g.PLATE_UNDER  # the plate's underside, under its centre
    return bool(np.hypot(*(lid.pos[:2] - tub.pos[:2])) <= CENTRE_M and abs(under[2] - rim) <= SEAT_M
                and np.linalg.norm(lid.lin_vel - tub.lin_vel) <= RIDE_MPS)


def let_go(lid, tcps) -> bool:
    """Neither gripper holds the lid: both grasp points 5 cm or more from its grip fin's centre."""
    if tcps is None:
        return False
    tab = lid.pos + _up(lid) * g.TAB_CENTRE_Z
    return all(np.linalg.norm(p - tab) >= LET_GO_M for p in tcps)


def lidded(objects, tub: str, tcps) -> str | None:
    """The lid seated on ``tub`` and let go, if any."""
    for lid in LIDS:
        if seated(objects[lid], objects[f"carrier_{tub}"]) and let_go(objects[lid], tcps):
            return lid
    return None


def update(objects: dict, tcps) -> None:
    """One control step of the monitor: judge each tub, once, at the step its centre reaches the end line."""
    tcps = None if tcps is None else [np.asarray(p, float) for p in tcps]
    _STATE["tcps"] = tcps
    for tub in TUBS:
        if tub not in _STATE["judged"] and objects[f"carrier_{tub}"].pos[1] >= END_Y:
            _STATE["judged"][tub] = lidded(objects, tub, tcps)


DEBUG = bool(os.environ.get("LID_DEBUG"))  # print lid poses and grasps (for checking a sim run)


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.scene import object_states

    tcps = [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0] for side in ("left", "right")]
    objects = object_states(environment, [*LIDS, *(f"carrier_{t}" for t in TUBS)])
    update(objects, tcps)
    if DEBUG:
        gaps = [float(np.sum(np.abs(arm_joints(environment, side, FINGER_JOINTS)))) for side in ("left", "right")]
        _debug(objects, tcps, gaps)


def _debug(objects, tcps, gaps) -> None:
    """Every 0.5 s each lid's pose; every step a gripper is within 6 cm of a lid's fin, its TCP relative to the fin's
    centre and its pad gap; and each tub's judgment as it is made."""
    step = _STATE["step"] = _STATE.get("step", 0) + 1
    t = step * DT
    if step % 10 == 0:
        for n in LIDS:
            o = objects[n]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, _up(o)[2]))))
            print(f"lid_debug t={t:.2f} {n} pos={o.pos.round(4).tolist()} tilt={tilt:.1f} "
                  f"vel={o.lin_vel.round(3).tolist()}")
    for side, tcp, gap in zip(("left", "right"), tcps, gaps, strict=True):
        for n in LIDS:
            fin = objects[n].pos + _up(objects[n]) * g.TAB_CENTRE_Z
            r = tcp - fin
            if np.linalg.norm(r) < 0.06:
                print(f"lid_debug t={t:.2f} {side} near {n}: tcp-fin={r.round(4).tolist()} pad_gap={gap:.4f} "
                      f"tcp={tcp.round(4).tolist()}")
    for tub, lid in _STATE["judged"].items():
        if tub not in _STATE.setdefault("printed", set()):
            _STATE["printed"].add(tub)
            print(f"lid_debug t={t:.2f} {tub} judged at the end line: {lid or 'not lidded'}")


def judged() -> dict[str, str | None]:
    """The tubs past the end line so far, with the lid seated on each then (None: not lidded)."""
    return dict(_STATE["judged"])


# -- scoring -------------------------------------------------------------------------------------------------------


def success(objects):
    """All three tubs reached the end line lidded."""
    return bool(len(_STATE["judged"]) == 3 and all(_STATE["judged"].values()))


def distance(objects):
    """Metres from the nearest free lid to a seat on the next tub not yet judged (0 once all are judged)."""
    left = [t for t in TUBS if t not in _STATE["judged"]]
    if not left:
        return 0.0
    tub = objects[f"carrier_{left[0]}"]
    seat = tub.pos + np.array([0.0, 0.0, RIM_Z - CARRIER_Z - g.PLATE_UNDER])
    used = {lid for lid in _STATE["judged"].values() if lid}
    return float(min(np.linalg.norm(objects[n].pos - seat) for n in LIDS if n not in used))


def score(objects):
    """1/3 for each tub that reached the end line with a lid seated on it (centre within 1.5 cm of its axis, plate's
    underside within 5 mm of its rim, level within 5 degrees, moving with it) and let go. A tub that hasn't reached
    the end line yet counts 0."""
    return sum(1 for lid in _STATE["judged"].values() if lid) / 3


# -- the scripted demo ---------------------------------------------------------------------------------------------
# The left arm takes each lid by its grip fin (the fingers close across its thickness, along y), carries it over
# the line
# ahead of the next tub, and lowers it onto the tub while following it (one waypoint per control step), lets go on
# the rim and rises still following, then fetches the next lid. A lag behind the moving tub shifts the lid along the
# line; the skirt clears the walls by 1 cm on each side.
DT = 1 / 20
GRIP_YAW = 0.0  # the fingers close along y, along the line (turned 90 degrees, the wrist runs out of range here)
# TCP above the plate's top: the pad tips 1.6 cm over the plate. The pads' collision faces meet the fin from their tips
# up to 5 cm above them (the finger meshes, measured from the robot's USD), so all 2.9 cm of fin above the tips is held.
TAB_GRIP = PAD_TIP_PAST_TCP + 0.015
TCP_OVER_ORIGIN = g.PLATE_T / 2 + TAB_GRIP  # 0.0315: TCP above the lid's origin while held
GRASP_Z = LID_REST_Z + TCP_OVER_ORIGIN
PRE_Z = GRASP_Z + 0.06  # over a lid on the table: the open pad tips 3.1 cm above its grip fin
CARRY_Z = RIM_Z + 0.03 - g.SKIRT_BOTTOM + TCP_OVER_ORIGIN  # held lid's skirt 3 cm above the rims
DROP_M = 0.003  # a lid is let go this far above its seat
PLACE_Z = RIM_Z - g.PLATE_UNDER + DROP_M + TCP_OVER_ORIGIN
MEET_Y = 0.29  # where the held lid meets each tub over the line
# The gripper takes about 1 s to close on the fin (and to open off it) after the command: the arm holds still at
# the fin for GRIP_S after the close command and at the rim for OPEN_S after the open command. Lift and carry are
# slow (peak accelerations under 0.35 m/s^2), so the held lid doesn't swing in the pinch.
TO_LID_S, DESCEND_S, CLOSE_S, GRIP_S, LIFT_S, ACROSS_S = 1.1, 0.7, 0.5, 0.9, 1.4, 2.0
LOWER_S, SETTLE_S, RELEASE_S, OPEN_S, RISE_S = 1.0, 0.2, 0.5, 0.8, 0.6
ORDER = dict(zip(TUBS, LIDS[::-1], strict=True))  # the far lid first, so no held lid passes over a lidded tub


def _track(tub: str, t0: float, seconds: float, z0: float, z1: float, f0: float, f1: float, label: str):
    """Waypoints one control step apart over tub ``tub``'s axis from ``t0``, while the TCP goes from ``z0`` to ``z1``
    and the fingers from ``f0`` to ``f1`` (both smoothly)."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        y = carrier_y(tub, t0 + k * DT)
        out.append(Waypoint(DT, (LINE_X, y, z0 + s * (z1 - z0)), f0 + s * (f1 - f0), label, GRIP_YAW))
    return out


END_S = 37.0  # the plan ends here: the last tub is judged at 34.25 s and success holds from 35.25 s


def demo():
    """Left arm, for each tub in turn: over its lid, down, close on the grip fin and hold while the fingers close,
    slowly straight up to carrying height, across to the line ahead of the tub; then, following the tub from its
    meeting time (7.4 s, then every 10.7 s), lower the lid onto its rim, let go 3 mm above it, hold while the fingers
    open, and rise. After the third it moves back over the table and waits."""
    plan: list[Waypoint] = []
    t = 0.0
    for tub, lid in ORDER.items():
        ly = LID_YS[lid]
        plan += [
            Waypoint(TO_LID_S, (LID_X, ly, PRE_Z), FINGER_OPEN, f"over {lid}", GRIP_YAW),
            Waypoint(DESCEND_S, (LID_X, ly, GRASP_Z), FINGER_OPEN, f"down to {lid}", GRIP_YAW),
            Waypoint(CLOSE_S, (LID_X, ly, GRASP_Z), FINGER_CLOSED, f"close on {lid}", GRIP_YAW),
            Waypoint(GRIP_S, (LID_X, ly, GRASP_Z), FINGER_CLOSED, f"grip {lid}", GRIP_YAW),
            Waypoint(LIFT_S, (LID_X, ly, CARRY_Z), FINGER_CLOSED, f"lift {lid}", GRIP_YAW),
            Waypoint(ACROSS_S, (LINE_X, MEET_Y, CARRY_Z), FINGER_CLOSED, f"across with {lid}", GRIP_YAW),
        ]
        t += TO_LID_S + DESCEND_S + CLOSE_S + GRIP_S + LIFT_S + ACROSS_S
        meet = arrival(tub, MEET_Y)
        wait = math.ceil((meet - t) / DT - 1e-6) * DT  # whole control steps, starting down once the tub is there
        assert wait >= 0.0, f"{tub}: {-wait:.2f} s late"
        if wait > 0:
            plan.append(Waypoint(wait, (LINE_X, MEET_Y, CARRY_Z), FINGER_CLOSED, f"wait for {tub}", GRIP_YAW))
            t += wait
        plan += _track(tub, t, LOWER_S, CARRY_Z, PLACE_Z, FINGER_CLOSED, FINGER_CLOSED, f"lower onto {tub}")
        t += LOWER_S
        plan += _track(tub, t, SETTLE_S, PLACE_Z, PLACE_Z, FINGER_CLOSED, FINGER_CLOSED, f"settle on {tub}")
        t += SETTLE_S
        plan += _track(tub, t, RELEASE_S, PLACE_Z, PLACE_Z, FINGER_CLOSED, FINGER_OPEN, f"release on {tub}")
        t += RELEASE_S
        plan += _track(tub, t, OPEN_S, PLACE_Z, PLACE_Z, FINGER_OPEN, FINGER_OPEN, f"open on {tub}")
        t += OPEN_S
        plan += _track(tub, t, RISE_S, PLACE_Z, CARRY_Z + 0.055, FINGER_OPEN, FINGER_OPEN, f"rise off {tub}")
        t += RISE_S
    plan += [
        Waypoint(1.0, (LID_X + 0.04, 0.30, CARRY_Z), FINGER_OPEN, "back over the table", GRIP_YAW),
    ]
    # yam_scripted ends the rollout 3 s after the plan does: wait until the last tub has been judged and success held
    plan.append(Waypoint(END_S - sum(w.duration for w in plan), (LID_X + 0.04, 0.30, CARRY_Z), FINGER_OPEN, "wait",
                         GRIP_YAW))
    return {"left": plan}
