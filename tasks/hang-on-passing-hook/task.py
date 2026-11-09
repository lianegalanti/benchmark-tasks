"""On the station table, two orange plastic hanger parts stand upright, one in front of each arm, 15 cm ahead of the
arm bases: each is a block (8.4 x 3 x 3 cm) with a rectangular eye on top (opening 6 cm wide, 6.5 cm tall, facing
the robot) and a small grip tab (2 x 1.2 x 2.5 cm) on top of the eye. Above the far side of the table runs an overhead
conveyor rail (40 cm above the table), carrying three steel hooks 30 cm apart. Each hook hangs from the rail on a
stem, 47.7 cm in front of the arm bases, and ends in a bar (1.2 cm square, 23 cm long, 12 cm above the table)
pointing back toward the robot with an upturned tip (1.5 cm high) at its end. The hooks enter from the robot's left
and travel toward its right at a steady 5 cm/s; they never stop. A red tape mark on the table, 60 cm right of the
midpoint between the arms, marks where hooks leave the station.

The robot must hang each part on its own hook by its eye (the hook's bar through the eye) before that hook passes
the red exit mark, and let go of it.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, UsdPart, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
# "end": the score keeps its own record over the trial (below): which part was on which hook as that hook left.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Hang the two orange parts on the passing conveyor hooks by the loop on top of each part, one part "
               "per hook, before the hooks reach the red mark, and let go of them. The hooks keep moving and won't "
               "stop for you.")

STEEL = (0.45, 0.47, 0.50)
RAIL_GREY = (0.30, 0.31, 0.33)
RED = (0.85, 0.08, 0.08)


def _f(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects."""
    return tuple(float(x) for x in v)


def conveyor(start):
    """A path at the conveyor's constant speed toward -y from ``start`` (the hooks never stop)."""
    a = np.asarray(start, float)
    vel = np.array([0.0, -g.SPEED, 0.0])

    def path(t: float):
        return a + vel * t, vel.copy()

    return path


def _hook(i: int) -> list[DrivenBox]:
    return [DrivenBox(f"hook{i}_{part}", pos=_f(c), size=_f(s), color=STEEL, path=conveyor(c))
            for part, (c, s) in g.hook_boxes(i, 0.0).items()]


HOOKS = [box for i in range(g.N_HOOKS) for box in _hook(i)]
RAIL = Box("rail", pos=_f(g.RAIL_CENTRE), size=_f(g.RAIL), color=RAIL_GREY)
EXIT_MARK = Box("exit_mark", pos=_f((0.36, g.EXIT_Y, g.TOP + 0.001)), size=(0.30, 0.02, 0.002), color=RED)
PARTS = [UsdPart(name, usd="assets/hanger_part.usda", pos=_f(g.part_start(name)), size=(0.03, g.OUT_W, g.HEIGHT),
                 mass=g.MASS, color=(0.95, 0.45, 0.10), torsional_patch=0.01) for name in g.PARTS]
OBJECTS = [RAIL, EXIT_MARK, *HOOKS, *PARTS]

# -- monitoring and scoring ------------------------------------------------------------------------------------------
LIFTED = 0.03  # m: a part's lowest point this far above the table counts as lifted
OFF_TABLE = 0.01  # m: a hung part's lowest point at least this far above the table
LET_GO = 0.05  # m: every gripper's grasp point (between its finger pads) at least this far from the part's tab
HUNG, ON_HOOK, LIFT_CREDIT = 0.5, 0.25, 0.1  # per part

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(best={p: 0.0 for p in g.PARTS}, credited={p: None for p in g.PARTS}, judged=set())


reset_state()


def update(parts: dict, hook_ys, grasps) -> None:
    """Take one control step: each part's (origin, rotation matrix), each hook's bar position along the rail (y),
    and each gripper's grasp point (station frame).

    A part reaches ON_HOOK once a hook's bar passes through its eye, and LIFT_CREDIT once its lowest point is 3 cm
    above the table. When a hook's bar first reaches the exit mark, one not-yet-credited part hanging on it is
    credited HUNG: the bar through its eye, its lowest point 1 cm or more above the table, and every grasp point at
    least 5 cm from its tab (let go). A hook credits at most one part; a credited part stays credited."""
    for name, (pos, R) in parts.items():
        if any(g.bar_through_eye(pos, R, y) for y in hook_ys):
            _STATE["best"][name] = max(_STATE["best"][name], ON_HOOK)
        if g.lowest_z(pos, R) >= g.TOP + LIFTED:
            _STATE["best"][name] = max(_STATE["best"][name], LIFT_CREDIT)
    for i, y in enumerate(hook_ys):
        if i in _STATE["judged"] or y > g.EXIT_Y:
            continue
        _STATE["judged"].add(i)
        for name, (pos, R) in parts.items():
            if _STATE["credited"][name] is not None:
                continue
            tab = g.tab_centre(pos, R)
            if (g.bar_through_eye(pos, R, y) and g.lowest_z(pos, R) >= g.TOP + OFF_TABLE
                    and all(np.linalg.norm(np.asarray(p) - tab) >= LET_GO for p in grasps)):
                _STATE["credited"][name] = i
                break


def _grasps(environment):
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    return [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0] for side in ("left", "right")]


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    names = [*g.PARTS, *(f"hook{i}_bar" for i in range(g.N_HOOKS))]
    s = object_states(environment, names)
    parts = {p: (s[p].pos, g.rot(s[p].quat_xyzw)) for p in g.PARTS}
    update(parts, [float(s[f"hook{i}_bar"].pos[1]) for i in range(g.N_HOOKS)], _grasps(environment))


def success(objects) -> bool:
    """Both parts credited: each was hanging on its own hook, let go, when that hook passed the exit mark."""
    return all(v is not None for v in _STATE["credited"].values())


def score(objects) -> float:
    """Per part, summed: 0.5 if it was hanging on its own hook (bar through the eye, lowest point 1 cm or more above
    the table, no gripper within 5 cm of its tab) when that hook passed the exit mark; otherwise 0.25 if a hook's bar
    was at some point through its eye; otherwise 0.1 if it was at some point lifted 3 cm off the table; else 0."""
    return float(sum(HUNG if _STATE["credited"][p] is not None else _STATE["best"][p] for p in g.PARTS))


# -- the scripted demo ---------------------------------------------------------------------------------------------
GRIP_DEPTH = 0.010  # the TCP this far below the tab's top: the pad tips stop 2 mm above the eye's top bar
TCP_ABOVE_ORIGIN = g.TAB_TOP - GRIP_DEPTH
TRAVEL_Z = g.TOP + 0.21  # TCP height while carrying
APPROACH_X = 0.195  # the part's origin in front of the hook's tip, before threading
RELEASE_OPEN = -0.025  # finger target when letting go: pads 5 cm apart, clear of the 1.2 cm tab
DT = 0.05  # one control step at 20 Hz: while following, a waypoint per step
PARK = {"left": (0.16, 0.33, g.TOP + 0.20), "right": (0.16, -0.33, g.TOP + 0.20)}  # behind the hooks, above the parts
# who hangs what: (part, hook, when the follow starts). The meeting points are 2 cm toward the arm from each hook's
# entry into the arm's comfortable reach.
JOBS = {"left": ("part_a", 0, (g.Y0 - 0.42) / g.SPEED), "right": ("part_b", 1, (g.Y0 + g.SPACING + 0.08) / g.SPEED)}
# the follow, in seconds after it starts: (end, phase)
PHASES = ((0.5, "match"), (1.5, "thread"), (1.7, "settle"), (2.5, "lower"), (2.7, "hold"), (3.1, "release"),
          (3.4, "ride open"), (4.0, "rise"))
FOLLOW_S = PHASES[-1][0]


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def follow_target(hook: int, t0: float, t: float):
    """(part origin x, z, finger, phase) at time ``t`` of a follow that started at ``t0``; y is the hook's."""
    s = t - t0
    begin = 0.0
    for end, phase in PHASES:
        if s <= end + 1e-9:
            break
        begin = end
    u = _smooth((s - begin) / (end - begin))
    x = APPROACH_X if phase == "match" else APPROACH_X + u * (g.HUNG_X - APPROACH_X) if phase == "thread" \
        else g.HUNG_X
    lowered = g.HUNG_Z + 0.003  # let go 3 mm above resting on the bar
    z = g.THREAD_Z if phase in ("match", "thread", "settle") else g.THREAD_Z + u * (lowered - g.THREAD_Z) \
        if phase == "lower" else lowered + u * 0.06 if phase == "rise" else lowered
    finger = FINGER_CLOSED if phase not in ("release", "ride open", "rise") else \
        FINGER_CLOSED + u * (RELEASE_OPEN - FINGER_CLOSED) if phase == "release" else RELEASE_OPEN
    return x, z, finger, phase


def arm_plan(side: str) -> list[Waypoint]:
    """Pick the part by its tab, wait above its start, carry it in front of the hook's tip as the follow starts, then
    a waypoint every control step: match the hook, thread the eye over the tip and onto the bar, lower it onto the
    bar, open the fingers, ride along open, rise while following, and park."""
    part, hook, t0 = JOBS[side]
    px, py = g.PART_XY[part]
    grasp_z = g.part_start(part)[2] + TCP_ABOVE_ORIGIN
    plan = [
        Waypoint(1.5, (px, py, TRAVEL_Z), FINGER_OPEN, "above the part"),
        Waypoint(1.2, (px, py, grasp_z), FINGER_OPEN, "descend"),
        Waypoint(0.6, (px, py, grasp_z), FINGER_CLOSED, "close"),
        Waypoint(1.0, (px, py, TRAVEL_Z), FINGER_CLOSED, "pick up"),
    ]
    carry_s = 1.3
    wait = t0 - carry_s - sum(w.duration for w in plan)
    if wait > 0:
        plan.append(Waypoint(wait, (px, py, TRAVEL_Z), FINGER_CLOSED, "wait"))
    plan.append(Waypoint(carry_s, (APPROACH_X, g.hook_y(hook, t0), g.THREAD_Z + TCP_ABOVE_ORIGIN), FINGER_CLOSED,
                         "carry"))
    n = round(FOLLOW_S / DT)
    for k in range(1, n + 1):
        t = t0 + k * DT
        x, z, finger, phase = follow_target(hook, t0, t)
        plan.append(Waypoint(DT, (x, g.hook_y(hook, t), z + TCP_ABOVE_ORIGIN), finger, phase))
    plan.append(Waypoint(1.2, PARK[side], RELEASE_OPEN, "park"))
    # yam_scripted ends the rollout 3 s after the plan does: stay parked until hook 1 has passed and success held
    plan.append(Waypoint(END_S - sum(w.duration for w in plan), PARK[side], RELEASE_OPEN, "wait parked"))
    return plan


END_S = 35.0  # the plans end here: hook 1 passes the exit mark at 32 s and success holds from 33 s


def demo():
    """Left arm: part A onto hook 0 as it passes in front of the left arm (follow from 5.6 s, let go by 8.7 s).
    Right arm: part B onto hook 1 as it passes in front of the right arm (follow from 21.6 s, let go by 24.7 s).
    Hook 0 leaves at 26 s, hook 1 at 32 s: success holds from then, and the trial ends about 33 s."""
    return {side: arm_plan(side) for side in ("left", "right")}
