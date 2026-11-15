"""On the station table, two plain steel strut plates (3-hole flat plate fittings, 136.5 x 41.3 x 6.35 mm, 0.26 kg,
three 14.3 mm holes on the centre line) stand on end in two slotted wooden loading blocks, one in front of each arm,
holes facing the robot. Above the far side of the table runs an overhead conveyor rail (40 cm above the table)
carrying three hooks of 4 mm steel wire, 30 cm apart: each hangs from the rail on a stem, 46 cm in front of the arm
bases, and ends in a horizontal leg (20 cm long, 15 cm above the table) pointing back toward the robot, with a 4 mm
upturn at its end. The hooks enter from the robot's left and travel toward its right at a steady 5 cm/s; they never
stop. A red tape mark on the table, 60 cm right of the midpoint between the arms, marks where hooks leave the station.

The robot must hang each plate on its own hook by an end hole (the hook's leg through the hole, the hole above the
plate's centre of mass, so the plate hangs plumb below the hook) before that hook passes the red exit mark, and let
go of it.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, UsdPart, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
# "end": the score keeps its own record over the trial (below): which plate was on which hook as that hook left.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Hang the two steel plates on the passing conveyor hooks, one plate per hook, each by the hole at one "
               "of its ends so that it hangs straight down, before the hooks reach the red mark, and let go of them. "
               "The hooks keep moving and won't stop for you.")

WIRE_BLACK = (0.16, 0.16, 0.17)
RAIL_GREY = (0.30, 0.31, 0.33)
RED = (0.85, 0.08, 0.08)
WOOD = (0.62, 0.45, 0.27)
ZINC = (0.72, 0.74, 0.77)


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
    return [DrivenBox(f"hook{i}_{part}", pos=_f(c), size=_f(s), color=WIRE_BLACK, path=conveyor(c),
                      friction=(0.6, 0.5))
            for part, (c, s) in g.hook_boxes(i, 0.0).items()]


HOOKS = [box for i in range(g.N_HOOKS) for box in _hook(i)]
RAIL = Box("rail", pos=_f(g.RAIL_CENTRE), size=_f(g.RAIL), color=RAIL_GREY)
EXIT_MARK = Box("exit_mark", pos=_f((0.36, g.EXIT_Y, g.TOP + 0.001)), size=(0.30, 0.02, 0.002), color=RED)
HOLDERS = [Box(f"block_{p[-1]}_{w}", pos=_f(c), size=_f(sz), color=WOOD)
           for p in g.PARTS for w, (c, sz) in g.holder_boxes(p).items()]
PARTS = [UsdPart(name, usd="assets/strut_plate.usda", pos=_f(g.part_start(name)),
                 size=(float(g.THICK), float(g.WIDTH), float(g.HEIGHT)), mass=float(g.MASS), color=ZINC,
                 friction=(0.6, 0.5), torsional_patch=0.01) for name in g.PARTS]
OBJECTS = [RAIL, EXIT_MARK, *HOLDERS, *HOOKS, *PARTS]

# -- monitoring and scoring ------------------------------------------------------------------------------------------
LIFTED = 0.03  # m: a plate's lowest point this far above the table (clear of its 2.5 cm slot) counts as lifted
OFF_TABLE = 0.01  # m: a hung plate's lowest point at least this far above the table
REST_TOL = 0.005  # m: a hung plate rests on the hook's leg: at most this gap between the leg and the hole's edge
PLUMB_DEG = 10.0  # a hung plate's length within this of vertical
STILL_SPIN = 0.5  # rad/s: settled: the plate turning slower than this
STILL_SPEED = 0.03  # m/s: settled: the plate moving within this of its hook's velocity
CLEAR = 0.02  # m: let go: every link of both arms (fingers, gripper and the rest) at least this far from the plate
HANG_S = 1.0  # s: a plate must have hung, settled and free, this long without a break when its hook reaches the mark
LAG = 0.01  # m: a hook whose leg falls this far behind its 5 cm/s schedule (stopped or slowed) credits no plate
HUNG, ON_HOOK, LIFT_CREDIT = 0.5, 0.25, 0.1  # per plate
HOOK_VEL = np.array([0.0, -g.SPEED, 0.0])

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(best={p: 0.0 for p in g.PARTS}, credited={p: None for p in g.PARTS}, judged=set(), since={},
                  stalled=set())


reset_state()


def arm_clearance(pos, R, hulls) -> float:
    """The smallest distance (m) between the plate, at origin ``pos`` and rotation ``R``, and the arms' collision
    hulls ``hulls`` (a list of (link, analysis.Hull) in the station frame); negative when they overlap. Each hull is
    tested against points on the plate's surface (5 mm apart) and the plate's boxes against the hull's vertices."""
    from yam_arena import analysis

    pts = g.part_points(pos, R)
    lo, hi = pts.min(0) - 0.05, pts.max(0) + 0.05
    best = np.inf
    for _, h in hulls:
        if np.any(h.vertices.max(0) < lo) or np.any(h.vertices.min(0) > hi):
            continue  # more than 5 cm away
        best = min(best, float(analysis._signed(pts, h).min()))
        local = (h.vertices - pos) @ R  # into the part frame
        for c, sz in g.PART_BOXES.values():
            best = min(best, float(analysis.box_signed_distance(local, c, np.eye(3), sz).min()))
    return best


def part_clearance(a, b) -> float:
    """The smallest distance (m) between two plates, each given as (origin, rotation matrix); negative when they
    overlap. Points on each plate's surface (5 mm apart) against the other's boxes, both ways."""
    best = np.inf
    for (pa, Ra), (pb, Rb) in ((a, b), (b, a)):
        local = (g.part_points(pa, Ra) - pb) @ Rb  # into b's frame
        for c, sz in g.PART_BOXES.values():
            best = min(best, float(analysis_box(local, c, sz).min()))
    return best


def analysis_box(points, centre, size):
    from yam_arena import analysis

    return analysis.box_signed_distance(points, np.asarray(centre), np.eye(3), np.asarray(size))


def hanging_free(state, hook_bar_y: float, hulls, others=()) -> bool:
    """The hook alone holds the plate up, and it has settled there: the hook's leg through an end hole that is
    above the plate's centre of mass, resting on the hole's upper edge (gap at most 5 mm); the plate within 10
    degrees of plumb, turning slower than 0.5 rad/s and moving within 3 cm/s of the hook; its lowest point 1 cm or
    more above the table; every link of both arms at least 2 cm from every part of it, and every other plate
    (``others``: (origin, rotation) pairs) at least 2 cm from it. ``state``: (origin, rotation, velocity, spin)."""
    pos, R, vel, spin = state
    gap = g.rest_gap(pos, R, hook_bar_y)
    if gap is None or gap > REST_TOL or g.tilt_deg(R) > PLUMB_DEG or g.lowest_z(pos, R) < g.TOP + OFF_TABLE:
        return False
    if np.linalg.norm(spin) > STILL_SPIN or np.linalg.norm(np.asarray(vel) - HOOK_VEL) > STILL_SPEED:
        return False
    if any(part_clearance((pos, R), o) < CLEAR for o in others):
        return False
    return arm_clearance(pos, R, hulls() if callable(hulls) else hulls) >= CLEAR


def update(parts: dict, hook_ys, hulls, t: float) -> None:
    """Take one control step at episode time ``t``: each plate's (origin, rotation matrix, velocity, spin), each
    hook's leg position along the rail (y), and both arms' collision hulls in the station frame (a list, or a
    function returning it).

    A plate reaches ON_HOOK once a hook's leg passes through any of its holes, and LIFT_CREDIT once its lowest point
    is 3 cm above the table. A plate hangs freely on a hook while the hook alone supports it and it has settled
    (hanging_free). A hook whose leg is ever more than 1 cm behind its schedule (the robot stopped or slowed it)
    credits no plate. When a hook's leg first reaches the exit mark's centre line, one not-yet-credited plate that
    has hung freely on it, without a break, for the last 1 s is credited HUNG. A hook credits at most one plate; a
    credited plate stays credited."""
    for name, (pos, R, *_rest) in parts.items():
        if any(g.bar_through_hole(pos, R, y) for y in hook_ys):
            _STATE["best"][name] = max(_STATE["best"][name], ON_HOOK)
        if g.lowest_z(pos, R) >= g.TOP + LIFTED:
            _STATE["best"][name] = max(_STATE["best"][name], LIFT_CREDIT)
    cache = {}

    def arm_hulls():
        if "h" not in cache:
            cache["h"] = hulls() if callable(hulls) else hulls
        return cache["h"]

    since = _STATE["since"]
    for i, y in enumerate(hook_ys):
        if i in _STATE["judged"]:
            continue
        if y - g.hook_y(i, t) > LAG:
            _STATE["stalled"].add(i)  # held back: a real slip-clutch drive would have stopped or slowed
        for name, state in parts.items():
            others = [v[:2] for k, v in parts.items() if k != name]
            if (_STATE["credited"][name] is None and i not in _STATE["stalled"]
                    and hanging_free(state, y, arm_hulls, others)):
                since.setdefault((name, i), t)
            else:
                since.pop((name, i), None)
        if y > g.EXIT_Y:
            continue
        _STATE["judged"].add(i)
        for name in parts:
            start = since.get((name, i))
            if _STATE["credited"][name] is None and start is not None and t - start >= HANG_S - 1e-6:
                _STATE["credited"][name] = i
                break


def _arm_hulls(environment):
    """Both arms' URDF collision hulls placed at the simulated link poses (station frame)."""
    from motionkit import _np
    from yam_arena import analysis

    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        poses = {}
        for name, pose in zip(robot.body_names, _np(robot.data.body_link_pose_w)[0], strict=True):
            m = analysis.pose7_to_matrix(pose)
            m[:3, 3] -= origin
            poses[name] = m
        out += analysis.world_hulls(poses)
    return out


@on_step
def _monitor(environment) -> None:
    from motionkit import clock
    from yam_arena.scene import object_states

    names = [*g.PARTS, *(f"hook{i}_bar" for i in range(g.N_HOOKS))]
    s = object_states(environment, names)
    parts = {p: (s[p].pos, g.rot(s[p].quat_xyzw), np.asarray(s[p].lin_vel), np.asarray(s[p].ang_vel))
             for p in g.PARTS}
    update(parts, [float(s[f"hook{i}_bar"].pos[1]) for i in range(g.N_HOOKS)], lambda: _arm_hulls(environment),
           clock())


def success(objects) -> bool:
    """Both plates credited: each hung freely and settled on its own hook for the last 1 s before that hook passed
    the exit mark."""
    return all(v is not None for v in _STATE["credited"].values())


def score(objects) -> float:
    """Per plate, summed: 0.5 if it hung freely on its own hook for the whole last 1 s before that hook's leg passed
    over the exit mark's centre line (the leg through an end hole above the centre of mass, resting on it within
    5 mm; within 10 degrees of plumb, turning under 0.5 rad/s and within 3 cm/s of the hook's velocity; lowest point
    1 cm or more above the table; every link of both arms and the other plate at least 2 cm from it), and the robot
    never held that hook back (its leg never more than 1 cm behind its 5 cm/s schedule); otherwise 0.25 if a hook's
    leg was at some point through one of its holes; otherwise 0.1 if it was at some point lifted 3 cm off the table;
    else 0."""
    return float(sum(HUNG if _STATE["credited"][p] is not None else _STATE["best"][p] for p in g.PARTS))


# -- the scripted demo ---------------------------------------------------------------------------------------------
GRIP_DEPTH = 0.012  # the TCP this far below the plate's top end: the pads pinch its two long edges beside hole 0
TCP_ABOVE_ORIGIN = g.PART_TOP - GRIP_DEPTH
TRAVEL_Z = g.TOP + 0.21  # TCP height while carrying
APPROACH_X = g.TIP_X - 0.04  # the plate's origin in front of the hook's upturn, before threading
RELEASE_OPEN = -0.036  # finger target when letting go: pads 7.2 cm apart, 1.5 cm clear of the plate's edges
# finger target while holding: 2.5 mm inside the plate's half width, so each finger presses with a few newtons, the two
# balance only with the plate centred between them, and it can't drift sideways in the hand. (Commanded fully closed,
# both fingers push at their effort limit and the plate floats sideways between them: in Isaac Sim it slid 1.8 cm
# off centre while being lifted, and hole 0 missed the hook's wire.)
GRIP = -g.WIDTH / 2 + 0.0025
# where the arm's TCP actually is, relative to its target, while it carries the plate along with a hook (measured in
# Isaac Sim): one control step behind the hook (2.5 mm, toward +y) and sagging 3 mm under gravity. The plan aims
# that much ahead and above, so the hole meets the wire and its 4 mm upturn centred (3.1 mm clear above and below).
TRACKING = np.array([0.0, 0.0025, -0.003])
DT = 0.05  # one control step at 20 Hz: while following, a waypoint per step
PARK = {"left": (0.16, 0.33, g.TOP + 0.20), "right": (0.16, -0.33, g.TOP + 0.20)}  # behind the hooks, over the blocks
# who hangs what: (part, hook, when the follow starts). The meeting points are 2 cm toward the arm from each hook's
# entry into the arm's comfortable reach.
JOBS = {"left": ("plate_a", 0, (g.Y0 - 0.42) / g.SPEED), "right": ("plate_b", 1, (g.Y0 + g.SPACING + 0.08) / g.SPEED)}
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
    lowered = g.HUNG_Z + 0.002  # let go with the hole's edge 2 mm above the leg
    z = g.THREAD_Z if phase in ("match", "thread", "settle") else g.THREAD_Z + u * (lowered - g.THREAD_Z) \
        if phase == "lower" else lowered + u * 0.06 if phase == "rise" else lowered
    finger = GRIP if phase not in ("release", "ride open", "rise") else \
        GRIP + u * (RELEASE_OPEN - GRIP) if phase == "release" else RELEASE_OPEN
    return x, z, finger, phase


def arm_plan(side: str) -> list[Waypoint]:
    """Pick the plate out of its block by pinching its two long edges at the top end, wait above its block, carry it
    in front of the hook's upturn as the follow starts (hole 0 level with the leg), then a waypoint every control
    step: match the hook, slide hole 0 over the upturn and onto the leg, lower its edge onto the leg, open the
    fingers, ride along open, rise while following, and park."""
    part, hook, t0 = JOBS[side]
    px, py = g.PART_XY[part]
    grasp_z = g.part_start(part)[2] + TCP_ABOVE_ORIGIN
    plan = [
        Waypoint(1.5, (px, py, TRAVEL_Z), FINGER_OPEN, "above the plate"),
        Waypoint(1.2, (px, py, grasp_z), FINGER_OPEN, "descend"),
        Waypoint(0.6, (px, py, grasp_z), GRIP, "close"),
        Waypoint(1.0, (px, py, TRAVEL_Z), GRIP, "pick up"),
    ]
    carry_s = 1.3
    wait = t0 - carry_s - sum(w.duration for w in plan)
    if wait > 0:
        plan.append(Waypoint(wait, (px, py, TRAVEL_Z), GRIP, "wait"))
    plan.append(Waypoint(carry_s, _f(np.array([APPROACH_X, g.hook_y(hook, t0), g.THREAD_Z + TCP_ABOVE_ORIGIN])
                                     - TRACKING), GRIP, "carry"))
    n = round(FOLLOW_S / DT)
    for k in range(1, n + 1):
        t = t0 + k * DT
        x, z, finger, phase = follow_target(hook, t0, t)
        plan.append(Waypoint(DT, _f(np.array([x, g.hook_y(hook, t), z + TCP_ABOVE_ORIGIN]) - TRACKING), finger, phase))
    plan.append(Waypoint(1.2, PARK[side], RELEASE_OPEN, "park"))
    # yam_scripted ends the rollout 3 s after the plan does: stay parked until hook 1 has passed and success held
    plan.append(Waypoint(END_S - sum(w.duration for w in plan), PARK[side], RELEASE_OPEN, "wait parked"))
    return plan


END_S = 35.0  # the plans end here: hook 1 passes the exit mark at 32 s and success holds from 33 s


def demo():
    """Left arm: plate A onto hook 0 as it passes in front of the left arm (follow from 5.6 s, let go by 8.7 s).
    Right arm: plate B onto hook 1 as it passes in front of the right arm (follow from 21.6 s, let go by 24.7 s).
    Hook 0 leaves at 26 s, hook 1 at 32 s: success holds from then, and the trial ends about 33 s."""
    return {side: arm_plan(side) for side in ("left", "right")}
