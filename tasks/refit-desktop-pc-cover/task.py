"""On the station table, ahead of the right arm: a small-form-factor desktop PC (steel case 29.2 x 29 x 9.4 cm, front
away from the robot) lying flat with its cover off, so its motherboard, CPU cooler, memory, power supply and drive
are open to view. Its rim is folded in, with three hook slots in each side flange and an orange spring latch button
in the left flange. Ahead of the left arm, the case's steel cover (29.2 x 29.5 cm, 0.6 kg) lies on the table, outer
face up, resting on its side lips; two small black pull tabs stand on its rear edge.

The robot must put the cover back on: lower it flat so its six hooks drop through the slots, slide it 13 to 16 mm
toward the case's front until the latch button pops up through the cover's latch hole, and let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Slider, on_reset, on_step, part  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Put the cover back on the computer so that it latches, then let go of it."

# -- objects ----------------------------------------------------------------------------------------------------------
LATCH = Slider("latch", rest=(g.LATCH_XY[0], g.LATCH_XY[1], g.RIM), boxes=tuple(g.latch_boxes()),
               travel=g.LATCH_TRAVEL, rate=g.LATCH_RATE, damping=3.0, mass=0.005, friction=0.3)
CASE = part("pc_case", pos=(g.CASE_XY[0], g.CASE_XY[1], g.TOP), boxes=g.case_boxes(), mass=g.CASE_MASS,
            sliders=(LATCH,), color=g.STEEL, friction=0.8)
COVER = part("pc_cover", pos=g.COVER_START, boxes=g.cover_boxes(), mass=g.COVER_MASS, color=g.STEEL, friction=0.6)
OBJECTS = [CASE, COVER]

# -- monitor and scoring ----------------------------------------------------------------------------------------------
DROP_SPEED = 0.55        # m/s downward: the speed a 1.5 cm free fall reaches
KNOCK_M, KNOCK_DEG = 0.010, 2.0
CLEAR_M = 0.03           # every gripper point this far from the cover counts as let go
STILL_SPEED, STILL_SPIN = 0.01, 0.2
FINGER_T = 0.010         # finger thickness outside its pad (m), for the gripper's sampled points

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(violation=False, why=[], stage="off", released=True, still=True, case0=None)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def gripper_points(tcp, rot, opening: float) -> np.ndarray:
    """Points on one gripper (world): its TCP, the wrist, and both fingers' outer faces from the pad tips up 4 cm."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    x, z = rot[:, 0], rot[:, 2]
    pts = [tcp, tcp - 0.10 * z]
    for side in (-1.0, 1.0):
        for up in (-0.013, 0.0, 0.027):
            pts.append(tcp + side * (opening / 2 + FINGER_T) * x - up * z)
            pts.append(tcp + side * (opening / 2) * x - up * z)
    return np.array(pts)


def _cover_distance(points, cover_pos, cover_q) -> float:
    r = g.quat_matrix(cover_q)
    local = (r.T @ (np.asarray(points, float) - np.asarray(cover_pos, float)).T).T
    lo, hi = g.COVER_HULL
    plate = g.dist_to_box(local, lo, np.array([hi[0], hi[1], g.PLATE_T]))
    gr = g.GRIP
    tabs = [g.dist_to_box(local, np.array([gr["x0"], side * g.TAB_Y - g.TAB_W / 2, 0.0]),
                          np.array([gr["x1"], side * g.TAB_Y + g.TAB_W / 2, hi[2]])) for side in (1, -1)]
    return float(np.min(np.minimum(plate, np.minimum(*tabs))))


def update(case, cover, latch_q: float, grippers) -> None:
    """One control step's bookkeeping. ``case`` and ``cover`` have ``pos`` (body origin), ``quat_xyzw``, ``lin_vel``
    and ``ang_vel``; ``latch_q`` is the latch plunger's joint position (m, 0 fully up); ``grippers`` lists each arm's
    sampled gripper points (world)."""
    cpos, cq = np.asarray(case.pos, float), np.asarray(case.quat_xyzw, float)
    if _STATE["case0"] is None:
        _STATE["case0"] = (cpos.copy(), cq.copy())
    p0, q0 = _STATE["case0"]
    turned = g.quat_matrix(q0).T @ g.quat_matrix(cq)
    angle = math.degrees(math.acos(float(np.clip((np.trace(turned) - 1) / 2, -1.0, 1.0))))
    if np.linalg.norm(cpos[:2] - p0[:2]) > KNOCK_M or angle > KNOCK_DEG:
        _flag("the case was moved")
    if cover.lin_vel[2] < -DROP_SPEED:
        _flag("the cover was dropped")
    for name, s in (("case", case), ("cover", cover)):
        if s.pos[2] < g.TOP - 0.03:
            _flag(f"the {name} fell off the table")
    rc = g.quat_matrix(cq)
    to_case = lambda pts: (rc.T @ (np.asarray(pts, float) - cpos).T).T  # noqa: E731
    for pts in grippers:
        if np.any(g.inside_case(to_case(pts))):
            _flag("a gripper went into the open case")
    rv = g.quat_matrix(cover.quat_xyzw)
    edge = np.array([[sx * g.HX, sy * g.HY, 0.0] for sx in (-1, 0, 1) for sy in (-1, 0, 1)])
    cover_pts = to_case((rv @ edge.T).T + np.asarray(cover.pos, float))
    if np.any(g.inside_case(cover_pts) & (cover_pts[:, 2] < g.RIM - 0.010)):
        _flag("the cover went into the open case")
    offset, rot = g.cover_in_case(cpos, cq, cover.pos, cover.quat_xyzw)
    _STATE["stage"] = g.stage(offset, rot, latch_q)
    _STATE["released"] = all(_cover_distance(pts, cover.pos, cover.quat_xyzw) >= CLEAR_M for pts in grippers)
    _STATE["still"] = bool(np.linalg.norm(cover.lin_vel) < STILL_SPEED and np.linalg.norm(cover.ang_vel) < STILL_SPIN)


@on_step
def _monitor(environment) -> None:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.scene import object_states

    base = environment.unwrapped
    grippers = []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        q = robot.data.joint_pos
        q = getattr(q, "torch", q).detach().cpu().numpy()[0].astype(float)
        names = list(robot.joint_names)
        arm = np.array([q[names.index(n)] for n in ARM_JOINTS])
        opening = float(sum(abs(q[names.index(n)]) for n in FINGER_JOINTS))
        tcp, rot = ik.tcp_pose(arm, station.BASES[side])
        grippers.append(gripper_points(tcp, rot, opening))
    states = object_states(environment, ["pc_case", "pc_cover"])
    update(states["pc_case"], states["pc_cover"], states["pc_case"].sliders.get("latch", -1.0), grippers)


def _done() -> bool:
    return bool(not _STATE["violation"] and _STATE["stage"] == "latched" and _STATE["released"] and _STATE["still"])


def success(objects) -> bool:
    """The cover is latched (seated, slid 13 to 16 mm, latch button up through its hole), no gripper within 3 cm of it,
    still, and nothing has gone wrong all trial. Must hold for 1 s."""
    return _done()


def score(objects) -> float:
    """0 after any violation. Otherwise, at the end: 1.0 latched, let go and still; 0.75 latched but still held (or a
    gripper within 3 cm); 0.5 seated and slid at least 9 mm (every hook under the flange) but not latched; 0.25 seated
    flat on the rim with all hooks down through their slots; 0 otherwise."""
    if _STATE["violation"]:
        return 0.0
    if _done():
        return 1.0
    return {"latched": 0.75, "engaged": 0.5, "seated": 0.25}.get(_STATE["stage"], 0.0)


# -- the scripted demo (open loop, from the fixed starting layout) ----------------------------------------------------
CARRY_UP = 0.035          # cover underside this far above the rim while carried: lips 2.3 cm clear of the case
PUSH_TO = 0.0205          # commanded slide (m): 4.5 mm past the stems' stop at 16 mm (the pushing arms lag ~5 mm)
DROP_AIM = 0.003          # lowered 3 mm into the window: the loaded arms trail their targets by up to 5 mm
PRESS = 0.0015            # grips commanded this far below seated, so the arms hold the cover flat on the rim (else
                          # the latch plunger tilts it up and the hooks' toes jam under the flange as it slides)


def cover_origin(s: float, up: float = 0.0) -> np.ndarray:
    """The cover's origin over the case, slid ``s`` (0 = hooks straight over the slots' drop position), ``up`` above
    the rim."""
    return np.array([g.CASE_XY[0] - g.SLIDE + s, g.CASE_XY[1], g.TOP + g.RIM + up])


def demo():
    """Each arm pinches one of the cover's two rear pull tabs, 12 cm either side of its middle (fingers closing front
    to back). Together they lift it 3.5 cm above the rim, carry it sideways over the case to 3 mm inside the hooks' drop window, lower it flat so the
    hooks drop through the slots, push it toward the front against the hooks' stop (16 mm) so the latch pops up, let
    go and back away."""
    off = g.grip_tcp_offset()
    start = np.asarray(g.COVER_START, float)
    above_rim = g.TOP + g.RIM + CARRY_UP
    plans = {}
    for side, dy in (("left", g.GRASP_DY), ("right", -g.GRASP_DY)):
        yaw = g.YAW[side]

        def at(origin, dz=0.0, dx=0.0):
            p = np.asarray(origin, float) + off + np.array([dx, dy, dz])
            return tuple(float(v) for v in p)

        lift = start.copy()
        lift[2] = above_rim
        drop = cover_origin(DROP_AIM, -PRESS)
        home = cover_origin(PUSH_TO, -PRESS)
        plans[side] = [
            Waypoint(2.5, at(start, 0.06), FINGER_OPEN, "above the grip", yaw),
            Waypoint(1.5, at(start), FINGER_OPEN, "down to the grip", yaw),
            Waypoint(1.5, at(start), FINGER_CLOSED, "pinch the grip", yaw),
            Waypoint(1.5, at(lift), FINGER_CLOSED, "lift", yaw),
            Waypoint(3.0, at(cover_origin(DROP_AIM, CARRY_UP)), FINGER_CLOSED, "carry over the case", yaw),
            Waypoint(0.8, at(cover_origin(DROP_AIM, CARRY_UP)), FINGER_CLOSED, "settle", yaw),
            Waypoint(2.5, at(drop), FINGER_CLOSED, "lower: hooks into slots", yaw),
            Waypoint(0.8, at(drop), FINGER_CLOSED, "seated", yaw),
            Waypoint(3.0, at(home), FINGER_CLOSED, "push to the latch", yaw),
            Waypoint(0.8, at(home), FINGER_CLOSED, "latched", yaw),
            Waypoint(1.5, at(home, PRESS), FINGER_OPEN, "let go", yaw),
            Waypoint(1.0, at(home, 0.04), FINGER_OPEN, "lift away", yaw),
            Waypoint(1.5, at(home, 0.07, -0.02), FINGER_OPEN, "back off", yaw),
        ]
    return plans
