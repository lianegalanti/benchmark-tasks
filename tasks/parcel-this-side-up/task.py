"""On the station table, in front of the left arm: a small brown cardboard parcel (7 x 7 x 14 cm, 300 g) lies on one of
its long sides, its length running across the table. Each of its four long sides carries a white label with a bold
black "this way up" arrow, and one end is a red top panel: the arrows point along the parcel to that end, which is
the end toward the robot's right. In front of the middle of the robot, a little to its right, is a green square of
floor-marking tape (12 x 12 cm): the pallet mark.

The robot must stand the parcel up on the pallet mark the way its arrows say, red top panel up, and let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import Shape, UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
# "end": the score keeps the best "lifted or tipped up" milestone itself (below), so that a parcel knocked off the
# table can still zero it, which "best" can't express.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Put the parcel on the green pallet mark."

GREEN = (0.15, 0.62, 0.25)


def _f(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects."""
    return tuple(float(x) for x in v)


PARCEL = UsdBody("parcel", usd="assets/parcel.usda", pos=_f(g.PARCEL_START), rot_xyzw=_f(g.PARCEL_START_XYZW),
                 size=_f(g.SIZE), mass=g.MASS, friction=(0.9, 0.8), torsional_patch=0.01)
MARK = Shape("pallet_mark", pos=_f((*g.MARK_XY, g.TOP + g.MARK[2] / 2)), size=_f(g.MARK), color=GREEN)
OBJECTS = [MARK, PARCEL]

# -- monitoring and scoring ------------------------------------------------------------------------------------------
STILL_SPEED, STILL_SPIN = 0.02, 0.5   # m/s, rad/s
PAD_TIP = 0.013                       # fingertips past the grasp point (yam_arena.constants.PAD_TIP_PAST_TCP)
WRIST = 0.134                         # the gripper link's origin behind the grasp point (TCP_OFFSET)

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(best=0.0, fallen=False, on_mark=False, right_way=False, held=False, clear=False, still=False,
                  tilt=0.0, centre_off=1.0, low=0.0, score=0.0)


_reset_state()


def update(pos, R, lin_vel, ang_vel, grippers) -> float:
    """Take one control step: the parcel's centre, rotation and velocities, and per gripper (grasp point, finger
    opening, wrist point, fingertip point), all station frame. Returns the score now."""
    pos = np.asarray(pos, float)
    k = g.classify(pos, R)
    if k["fallen"]:
        _STATE["fallen"] = True
    held, clear = False, True
    for grasp, opening, wrist, tip in grippers:
        near = float(g.box_distance([grasp], pos, R)[0]) < g.HOLD_IN
        held = held or (near and opening < g.HOLD_OPEN)
        clear = clear and g.segment_distance(wrist, tip, pos, R) >= g.CLEAR
    still = bool(np.linalg.norm(lin_vel) < STILL_SPEED and np.linalg.norm(ang_vel) < STILL_SPIN)
    if k["lifted"] or k["upright_any"]:
        _STATE["best"] = max(_STATE["best"], 0.25)
    let_go = bool(not held and clear)
    if _STATE["fallen"]:
        s = 0.0
    elif k["on_mark"] and let_go and still:
        s = 1.0 if k["upright_right"] else 0.5
    else:
        s = _STATE["best"]
    _STATE.update(on_mark=k["on_mark"], right_way=k["upright_right"], held=held, clear=clear, still=still,
                  tilt=k["tilt"], centre_off=k["centre_off"], low=k["low"], score=s)
    return s


def _grippers(environment):
    from scenekit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = []
    for side in ("left", "right"):
        q = arm_joints(environment, side, ARM_JOINTS)
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        opening = float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())
        out.append((tcp, opening, tcp - rot[:, 2] * WRIST, tcp + rot[:, 2] * PAD_TIP))
    return out


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    p = object_states(environment, ["parcel"])["parcel"]
    update(p.pos, g.rot(p.quat_xyzw), p.lin_vel, p.ang_vel, _grippers(environment))


def success(objects) -> bool:
    """Standing on the mark the right way up: its centre within 3 cm of the mark's centre, resting on it, its up axis
    (the way the arrows point) within 15 degrees of straight up, let go (no gripper holding it, every gripper's centre
    line 2 cm or more from it) and still; held 1 s, and the parcel never went off the table."""
    return bool(_STATE["score"] == 1.0)


def score(objects) -> float:
    """0 once the parcel has gone off the table. Otherwise, at the end: 1.0 standing on the mark the right way up, let
    go and still; 0.5 resting on the mark, let go and still, any other way (upside down, or lying); else 0.25 if it was
    at some point lifted 3 cm or tipped upright (either end up, within 15 degrees), else 0."""
    return float(_STATE["score"])
