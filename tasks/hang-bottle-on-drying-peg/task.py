"""On the station table, in front of the left arm: a washed baby bottle (a straight wide-neck bottle without its teat
and cap: an open plastic tube 6 cm across and 12 cm long, closed at one end, with an amber screw collar at the open
mouth) lies on its side on a small dish towel, its mouth toward the robot. Ahead of it, nearer the middle, stands a
white drying rack: a flat base with two upright grey pegs, 1.2 cm thick and 15 cm tall, 12 cm apart.

The robot must hang the bottle mouth-down on one of the pegs, so that it hangs on the peg's tip by its inside bottom
with its mouth clear of the base, and let go.

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
# "end": the score keeps the best milestone reached itself (below), so that a bottle knocked off the table can still
# zero it, which "best" can't express.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Hang the washed baby bottle mouth-down on a peg of the drying rack to dry."

WHITE = (0.93, 0.93, 0.91)
GREY = (0.62, 0.64, 0.67)
TOWEL_BLUE = (0.62, 0.76, 0.90)


def _f(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects."""
    return tuple(float(x) for x in v)


BOTTLE = UsdBody("bottle", usd="assets/bottle.usda", pos=_f(g.BOTTLE_START), rot_xyzw=_f(g.BOTTLE_START_XYZW),
                 size=(g.LENGTH, 2 * g.R_OUT, 2 * g.R_OUT), mass=g.MASS, friction=(0.9, 0.8),
                 torsional_patch=0.01)
TOWEL = Shape("towel", pos=_f((*g.TOWEL_XY, g.TOP + g.TOWEL[2] / 2)), size=_f(g.TOWEL), color=TOWEL_BLUE)
RACK_BASE = Shape("rack_base", pos=_f((*g.RACK_XY, g.TOP + g.BASE[2] / 2)), size=_f(g.BASE), color=WHITE)


def _peg(i: int, xy) -> Shape:
    # a capsule from just inside the base (its lower cap buried) up to the rounded tip at PEG_TIP
    height = g.PEG_TIP - g.BASE_TOP - g.PEG_R
    z = (g.PEG_TIP + g.BASE_TOP - g.PEG_R) / 2
    return Shape(f"peg_{i}", kind="capsule", pos=_f((*xy, z)), radius=g.PEG_R, height=float(height), axis="Z",
                 size=_f((2 * g.PEG_R, 2 * g.PEG_R, height + 2 * g.PEG_R)), color=GREY)


PEGS = [_peg(i, xy) for i, xy in enumerate(g.PEGS)]
OBJECTS = [TOWEL, RACK_BASE, *PEGS, BOTTLE]

# -- monitoring and scoring ------------------------------------------------------------------------------------------
STILL_SPEED, STILL_SPIN = 0.02, 0.5   # m/s, rad/s
PAD_TIP = 0.013                       # fingertips past the grasp point (yam_arena.constants.PAD_TIP_PAST_TCP)

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(best=0.0, fallen=False, on_peg=False, hanging=False, held=False, clear=False, still=False,
                  depth=-1.0, tilt=0.0, lowest=0.0)


_reset_state()


def milestone(pos, R) -> float:
    """The score this bottle pose alone is worth: 0.5 with a peg at least 3 cm inside it through its mouth and its
    mouth within 30 degrees of straight down; 0.25 with its lowest point at least 3 cm above the table; else 0."""
    depth = max(g.peg_depth(pos, R, xy) for xy in g.PEGS)
    if depth >= g.INSIDE_MIN and g.tilt_from_down(R) <= g.MOUTH_DOWN:
        return 0.5
    if g.lowest_z(pos, R) >= g.TOP + g.LIFTED:
        return 0.25
    return 0.0


def update(pos, R, lin_vel, ang_vel, grippers) -> None:
    """Take one control step: the bottle's pose and velocities, and per gripper (grasp point, finger opening, wrist
    point, fingertip point), all station frame."""
    pos = np.asarray(pos, float)
    if pos[2] < g.FALLEN:
        _STATE["fallen"] = True
    depth = max(g.peg_depth(pos, R, xy) for xy in g.PEGS)
    tilt, lowest = g.tilt_from_down(R), g.lowest_z(pos, R)
    a0, a1 = g.axis_segment(pos, R)
    held, clear = False, True
    for grasp, opening, wrist, tip in grippers:
        near = g.seg_dist(grasp, grasp, a0, a1) < g.HOLD_R
        held = held or (near and opening < g.HOLD_OPEN)
        clear = clear and g.seg_dist(wrist, tip, a0, a1) >= g.GRIPPER_CLEAR
    on_peg = depth >= g.INSIDE_MIN and tilt <= g.MOUTH_DOWN
    still = bool(np.linalg.norm(lin_vel) < STILL_SPEED and np.linalg.norm(ang_vel) < STILL_SPIN)
    _STATE.update(on_peg=on_peg, hanging=on_peg and lowest >= g.BASE_TOP + g.RIM_CLEAR, held=held, clear=clear,
                  still=still, depth=depth, tilt=tilt, lowest=lowest,
                  best=max(_STATE["best"], milestone(pos, R)))


def _grippers(environment):
    from scenekit import arm_joints, robot_links
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    links = robot_links(environment)
    out = []
    for side in ("left", "right"):
        q = arm_joints(environment, side, ARM_JOINTS)
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        opening = float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())
        wrist = links[side].get("gripper", tcp - rot[:, 2] * 0.134)
        out.append((tcp, opening, wrist, tcp + rot[:, 2] * PAD_TIP))
    return out


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    b = object_states(environment, ["bottle"])["bottle"]
    update(b.pos, g.rot(b.quat_xyzw), b.lin_vel, b.ang_vel, _grippers(environment))


def success(objects) -> bool:
    """Hanging on a peg: the peg at least 3 cm inside through the mouth, the mouth within 30 degrees of straight down,
    the bottle's lowest point at least 5 mm above the rack base, no gripper holding it and both grippers' centre
    lines at least 8 cm from its axis, and still; held 1 s, and the bottle never went off the table."""
    S = _STATE
    return bool(not S["fallen"] and S["hanging"] and not S["held"] and S["clear"] and S["still"])


def score(objects) -> float:
    """Partial credit (success scores 1.0): 0 once the bottle has gone off the table. Otherwise the best reached
    during the trial: 0.5 once a peg was at least 3 cm inside it through its mouth with the mouth within 30 degrees
    of straight down; 0.25 once its lowest point was at least 3 cm above the table (lifted); else 0."""
    return 0.0 if _STATE["fallen"] else float(_STATE["best"])
