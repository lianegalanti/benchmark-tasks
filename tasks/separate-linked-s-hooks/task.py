"""On the station table, in front of the left arm, stands a shop display: an upright plywood board with a chrome rail
on two brackets along its top, and a straight chrome peg lower down to the right, pointing at the robot. Two large
aluminium S-hooks (15 cm tall, 10 mm rod) hang hooked together from the rail: hook A hangs on the rail by its top
curl, edge-on to the robot, and hook B hangs from hook A's bottom curl by its own top curl, facing the robot.

The robot must separate the two hooks and hang each one by itself, one on the rail and the other on the peg, without
putting either of them down.

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
# "end": the score keeps the best milestone reached itself (below), so that a hook put down on the table can still
# zero it, which "best" can't express.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("The two S-hooks on the display rail are hooked into each other. Separate them and hang each one by "
               "itself, one on the rail and the other on the peg, without putting either down.")

PLYWOOD = (0.76, 0.62, 0.44)
CHROME = (0.55, 0.57, 0.60)
NAMES = ("hook_a", "hook_b")


def _f(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects."""
    return tuple(float(x) for x in v)


def _hook(name: str, pos, xyzw) -> UsdBody:
    return UsdBody(name, usd="assets/s_hook.usda", pos=_f(pos), rot_xyzw=_f(xyzw), size=(g.WIDTH, 2 * g.ROD_R, g.HEIGHT),
                   mass=g.MASS, friction=(0.7, 0.6), solver_iters=32, torsional_patch=0.005)


HOOK_A = _hook("hook_a", g.A_START, g.A_START_XYZW)
HOOK_B = _hook("hook_b", g.B_START, g.B_START_XYZW)
BOARD = Shape("board", pos=_f(g.BOARD_POS), size=_f(g.BOARD), color=PLYWOOD)
BRACKETS = [Shape(f"bracket_{i}", pos=_f(((g.BOARD_FACE + g.RAIL_X - 0.008) / 2, y, g.RAIL_Z)), size=_f(g.BRACKET),
                  color=CHROME) for i, y in enumerate(g.BRACKET_Y)]
_RAIL_LEN = g.RAIL_Y[1] - g.RAIL_Y[0]
RAIL = Shape("rail", kind="capsule", pos=_f((g.RAIL_X, sum(g.RAIL_Y) / 2, g.RAIL_Z)), radius=g.SUPPORT_R,
             height=float(_RAIL_LEN - 2 * g.SUPPORT_R), axis="Y", size=_f((2 * g.SUPPORT_R, _RAIL_LEN, 2 * g.SUPPORT_R)),
             color=CHROME)
# the peg: its rear cap buried in the board, its rounded tip at PEG_TIP_X
PEG = Shape("peg", kind="capsule", pos=_f(((g.BOARD_FACE + g.PEG_TIP_X + g.SUPPORT_R) / 2, g.PEG_Y, g.PEG_Z)),
            radius=g.SUPPORT_R, height=float(g.PEG_LEN - g.SUPPORT_R), axis="X",
            size=_f((g.PEG_LEN, 2 * g.SUPPORT_R, 2 * g.SUPPORT_R)), color=CHROME)
OBJECTS = [BOARD, *BRACKETS, RAIL, PEG, HOOK_A, HOOK_B]

# -- monitoring and scoring ------------------------------------------------------------------------------------------
PAD_TIP = 0.013                       # fingertips past the grasp point (yam_arena.constants.PAD_TIP_PAST_TCP)

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(best=0.0, down=False, on=(None, None), linked=True, gap=0.0, held=False, clear=False,
                  still=False)


_reset_state()


def milestone(pa, Ra, pb, Rb) -> float:
    """The score this pair of hook poses alone is worth: 0.5 once they are no longer hooked together (neither hook's
    rod passes through a curl of the other); 0.25 once they are at least 1 cm apart everywhere (one lifted off the
    other); else 0."""
    if not g.linked(pa, Ra, pb, Rb):
        return 0.5
    if g.gap(pa, Ra, pb, Rb) >= g.APART:
        return 0.25
    return 0.0


def update(hooks, grippers) -> None:
    """Take one control step: per hook (position, rotation matrix, linear and angular velocity), and per gripper
    (grasp point, finger opening, wrist point, fingertip point), all station frame."""
    (pa, Ra, va, wa), (pb, Rb, vb, wb) = [(np.asarray(p, float), R, v, w) for p, R, v, w in hooks]
    for p, R, _, _ in ((pa, Ra, va, wa), (pb, Rb, vb, wb)):
        if g.lowest_z(p, R) < g.TOP + g.ON_TABLE:
            _STATE["down"] = True
    held, clear = False, True
    for grasp, opening, wrist, tip in grippers:
        line = g.seg_points(wrist, tip)
        for p, R in ((pa, Ra), (pb, Rb)):
            held = held or (g.dist_to_hook([grasp], p, R) < g.HOLD_R and opening < g.HOLD_OPEN)
            clear = clear and g.dist_to_hook(line, p, R) >= g.GRIPPER_CLEAR
    still = all(np.linalg.norm(v) < g.STILL_SPEED and np.linalg.norm(w) < g.STILL_SPIN for _, _, v, w in hooks)
    _STATE.update(on=(g.on_support(pa, Ra), g.on_support(pb, Rb)), linked=g.linked(pa, Ra, pb, Rb),
                  gap=g.gap(pa, Ra, pb, Rb), held=held, clear=clear, still=bool(still),
                  best=max(_STATE["best"], milestone(pa, Ra, pb, Rb)))


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

    s = object_states(environment, list(NAMES))
    update([(s[n].pos, g.rot(s[n].quat_xyzw), s[n].lin_vel, s[n].ang_vel) for n in NAMES], _grippers(environment))


def success(objects) -> bool:
    """Each hook hangs by itself on its own support: one hook's curl around the rail and the other's around the
    peg, the hooks no longer hooked together and not touching, no gripper holding either and both grippers' centre
    lines at least 4 cm from both hooks, both still; held 1 s, and neither hook was ever put down on the table."""
    S = _STATE
    on = set(S["on"])
    return bool(not S["down"] and on == {"rail", "peg"} and not S["linked"] and S["gap"] >= g.ALONE
                and not S["held"] and S["clear"] and S["still"])


def score(objects) -> float:
    """Partial credit (success scores 1.0): 0 once either hook has touched the table (put down or dropped).
    Otherwise the best reached during the trial: 0.5 once the hooks were no longer hooked together; 0.25 once
    they were at least 1 cm apart everywhere (one lifted off the other); else 0."""
    return 0.0 if _STATE["down"] else float(_STATE["best"])
