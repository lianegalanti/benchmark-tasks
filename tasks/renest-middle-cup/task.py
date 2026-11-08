"""On the station table: a set of three nesting square cups (teal plastic, open at the top). The large cup (12.6 cm
wide, 3.2 cm tall) stands a little left of centre, 22 cm in front of the arms, with the small cup (5 cm wide, 9.5 cm
tall) standing inside it. The medium cup (8.8 cm wide, 6.5 cm tall) stands on its own in front of the right arm.
Nested, each cup has 1.5 cm of clearance to the walls of the next larger one.

The medium cup belongs between the other two, so the robot must take the small cup out of the large one first, put
the medium cup into the large one, and put the small cup back into the medium one (or nest the small cup in the
medium one on the table and then the pair in the large one).

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

import cups as c  # noqa: E402

SCENE = "station"
TOP = c.TOP
SCORE_OVER = "best"  # partial credit for the furthest the trial got; success scores 1.0

INSTRUCTION = "Put the loose cup away in the nested set, so that all three cups are nested inside one another."

# -- objects -------------------------------------------------------------------------------------------------------
CUP_BODIES = {
    key: UsdBody(f"cup_{key}", usd=f"assets/cup_{key}.usda", pos=tuple(float(v) for v in c.START[key]),
                 size=(cup.outer, cup.outer, cup.height), mass=cup.mass, friction=(1.0, 0.8), torsional_patch=0.01)
    for key, cup in c.CUPS.items()
}
OBJECTS = list(CUP_BODIES.values())
NAME = {key: body.name for key, body in CUP_BODIES.items()}

# -- scoring -------------------------------------------------------------------------------------------------------
TILT_MAX = math.radians(10.0)  # a cup is upright when its axis is within this of vertical
SEAT_TOL = 0.01                # a nested cup's underside within this of the floor of the cup around it
CENTRE_TOL = c.GAP + 0.003     # ... and its centre within this of that cup's centre, along each of its sides
TABLE_TOL = 0.01               # the large cup's underside within this of the table top
LIFT_M = 0.01                  # the small cup lifted out: its underside this far above the large cup's rim
CLEAR_M = 0.02                 # let go: both grippers' TCP and pad tips this far outside the stack's outline
PAD_TIP = 0.013                # m: pad tips past the TCP (yam_arena.constants.PAD_TIP_PAST_TCP)

_STATE: dict = {}


def _rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def upright(state) -> bool:
    return bool(_rot(state.quat_xyzw)[2, 2] >= math.cos(TILT_MAX))


def seated(objects, inner: str, outer: str) -> bool:
    """Cup ``inner`` stands upright inside cup ``outer``, also upright: its underside within 1 cm of ``outer``'s floor
    and its centre within 1.8 cm of ``outer``'s centre along each of ``outer``'s sides (1.5 cm is touching a wall)."""
    a, b = objects[NAME[inner]], objects[NAME[outer]]
    if not (upright(a) and upright(b)):
        return False
    rot = _rot(b.quat_xyzw)
    local = rot.T @ (np.asarray(a.pos, float) - np.asarray(b.pos, float))
    return bool(abs(local[2] - c.WALL) <= SEAT_TOL and abs(local[0]) <= CENTRE_TOL and abs(local[1]) <= CENTRE_TOL)


def small_lifted(objects) -> bool:
    """The small cup's underside is at least 1 cm above the large cup's rim."""
    small, large = objects[NAME["small"]], objects[NAME["large"]]
    return bool(small.pos[2] >= large.pos[2] + c.LARGE.height + LIFT_M)


def _stack_box(objects):
    """The full nest's outline: the large cup's footprint, from the table to the small cup's nested rim + 1 cm."""
    large = objects[NAME["large"]]
    half = c.LARGE.outer / 2
    lo = np.array([large.pos[0] - half, large.pos[1] - half, TOP])
    hi = np.array([large.pos[0] + half, large.pos[1] + half, TOP + c.rim_when_nested("small") + 0.01])
    return lo, hi


def grippers_clear(objects) -> bool:
    """Both grippers' TCP and pad tips at least 2 cm outside the stack's outline (measured each control step)."""
    points = _STATE.get("grip_points")
    if points is None:
        return False
    lo, hi = _stack_box(objects)
    for p in points:
        d = np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0))
        if d < CLEAR_M:
            return False
    return True


@on_reset
def _reset_state(environment=None) -> None:
    _STATE["grip_points"] = None


@on_step
def _grippers(environment) -> None:
    """Each gripper's TCP and pad-tip point, station frame."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    from scenekit import arm_joints

    points = []
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        points += [np.asarray(tcp, float), np.asarray(tcp, float) + rot[:, 2] * PAD_TIP]
    _STATE["grip_points"] = points


def success(objects) -> bool:
    """All three nested in order (medium seated in large, small seated in medium), the large cup upright on the
    table, every cup still, and both grippers clear of the stack; held for 1 s."""
    large = objects[NAME["large"]]
    return bool(seated(objects, "medium", "large") and seated(objects, "small", "medium")
                and upright(large) and abs(large.pos[2] - TOP) <= TABLE_TOL
                and all(objects[n].resting() for n in NAME.values()) and grippers_clear(objects))


def distance(objects) -> float:
    """Metres from the medium cup's underside to its seat on the large cup's floor."""
    large, medium = objects[NAME["large"]], objects[NAME["medium"]]
    goal = np.asarray(large.pos, float) + [0.0, 0.0, c.WALL]
    return float(np.linalg.norm(np.asarray(medium.pos, float) - goal))


def score(objects) -> float:
    """Partial credit, the best reached over the trial (success scores 1.0): 0.5 once the medium cup is seated in the
    large one, or the small cup seated in the medium one; 0.25 once the small cup has been lifted out of the large one
    (underside 1 cm above its rim); otherwise 0."""
    if seated(objects, "medium", "large") or seated(objects, "small", "medium"):
        return 0.5
    if small_lifted(objects):
        return 0.25
    return 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Grip heights: the TCP 6 mm below a cup's rim, so the pads hold its top 1.9 cm and the tips stay 1.5 cm or more
# above the rim of the cup it is going into.
GRIP_BELOW_RIM = 0.006
PINCH_DY = -(c.MEDIUM.outer - c.WALL) / 2   # the medium cup is pinched by its right-hand (-y) wall
S_TRAVEL = TOP + 0.175   # the small cup's underside 1.7 cm above the nested medium cup's rim
M_TRAVEL = TOP + 0.13    # the medium cup's underside 3.9 cm above the large cup's rim
ASIDE = (0.20, 0.22)     # where the left arm holds the small cup out of the way
PARK = {"left": (0.20, 0.24, TOP + 0.18), "right": (0.20, -0.24, TOP + 0.18)}
RELEASE_ABOVE = 0.004    # let go with the cup's underside this far above its seat


def _tcp_for(key: str, underside_z: float) -> float:
    return underside_z + c.CUPS[key].height - GRIP_BELOW_RIM


def demo():
    """Left arm: lift the small cup out of the large one and hold it aside. Right arm: pinch the medium cup by its
    right-hand wall, put it into the large cup once the small one is out, and park. Left arm: put the small cup into
    the medium one and park. About 19 s."""
    lx, ly = c.LARGE_XY
    mx, my = c.MEDIUM_XY
    s_grip = _tcp_for("small", c.START["small"][2])
    s_place = _tcp_for("small", TOP + 2 * c.WALL + RELEASE_ABOVE)
    m_grip = _tcp_for("medium", TOP)
    m_place = _tcp_for("medium", TOP + c.WALL + RELEASE_ABOVE)
    O, C = FINGER_OPEN, FINGER_CLOSED

    left = [
        Waypoint(2.0, (lx, ly, S_TRAVEL), O, "above the small cup"),
        Waypoint(1.2, (lx, ly, s_grip), O, "descend to the small cup"),
        Waypoint(0.6, (lx, ly, s_grip), C, "grip the small cup"),
        Waypoint(1.4, (lx, ly, S_TRAVEL), C, "lift it out"),
        Waypoint(1.4, (*ASIDE, S_TRAVEL), C, "hold it aside"),
    ]
    right = [
        Waypoint(2.0, (mx, my + PINCH_DY, M_TRAVEL), O, "above the medium cup's wall"),
        Waypoint(1.2, (mx, my + PINCH_DY, m_grip), O, "descend astride the wall"),
        Waypoint(0.6, (mx, my + PINCH_DY, m_grip), C, "pinch the wall"),
        Waypoint(1.2, (mx, my + PINCH_DY, M_TRAVEL), C, "lift the medium cup"),
    ]
    right.append(Waypoint(6.8 - sum(w.duration for w in right), (mx, my + PINCH_DY, M_TRAVEL), C,
                          "wait for the small cup to be out"))
    right += [
        Waypoint(1.6, (lx, ly + PINCH_DY, M_TRAVEL), C, "carry over the large cup"),
        Waypoint(1.5, (lx, ly + PINCH_DY, m_place), C, "lower into the large cup"),
        Waypoint(0.6, (lx, ly + PINCH_DY, m_place), O, "release the medium cup"),
        Waypoint(0.8, (lx, ly + PINCH_DY, M_TRAVEL), O, "up"),
        Waypoint(1.2, PARK["right"], O, "park"),
    ]
    right_done = sum(w.duration for w in right)
    left.append(Waypoint(right_done - sum(w.duration for w in left), (*ASIDE, S_TRAVEL), C,
                         "wait for the medium cup to be in"))
    left += [
        Waypoint(1.5, (lx, ly, S_TRAVEL), C, "over the nested medium cup"),
        Waypoint(1.5, (lx, ly, s_place), C, "lower into the medium cup"),
        Waypoint(0.6, (lx, ly, s_place), O, "release the small cup"),
        Waypoint(0.8, (lx, ly, S_TRAVEL), O, "up"),
        Waypoint(1.2, PARK["left"], O, "park"),
    ]
    return {"left": left, "right": right}
