"""On the station table, between the arms and a little ahead of them: a closed light-blue vaccine cooler (an open box
16 x 20 cm, 9 cm tall, with a loose lift-off lid whose skirt hangs over the walls and a dark handle bar on top). Inside
stand two vaccine cartons (3.5 x 4.5 x 7 cm): an orange one on the robot's right and a white one on its left. A green
tray (12 x 12 cm) lies on the table in front of the right arm.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import Shape, UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 2.0

INSTRUCTION = "Take the orange vaccine carton out of the cooler and put it on the green tray."

COOLER_BLUE = (0.30, 0.52, 0.78)
GREEN = (0.20, 0.62, 0.30)
ORANGE = (0.92, 0.45, 0.10)
WHITE = (0.92, 0.92, 0.90)


# -- objects --------------------------------------------------------------------------------------------------------
def _fixed(name, lo, hi, color):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Shape(name, pos=tuple(float(v) for v in (lo + hi) / 2), size=tuple(float(v) for v in hi - lo), color=color)


def _cooler():
    cx, cy = g.COOLER_XY
    hx, hy = g.COOLER_HALF
    w, t0, rim = g.WALL, g.TABLE, g.RIM
    return [
        _fixed("cooler_floor", (cx - hx, cy - hy, t0), (cx + hx, cy + hy, t0 + w), COOLER_BLUE),
        _fixed("cooler_wall_front", (cx + hx - w, cy - hy, t0 + w), (cx + hx, cy + hy, rim), COOLER_BLUE),
        _fixed("cooler_wall_back", (cx - hx, cy - hy, t0 + w), (cx - hx + w, cy + hy, rim), COOLER_BLUE),
        _fixed("cooler_wall_left", (cx - hx + w, cy + hy - w, t0 + w), (cx + hx - w, cy + hy, rim), COOLER_BLUE),
        _fixed("cooler_wall_right", (cx - hx + w, cy - hy, t0 + w), (cx + hx - w, cy - hy + w, rim), COOLER_BLUE),
    ]


LID = UsdBody("lid", usd="assets/lid.usda", pos=(float(g.COOLER_XY[0]), float(g.COOLER_XY[1]), float(g.RIM)),
              size=(2 * g.LID_HALF[0], 2 * g.LID_HALF[1], 0.067), mass=g.LID_MASS, friction=(0.8, 0.7),
              torsional_patch=0.01)


def _carton(name, color):
    x, y = g.CARTON_XY[name]
    return Box(name, pos=(float(x), float(y), float(g.FLOOR_TOP + g.CARTON[2] / 2)), size=tuple(g.CARTON),
               mass=g.CARTON_MASS, color=color)


CARTONS = {"orange_carton": _carton("orange_carton", ORANGE), "white_carton": _carton("white_carton", WHITE)}
TRAY = _fixed("tray", (g.TRAY_XY[0] - g.TRAY[0] / 2, g.TRAY_XY[1] - g.TRAY[1] / 2, g.TABLE),
              (g.TRAY_XY[0] + g.TRAY[0] / 2, g.TRAY_XY[1] + g.TRAY[1] / 2, g.TRAY_TOP), GREEN)
OBJECTS = [*_cooler(), LID, *CARTONS.values(), TRAY]

# -- monitor and scoring --------------------------------------------------------------------------------------------
HOLD_R = 0.03            # m: a gripper holds a carton (or the lid's handle) when its grasp point is this close to it
HOLD_CARTON = 0.06       # m: ... and its fingers are less than this far apart (3.5 cm across a carton)
HOLD_HANDLE = 0.05       # m: ... (2.5 cm across the handle)

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(violation=False, why=[], seated=True, delivered=False, out=False, white_in=True)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def _held(grips, point, width) -> bool:
    return any(np.linalg.norm(np.asarray(tcp) - point) < HOLD_R and opening < width for tcp, opening in grips)


def update(states, grips) -> None:
    """One control step's bookkeeping. ``states`` maps "lid", "orange_carton" and "white_carton" to objects with
    ``pos``, ``quat_xyzw`` and ``lin_vel``; ``grips`` lists each arm's (grasp point, finger opening in m)."""
    lid, orange, white = states["lid"], states["orange_carton"], states["white_carton"]
    for name, s in states.items():
        if g.off_table(s.pos):
            _flag(f"{name} fell off the table")
    white_in = g.in_cooler(white.pos)
    if not white_in:
        _flag("the white carton was taken out of the cooler")
    lid_held = _held(grips, g.handle_centre(lid.pos, lid.quat_xyzw), HOLD_HANDLE)
    orange_held = _held(grips, np.asarray(orange.pos), HOLD_CARTON)
    still = float(np.linalg.norm(orange.lin_vel)) < 0.01
    _STATE.update(seated=bool(g.lid_seated(lid.pos, lid.quat_xyzw) and not lid_held),
                  delivered=bool(g.on_tray(orange.pos) and not orange_held and still),
                  out=not g.in_cooler(orange.pos), white_in=white_in,
                  lid_pos=np.asarray(lid.pos, float).copy(), lid_quat=np.asarray(lid.quat_xyzw, float).copy())


@on_step
def _monitor(environment) -> None:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.scene import object_states

    from scenekit import arm_joints

    grips = []
    for side in ("left", "right"):
        tcp, _ = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        grips.append((tcp, float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())))
    update(object_states(environment, ["lid", *CARTONS]), grips)


def success(objects) -> bool:
    """The orange carton rests on the tray, let go; the lid rests seated on the cooler, let go; the white carton is
    still inside; no violation so far. Held for 2 s."""
    return bool(not _STATE["violation"] and _STATE["delivered"] and _STATE["seated"] and _STATE["white_in"])


def score(objects) -> float:
    """0 after a violation (the white carton taken out of the cooler, or anything off the table). Otherwise, at the
    end: 1.0 orange carton on the tray and lid back on; 0.5 orange carton on the tray, lid not back on; 0.25 lid on and
    the orange carton out of the cooler but not on the tray; 0 otherwise (including doing nothing)."""
    if _STATE["violation"]:
        return 0.0
    if _STATE["delivered"]:
        return 1.0 if _STATE["seated"] else 0.5
    return 0.25 if (_STATE["seated"] and _STATE["out"]) else 0.0
