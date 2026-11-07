"""On the station table, in front of the robot and a little to the right of the midpoint between the arms, a white
PTFE sheet (20 x 30 cm, 3 mm thick) is taped to the table. On it lies an orange 3D-printed bar (31 x 31 mm in
section, 17 cm long, 171 g) along the line from the left arm to the right arm. The bar's left end has a square hole
19 mm across and 7 cm deep, closed at its bottom; the hole's mouth is 30 cm in front of the arm bases and 5 cm to the
left of their midpoint. The bar's underside is low-friction tape, so the bar slides on the sheet when pushed. A steel
dowel pin (16 mm across, 7 cm long, 110 g) lies on the bare table in front of the left arm, 30 cm out, along the same
line as the bar.

The pin has to go all the way into the hole, flush with the bar's end, without the bar moving: pushing the pin in
with one arm only shoves the bar along the sheet, so the other arm must hold the bar still.

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
SCORE_OVER = "best"  # partial credit for the best reached; success (1.0) ends the trial
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Press the steel pin all the way into the hole in the end of the orange bar, until it is flush with "
               "the end. The bar is loose on a slippery sheet and must not move: hold it still with one gripper "
               "while the other presses the pin in.")

SHEET = Shape("sheet", pos=(*g.SHEET_XY, g.TOP + g.SHEET_SIZE[2] / 2), size=g.SHEET_SIZE, color=(0.95, 0.95, 0.93),
              friction=g.SHEET_FRICTION, friction_min=True)
BAR = UsdBody("bar", usd="assets/bar.usda", pos=g.MOUTH, size=(g.BAR_W, g.BAR_LEN, g.BAR_W), mass=g.BAR_MASS,
              friction=g.BAR_FRICTION)
# WHY: the torsional patch lets the pads resist the pin pivoting about the closing axis while it is held by one end.
# WHY: PhysX has no rolling resistance, so a round pin lying on the table rolls away from a numerical nudge (it rolled
# ~10 cm in the first 5 s of the first sim run); angular damping stands in for the rolling resistance of a real table.
PEG = UsdBody("peg", usd="assets/peg.usda", pos=g.PEG_START, size=(g.PEG_D, g.PEG_LEN, g.PEG_D), mass=g.PEG_MASS,
              friction=g.PEG_FRICTION, torsional_patch=0.008, angular_damping=g.PEG_ANGULAR_DAMPING)
OBJECTS = [SHEET, BAR, PEG]

_STATE: dict = {}


# -- geometry of the measured state ----------------------------------------------------------------------------------


def _rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def bar_frame(bar):
    """The bar's origin (the mouth's centre) and rotation, station frame."""
    return np.asarray(bar.pos, float), _rot(bar.quat_xyzw)


def peg_ends_local(objects):
    """The centres of the pin's two ends in the bar's frame, the outer one (nearer the mouth, larger y) first."""
    origin, rot = bar_frame(objects["bar"])
    peg = objects["peg"]
    axis = _rot(peg.quat_xyzw) @ np.array([0.0, 1.0, 0.0])
    ends = [np.asarray(peg.pos, float) + s * axis * g.PEG_LEN / 2 for s in (1.0, -1.0)]
    local = [rot.T @ (e - origin) for e in ends]
    return tuple(sorted(local, key=lambda p: -p[1]))


def _in_section(p) -> bool:
    """A point's centre lies inside the hole's square cross-section."""
    return bool(abs(p[0]) <= g.HOLE / 2 and abs(p[2]) <= g.HOLE / 2)


def tip_in(objects) -> bool:
    """The pin's leading end is at least 5 mm inside the mouth, within the hole's cross-section."""
    _, lead = peg_ends_local(objects)
    return bool(_in_section(lead) and -g.HOLE_DEPTH - 0.005 <= lead[1] <= -g.TIP_IN)


def seated(objects) -> bool:
    """Both ends of the pin inside the hole's cross-section, its outer end at most 2 mm out of the mouth."""
    outer, lead = peg_ends_local(objects)
    return bool(_in_section(outer) and _in_section(lead) and outer[1] <= g.SEATED_PROUD)


def bar_move(objects) -> float:
    """How far the bar is from where it started: the larger displacement of its two end centres (m)."""
    origin, rot = bar_frame(objects["bar"])
    start = _STATE.get("start", g.bar_points())
    now = {"mouth": origin, "far_end": origin + rot @ np.array([0.0, -g.BAR_LEN, 0.0])}
    return float(max(np.linalg.norm(now[k] - np.asarray(start[k], float)) for k in now))


# -- the monitor: the bar's largest displacement over the trial -------------------------------------------------------


def reset_tracking(objects=None) -> None:
    """Start a trial: the bar's start is where it lies now (or its nominal pose)."""
    _STATE.clear()
    if objects is not None:
        origin, rot = bar_frame(objects["bar"])
        _STATE["start"] = {"mouth": origin.copy(), "far_end": origin + rot @ np.array([0.0, -g.BAR_LEN, 0.0])}
    _STATE["max_move"] = 0.0


def track(objects) -> None:
    """After every control step: keep the bar's largest displacement so far."""
    _STATE["max_move"] = max(_STATE.get("max_move", 0.0), bar_move(objects))


reset_tracking()


@on_reset
def _on_reset(environment) -> None:
    from yam_arena.scene import object_states

    reset_tracking(object_states(environment, ["bar", "peg"]))


@on_step
def _on_step(environment) -> None:
    from yam_arena.scene import object_states

    track(object_states(environment, ["bar", "peg"]))


def max_move() -> float:
    return float(_STATE.get("max_move", 0.0))


# -- scoring ----------------------------------------------------------------------------------------------------------


def success(objects) -> bool:
    """The pin fully seated (its outer end at most 2 mm out of the mouth), pin and bar still, and the bar never more
    than 1 cm from where it started at any time in the trial; held 1 s."""
    return bool(seated(objects) and max_move() <= g.STILL_MOVE
                and objects["peg"].resting() and objects["bar"].resting())


def distance(objects) -> float:
    """Metres the pin's outer end is from flush with the mouth (0 when seated)."""
    outer, _ = peg_ends_local(objects)
    return float(np.linalg.norm(outer - np.array([0.0, min(outer[1], 0.0), 0.0])) if _in_section(outer)
                 else np.linalg.norm(outer))


def score(objects) -> float:
    """Partial credit, the best reached in the trial (success scores 1.0): 0.5 while the pin is fully seated and the
    bar has never moved more than 5 cm; 0.25 while the pin's leading end is at least 5 mm inside the hole."""
    if seated(objects) and max_move() <= g.PARTIAL_MOVE:
        return 0.5
    if tip_in(objects):
        return 0.25
    return 0.0
