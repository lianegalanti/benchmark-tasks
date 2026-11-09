"""On the station table, a conveyor line runs across the table from the robot's left to its right, 32 cm in front of
the arm bases: a black track bed (10 cm wide) with three grey carrier plates (10 x 10 cm) riding on it, 24 cm apart,
at 2 cm/s. Each carrier holds one bottle (a white plastic cylinder 6 cm across and 13 cm tall with a red cap band at
its top end). The two in front stand upright; the last one has fallen over and lies on its side along the line, cap
pointing downstream. Downstream, 18 cm right of the midpoint between the arms, the line enters a pair of steel guide
rails (9 cm apart) that lead to the next machine's star wheel; red tape marks the guide's mouth across the table.

The robot must stand the fallen bottle back upright, cap up, on its own carrier before that carrier reaches the guide,
and let go of it, without disturbing the upright bottles.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import Bottle, DrivenBox, belt_run, on_reset, on_step  # noqa: E402
from yam_arena.station import TABLE_CENTER, TABLE_SIZE  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the score is decided by the monitor below at the moment the fallen bottle's carrier reaches the guide's
# mouth, and kept from then on (score() returns that decision); cleared at every episode start.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0
INSTRUCTION = ("One bottle has fallen over on the conveyor. Stand it back up, cap up, on its own carrier before it "
               "reaches the guide rails, and let go of it, without disturbing the upright bottles.")
TABLE_RIGHT_Y = TABLE_CENTER[1] - TABLE_SIZE[1] / 2  # -0.65: the table's right edge (y is to the robot's left)


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the conveyor line ---------------------------------------------------------------------------------------------
LINE_X = 0.32  # the line's centre, metres in front of the arm bases
BED = (0.10, 1.28, 0.015)  # the fixed track bed: width (x), length (y), height; it ends 1 cm inside each table edge
CARRIER = (0.10, 0.10, 0.015)
CARRIER_Z = TOP + BED[2] + 0.002 + CARRIER[2] / 2  # 2 mm above the bed: carriers ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
SPEED = 0.02  # m/s, toward -y (the robot's right)
RAMP_S = 0.5  # the line speeds up from rest over this long at the start of the trial
SPACING = 0.24  # between carriers along the line
TURN_Y = TABLE_RIGHT_Y - CARRIER[1] / 2 - 0.02  # -0.72: past the table's edge, carriers turn down out of the way
DROP, DROP_SPEED = 0.30, 0.15
BED_BOX = Box("track_bed", pos=(LINE_X, 0.0, TOP + BED[2] / 2), size=BED, color=(0.08, 0.08, 0.09))

# -- the bottles ---------------------------------------------------------------------------------------------------
R = 0.03  # bottle radius
H = 0.13  # bottle height
MASS = 0.35  # kg: a filled and capped 330 ml HDPE bottle
WHITE, CAP_RED = (0.92, 0.92, 0.90), (0.80, 0.10, 0.10)
FALLEN = "bottle_fallen"
# from the front of the line (downstream, -y) back: two upright, then the fallen one
NAMES = ("bottle_a", "bottle_b", FALLEN)
UPRIGHT = tuple(n for n in NAMES if n != FALLEN)
FALLEN_Y0 = 0.35  # the fallen bottle's carrier centre at the start
STARTS = {n: FALLEN_Y0 + (i - 2) * SPACING for i, n in enumerate(NAMES)}  # -0.13, 0.11, 0.35
CARRIER_OF = {n: f"carrier_{n.split('_')[1]}" for n in NAMES}
PATHS = {n: belt_run((LINE_X, y, CARRIER_Z), SPEED, RAMP_S, TURN_Y, DROP, DROP_SPEED, direction=-1)
         for n, y in STARTS.items()}
CARRIERS = [DrivenBox(CARRIER_OF[n], pos=_floats((LINE_X, STARTS[n], CARRIER_Z)), size=CARRIER,
                      color=(0.5, 0.52, 0.55), path=PATHS[n]) for n in NAMES]
LYING_XYZW = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # +90 deg about x: the cap end points -y
BOTTLES = [
    Bottle(n, pos=_floats((LINE_X, STARTS[n], CARRIER_TOP + (R if n == FALLEN else H / 2) + 0.001)), radius=R,
           height=H, mass=MASS, color=WHITE, cap_color=CAP_RED,
           rot_xyzw=LYING_XYZW if n == FALLEN else (0.0, 0.0, 0.0, 1.0))
    for n in NAMES
]


def carrier_y(name: str, t: float) -> float:
    """Where the carrier of bottle ``name`` is along the line (its centre's y) at episode time ``t``."""
    return float(PATHS[name](t)[0][1])


def arrival(name: str, y: float) -> float:
    """When the carrier of bottle ``name`` reaches ``y`` (after the ramp; the line runs at constant speed then)."""
    return float((STARTS[name] - y) / SPEED + RAMP_S / 2)


# -- the guide (steel rails on posts) and the red tape at its mouth --------------------------------------------------
GUIDE_Y = -0.18  # the guide's mouth: the fallen bottle must be upright on its carrier when the carrier gets here
GUIDE_END_Y = -0.50
GUIDE_GAP = 0.09  # between the rails' inner faces: 1.5 cm each side of an upright bottle
RAIL = (0.01, GUIDE_Y - GUIDE_END_Y, 0.06)  # thickness (x), length (y), height (z)
RAIL_Z = CARRIER_TOP + 0.03 + RAIL[2] / 2  # the rails run 3 to 9 cm above the carriers' tops
POST = 0.012


def _guide() -> list[Box]:
    out = []
    y_mid = (GUIDE_Y + GUIDE_END_Y) / 2
    for side, sign in (("near", -1), ("far", 1)):
        x = LINE_X + sign * (GUIDE_GAP / 2 + RAIL[0] / 2)
        out.append(Box(f"guide_rail_{side}", pos=_floats((x, y_mid, RAIL_Z)), size=RAIL, color=(0.70, 0.72, 0.75)))
        post_x = LINE_X + sign * (BED[0] / 2 + 0.01 + POST / 2)  # beside the bed, holding the rail from outside
        post_h = RAIL_Z + RAIL[2] / 2 - TOP
        for k, y in enumerate((GUIDE_Y - 0.03, GUIDE_END_Y + 0.03)):
            out.append(Box(f"guide_post_{side}_{k}", pos=_floats((post_x, y, TOP + post_h / 2)),
                           size=(POST, POST, post_h), color=(0.55, 0.57, 0.60)))
        # a bracket from the post's top to the rail
        bx = (x + post_x) / 2
        bw = abs(post_x - x) + POST
        for k, y in enumerate((GUIDE_Y - 0.03, GUIDE_END_Y + 0.03)):
            out.append(Box(f"guide_bracket_{side}_{k}", pos=_floats((bx, y, RAIL_Z + RAIL[2] / 2 - 0.005)),
                           size=(bw, POST, 0.01), color=(0.55, 0.57, 0.60)))
    return out


TAPE = (0.06, 0.012, 0.002)
TAPES = [
    Box(f"guide_tape_{side}", pos=_floats((LINE_X + sign * (BED[0] / 2 + 0.03 + TAPE[0] / 2), GUIDE_Y,
                                           TOP + TAPE[2] / 2)), size=TAPE, color=(0.9, 0.05, 0.05))
    for side, sign in (("near", -1), ("far", 1))
]
GUIDE = _guide()

OBJECTS = [BED_BOX, *GUIDE, *TAPES, *CARRIERS, *BOTTLES]

# -- thresholds ------------------------------------------------------------------------------------------------------
UPRIGHT_DEG = 10.0  # the fallen bottle stands upright: its axis within this of vertical, cap up
CENTRE_X, CENTRE_Y = 0.015, 0.025  # ... centred on its carrier: its centre within these of the carrier's, across
#                                     the line (what the guide's 1.5 cm side clearance allows) and along it
SEAT_M = 0.01  # ... and its underside within this of the carrier's top (or, off the line, of the table top)
STILL_MS = 0.05  # ... and moving under this, relative to the carrier (or the table)
LET_GO_M = 0.08  # let go: both grippers' grasp points at least this far from its centre
LIFTED_M = 0.02  # off the line: its lowest point this far above the carriers' tops, or wholly beside the bed
DISTURB_DEG = 20.0  # an upright bottle is disturbed once it tilts more than this,
DISTURB_SLIDE_M = 0.03  # or its centre moves this far (horizontally) from its carrier's centre,
DISTURB_RAISE_M = 0.02  # or its underside rises this far above its carrier's top
PENALTY = 0.5  # taken off the score for each upright bottle disturbed

# -- the monitor ---------------------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(disturbed={}, frozen=set(), decided=None, level=0.0, tcps=None, why="")


reset_state()


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_deg(b) -> float:
    """Degrees between the bottle's axis (base to cap) and straight up: 0 upright, 90 lying, 180 upside down."""
    return float(math.degrees(math.acos(float(np.clip(rot(b.quat_xyzw)[2, 2], -1.0, 1.0)))))


def _horizontal(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def disturbed(b, carrier) -> str:
    """Why an upright bottle counts as disturbed, or '' if it doesn't."""
    if tilt_deg(b) > DISTURB_DEG:
        return "tilted"
    if _horizontal(b.pos, carrier.pos) > DISTURB_SLIDE_M:
        return "moved on or off its carrier"
    if b.low[2] - carrier.high[2] > DISTURB_RAISE_M:
        return "raised off its carrier"
    return ""


def _let_go(b) -> bool:
    tcps = _STATE["tcps"]
    return tcps is not None and all(np.linalg.norm(np.asarray(p) - b.pos) >= LET_GO_M for p in tcps)


def _over_bed(b) -> bool:
    """Some part of the bottle is over the track bed's strip."""
    return bool(b.low[0] < LINE_X + BED[0] / 2 and b.high[0] > LINE_X - BED[0] / 2)


def level(objects) -> tuple[float, str]:
    """What the fallen bottle's state is worth now, before the penalty for disturbed bottles."""
    b, c = objects[FALLEN], objects[CARRIER_OF[FALLEN]]
    upright = tilt_deg(b) <= UPRIGHT_DEG
    dx, dy = abs(b.pos[0] - c.pos[0]), abs(b.pos[1] - c.pos[1])
    on_carrier = dx <= CARRIER[0] / 2 and dy <= CARRIER[1] / 2 and abs(b.low[2] - c.high[2]) <= SEAT_M
    centred = dx <= CENTRE_X and dy <= CENTRE_Y
    with_carrier = np.linalg.norm(np.asarray(b.lin_vel) - np.asarray(c.lin_vel)) < STILL_MS
    let_go = _let_go(b)
    on_table = b.pos[2] > TOP
    off_line = on_table and (not _over_bed(b) or b.low[2] >= CARRIER_TOP + LIFTED_M)
    if upright and on_carrier and centred and with_carrier and let_go:
        return 1.0, "upright and centred on its carrier, let go"
    if upright and on_carrier:
        return 0.5, "upright on its carrier, but still held or off-centre"
    if upright and let_go and off_line and abs(b.low[2] - TOP) <= SEAT_M and np.linalg.norm(b.lin_vel) < STILL_MS:
        return 0.5, "upright and let go, off the line"
    if off_line:
        return 0.25, "off the line, not standing upright"
    return 0.0, "on the line, not upright on its carrier" if on_table else "off the table"


def update(objects, tcps) -> None:
    """One control step: record disturbed upright bottles (each until its own carrier reaches the guide's mouth), the
    grippers' grasp points, and, once the fallen bottle's carrier reaches the guide's mouth, the decision."""
    _STATE["tcps"] = None if tcps is None else [np.asarray(p, float) for p in tcps]
    if _STATE["decided"] is not None:
        return
    for n in UPRIGHT:
        if n in _STATE["frozen"] or n in _STATE["disturbed"]:
            continue
        carrier = objects[CARRIER_OF[n]]
        why = disturbed(objects[n], carrier)
        if why:
            _STATE["disturbed"][n] = why
        if carrier.pos[1] <= GUIDE_Y:
            _STATE["frozen"].add(n)
    lv, why = level(objects)
    _STATE.update(level=lv, why=why)
    if objects[CARRIER_OF[FALLEN]].pos[1] <= GUIDE_Y:
        _STATE["decided"] = _penalised(lv)


def _penalised(lv: float) -> float:
    return float(max(0.0, lv - PENALTY * len(_STATE["disturbed"])))


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.scene import object_states

    tcps = [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0] for side in ("left", "right")]
    names = [*NAMES, *(CARRIER_OF[n] for n in NAMES)]
    update(object_states(environment, names), tcps)


# -- scoring -------------------------------------------------------------------------------------------------------


def success(objects):
    """Decided at the guide's mouth: the fallen bottle stood upright (within 10 degrees, cap up) on its own carrier,
    centre within 1.5 cm of the carrier's centre across the line and 2.5 cm along it, riding with it, both grasp points 8 cm or more from its centre, and no
    upright bottle disturbed."""
    return bool(_STATE["decided"] is not None and _STATE["decided"] >= 1.0)


def score(objects):
    """The decision taken when the fallen bottle's carrier reached the guide's mouth (before then, what the state is
    worth now): 1.0 upright and centred on its carrier and let go; 0.5 upright on its carrier but still held or
    off-centre, or upright and let go on the table off the line; 0.25 off the line but not upright; 0 still on the line not upright (or off the table). Minus 0.5
    for each upright bottle disturbed (tilted over 20 degrees, moved 3 cm on its carrier or raised 2 cm off it), not
    below 0."""
    if _STATE["decided"] is not None:
        return _STATE["decided"]
    return _penalised(_STATE["level"])


def distance(objects):
    """Metres from the fallen bottle's centre to where it would stand upright on its carrier."""
    c = objects[CARRIER_OF[FALLEN]]
    goal = np.array([c.pos[0], c.pos[1], c.high[2] + H / 2])
    return float(np.linalg.norm(objects[FALLEN].pos - goal))
