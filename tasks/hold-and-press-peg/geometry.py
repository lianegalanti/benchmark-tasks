"""Dimensions and layout of the hold-and-press-peg task, in the station frame (x forward, y left, z up, metres).
Pure Python: task.py, demo.py, make_assets.py and the tests read it.

The bar lies along y on a low-friction sheet, its open end (the mouth of a square blind hole) facing the left arm.
Its USD origin is the centre of the mouth, so in the bar's own frame the hole runs from y = 0 to y = -HOLE_DEPTH
along -y, centred on x = 0, z = 0.
"""

from __future__ import annotations

import math

from yam_arena.task_file import TABLE_TOP

TOP = TABLE_TOP["station"]

# -- the low-friction sheet (PTFE, taped to the table) ------------------------------------------------------------
SHEET_SIZE = (0.20, 0.30, 0.003)
SHEET_XY = (0.30, -0.035)
SHEET_TOP = TOP + SHEET_SIZE[2]
SHEET_FRICTION = (0.08, 0.06)  # static, dynamic: PTFE on PTFE, combined as the minimum so it rules the bar's contact

# -- the bar (3D-printed, solid PLA) with its blind hole ---------------------------------------------------------
HOLE = 0.019  # square hole, side
WALL = 0.006  # wall around the hole
BAR_W = HOLE + 2 * WALL  # 31 mm across (x) and tall (z)
BAR_LEN = 0.170  # along y
HOLE_DEPTH = 0.070
PLA_KG_M3 = 1240.0
BAR_MASS = round(PLA_KG_M3 * (BAR_W * BAR_W * BAR_LEN - HOLE * HOLE * HOLE_DEPTH), 3)
AXIS_Z = SHEET_TOP + BAR_W / 2  # the hole's axis, with the bar lying on the sheet
MOUTH = (0.30, 0.05, AXIS_Z)  # centre of the hole's mouth (the bar's origin)
FAR_END = (MOUTH[0], MOUTH[1] - BAR_LEN, AXIS_Z)
BAR_FRICTION = (1.0, 0.9)

# -- the peg (a ground steel dowel pin, lying on the bare table in front of the left arm) ------------------------
PEG_D = 0.016
PEG_R = PEG_D / 2
PEG_LEN = HOLE_DEPTH  # flush with the bar's end when fully in
STEEL_KG_M3 = 7850.0
PEG_MASS = round(STEEL_KG_M3 * math.pi * PEG_R**2 * PEG_LEN, 3)
PEG_START = (0.30, 0.23, TOP + PEG_R + 0.0005)  # centre; axis along y
PEG_FRICTION = (1.0, 0.9)
PEG_ANGULAR_DAMPING = 5.0  # 1/s: stands in for rolling resistance, which PhysX lacks (a lying cylinder rolls freely)

# -- scoring tolerances ---------------------------------------------------------------------------------------------
SEATED_PROUD = 0.002  # fully seated: the peg's outer end at most this far out of the mouth
TIP_IN = 0.005  # the peg's leading end at least this far inside the mouth counts as in the hole
STILL_MOVE = 0.010  # full score: the bar never moved more than this
PARTIAL_MOVE = 0.050  # half score: seated, the bar never moved more than this


def bar_points() -> dict[str, tuple[float, float, float]]:
    """The two points whose displacement measures how far the bar moved: the mouth's centre and the far end's."""
    return {"mouth": MOUTH, "far_end": FAR_END}


def bar_boxes() -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """The bar's colliders as (centre, size) in the bar's frame (origin at the mouth's centre)."""
    h, w, d = HOLE / 2, BAR_W / 2, HOLE_DEPTH
    solid = BAR_LEN - d
    return {
        "floor": ((0.0, -d / 2, -(h + WALL / 2)), (BAR_W, d, WALL)),
        "roof": ((0.0, -d / 2, h + WALL / 2), (BAR_W, d, WALL)),
        "wall_pos_x": ((h + WALL / 2, -d / 2, 0.0), (WALL, d, HOLE)),
        "wall_neg_x": ((-(h + WALL / 2), -d / 2, 0.0), (WALL, d, HOLE)),
        "solid_end": ((0.0, -(d + solid / 2), 0.0), (BAR_W, solid, 2 * w)),
    }
