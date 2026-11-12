"""Geometry of the hang-on-passing-hook scene: the overhead hook conveyor, the two hanger parts, and the tests the
score uses (is a hook's bar through a part's eye, how low is the part). Pure NumPy, so the task file and the asset
script can import it before Isaac Sim starts.

Station frame: x forward (away from the arms), y left, z up, metres. The hooks travel toward -y (from the robot's
left to its right) at a constant speed and never stop.

A hook (seen from the side, robot on the left):

          rail ======
                  |   <- stem, hanging from a carrier on the overhead rail
                  |
    tip ->  |     |
            |_____|   <- bar, pointing back toward the robot; a part hangs on it by its eye

A part (seen from the robot, facing +x), in its own frame: origin at the centre of the eye's opening, x through the
opening (the bar goes along x through it), y across, z up:

            [ ]       <- grip tab
         +-------+    <- top bar: rests on the hook's bar
         |       |
         |  eye  |    <- opening 6.0 cm wide, 6.5 cm tall
         |       |
       +-----------+
       |   block   |  <- the body, 8.4 x 3 x 3 cm
       +-----------+
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # yam_arena.task_file.TABLE_TOP["station"]
TABLE_EDGE_Y = 0.65  # the station table spans y -0.65..0.65 (yam_arena.station TABLE_CENTER, TABLE_SIZE)

# -- the conveyor ------------------------------------------------------------------------------------------------------
SPEED = 0.05  # m/s toward -y, constant
SPACING = 0.30  # between hooks along the rail
N_HOOKS = 3
Y0 = 0.70  # hook 0's bar at t = 0: just past the table's left edge; hooks 1 and 2 follow 30 and 60 cm behind
EXIT_Y = -0.60  # the red exit mark: each hook is checked as its bar passes it (5 cm inside the table's right edge)

BAR = 0.012  # the hook's square steel section (bar, tip and stem)
GAP = 0.001  # between the boxes of one hook, so they never touch each other
BAR_TOP = TOP + 0.12  # the bar's upper face: what a hung part's eye rests on
BAR_Z = BAR_TOP - BAR / 2
TIP_RISE = 0.015  # the tip stands this far above the bar's top
TIP_X = 0.235  # centre of the upturned tip (the hook's end nearest the robot)
BAR_X0 = TIP_X + BAR / 2 + GAP  # the bar's near end
BAR_X1 = 0.470  # its far end, at the stem
BAR_LEN = BAR_X1 - BAR_X0
BAR_XC = (BAR_X0 + BAR_X1) / 2
TIP_Z0, TIP_Z1 = BAR_Z - BAR / 2, BAR_TOP + TIP_RISE
STEM_X = BAR_X1 + GAP + BAR / 2
RAIL_BOTTOM = TOP + 0.405
STEM_Z0, STEM_Z1 = BAR_Z - BAR / 2, RAIL_BOTTOM - 0.005  # 5 mm under the rail: the rail is not touched in the sim
RAIL = (0.04, 3.2, 0.03)  # x, y, z
RAIL_CENTRE = (STEM_X, -0.20, RAIL_BOTTOM + RAIL[2] / 2)  # y -1.80..1.40: under every hook for the whole run


def hook_y(i: int, t: float) -> float:
    """Where hook ``i``'s bar is along the rail (y) at episode time ``t``."""
    return Y0 + i * SPACING - SPEED * t


def hook_boxes(i: int, t: float = 0.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Hook ``i``'s three boxes at time ``t``: name -> (centre, size)."""
    y = hook_y(i, t)
    return {
        "bar": (np.array([BAR_XC, y, BAR_Z]), np.array([BAR_LEN, BAR, BAR])),
        "tip": (np.array([TIP_X, y, (TIP_Z0 + TIP_Z1) / 2]), np.array([BAR, BAR, TIP_Z1 - TIP_Z0])),
        "stem": (np.array([STEM_X, y, (STEM_Z0 + STEM_Z1) / 2]), np.array([BAR, BAR, STEM_Z1 - STEM_Z0])),
    }


def exit_time(i: int) -> float:
    """When hook ``i``'s bar passes the exit mark."""
    return (Y0 + i * SPACING - EXIT_Y) / SPEED


# -- the part (part frame) ---------------------------------------------------------------------------------------------
W_IN, H_IN = 0.060, 0.065  # the eye's opening: width (y), height (z)
EYE_T = 0.012  # thickness of the eye's top bar (z) and side posts (y)
EYE_D = 0.015  # the eye's depth along x
OUT_W = W_IN + 2 * EYE_T  # 8.4 cm
BLOCK = (0.030, OUT_W, 0.030)
TAB = (0.020, 0.012, 0.025)  # the grip tab on top: 1.2 cm thick across y, where the fingers close
MASS = 0.15

_BOXES = {  # name -> (centre, size) in the part frame
    "top_bar": ((0.0, 0.0, H_IN / 2 + EYE_T / 2), (EYE_D, OUT_W, EYE_T)),
    "post_left": ((0.0, W_IN / 2 + EYE_T / 2, 0.0), (EYE_D, EYE_T, H_IN)),
    "post_right": ((0.0, -(W_IN / 2 + EYE_T / 2), 0.0), (EYE_D, EYE_T, H_IN)),
    "block": ((0.0, 0.0, -H_IN / 2 - BLOCK[2] / 2), BLOCK),
    "tab": ((0.0, 0.0, H_IN / 2 + EYE_T + TAB[2] / 2), TAB),
}
PART_BOXES = {k: (np.array(c, float), np.array(s, float)) for k, (c, s) in _BOXES.items()}
PART_BOTTOM = -H_IN / 2 - BLOCK[2]  # lowest point, part frame
TAB_TOP = H_IN / 2 + EYE_T + TAB[2]
TAB_CENTRE = np.array([0.0, 0.0, H_IN / 2 + EYE_T + TAB[2] / 2])
HEIGHT = TAB_TOP - PART_BOTTOM  # 13.2 cm
# centre of mass, for a uniform material (the USD file sets it explicitly)
_VOL = {k: float(np.prod(s)) for k, (_, s) in PART_BOXES.items()}
COM = sum(_VOL[k] * PART_BOXES[k][0] for k in PART_BOXES) / sum(_VOL.values())

PARTS = ("part_a", "part_b")
PART_XY = {"part_a": (0.15, 0.40), "part_b": (0.15, -0.40)}  # standing on the table, in front of each arm


def part_start(name: str) -> np.ndarray:
    """A part's origin standing upright on the table (its eye facing the robot)."""
    x, y = PART_XY[name]
    return np.array([x, y, TOP - PART_BOTTOM])


# -- hanging ----------------------------------------------------------------------------------------------------------
HUNG_Z = BAR_TOP - H_IN / 2  # the part's origin when its eye's top bar rests on a hook's bar, upright
THREAD_Z = (TIP_Z0 + TIP_Z1) / 2  # the origin height at which the bar and the tip both pass through the opening
HUNG_X = 0.285  # where along the bar the demo hangs a part (5 cm past the tip)


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def corners(centre, size) -> np.ndarray:
    c, h = np.asarray(centre, float), np.asarray(size, float) / 2
    return np.array([c + h * s for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)])


def part_points(pos, R, step: float = 0.005) -> np.ndarray:
    """Points on the surfaces of the part's boxes in the world (a grid of ``step`` spacing): for clearance checks."""
    out = []
    for c, s in PART_BOXES.values():
        out.append(box_surface(c, s, step))
    return np.concatenate(out) @ R.T + np.asarray(pos, float)


def box_surface(centre, size, step: float = 0.005) -> np.ndarray:
    c, s = np.asarray(centre, float), np.asarray(size, float)
    axes = [np.linspace(-v / 2, v / 2, max(2, math.ceil(v / step) + 1)) for v in s]
    g = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    on = np.any(np.isclose(np.abs(g), s / 2), axis=1)
    return g[on] + c


def lowest_z(pos, R) -> float:
    """The part's lowest point (world z)."""
    return float(min((corners(c, s) @ R.T)[:, 2].min() for c, s in PART_BOXES.values()) + pos[2])


def _bar_crossing(pos, R, hook_bar_y: float):
    """Where a hook's bar centre line (from the bar's near end to the stem), at ``hook_bar_y`` along the rail, crosses
    the plane of the part's eye, in the part frame; None if it doesn't cross it."""
    a = np.array([BAR_X0, hook_bar_y, BAR_Z])
    b = np.array([BAR_X1, hook_bar_y, BAR_Z])
    pa, pb = R.T @ (a - pos), R.T @ (b - pos)  # into the part frame
    if pa[0] * pb[0] > 0 or pa[0] == pb[0]:
        return None  # the bar doesn't cross the eye's plane
    return pa + (pb - pa) * (pa[0] / (pa[0] - pb[0]))


def bar_through_eye(pos, R, hook_bar_y: float) -> bool:
    """Whether a hook's bar, at ``hook_bar_y`` along the rail, passes through the part's eye: its centre line (from the
    bar's near end to the stem) crosses the plane of the eye inside the opening."""
    p = _bar_crossing(pos, R, hook_bar_y)
    return p is not None and bool(abs(p[1]) < W_IN / 2 and abs(p[2]) < H_IN / 2)


def rest_gap(pos, R, hook_bar_y: float) -> float | None:
    """With the hook's bar through the part's eye: the gap (m) between the bar's upper face and the underside of the
    eye's top bar, where the bar crosses the eye (0 when the part rests on the bar, negative if they overlap). None
    if the bar is not through the eye."""
    if not bar_through_eye(pos, R, hook_bar_y):
        return None
    p = _bar_crossing(pos, R, hook_bar_y)
    return float(H_IN / 2 - (p[2] + BAR / 2))


def tab_centre(pos, R) -> np.ndarray:
    return np.asarray(pos, float) + R @ TAB_CENTRE
