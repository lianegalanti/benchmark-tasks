"""Geometry of the hang-on-passing-hook scene: the overhead conveyor with its wire hooks, the two steel strut plates
in their loading block, and the tests the score uses (which hole a hook's leg passes through, does the plate rest on
it by a hole above its centre of mass, how far is it from plumb, how low is it). Pure NumPy, so the task file and the
asset script can import it before Isaac Sim starts.

Station frame: x forward (away from the arms), y left, z up, metres. The hooks travel toward -y (from the robot's
left to its right) at a constant speed and never stop.

A hook, bent from 4 mm steel wire (seen from the side, robot on the left):

          rail ======
                  |   <- stem, hanging from a carrier on the overhead rail
                  |
          tip ->  |
            '-----|   <- leg, pointing back toward the robot, with a 4 mm upturn at its end

The part: a 3-hole flat plate strut fitting (Unistrut P1066 pattern), 1/4 in (6.35 mm) hot-rolled steel,
5-3/8 x 1-5/8 in (136.5 x 41.3 mm), three 9/16 in (14.3 mm) holes on its centre line, 1-7/8 in (47.6 mm) apart and
13/16 in (20.6 mm) from each end. Part frame: origin at the centre of hole 0 (the end hole that is up at the start),
x through the holes (the plate's thickness), y across its width, z along its length (up when standing):

        +-----+   <- end 0
        | [0] |   <- hole 0: a hanging hole, 4.8 cm above the centre of mass
        |     |
        | [1] |   <- hole 1, at the centre of mass: not a hanging hole
        |     |
        | [2] |   <- hole 2: the other hanging hole (the plate hung upside down)
        +-----+
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # yam_arena.task_file.TABLE_TOP["station"]
TABLE_EDGE_Y = 0.65  # the station table spans y -0.65..0.65 (yam_arena.station TABLE_CENTER, TABLE_SIZE)
INCH = 0.0254

# -- the conveyor ------------------------------------------------------------------------------------------------------
SPEED = 0.05  # m/s toward -y, constant
SPACING = 0.30  # between hooks along the rail
N_HOOKS = 3
Y0 = 0.70  # hook 0's leg at t = 0: just past the table's left edge; hooks 1 and 2 follow 30 and 60 cm behind
EXIT_Y = -0.60  # the red exit mark: each hook is checked as its leg passes it (5 cm inside the table's right edge)

WIRE = 0.004  # the hook's wire (modelled square)
GAP = 0.001  # between the boxes of one hook, so they never touch each other
BAR_TOP = TOP + 0.150  # the leg's upper face: what a hung plate's hole rests on
BAR_Z = BAR_TOP - WIRE / 2
TIP_RISE = 0.004  # the upturned end stands this far above the leg's top
TIP_X = 0.255  # centre of the upturned end (the hook's end nearest the robot)
STEM_X = 0.460  # the stem, under the rail
BAR_X0 = TIP_X + WIRE / 2 + GAP  # the leg's near end
BAR_X1 = STEM_X - WIRE / 2 - GAP  # its far end, at the stem
BAR_LEN = BAR_X1 - BAR_X0
BAR_XC = (BAR_X0 + BAR_X1) / 2
TIP_Z0, TIP_Z1 = BAR_Z - WIRE / 2, BAR_TOP + TIP_RISE
RAIL_BOTTOM = TOP + 0.405
STEM_Z0, STEM_Z1 = BAR_Z - WIRE / 2, RAIL_BOTTOM - 0.005  # 5 mm under the rail: the rail is not touched in the sim
RAIL = (0.04, 3.2, 0.03)  # x, y, z
RAIL_CENTRE = (STEM_X, -0.20, RAIL_BOTTOM + RAIL[2] / 2)  # y -1.80..1.40: over every hook for the whole run


def hook_y(i: int, t: float) -> float:
    """Where hook ``i``'s leg is along the rail (y) at episode time ``t``."""
    return Y0 + i * SPACING - SPEED * t


def hook_boxes(i: int, t: float = 0.0) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Hook ``i``'s three boxes at time ``t``: name -> (centre, size). "bar" is the leg a plate hangs on."""
    y = hook_y(i, t)
    return {
        "bar": (np.array([BAR_XC, y, BAR_Z]), np.array([BAR_LEN, WIRE, WIRE])),
        "tip": (np.array([TIP_X, y, (TIP_Z0 + TIP_Z1) / 2]), np.array([WIRE, WIRE, TIP_Z1 - TIP_Z0])),
        "stem": (np.array([STEM_X, y, (STEM_Z0 + STEM_Z1) / 2]), np.array([WIRE, WIRE, STEM_Z1 - STEM_Z0])),
    }


def exit_time(i: int) -> float:
    """When hook ``i``'s leg passes the exit mark."""
    return (Y0 + i * SPACING - EXIT_Y) / SPEED


# -- the part (part frame) ---------------------------------------------------------------------------------------------
THICK = 0.25 * INCH  # 6.35 mm (x)
WIDTH = 1.625 * INCH  # 41.3 mm (y)
LENGTH = 5.375 * INCH  # 136.5 mm (z)
HOLE = 0.5625 * INCH  # 14.3 mm (modelled as a square opening of this size)
PITCH = 1.875 * INCH  # 47.6 mm between hole centres
END = 13 / 16 * INCH  # 20.6 mm from each end to the nearest hole's centre
R_HOLE = HOLE / 2
HOLE_Z = (0.0, -PITCH, -2 * PITCH)  # hole centres along z
HANGING_HOLES = (0, 2)  # the end holes; hole 1 is at the centre of mass
STEEL_DENSITY = 7850.0
# the real plate's mass, with round holes: 0.257 kg
MASS = round(STEEL_DENSITY * THICK * (LENGTH * WIDTH - 3 * math.pi * R_HOLE ** 2), 3)


def _part_boxes() -> dict[str, tuple[tuple, tuple]]:
    side_w = WIDTH / 2 - R_HOLE
    boxes = {"end_0": ((0.0, 0.0, (R_HOLE + END) / 2), (THICK, WIDTH, END - R_HOLE))}
    for k, z in enumerate(HOLE_Z):
        boxes[f"hole{k}_left"] = ((0.0, R_HOLE + side_w / 2, z), (THICK, side_w, HOLE))
        boxes[f"hole{k}_right"] = ((0.0, -(R_HOLE + side_w / 2), z), (THICK, side_w, HOLE))
        if k < 2:
            boxes[f"web_{k}{k + 1}"] = ((0.0, 0.0, z - PITCH / 2), (THICK, WIDTH, PITCH - HOLE))
    boxes["end_2"] = ((0.0, 0.0, HOLE_Z[2] - (R_HOLE + END) / 2), (THICK, WIDTH, END - R_HOLE))
    return boxes


PART_BOXES = {k: (np.array(c, float), np.array(s, float)) for k, (c, s) in _part_boxes().items()}
PART_BOTTOM = HOLE_Z[2] - END  # lowest point, part frame (end 2)
PART_TOP = END  # end 0
HEIGHT = PART_TOP - PART_BOTTOM  # 136.5 mm
COM = np.array([0.0, 0.0, -PITCH])  # the plate is symmetric: its centre of mass is at hole 1

PARTS = ("plate_a", "plate_b")
PART_XY = {"plate_a": (0.15, 0.40), "plate_b": (0.15, -0.40)}  # standing in their loading blocks

# -- the loading blocks: a slot 8.5 x 46 mm, 25 mm deep, each plate standing in one on its end 2 --------------------
SLOT = (0.0085, 0.046)  # x, y: 2.15 mm and 4.7 mm of play around the plate
SLOT_DEPTH = 0.025
WALL = 0.015


def holder_boxes(name: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """The four walls of a plate's loading block, standing on the table: name -> (centre, size)."""
    x, y = PART_XY[name]
    z = TOP + SLOT_DEPTH / 2
    out = {}
    for s, tag in ((1, "far"), (-1, "near")):
        out[tag] = (np.array([x + s * (SLOT[0] + WALL) / 2, y, z]), np.array([WALL, SLOT[1] + 2 * WALL, SLOT_DEPTH]))
    for s, tag in ((1, "left"), (-1, "right")):
        out[tag] = (np.array([x, y + s * (SLOT[1] + WALL) / 2, z]), np.array([SLOT[0], WALL, SLOT_DEPTH]))
    return out


def part_start(name: str) -> np.ndarray:
    """A plate's origin (hole 0's centre) standing upright in its block, end 2 on the table, holes facing the
    robot."""
    x, y = PART_XY[name]
    return np.array([x, y, TOP - PART_BOTTOM])


# -- hanging ----------------------------------------------------------------------------------------------------------
HUNG_Z = BAR_TOP - R_HOLE  # the origin when hole 0's upper edge rests on a hook's leg, upright
THREAD_Z = (TIP_Z0 + TIP_Z1) / 2  # the origin height at which the leg and its upturn both pass through hole 0
HUNG_X = TIP_X + 0.05  # where along the leg the demo hangs a plate (5 cm past the upturn)


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def corners(centre, size) -> np.ndarray:
    c, h = np.asarray(centre, float), np.asarray(size, float) / 2
    return np.array([c + h * s for s in np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)])


def part_points(pos, R, step: float = 0.005) -> np.ndarray:
    """Points on the surfaces of the plate's boxes in the world (a grid of ``step`` spacing): for clearance checks."""
    out = []
    for c, s in PART_BOXES.values():
        out.append(box_surface(c, s, step))
    return np.concatenate(out) @ R.T + np.asarray(pos, float)


def box_surface(centre, size, step: float = 0.005) -> np.ndarray:
    c, s = np.asarray(centre, float), np.asarray(size, float)
    axes = [np.linspace(-v / 2, v / 2, max(2, math.ceil(v / step) + 1)) for v in s]
    grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
    on = np.any(np.isclose(np.abs(grid), s / 2), axis=1)
    return grid[on] + c


def lowest_z(pos, R) -> float:
    """The plate's lowest point (world z)."""
    return float(min((corners(c, s) @ R.T)[:, 2].min() for c, s in PART_BOXES.values()) + pos[2])


def tilt_deg(R) -> float:
    """How far the plate's length is from plumb (degrees), either end up."""
    return math.degrees(math.acos(min(1.0, abs(float(R[2, 2])))))


def _bar_crossing(pos, R, hook_bar_y: float):
    """Where a hook leg's centre line (from its near end to the stem), at ``hook_bar_y`` along the rail, crosses the
    plate's mid-plane, in the part frame; None if it doesn't cross it."""
    a = np.array([BAR_X0, hook_bar_y, BAR_Z])
    b = np.array([BAR_X1, hook_bar_y, BAR_Z])
    pa, pb = R.T @ (a - pos), R.T @ (b - pos)  # into the part frame
    if pa[0] * pb[0] > 0 or pa[0] == pb[0]:
        return None  # the leg doesn't cross the plate's plane
    return pa + (pb - pa) * (pa[0] / (pa[0] - pb[0]))


def hole_on_bar(pos, R, hook_bar_y: float) -> int | None:
    """Which of the plate's holes a hook's leg, at ``hook_bar_y`` along the rail, passes through (its centre line
    crosses the plate's mid-plane inside that hole), or None."""
    p = _bar_crossing(pos, R, hook_bar_y)
    if p is None or abs(p[1]) >= R_HOLE:
        return None
    for k, z in enumerate(HOLE_Z):
        if abs(p[2] - z) < R_HOLE:
            return k
    return None


def bar_through_hole(pos, R, hook_bar_y: float) -> bool:
    return hole_on_bar(pos, R, hook_bar_y) is not None


def hole_above_com(pos, R, k: int) -> float:
    """How far (m, world z) hole ``k``'s centre is above the plate's centre of mass."""
    return float((R @ (np.array([0.0, 0.0, HOLE_Z[k]]) - COM))[2])


def rest_gap(pos, R, hook_bar_y: float) -> float | None:
    """With a hook's leg through one of the plate's end holes, and that hole above the centre of mass: the gap (m)
    between the leg's upper face and the hole's upper edge, where the leg crosses the plate (0 when the plate hangs
    on the leg, negative if they overlap). None if the leg isn't through an end hole that is above the centre of
    mass."""
    k = hole_on_bar(pos, R, hook_bar_y)
    if k not in HANGING_HOLES:
        return None
    up = 1.0 if R[2, 2] >= 0 else -1.0  # the plate's +z up (hole 0 on top) or upside down (hole 2 on top)
    if up * (HOLE_Z[k] - COM[2]) <= 0:
        return None  # that hole is below the centre of mass
    p = _bar_crossing(pos, R, hook_bar_y)
    return float(R_HOLE - (up * (p[2] - HOLE_Z[k]) + WIRE / 2))


def flipped() -> np.ndarray:
    """The plate turned end for end (half a turn about x): hole 2 on top."""
    return np.diag([1.0, -1.0, -1.0])
