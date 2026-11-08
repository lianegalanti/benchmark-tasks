"""Dimensions, layout and the geometric tests the scorer uses for the linked S-hooks task. NumPy only, no Isaac Sim.

Station frame: x forward from the arms, y to the robot's left, z up, metres.

Each S-hook's own frame has its origin at the hook's centre (its centre of mass: the hook is point-symmetric about it),
the hook lying in its x-z plane: the top curl is centred at (0, 0, +H) and its free tip points to +x, the bottom curl
is centred at (0, 0, -H) and its free tip points to -x, and the straight middle stroke runs from the top curl's -x side
diagonally down to the bottom curl's +x side. Both curl centres lie on the hook's z axis, so a hook hanging by either
curl hangs upright, with its middle stroke under the curl.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # yam_arena.task_file.TABLE_TOP["station"]

# -- the S-hooks: 10 mm aluminium rod bent into an S, 15 cm tall ------------------------------------------------------
ROD_R = 0.005                       # 10 mm rod
CURL_R = 0.028                      # curl radius to the rod's centre line: 4.6 cm clear inside each curl
H = 0.042                           # each curl's centre this far above / below the hook's centre
TIP_DEG = 20.0                      # each curl ends 20 degrees past the vertical through its centre (160 degree curls)
N_ARC = 8                           # capsules per curl (20 degrees each; a chord's middle is at the curl's top)
MASS = 0.06                         # kg
HEIGHT = 2 * (H + CURL_R + ROD_R)   # 15.0 cm outside
WIDTH = 2 * (CURL_R + ROD_R)        # 6.6 cm outside
INNER = CURL_R - ROD_R              # 2.3 cm: a curl's inside radius
# A curl's top (and bottom) is the middle of a straight 20-degree chord, this far from the curl's centre.
CHORD = CURL_R * math.cos(math.radians(10.0))


def _arc(cz: float, a0: float, a1: float) -> list[np.ndarray]:
    return [np.array([CURL_R * math.cos(a), 0.0, cz + CURL_R * math.sin(a)])
            for a in np.linspace(math.radians(a0), math.radians(a1), N_ARC + 1)]


def hook_polyline() -> np.ndarray:
    """The rod's centre line in the hook frame, as the points between its capsules: the top curl from its tip
    (20 degrees above the curl's centre, at +x) over the top to its -x side, the middle stroke, and the bottom curl
    from its +x side round the bottom to its tip (20 degrees below its centre, at -x)."""
    top = _arc(H, TIP_DEG, 180.0)
    bottom = _arc(-H, 0.0, -180.0 + TIP_DEG)
    return np.array(top + bottom)  # top[-1] -> bottom[0] is the middle stroke


POLY = hook_polyline()
SEGMENTS = [(POLY[i], POLY[i + 1]) for i in range(len(POLY) - 1)]
CURLS = (np.array([0.0, 0.0, H]), np.array([0.0, 0.0, -H]))   # top, bottom curl centres (hook frame)
STROKE = (POLY[N_ARC], POLY[N_ARC + 1])
STROKE_DIR = (STROKE[1] - STROKE[0]) / np.linalg.norm(STROKE[1] - STROKE[0])
# the curl's opening: from its tip's end to the middle stroke, clear of both rods
GAP = float(abs(np.cross(STROKE_DIR, POLY[0] - STROKE[0])[1])) - 2 * ROD_R


def dense(n_per_seg: int = 8) -> np.ndarray:
    """Points along the centre line, about 2 mm apart (hook frame)."""
    pts = [a + (b - a) * s for a, b in SEGMENTS for s in np.linspace(0.0, 1.0, n_per_seg, endpoint=False)]
    return np.array(pts + [POLY[-1]])


DENSE = dense()

# -- the display: an upright back board with a rail on two brackets and a straight peg --------------------------------
BOARD_FACE = 0.40                    # x of the board's front face
BOARD = (0.015, 0.32, 0.40)          # thickness (x), width (y), height (z)
BOARD_POS = (BOARD_FACE + BOARD[0] / 2, 0.07, TOP + BOARD[2] / 2)
SUPPORT_R = 0.006                    # 12 mm chrome rail and peg

RAIL_X, RAIL_Z = 0.30, TOP + 0.325   # the rail runs along y
RAIL_Y = (0.04, 0.20)
BRACKET = (BOARD_FACE - (RAIL_X - 0.008), 0.012, 0.02)   # x, y, z: from the rail to the board
BRACKET_Y = (RAIL_Y[0] + BRACKET[1] / 2, RAIL_Y[1] - BRACKET[1] / 2)

PEG_Y, PEG_Z = -0.02, TOP + 0.26      # the peg points straight at the robot (along -x) from the board
PEG_TIP_X = 0.29
PEG_LEN = BOARD_FACE - PEG_TIP_X

# each support as (name, axis start, axis end)
SUPPORTS = (
    ("rail", np.array([RAIL_X, RAIL_Y[0], RAIL_Z]), np.array([RAIL_X, RAIL_Y[1], RAIL_Z])),
    ("peg", np.array([BOARD_FACE, PEG_Y, PEG_Z]), np.array([PEG_TIP_X, PEG_Y, PEG_Z])),
)

# -- the start: hook A on the rail, hook B hanging from hook A's bottom curl -----------------------------------------
HOOK_Y = 0.12
DROP = 0.001                         # each hook starts 1 mm above what it hangs from (as placed), and drops onto it
# Hook A hangs from the rail by its top curl, in the x-z plane (the rail runs through it): hook frame = station frame.
A_START = np.array([RAIL_X, HOOK_Y, RAIL_Z - (CHORD - ROD_R - SUPPORT_R) - H + DROP])
A_START_XYZW = (0.0, 0.0, 0.0, 1.0)
A_BAR = A_START + np.array([0.0, 0.0, -H - CHORD])     # the lowest point of hook A's bottom curl (centre line)
# Hook B hangs by its top curl from that lowest point, turned a quarter turn about z: it faces the robot (its plane is
# y-z), with its top curl's tip toward +y (the robot's left).
B_START = A_BAR + np.array([0.0, 0.0, -(CHORD - 2 * ROD_R) + DROP - H])
B_START_XYZW = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))

# -- scoring thresholds -----------------------------------------------------------------------------------------------
ON_TABLE = 0.005                     # a hook whose lowest point is this close to the table top has been put down
ENCIRCLE = INNER + 0.002             # a rod passes through a curl when it crosses the curl's plane this close to its centre
APART = 0.01                         # "lifted off": the two hooks' surfaces at least this far apart everywhere
ALONE = 0.003                        # hanging alone: the hooks' surfaces at least this far apart
HOLD_R = 0.03                        # a gripper holds a hook when its grasp point is this close to the rod's centre line
HOLD_OPEN = 0.03                     # ... and its fingers are closer together than this
GRIPPER_CLEAR = 0.04                 # let go: each gripper's centre line this far from every hook's centre line
STILL_SPEED, STILL_SPIN = 0.02, 0.5  # m/s, rad/s


def rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def world(pos, R, pts) -> np.ndarray:
    return np.asarray(pts, float) @ np.asarray(R).T + np.asarray(pos, float)


def lowest_z(pos, R) -> float:
    """Height of the hook's lowest point."""
    return float(world(pos, R, DENSE)[:, 2].min() - ROD_R)


def _pierces(pos, R, a0, a1) -> bool:
    """Whether segment a0-a1 (station frame) passes through either curl of the hook at ``pos``, ``R``: it crosses the
    hook's plane within ENCIRCLE of a curl's centre."""
    p0 =(np.asarray(a0, float) - pos) @ R
    p1 = (np.asarray(a1, float) - pos) @ R
    if p0[1] * p1[1] > 0.0 or p0[1] == p1[1]:
        return False
    s = p0[1] / (p0[1] - p1[1])
    hit = p0 + s * (p1 - p0)
    return any(math.hypot(hit[0] - c[0], hit[2] - c[2]) < ENCIRCLE for c in CURLS)


def on_support(pos, R) -> str | None:
    """The support (``"rail"`` or ``"peg"``) passing through one of the hook's curls, else None."""
    pos = np.asarray(pos, float)
    for name, a0, a1 in SUPPORTS:
        if _pierces(pos, R, a0, a1):
            return name
    return None


def linked(pa, Ra, pb, Rb) -> bool:
    """Whether either hook's rod passes through a curl of the other (they are hooked together)."""
    wa, wb = world(pa, Ra, POLY), world(pb, Rb, POLY)
    pa, pb = np.asarray(pa, float), np.asarray(pb, float)
    return any(_pierces(pb, Rb, wa[i], wa[i + 1]) for i in range(len(wa) - 1)) or \
        any(_pierces(pa, Ra, wb[i], wb[i + 1]) for i in range(len(wb) - 1))


def gap(pa, Ra, pb, Rb) -> float:
    """Shortest distance between the two hooks' surfaces (from their centre lines, sampled every 2 mm)."""
    wa, wb = world(pa, Ra, DENSE), world(pb, Rb, DENSE)
    d = np.linalg.norm(wa[:, None, :] - wb[None, :, :], axis=2).min()
    return float(d - 2 * ROD_R)


def seg_points(a0, a1, n: int = 41) -> np.ndarray:
    s = np.linspace(0.0, 1.0, n)
    return np.asarray(a0, float) + np.outer(s, np.asarray(a1, float) - np.asarray(a0, float))


def dist_to_hook(points, pos, R) -> float:
    """Shortest distance from any of ``points`` to the hook's centre line."""
    w = world(pos, R, DENSE)
    return float(np.linalg.norm(np.asarray(points, float)[:, None, :] - w[None, :, :], axis=2).min())
