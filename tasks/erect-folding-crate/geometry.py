"""Dimensions and kinematics of the folding crate, in plain NumPy (no Isaac Sim).

The crate is a standard 400 x 300 x 174 mm polypropylene folding produce crate (internal 374 x 274 x 166 mm,
0.88 kg). Crate frame: origin at the middle of the base's underside, x across the crate (300 mm, away from the
robot), y along it (400 mm, the robot's left), z up. On the station it stands square to the table, its centre at
(XC, 0) and its underside on the table top.

Parts (all rigid; one articulation whose root is the base):

- base: a 6 mm floor, two side rims (the long sides, 68 mm tall) and two end rims (50 mm tall);
- four walls, each a 10 mm plate hinged along the inner top edge of its rim, folding inward. The short walls (ends,
  y = +-) fold first and lie lowest; the long walls (sides, x = +-) hinge 18 mm higher and lie on top. Joint angle
  q = 0 folded flat, pi/2 upright (the stop);
- each wall has a 120 x 36 mm handhold slot under a round 18 mm top rail;
- four catches: a sprung pawl in each long wall at each end, sliding along the wall's normal. At rest it stands
  6 mm out of the wall's inner face, just inside where the short wall stands when up. Its inboard side is a 45
  degree ramp, so a short wall swung up pushes it in; once the wall is past, it springs out behind the wall's inner
  face and holds the wall up. Joint s = 0 out (rest), negative pushed in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# -- placement (station frame) ------------------------------------------------------------------------------------
TOP = 0.7475             # station table top (yam_arena.task_file.TABLE_TOP["station"])
XC = 0.26                # crate centre, forward of the arm bases
YC = 0.0

# -- catalogue crate --------------------------------------------------------------------------------------------------
LENGTH, WIDTH, HEIGHT = 0.400, 0.300, 0.174
INNER_LENGTH, INNER_WIDTH = 0.374, 0.274
MASS = 0.88

T_WALL = 0.010           # wall plate thickness
FLOOR = 0.006            # floor thickness
SHORT_HINGE_Z = 0.050    # top of the end rims: short walls hinge here
LONG_HINGE_Z = 0.068     # top of the side rims: long walls hinge here, above the folded short walls
IN_X = INNER_WIDTH / 2   # 0.137: inner face of the side rims and of the upright long walls
IN_Y = INNER_LENGTH / 2  # 0.187: inner face of the end rims and of the upright short walls
SHORT_H = HEIGHT - SHORT_HINGE_Z   # 0.124
LONG_H = HEIGHT - LONG_HINGE_Z     # 0.106
SHORT_GAP = 0.007                  # each side, between a short wall's plate and the long walls' inner faces
SHORT_W = INNER_WIDTH - 2 * SHORT_GAP   # 0.260: the plate stays clear of the catches (6 mm out)
HOOK_R = 0.004                     # a hook on each side edge of a short wall: a 8 mm round lug, upright, ...
HOOK_REACH = IN_X - 0.0015         # ... reaching to 1.5 mm from the long wall (0.1355), over 4.5 mm of the pawl
HOOK_LEN = 0.030                   # straight length; with its rounded ends it spans 38 mm of the wall's height
LONG_W = LENGTH - 0.002                 # 0.398

# Handhold: slot and round top rail, centred along the wall.
RAIL_R = 0.009
SLOT_LEN = 0.120
SLOT_H = 0.036

# Catches.
PAWL_OUT = 0.006         # how far a pawl at rest stands out of the long wall's inner face
PAWL_TRAVEL = 0.007      # it can be pushed in this far (1 mm more than it stands out)
PAWL_W = 0.011           # along the wall
PAWL_H = 0.020           # tall
PAWL_GAP = 0.001         # between its square face and an upright short wall's inner face
PAWL_Z = 0.145           # its centre height above the crate's underside
PAWL_RATE = 600.0        # N/m spring, no preload
SQUARE = 0.004           # the square-faced part of the head, along the wall; the rest is the ramp

# Masses (kg); together the catalogue 0.88 kg.
M_LONG, M_SHORT, M_PAWL = 0.13, 0.10, 0.005
M_BASE = MASS - 2 * M_LONG - 2 * M_SHORT - 4 * M_PAWL

PP_FRICTION = 0.35       # polypropylene on the table and on the grippers' pads ("min" combine)

# Colours.
CRATE_GREEN = (0.10, 0.45, 0.25)
CATCH_ORANGE = (1.00, 0.50, 0.05)


def rot_axis(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + s * k + (1 - c) * k @ k


def quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    tr = np.trace(r)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        q = ((r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s)
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        q = ((r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        q = ((r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s)
    q = np.array(q, float)
    q = q if q[0] >= 0 else -q
    return tuple(float(v) for v in q / np.linalg.norm(q))


def quat_wxyz_matrix(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# A box: (centre, size, colour, rotation wxyz) in its body's frame. A capsule: (centre, radius, length of the
# straight part, colour, axis "X" or "Z" of the body).
@dataclass(frozen=True)
class Wall:
    """A wall's body frame at q = 0 (folded): origin on the hinge line at the wall's middle; x along the hinge,
    oriented so raising is +q about it; y the wall's inward normal when upright (it points down when folded);
    z along the wall from the hinge to its free edge (inward when folded, up when upright). The plate occupies
    u in [-w/2, w/2], v (z) in [0, h], and w (y) in [-t, 0]: its outer face is at y = -t."""

    name: str
    inward: tuple[float, float, float]   # horizontal unit vector from the hinge toward the crate's middle
    hinge: tuple[float, float, float]    # a point on the hinge line, crate frame
    height: float
    width: float
    mass: float

    @property
    def axis(self) -> np.ndarray:
        return np.cross(self.inward, (0.0, 0.0, 1.0))

    def frame(self, q: float = 0.0) -> np.ndarray:
        """Rotation of the body frame in the crate frame at joint angle q."""
        a = self.axis
        c = np.asarray(self.inward, float)
        b = np.cross(c, a)
        flat = np.column_stack([a, b, c])
        return rot_axis(a, q) @ flat

    def point(self, local, q: float = 0.0) -> np.ndarray:
        return np.asarray(self.hinge, float) + self.frame(q) @ np.asarray(local, float)

    @property
    def rail_local(self) -> np.ndarray:
        return np.array([0.0, -T_WALL / 2, self.height - RAIL_R])


WALLS = {
    "wall_left": Wall("wall_left", (0.0, -1.0, 0.0), (0.0, IN_Y, SHORT_HINGE_Z), SHORT_H, SHORT_W, M_SHORT),
    "wall_right": Wall("wall_right", (0.0, 1.0, 0.0), (0.0, -IN_Y, SHORT_HINGE_Z), SHORT_H, SHORT_W, M_SHORT),
    "wall_far": Wall("wall_far", (-1.0, 0.0, 0.0), (IN_X, 0.0, LONG_HINGE_Z), LONG_H, LONG_W, M_LONG),
    "wall_near": Wall("wall_near", (1.0, 0.0, 0.0), (-IN_X, 0.0, LONG_HINGE_Z), LONG_H, LONG_W, M_LONG),
}
SHORT = ("wall_left", "wall_right")
LONG = ("wall_far", "wall_near")
UP = math.pi / 2


def _span(x0, x1, y0, y1, z0, z1, color=CRATE_GREEN):
    return (((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color, (1.0, 0.0, 0.0, 0.0))


def base_boxes():
    """The base in the crate frame: floor, side rims (long sides) and end rims (short ends)."""
    L, W = LENGTH / 2, WIDTH / 2
    return [
        _span(-W, W, -L, L, 0.0, FLOOR),
        _span(IN_X, W, -L, L, FLOOR, LONG_HINGE_Z), _span(-W, -IN_X, -L, L, FLOOR, LONG_HINGE_Z),
        _span(-IN_X, IN_X, IN_Y, L, FLOOR, SHORT_HINGE_Z), _span(-IN_X, IN_X, -L, -IN_Y, FLOOR, SHORT_HINGE_Z),
    ]


def wall_shapes(wall: Wall):
    """(boxes, capsules) of a wall in its body frame: the plate with its handhold slot, and the round top rail."""
    hw, h, t = wall.width / 2, wall.height, T_WALL
    s0, s1 = h - 2 * RAIL_R - SLOT_H, h - 2 * RAIL_R      # the slot's bottom and top (the rail's underside)
    sl = SLOT_LEN / 2
    boxes = [
        _span(-hw, hw, -t, 0.0, 0.0, s0),                     # panel below the slot
        _span(-hw, -sl, -t, 0.0, s0, h), _span(sl, hw, -t, 0.0, s0, h),   # either side of the slot, full height
    ]
    caps = [((0.0, -t / 2, h - RAIL_R), RAIL_R, SLOT_LEN, CRATE_GREEN, "X")]
    if wall.name in SHORT:
        # The corner hooks: upright lugs at the pawls' height, flush with the plate's inner face. A lug's round side
        # rides over a pawl's ramp and pushes it in; its inner side then sits 1 mm from the pawl's square face.
        v = PAWL_Z - SHORT_HINGE_Z
        for side in (-1, 1):
            caps.append(((side * (HOOK_REACH - HOOK_R), -HOOK_R, v), HOOK_R, HOOK_LEN, CATCH_ORANGE, "Z"))
    return boxes, caps


@dataclass(frozen=True)
class Pawl:
    """A catch: a pawl in long wall ``wall`` at end ``end`` (+1: the left end, y > 0) holding short wall
    ``holds``. Its body frame has the long wall's orientation; its origin, at rest, is on the wall's inner face."""

    name: str
    wall: str
    end: int
    holds: str

    @property
    def u(self) -> float:
        """Its centre along the long wall's x axis (the crate's +-y)."""
        y = self.end * (IN_Y - PAWL_GAP - PAWL_W / 2)
        return float(y / WALLS[self.wall].axis[1])

    @property
    def out(self) -> float:
        """+1 or -1: the direction along the wall's x axis toward the end (and the short wall)."""
        return float(np.sign(self.u))

    @property
    def rest_local(self) -> np.ndarray:
        """Its origin in the long wall's body frame."""
        return np.array([self.u, 0.0, PAWL_Z - LONG_HINGE_Z])

    def boxes(self):
        """The head in the pawl's body frame: the square part next to the short wall, a 45 degree ramp on the
        inboard side, and a short stem inside the wall."""
        s, p = self.out, PAWL_OUT
        half = PAWL_W / 2
        sq0 = half - SQUARE                                   # where the square part starts (sigma, toward the end)
        out = [
            ((s * (sq0 + half) / 2, p / 2, 0.0), (SQUARE, p, PAWL_H), CATCH_ORANGE, (1.0, 0.0, 0.0, 0.0)),
            ((0.0, -0.0015, 0.0), (PAWL_W, 0.003, PAWL_H), CATCH_ORANGE, (1.0, 0.0, 0.0, 0.0)),   # stem
        ]
        thick = 0.003
        length = p * math.sqrt(2) + 0.002
        mid = np.array([sq0 - p / 2, p / 2])                  # (sigma, w) midpoint of the slope
        normal = np.array([-1.0, 1.0]) / math.sqrt(2)         # toward the incoming short wall
        c = mid - normal * thick / 2
        ang = math.atan2(1.0, s)                              # box x along the slope, (s, 1)/sqrt(2) in (u, w)
        out.append(((s * c[0], c[1], 0.0), (length, thick, PAWL_H), CATCH_ORANGE,
                    (math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2))))
        return out


PAWLS = {
    "catch_far_left": Pawl("catch_far_left", "wall_far", +1, "wall_left"),
    "catch_far_right": Pawl("catch_far_right", "wall_far", -1, "wall_right"),
    "catch_near_left": Pawl("catch_near_left", "wall_near", +1, "wall_left"),
    "catch_near_right": Pawl("catch_near_right", "wall_near", -1, "wall_right"),
}

# -- judging ------------------------------------------------------------------------------------------------------
UPRIGHT_TOL = math.radians(1.0)    # a latched corner's walls are within this of upright (q = pi/2)
PAWL_OUT_TOL = 0.001               # ... and its pawl is within 1 mm of fully out
VERTICAL_TOL = math.radians(3.0)   # full credit: every wall within 3 degrees of vertical
FLAT_TOL = math.radians(3.0)       # full credit: the base within 3 degrees of flat


def hinge_joint(wall: str) -> str:
    return f"{wall}_hinge"


def pawl_joint(pawl: str) -> str:
    return f"{pawl}_slide"


def corner_latched(joints: dict, pawl: str) -> bool:
    """A catch holds: its long wall and the short wall it holds are both within 1 degree of upright and its pawl is
    within 1 mm of fully out. With the short wall that close to upright, its inner face is past the pawl's square
    face (the pawl stands 1 mm inside it), so the pawl could only be out if the wall had gone past it."""
    p = PAWLS[pawl]
    up = all(abs(joints.get(hinge_joint(w), 0.0) - UP) <= UPRIGHT_TOL for w in (p.wall, p.holds))
    return bool(up and joints.get(pawl_joint(pawl), -1.0) >= -PAWL_OUT_TOL)


def crate_to_station(p) -> np.ndarray:
    return np.asarray(p, float) + np.array([XC, YC, TOP])


def rail_world(wall: str, q: float) -> np.ndarray:
    """The rail's centre (station frame) with the wall at angle q and the crate at its start pose."""
    w = WALLS[wall]
    return crate_to_station(w.point(w.rail_local, q))
