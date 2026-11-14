"""Geometry of the cabin-filter scene: a passenger-side dashboard section with its glovebox (a bin on a bottom hinge
whose two side walls carry sprung stop bumps), the HVAC filter housing behind it, and two cabin air filters. Pure
NumPy, so the task file, the asset script and the tests can import it before Isaac Sim starts.

Frames (metres):
- station: x forward from the arm bases, y to the robot's left, z up; the table top is at ``TABLE``.
- dash frame D: origin on the table top straight below the glovebox hinge axis; axes as the station's. The dash
  section stands square to the robot, its hinge axis ``X_HINGE`` ahead of the arm bases.
- bin frame B: origin on the hinge axis (D (0, 0, ``HINGE_Z``)); at angle 0 (shut) its axes are D's. The bin turns
  about the station's -y axis by the joint angle ``theta``: positive swings its top toward the robot. A point at
  polar angle ``phi`` in the bin's x-z plane (from +x toward +z) is at polar angle ``phi + theta`` in the world.
- pin frames: the two flexing upper side-wall panels (one per side) carry the stop bumps. Each is a link whose
  frame coincides with B at rest and slides along B's y axis; positive is inward (toward the bin's middle).
- filter frame F: origin at the centre of the filter's frame body; +x points into the slot (the end without the
  pull tab), +z is the airflow-in face (the face with the key rib).

A box is (name, centre, size, rotation) in its frame.
"""

from __future__ import annotations

import math

import numpy as np

TABLE = 0.7475                          # yam_arena.task_file.TABLE_TOP["station"]
X_HINGE = 0.45                          # hinge axis, ahead of the arm bases
HINGE_Z = 0.135                         # hinge axis above the table: the dash section's mounting height
EYE = np.eye(3)


def rot_y(pitch: float) -> np.ndarray:
    """Rotation about +y (positive turns +x down)."""
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_axis(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def lohi(name: str, lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (name, (lo + hi) / 2, hi - lo, EYE)


# -- glovebox bin (bin frame) ---------------------------------------------------------------------------------------
HALF_W = 0.18                           # outer half width of the bin (36 cm wide)
WALL_T = 0.006                          # moulded polypropylene walls
PANEL = (-0.026, -0.006, -0.004, 0.205)  # front panel (the lid you see): x range, z range
FLOOR_D = 0.060                         # floor depth behind the front panel's inner face
BACK_TOP = (0.130, 0.160)               # top of the slanted back wall (x, z); it rises from (FLOOR_D, 0)
FLEX_Z = 0.050                          # side walls above this height are the flexing panels (pin links)
BIN_MASS = 0.85                         # kg: an empty glovebox bin with its lid, latch and damper arm
BIN_COM = (0.010, 0.0, 0.085)           # bin frame: the lid's weight puts it in front of the hinge once open
PIN_MASS = 0.03
# OPEN_STOP (the angle at which the bin rests on its stops) follows from the geometry: see the end of this section
LOWERED = math.radians(110.0)           # lowered past the stops, it hangs on its hinge stop here
HINGE_DAMPING = 0.25                    # N m s/rad, the glovebox's damper
# the flexing panels: sprung outward, travel 0..PIN_TRAVEL inward
PIN_TRAVEL = 0.014
PIN_RATE = 900.0                        # N/m
PIN_PRELOAD = 2.7                       # N at rest (spring rest point 3 mm outside the outer stop)
PIN_TARGET = -PIN_PRELOAD / PIN_RATE
PIN_DAMPING = 3.0                       # N s/m
# the stop bump on each flexing panel, near its top back corner
BUMP_C = np.array([0.100, 0.090])       # (x, z), bin frame: near the flexing panel's back edge
BUMP_OUT = 0.037                        # nominal height; the square part stands 36 mm out (TOP_DROP lower), 6 mm past
                                        # the opening's edge, and the chamfer peaks at 38.1 mm
BUMP_T = 0.014                          # along its path (tangential): the square-faced part
BUMP_R = 0.018                          # radial extent
RAMP_RISE = 0.012                       # the outer 12 mm of the bump is chamfered on its trailing side ...
RAMP_L = RAMP_RISE / math.tan(math.radians(20.0))   # ... at 20 degrees to its path
RAMP_W = 0.010                          # thickness of the chamfer slab
RAMP_OVER = 0.003                       # the chamfer runs this far over the square part (peaking 2.1 mm above it)
TOP_DROP = 0.001                        # the square part's top is this much lower than the chamfer at its edge
# a moulded thumb pad on each flexing panel's outer face: where it is pushed in
PAD = (-0.004, 0.050, 0.070, 0.130)     # x range, z range (bin frame)
PAD_OUT = 0.026                         # stands out from the wall's outer face (to 4 mm inside the opening's edge)
BUMP_PHI = math.atan2(BUMP_C[1], BUMP_C[0])
BUMP_RAD = float(np.hypot(*BUMP_C))


def back_x(z: float) -> float:
    """x of the back wall's outer face at height ``z`` (bin frame)."""
    return FLOOR_D + (BACK_TOP[0] - FLOOR_D) * z / BACK_TOP[1]


def bump_frame(sign: float) -> np.ndarray:
    """Rotation of a bump (bin frame): columns tangential (opening direction), outward (sign * y), and their cross
    product."""
    t = np.array([-math.sin(BUMP_PHI), 0.0, math.cos(BUMP_PHI)])
    o = np.array([0.0, sign, 0.0])
    return np.column_stack([t, o, np.cross(t, o)])


def bin_boxes():
    """The bin's fixed parts (bin frame)."""
    x0, x1, z0, z1 = PANEL
    out = [lohi("panel", (x0, -HALF_W, z0), (x1, HALF_W, z1)),
           lohi("floor", (x1, -HALF_W, 0.0), (FLOOR_D, HALF_W, WALL_T))]
    # the back wall: a slab along the line from (FLOOR_D, 0) to BACK_TOP, its outer face on that line
    d = np.array([BACK_TOP[0] - FLOOR_D, BACK_TOP[1]])
    length = float(np.linalg.norm(d))
    lean = math.atan2(d[0], d[1])                     # from vertical toward +x
    inward = np.array([-math.cos(lean), math.sin(lean)])  # into the bin, normal to the wall
    mid = np.array([FLOOR_D, 0.0]) + d / 2 + inward * WALL_T / 2
    out.append(("back", np.array([mid[0], 0.0, mid[1]]), np.array([WALL_T, 2 * HALF_W, length]), rot_y(lean)))
    # fixed lower side walls, stepped to follow the slanted back
    for sign, side in ((1, "left"), (-1, "right")):
        for k in range(4):
            z0s, z1s = k * FLEX_Z / 4, (k + 1) * FLEX_Z / 4
            y0, y1 = sorted((sign * (HALF_W - WALL_T), sign * HALF_W))
            out.append(lohi(f"side_{side}_{k}", (x1, y0, z0s), (back_x(z0s), y1, z1s)))
    return out


def pin_boxes(sign: float):
    """One flexing panel (pin link, frame = bin frame at rest): two wall strips and the stop bump (a square leading
    face and a 45-degree ramp behind it)."""
    side = "left" if sign > 0 else "right"
    y0, y1 = sorted((sign * (HALF_W - WALL_T), sign * HALF_W))
    zs = (FLEX_Z, 0.110, BACK_TOP[1])
    out = [lohi(f"flex_{side}_{k}", (PANEL[1], y0, zs[k]), (back_x(zs[k]), y1, zs[k + 1])) for k in range(2)]
    out.append(lohi(f"pad_{side}", (PAD[0], sign * HALF_W if sign < 0 else HALF_W, PAD[2]),
                    (PAD[1], sign * (HALF_W + PAD_OUT) if sign < 0 else HALF_W + PAD_OUT, PAD[3]))
               if sign > 0 else lohi(f"pad_{side}", (PAD[0], -HALF_W - PAD_OUT, PAD[2]), (PAD[1], -HALF_W, PAD[3])))
    r = bump_frame(sign)
    base = np.array([BUMP_C[0], sign * HALF_W, BUMP_C[1]])          # the bump's root on the wall's outer face

    def at(t, o):                                                  # bin-frame point at (tangential, outward)
        return base + r[:, 0] * t + r[:, 1] * o

    h_low = BUMP_OUT - RAMP_RISE
    t_back = -BUMP_T / 2 - RAMP_L
    # the low part runs the whole length; the high part has the square leading face; the chamfer joins them
    out.append((f"bump_{side}", at((t_back + BUMP_T / 2) / 2, h_low / 2), np.array([BUMP_T / 2 - t_back, h_low, BUMP_R]), r))
    # the square part's top sits TOP_DROP below the chamfer's peak, so its trailing top edge is buried in the chamfer
    top = RAMP_RISE - TOP_DROP
    out.append((f"bumptop_{side}", at(0.0, h_low + top / 2), np.array([BUMP_T, top, BUMP_R]), r))
    # the chamfer runs on RAMP_OVER past the square part's trailing edge (ending a little proud of its top), so the
    # fascia's edge steps down onto the flat instead of catching on that edge
    length, rise = RAMP_L + RAMP_OVER, RAMP_RISE * (RAMP_L + RAMP_OVER) / RAMP_L
    d = np.array([length, rise]) / math.hypot(length, rise)             # along the chamfer (t, out)
    n = np.array([d[1], -d[0]])                                         # into the slab
    mid = np.array([t_back + length / 2, h_low + rise / 2]) + n * RAMP_W / 2
    turn = rot_axis((0, 0, 1), math.atan2(d[1], d[0]))
    out.append((f"ramp_{side}", at(*mid), np.array([math.hypot(length, rise), RAMP_W, BUMP_R]), r @ turn))
    return out


def bin_rot(theta: float) -> np.ndarray:
    return rot_y(-theta)


# -- dash section (dash frame) --------------------------------------------------------------------------------------
FASCIA = (-0.026, -0.006)               # the fascia's thickness (x range); the shut lid sits flush in it
OPEN_HALF_W = 0.210                     # glovebox opening: half width ...
OPEN_TOP = 0.265                        # ... from the hinge up to this (bin frame z)
OPEN_BOTTOM = -0.050
DASH_HALF_W = 0.27
DASH_TOP = 0.30                         # top of the cropped dash section above the hinge
CAVITY_HALF_W = 0.226                   # the cavity's side walls behind the fascia
X_FACE = 0.140                          # the filter housing's face (dash x)
FACE_T = 0.008
SLOT_Y = 0.0                            # slot centre (dash y)
SLOT_FLOOR = HINGE_Z + 0.082            # slot floor (dash z); a seated filter rests on it
F_SIZE = (0.200, 0.215, 0.030)          # filter frame: depth (x), width (y), thickness (z)
SLOT_HALF_W = F_SIZE[1] / 2 + 0.00375   # 3.75 mm play each side
SLOT_H = F_SIZE[2] + 0.006              # 6 mm play above
RIB = ((0.060, 0.070), 0.005)           # the filter's key rib: y range (filter frame), height above the top face
KEY = ((0.0575, 0.0725), 0.009)         # the slot's key channel: y range (dash, from SLOT_Y), depth above the slot
TUNNEL_T = 0.004
SILL_DROP = 0.016                       # the lead-in sill in front of the slot starts this far below its floor
SIDE_LEAD = 0.010                       # the side cheeks widen the slot's mouth by this much each side
HOOD_RISE = 0.010                       # the lead-in hood above the slot's mouth (either side of the tab) rises this much
def dash_boxes():
    """The fixed dash section (dash frame): fascia round the glovebox opening (the rear faces of its two side members
    are the stops the bin's bumps catch on), the cavity behind it and the filter housing with its slot."""
    h = HINGE_Z
    f0, f1 = FASCIA
    W, top = DASH_HALF_W, h + DASH_TOP
    out = [lohi("knee", (f0, -W, 0.0), (f1, W, h + OPEN_BOTTOM)),
           lohi("fascia_top", (f0, -W, h + OPEN_TOP), (f1, W, top)),
           lohi("fascia_left", (f0, OPEN_HALF_W, h + OPEN_BOTTOM), (f1, W, h + OPEN_TOP)),
           lohi("fascia_right", (f0, -W, h + OPEN_BOTTOM), (f1, -OPEN_HALF_W, h + OPEN_TOP)),
           lohi("base", (f1, -W, 0.0), (X_FACE + 0.23, W, h - 0.060)),
           lohi("cavity_floor", (f1, -CAVITY_HALF_W, h - 0.060), (X_FACE, CAVITY_HALF_W, h + OPEN_BOTTOM)),
           lohi("cavity_left", (f1, CAVITY_HALF_W, h - 0.060), (X_FACE + 0.23, W, top)),
           lohi("cavity_right", (f1, -W, h - 0.060), (X_FACE + 0.23, -CAVITY_HALF_W, top)),
           lohi("cavity_ceiling", (f1, -CAVITY_HALF_W, h + OPEN_TOP), (X_FACE + 0.23, CAVITY_HALF_W, top))]
    # the housing face with the slot (and its key channel) cut out
    x0, x1 = X_FACE, X_FACE + FACE_T
    sy0, sy1 = SLOT_Y - SLOT_HALF_W, SLOT_Y + SLOT_HALF_W
    z0, z1 = SLOT_FLOOR, SLOT_FLOOR + SLOT_H
    k0, k1 = SLOT_Y + KEY[0][0], SLOT_Y + KEY[0][1]
    kz = z1 + KEY[1]
    C = CAVITY_HALF_W
    out += [lohi("face_low", (x0, -C, h - 0.060), (x1, C, z0)),
            lohi("face_high", (x0, -C, kz), (x1, C, h + OPEN_TOP)),
            lohi("face_left", (x0, sy1, z0), (x1, C, kz)),
            lohi("face_right", (x0, -C, z0), (x1, sy0, kz)),
            lohi("face_key_l", (x0, k1, z1), (x1, sy1, kz)),
            lohi("face_key_r", (x0, sy0, z1), (x1, k0, kz))]
    # the tunnel behind the face: floor, ceiling (with the channel), side walls, back stop
    b = X_FACE + 0.006 + F_SIZE[0] + 0.006
    T = TUNNEL_T
    out += [lohi("tunnel_floor", (x1, sy0 - T, z0 - T), (b, sy1 + T, z0)),
            lohi("tunnel_ceil_l", (x1, k1, z1), (b, sy1 + T, z1 + T)),
            lohi("tunnel_ceil_r", (x1, sy0 - T, z1), (b, k0, z1 + T)),
            lohi("tunnel_key", (x1, k0 - T, kz), (b, k1 + T, kz + T)),
            lohi("tunnel_key_l", (x1, k1, z1), (b, k1 + T, kz)),
            lohi("tunnel_key_r", (x1, k0 - T, z1), (b, k0, kz)),
            lohi("tunnel_left", (x1, sy1, z0 - T), (b, sy1 + T, z1)),
            lohi("tunnel_right", (x1, sy0 - T, z0 - T), (b, sy0, z1)),
            lohi("tunnel_back", (b, sy0 - T, z0 - T), (b + T, sy1 + T, kz + T))]
    # a moulded lead-in sill under the slot's mouth: a 45-degree ramp rising from 2 cm below the floor to the floor
    ramp = math.radians(45.0)
    length = SILL_DROP / math.sin(ramp)
    c = np.array([x0 - SILL_DROP / math.tan(ramp) / 2, SLOT_Y, z0 - SILL_DROP / 2])
    c += np.array([math.sin(ramp), 0.0, -math.cos(ramp)]) * 0.004            # top face on the ramp line
    out.append(("sill", c, np.array([length, 2 * SLOT_HALF_W, 0.008]), rot_y(-ramp)))
    # and a lead-in hood over the mouth: 45-degree ramps rising from the slot's top edge, clear of the tab and fingers
    length = HOOD_RISE / math.sin(ramp)
    for side, (h0, h1) in (("l", (k1 - SLOT_Y, SLOT_HALF_W)), ("r", (-SLOT_HALF_W, -(k1 - SLOT_Y)))):
        c = np.array([x0 - HOOD_RISE / math.tan(ramp) / 2, SLOT_Y + (h0 + h1) / 2, z1 + HOOD_RISE / 2])
        c += np.array([-math.sin(ramp), 0.0, math.cos(ramp)]) * -0.004 * -1
        out.append((f"hood_{side}", c, np.array([length, h1 - h0, 0.008]), rot_y(ramp)))
    # and lead-ins at the slot's side edges: 45-degree cheeks in front of the face, widening the mouth by SIDE_LEAD
    for side, sg in (("l", 1.0), ("r", -1.0)):
        r = rot_axis((0, 0, 1), sg * math.radians(45.0))
        edge = np.array([x0, SLOT_Y + sg * SLOT_HALF_W, (z0 + z1) / 2])
        along = np.array([-1.0, sg, 0.0]) / math.sqrt(2.0)          # from the slot edge forward and outward
        outward = np.array([1.0, sg, 0.0]) / math.sqrt(2.0)         # into the cheek
        c = edge + along * SIDE_LEAD * math.sqrt(2.0) / 2 + outward * 0.004
        out.append((f"cheek_{side}", c, np.array([0.008, SIDE_LEAD * math.sqrt(2.0), z1 - z0]), r))
    return out


def hinge_world() -> np.ndarray:
    return dash_to_world((0.0, 0.0, HINGE_Z))


def dash_to_world(p) -> np.ndarray:
    return np.asarray(p, float) + np.array([X_HINGE, 0.0, TABLE])


# -- filters (filter frame) -----------------------------------------------------------------------------------------
F_MASS = 0.15
TAB = ((-0.135, -0.100), 0.008, (-0.015, 0.045))  # pull tab on the front edge: x range, half thickness, z range
GRIP = np.array([-0.1260, 0.0, 0.026])           # where a gripper's TCP holds the tab (filter frame)


def filter_boxes():
    dx, dy, dz = F_SIZE
    (t0, t1), ty, (tz0, tz1) = TAB
    (r0, r1), rh = RIB
    return [("frame", np.zeros(3), np.array(F_SIZE), EYE),
            lohi("tab", (t0, -ty, tz0), (t1, ty, tz1)),
            lohi("rib", (-dx / 2, r0, dz / 2), (dx / 2, r1, dz / 2 + rh))]


SEAT_C = np.array([X_FACE + F_SIZE[0] / 2, SLOT_Y, SLOT_FLOOR + F_SIZE[2] / 2])   # seated filter centre, dash frame
# the new filter lies on a stand (an upturned parts bin) to the right; the table to the left is clear for the old one
STAND = (np.array([0.36, -0.45]), np.array([0.30, 0.26, 0.10]))   # centre (station xy) and size
STAND_TOP = TABLE + STAND[1][2]
NEW_START = np.array([0.33, -0.45, STAND_TOP + F_SIZE[2] / 2])
OLD_DOWN = np.array([0.30, 0.40])                                  # where the demo sets the old filter down
OLD_START = dash_to_world(SEAT_C + [0.0, 0.0, 0.0005])


# -- placing boxes in the station frame -----------------------------------------------------------------------------
def world_boxes(boxes, origin, rot):
    """(name, centre, rotation, half sizes) in the station frame for boxes in a frame at ``origin`` with ``rot``."""
    return [(n, np.asarray(origin, float) + rot @ np.asarray(c, float), rot @ r, np.asarray(s, float) / 2)
            for n, c, s, r in boxes]


def dash_world():
    return world_boxes(dash_boxes(), dash_to_world((0, 0, 0)), EYE)


def bin_world(theta: float, pins=(0.0, 0.0)):
    """The bin and its two flexing panels at hinge angle ``theta`` with the panels pushed in by ``pins`` (left,
    right)."""
    r = bin_rot(theta)
    o = hinge_world()
    out = world_boxes(bin_boxes(), o, r)
    for sign, d in ((1.0, pins[0]), (-1.0, pins[1])):
        out += world_boxes(pin_boxes(sign), o + r @ np.array([0.0, -sign * d, 0.0]), r)
    return out


def filter_world(centre, rot=EYE):
    return world_boxes(filter_boxes(), centre, rot)


def bin_point(p_bin, theta: float) -> np.ndarray:
    """A bin-frame point in the station frame."""
    return hinge_world() + bin_rot(theta) @ np.asarray(p_bin, float)


# -- simple box maths -----------------------------------------------------------------------------------------------
def box_points(c, r, h, n: int = 5) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n)
    g = np.array(np.meshgrid(t, t, t)).reshape(3, -1).T
    return c + (g * h) @ r.T


def point_box_dist(points, c, r, h) -> np.ndarray:
    """Distance from each point to a box (0 inside)."""
    local = (np.atleast_2d(points) - c) @ r
    return np.linalg.norm(np.maximum(np.abs(local) - h, 0.0), axis=1)


def sat_overlap(a, b) -> float:
    """Separating-axis test for two boxes: the smallest overlap along any of the 15 axes (<= 0: separated)."""
    (ca, ra, ha), (cb, rb, hb) = a, b
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    d = np.asarray(cb, float) - np.asarray(ca, float)
    best = math.inf
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        pa = float(np.abs(ra.T @ ax) @ ha)
        pb = float(np.abs(rb.T @ ax) @ hb)
        best = min(best, pa + pb - abs(float(d @ ax)))
    return best


def boxes_gap(a, b, n: int = 7) -> float:
    """Distance between two boxes: negative (the overlap depth, exact by separating axes) when they overlap,
    otherwise approximated from lattices of points on each."""
    (ca, ra, ha), (cb, rb, hb) = a, b
    over = sat_overlap(a, b)
    if over > 0:
        return -over
    pa, pb = box_points(ca, ra, ha, n), box_points(cb, rb, hb, n)
    return float(min(point_box_dist(pa, cb, rb, hb).min(), point_box_dist(pb, ca, ra, ha).min()))


def gap(a, b) -> float:
    """Smallest gap between two lists of world boxes (name, centre, rotation, half sizes)."""
    return min(boxes_gap(x[1:], y[1:]) for x in a for y in b)


# -- the stops --------------------------------------------------------------------------------------------------------
def bump_parts(theta: float, travel: float = 0.0, side: str = "left"):
    """One side's bump (station frame) at hinge angle ``theta`` with its panel pushed in by ``travel``."""
    pins = (travel, 0.0) if side == "left" else (0.0, travel)
    return [b for b in bin_world(theta, pins) if b[0].endswith(side) and b[0].startswith(("bump", "ramp"))]


def _stop_angle() -> float:
    """The angle at which the bumps' square leading faces meet the rear of the fascia (the bin rests there)."""
    fascia = [b for b in dash_world() if b[0] == "fascia_left"]
    lo = math.radians(20.0)
    while gap(bump_parts(lo + math.radians(0.5)), fascia) > 0:
        lo += math.radians(0.5)
    hi = lo + math.radians(0.5)
    for _ in range(20):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(bump_parts(mid), fascia) > 0 else (lo, mid)
    return lo


OPEN_STOP = _stop_angle()
