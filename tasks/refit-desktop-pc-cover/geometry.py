"""Geometry of the refit-desktop-pc-cover scene: a small-form-factor desktop PC with its cover off, and the cover.

Pure NumPy, so the task file and the tests can import it before Isaac Sim starts.

Chassis frame ("case frame"): origin at the centre of the case's footprint on the underside of its floor, x toward
the case's front (away from the robot), y to the robot's left, z up. The cover's own frame is the same frame moved
with the cover: when the cover is fitted, the two frames differ only by RIM in z.

Sizes are those of a common small-form-factor office desktop (about 29 x 29 x 9.4 cm): a 1 mm steel case with a
folded-in rim, and a 0.8 mm steel cover with side lips, six hooks, a moulded rear grip and a hole for the latch.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                            # yam_arena.task_file.TABLE_TOP["station"]
MM = 0.001

# -- case (chassis frame) -------------------------------------------------------------------------------------------
HX, HY = 0.146, 0.145                   # half depth (x) and half width (y): 292 x 290 mm outside
FOOT = 0.004                            # rubber feet
SHEET = 0.001                           # case steel
BODY_H = 0.090                          # floor underside to the rim's top face
RIM = FOOT + BODY_H                     # rim top above the table (and above the case origin, which is on the table)
FLANGE_W = 0.025                        # the rim's folded-in side flanges (y), 1 mm thick
END_FLANGE_W = 0.015                    # rear and front flanges (x)
CASE_MASS = 4.6                         # kg, with its board, drive and power supply

# slots: three per side flange, 23 mm (x) long, centred in the flange's width
SLOT_PLAY = 0.0025                      # side play either side of an 8 mm hook
SLOT_LEN, SLOT_W = 0.023, 0.008 + 2 * SLOT_PLAY
SLOT_Y = 0.1315                         # slot centre |y|: 6 to 7 mm of flange either side of a slot
SLOT_CX = (-0.090, 0.000, 0.090)        # slot centres (x)

# hooks (cover frame, cover fitted): stem 3 mm (x) at the slot's front end, toe 12 mm (x) pointing to the front
HOOK_W = 0.008                          # y
STEM_T = 0.003
TOE_LEN = 0.012
TOE_T = 0.0015
TOE_GAP = 0.0015                        # toe top to the flange's underside, cover fitted
HOOK_DROP = SHEET + TOE_GAP + TOE_T     # 4 mm below the cover's underside
# A hook's stem rear face, cover fitted. The slot runs from 19 mm behind it to 4 mm in front of it.
HOOK_X = tuple(round(c + SLOT_LEN / 2 - 0.004, 6) for c in SLOT_CX)   # stem rear face = slot front end - 4 mm

SLIDE = 0.015                           # the cover's travel from "dropped in" (s = 0) to fitted (s = 15 mm)
# the hooks fit through the slots for s from DROP_MIN to DROP_MAX (rear and front clearances of the slot)
DROP_MIN = -0.004                       # a hook's rear end meets its slot's rear end
DROP_MAX = 0.007                        # a toe's tip meets its slot's front end
ENGAGED_S = 0.009                       # every toe at least 2 mm under the flange in front of its slot
STOP_S = 0.016                          # the stems reach the slots' front ends

# latch: a spring plunger in the left flange; it pops up through a 12 x 12 mm hole in the cover when the cover is home
LATCH_XY = (-0.045, SLOT_Y)
LATCH_SIDE = 0.008
LATCH_UP = 0.003                        # free plunger stands 3 mm above the rim
LATCH_TRAVEL = 0.004
LATCH_RATE = 600.0                      # N/m: 1.8 N with the cover on it, much less than the cover's 5.9 N weight
HOLE = 0.012
LATCH_S = (SLIDE - (HOLE - LATCH_SIDE) / 2, SLIDE + (HOLE - LATCH_SIDE) / 2)   # 13..17 mm: the plunger fits the hole
LATCHED_S = (LATCH_S[0], 0.016)         # 13..16 mm: where a latched cover can be (the stems stop it at 16)
LATCHED_Q = -0.0015                     # plunger joint above this (m): at least 1.5 mm up into the hole

# -- cover (cover frame: origin on the plate's underside, at the case's centre when fitted) ---------------------------
PLATE_T = 0.0008
LIP_GAP = 0.0015                        # side lips stand 1.5 mm off the case's side walls
LIP_Y = HY + LIP_GAP                    # inner face of a lip
LIP_DEPTH = 0.012
COVER_HY = LIP_Y + SHEET                # 0.147
FLARE = 0.003                           # the lips' bottom 3 mm is bent out 45 degrees: a 3 mm lead-in
GRIP = dict(x0=-HX, x1=-HX + 0.016, h=0.025)   # two moulded plastic pull tabs on the plate's rear edge ...
TAB_Y, TAB_W = 0.120, 0.040             # ... centred 12 cm either side of the middle, 4 cm wide
COVER_MASS = 0.60
COVER_HULL = (np.array([-HX, -COVER_HY - FLARE, -LIP_DEPTH - 0.001]), np.array([HX, COVER_HY + FLARE, PLATE_T + GRIP["h"]]))

# -- layout on the station (world frame) ------------------------------------------------------------------------------
CASE_XY = (0.276, -0.160)               # case centre: rear face at x 0.13, right side at y -0.305
COVER_START_Y = 0.160                   # the cover lies on the table to the case's left ...
START_S = 0.000                         # ... level with where it is lowered (4 mm into the hooks' 11 mm drop window)
_S45 = math.sqrt(0.5)
LIP_FOOT = (LIP_DEPTH - FLARE) + FLARE / 2 + ((FLARE * math.sqrt(2) + SHEET) / 2 + SHEET / 2) * _S45  # lowest point
COVER_START = (CASE_XY[0] - SLIDE + START_S, COVER_START_Y, TOP + LIP_FOOT)   # resting on its lips' flared feet
GRASP_DY = TAB_Y                        # each arm pinches one tab
PAD_ABOVE_PLATE = 0.006                 # pad tips this far above the plate's top
YAW = {"left": -math.pi / 2, "right": math.pi / 2}   # grips turned so the fingers close along x (IK reaches these)


def case_world(p) -> np.ndarray:
    """A chassis-frame point on the station, with the case where it starts."""
    p = np.asarray(p, float)
    return np.array([CASE_XY[0] + p[0], CASE_XY[1] + p[1], TOP + p[2]])


def grip_tcp_offset() -> np.ndarray:
    """Where the TCP sits in the cover frame when an arm pinches the grip (y = 0 here)."""
    from_tip = 0.013                    # yam_arena.constants.PAD_TIP_PAST_TCP
    return np.array([(GRIP["x0"] + GRIP["x1"]) / 2, 0.0, PLATE_T + PAD_ABOVE_PLATE + from_tip])


# -- part shapes ----------------------------------------------------------------------------------------------------
STEEL = (0.62, 0.64, 0.67)
DARK_STEEL = (0.33, 0.34, 0.36)
BLACK = (0.08, 0.08, 0.09)
GREEN_PCB = (0.10, 0.42, 0.22)
ALU = (0.75, 0.76, 0.78)
ORANGE = (0.95, 0.45, 0.08)
BLUE = (0.15, 0.25, 0.60)


def _b(x0, x1, y0, y1, z0, z1, color):
    return (((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)


def _slot_band_segments(x0: float, x1: float, gaps) -> list[tuple[float, float]]:
    """[x0, x1] minus the given (start, end) gaps, as solid pieces."""
    out, x = [], x0
    for a, b in sorted(gaps):
        if a > x:
            out.append((x, a))
        x = max(x, b)
    if x < x1:
        out.append((x, x1))
    return out


def slots() -> list[tuple[float, float]]:
    return [(c - SLOT_LEN / 2, c + SLOT_LEN / 2) for c in SLOT_CX]


def case_boxes() -> list:
    """Floor, feet, walls, the folded-in rim with its slots, and the parts inside."""
    z0, zr = FOOT, RIM
    s = SHEET
    out = [_b(-HX, HX, -HY, HY, z0, z0 + s, DARK_STEEL)]
    out += [_b(fx - 0.006, fx + 0.006, fy - 0.006, fy + 0.006, 0.0, z0, BLACK)
            for fx in (-HX + 0.02, HX - 0.02) for fy in (-HY + 0.02, HY - 0.02)]
    out += [_b(-HX, -HX + s, -HY, HY, z0 + s, zr, STEEL), _b(HX - s, HX, -HY, HY, z0 + s, zr, STEEL),
            _b(-HX + s, HX - s, HY - s, HY, z0 + s, zr, STEEL), _b(-HX + s, HX - s, -HY, -HY + s, z0 + s, zr, STEEL)]
    fz0 = zr - s
    out += [_b(-HX + s, -HX + END_FLANGE_W, -HY + s, HY - s, fz0, zr, STEEL),
            _b(HX - END_FLANGE_W, HX - s, -HY + s, HY - s, fz0, zr, STEEL)]
    xa, xb = -HX + END_FLANGE_W, HX - END_FLANGE_W
    latch_gap = (LATCH_XY[0] - LATCH_SIDE / 2 - 0.0005, LATCH_XY[0] + LATCH_SIDE / 2 + 0.0005)
    for side in (1, -1):
        outer = sorted((side * (HY - s), side * (SLOT_Y + SLOT_W / 2)))
        inner = sorted((side * (SLOT_Y - SLOT_W / 2), side * (HY - FLANGE_W)))
        band = sorted((side * (SLOT_Y + SLOT_W / 2), side * (SLOT_Y - SLOT_W / 2)))
        out += [_b(xa, xb, *outer, fz0, zr, STEEL), _b(xa, xb, *inner, fz0, zr, STEEL)]
        gaps = slots() + ([latch_gap] if side == 1 else [])
        out += [_b(a, b, *band, fz0, zr, STEEL) for a, b in _slot_band_segments(xa, xb, gaps)]
    # inside: motherboard, CPU cooler with its shroud, two memory modules, power supply, 3.5-inch drive
    fl = z0 + s
    out += [
        _b(-0.130, 0.090, -0.115, 0.095, fl + 0.006, fl + 0.0076, GREEN_PCB),
        _b(-0.090, 0.000, -0.020, 0.070, fl + 0.0076, fl + 0.055, ALU),
        _b(0.010, 0.143, 0.075, 0.079, fl + 0.0076, fl + 0.039, BLUE),
        _b(0.010, 0.143, 0.083, 0.087, fl + 0.0076, fl + 0.039, BLUE),
        _b(-0.140, 0.000, -0.140, -0.040, fl, fl + 0.075, DARK_STEEL),
        _b(0.020, 0.140, -0.140, -0.038, fl, fl + 0.026, ALU),
    ]
    return out


def latch_boxes() -> list:
    """The plunger, in the chassis frame at its rest position: 8 x 8 mm, 13 mm tall, its top 3 mm above the rim."""
    x, y = LATCH_XY
    h = LATCH_SIDE / 2
    return [_b(x - h, x + h, y - h, y + h, RIM + LATCH_UP - 0.013, RIM + LATCH_UP, ORANGE)]


def cover_boxes() -> list:
    """Plate (with the latch hole), side lips, six hooks and the rear grip, in the cover frame."""
    t = PLATE_T
    hx0, hx1 = LATCH_XY[0] - HOLE / 2, LATCH_XY[0] + HOLE / 2
    hy0, hy1 = LATCH_XY[1] - HOLE / 2, LATCH_XY[1] + HOLE / 2
    out = [_b(-HX, hx0, -COVER_HY, COVER_HY, 0.0, t, STEEL), _b(hx1, HX, -COVER_HY, COVER_HY, 0.0, t, STEEL),
           _b(hx0, hx1, -COVER_HY, hy0, 0.0, t, STEEL), _b(hx0, hx1, hy1, COVER_HY, 0.0, t, STEEL)]
    straight = LIP_DEPTH - FLARE
    out += [_b(-HX, HX, LIP_Y, COVER_HY, -straight, 0.0, STEEL), _b(-HX, HX, -COVER_HY, -LIP_Y, -straight, 0.0, STEEL)]
    half = math.radians(22.5)
    for side in (1, -1):   # the flare: a 1 mm sheet from the lip's foot, 45 degrees outward and down
        cy = side * (LIP_Y + SHEET / 2 + FLARE / 2)
        q = (math.cos(half), -side * math.sin(half), 0.0, 0.0)
        out.append(((0.0, cy, -straight - FLARE / 2), (2 * HX, FLARE * math.sqrt(2) + SHEET, SHEET), STEEL, q))
    for side in (1, -1):
        y0, y1 = side * SLOT_Y - HOOK_W / 2, side * SLOT_Y + HOOK_W / 2
        for hx in HOOK_X:
            out.append(_b(hx, hx + STEM_T, y0, y1, -HOOK_DROP + TOE_T, 0.0, STEEL))
            out.append(_b(hx, hx + TOE_LEN, y0, y1, -HOOK_DROP, -HOOK_DROP + TOE_T, STEEL))
    g = GRIP
    for side in (1, -1):
        out.append(_b(g["x0"], g["x1"], side * TAB_Y - TAB_W / 2, side * TAB_Y + TAB_W / 2, t, t + g["h"], BLACK))
    return out


# -- state tests (pure geometry; the task file feeds them the simulated poses) --------------------------------------
def quat_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def cover_in_case(case_pos, case_q, cover_pos, cover_q):
    """The cover's pose in the chassis frame: (offset from its fitted pose, rotation matrix)."""
    rc = quat_matrix(case_q)
    p = rc.T @ (np.asarray(cover_pos, float) - np.asarray(case_pos, float)) - np.array([0.0, 0.0, RIM])
    return p, rc.T @ quat_matrix(cover_q)


PLATE_CORNERS = np.array([[sx * HX, sy * HY, 0.0] for sx in (-1, 1) for sy in (-1, 1)])


def slide_of(offset) -> float:
    """How far (m) the cover has slid toward the front, counted from where its hooks drop straight in (s = 0)."""
    return float(offset[0] + SLIDE)


def seated(offset, rot) -> bool:
    """The cover lies flat on the rim: every corner of its plate's underside within -1..+2 mm of the rim's top, square
    to the case within 2 degrees, centred within 2 mm across, and slid -6 to +18 mm from the drop position. Only with
    every hook down through its slot can it lie this low."""
    corners = (rot @ PLATE_CORNERS.T).T + offset
    yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    s = slide_of(offset)
    return bool(np.all(corners[:, 2] > -0.001) and np.all(corners[:, 2] < 0.002) and abs(yaw) < 2.0
                and abs(offset[1]) < 0.002 and -0.006 < s < STOP_S + 0.002)


def stage(offset, rot, latch_q: float) -> str:
    """'latched', 'engaged', 'seated' or 'off' (the furthest mechanical stage the cover is in now)."""
    if not seated(offset, rot):
        return "off"
    s = slide_of(offset)
    if s >= ENGAGED_S and latch_q > LATCHED_Q and LATCHED_S[0] - 0.0005 <= s <= LATCHED_S[1] + 0.0005:
        return "latched"
    if s >= ENGAGED_S:
        return "engaged"
    return "seated"


def dist_to_box(points, lo, hi) -> np.ndarray:
    """Distance from each point to an axis-aligned box (0 inside)."""
    p = np.atleast_2d(points)
    d = np.maximum(np.maximum(lo - p, 0.0), p - hi)
    return np.linalg.norm(d, axis=1)


def inside_case(points_case) -> np.ndarray:
    """Which chassis-frame points are inside the open case: within its walls and below the rim's top."""
    p = np.atleast_2d(points_case)
    return (np.abs(p[:, 0]) < HX - SHEET) & (np.abs(p[:, 1]) < HY - SHEET) & (p[:, 2] < RIM) & (p[:, 2] > FOOT)
