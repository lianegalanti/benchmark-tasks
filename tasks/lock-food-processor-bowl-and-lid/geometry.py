"""Dimensions, layout and shape lists for the food-processor task. Pure NumPy; shared by task.py, demo.py and the tests.

Station frame: x forward, y to the robot's left, z up (m). Angles about a vertical axis are measured from +x toward
+y (counter-clockwise seen from above); locking turns are clockwise (negative).

Three frames, all z up:
- B (base): on the drive shaft's axis, at the table top. The base is fixed; B is the station frame shifted to
  ``BASE_XY``.
- W (work bowl): on the bowl's axis, at the underside of its floor. Seated and locked on the base, W = B raised
  ``SEAT_Z`` with no turn. Its "turn" is its yaw about its axis relative to B: 0 = locked against the stops.
- L (lid): on the lid's axis, at the underside of its top. Seated and locked on the bowl, L = W raised ``BOWL_H`` with no
  turn; its turn is its yaw relative to W.

The mechanism (a 9-cup food processor of the common kind, all polycarbonate except the motor base):
- **Bowl on base: a three-lug bayonet.** The base's round coupling boss (110 mm across, 20 mm tall) carries three
  ledges round its top edge. The bowl's foot is a skirt that drops over the boss; three lugs on the skirt's inside
  pass down through the gaps between the ledges and, turned 28 degrees clockwise, slide under them until the lugs
  meet the stops. A sprung plunger in one lug clicks up into a notch in the ledge above it at the stop (the detent).
  The bowl's centre tube slides over the drive shaft first, with 2 mm play, and centres the bowl.
- **Lid on bowl: a four-lug bayonet.** The lid's skirt drops over the bowl's rim; four lugs inside it pass down
  between four ledges on the outside of the rim and, turned 28 degrees clockwise, slide under them to their stops.
  A sprung plunger in one rim ledge drops into a notch in the lug below it at the stop (the detent).
- **Safety interlock.** A sprung switch rod runs down the inside of the bowl's handle; its top stands 9 mm proud of the
  handle's top. The lid has an interlock tab that sticks out over the handle. Only when the lid is turned home does
  the tab (its leading edge a 30-degree ramp) ride over the rod and push it 7 mm down; the rod's lower end then closes
  the motor switch in the base under the handle, which it can reach only when the bowl is locked. The motor runs
  only with both bayonets home.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import station

TABLE = float(station.TABLE_TOP)
DEG = math.pi / 180.0

# -- the motor base (B frame) ------------------------------------------------------------------------------------------
BASE_XY = (0.415, 0.0)           # the drive shaft's axis on the table
HOUSING = ((-0.100, -0.085, 0.0), (0.100, 0.085, 0.095))   # the motor housing (20 x 17 x 9.5 cm)
DECK = 0.095                     # the housing's top (where the boss stands)
BOSS_R = 0.055                   # the coupling boss
SEAT_Z = 0.115                   # the boss's top: the bowl's floor rests on it
SHAFT_R = 0.0125                 # the drive shaft's hub
SHAFT_TOP = 0.165                # its top, 5 cm above the boss, then a 1 cm cone to a point
SHAFT_CONE = 0.010
BASE_MASS = 4.2                  # kg (real: rubber feet; fixed in the simulation)

BASE_LEDGE_R = (0.053, 0.065)    # the three ledges round the boss's top edge (radial span) ...
BASE_LEDGE_Z = (0.106, 0.115)    # ... flush with its top, 9 mm thick; under them the lug channel down to the deck
BASE_STOP_R = (0.053, 0.066)     # each ledge's stop: a post at its clockwise end, deck to boss top
BOWL_LUG_ANGLES = (30.0, 150.0, 270.0)   # degrees, bowl frame (= base frame when locked)

# -- the work bowl (W frame) -------------------------------------------------------------------------------------------
BOWL_R_IN, BOWL_R_OUT = 0.087, 0.090     # wall: 18.0 cm outside, 3 mm polycarbonate
BOWL_H = 0.125                   # rim height above the floor's underside
FLOOR_T = 0.004
TUBE_R = (0.0145, 0.0175)        # the centre tube: 2 mm play round the shaft's hub
TUBE_TOP = 0.105
SKIRT_R = (0.071, 0.074)         # the foot skirt: 6 mm clear of the base's ledges (it cannot perch on them)
SKIRT_Z = (-0.0175, 0.0)
LUG_R = (0.0575, 0.071)          # bowl lugs: 2.5 mm clear of the boss
LUG_Z = (-0.0175, -0.0125)       # seated: 2.5 mm above the deck, 3.5 mm below the ledges
LUG_HALF = 11.0                  # degrees: each lug spans 22 degrees
LEDGE_SPAN = (-11.5, 13.5)       # degrees about a lug's locked centre: a ledge covers the locked lug ...
STOP_SPAN = (-17.0, -11.5)       # ... and its stop sits clockwise of it: the locked lug is 0.5 degree off its stop
TURN = 28.0                      # degrees: the drop position is this far counter-clockwise of locked
BOWL_MASS = 0.55                 # kg, with its handle (polycarbonate)
WALL_N = 32                      # the wall as 32 flat panels; a panel is centred on each rim grip point
WALL_PHASE = 62.0 - 5 * 360.0 / WALL_N   # degrees: a panel centred on the rim grip (62)

# handle and switch rod, on the bowl's right front when locked
HANDLE_ANGLE = 242.0             # degrees, bowl frame: front right when locked, opposite the left hand's rim grip
HANDLE_HALF_W = 0.015            # 30 mm wide
HANDLE_GRIP_R = (0.118, 0.142)   # the upright grip, 24 mm deep
HANDLE_Z = (0.015, BOWL_H - 0.026)
BRIDGE_T = 0.012                 # top and bottom bridges to the wall
ROD_R = 0.1145                   # the switch rod's axis (radius): 5.5 mm outside the lid's skirt ...
ROD_HALF = 0.003                 # ... 6 mm square
ROD_TOP = BOWL_H - 0.017         # its top at rest: 9 mm proud of the handle's top
ROD_LEN = 0.060
ROD_TRAVEL = 0.009
ROD_ACTUATE = 0.005              # pressed this far, it closes the motor switch
ROD_RATE, ROD_PRELOAD = 60.0, 1.0      # N/m, N (1.4 N pressed 7 mm)

# the rim's lid ledges (four), on the outside of the wall
RIM_LEDGE_R = (0.088, 0.098)
RIM_LEDGE_Z = (BOWL_H - 0.0155, BOWL_H - 0.0065)
RIM_STOP_R = (0.088, 0.098)
RIM_RAMP_LEN = 0.008             # each rim ledge's entry end is a lead-in ramp, 8 mm long, rising its full 9 mm
RIM_RAMP_DEG = RIM_RAMP_LEN / ((0.088 + 0.098) / 2) / DEG
RIM_STOP_Z = (BOWL_H - 0.026, BOWL_H - 0.0065)
LID_LUG_ANGLES = (18.75, 108.75, 198.75, 288.75)   # degrees, lid frame (= bowl frame when locked)

# the detents: a sprung plunger with a 45-degree diamond tip (a square bar turned 45 degrees)
DIAMOND = 0.004                  # the bowl detent's tip: a 4 mm square bar (45-degree flanks 2.8 mm tall) ...
DIAMOND_HALF = DIAMOND / math.sqrt(2.0)        # the tip's height above the bar's centre line
LID_DIAMOND = 0.008              # ... the lid detent's: an 8 mm bar (flanks 5.7 mm tall, so a lug riding high still
#                                  meets a flank)
DETENT_PROUD = 0.005             # the tip stands this far past the lug's (or ledge's) face at rest ...
LID_DETENT_PROUD = 0.005           # the lid detent's tip stands 5 mm below its ledge (1.5 mm into the lug's path)
DETENT_TRAVEL = 0.006            # ... and can be pushed this far back
NOTCH_HALF = 0.003               # the notch it clicks into: 6 mm wide
LOCK_PLUNGER_R = (0.059, 0.064)  # in bowl lug 0 (pointing up), under base ledge 0
LID_PLUNGER_R = (0.0935, 0.0975)  # in rim ledge 0 (pointing down), over lid lug 0
DETENT_RATE, DETENT_PRELOAD = 300.0, 3.0      # N/m, N (the bowl's detent; 3.45 N at 1.5 mm)
LID_DETENT_RATE, LID_DETENT_PRELOAD = 300.0, 3.0
DETENT_DAMPING = 2.0             # N s/m
ROD_DAMPING = 2.0
PLUNGER_MASS = 0.01              # kg

# -- the lid (L frame) -------------------------------------------------------------------------------------------------
LID_R = 0.106                    # top and skirt outside radius
LID_T = 0.004
LID_SKIRT_R = (0.103, 0.106)     # 5 mm clear of the rim ledges (it cannot perch on them)
LID_SKIRT_Z = (-0.024, 0.0)
LID_LUG_R = (0.0925, 0.103)      # 2.5 mm clear of the wall
LID_LUG_Z = (-0.024, -0.019)     # seated: 3.5 mm below the rim ledges
LID_MASS = 0.25
TUBE_ANGLE = 270.0               # degrees, lid frame: the feed tube on the lid's right when locked
FEED_R = 0.058                   # the feed tube's centre (radius)
FEED_RADIAL, FEED_TANGENT = 0.040, 0.070   # its outside: 40 x 70 mm (the narrow way radial)
FEED_WALL = 0.003
FEED_TOP = 0.104                 # 10 cm above the lid's top
TAB_ANGLE = HANDLE_ANGLE         # the interlock tab, over the handle when locked
TAB_R = (0.106, 0.124)
TAB_Z = (-0.024, -0.016)         # seated: its underside 2 mm over the handle's top, 7 mm under the rod's top
TAB_FLAT = (-0.006, 0.012)       # m along the tangent (+ = counter-clockwise) at ROD_R: the flat underside ...
RAMP = 30.0 * DEG                # ... and, clockwise of it (its leading side when locking), a 30-degree ramp
RAMP_LEN = (TAB_Z[1] - TAB_Z[0]) / math.tan(RAMP)

# -- start layout (station) --------------------------------------------------------------------------------------------
BOWL0 = (0.20, 0.0)              # the bowl stands on its foot in front of the base, ...
BOWL0_YAW = TURN                 # ... turned to the drop position (handle on its right)
LID0 = (0.27, -0.30)             # the lid stands on its skirt to the right, ...
LID0_YAW = TURN                  # ... also turned to its drop position (feed tube on its right, toward the back)
REST_GAP = 0.0005                # start this far above the table


def bowl_start_z() -> float:
    return TABLE + (-SKIRT_Z[0]) + REST_GAP


def lid_start_z() -> float:
    return TABLE + (-LID_SKIRT_Z[0]) + REST_GAP


def base_origin() -> np.ndarray:
    return np.array([BASE_XY[0], BASE_XY[1], TABLE])


def seated_bowl_origin() -> np.ndarray:
    return base_origin() + [0.0, 0.0, SEAT_Z]


def seated_lid_origin() -> np.ndarray:
    return seated_bowl_origin() + [0.0, 0.0, BOWL_H]


# -- rotations ---------------------------------------------------------------------------------------------------------
def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def q_mul(a, b):
    """Quaternion product (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def q_axis(axis: str, a: float):
    c, s = math.cos(a / 2), math.sin(a / 2)
    return {"x": (c, s, 0.0, 0.0), "y": (c, 0.0, s, 0.0), "z": (c, 0.0, 0.0, s)}[axis]


def q_matrix(wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def yaw_of(r) -> float:
    """Yaw (rad) of a rotation matrix about z."""
    r = np.asarray(r, float)
    return math.atan2(float(r[1, 0]), float(r[0, 0]))


def tilt_of(r) -> float:
    """Angle (rad) between a rotation's z axis and the vertical."""
    return math.acos(max(-1.0, min(1.0, float(np.asarray(r, float)[2, 2]))))


def xyzw(r) -> tuple:
    from yam_arena.ik import matrix_to_quat_xyzw

    return tuple(float(v) for v in matrix_to_quat_xyzw(np.asarray(r, float)))


# -- shapes: (centre, size, colour, rotation w x y z) boxes in a body frame -------------------------------------------
CLEAR_PC = (0.86, 0.90, 0.93)    # clear polycarbonate (drawn light grey-blue)
GREY_PC = (0.55, 0.58, 0.62)
WHITE = (0.94, 0.94, 0.92)
DARK = (0.13, 0.13, 0.14)
STEEL = (0.62, 0.64, 0.66)
RED = (0.80, 0.18, 0.15)


def _f(v) -> tuple:
    return tuple(float(x) for x in v)


def radial_box(angle_deg: float, r0: float, r1: float, z0: float, z1: float, tangent: float, color,
               t_off: float = 0.0, tilt=None):
    """A box between radii ``r0`` and ``r1`` (along the radius at ``angle_deg``), ``tangent`` long across it, centred
    ``t_off`` along the tangent (+ = counter-clockwise), from z0 to z1; optionally turned ``tilt`` (rad) about its radial
    axis. Its inner face is flat at ``r0``."""
    a = angle_deg * DEG
    radial, tang = np.array([math.cos(a), math.sin(a), 0.0]), np.array([-math.sin(a), math.cos(a), 0.0])
    c = radial * (r0 + r1) / 2 + tang * t_off + np.array([0.0, 0.0, (z0 + z1) / 2])
    q = q_axis("z", a)
    if tilt:
        q = q_mul(q, q_axis("x", tilt))
    return (_f(c), _f((r1 - r0, tangent, z1 - z0)), color, _f(q))


def arc_boxes(a0: float, a1: float, r0: float, r1: float, z0: float, z1: float, color, step: float = 12.0):
    """Boxes covering the arc a0..a1 (degrees) between radii r0 and r1: each segment's outer corners on r1's chord
    ends, so the arc is covered without gaps (the corners stick out at most r1 (1/cos(step/2) - 1))."""
    n = max(1, math.ceil((a1 - a0) / step - 1e-9))
    d = (a1 - a0) / n
    out = []
    for i in range(n):
        mid = a0 + (i + 0.5) * d
        out.append(radial_box(mid, r0, r1, z0, z1, 2 * r1 * math.tan(d / 2 * DEG), color))
    return out


def ring_boxes(r0: float, r1: float, z0: float, z1: float, n: int, color, phase: float = 0.0):
    """A closed ring of ``n`` panels between r0 and r1 (each panel's inner face flat at r0), centred at phase + k
    360/n degrees."""
    d = 360.0 / n
    return [radial_box(phase + i * d, r0, r1, z0, z1, 2 * r1 * math.tan(d / 2 * DEG), color) for i in range(n)]


def disc_boxes(r0: float, r1: float, z0: float, z1: float, n: int, color, phase: float = 0.0):
    return ring_boxes(r0, r1, z0, z1, n, color, phase)


def diamond_box(angle_deg: float, r0: float, r1: float, tip_z: float, up: bool, color, side: float = DIAMOND):
    """The detent plunger's tip: a ``side`` square bar along the radius, turned 45 degrees, its edge ``tip_z``
    (pointing up if ``up``, else down)."""
    half = side / math.sqrt(2.0)
    zc = tip_z - half if up else tip_z + half
    return radial_box(angle_deg, r0, r1, zc - side / 2, zc + side / 2, side, color, tilt=45.0 * DEG)


# -- the base -----------------------------------------------------------------------------------------------------------
def base_boxes():
    """The motor base, B frame: housing and the ledges and stops on the boss. (The boss and shaft are cylinders.)"""
    (x0, y0, z0), (x1, y1, z1) = HOUSING
    out = [(_f(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)), _f((x1 - x0, y1 - y0, z1 - z0)), WHITE,
            (1.0, 0.0, 0.0, 0.0))]
    # control paddles on the housing's front (visual, flush): they still collide, as on a real base
    out.append((_f((x0 - 0.002, -0.03, 0.05)), (0.004, 0.025, 0.03), DARK, (1.0, 0.0, 0.0, 0.0)))
    out.append((_f((x0 - 0.002, 0.03, 0.05)), (0.004, 0.025, 0.03), DARK, (1.0, 0.0, 0.0, 0.0)))
    for k, lam in enumerate(BOWL_LUG_ANGLES):
        lo, hi = lam + LEDGE_SPAN[0], lam + LEDGE_SPAN[1]
        if k == 0:     # ledge 0 has the detent's notch over the locked plunger
            nh = NOTCH_HALF / ((LOCK_PLUNGER_R[0] + LOCK_PLUNGER_R[1]) / 2) / DEG
            out += arc_boxes(lo, lam - nh, *BASE_LEDGE_R, *BASE_LEDGE_Z, GREY_PC, step=6.0)
            out += arc_boxes(lam + nh, hi, *BASE_LEDGE_R, *BASE_LEDGE_Z, GREY_PC, step=6.0)
        else:
            out += arc_boxes(lo, hi, *BASE_LEDGE_R, *BASE_LEDGE_Z, GREY_PC, step=6.0)
        out += arc_boxes(lam + STOP_SPAN[0], lam + STOP_SPAN[1], *BASE_STOP_R, DECK, SEAT_Z, DARK, step=6.0)
    return out


def base_cylinders():
    """(centre, radius, height, colour, axis) cylinders and (centre, radius, height, colour) cones, B frame."""
    cyl = [((0.0, 0.0, (DECK + SEAT_Z) / 2), BOSS_R, SEAT_Z - DECK, GREY_PC, "Z"),
           ((0.0, 0.0, (SEAT_Z + SHAFT_TOP) / 2), SHAFT_R, SHAFT_TOP - SEAT_Z, STEEL, "Z")]
    cones = [((0.0, 0.0, SHAFT_TOP + SHAFT_CONE / 2), SHAFT_R, SHAFT_CONE, STEEL)]
    return cyl, cones


# -- the bowl -----------------------------------------------------------------------------------------------------------
def bowl_boxes():
    """The work bowl's root body, W frame."""
    out = []
    out += ring_boxes(BOWL_R_IN, BOWL_R_OUT, FLOOR_T, BOWL_H, WALL_N, CLEAR_PC, WALL_PHASE)
    out += disc_boxes(TUBE_R[1] - 0.001, BOWL_R_OUT, 0.0, FLOOR_T, 24, CLEAR_PC)
    out += ring_boxes(TUBE_R[0], TUBE_R[1], 0.0, TUBE_TOP, 12, CLEAR_PC)
    out += ring_boxes(SKIRT_R[0], SKIRT_R[1], SKIRT_Z[0], SKIRT_Z[1], WALL_N, CLEAR_PC, WALL_PHASE)
    for k, lam in enumerate(BOWL_LUG_ANGLES):
        out += arc_boxes(lam - LUG_HALF, lam + LUG_HALF, LUG_R[0], LUG_R[1] + 0.001, *LUG_Z, GREY_PC, step=6.0)
    # the rim's four lid ledges, their stops, and ledge 0 holds the lid detent's plunger
    for mu in LID_LUG_ANGLES:
        end = mu + LEDGE_SPAN[1]
        out += arc_boxes(mu + LEDGE_SPAN[0], end - RIM_RAMP_DEG, RIM_LEDGE_R[0], RIM_LEDGE_R[1], *RIM_LEDGE_Z,
                         CLEAR_PC, step=6.0)
        out.append(rim_ramp_box(end))
        out += arc_boxes(mu + STOP_SPAN[0], mu + STOP_SPAN[1], RIM_STOP_R[0], RIM_STOP_R[1], *RIM_STOP_Z, GREY_PC,
                         step=6.0)
    # the handle: an upright grip joined to the wall by a top and a bottom bridge
    w = 2 * HANDLE_HALF_W
    out.append(radial_box(HANDLE_ANGLE, *HANDLE_GRIP_R, *HANDLE_Z, w, GREY_PC))
    out.append(radial_box(HANDLE_ANGLE, BOWL_R_IN + 0.001, HANDLE_GRIP_R[0], HANDLE_Z[1] - BRIDGE_T, HANDLE_Z[1], w,
                          GREY_PC))
    out.append(radial_box(HANDLE_ANGLE, BOWL_R_IN + 0.001, HANDLE_GRIP_R[0], HANDLE_Z[0], HANDLE_Z[0] + BRIDGE_T, w,
                          GREY_PC))
    return out


def rim_ramp_box(end_deg: float):
    """The lead-in at a rim ledge's entry end (its counter-clockwise end, where a lid lug comes in): a slab whose
    underside rises from the ledge's underside, RIM_RAMP_LEN before the end, to the ledge's top at the end, so a lug
    riding a little high is cammed down under the ledge instead of catching on its end."""
    z0, z1 = RIM_LEDGE_Z
    beta = math.atan2(z1 - z0, RIM_RAMP_LEN)
    th = 0.0025
    mid = np.array([-RIM_RAMP_LEN / 2, (z0 + z1) / 2])          # (tangent from the end, z) of the underside's middle
    c = mid + np.array([-math.sin(beta), math.cos(beta)]) * th / 2
    length = math.hypot(RIM_RAMP_LEN, z1 - z0)
    return radial_box(end_deg, *RIM_LEDGE_R, c[1] - th / 2, c[1] + th / 2, length, CLEAR_PC, t_off=c[0], tilt=beta)


def rim_ramp_underside_z(t: float) -> float:
    """Height (W frame) of a rim ledge's underside at tangent position t (m) measured back from its entry end."""
    if t >= 0.0:
        return math.inf
    if t > -RIM_RAMP_LEN:
        return RIM_LEDGE_Z[0] + (t + RIM_RAMP_LEN) * (RIM_LEDGE_Z[1] - RIM_LEDGE_Z[0]) / RIM_RAMP_LEN
    return RIM_LEDGE_Z[0]


def rod_boxes():
    """The switch rod, in its link frame (= W at rest): a 6 mm square rod, its top ROD_TOP."""
    return [radial_box(HANDLE_ANGLE, ROD_R - ROD_HALF, ROD_R + ROD_HALF, ROD_TOP - ROD_LEN, ROD_TOP, 2 * ROD_HALF,
                       RED)]


def lock_plunger_boxes():
    """The bowl's detent plunger (in lug 0, pointing up), link frame = W at rest."""
    lam = BOWL_LUG_ANGLES[0]
    tip = LUG_Z[1] + DETENT_PROUD
    return [diamond_box(lam, *LOCK_PLUNGER_R, tip, True, DARK),
            radial_box(lam, *LOCK_PLUNGER_R, LUG_Z[0] + 0.001, tip - DIAMOND_HALF, DIAMOND * 0.7, DARK)]


def lid_plunger_boxes():
    """The lid detent's plunger (in rim ledge 0, pointing down), link frame = W at rest."""
    mu = LID_LUG_ANGLES[0]
    tip = RIM_LEDGE_Z[0] - LID_DETENT_PROUD
    return [diamond_box(mu, *LID_PLUNGER_R, tip, False, DARK, LID_DIAMOND),
            radial_box(mu, *LID_PLUNGER_R, tip + LID_DIAMOND / math.sqrt(2.0), RIM_LEDGE_Z[1], LID_DIAMOND * 0.7,
                       DARK)]


# -- the lid ------------------------------------------------------------------------------------------------------------
def tab_boxes():
    """The interlock tab, L frame: the flat part and its ramp (leading, clockwise side). Positions along the tab are
    measured along its tangent line; the rod meets that line at t = 0 when the lid is locked."""
    a = TAB_ANGLE
    f0, f1 = TAB_FLAT
    flat = radial_box(a, *TAB_R, *TAB_Z, f1 - f0, GREY_PC, t_off=(f0 + f1) / 2)
    # the ramp: a 2.5 mm slab whose underside runs from the flat's leading edge (at TAB_Z[0]) up at 30 degrees
    t = 0.0025
    length = RAMP_LEN / math.cos(RAMP)
    under_mid = np.array([f0 - RAMP_LEN / 2, (TAB_Z[0] + TAB_Z[1]) / 2])     # (tangent, z) of the underside
    normal = np.array([math.sin(RAMP), math.cos(RAMP)])   # into the slab from its underside, (tangent, z)
    c = under_mid + normal * t / 2
    ramp = radial_box(a, *TAB_R, c[1] - t / 2, c[1] + t / 2, length, GREY_PC, t_off=c[0], tilt=-RAMP)
    # a web joining the tab to the skirt above the ramp
    web = radial_box(a, LID_SKIRT_R[0], TAB_R[0], TAB_Z[0] + 0.002, LID_SKIRT_Z[1], 0.024, GREY_PC)
    return [flat, ramp, web]


def ramp_underside_z(t: float) -> float:
    """Height (L frame) of the tab's underside at tangent position t (m, at the rod's radius); +inf off the tab."""
    if TAB_FLAT[0] <= t <= TAB_FLAT[1]:
        return TAB_Z[0]
    if TAB_FLAT[0] - RAMP_LEN <= t < TAB_FLAT[0]:
        return TAB_Z[0] + (TAB_FLAT[0] - t) * math.tan(RAMP)
    return math.inf


def lid_boxes():
    """The lid, L frame: top, skirt, lugs (lug 0 notched for the detent), feed tube and interlock tab."""
    out = []
    out += disc_boxes(0.0, LID_R, 0.0, LID_T, 24, CLEAR_PC)
    out += ring_boxes(LID_SKIRT_R[0], LID_SKIRT_R[1], LID_SKIRT_Z[0], LID_SKIRT_Z[1], 40, CLEAR_PC)
    nh = NOTCH_HALF / ((LID_PLUNGER_R[0] + LID_PLUNGER_R[1]) / 2) / DEG
    for k, mu in enumerate(LID_LUG_ANGLES):
        spans = [(mu - LUG_HALF, mu - nh), (mu + nh, mu + LUG_HALF)] if k == 0 else [(mu - LUG_HALF, mu + LUG_HALF)]
        for a0, a1 in spans:
            out += arc_boxes(a0, a1, LID_LUG_R[0], LID_LUG_R[1] + 0.001, *LID_LUG_Z, GREY_PC, step=6.0)
    # the feed tube: four walls round a 40 x 70 mm outline (radial x tangential), 10 cm tall
    a = TUBE_ANGLE
    hr, ht, w = FEED_RADIAL / 2, FEED_TANGENT / 2, FEED_WALL
    z0, z1 = LID_T, FEED_TOP
    out.append(radial_box(a, FEED_R - hr, FEED_R - hr + w, z0, z1, 2 * ht, CLEAR_PC))
    out.append(radial_box(a, FEED_R + hr - w, FEED_R + hr, z0, z1, 2 * ht, CLEAR_PC))
    out.append(radial_box(a, FEED_R - hr + w, FEED_R + hr - w, z0, z1, w, CLEAR_PC, t_off=ht - w / 2))
    out.append(radial_box(a, FEED_R - hr + w, FEED_R + hr - w, z0, z1, w, CLEAR_PC, t_off=-ht + w / 2))
    out += tab_boxes()
    return out


def feed_tube_centre_L(z: float = 0.0) -> np.ndarray:
    a = TUBE_ANGLE * DEG
    return np.array([FEED_R * math.cos(a), FEED_R * math.sin(a), z])


# -- the mechanism, in numbers (for the tests and the README) ------------------------------------------------------------
def arc_gap_deg(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Angular gap between two arcs (degrees, positive if apart, negative if they overlap), on a circle."""
    def norm(x):
        return (x + 180.0) % 360.0 - 180.0
    ca, ha = (a[0] + a[1]) / 2, (a[1] - a[0]) / 2
    cb, hb = (b[0] + b[1]) / 2, (b[1] - b[0]) / 2
    return abs(norm(ca - cb)) - ha - hb


def lugs_clear_of_ledges(turn_deg: float, lugs, half=LUG_HALF) -> float:
    """Smallest angular gap (degrees) between the lugs, turned ``turn_deg`` from locked, and the ledges and stops."""
    worst = math.inf
    for lam in lugs:
        lug = (lam + turn_deg - half, lam + turn_deg + half)
        for mu in lugs:
            worst = min(worst, arc_gap_deg(lug, (mu + STOP_SPAN[0], mu + LEDGE_SPAN[1])))
    return worst


def lug_under_ledge(turn_deg: float) -> bool:
    """Every lug wholly under its ledge (turned ``turn_deg`` from locked)."""
    return bool(LEDGE_SPAN[0] <= turn_deg - LUG_HALF and turn_deg + LUG_HALF <= LEDGE_SPAN[1])


STOP_TURN = STOP_SPAN[1] + LUG_HALF   # degrees: the turn at which the lugs meet their stops (-0.5)


# -- outlines for the "let go" check: solid cylinders and boxes, in the body's frame --------------------------------------
def bowl_outline():
    """(radius, z0, z1) cylinders and boxes (centre, half sizes, rotation) round the bowl, W frame."""
    cyls = [(RIM_LEDGE_R[1], SKIRT_Z[0], BOWL_H)]
    a = HANDLE_ANGLE * DEG
    r = rot_z(a)
    c = r @ np.array([(BOWL_R_OUT + HANDLE_GRIP_R[1]) / 2, 0.0, (HANDLE_Z[0] + ROD_TOP) / 2])
    boxes = [(c, np.array([(HANDLE_GRIP_R[1] - BOWL_R_OUT) / 2, HANDLE_HALF_W, (ROD_TOP - HANDLE_Z[0]) / 2]), r)]
    return cyls, boxes


def lid_outline():
    cyls = [(LID_R, LID_SKIRT_Z[0], LID_T)]
    a = TUBE_ANGLE * DEG
    r = rot_z(a)
    c = r @ np.array([FEED_R, 0.0, (LID_T + FEED_TOP) / 2])
    boxes = [(c, np.array([FEED_RADIAL / 2, FEED_TANGENT / 2, (FEED_TOP - LID_T) / 2]), r)]
    a = TAB_ANGLE * DEG
    r = rot_z(a)
    t0, t1 = TAB_FLAT[0] - RAMP_LEN, TAB_FLAT[1]
    c = r @ np.array([(LID_R + TAB_R[1]) / 2, (t0 + t1) / 2, (TAB_Z[0] + TAB_Z[1]) / 2])
    boxes.append((c, np.array([(TAB_R[1] - LID_R) / 2, (t1 - t0) / 2, (TAB_Z[1] - TAB_Z[0]) / 2]), r))
    return cyls, boxes


def dist_to_outline(points, cyls, boxes) -> float:
    """Smallest distance from points (rows, body frame) to the outline (0 inside)."""
    p = np.atleast_2d(np.asarray(points, float))
    best = math.inf
    for radius, z0, z1 in cyls:
        dr = np.maximum(np.linalg.norm(p[:, :2], axis=1) - radius, 0.0)
        dz = np.maximum(np.maximum(z0 - p[:, 2], p[:, 2] - z1), 0.0)
        best = min(best, float(np.sqrt(dr * dr + dz * dz).min()))
    for c, h, r in boxes:
        local = (p - c) @ r
        d = np.maximum(np.abs(local) - h, 0.0)
        best = min(best, float(np.linalg.norm(d, axis=1).min()))
    return best


def box_points(c, r, h, n: int = 4) -> np.ndarray:
    """Points on a box's surface grid (centre, rotation, half sizes)."""
    u = np.linspace(-1, 1, n)
    grid = np.array([[a, b, d] for a in u for b in u for d in u])
    grid = grid[np.abs(grid).max(axis=1) == 1.0]
    return np.asarray(c) + (grid * np.asarray(h)) @ np.asarray(r).T


def all_boxes_extent(boxes):
    """(lo, hi) corners of a list of boxes (body frame)."""
    pts = []
    for c, s, _, q in boxes:
        r = q_matrix(q)
        h = np.asarray(s) / 2
        corners = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]) * h
        pts.append(np.asarray(c) + corners @ r.T)
    pts = np.vstack(pts)
    return pts.min(axis=0), pts.max(axis=0)
