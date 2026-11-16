"""Dimensions of the drum-brake corner and the adjusting spoon, the adjuster's model (clicks, clearance, drum drag), and
the box maths shared by make_assets.py, task.py, demo.py and the tests. Pure Python (NumPy).

Frames
- Station: x forward, y to the robot's left, z up (m); the table top is at ``TABLE``.
- B, the corner's frame (the backing plate's): origin on the axle's axis, in the backing plate's outboard face; axes
  parallel to the station's. The axle runs along y: the backing plate faces the robot's right (inboard, -y) and the
  drum's outer face the robot's left (+y). The corner is fixed, so B is the station frame shifted by ``B0``; the
  scorer still measures everything from the corner's measured pose.
- S, the spoon's frame: origin at the centre of its handle, x along the spoon toward the blade's tip, y across its
  width, z through its thickness (up when it lies on the table).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from yam_arena import station

    TABLE = float(station.TABLE_TOP)
except Exception:  # pragma: no cover - make_assets.py may run without yam_arena
    TABLE = 0.7475

# -- the corner (B frame) ----------------------------------------------------------------------------------------------
B0 = np.array([0.30, -0.05, TABLE + 0.18])     # station position of B's origin: axle axis 18 cm above the table

PLATE_R = 0.135          # backing plate: 270 mm across, 3 mm pressed steel, from y = -3 mm to 0
PLATE_T = 0.003
LIP_R = (0.130, 0.135)   # its dust lip: a 5 mm rim standing 10 mm outboard round the drum's open edge
LIP_Y = 0.010

DRUM_R_IN = 0.1143       # 9.00 in (228.6 mm) brake drum
DRUM_R_OUT = 0.128
DRUM_Y = (0.003, 0.068)  # braking band: open edge 3 mm outboard of the plate, 65 mm deep
FACE_Y = (0.068, 0.076)  # outer face (8 mm)
CAP = (0.025, 0.006)     # hub centre cap: radius, height
STUD_R = 0.006           # five M12 wheel studs on a 114.3 mm circle, standing 45 mm proud of the face
STUD_PCD_R = 0.05715
STUD_LEN = 0.045
N_STUDS = 5
DRUM_MASS = 6.0          # kg (cast-iron 9 in drum)
N_SEG = 24               # the dust lip as 24 segments
N_BAND = 240             # the drum's braking band as 240 flat segments (1.5 deg: 0.01 mm from round)

# Star wheel: the self-adjuster's toothed wheel on the adjusting screw, which spans the shoes' lower ends across the
# bottom of the brake. Its axis runs along x (tangent at the bottom), 42 mm outboard of the plate.
SW = np.array([0.0, 0.042, -0.085])
N_TEETH = 10
TOOTH_R = (0.0095, 0.015)   # teeth from the root (19 mm) to the tips (30 mm across)
TOOTH_T = 0.002             # tooth thickness (tangential)
SW_W = 0.006                # wheel thickness along its axis
PITCH = 2 * math.pi / N_TEETH
SW_MASS = 0.03
SCREW_R, SOCKET_R = 0.0045, 0.006
SCREW_X = (0.0035, 0.040)   # the screw (+x) and its socket (-x) either side of the wheel, to the shoe forks

# Adjusting slot through the backing plate below the star wheel (its rubber plug is out): 22 mm wide (x) and 16 mm
# tall, its bottom edge 1.2 mm below the wheel's axis.
SLOT_HX = 0.011
SLOT_Z = (SW[2] - 0.0012, SW[2] - 0.0012 + 0.016)

# Shoes (fixed in the model; see fidelity): two 50 mm wide shoes, lining 5 mm, table 2 mm, web 4 mm.
# Ford's linings: 1.75 in wide; primary 6.10 in and secondary 8.65 in long (arcs of 78 and 111 degrees at the drum)
SHOE_Y = (0.0145, 0.059)
WEB_Y = (0.035, 0.039)
WEB_R0 = 0.075

# -- the adjuster's model and the brake's calibration --------------------------------------------------------------
# The real assembly: a 1979-1993 Ford Mustang 9 in duo-servo rear drum brake (drum bore 9.000 in, 1.75 in linings;
# Ford workshop manual). Sources for every number are in the README ("Calibration").
TIGHTEN = -1.0           # sign of the star joint's angle (about B's +x) that spreads the shoes: tip pushes teeth up
C0_MM = 0.975            # shoe-to-drum clearance at the start (mm, each lining's crown to the drum)
CLICK_MM = 0.05          # each click moves each shoe's lining 0.05 mm out (the drive target): 20 clicks per mm
N_CONTACT = 19.5         # clicks at which the linings reach the drum (C0_MM / CLICK_MM): touching at 20
N_MAX = 40               # the screw's travel (joint limit)
DEMO_CLICKS = 25         # the demo's fixed count


def target_mm(n: float) -> float:
    """Each shoe's commanded outward travel after ``n`` clicks (mm): where the adjuster would put the lining if the
    drum weren't there."""
    return CLICK_MM * n


# The shoes are pressed onto the drum through a compliance (the shoe, its lining and the return springs): each
# shoe's drive is a spring K_SHOE toward its commanded travel. Once a lining touches the drum, the drum stops it and
# the spring's stretch is the normal force N on the drum (read in the sim from the shoe joint's measured position).
# The drum's turning torque is MU x R x (N_lead + N_trail) plus the hub's own drag.
MU_LINING = 0.40         # organic drum lining, SAE J866 edge code "FF" (0.35-0.45): the middle
K_SHOE = 93.0e3          # N/m per shoe: model compliance, sized so the light-drag window spans 7 clicks
D_SHOE = 300.0           # N s/m
SHOE_FORCE_MAX = 600.0   # N: the drive's cap
T_BEAR = 0.15            # N m: hub bearing and seal drag (shoes clear), as the drum joint's damping at OMEGA_REF
OMEGA_REF = math.radians(30.0)
BEAR_DAMPING = T_BEAR / OMEGA_REF
# Light drag, as a turning torque at the drum (or a pull at a wheel stud, on the 114.3 mm circle):
T_LO = 1.0               # N m (17.5 N at a stud): clearly felt: over six times the hub's free drag
T_HI = 3.9               # N m (69 N at a stud): the most residual drag a service manual accepts
T_LOCK = 4.8             # N m: past this the brake is dragging hard; latched (the pawl stops it backing off)
STUD_ARM = 0.05715       # m: torque = pull at a stud x this


def turning_torque(f_lead: float, f_trail: float) -> float:
    """Torque (N m) to turn the drum by hand (at OMEGA_REF) with these lining-to-drum normal forces (N)."""
    return T_BEAR + MU_LINING * DRUM_R_IN * (max(f_lead, 0.0) + max(f_trail, 0.0))


def shoe_force(target_m: float, travel_m: float) -> float:
    """A shoe's normal force on the drum (N) from its commanded and measured outward travel (m)."""
    return max(0.0, K_SHOE * (target_m - travel_m))


def ideal_torque(n: float) -> float:
    """The model's torque after ``n`` clicks with the drum perfectly rigid and round (for the tests and the README)."""
    f = shoe_force(target_mm(n) / 1000, min(target_mm(n), C0_MM) / 1000)
    return turning_torque(f, f)


# Star wheel detent: the self-adjuster's spring-loaded lever rides on the teeth. Turning the wheel one tooth takes
# CLICK_TORQUE; it can't be turned back (the lever acts as a pawl).
CLICK_TORQUE = 0.10      # N m
DETENT_K = 0.6           # N m/rad drive spring toward the nearest notch (capped at CLICK_TORQUE)
DETENT_D = 0.003         # N m s/rad
PAWL = 0.6               # teeth: the pawl stops the wheel this far behind its last notch (past the half-way click)

# -- the spoon (S frame) -------------------------------------------------------------------------------------------
HANDLE = (0.100, 0.024, 0.018)        # moulded handle: length, width, height (S origin at its centre)
BLADE = (0.125, 0.012, 0.0016)        # flat spring-steel blade beyond the handle (1/16 in thick)
BLADE_Z = -HANDLE[2] / 2 + BLADE[2] / 2   # the blade is flush with the handle's flat underside
TIP_X = HANDLE[0] / 2 + BLADE[0]          # 0.175: the blade's end
SPOON_LEN = HANDLE[0] + BLADE[0]          # 0.225 m overall
SPOON_MASS = 0.09
GRASP_Z = HANDLE[2] / 2                   # the TCP sits on the handle's top face when holding it

# where it lies: flat on the table, along +y (blade toward the corner), right of the corner, in front of the right arm
SPOON_START = np.array([0.20, -0.265, TABLE + HANDLE[2] / 2 + 0.0005])
YAW90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # S x -> station +y


def b_to_world(p) -> np.ndarray:
    return np.asarray(p, float) + B0


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_wxyz(r: np.ndarray) -> tuple[float, float, float, float]:
    r = np.asarray(r, float)
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-6:
        return (w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w))
    i = int(np.argmax(np.diag(r)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(max(0.0, 1.0 + r[i, i] - r[j, j] - r[k, k])) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[0] = (r[k, j] - r[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (r[j, i] + r[i, j]) / s
    q[1 + k] = (r[k, i] + r[i, k]) / s
    return tuple(float(v) for v in q)


def quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = quat_wxyz(r)
    return (x, y, z, w)


# -- the corner's shapes -----------------------------------------------------------------------------------------------
# Each box: (name, centre, size, rotation matrix) in its link's frame. Each cylinder: (name, centre, radius, length,
# axis "x"/"y"/"z").


def plate_boxes():
    """The backing plate (3 mm, 270 mm across) as horizontal strips, with the adjusting slot cut out of it, and its
    dust lip as 24 segments."""
    out = []
    edges = sorted({*np.round(np.arange(-PLATE_R, PLATE_R + 1e-9, 0.010), 6), *SLOT_Z})
    edges = [e for e in edges if -PLATE_R <= e <= PLATE_R]
    yc = -PLATE_T / 2
    for z0, z1 in zip(edges[:-1], edges[1:]):
        zz = max(abs(z0), abs(z1))
        if z0 < 0 < z1:
            zz = 0.0
        half = math.sqrt(max(PLATE_R ** 2 - zz ** 2, 0.0))
        if half < 0.01:
            half = 0.01
        zc, hz = (z0 + z1) / 2, z1 - z0
        if z0 >= SLOT_Z[0] - 1e-9 and z1 <= SLOT_Z[1] + 1e-9:
            for s in (-1, 1):      # either side of the slot
                x0, x1 = SLOT_HX, half
                out.append((f"plate_{len(out)}", (s * (x0 + x1) / 2, yc, zc), (x1 - x0, PLATE_T, hz), np.eye(3)))
        else:
            out.append((f"plate_{len(out)}", (0.0, yc, zc), (2 * half, PLATE_T, hz), np.eye(3)))
    for i in range(N_SEG):
        a = 2 * math.pi * i / N_SEG
        r = (LIP_R[0] + LIP_R[1]) / 2
        w = 2 * LIP_R[1] * math.tan(math.pi / N_SEG)
        out.append((f"lip_{i}", (r * math.cos(a), LIP_Y / 2, r * math.sin(a)), (LIP_R[1] - LIP_R[0], LIP_Y, w),
                    rot_y(-a)))
    return out


def _arc(name, r0, r1, y, a0, a1, step):
    """Boxes along an arc in the x-z plane (angle from +x toward +z), radially r0..r1, axially y."""
    out = []
    n = max(1, int(math.ceil((a1 - a0) / step)))
    da = (a1 - a0) / n
    for i in range(n):
        a = a0 + (i + 0.5) * da
        r = (r0 + r1) / 2
        w = 2 * r1 * math.tan(da / 2)
        out.append((f"{name}_{i}", (r * math.cos(a), (y[0] + y[1]) / 2, r * math.sin(a)), (r1 - r0, y[1] - y[0], w),
                    rot_y(-a)))
    return out


# (link and joint name, middle angle, sign of its outward travel along B's x, half the lining's arc)
SHOES = (("shoe_primary", 0.0, 1.0, 6.10 * 0.0254 / 2 / DRUM_R_IN),
         ("shoe_secondary", math.pi, -1.0, 8.65 * 0.0254 / 2 / DRUM_R_IN))
PAD_R = 0.005            # each lining's contact: three 5 mm spheres along its crown line, their outer points on the
PAD_ANGLES = (-0.5, 0.0, 0.5)   # lining's face (rad from the shoe's middle), axially in the lining's middle
SHOE_MASS = 0.6          # kg each


def shoe_visual_boxes(side: str):
    """One shoe (its link's frame = B at zero travel): lining, table and web as arc segments, for looks only; the
    lining's face is C0_MM inside the drum."""
    mid, span = {n: (m, h) for n, m, _, h in SHOES}[side]
    r_out = DRUM_R_IN - C0_MM / 1000
    a0, a1 = mid - span, mid + span
    out = _arc(f"lining_{side}", r_out - 0.005, r_out - 0.0003, SHOE_Y, a0, a1, math.radians(8))
    out += _arc(f"table_{side}", r_out - 0.007, r_out - 0.005, SHOE_Y, a0, a1, math.radians(8))
    out += _arc(f"web_{side}", WEB_R0, r_out - 0.007, WEB_Y, a0, a1, math.radians(8))
    return out


def shoe_pads(side: str):
    """The lining's contact spheres (name, centre, radius) in the shoe link's frame: their outer points lie on the
    lining's face, C0_MM inside the drum at zero travel."""
    mid = {n: m for n, m, _, _ in SHOES}[side]
    r = DRUM_R_IN - C0_MM / 1000 - PAD_R
    y = (SHOE_Y[0] + SHOE_Y[1]) / 2
    return [(f"pad_{i}", (r * math.cos(mid + a), y, r * math.sin(mid + a)), PAD_R) for i, a in enumerate(PAD_ANGLES)]


def base_boxes():
    """Fixed parts inside the drum: the wheel cylinder at the top, between the shoes' upper ends."""
    return [("wheel_cylinder", (0.0, 0.035, 0.083), (0.070, 0.028, 0.022), np.eye(3))]


def shoe_cylinders():
    """The adjusting screw (+x) and its socket (-x) either side of the star wheel, and the anchor pin."""
    y, z = SW[1], SW[2]
    x0, x1 = SCREW_X
    return [("screw", ((x0 + x1) / 2, y, z), SCREW_R, x1 - x0, "x"),
            ("socket", (-(x0 + x1) / 2, y, z), SOCKET_R, x1 - x0, "x"),
            ("anchor_pin", (0.0, 0.035, 0.103), 0.008, 0.05, "y")]


def stand_boxes():
    """The bench stand (fixed): an upright post behind the corner, an arm forward to the axle stub."""
    return [("stand_foot", (0.23, -0.045, -B0[2] + TABLE + 0.006), (0.12, 0.12, 0.012), np.eye(3)),
            ("stand_post", (0.23, -0.045, (-B0[2] + TABLE + 0.012) / 2), (0.05, 0.05, B0[2] - TABLE - 0.012),
             np.eye(3)),
            ("stand_arm", (0.115, -0.045, 0.0), (0.18, 0.036, 0.04), np.eye(3))]


def stand_cylinders():
    return [("axle_stub", (0.0, -PLATE_T - 0.03, 0.0), 0.035, 0.06, "y")]


def drum_boxes():
    """The drum's braking band as N_BAND flat segments (drum link frame = B at drum angle 0). Each segment's inner face
    is a chord of the 228.6 mm bore, so the band is round to 0.01 mm."""
    out = []
    da = 2 * math.pi / N_BAND
    t = DRUM_R_OUT - DRUM_R_IN
    for i in range(N_BAND):
        a = (i + 0.5) * da
        r = DRUM_R_IN + t / 2
        w = 2 * DRUM_R_IN * math.tan(da / 2) + 0.0002   # chord width, overlapping its neighbours by 0.1 mm
        out.append((f"band_{i}", (r * math.cos(a), (DRUM_Y[0] + DRUM_Y[1]) / 2, r * math.sin(a)),
                    (t, DRUM_Y[1] - DRUM_Y[0], w), rot_y(-a)))
    return out


def stud_angle(i: int) -> float:
    """Angle of stud ``i`` from +x toward +z at drum angle 0 (stud 0 at the top)."""
    return math.pi / 2 + 2 * math.pi * i / N_STUDS


def drum_cylinders():
    out = [("face", (0.0, (FACE_Y[0] + FACE_Y[1]) / 2, 0.0), DRUM_R_OUT, FACE_Y[1] - FACE_Y[0], "y"),
           ("cap", (0.0, FACE_Y[1] + CAP[1] / 2, 0.0), CAP[0], CAP[1], "y")]
    for i in range(N_STUDS):
        a = stud_angle(i)
        out.append((f"stud_{i}", (STUD_PCD_R * math.cos(a), FACE_Y[1] + STUD_LEN / 2, STUD_PCD_R * math.sin(a)),
                    STUD_R, STUD_LEN, "y"))
    return out


def star_boxes():
    """The star wheel's teeth (star link frame: origin at SW, axes as B, at joint angle 0). At angle 0 a tooth gap is
    centred on -y (toward the slot): teeth at +-18 degrees either side of it."""
    out = []
    r = (TOOTH_R[0] + TOOTH_R[1]) / 2
    for k in range(N_TEETH):
        a = math.pi + PITCH / 2 + k * PITCH          # angle in the y-z plane from +y toward +z
        out.append((f"tooth_{k}", (0.0, r * math.cos(a), r * math.sin(a)), (SW_W, TOOTH_R[1] - TOOTH_R[0], TOOTH_T),
                    rot_x(a)))
    return out


def star_cylinders():
    return [("hub", (0.0, 0.0, 0.0), 0.010, SW_W, "x")]


# -- the stroke (2-D, in the star wheel's y-z plane) ----------------------------------------------------------------
def tooth_rects(phi: float):
    """Teeth as 2-D rectangles (centre (y, z), half sizes (radial, tangential), angle) relative to SW, with the wheel
    turned ``phi`` in the tightening sense (teeth on the slot's side move up)."""
    r = (TOOTH_R[0] + TOOTH_R[1]) / 2
    out = []
    for k in range(N_TEETH):
        a = math.pi + PITCH / 2 + k * PITCH - phi      # tightening moves the -y side's teeth up: angle decreases
        out.append((np.array([r * math.cos(a), r * math.sin(a)]), np.array([(TOOTH_R[1] - TOOTH_R[0]) / 2,
                                                                           TOOTH_T / 2]), a))
    return out


def _rect_corners(c, h, a):
    u = np.array([math.cos(a), math.sin(a)])
    v = np.array([-math.sin(a), math.cos(a)])
    return np.array([c + sx * h[0] * u + sy * h[1] * v for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _seg_point(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-18), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def rect_gap(r1, r2) -> float:
    """Signed gap between two 2-D rectangles (c, h, angle): distance if apart, minus the smallest SAT overlap if not."""
    p, q = _rect_corners(*r1), _rect_corners(*r2)
    overlap = math.inf
    for poly in (p, q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            a, b = p @ n, q @ n
            sep = max(b.min() - a.max(), a.min() - b.max())
            if sep > 0:
                return _poly_dist(p, q)
            overlap = min(overlap, -sep)
    return -overlap


def _poly_dist(p, q) -> float:
    d = math.inf
    for a_poly, b_poly in ((p, q), (q, p)):
        for x in a_poly:
            for j in range(4):
                d = min(d, _seg_point(x, b_poly[j], b_poly[(j + 1) % 4]))
    return d


# Spoon stroke parameters: the blade's centre line always crosses the backing plate's mid-plane at the slot's
# fulcrum height (FULCRUM), and the spoon is described by its angle ``alpha`` (tip up, rad) and how far its tip
# reaches past that crossing point, ``d`` (m).
FULCRUM = np.array([0.0, -PLATE_T / 2, SW[2]])       # B frame
# The stroke lifts the tooth BELOW the blade: the blade goes in tip down (ALPHA_LOW), its end under the lower tooth
# of the gap facing the slot, and the handle is pressed down (pivoting on the slot's edge) until the blade is level
# (ALPHA_TOP): its end lifts that tooth up past the wheel's axis, past half a tooth, and the wheel clicks on. The
# tooth's push on the blade then points into the slot (it drives the blade's end in toward the hub, not out of the
# teeth, as levering the tooth above would), so how far the wheel turns depends on the lever angle and hardly on how
# deep the blade's end is: 29-34 degrees with the end anywhere from 4 mm short to 1 mm deep (a click is past 18; two
# would take 54). Mid-stroke the blade's end reaches the root of the tooth and slides up the hub, bottomed there.
ALPHA_LOW = math.radians(-14.0)                      # the blade going in, tip down
ALPHA_TOP = math.radians(1.5)                        # the end of the lever stroke
D_ENGAGE = 0.035                                     # the blade's end under the lower tooth, 12.4 mm from the axis
D_BACK = D_ENGAGE - 0.011                            # withdrawn 11 mm, clear of the teeth


def blade_dir(alpha: float) -> np.ndarray:
    """Unit vector along the blade toward its tip (B frame) at lever angle ``alpha``."""
    return np.array([0.0, math.cos(alpha), math.sin(alpha)])


def spoon_pose_b(alpha: float, d: float):
    """(S origin, rotation S->B) with the blade's tip end ``d`` past the fulcrum along the blade at angle alpha."""
    u = blade_dir(alpha)
    r = rot_x(alpha) @ YAW90
    tip_centre = FULCRUM + d * u                      # the blade's end, on its centre line
    origin = tip_centre - r @ np.array([TIP_X, 0.0, BLADE_Z])
    return origin, r


def blade_rect_2d(alpha: float, d: float, length: float = 0.04):
    """The blade's last ``length`` as a 2-D rectangle in the wheel's y-z plane, relative to SW."""
    u = blade_dir(alpha)
    end = FULCRUM + d * u
    c = end - u * length / 2
    return (np.array([c[1] - SW[1], c[2] - SW[2]]), np.array([length / 2, BLADE[2] / 2]), alpha)


def blade_tooth_gap(alpha: float, d: float, phi: float) -> float:
    return min(rect_gap(blade_rect_2d(alpha, d), t) for t in tooth_rects(phi))


def pushed_to(alpha: float, d: float = D_ENGAGE, step: float = math.radians(0.25)) -> float:
    """How far (rad) the blade at ``alpha`` must have turned the wheel from its notch so the two don't overlap."""
    phi = 0.0
    while blade_tooth_gap(alpha, d, phi) < 0.0 and phi < PITCH:
        phi += step
    return phi


def slot_rects():
    """The slot's bottom and top edges as 2-D rectangles in the y-z plane (B frame, plate cross-section)."""
    t = PLATE_T
    bottom = (np.array([-t / 2, SLOT_Z[0] - 0.005]), np.array([t / 2, 0.005]), 0.0)
    top = (np.array([-t / 2, SLOT_Z[1] + 0.005]), np.array([t / 2, 0.005]), 0.0)
    return bottom, top


def blade_slot_gap(alpha: float, d: float) -> float:
    """Gap between the blade (in the slot) and the slot's top and bottom edges, in the y-z plane."""
    u = blade_dir(alpha)
    end = FULCRUM + d * u
    length = 0.10
    c = end - u * length / 2
    blade = (np.array([c[1], c[2]]), np.array([length / 2, BLADE[2] / 2]), alpha)
    return min(rect_gap(blade, r) for r in slot_rects())


# -- box maths (3-D) ---------------------------------------------------------------------------------------------------
def box_points(c, r, h, n: int = 5) -> np.ndarray:
    u = np.linspace(-1, 1, n)
    g = np.array([[a, b, d] for a in u for b in u for d in u])
    g = g[np.abs(g).max(axis=1) == 1.0]
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def point_box_dist(points, c, r, h) -> np.ndarray:
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    d = np.maximum(np.abs(local) - np.asarray(h), 0.0)
    return np.linalg.norm(d, axis=-1)


def boxes_gap(a, b, n: int = 7) -> float:
    """Approximate gap between two boxes (centre, rotation, half sizes); 0 if they touch or overlap."""
    return float(min(point_box_dist(box_points(*a, n), *b).min(), point_box_dist(box_points(*b, n), *a).min()))


def point_cyl_dist(points, c, axis, radius, half) -> np.ndarray:
    """Distance from points to a solid cylinder (centre, unit axis, radius, half length); 0 inside."""
    p = np.atleast_2d(points) - np.asarray(c)
    t = p @ np.asarray(axis)
    radial = np.linalg.norm(p - np.outer(t, axis), axis=1)
    return np.hypot(np.maximum(radial - radius, 0.0), np.maximum(np.abs(t) - half, 0.0))


AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


def spoon_boxes_s():
    """The spoon as boxes (centre, rotation, half sizes) in S."""
    hx = HANDLE[0] / 2
    return [(np.zeros(3), np.eye(3), np.array(HANDLE) / 2),
            (np.array([hx + BLADE[0] / 2, 0.0, BLADE_Z]), np.eye(3), np.array(BLADE) / 2)]


def spoon_boxes(pos, rot):
    rot = np.asarray(rot, float)
    return [(np.asarray(pos, float) + rot @ c, rot @ r, h) for c, r, h in spoon_boxes_s()]


def corner_boxes_world(drum_angle: float = 0.0, star_angle: float = 0.0, base=None, base_rot=None, band: bool = True):
    """Every box of the corner in the station frame (centre, rotation, half sizes), for clearance checks (``band``
    False: without the drum's 240 band segments, which ``drum_world_cylinders`` covers as one cylinder)."""
    p0 = B0 if base is None else np.asarray(base, float)
    r0 = np.eye(3) if base_rot is None else np.asarray(base_rot, float)
    out = []
    for _, c, s, r in plate_boxes() + base_boxes() + stand_boxes() + shoe_visual_boxes("shoe_primary") \
            + shoe_visual_boxes("shoe_secondary"):
        out.append((p0 + r0 @ np.asarray(c), r0 @ r, np.asarray(s) / 2))
    rd = rot_y(drum_angle)
    for _, c, s, r in drum_boxes() if band else []:
        out.append((p0 + r0 @ rd @ np.asarray(c), r0 @ rd @ r, np.asarray(s) / 2))
    rs = rot_x(star_angle)
    for _, c, s, r in star_boxes():
        out.append((p0 + r0 @ (SW + rs @ np.asarray(c)), r0 @ rs @ r, np.asarray(s) / 2))
    return out


def corner_cylinders_world(drum_angle: float = 0.0, base=None, base_rot=None):
    """Every cylinder of the corner (centre, axis, radius, half length) in the station frame."""
    p0 = B0 if base is None else np.asarray(base, float)
    r0 = np.eye(3) if base_rot is None else np.asarray(base_rot, float)
    out = []
    for _, c, rad, length, ax in shoe_cylinders() + stand_cylinders():
        out.append((p0 + r0 @ np.asarray(c), r0 @ AXES[ax], rad, length / 2))
    rd = rot_y(drum_angle)
    for _, c, rad, length, ax in drum_cylinders():
        out.append((p0 + r0 @ rd @ np.asarray(c), r0 @ rd @ AXES[ax], rad, length / 2))
    for _, c, rad, length, ax in star_cylinders():
        out.append((p0 + r0 @ (SW + np.asarray(c)), r0 @ AXES[ax], rad, length / 2))
    return out


def stud_world(i: int, drum_angle: float) -> np.ndarray:
    """Centre of stud ``i``'s free end region (station), the drum turned ``drum_angle`` about +y."""
    a = stud_angle(i)
    p = rot_y(drum_angle) @ np.array([STUD_PCD_R * math.cos(a), 0.0, STUD_PCD_R * math.sin(a)])
    return B0 + p


def drum_world_cylinders(drum_angle: float = 0.0, base=None, base_rot=None):
    """The drum as two cylinders (band + face, and the studs' ring) for the arms' clearance check."""
    p0 = B0 if base is None else np.asarray(base, float)
    r0 = np.eye(3) if base_rot is None else np.asarray(base_rot, float)
    ax = r0 @ AXES["y"]
    return [(p0 + r0 @ np.array([0.0, (DRUM_Y[0] + FACE_Y[1]) / 2, 0.0]), ax, DRUM_R_OUT,
             (FACE_Y[1] - DRUM_Y[0]) / 2),
            (p0 + r0 @ np.array([0.0, FACE_Y[1] + STUD_LEN / 2, 0.0]), ax, STUD_PCD_R + STUD_R, STUD_LEN / 2)]
