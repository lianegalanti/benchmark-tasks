"""Dimensions, layout and frame maths for the glad-hand coupling task. Pure NumPy; shared by make_assets.py, task.py,
demo.py and the tests.

Station frame: x forward, y to the robot's left, z up (m).

Glad hand frame G (every glad hand is the same casting): origin at the centre of the seal face, +z out of the face,
+x along the body toward its flange (the "tongue"), so the hose port and the lip (the hook) are on the -x side.

    flange  x  32..48 mm, |y| <= 11, z -5..-1   (a 4 mm tongue just behind the face plane), with a 2.5 mm tip tooth
            (x 49.5..52, |y| <= 6) turned back toward the casting's back, z -9..-5
    lip     a curved hook round the seal axis on the -x side: an overhang (radius 32..47.5 mm, z 9..13) over the
            mate's flange, a curved riser (radius 54.5..59.5) rising to z = +13, leaving a 7 mm groove between them
            for the mate's tip tooth (radius 49.5..52), flared into a funnel at its mouth; a chamfered entry on its
            +y end and a stop wall at its -y end (the flange's leading face stops on it at y = -11). The tooth in the
            groove locates the coupling along the long axis: about 2 mm of play each way.
    boss    the seal boss, radius <= 24 mm, z -14..0; the back block (the grip) 44 x 44 mm, z -32..-14
    neck    toward the hose port, x -62..-22, |y| <= 9, z -30..-16; the hose enters at PORT

Two glad hands couple face to face with their long axes turned 90 degrees apart, seals pressed together; turning
one 90 degrees about the common seal axis swings each flange under the other's lip until it hits the stop. Coupled,
the bodies lie in line (long axes opposite). The trailer's seal ring is sprung: compressed 3 mm the coupling's own
lips hold the faces together. Seal-face separation s (between the faces' planes): s = RING_PROUD + STUB_PROUD - c,
with c the ring's compression; locked, the flange's back bears on the mate's overhang at s = S_LOCK.

A tractor glad hand's pose relative to its trailer glad hand T is ``rel(phi, s)``: rotated 180 degrees about T's y
(face to face), turned ``phi`` about T's z, its face s above T's face. phi = 0 is locked (on the stop), phi = -90
degrees is the entry angle (flange clear of the lips).
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import station

TABLE = float(station.TABLE_TOP)
MM = 0.001


# -- rotations ----------------------------------------------------------------------------------------------------------
def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


FLIP = rot_y(math.pi)          # face to face: G's x and z reversed, y kept


def quat_xyzw(r) -> tuple:
    from yam_arena.ik import matrix_to_quat_xyzw

    return tuple(float(v) for v in matrix_to_quat_xyzw(np.asarray(r, float)))


# -- the glad hand casting (G frame, m) ----------------------------------------------------------------------------------
GH_MASS = 0.35               # kg, cast aluminium
GH_COM = np.array([-0.008, 0.0, -0.016])
GH_SIZE = (0.115, 0.060, 0.045)   # m, overall (port end to flange tip, lip/stop to chamfer, back to lip top)
PORT = np.array([-0.062, 0.0, -0.023])    # where the hose enters (end of the neck), G frame

FLANGE = ((0.032, -0.011, -0.005), (0.048, 0.011, -0.001))
WEB = ((0.018, -0.008, -0.005), (0.034, 0.008, -0.001))
BACK = ((-0.022, -0.022, -0.032), (0.022, 0.022, -0.014))
NECK = ((-0.062, -0.009, -0.030), (-0.022, 0.009, -0.016))
TIP = ((0.048, -0.006, -0.005), (0.052, 0.006, -0.001))        # the flange's tip, carrying ...
TOOTH = ((0.0495, -0.006, -0.009), (0.052, 0.006, -0.005))     # ... the tooth that drops into the mate's groove
STOP = ((-0.054, -0.019, 0.0), (-0.032, -0.011, 0.013))
LIP_ARC = (math.radians(180 - 22), math.radians(180 + 20))     # overhang's angular span (about G z, from +x)
RISER_ARC = (math.radians(180 - 32), math.radians(180 + 24))
CHAMFER_ARC = (math.radians(180 - 32), math.radians(180 - 22))  # the entry, ahead of the overhang
OVER_R = (0.032, 0.0475)     # overhang's radial span
RISER_R = (0.0545, 0.0595)   # riser's: the groove between them is 7 mm, about 2 mm of play each side of the tooth
CHAMFER_R = (0.032, 0.043)   # the chamfer stops short of the groove's funnel
FUNNEL = math.radians(14)    # the groove's mouth flares over this angle ahead of the lip ...
FUNNEL_FLARE = 0.0045        # ... by this much on each side (riser out to 59 mm, inner wall in to 43 mm)
FUNNEL_Z = 0.010             # inner funnel wall starts 1 mm above the overhang's underside
LIP_Z = (0.009, 0.013)       # overhang underside and top
RISER_Z0 = -0.016
BOSS_HALF = (0.024, 0.010, 0.007)           # three of these, turned 0/60/120 degrees: a 12-sided boss
BOSS_Z = -0.007
SEAL_HALF_XY = (0.019, 0.008)               # three of these too: the seal (stub on the tractor, ring on the trailer)
CHAMFER_RISE = 0.003         # the lip's entry: its underside rises 3 mm over the 10 degrees ahead of the overhang
CHAMFER_T = 0.001            # thin, so its top stays below the lip top (13 mm)
SEG = math.radians(8)        # the curved parts are built from boxes spanning at most this angle

STUB_PROUD = 0.0015          # tractor seal stands this proud of its face (rigid)
RING_PROUD = 0.0055          # trailer seal ring stands this proud at rest ...
RING_TRAVEL = 0.0055         # ... and can be pressed in this far (then flush: the faces' stubs meet the face)
RING_DEPTH = 0.008           # ring's depth below the face at rest
S_REST = RING_PROUD + STUB_PROUD             # face separation when the seals just touch (7 mm)
S_MIN = S_REST - RING_TRAVEL                 # ring fully in (1.5 mm)
OVER_Z = LIP_Z[0]                            # overhang underside (9 mm)
FLANGE_BACK = -FLANGE[0][2]                  # flange's back face depth behind the face plane (5 mm)
S_LOCK = OVER_Z - FLANGE_BACK                # 4 mm: the flange's back bears on the mate's overhang
C_LOCK = S_REST - S_LOCK                     # 3 mm of seal compression when locked
RING_RATE = 2000.0           # N/m
RING_PRELOAD = 3.0           # N at rest (ring at its outer stop)
RING_TARGET = -RING_PRELOAD / RING_RATE
RING_DAMPING = 20.0          # N s/m
RING_MASS = 0.01             # kg
ENTRY = math.radians(-90.0)  # phi at which the tractor glad hand meets the trailer's, flanges clear of the lips


def seal_force(c: float) -> float:
    return RING_PRELOAD + RING_RATE * c


def _lohi(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return (lo + hi) / 2, np.eye(3), (hi - lo) / 2


def arc_boxes(r0: float, r1: float, a0: float, a1: float, z0: float, z1: float):
    """An annular sector (radius r0..r1, angle a0..a1 about G z, height z0..z1) as boxes spanning at most SEG each:
    each box's inner face is a chord at r0 and its outer face a chord at r1, so the boxes' corners stick out past r1
    by at most r1 (1 / cos(SEG / 2) - 1) (0.12 mm at 48 mm) and nothing reaches inside r0."""
    n = max(1, math.ceil((a1 - a0) / SEG - 1e-9))
    out = []
    for i in range(n):
        lo, hi = a0 + (a1 - a0) * i / n, a0 + (a1 - a0) * (i + 1) / n
        c, h = (lo + hi) / 2, (hi - lo) / 2
        rot = rot_z(c)
        out.append((rot @ np.array([(r0 + r1) / 2, 0.0, (z0 + z1) / 2]), rot,
                    np.array([(r1 - r0) / 2, r1 * math.tan(h), (z1 - z0) / 2])))
    return out


def chamfer_box():
    """The slab under the lip's entry, as (centre, rotation, half) in G: its underside rises from z 9 mm where the
    overhang starts to 12 mm at the entry end, radius 32..43 mm."""
    a0, a1 = CHAMFER_ARC
    c = (a0 + a1) / 2
    r_mid = sum(CHAMFER_R) / 2
    run = r_mid * (a1 - a0)                          # arc length at mid radius
    tilt = math.atan2(CHAMFER_RISE, run)             # underside rises toward the entry (toward smaller angles)
    rot = rot_z(c) @ rot_x(-tilt)
    centre = rot_z(c) @ np.array([r_mid, 0.0, OVER_Z + CHAMFER_RISE / 2]) + rot[:, 2] * CHAMFER_T / 2
    return centre, rot, np.array([(CHAMFER_R[1] - CHAMFER_R[0]) / 2, math.hypot(run, CHAMFER_RISE) / 2,
                                  CHAMFER_T / 2])


def wall_box(a0: float, r0: float, a1: float, r1: float, thick: float, outward: bool, z0: float, z1: float):
    """A straight wall whose guiding face runs from (angle a0, radius r0) to (a1, r1) in G's xy plane, ``thick``
    deep away from the groove (outward for the riser side, inward for the inner side)."""
    p0 = np.array([r0 * math.cos(a0), r0 * math.sin(a0)])
    p1 = np.array([r1 * math.cos(a1), r1 * math.sin(a1)])
    t = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-t[1], t[0]])
    mid = (p0 + p1) / 2
    if (n @ mid > 0) != outward:
        n = -n
    c = mid + n * thick / 2
    yaw = math.atan2(t[1], t[0])
    return (np.array([c[0], c[1], (z0 + z1) / 2]), rot_z(yaw),
            np.array([np.linalg.norm(p1 - p0) / 2 + 0.0005, thick / 2, (z1 - z0) / 2]))


def lip_boxes():
    """The curved lip: overhang segments, riser segments, the chamfered entry and the stop wall (G frame)."""
    out = [(f"overhang{i}", b) for i, b in enumerate(arc_boxes(*OVER_R, *LIP_ARC, *LIP_Z))]
    out += [(f"riser{i}", b) for i, b in enumerate(arc_boxes(*RISER_R, *RISER_ARC, RISER_Z0, LIP_Z[1]))]
    a = RISER_ARC[0]
    out += [("funnel_out", wall_box(a - FUNNEL, RISER_R[0] + FUNNEL_FLARE, a, RISER_R[0], 0.005, True, RISER_Z0,
                                    LIP_Z[1])),
            ("funnel_in", wall_box(a - FUNNEL / 2, OVER_R[1] - FUNNEL_FLARE, LIP_ARC[0], OVER_R[1], 0.004, False,
                                   FUNNEL_Z, LIP_Z[1]))]
    out += [("chamfer", chamfer_box()), ("stop", _lohi(*STOP))]
    return out


def casting_boxes():
    """Every collision box of a glad hand's casting in G, as (name, (centre, rotation, half))."""
    out = [("back", _lohi(*BACK)), ("neck", _lohi(*NECK)), ("web", _lohi(*WEB)), ("flange", _lohi(*FLANGE)),
           ("tip", _lohi(*TIP)), ("tooth", _lohi(*TOOTH))] + lip_boxes()
    for i, yaw in enumerate((0.0, 60.0, 120.0)):
        out.append((f"boss{i}", (np.array([0.0, 0.0, BOSS_Z]), rot_z(math.radians(yaw)), np.array(BOSS_HALF))))
    return out


def stub_boxes():
    """The tractor glad hand's seal (rigid, proud of the face), G frame."""
    h = np.array([*SEAL_HALF_XY, STUB_PROUD / 2])
    return [(f"seal{i}", (np.array([0.0, 0.0, STUB_PROUD / 2]), rot_z(math.radians(y)), h))
            for i, y in enumerate((0.0, 60.0, 120.0))]


def ring_boxes(c: float = 0.0):
    """The trailer glad hand's seal ring at compression ``c``, G frame."""
    h = np.array([*SEAL_HALF_XY, (RING_PROUD + RING_DEPTH) / 2])
    z = (RING_PROUD - RING_DEPTH) / 2 - c
    return [(f"ring{i}", (np.array([0.0, 0.0, z]), rot_z(math.radians(y)), h))
            for i, y in enumerate((0.0, 60.0, 120.0))]


def tractor_boxes():
    return casting_boxes() + stub_boxes()


# -- layout (station frame) -----------------------------------------------------------------------------------------------
FACE_Z = TABLE + 0.080       # trailer glad hands' seal faces (face up)
COLOURS = ("blue", "red")    # blue (service) on the robot's left, red (emergency) on its right
SIDE = {"blue": "left", "red": "right"}
TRAILER_GH = {"blue": np.array([0.35, 0.21, FACE_Z]), "red": np.array([0.35, -0.21, FACE_Z])}
# each trailer glad hand's yaw: its +x (flange) direction. Coupled, the tractor's hose port points along T's +x, so
# each is turned to point that way at (blue) or a little past (red, 24 degrees, for the right wrist's range) its
# line's guide on the tender post: the line's pull then runs along the coupled pair's long axis (taken by the tip
# teeth in the grooves) and does not turn it open (red: it turns it onto its stop)
TRAILER_YAW = {"blue": math.radians(-144.2), "red": math.radians(120.0)}
R_TRAILER = {c: rot_z(TRAILER_YAW[c]) for c in COLOURS}

NOSE = np.array([0.52, 0.0, TABLE])          # trailer nose section's origin: the bulkhead's back face, on the table
BULKHEAD = ((-0.020, -0.36, 0.002), (0.0, 0.36, 0.30))          # nose frame (= station axes), m
BRACKET = ((-0.21, -0.33, 0.030), (-0.020, 0.33, 0.038))      # the glad-hand mounting bracket (steel plate)
PEDESTAL_HALF = (0.018, 0.018)

TENDER = np.array([0.10, 0.0, TABLE])        # the tractor's tender post, on the table
POST = ((-0.015, -0.015, 0.002), (0.015, 0.015, 0.235))         # tender frame (= station axes)
GUIDE = {"blue": np.array([0.10, 0.03, TABLE + 0.22]), "red": np.array([0.10, -0.03, TABLE + 0.22])}   # pivots
HOSE_R = 0.008               # hose (16 mm OD rubber air line)
HOSE_VIS = 0.34              # straight hose segment drawn from the glad hand back through the guide
HOSE_MASS = 0.10             # kg of hose carried with the glad hand
LINK_MASS = 0.01             # kg, the swivel links
L_MIN, L_MAX = 0.10, 0.48    # hose length between the guide and the port: slack .. fully paid out (its end stop)
COIL_RATE = 120.0            # N/m: the coiled line's return pull on the glad hand, zero with it parked (about 15 N
#                              at the farthest point of the carry, about 3.5-6 N coupled; an estimate for the working end
#                              of a 15 ft coiled nylon air line, which takes on the order of 10-20 N to hold stretched)
SLIDE_DAMPING = 2.0          # N s/m
BEND_MAX = math.radians(80)  # the glad hand's swivel's middle joint (keeps the hose off the face normal)
PITCH_LIM = (math.radians(-60), math.radians(85))

PARK_Z = TABLE + 0.070       # parked tractor glad hands: seal face (face down) at this height
PARK = {"blue": np.array([0.25, 0.12, PARK_Z]), "red": np.array([0.25, -0.12, PARK_Z])}
PARK_TURN = {"blue": 0.0, "red": 0.0}   # parked turned this far (about the vertical) from the entry angle


def rel(phi: float, s: float):
    """Tractor glad hand pose in its trailer glad hand's frame T: (position, rotation)."""
    return np.array([0.0, 0.0, s]), rot_z(phi) @ FLIP


def tractor_pose(colour: str, phi: float, s: float):
    """Tractor glad hand pose (station) at turn phi and separation s on its own trailer glad hand."""
    p, r = rel(phi, s)
    rt = R_TRAILER[colour]
    return TRAILER_GH[colour] + rt @ p, rt @ r


def park_pose(colour: str):
    """Parked: face down on its stand, turned PARK_TURN from the entry angle."""
    _, r = tractor_pose(colour, ENTRY, 0.05)
    return PARK[colour].copy(), rot_z(PARK_TURN[colour]) @ r


# the parking stand: two steel pads, under the lip (G x -56..-34) and under the flange (G x 34..47)
LIP_LOW = LIP_Z[1]           # face down, the lip's lowest face is 13 mm below the seal face ...
FLANGE_LOW = FLANGE[1][2]    # ... and the flange's 1 mm above it (the seal stub, 1.5 mm below, rests on nothing)


def stand_pads(colour: str):
    """The stand's two pads as (centre, rotation, half) in the station frame, standing on the table, their tops
    touching the parked glad hand's lip and flange."""
    p, r = park_pose(colour)
    yaw = math.atan2(r[1, 0], r[0, 0])
    out = []
    for (x0, x1), top in (((-0.056, -0.034), p[2] - LIP_LOW), ((0.034, 0.047), p[2] - FLANGE_LOW)):
        c = p + r @ np.array([(x0 + x1) / 2, 0.0, 0.0])
        h = np.array([(x1 - x0) / 2, 0.010, (top - TABLE) / 2])
        out.append((np.array([c[0], c[1], TABLE + h[2]]), rot_z(yaw), h))
    return out


# -- the line: guide pivot (yaw z, pitch y), slide along the hose, three-axis swivel at the port -------------------------
def chain_angles(colour: str, pos, rot):
    """Joint angles (absolute) that put the glad hand (G origin ``pos``, rotation ``rot``) on the line:
    psi, theta (hose direction at the guide), length, a, b, c (swivel: R_hose^T R_G = Rx(a) Ry(b) Rz(c))."""
    port = np.asarray(pos, float) + np.asarray(rot, float) @ PORT
    d = port - GUIDE[colour]
    length = float(np.linalg.norm(d))
    psi = math.atan2(d[1], d[0])
    theta = math.atan2(-d[2], math.hypot(d[0], d[1]))
    rh = rot_z(psi) @ rot_y(theta)
    w = rh.T @ np.asarray(rot, float)
    b = math.asin(max(-1.0, min(1.0, float(w[0, 2]))))
    a = math.atan2(-w[1, 2], w[2, 2])
    c = math.atan2(-w[0, 1], w[0, 0])
    return np.array([psi, theta, length, a, b, c])


def chain_pose(colour: str, q):
    """Forward kinematics of the line: the glad hand's (G origin, rotation) from absolute joint values."""
    psi, theta, length, a, b, c = [float(v) for v in q]
    rh = rot_z(psi) @ rot_y(theta)
    port = GUIDE[colour] + rh[:, 0] * length
    r = rh @ rot_x(a) @ rot_y(b) @ rot_z(c)
    return port - r @ PORT, r


CHAIN_JOINTS = ("yaw", "pitch", "slide", "roll", "bend", "twist")


def start_chain(colour: str):
    return chain_angles(colour, *park_pose(colour))


# -- measuring a coupling -------------------------------------------------------------------------------------------------
def coupling(colour_t: str, pos, rot, trailer_pos=None, trailer_rot=None):
    """A tractor glad hand (station pose) measured in trailer glad hand ``colour_t``'s frame: (phi, tilt, offset, s).
    phi: turn about the seal axis (0 = on the stop, negative = short of it); tilt: angle between the faces' normals
    (0 = face to face); offset: its face centre's distance from the seal axis; s: face separation."""
    tp = TRAILER_GH[colour_t] if trailer_pos is None else np.asarray(trailer_pos, float)
    tr = R_TRAILER[colour_t] if trailer_rot is None else np.asarray(trailer_rot, float)
    p = tr.T @ (np.asarray(pos, float) - tp)
    r = tr.T @ np.asarray(rot, float)
    tilt = math.acos(max(-1.0, min(1.0, -float(r[2, 2]))))
    mx = r[:, 0]
    phi = math.atan2(-mx[1], -mx[0])
    return phi, tilt, float(math.hypot(p[0], p[1])), float(p[2])


# -- box maths --------------------------------------------------------------------------------------------------------------
def place(boxes, pos, rot):
    """Boxes (centre, rotation, half) given in a frame at (pos, rot), into the parent frame."""
    pos, rot = np.asarray(pos, float), np.asarray(rot, float)
    return [(pos + rot @ c, rot @ r, h) for c, r, h in boxes]


def box_points(c, r, h, n: int = 5) -> np.ndarray:
    u = np.linspace(-1, 1, n)
    g = np.array([[a, b, d] for a in u for b in u for d in u])
    g = g[np.abs(g).max(axis=1) == 1.0]
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def point_box_dist(points, c, r, h) -> np.ndarray:
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    d = np.maximum(np.abs(local) - np.asarray(h), 0.0)
    return np.linalg.norm(d, axis=-1)


def boxes_overlap(a, b) -> bool:
    """Exact separating-axis test for two boxes (centre, rotation, half)."""
    ca, ra, ha = (np.asarray(v, float) for v in a)
    cb, rb, hb = (np.asarray(v, float) for v in b)
    t = cb - ca
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    for ax in axes:
        n = np.linalg.norm(ax)
        if n < 1e-9:
            continue
        ax = ax / n
        pa = sum(ha[i] * abs(ax @ ra[:, i]) for i in range(3))
        pb = sum(hb[i] * abs(ax @ rb[:, i]) for i in range(3))
        if abs(t @ ax) > pa + pb + 1e-9:
            return False
    return True


def boxes_gap(a, b, n: int = 7) -> float:
    """Gap between two boxes: 0 if they overlap (exact), else sampled (an upper bound within a fraction of a mm)."""
    if boxes_overlap(a, b):
        return 0.0
    return float(min(point_box_dist(box_points(*a, n), *b).min(), point_box_dist(box_points(*b, n), *a).min()))


def grow(box, d: float):
    c, r, h = box
    return c, r, np.asarray(h, float) + d


def lohi(lo, hi, offset=(0.0, 0.0, 0.0)):
    c, r, h = _lohi(lo, hi)
    return c + np.asarray(offset, float), r, h
