"""Dimensions of the manikin (head, neck, soft lower nose with its two nostrils), the swab and its transport tube; the
frames that tie them together; and the distance maths shared by make_assets.py, task.py, demo.py and the tests.
Pure Python (NumPy).

Frames
- Station: x forward, y to the robot's left, z up (m); the table top is at ``TABLE``.
- M, the manikin's root: on the table under the middle of its bust; axes parallel to the station's. The manikin faces
  the robot (its face toward -x), so the patient's left is the station's -y (the robot's right).
- H, the head link: origin on the neck pivot (the top of the neck, 65 mm below the head's centre); axes parallel to
  M's when the neck is straight. The neck pitches about H's y (positive: the head tips back, chin up) and yaws
  about its z.
- N, the lower nose (soft: alae, columella, tip and both nostrils): origin at the middle of the two nostril openings;
  its x axis U runs up each nostril (the vestibule's axis: up and back, 65 degrees above horizontal), y is H's y, and
  z (F) points across the nostril toward the nose tip. The lower nose slides on two sprung joints in N's y-z plane
  (soft tissue): ``nose_f`` along F and ``nose_y`` along y.
- S, the swab: origin at the centre of its cap's grip section, x along the swab toward the tip, y across the grip's
  two flats (the fingers close along it), z = x cross y.
- T, the tube: origin at the centre of its rim, z up its axis (out of the tube).
"""

from __future__ import annotations

import math

import numpy as np

try:
    from yam_arena import station

    TABLE = float(station.TABLE_TOP)
except Exception:  # pragma: no cover - make_assets.py may run without yam_arena
    TABLE = 0.7475

# -- the manikin -------------------------------------------------------------------------------------------------------
HX = 0.34
M0 = np.array([HX, 0.0, TABLE])              # station position of M

# Bust (fixed): shoulders 400 mm wide, chest 190 mm deep, 105 mm high at the shoulders; neck 110 mm across.
BUST = (0.19, 0.40, 0.085)                   # box: depth, width, height (from the table)
BUST_C = np.array([0.012, 0.0, BUST[2] / 2])
TRAP = (0.15, 0.24, 0.035)                   # trapezius / upper chest, on top of the bust
TRAP_C = np.array([0.020, 0.0, BUST[2] + TRAP[2] / 2])
SHOULDER_R = 0.045                           # rounded shoulder tops: a capsule across
SHOULDER_C = np.array([0.012, 0.0, 0.065])
SHOULDER_LEN = 0.31                          # cylindrical part, along y
NECK_R = 0.055
NECK_Z = (0.10, 0.215)
NECK_C = np.array([0.010, 0.0, 0.0])         # neck axis (x, y) in M

PIVOT = np.array([0.0, 0.0, 0.200])          # neck pivot in M: H's origin
HEAD_C = np.array([0.0, 0.0, 0.065])         # head centre in H (265 mm above the table)
HEAD_AX = np.array([0.096, 0.076, 0.115])    # semi-axes: 192 mm front to back, 152 mm wide, 230 mm tall
HEAD_MASS = 1.3
NECK_K = 20.0                                # N m/rad: the neck's spring toward straight (pitch and yaw) ...
NECK_D = 0.6                                 # N m s/rad
NECK_CAP = 0.2                               # ... with its torque capped at 0.2 N m (a patient yields to a push)
NECK_LIMIT = math.radians(30.0)
HEAD_PUSHED = math.radians(15.0)             # scorer: the head turned this far from straight was forced

# Lower nose (N frame). The vestibule axis U is 65 degrees above horizontal, tilted back.
THETA = math.radians(65.0)
U = np.array([math.cos(THETA), 0.0, math.sin(THETA)])     # in H
F = np.array([-math.sin(THETA), 0.0, math.cos(THETA)])    # in H: across the nostril, toward the nose tip
R_N = np.column_stack([U, [0.0, 1.0, 0.0], F])            # N axes in H
N0 = HEAD_C + np.array([-0.1075, 0.0, -0.052])            # N origin in H
NOSE_K = 250.0                               # N/m: the lower nose's spring, each way (soft tissue)
NOSE_D = 3.0                                 # N s/m
NOSE_LIMIT = 0.010                           # m: the soft nose's travel each way
NOSE_MASS = 0.03

# The two nostrils (vestibules): elliptical, 15 mm long (along F) and 10 mm wide, 26 mm deep; patient's left is -y.
NOSTRIL_Y = {"left": -0.009, "right": 0.009}  # lumen centres in N's y
SEMI_F, SEMI_Y = 0.0075, 0.005
LUMEN_DEPTH = 0.026
WALL = 0.002                                 # the lining's thickness (16 staves round each nostril)
N_STAVES = 16

# Rules on the swab's tip (shared by the scorer, the tests and the README)
DEPTH_BAND = (0.015, 0.022)                  # m: tip end past the nostril opening while circles count (whole tip in)
TOO_DEEP = 0.025                             # m: past the vestibule, toward the nasal valve and turbinates
FORCE_BAND = (0.3, 2.0)                      # N on the nostril wall while circles count
DISP_BAND = (FORCE_BAND[0] / NOSE_K, FORCE_BAND[1] / NOSE_K)   # 1.2 .. 8 mm of the soft nose's travel
CIRCLES = 4                                  # full circles per nostril
TOUCH = 0.001                                # m: the tip this close to anything but a nostril's lining or its tube
                                             # bore is contaminated
ON_WALL = 0.001                              # m: a circle step counts only with the tip this close to the lining it
                                             # pushes toward

# -- the swab (S frame) ------------------------------------------------------------------------------------------------
# A sterile foam-tipped anterior nasal swab fixed in the cap of its transport tube: the cap is the handle.
GRIP = np.array([0.044, 0.016, 0.020])        # the cap's grip section: 44 mm long, 16 mm across its two flats
FLANGE = (0.009, 0.022, 0.024)               # radius, x range: 18 mm flange that seats on the tube's rim
PLUG = (0.005, 0.024, 0.030)                 # 10 mm plug that goes into the tube: 6 mm straight ...
CONE = (0.005, 0.0025, 0.030, 0.036)         # ... then tapered to 5 mm over 6 mm (the lead-in): r0, r1, x0, x1
SHAFT = (0.00125, 0.036, 0.126)              # 2.5 mm rigid polystyrene shaft, 90 mm
TIP = (0.00275, 0.126, 0.141)                # foam tip, 5.5 mm across, 15 mm long
TIP_END = TIP[2]
SWAB_LEN = TIP_END + GRIP[0] / 2             # 163 mm overall
SWAB_MASS = 0.008
TCP_IN_S = -0.008                            # the TCP sits 8 mm past the cap's axis, toward the fingertips
GRIP_IN_S = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # S axes in TCP coordinates (columns)

# -- the tube and its rack (T frame) -----------------------------------------------------------------------------------
TUBE_POS = np.array([0.21, -0.17, TABLE + 0.153])   # station: centre of the rim
TUBE_R_IN, TUBE_R_OUT = 0.00625, 0.00775     # 12.5 mm bore, 15.5 mm outside
TUBE_LEN = 0.150
TUBE_BOTTOM = 0.002
RACK = (0.050, 0.050, 0.060)                 # rack block round the tube's foot: 50 x 50 mm, 60 mm tall
RACK_HOLE = 0.017
SEAT_TOL = 0.002                             # seated: the flange within 2 mm of the rim ...
SEAT_TILT = math.radians(8.0)                # ... the swab within 8 degrees of the tube's axis
SEAT_OFF = 0.003                             # ... and within 3 mm of its axis at the flange

# Start: the swab in its tube, the flange resting on the rim; flats facing +-y.
R_SWAB0 = np.column_stack([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
SWAB_START = TUBE_POS + np.array([0.0, 0.0, FLANGE[2] + 0.0005])


# -- small maths -------------------------------------------------------------------------------------------------------
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
    t = np.trace(r)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        w, x, y, z = (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        w, x, y, z = (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
        w, x, y, z = (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    q = q / np.linalg.norm(q)
    return tuple(float(v) for v in q)


def quat_xyzw(r: np.ndarray) -> tuple[float, float, float, float]:
    w, x, y, z = quat_wxyz(r)
    return (x, y, z, w)


def rot_from_xyzw(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- link poses (station frame) at given joint values ------------------------------------------------------------------
def head_pose(pitch: float = 0.0, yaw: float = 0.0, m0=None, m_rot=None):
    """The head link's (position, rotation) with the neck at ``pitch``/``yaw``."""
    p0 = M0 if m0 is None else np.asarray(m0, float)
    r0 = np.eye(3) if m_rot is None else np.asarray(m_rot, float)
    return p0 + r0 @ PIVOT, r0 @ rot_y(pitch) @ rot_z(yaw)


def nose_pose(head_pos, head_rot, f: float = 0.0, y: float = 0.0):
    """The lower nose's (position, rotation) on a head at ``head_pos``/``head_rot``, displaced ``f``/``y``."""
    hr = np.asarray(head_rot, float)
    r = hr @ R_N
    return np.asarray(head_pos, float) + hr @ N0 + r @ np.array([0.0, y, f]), r


def nostril_station(side: str, depth: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """A point on nostril ``side``'s axis, ``depth`` past its opening, in the station frame (nose undisplaced)."""
    hp, hr = head_pose(pitch, yaw)
    npos, nrot = nose_pose(hp, hr)
    return npos + nrot @ np.array([depth, NOSTRIL_Y[side], 0.0])


def u_station(pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    return head_pose(pitch, yaw)[1] @ U


def f_station(pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    return head_pose(pitch, yaw)[1] @ F


# -- the nose's parts (N frame: x = U, y, z = F) -----------------------------------------------------------------------
def ellipse_point(t: float):
    return np.array([SEMI_F * math.cos(t), SEMI_Y * math.sin(t)])     # (f, y) offsets from a lumen's centre


def lumen_staves(side: str):
    """The lining round one nostril as boxes (name, centre, size, rotation) in N: their inner faces on the ellipse."""
    out = []
    per = math.pi * (3 * (SEMI_F + SEMI_Y) - math.sqrt((3 * SEMI_F + SEMI_Y) * (SEMI_F + 3 * SEMI_Y)))
    width = per / N_STAVES * 1.25
    yc = NOSTRIL_Y[side]
    for k in range(N_STAVES):
        t = 2 * math.pi * k / N_STAVES
        f, y = ellipse_point(t)
        n = np.array([SEMI_Y * math.cos(t), SEMI_F * math.sin(t)])
        n /= np.linalg.norm(n)                            # outward normal (f, y)
        tan = np.array([-n[1], n[0]])
        c = np.array([(LUMEN_DEPTH - 0.001) / 2, yc + y + n[1] * WALL / 2, f + n[0] * WALL / 2])
        rot = np.column_stack([[1.0, 0.0, 0.0], [0.0, tan[1], tan[0]], [0.0, n[1], n[0]]])
        if np.linalg.det(rot) < 0:
            rot[:, 1] *= -1
        out.append((f"lining_{side}_{k:02d}", c, np.array([LUMEN_DEPTH + 0.001, width, WALL]), rot))
    return out


def nose_outer_boxes():
    """The lower nose's outer parts (skin and cartilage) as boxes (name, centre, size) in N, axis aligned: none of
    them reaches into a nostril."""
    yo = max(abs(v) for v in NOSTRIL_Y.values()) + SEMI_Y + WALL        # 16 mm: outside the linings
    yi = min(abs(v) for v in NOSTRIL_Y.values()) - SEMI_Y - WALL        # 2 mm: the columella between them
    fo = SEMI_F + WALL                                                  # 9.5 mm
    return [("columella", np.array([0.013, 0.0, 0.0]), np.array([0.026, 2 * yi, 2 * SEMI_F])),
            ("ala_left", np.array([0.010, -(yo + 0.0015), 0.0]), np.array([0.020, 0.003, 2 * fo])),
            ("ala_right", np.array([0.010, yo + 0.0015, 0.0]), np.array([0.020, 0.003, 2 * fo])),
            ("tip", np.array([0.014, 0.0, fo + 0.0035]), np.array([0.028, 2 * yo + 0.006, 0.007])),
            ("sill", np.array([0.006, 0.0, -(fo + 0.0015)]), np.array([0.012, 2 * yo + 0.006, 0.003]))]


# -- the head's and bust's colliders -----------------------------------------------------------------------------------
LIPS = [("lip_upper", HEAD_C + np.array([-0.080, 0.0, -0.068]), 0.0065, 0.036),   # capsules along y: centre, r, length
        ("lip_lower", HEAD_C + np.array([-0.077, 0.0, -0.083]), 0.006, 0.032)]


def ellipsoid_hull_points(n_lat: int = 6, n_lon: int = 10) -> np.ndarray:
    """Vertices of the head's convex collider (in H): rings on the ellipsoid, a vertex at each pole and one at the
    back of the head on its mid-line."""
    pts = [HEAD_C + np.array([0.0, 0.0, HEAD_AX[2]]), HEAD_C - np.array([0.0, 0.0, HEAD_AX[2]])]
    for i in range(1, n_lat + 1):
        lat = -math.pi / 2 + math.pi * i / (n_lat + 1)
        for j in range(n_lon):
            lon = 2 * math.pi * j / n_lon
            pts.append(HEAD_C + HEAD_AX * np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon),
                                                    math.sin(lat)]))
    return np.array(pts)


HULL_LAT, HULL_LON = 7, 8


def hull_triangles():
    """The head's convex collider (in H) as triangles: lat-long faces on ellipsoid_hull_points."""
    pts = ellipsoid_hull_points(HULL_LAT, HULL_LON)      # [north pole, south pole, rings from south to north]
    ring = lambda i: 2 + i * HULL_LON                    # noqa: E731
    faces = []
    for j in range(HULL_LON):
        faces.append((1, ring(0) + (j + 1) % HULL_LON, ring(0) + j))
        faces.append((ring(HULL_LAT - 1) + j, ring(HULL_LAT - 1) + (j + 1) % HULL_LON, 0))
    for i in range(HULL_LAT - 1):
        for j in range(HULL_LON):
            k = (j + 1) % HULL_LON
            faces.append((ring(i) + j, ring(i) + k, ring(i + 1) + k))
            faces.append((ring(i) + j, ring(i + 1) + k, ring(i + 1) + j))
    return pts, faces


def hull_back_x(y: float, z: float) -> float:
    """How far back (H's +x) the head's collider reaches at (y, z) in H; -inf if (y, z) misses it."""
    pts, faces = hull_triangles()
    best = -math.inf
    for f in faces:
        a, b, c = pts[list(f)]
        m = np.array([[b[1] - a[1], c[1] - a[1]], [b[2] - a[2], c[2] - a[2]]])
        if abs(np.linalg.det(m)) < 1e-12:
            continue
        u, v = np.linalg.solve(m, [y - a[1], z - a[2]])
        if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
            best = max(best, float(a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0])))
    return best


_HULL_PLANES: list = []


def hull_planes():
    """The head collider's face planes (unit outward normal n, offset d: n . p = d on the face), in H."""
    if not _HULL_PLANES:
        pts, faces = hull_triangles()
        for f in faces:
            a, b, c = pts[list(f)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-12:
                continue
            n = n / np.linalg.norm(n)
            if n @ (a - HEAD_C) < 0:
                n = -n
            _HULL_PLANES.append((n, float(n @ a)))
    return _HULL_PLANES


def hull_dist(points_h) -> np.ndarray:
    """Distance (a lower bound outside; exact inside, negative) from points in H to the head's convex collider."""
    p = np.atleast_2d(points_h)
    n = np.array([pl[0] for pl in hull_planes()])
    d = np.array([pl[1] for pl in hull_planes()])
    return (p @ n.T - d).max(axis=1)


def bust_boxes():
    """(name, centre, size) in M."""
    return [("bust", BUST_C, np.array(BUST)), ("trapezius", TRAP_C, np.array(TRAP))]


# -- the swab ----------------------------------------------------------------------------------------------------------
def swab_from_tcp(tcp, rot):
    """The swab's (position, rotation) held at TCP pose ``tcp``/``rot``."""
    rot = np.asarray(rot, float)
    return np.asarray(tcp, float) + rot @ np.array([0.0, 0.0, TCP_IN_S]), rot @ GRIP_IN_S


def tcp_for_swab(pos, rot):
    """The TCP pose that holds the swab at (``pos``, ``rot``)."""
    r_tcp = np.asarray(rot, float) @ GRIP_IN_S.T
    return np.asarray(pos, float) - r_tcp @ np.array([0.0, 0.0, TCP_IN_S]), r_tcp


def tip_segment(pos, rot):
    """The foam tip as a segment (a, b) with radius TIP[0], station frame; b is the end's centre of curvature."""
    r = TIP[0]
    rot = np.asarray(rot, float)
    p = np.asarray(pos, float)
    return p + rot @ np.array([TIP[1] + r, 0.0, 0.0]), p + rot @ np.array([TIP[2] - r, 0.0, 0.0])


def tip_end(pos, rot) -> np.ndarray:
    return np.asarray(pos, float) + np.asarray(rot, float) @ np.array([TIP_END, 0.0, 0.0])


def swab_boxes(pos, rot):
    """The swab as boxes (centre, rotation, half sizes), station frame, for the grippers' clearance: the cap (grip,
    flange and plug) and the shaft with its tip."""
    rot = np.asarray(rot, float)
    p = np.asarray(pos, float)
    parts = [(np.zeros(3), GRIP / 2),
             (np.array([(FLANGE[1] + CONE[3]) / 2, 0.0, 0.0]), np.array([(CONE[3] - FLANGE[1]) / 2, FLANGE[0],
                                                                         FLANGE[0]])),
             (np.array([(SHAFT[1] + TIP[2]) / 2, 0.0, 0.0]), np.array([(TIP[2] - SHAFT[1]) / 2, TIP[0], TIP[0]]))]
    return [(p + rot @ c, rot, h) for c, h in parts]


# -- the tube (T frame) ------------------------------------------------------------------------------------------------
def tube_staves(n: int = 12):
    """The tube's wall as boxes (name, centre, size, rotation about z) in T."""
    rm = (TUBE_R_IN + TUBE_R_OUT) / 2
    w = 2 * math.pi * rm / n * 1.2
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        out.append((f"wall_{k:02d}", np.array([rm * math.cos(a), rm * math.sin(a), -TUBE_LEN / 2]),
                    np.array([TUBE_R_OUT - TUBE_R_IN, w, TUBE_LEN]), a))
    return out


def rack_boxes():
    """The rack block round the tube's foot as four walls (name, centre, size) in T."""
    hx, hz = RACK[0] / 2, RACK[2]
    t = (RACK[0] - RACK_HOLE) / 2
    zc = -TUBE_LEN - 0.003 + hz / 2
    return [("rack_front", np.array([hx - t / 2, 0.0, zc]), np.array([t, RACK[1], hz])),
            ("rack_back", np.array([-hx + t / 2, 0.0, zc]), np.array([t, RACK[1], hz])),
            ("rack_left", np.array([0.0, hx - t / 2, zc]), np.array([RACK_HOLE, t, hz])),
            ("rack_right", np.array([0.0, -hx + t / 2, zc]), np.array([RACK_HOLE, t, hz]))]


# -- distance maths ----------------------------------------------------------------------------------------------------
def seg_points(a, b, n: int = 7) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def point_box_dist(points, c, r, h) -> np.ndarray:
    local = (np.atleast_2d(points) - np.asarray(c)) @ np.asarray(r)
    d = np.maximum(np.abs(local) - np.asarray(h), 0.0)
    return np.linalg.norm(d, axis=-1)


def box_points(c, r, h, n: int = 5) -> np.ndarray:
    u = np.linspace(-1, 1, n)
    g = np.array([[a, b, d] for a in u for b in u for d in u])
    g = g[np.abs(g).max(axis=1) == 1.0]
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def boxes_gap(a, b, n: int = 5) -> float:
    """Approximate gap between two boxes (centre, rotation, half sizes); 0 if they touch or overlap."""
    return float(min(point_box_dist(box_points(*a, n), *b).min(), point_box_dist(box_points(*b, n), *a).min()))


def point_cyl_dist(points, c, axis, radius, half) -> np.ndarray:
    """Distance from points to a solid cylinder (centre, unit axis, radius, half length); 0 inside."""
    p = np.atleast_2d(points) - np.asarray(c)
    t = p @ np.asarray(axis)
    radial = np.linalg.norm(p - np.outer(t, axis), axis=1)
    return np.hypot(np.maximum(radial - radius, 0.0), np.maximum(np.abs(t) - half, 0.0))


def point_seg_dist(points, a, b) -> np.ndarray:
    p = np.atleast_2d(points)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip((p - a) @ ab / float(ab @ ab), 0.0, 1.0)
    return np.linalg.norm(p - (a + np.outer(t, ab)), axis=1)


def ellipsoid_dist(points, c, rot, axes) -> np.ndarray:
    """Approximate signed distance from points to an ellipsoid (centre, rotation, semi-axes): first order in the
    level set, exact on the axes; negative inside."""
    local = (np.atleast_2d(points) - np.asarray(c)) @ np.asarray(rot)
    q = local / np.asarray(axes)
    f = np.linalg.norm(q, axis=1)
    grad = np.linalg.norm(local / np.asarray(axes) ** 2, axis=1) / np.maximum(f, 1e-9)
    return (f - 1.0) / np.maximum(grad, 1e-9)


AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
