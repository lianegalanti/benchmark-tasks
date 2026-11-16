"""Dimensions of the hub and brake rotor, its two seized retaining screws, the hand impact driver, the hammer and the
driver's bench block; the blow and screw model; and the box/cylinder maths shared by make_assets.py, task.py,
demo.py and the tests. Pure Python (NumPy).

Frames
- Station: x forward, y to the robot's left, z up (m); the table top is at ``TABLE``.
- F, the fixture's frame: origin on the table top under the hub's axis, axes parallel to the station's. The hub and
  knuckle, off the car, stand hub-up on a bench stand, so the rotor lies flat and both screws point straight up. F is
  fixed; the scorer still measures everything from the fixture's measured pose.
- D, the driver's body frame: origin where the bit's tip is with the stroke fully out, z along the driver's axis toward
  the striking cap. The anvil (nose and bit) slides up to 3 mm along z into the body against its return spring.
- H, the hammer's frame: origin on the handle's axis at the centre of the head, z along the head toward the striking
  face, y along the handle toward its end.
- K, a screw's frame: origin at the centre of the head's top face (flush with the rotor's hat), z up its axis.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from yam_arena import station

    TABLE = float(station.TABLE_TOP)
except Exception:  # pragma: no cover - make_assets.py may run without yam_arena
    TABLE = 0.7475

# -- the hub and rotor (F frame) ---------------------------------------------------------------------------------------
F0 = np.array([0.40, -0.06, TABLE])     # station position of F's origin: the hub's axis

FOOT = (0.16, 0.012)                    # stand's foot plate: side, thickness
POST_R = 0.035                          # stand's post up to the rotor
RING_R = (0.078, 0.131)                 # 262 mm front rotor: friction ring, 22 mm thick (vents not modelled)
RING_Z = (0.060, 0.082)
HAT_R = 0.075                           # rotor hat (150 mm), its face 45 mm above the ring's underside
HAT_Z = (0.060, 0.099)
FACE_Z = (0.099, 0.105)                 # the hat's face: 6 mm, the screw heads flush with its top
FACE = FACE_Z[1]
PILOT = (0.028, 0.006)                  # hub pilot through the rotor's centre bore: radius, height above the face
STUD_R, STUD_LEN, STUD_PCD_R = 0.006, 0.038, 0.050   # four M12 x 1.5 wheel studs on a 4 x 100 mm circle
STUD_ANGLES = tuple(math.radians(a) for a in (45.0, 135.0, 225.0, 315.0))
N_SEG = 24

# The two retaining screws: M6 x 1.0 countersunk Phillips (ISO 7046, PH3 recess), on a 40 mm radius between the studs,
# one either side of the hub along y. Only the one on the robot's left (+y), "a", is still in: the other has already
# been taken out, and its hole is open.
SCREW_PCD_R = 0.040
SCREWS = {"a": np.array([0.0, SCREW_PCD_R, FACE])}
HOLES = (np.array([0.0, SCREW_PCD_R, FACE]), np.array([0.0, -SCREW_PCD_R, FACE]))   # the hat's openings, both screws'
OUT_HOLE = HOLES[1]                                                                  # the screw already out
HOLE_HALF = 0.007                       # the hat face's opening round each head (12 mm head)
HEAD_R = 0.0058

# PH3 recess (K frame): 6.6 mm across at the top, 3.6 mm deep, its walls 34.8 degrees off the axis, as eight petals
RECESS_R = 0.0033
RECESS_BOTTOM_R = 0.0008
RECESS_D = 0.0036
RECESS_TAN = (RECESS_R - RECESS_BOTTOM_R) / RECESS_D
N_PETALS = 8
PETAL_T = 0.0012

# -- the hand impact driver (D frame) ----------------------------------------------------------------------------------
# A 3/8 in-drive hand impact driver with a PH3 impact bit: (name, radius, z0, z1) from the bit's tip up, with the stroke
# fully out. The first five (bit and nose) are the anvil; the knurled body and the cap are the body.
DRIVER_PARTS = (("tip1", 0.0010, 0.0, 0.0012), ("tip2", 0.0019, 0.0012, 0.0024), ("tip3", 0.0028, 0.0024, 0.0036),
                ("shank", 0.0040, 0.0036, 0.030), ("nose", 0.011, 0.030, 0.050), ("body", 0.018, 0.050, 0.155),
                ("cap", 0.0195, 0.155, 0.170))
DRIVER_LEN = 0.170
CAP_R = 0.0195
DRIVER_MASS = 0.65
BODY_MASS, ANVIL_MASS, SLIDER_MASS = 0.50, 0.13, 0.02
DRIVER_COM = 0.105
GRIP_Z = 0.1025                         # the hand holds the knurled body at its middle
N_ANVIL_PARTS = 5

# The driver's mechanism: a blow on the cap drives the body down over the anvil (3 mm stroke) against its return spring;
# a helical cam between them turns the anvil (and the bit) as the stroke closes: 12 degrees over the full stroke.
STROKE = 0.003                          # m: the stroke's travel
SPRING_K = 5000.0                       # N/m: the return spring ...
SPRING_PRELOAD = 20.0                   # ... preloaded to 20 N (35 N at the bottom): a hand's push barely moves it
CAM = math.radians(12.0) / STROKE       # rad per m of stroke: the cam's helix
BOTTOM = 0.0026                         # m: the stroke has bottomed (the cam has driven the bit its full turn)
DEEPEN = 0.002                          # m: a good blow drives the stroke at least this much deeper than it was just
                                        # before the contact (so a body already pressed down then tapped doesn't count)
HELD = 0.005                            # m: a blow counts only on a held driver: a gripper this close to its body

# How deep the bit's tip sits when it is fully home in the recess (the third step of its cone meets the walls)
SEAT_DEPTH = min([z0 + (RECESS_R - r) / RECESS_TAN for _, r, z0, _ in DRIVER_PARTS[:3]] + [RECESS_D])

# -- the hammer (H frame) ------------------------------------------------------------------------------------------
# A 16 oz ball-peen hammer: 0.45 kg forged head (28 mm face), 330 mm hickory handle; 0.62 kg in all.
FACE_R = 0.0145
HEAD_FACE_Z = (-0.005, 0.060)          # the face's cylinder, along H's z
HEAD_PEEN_Z = (-0.040, -0.005)
PEEN_R = 0.012
HANDLE = (0.022, 0.323, 0.030)         # x (thickness), y (length past the head), z (depth, in the swing plane)
HANDLE_Y0 = 0.012
FACE_TIP = HEAD_FACE_Z[1]
HAMMER_MASS = 0.62
HEAD_MASS, HANDLE_MASS = 0.45, 0.17
HAMMER_COM = np.array([0.0, HANDLE_MASS * (HANDLE_Y0 + HANDLE[1] / 2) / HAMMER_MASS,
                       HEAD_MASS * 0.012 / HAMMER_MASS])
HAMMER_GRIP_Y = 0.120                  # the hand holds the handle 12 cm from the head

# -- the driver's bench block (B frame = station, fixed): a hardwood block with a square pocket ----------------------
BLOCK_C = np.array([0.46, -0.29])
BLOCK = (0.08, 0.07)                   # side, height
POCKET = (0.030, 0.055)                # square pocket: side, depth

# -- start poses (station) -------------------------------------------------------------------------------------------
# -- the hammer's stand (fixed): a stepped hardwood block; the hammer stands on its face, its handle on the step ------
STAND_C = np.array([0.40, 0.125])       # under the hammer's head
STAND = (0.07, 0.10)                    # side (x), height of the lower step (the face stands on it)
STAND_STEP = (0.030, 0.070, 0.045)      # the upper step: from y, to y (from the head's axis), height above the lower

DRIVER_START = np.array([BLOCK_C[0], BLOCK_C[1], TABLE + BLOCK[1] - 0.050])   # tip; body's shoulder on the block
HAMMER_START = np.array([STAND_C[0], STAND_C[1], TABLE + STAND[1] + FACE_TIP + 0.0005])  # face down on the stand


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


HAMMER_STRIKE = rot_y(math.pi)         # face down, handle toward +y


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


def screw_world(name: str, base=None, base_rot=None) -> np.ndarray:
    """Station position of a screw head's top centre."""
    p0 = F0 if base is None else np.asarray(base, float)
    r0 = np.eye(3) if base_rot is None else np.asarray(base_rot, float)
    return p0 + r0 @ SCREWS[name]


# -- the blow and the screw model --------------------------------------------------------------------------------------
V_BLOW = 1.8             # m/s: the hammer's face must meet the cap at least this fast (~0.7 J in the head)
J_MIN = 0.2              # N s: ... and the contact between them must pass at least this impulse in one physics step
TURN_PER_BLOW = math.radians(12.0)   # one good blow turns the bit (and the screw) through the cam's stroke
LOOSE = math.radians(30.0)           # turned this far, the rust bond is broken: the screw is loose
NEAR = 0.008             # m: a blow lands on a screw if the bit's tip is this close to its recess
SEAT_R = 0.0010          # m: seated: the tip within this of the screw's axis ...
SEAT_MIN = 0.0020        # ... and at least this deep in the recess (fully home is SEAT_DEPTH, 3.1 mm) ...
SQUARE = math.radians(5.0)           # ... and the driver's axis within this of the screw's
RUIN = 2                 # cam-outs that strip a recess
CAP_LAT = 0.0195         # m: a square blow: the face's centre over the cap (within its radius of the axis) ...
FACE_TILT = math.radians(20.0)       # ... and the face within this of square to it
H_ARM = 0.050            # m: the face must have been at least this far off the cap before the next blow counts
WINDOW = 12              # physics steps (50 ms) after the contact in which the stroke must bottom


def seated(tip, axis, screw, screw_axis=(0.0, 0.0, 1.0)) -> tuple[bool, float, float, float]:
    """(seated, radial offset, depth below the head's top, tilt) for the bit's tip and the driver's axis (unit, toward
    the cap) against one screw (head top centre, its axis up)."""
    tip, axis, screw, sa = (np.asarray(v, float) for v in (tip, axis, screw, screw_axis))
    rel = tip - screw
    depth = float(-rel @ sa)
    radial = float(np.linalg.norm(rel + depth * sa))
    tilt = float(math.acos(max(-1.0, min(1.0, float(axis @ sa)))))
    return (radial <= SEAT_R and depth >= SEAT_MIN and tilt <= SQUARE), radial, depth, tilt


def blow_outcome(speed: float, lateral: float, face_tilt: float, tip, axis, screws: dict, ruined: dict,
                 stroke: float = STROKE, held: bool = True):
    """What one blow (a contact passing at least J_MIN between the face and the cap) does: (kind, screw name or None,
    details). ``screws``: name -> head top centre (station); ``tip``/``axis``: the bit just before the blow; ``stroke``:
    how much deeper than just before the contact the stroke went in the WINDOW after it; ``held``: a gripper within
    HELD of the driver's body at the last control step. Kinds: "unheld" (no hand on the driver: nothing), "glancing" (face not square on the cap: nothing),
    "tap" (under V_BLOW: nothing), "air" (the bit on no screw: nothing), "stripped" (the screw's recess is ruined:
    nothing), "camout" (the bit was not seated and square: it cams out and damages the recess), "short" (seated and
    square, but the blow did not drive the stroke 2 mm deeper: the cam did not drive the bit its turn; nothing) and "good" (the screw turns
    TURN_PER_BLOW)."""
    if not held:
        return "unheld", None, {}
    if lateral > CAP_LAT or face_tilt > FACE_TILT:
        return "glancing", None, {}
    if speed < V_BLOW:
        return "tap", None, {}
    tip = np.asarray(tip, float)
    name = min(screws, key=lambda n: float(np.linalg.norm(tip - screws[n])))
    if float(np.linalg.norm(tip - screws[name])) > NEAR:
        return "air", None, {}
    if ruined.get(name):
        return "stripped", name, {}
    ok, radial, depth, tilt = seated(tip, axis, screws[name])
    det = {"radial_mm": round(radial * 1000, 2), "depth_mm": round(depth * 1000, 2),
           "tilt_deg": round(math.degrees(tilt), 2), "deepen_mm": round(stroke * 1000, 2)}
    if not ok:
        return "camout", name, det
    return ("good" if stroke >= DEEPEN else "short"), name, det


def screw_value(turned: float, camouts: int) -> float:
    """The screw's share of the score (out of 0.8): 0.8 x the fraction of the 30 degree breakaway turned, less 0.1 per
    cam-out; 0 once two cam-outs have stripped it."""
    if camouts >= RUIN:
        return 0.0
    return max(0.0, round(0.8 * min(turned / LOOSE, 1.0) - 0.1 * camouts, 4))


# -- shapes ----------------------------------------------------------------------------------------------------------
# Each box: (name, centre, size, rotation matrix) in its link's frame. Each cylinder: (name, centre, radius, length,
# axis "x"/"y"/"z").

def _arc(name, r0, r1, z, a0, a1, step):
    """Boxes round an arc in the x-y plane, radially r0..r1, vertically z = (z0, z1)."""
    out = []
    n = max(1, int(math.ceil((a1 - a0) / step - 1e-9)))
    da = (a1 - a0) / n
    for i in range(n):
        a = a0 + (i + 0.5) * da
        r = (r0 + r1) / 2
        w = 2 * r1 * math.tan(da / 2)
        out.append((f"{name}_{i}", (r * math.cos(a), r * math.sin(a), (z[0] + z[1]) / 2), (r1 - r0, w, z[1] - z[0]),
                    rot_z(a)))
    return out


def fixture_boxes():
    """The fixture's base link: the stand's foot, the rotor's friction ring, and the hat's face as strips along x with
    an opening round each screw head."""
    out = [("foot", (0.0, 0.0, FOOT[1] / 2), (FOOT[0], FOOT[0], FOOT[1]), np.eye(3))]
    out += _arc("ring", RING_R[0], RING_R[1], RING_Z, 0.0, 2 * math.pi, 2 * math.pi / N_SEG)
    holes = sorted({round(c[1] + s * HOLE_HALF, 6) for c in HOLES for s in (-1, 1)})
    edges = sorted({*np.round(np.arange(-HAT_R, HAT_R + 1e-9, 0.010), 6), *holes})
    edges = [e for e in edges if -HAT_R <= e <= HAT_R]
    zc, hz = (FACE_Z[0] + FACE_Z[1]) / 2, FACE_Z[1] - FACE_Z[0]
    for y0, y1 in zip(edges[:-1], edges[1:]):
        yy = 0.0 if y0 < 0 < y1 else max(abs(y0), abs(y1))
        half = max(math.sqrt(max(HAT_R ** 2 - yy ** 2, 0.0)), 0.01)
        yc, hy = (y0 + y1) / 2, y1 - y0
        in_hole = any(c[1] - HOLE_HALF - 1e-9 <= y0 and y1 <= c[1] + HOLE_HALF + 1e-9 for c in HOLES)
        if in_hole:
            for s in (-1, 1):
                out.append((f"face_{len(out)}", (s * (HOLE_HALF + half) / 2, yc, zc), (half - HOLE_HALF, hy, hz),
                            np.eye(3)))
        else:
            out.append((f"face_{len(out)}", (0.0, yc, zc), (2 * half, hy, hz), np.eye(3)))
    return out


def fixture_cylinders():
    out = [("post", (0.0, 0.0, (FOOT[1] + RING_Z[0]) / 2), POST_R, RING_Z[0] - FOOT[1], "z"),
           ("hat", (0.0, 0.0, (HAT_Z[0] + HAT_Z[1]) / 2), HAT_R, HAT_Z[1] - HAT_Z[0], "z"),
           ("pilot", (0.0, 0.0, FACE + PILOT[1] / 2), PILOT[0], PILOT[1], "z")]
    for i, a in enumerate(STUD_ANGLES):
        out.append((f"stud_{i}", (STUD_PCD_R * math.cos(a), STUD_PCD_R * math.sin(a), FACE + STUD_LEN / 2), STUD_R,
                    STUD_LEN, "z"))
    return out


def screw_boxes():
    """One screw head in K: the head's top as a ring of eight blocks, the recess's walls as eight petals (an
    eight-sided cone, 34.8 degrees off the axis) and its floor."""
    out = []
    da = 2 * math.pi / N_PETALS
    for i in range(N_PETALS):
        a = i * da
        r0, r1 = RECESS_R, HEAD_R
        w = 2 * r1 * math.tan(da / 2) * 1.05
        out.append((f"rim_{i}", ((r0 + r1) / 2 * math.cos(a), (r0 + r1) / 2 * math.sin(a), -0.001), (r1 - r0, w, 0.002),
                    rot_z(a)))
    slant = math.hypot(RECESS_R - RECESS_BOTTOM_R, RECESS_D)
    for i in range(N_PETALS):
        a = (i + 0.5) * da
        er = np.array([math.cos(a), math.sin(a), 0.0])
        et = np.array([-math.sin(a), math.cos(a), 0.0])
        ez = np.array([0.0, 0.0, 1.0])
        d = (-(RECESS_R - RECESS_BOTTOM_R) * er - RECESS_D * ez) / slant      # down the wall
        n = (RECESS_D * er - (RECESS_R - RECESS_BOTTOM_R) * ez) / slant      # into the metal
        mid = (RECESS_R + RECESS_BOTTOM_R) / 2 * er - RECESS_D / 2 * ez
        rot = np.column_stack([d, et, n])
        if np.linalg.det(rot) < 0:
            rot = np.column_stack([d, -et, n])
        w = 2 * RECESS_R * math.tan(da / 2) * 1.2
        out.append((f"petal_{i}", mid + n * PETAL_T / 2, (slant + 0.0006, w, PETAL_T), rot))
    out.append(("floor", (0.0, 0.0, -RECESS_D - 0.0005), (0.003, 0.003, 0.001), np.eye(3)))
    return out


def screw_marks():
    """The Phillips cross, drawn on the head (visual only, not colliding)."""
    return [(f"cross_{i}", (0.0, 0.0, -0.0001), (2 * RECESS_R, 0.0012, 0.0002), rot_z(i * math.pi / 2))
            for i in range(2)]


def driver_cylinders():
    return [(n, (0.0, 0.0, (z0 + z1) / 2), r, z1 - z0, "z") for n, r, z0, z1 in DRIVER_PARTS]


BODY_FACETS = 12      # the knurled body's collider: a 12-sided prism 36 mm across its flats (flat facets lie on the
                      # pads and resist rolling in the grip as the knurling does); facets face every 30 degrees


def anvil_cylinders():
    """The anvil's parts (its frame: the bit's tip, axes as D at stroke 0 and cam 0)."""
    return driver_cylinders()[:N_ANVIL_PARTS]


def body_cylinders():
    return driver_cylinders()[N_ANVIL_PARTS:]


def driver_body_boxes():
    """The body's 12 facets as boxes (D frame): 6 mm thick shells whose outer faces are 18 mm from the axis."""
    _, r, z0, z1 = DRIVER_PARTS[5]
    t = 0.006
    w = 2 * r * math.tan(math.pi / BODY_FACETS) * 1.02
    out = []
    for i in range(BODY_FACETS):
        a = 2 * math.pi * i / BODY_FACETS
        c = (r - t / 2) * np.array([math.cos(a), math.sin(a), 0.0]) + np.array([0.0, 0.0, (z0 + z1) / 2])
        out.append((f"facet_{i}", c, (t, w, z1 - z0), rot_z(a)))
    return out


def hammer_cylinders():
    return [("face", (0.0, 0.0, sum(HEAD_FACE_Z) / 2), FACE_R, HEAD_FACE_Z[1] - HEAD_FACE_Z[0], "z"),
            ("peen", (0.0, 0.0, sum(HEAD_PEEN_Z) / 2), PEEN_R, HEAD_PEEN_Z[1] - HEAD_PEEN_Z[0], "z")]


def hammer_boxes():
    return [("handle", (0.0, HANDLE_Y0 + HANDLE[1] / 2, 0.0), HANDLE, np.eye(3))]


def stand_boxes():
    """The hammer's stand (its frame: on the table under the head's axis): the lower step under the head, the upper
    step under the handle."""
    sx, h = STAND
    y0, y1, dh = STAND_STEP
    return [("lower", (0.0, (y1 - 0.03) / 2, h / 2), (sx, y1 + 0.03, h), np.eye(3)),
            ("upper", (0.0, (y0 + y1) / 2, h + dh / 2), (sx, y1 - y0, dh), np.eye(3))]


def block_boxes():
    """The block (its frame: centre of its foot, on the table) as four walls round the pocket and its floor."""
    s, h = BLOCK
    p, d = POCKET
    floor = h - d
    wall = (s - p) / 2
    return [("floor", (0.0, 0.0, floor / 2), (s, s, floor), np.eye(3)),
            ("wall_xp", ((p + wall) / 2, 0.0, (floor + h) / 2), (wall, s, d), np.eye(3)),
            ("wall_xm", (-(p + wall) / 2, 0.0, (floor + h) / 2), (wall, s, d), np.eye(3)),
            ("wall_yp", (0.0, (p + wall) / 2, (floor + h) / 2), (p, wall, d), np.eye(3)),
            ("wall_ym", (0.0, -(p + wall) / 2, (floor + h) / 2), (p, wall, d), np.eye(3))]


# -- box maths (3-D) ---------------------------------------------------------------------------------------------------
AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}


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


def cyl_box(c, radius, length, axis):
    """A cylinder's bounding box (centre, rotation, half sizes) in its link's frame."""
    h = {"x": (length / 2, radius, radius), "y": (radius, length / 2, radius), "z": (radius, radius, length / 2)}[axis]
    return np.asarray(c, float), np.eye(3), np.array(h)


def driver_boxes(origin, rot):
    """The driver as its parts' bounding boxes (station), from the body frame's origin (the tip with the stroke out)
    and D's rotation."""
    tip, rot = np.asarray(origin, float), np.asarray(rot, float)
    out = []
    for _, c, r, ln, ax in driver_cylinders():
        if r < 0.0035:         # the bit's tip steps (inside the recess when seated): skipped for clearance checks
            continue
        cc, _, h = cyl_box(c, r, ln, ax)
        out.append((tip + rot @ cc, rot, h))
    return out


def hammer_parts(origin, rot):
    """The hammer as boxes (station): the handle, and the head's two cylinders' bounding boxes."""
    o, rot = np.asarray(origin, float), np.asarray(rot, float)
    out = []
    for _, c, s, r in hammer_boxes():
        out.append((o + rot @ np.asarray(c), rot @ r, np.asarray(s) / 2))
    for _, c, rad, ln, ax in hammer_cylinders():
        cc, _, h = cyl_box(c, rad, ln, ax)
        out.append((o + rot @ cc, rot, h))
    return out


def fixture_world_cylinders(base=None, base_rot=None):
    """The hub and rotor as cylinders (centre, unit axis, radius, half length), station frame, for clearance checks:
    the ring (as a solid disc), the hat, the pilot, the studs and the screw heads."""
    p0 = F0 if base is None else np.asarray(base, float)
    r0 = np.eye(3) if base_rot is None else np.asarray(base_rot, float)
    out = [(p0 + r0 @ np.array([0.0, 0.0, sum(RING_Z) / 2]), r0 @ AXES["z"], RING_R[1], (RING_Z[1] - RING_Z[0]) / 2),
           (p0 + r0 @ np.array([0.0, 0.0, (HAT_Z[0] + FACE) / 2]), r0 @ AXES["z"], HAT_R, (FACE - HAT_Z[0]) / 2)]
    for _, c, rad, ln, ax in fixture_cylinders():
        out.append((p0 + r0 @ np.asarray(c), r0 @ AXES[ax], rad, ln / 2))
    return out


def fixture_world_boxes(base=None, base_rot=None):
    p0 = F0 if base is None else np.asarray(base, float)
    r0 = np.eye(3) if base_rot is None else np.asarray(base_rot, float)
    return [(p0 + r0 @ np.asarray(c), r0 @ r, np.asarray(s) / 2) for _, c, s, r in fixture_boxes()]


def gap_to_fixture(boxes, base=None, base_rot=None, n: int = 5) -> float:
    """Smallest distance from any of the boxes to the hub and rotor (0 if touching)."""
    pts = np.vstack([box_points(*b, n) for b in boxes])
    return float(min(point_cyl_dist(pts, *c).min() for c in fixture_world_cylinders(base, base_rot)))
