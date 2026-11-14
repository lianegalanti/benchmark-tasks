"""Dimensions, layout and box maths for the headrest task. Pure NumPy; shared by make_assets.py, task.py, demo.py and
the tests.

Station frame: x forward, y to the robot's left, z up (m). The seat frame F is the station frame shifted to ``SEAT``:
its origin is midway between the two guide sleeves' axes, level with the tops of their collars. The seat back faces
the robot (its front, where a passenger's back would be, is at x < 0).

The headrest frame H has its origin midway between the two post tips; z runs up the posts into the cushion, x through
the cushion's thickness (+x is its back), y along its width. Fitted, H's axes are F's and its origin is straight
below F's origin by the insertion depth. A headrest pose is written (o, R): H's origin and rotation in F (or the
station frame).

The lock (on the right sleeve) is a slider: its pawl tooth bears on the inboard face of the right post through a
window in the sleeve, its button stands out of the collar's outboard face. It is a prismatic joint along +y:
0 = at rest (the pawl in as far as it goes), positive = button pressed in (the pawl drawn back), up to LOCK_TRAVEL.

Each post's inboard face carries four notches. Going down, a notch's flat upper face stops on the pawl (to go lower
the button must be held in); going up, its sloped lower face pushes the pawl back (the headrest rises freely). The
first notch (nearest the tip) is square on both faces, so the headrest can't be pulled out without the button.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import station

TABLE = float(station.TABLE_TOP)
MM = 0.001

# -- the seat back section (cropped; F frame) -------------------------------------------------------------------------
SEAT = np.array([0.32, 0.0, TABLE + 0.105])    # F origin: midway between the sleeves, level with the collars' tops
C = float(SEAT[2] - TABLE)                     # the collars' tops stand 105 mm above the table
POST_Y = 0.100                                 # the sleeves' (and posts') axes at y = +/-POST_Y (200 mm apart)
COLLAR_H = 0.028                               # each collar stands 28 mm above the seat's top (the trim)
TRIM = -COLLAR_H                               # F z of the seat's top surface
BLOCK_X = 0.050                                # seat back section 100 mm thick (x), ...
BLOCK_Y = 0.220                                # ... 440 mm wide (the seat back's full width, its shoulders squared)
BORE = 0.008                                   # bore half width (16 mm: 1 mm play around a 14 mm post)
COLLAR = 0.014                                 # collar half width (28 mm square bezel)
FUNNEL_H, FUNNEL_TOP = 0.006, 0.013            # lead-in: the bore widens from 16 to 26 mm over the top 6 mm
LIP = 0.003                                    # collar / funnel wall thickness for the boxes

# -- the headrest (H frame) ---------------------------------------------------------------------------------------------
POST_R = 0.007                                 # 14 mm posts (steel tube, chromed)
POST_LEN = 0.130                               # each post stands 130 mm out of the cushion
POST_HIDDEN = 0.045                            # and runs 45 mm up inside it (visual only)
CUSH_H = 0.170                                 # cushion 170 mm tall ...
CUSH_W = 0.130                                 # ... 260 mm wide (half) ...
CUSH_T = 0.0525                                # ... 105 mm thick (half) across its middle ...
EAR_T = 0.032                                  # ... tapering to 64 mm at its ends (half thickness) ...
EAR_Y = 0.098                                  # ... outboard of |y| = 98 mm (32 mm wide end sections)
POST_MASS = 0.08                               # kg each (14 x 1.5 mm steel tube, 190 mm long)
CUSH_MASS = 0.84                               # kg: foam, cover and the steel frame inside
MASS = CUSH_MASS + 2 * POST_MASS               # 1.0 kg

# notches, measured as insertion depth: the depth of the post tips below the collars' tops when the pawl's centre is
# level with a notch's centre
PAWL_DEPTH = 0.012                             # the pawl's centre is 12 mm below the collar's top (below the lead-in)
DEPTHS = (0.050, 0.065, 0.080, 0.095)          # notches 1 (highest setting) .. 4 (lowest), 15 mm apart
NOTCHES = tuple(d - PAWL_DEPTH for d in DEPTHS)   # notch centres, measured up the post from its tip
NOTCH_H = 0.008                                # notch 8 mm tall ...
NOTCH_D = 0.003                                # ... and 3 mm deep
RAMP = 0.003                                   # sloped lower face (45 degrees) of notches 2 to 4
TIP_IN, TIP_RISE = 0.0055, 0.0095              # the tip's inboard edge chamfered 5.5 mm in over 9.5 mm (30 degrees)
PAWL_H = 0.005                                 # pawl tooth 5 mm tall, ...
PAWL_W = 0.008                                 # ... 8 mm wide (x), ...
PAWL_TIP = 0.0035                              # ... its face 3.5 mm from the bore's axis at rest (inboard side) ...
PAWL_BACK = 0.012                              # ... and running back to 12 mm
PAWL_Z = -PAWL_DEPTH
WINDOW = 1.5 * MM                              # the pawl fits in a notch over +/- this much (8 mm notch, 5 mm pawl)
SEATED = 1.5 * MM                              # locked and let down, the notch's flat face rests on the pawl:
LOCKED = tuple(d + SEATED for d in DEPTHS)     # the post tips are this deep
FLOOR = LOCKED[3]                              # the sleeves' closed ends stop the posts at the lowest locked depth

LOCK_TRAVEL = 0.0055                           # latch travel (pawl drawn back to 9 mm, clear of the bore)
LOCK_RATE = 200.0                              # N/m: the latch's spring, ...
LOCK_PRELOAD = 2.0                             # ... 2 N at rest, 3.1 N drawn fully back
LOCK_TARGET = -LOCK_PRELOAD / LOCK_RATE        # spring rest point beyond the stop (negative = further out)
LOCK_DAMPING = 5.0                             # N s/m
LOCK_MASS = 0.02
# the release button: on top of the bezel's outboard extension (its housing), pressed straight down; its stem's lower
# edge bears on the latch follower's 45-degree face and pushes it inboard, drawing the pawl back (pressed d, the latch
# moves d - BUTTON_GAP). Positions across the sleeve are r: y minus the right sleeve's y (positive inboard).
HOUSING = (-0.048, -COLLAR)                    # housing from 48 mm to 14 mm outboard of the right sleeve's axis
HOUSING_W = 0.010                              # 20 mm wide (x)
CAP = (-0.044, -0.028)                         # button cap 16 mm (y) x 14 mm (x), centred 36 mm outboard of the axis,
CAP_W = 0.007
BUTTON_PROUD = 0.006                           # standing 6 mm above the bezel at rest
BUTTON_GAP = 0.0003                            # free play before the wedge meets the follower
BUTTON_TRAVEL = 0.0075                         # its stop (the latch's stop comes first, at 5.8 mm)
BUTTON_RATE = 150.0                            # N/m: the button's return spring, ...
BUTTON_PRELOAD = 1.5                           # ... 1.5 N at rest
BUTTON_TARGET = -BUTTON_PRELOAD / BUTTON_RATE
BUTTON_DAMPING = 3.0
BUTTON_MASS = 0.01
FOLLOWER = (-0.026, -0.020)                    # the latch's follower block, ...
FOLLOWER_Z = (-0.016, -0.008)
SLOPE = ((-0.036, -0.018), (-0.026, -0.008))   # ... and its 45-degree face (r, z), facing outboard and up
STEM = (-0.032, SLOPE[1][0] - 0.0015 - BUTTON_GAP)   # the stem's inboard face touches the slope 9.5 mm down


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def quat_xyzw(r: np.ndarray) -> tuple:
    from yam_arena.ik import matrix_to_quat_xyzw

    return tuple(float(v) for v in matrix_to_quat_xyzw(r))


def to_station(p) -> np.ndarray:
    return np.asarray(p, float) + SEAT


# -- boxes: (centre, rotation, half sizes) ------------------------------------------------------------------------------
def lohi(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return ((lo + hi) / 2, np.eye(3), (hi - lo) / 2)


def slab(p0, p1, width: float, thick: float, plane: str, inward):
    """A thin box whose outer face is the segment p0 -> p1 in a vertical plane (``plane`` "yz": p = (y, z), the box
    spans x +/- width/2; "xz": p = (x, z), it spans y +/- width/2), extending ``thick`` toward ``inward`` (a 2-D
    direction in the same plane). Used for chamfers, ramps and the sleeves' lead-ins."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    d /= length
    n = np.array([-d[1], d[0]])
    if n @ np.asarray(inward, float) < 0:
        n = -n
    mid = (p0 + p1) / 2 + n * thick / 2
    if plane == "yz":
        # local y along the segment, local z along its normal: rotation about x
        r = np.column_stack([[1.0, 0.0, 0.0], [0.0, d[0], d[1]], [0.0, n[0], n[1]]])
        c = np.array([0.0, mid[0], mid[1]])
    else:
        # local x along the segment, local z along its normal: rotation about y
        r = np.column_stack([[d[0], 0.0, d[1]], [0.0, 1.0, 0.0], [n[0], 0.0, n[1]]])
        c = np.array([mid[0], 0.0, mid[1]])
    if np.linalg.det(r) < 0:
        r[:, 1] = -r[:, 1]
    return c, r, np.array([width / 2, length / 2, thick / 2]) if plane == "yz" else \
        np.array([length / 2, width / 2, thick / 2])


def shifted(box, offset):
    c, r, h = box
    return (np.asarray(c, float) + np.asarray(offset, float), r, h)


def posed(box, o, r):
    """A box given in H (or F), placed by the pose (o, R)."""
    c, rr, h = box
    return (np.asarray(o, float) + np.asarray(r) @ c, np.asarray(r) @ rr, h)


# -- the posts (H) ------------------------------------------------------------------------------------------------------
def post_boxes(side: float) -> dict:
    """One post's collision boxes in H; ``side`` +1 left, -1 right. Coordinates across the post are u, measured from
    its axis toward the other post (inboard, where the notches are): y = side * (POST_Y - u)."""
    def y_of(u0, u1):
        a, b = side * (POST_Y - u1), side * (POST_Y - u0)
        return min(a, b), max(a, b)

    def box(u0, u1, z0, z1):
        ya, yb = y_of(u0, u1)
        return lohi((-POST_R, ya, z0), (POST_R, yb, z1))

    r, land = POST_R, POST_R - NOTCH_D
    out = {"tip": box(-r, r - TIP_IN, 0.0, TIP_RISE), "core": box(-r, land, TIP_RISE, POST_LEN)}
    # the tip's chamfer, from (u = r - TIP_IN, z = 0) to (u = r, z = TIP_RISE), material toward -u, +z
    yz = lambda u, z: (side * (POST_Y - u), z)   # noqa: E731
    out["chamfer"] = slab(yz(r - TIP_IN, 0.0), yz(r, TIP_RISE), 2 * r, 0.003, "yz",
                          (side * 1.0, 1.0))
    edges = [TIP_RISE]
    for k, n in enumerate(NOTCHES):
        lower = n - NOTCH_H / 2 - (RAMP if k > 0 else 0.0)
        out[f"land{k}"] = box(land, r, edges[-1], lower)
        if k > 0:
            # sloped lower face: from the notch's floor (u = land, z = n - NOTCH_H/2) out to the land below
            out[f"ramp{k}"] = slab(yz(land, n - NOTCH_H / 2), yz(r, n - NOTCH_H / 2 - RAMP), 2 * r, 0.002, "yz",
                                   (side * 1.0, -1.0))
        edges.append(n + NOTCH_H / 2)
    out["land4"] = box(land, r, edges[-1], POST_LEN)
    return out


CUSHION = {
    "cushion": lohi((-CUSH_T, -EAR_Y, POST_LEN), (CUSH_T, EAR_Y, POST_LEN + CUSH_H)),
    "ear_left": lohi((-EAR_T, EAR_Y, POST_LEN), (EAR_T, CUSH_W, POST_LEN + CUSH_H)),
    "ear_right": lohi((-EAR_T, -CUSH_W, POST_LEN), (EAR_T, -EAR_Y, POST_LEN + CUSH_H)),
}


def headrest_boxes_h() -> dict:
    out = dict(CUSHION)
    for name, side in (("left", 1.0), ("right", -1.0)):
        out.update({f"post_{name}_{k}": b for k, b in post_boxes(side).items()})
    return out


HEADREST_H = headrest_boxes_h()
TIPS_H = {"left": np.array([0.0, POST_Y, 0.0]), "right": np.array([0.0, -POST_Y, 0.0])}
COM_H = np.array([0.0, 0.0, (CUSH_MASS * (POST_LEN + CUSH_H / 2) + 2 * POST_MASS * (POST_LEN + POST_HIDDEN - 0.095))
                  / MASS])


def headrest_boxes(o, r) -> dict:
    return {k: posed(b, o, r) for k, b in HEADREST_H.items()}


# -- the seat back section (F) ---------------------------------------------------------------------------------------------
def seat_boxes_f() -> dict:
    """The seat's fixed boxes in F: the upholstered block (with the two bores through it), each bore's closed end, the
    collars (bezels) with their lead-ins."""
    b, bottom = BORE, -C
    out = {
        "block_front": lohi((-BLOCK_X, -BLOCK_Y, bottom), (-b, BLOCK_Y, TRIM)),
        "block_back": lohi((b, -BLOCK_Y, bottom), (BLOCK_X, BLOCK_Y, TRIM)),
        "block_mid": lohi((-b, -POST_Y + b, bottom), (b, POST_Y - b, TRIM)),
        "block_left": lohi((-b, POST_Y + b, bottom), (b, BLOCK_Y, TRIM)),
        "block_right": lohi((-b, -BLOCK_Y, bottom), (b, -POST_Y - b, TRIM)),
    }
    for name, side in (("left", 1.0), ("right", -1.0)):
        y0 = side * POST_Y
        out[f"floor_{name}"] = lohi((-b, y0 - b, bottom), (b, y0 + b, -FLOOR))
        # collar walls from the trim up to the lead-in
        out[f"collar_{name}_front"] = lohi((-COLLAR, y0 - COLLAR, TRIM), (-b, y0 + COLLAR, -FUNNEL_H))
        out[f"collar_{name}_back"] = lohi((b, y0 - COLLAR, TRIM), (COLLAR, y0 + COLLAR, -FUNNEL_H))
        if side > 0:
            out[f"collar_{name}_in"] = lohi((-b, y0 + b, TRIM), (b, y0 + COLLAR, -FUNNEL_H))
        else:
            # the right sleeve's inboard wall has a slot for the pawl tooth
            out[f"collar_{name}_in"] = lohi((-b, y0 + b, TRIM), (b, y0 + COLLAR, PAWL_Z - PAWL_H / 2 - 0.001))
            out[f"collar_{name}_in_top"] = lohi((-b, y0 + b, PAWL_Z + PAWL_H / 2 + 0.001), (b, y0 + COLLAR,
                                                                                           -FUNNEL_H))
        out[f"collar_{name}_out"] = lohi((-b, y0 - COLLAR, TRIM), (b, y0 - b, -FUNNEL_H))
        # lead-in: four sloped faces from the bore (8 mm out, 6 mm down) to 13 mm out at the top
        for tag, plane, sgn in (("front", "xz", -1.0), ("back", "xz", 1.0), ("in", "yz", 1.0), ("out", "yz", -1.0)):
            if plane == "xz":
                box = slab((sgn * b, -FUNNEL_H), (sgn * FUNNEL_TOP, 0.0), 2 * COLLAR, LIP, "xz", (sgn, -1.0))
                out[f"funnel_{name}_{tag}"] = shifted(box, (0.0, y0, 0.0))
            else:
                box = slab((y0 + sgn * b, -FUNNEL_H), (y0 + sgn * FUNNEL_TOP, 0.0), 2 * COLLAR, LIP, "yz",
                           (sgn, -1.0))
                out[f"funnel_{name}_{tag}"] = box
    # the button's housing on the right bezel's outboard side: two side walls, an end wall and a floor, their tops
    # 9 mm below the bezel's top so that the fingertips pressing the cap (which stands on its stem above them) meet
    # nothing else
    y0, (r0, r1), w = -POST_Y, HOUSING, HOUSING_W
    top = -0.009
    out["housing_front"] = lohi((-w, y0 + r0, TRIM), (-CAP_W - 0.0005, y0 + r1, top))
    out["housing_back"] = lohi((CAP_W + 0.0005, y0 + r0, TRIM), (w, y0 + r1, top))
    out["housing_end"] = lohi((-CAP_W - 0.0005, y0 + r0, TRIM), (CAP_W + 0.0005, y0 + r0 + 0.002, top))
    out["housing_floor"] = lohi((-CAP_W - 0.0005, y0 + r0 + 0.002, TRIM), (CAP_W + 0.0005, y0 + r1, -0.024))
    return out


SEAT_F = seat_boxes_f()


def lock_boxes_f(q: float = 0.0, visual: bool = False) -> dict:
    """The latch's boxes in F, drawn back ``q`` (m): its pawl tooth (inboard of the right bore, through a slot in the
    bezel) and its follower (in the button's housing); with ``visual`` also the two bars joining them past the post
    (they don't collide)."""
    y0 = -POST_Y
    out = {
        "pawl": lohi((-PAWL_W / 2, y0 + PAWL_TIP + q, PAWL_Z - PAWL_H / 2),
                     (PAWL_W / 2, y0 + PAWL_BACK + q, PAWL_Z + PAWL_H / 2)),
        "follower": lohi((-0.005, y0 + FOLLOWER[0] + q, FOLLOWER_Z[0]), (0.005, y0 + FOLLOWER[1] + q, FOLLOWER_Z[1])),
        "slope": shifted(slab((y0 + SLOPE[0][0], SLOPE[0][1]), (y0 + SLOPE[1][0], SLOPE[1][1]), 0.010, 0.003, "yz",
                              (1.0, -1.0)), (0.0, q, 0.0)),
    }
    if visual:
        for s, tag in ((1.0, "front"), (-1.0, "back")):
            x0 = -s * (BORE + 0.001)
            out[f"bar_{tag}"] = lohi((min(x0, x0 - s * 0.002), y0 + FOLLOWER[1] + q, PAWL_Z - 0.0015),
                                     (max(x0, x0 - s * 0.002), y0 + PAWL_TIP + 0.004 + q, PAWL_Z + 0.0015))
    return out


def button_boxes_f(d: float = 0.0) -> dict:
    """The button's boxes in F, pressed down ``d`` (m): its cap and its stem, whose lower inboard edge bears on the
    follower's 45-degree face (BUTTON_GAP away at rest)."""
    y0 = -POST_Y
    out = {
        "cap": lohi((-CAP_W, y0 + CAP[0], 0.0), (CAP_W, y0 + CAP[1], BUTTON_PROUD)),
        "stem": lohi((-0.004, y0 + STEM[0], SLOPE[1][1] - 0.0015), (0.004, y0 + STEM[1], 0.0)),
    }
    return {k: shifted(b, (0.0, 0.0, -d)) for k, b in out.items()}


def latch_from_button(d: float) -> float:
    """The latch's travel for the button pressed ``d`` (the wedge at 45 degrees), up to its stop."""
    return float(min(max(0.0, d - BUTTON_GAP), LOCK_TRAVEL))


def seat_boxes_station() -> dict:
    return {k: shifted(b, SEAT) for k, b in SEAT_F.items()}


# -- the headrest's start: leaning on the seat back, post tips on the table ------------------------------------------------
REST_GAP = 0.0003


def _pose_at(lean: float):
    r = rot_y(lean)
    z0 = -C + REST_GAP + POST_R * math.sin(lean)          # lowest tip corner (x_H = +POST_R) on the table
    edge = r @ np.array([CUSH_T, 0.0, POST_LEN])          # the cushion's lower back edge, from H's origin
    return np.array([-edge[0], 0.0, z0]), r, z0 + edge[2]  # the edge over the collars' centres (x = 0)


def _solve_lean() -> float:
    lo, hi = math.radians(10), math.radians(35)           # edge height falls as the lean grows
    for _ in range(60):
        mid = (lo + hi) / 2
        if _pose_at(mid)[2] > REST_GAP:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


LEAN = _solve_lean()               # leaning back (top toward the seat) about 21 degrees


def start_pose():
    """(o, R) in F: post tips' lowest corners REST_GAP above the table, the cushion's lower back edge (middle
    section) resting REST_GAP above the collars' tops, over the sleeves' axes (x = 0)."""
    o, r, _ = _pose_at(LEAN)
    return o, r


# -- box maths --------------------------------------------------------------------------------------------------------------
def box_points(c, r, h, n: int = 5) -> np.ndarray:
    u = np.linspace(-1, 1, n)
    g = np.array([[a, b, d] for a in u for b in u for d in u])
    g = g[np.abs(g).max(axis=1) == 1.0]
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def box_corners(c, r, h) -> np.ndarray:
    g = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def point_box_dist(points, c, r, h) -> np.ndarray:
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    d = np.maximum(np.abs(local) - np.asarray(h), 0.0)
    return np.linalg.norm(d, axis=-1)


def point_box_depth(points, c, r, h) -> np.ndarray:
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    return np.maximum((np.asarray(h) - np.abs(local)).min(axis=-1), 0.0)


def boxes_gap(a, b, n: int = 7) -> float:
    """Approximate gap between two boxes (sampled; 0 if they touch or overlap)."""
    return float(min(point_box_dist(box_points(*a, n), *b).min(), point_box_dist(box_points(*b, n), *a).min()))


def depth_of(o, r, side: str) -> float:
    """How far a post's tip (its axis) is below the collars' top (F z = 0), for the headrest at (o, R) in F."""
    return float(-(np.asarray(o) + np.asarray(r) @ TIPS_H[side])[2])
