"""Dimensions of the booster seat and its tray, the ratchet that holds the tray, the tests the scorer uses, and the
demo's grip poses. Pure NumPy (no Isaac Sim), shared by task.py, the tests and the README.

Frames:
- booster frame: origin on the table under the middle of the seat's front edge; x toward the backrest, y to the
  robot's left, z up. The booster faces the robot, so its x is the station's x.
- tray frame: origin at the middle of the tray plate's underside; x toward the tray's back edge (the child's side),
  which goes toward the backrest when the tray is on.
Lengths in metres. "s" is +1 for the left side (y > 0) and -1 for the right.
"""

from __future__ import annotations

import math

import numpy as np

TABLE = 0.7475  # station table top (yam_arena.task_file.TABLE_TOP["station"])

# -- the booster seat (moulded polypropylene shell, about 40 x 33 x 23 cm, 2.5 kg) ----------------------------------
BOOSTER_XY = (0.455, 0.0)        # booster frame origin on the station table
BOOSTER_MASS = 2.5
BOOSTER_W, BOOSTER_D, BOOSTER_H = 0.40, 0.33, 0.23
SEAT_TOP = 0.03                 # seat surface: a low feeding-booster shell, 3 cm above the chair
ARM_IN, ARM_OUT = 0.135, 0.175  # each armrest spans |y| 13.5 - 17.5 cm (4 cm wide)
ARM_X0, ARM_X1 = 0.03, 0.29     # armrest front (with a 1.5 cm 45-degree lead-in on its top edge) and back
ARM_TOP = 0.15                  # armrest top: the tray plate rests here, 12 cm above the seat (lap room)
LEAD = 0.015                    # the lead-in chamfer on the armrest's front top edge

# The rack on each armrest's outer face: three teeth, each with a 45-degree ramp facing the front and a square face
# toward the back, then a square stop. A lip along the top closes the notches from above.
RACK_DEPTH = 0.008              # teeth stand 8 mm out from the armrest's outer face
TOOTH0 = 0.10                   # the first tooth's ramp starts here (booster x)
PITCH = 0.025                   # notch pitch
TOOTH_ROOT = 0.016              # tooth length at the root: 8 mm ramp + 8 mm flat crest
STOP_X = TOOTH0 + 3 * PITCH     # 0.175: the stop behind notch 3
BAND = (ARM_TOP - 0.040, ARM_TOP - 0.010)   # teeth and stop, booster z
LIP = (ARM_TOP - 0.010, ARM_TOP)            # lip over the notches, flush with the armrest top
LIP_X0 = 0.075                  # the lip starts 2.5 cm ahead of the first tooth
STOP_OUT = 0.020                # the stop stands 2 cm out: the catch can't ride over it
N_NOTCHES = 3


def tooth_start(k: int) -> float:
    """Booster x where tooth k's ramp starts (k = 0, 1, 2); notch k+1 lies behind tooth k."""
    return TOOTH0 + PITCH * k


def notch_seat(n: int) -> float:
    """Booster x of a catch's centre when it sits in notch ``n`` (1-3) pulled back against the tooth in front."""
    return tooth_start(n - 1) + TOOTH_ROOT + CATCH_W / 2


def notch_window(n: int) -> tuple[float, float]:
    """The catch centre's booster x range inside notch ``n``. Behind it, the tooth in front; ahead of it, the next
    ramp, which is 4 mm deep (the engaged limit) 4 mm past its start: a catch centred further on is riding it."""
    return tooth_start(n - 1) + TOOTH_ROOT - 0.004, tooth_start(n) + 0.001


# -- the tray (rigid polypropylene, 42 x 25 cm, 0.45 kg) -------------------------------------------------------------
TRAY_W, TRAY_D = 0.42, 0.25
TRAY_MASS = 0.45
PLATE_T = 0.006
FRONT_RIM = (0.004, 0.030)      # the front rim (the robot's side): 4 mm thick, 3 cm above the plate's top
SIDE_RIM_H = 0.020
RAIL_IN = (0.1285, 0.1335)      # rail walls under the plate, just inboard of each armrest (1.5 mm clearance)
RAIL_X = (-0.10, 0.12)
RAIL_DEPTH = 0.032              # rails and skirts reach 3.2 cm below the plate
SKIRT = (0.205, 0.209)          # outer skirt at each end, open around the catch
# The catch: a sprung slide in each tray end, moving along y. At rest its 8 mm wide, round-nosed tip reaches in to 2 mm
# off the armrest's face; it rides 6 mm out over a tooth (10 mm travel), and a 1.6 cm lug outside the tray end pulls it out.
CATCH_X = 0.10                  # tray x of the catches
CATCH_W = 0.008
CATCH_IN = 0.177                # |y| of the tip's inner face at rest
CATCH_Z = (-0.031, -0.017)      # tray z of the tip (7 mm under the lip, inside the teeth's band, when seated)
CATCH_TRAVEL = 0.010
CATCH_RATE = 600.0              # N/m: 3.6 N at the crest, about 12 N in all to click the tray on
CATCH_MASS = 0.02
LUG = (0.205, 0.218)
CATCH_REST_Y = 0.19             # |y| of the slide's frame

# -- the start and the notch-2 pose ----------------------------------------------------------------------------------
START_OFFSET = -0.13            # tray origin's booster x at the start: lying on the table, back edge 5 mm in front
SEATED_Z = ARM_TOP              # tray origin's booster z when it rests on the armrests


def offset_for_catch(x_catch: float) -> float:
    """Tray origin's booster x that puts the catches' centres at booster x ``x_catch``."""
    return x_catch - CATCH_X


def backrest_gap(n: int) -> float:
    """Room between the tray's back edge and the backrest with the catches in notch ``n``: the child's tummy."""
    return (BOOSTER_D - 0.04) - (offset_for_catch(notch_seat(n)) + TRAY_D / 2)


def front_overhang(n: int) -> float:
    """How far the tray's front edge sticks out past the booster's front face with the catches in notch ``n`` (the
    real-world check with a ruler)."""
    return -(offset_for_catch(notch_seat(n)) - TRAY_D / 2)


# -- scoring tests ---------------------------------------------------------------------------------------------------
ENGAGED_DEPTH = 0.004    # a catch is down in a notch when its tip is within 4 mm of the notch floor (half its depth)
SEATED_TOL = 0.005       # tray plate within 5 mm of the armrest tops
SEATED_TILT = 5.0        # deg
LATERAL_TOL = 0.006      # tray centred between the armrests within 6 mm
BOOSTER_MOVE = 0.03      # the booster moved more than 3 cm, or tipped more than 10 degrees: knocked
BOOSTER_TIP = 10.0
DROP_SPEED = 0.6         # m/s downward while no gripper holds it: the tray fell (about a 2 cm free fall)
RELEASE = 0.02           # every finger point at least 2 cm from the tray
STILL_TRAY = 0.01        # m/s
STILL_TCP = 0.001        # m per control step (2 cm/s)


def quat_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_deg(rot: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(rot[2, 2])))))


def tray_in_booster(booster_pos, booster_quat, tray_pos, tray_quat):
    """Tray origin in the booster frame, and the tray's rotation relative to the booster."""
    rb, rt = quat_matrix(booster_quat), quat_matrix(tray_quat)
    return rb.T @ (np.asarray(tray_pos, float) - np.asarray(booster_pos, float)), rb.T @ rt


def catch_tip(rel_pos, rel_rot, s: int, q: float) -> np.ndarray:
    """Booster-frame (x of its centre, |y| of its inner face) of side ``s``'s catch tip, pushed out by ``q``."""
    inner = np.array([CATCH_X, s * (CATCH_IN + abs(q)), sum(CATCH_Z) / 2])
    p = rel_pos + rel_rot @ inner
    return np.array([p[0], abs(p[1])])


def catch_notch(rel_pos, rel_rot, s: int, q: float) -> int | None:
    """The notch (1-3) side ``s``'s catch is down in, or None. Down in a notch: the tray seated on the armrests and
    centred between them, the catch tip within 4 mm of the notch floor (the armrest's face; teeth stand 8 mm proud),
    and the catch's centre inside that notch."""
    if not seated(rel_pos, rel_rot):
        return None
    x, y_in = catch_tip(rel_pos, rel_rot, s, q)
    if y_in - ARM_OUT > ENGAGED_DEPTH:
        return None
    for n in range(1, N_NOTCHES + 1):
        lo, hi = notch_window(n)
        if lo <= x <= hi:
            return n
    return None


def seated(rel_pos, rel_rot) -> bool:
    """The tray plate rests flat on the armrests (within 5 mm and 5 degrees), centred between them, and far enough
    back that both rails straddle the armrests by at least 3 cm."""
    return bool(abs(rel_pos[2] - SEATED_Z) <= SEATED_TOL and tilt_deg(rel_rot) <= SEATED_TILT
                and abs(rel_pos[1]) <= LATERAL_TOL and rel_pos[0] + RAIL_X[1] >= ARM_X0 + 0.03)


def _mm_box(x0, x1, y0, y1, z0, z1, color, rot=None):
    spec = (((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), color)
    return spec + ((rot,) if rot is not None else ())


def _span(s, a, b):
    """The y interval between |y| = a and b on side s, low end first."""
    return (min(s * a, s * b), max(s * a, s * b))


def _rot(axis: str, deg: float):
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return (math.cos(h), *(math.sin(h) * c for c in v))


def booster_boxes(shell, rack):
    """The booster as boxes (centre, size, colour[, rotation wxyz]) in the booster frame."""
    out = [
        _mm_box(0.0, BOOSTER_D, -BOOSTER_W / 2, BOOSTER_W / 2, 0.008, SEAT_TOP, shell),          # seat block
        _mm_box(BOOSTER_D - 0.04, BOOSTER_D, -BOOSTER_W / 2, BOOSTER_W / 2, SEAT_TOP, BOOSTER_H, shell),  # back
    ]
    for fx in (0.03, BOOSTER_D - 0.03):                                                       # non-slip feet
        for fy in (-0.17, 0.17):
            out.append(_mm_box(fx - 0.02, fx + 0.02, fy - 0.02, fy + 0.02, 0.0, 0.008, (0.15, 0.15, 0.15)))
    side = math.sqrt(2) * LEAD
    for s in (1, -1):
        y0, y1 = _span(s, ARM_IN, ARM_OUT)
        out += [
            _mm_box(ARM_X0 + LEAD, ARM_X1, y0, y1, SEAT_TOP, ARM_TOP, shell),                  # armrest
            _mm_box(ARM_X0, ARM_X0 + LEAD, y0, y1, SEAT_TOP, ARM_TOP - LEAD, shell),           # its nose
            # lead-in: a square turned 45 degrees about y whose upper front face runs from the nose up to the top
            (((ARM_X0 + LEAD, (y0 + y1) / 2, ARM_TOP - LEAD)), (side, y1 - y0, side), shell, _rot("y", 45)),
        ]
        ly0, ly1 = _span(s, ARM_OUT, ARM_OUT + RACK_DEPTH)
        out.append(_mm_box(LIP_X0, ARM_X1 - 0.01, ly0, ly1, *LIP, rack))                       # lip
        for k in range(N_NOTCHES):
            t = tooth_start(k)
            out.append(_mm_box(t + RACK_DEPTH, t + TOOTH_ROOT, ly0, ly1, *BAND, rack))         # crest and square face
            # the ramp: a square turned 45 degrees about z, centred on the armrest face under the crest's front
            # corner; its outer front face runs from the face at x = t out to the crest
            # (its apex stands 0.5 mm proud of the crest, so a catch sliding up never meets the crest's front edge)
            d = math.sqrt(2) * (RACK_DEPTH + 0.0005)
            out.append(((t + RACK_DEPTH, s * ARM_OUT, sum(BAND) / 2), (d, d, BAND[1] - BAND[0]), rack, _rot("z", 45)))
        sy0, sy1 = _span(s, ARM_OUT, ARM_OUT + STOP_OUT)
        out.append(_mm_box(STOP_X, ARM_X1 - 0.01, sy0, sy1, *BAND, rack))                     # stop
    return out


def tray_boxes(color, rim):
    """The tray body as boxes in the tray frame (the catches are separate sliders)."""
    hx, hy = TRAY_D / 2, TRAY_W / 2
    top = PLATE_T
    out = [
        _mm_box(-hx, hx, -hy, hy, 0.0, top, color),                                              # plate
        _mm_box(-hx, -hx + FRONT_RIM[0], -hy, hy, top, top + FRONT_RIM[1], rim),                  # front rim
        _mm_box(hx - 0.004, hx, -hy + 0.004, hy - 0.004, top, top + 0.010, color),               # back lip
    ]
    for s in (1, -1):
        out.append(_mm_box(-hx + FRONT_RIM[0], hx - 0.004, *_span(s, hy - 0.004, hy), top, top + SIDE_RIM_H, color))
        out.append(_mm_box(*RAIL_X, *_span(s, *RAIL_IN), -RAIL_DEPTH, 0.0, color))              # rail wall
        for xa, xb in ((RAIL_X[0], CATCH_X - 0.012), (CATCH_X + 0.012, RAIL_X[1])):
            out.append(_mm_box(xa, xb, *_span(s, *SKIRT), -RAIL_DEPTH, 0.0, color))           # skirt
    return out


def catch_boxes(s: int, color):
    """Side ``s``'s catch (stem, then the pull lug) in the tray frame; its round nose is :func:`catch_nose`."""
    return (
        _mm_box(CATCH_X - CATCH_W / 2, CATCH_X + CATCH_W / 2, *_span(s, CATCH_IN + CATCH_W / 2, LUG[0]), *CATCH_Z,
                color),
        _mm_box(CATCH_X - 0.008, CATCH_X + 0.008, *_span(s, *LUG), CATCH_Z[0], CATCH_Z[1] + 0.004, color),
    )


def catch_nose(s: int, color):
    """The catch tip's round nose: a vertical cylinder 8 mm across whose inner side is the tip's inner face. Against a
    tooth's square back face it bears flat (holds); against a ramp it slides."""
    r = CATCH_W / 2
    return ((CATCH_X, s * (CATCH_IN + r), sum(CATCH_Z) / 2), r, CATCH_Z[1] - CATCH_Z[0], color)


def point_box_distance(p, centre, size, rot_wxyz=None) -> float:
    d = np.asarray(p, float) - np.asarray(centre, float)
    if rot_wxyz is not None:
        w, x, y, z = rot_wxyz
        d = quat_matrix((x, y, z, w)).T @ d
    q = np.abs(d) - np.asarray(size, float) / 2
    return float(np.linalg.norm(np.maximum(q, 0.0)) + min(max(q[0], q[1], q[2]), 0.0))


TRAY_BOXES_PLAIN = tray_boxes((0, 0, 0), (0, 0, 0))


def tray_clearance(points, tray_pos, tray_quat) -> float:
    """Smallest distance from any of ``points`` (world) to the tray's boxes (catches included at rest)."""
    rt = quat_matrix(tray_quat)
    boxes = TRAY_BOXES_PLAIN + [b for s in (1, -1) for b in catch_boxes(s, (0, 0, 0))]
    best = math.inf
    for p in points:
        local = rt.T @ (np.asarray(p, float) - np.asarray(tray_pos, float))
        for b in boxes:
            best = min(best, point_box_distance(local, b[0], b[1]))
    return best


# -- the demo's grip (both arms, fingers closing along x across the front rim) ---------------------------------------
GRIP_Y = 0.19            # |y| of each grip on the front rim
GRIP_YAW = math.pi / 2   # fingers close along x
RIM_X = -TRAY_D / 2 + FRONT_RIM[0] / 2   # tray x of the front rim's centre
GRIP_BELOW_RIM_TOP = 0.010               # TCP 1 cm below the rim's top: pad tips 7 mm above the plate
TCP_IN_TRAY_Z = PLATE_T + FRONT_RIM[1] - GRIP_BELOW_RIM_TOP


def grip_tcp(offset: float, tray_z: float, s: int) -> tuple[float, float, float]:
    """World TCP for side ``s``'s grip with the tray origin at booster x ``offset`` and booster z ``tray_z``
    (the booster at its start pose)."""
    return (float(BOOSTER_XY[0] + offset + RIM_X), float(BOOSTER_XY[1] + s * GRIP_Y),
            float(TABLE + tray_z + TCP_IN_TRAY_Z))
