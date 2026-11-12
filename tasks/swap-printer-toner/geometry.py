"""Geometry of the toner-swap scene: a compact desktop laser printer with a hinged front door, its toner cartridge
bay (guide rails ending in a seat notch), and two toner cartridges. Pure NumPy, so the task file, the asset script
and the tests can import it before Isaac Sim starts.

Frames (metres):
- station: x forward from the arm bases, y to the robot's left, z up; the table top is at ``TABLE``.
- printer frame: origin on the table at the middle of the printer's front face; axes as the station's. The printer
  stands square to the robot, its front face ``X_FRONT`` ahead of the arm bases.
- door frame: origin on the hinge axis (bottom front edge of the bay opening). The door turns about the station's -y
  axis by its joint angle ``theta``: 0 is shut, positive swings its top toward the robot, 90 degrees is fully open.
- cartridge frame: origin at the centre of the cartridge's main body; at rest it is level and square to the
  printer, its handle on top and its two side bosses toward the back (+x).

A box is (name, centre, size, pitch, colour) in its frame; ``pitch`` turns it about +y (so its +x end goes down).
"""

from __future__ import annotations

import math

import numpy as np

TABLE = 0.7475                          # yam_arena.task_file.TABLE_TOP["station"]
X_FRONT = 0.27                          # printer front face, ahead of the arm bases

# -- printer body (printer frame) -----------------------------------------------------------------------------------
P_DEPTH, P_HALF_W, P_HEIGHT = 0.36, 0.20, 0.23
P_MASS = 7.5                            # kg, a compact monochrome laser printer
BASE_TOP = 0.05                         # bay floor (top of the paper-tray section)
FOOT = 0.008                            # rubber feet
FOOT_FRICTION = (0.9, 0.8)              # rubber feet on the table (static, dynamic)
ABS_FRICTION = (0.3, 0.25)              # the case, rails and door: smooth ABS
WALL_IN = 0.185                         # inner faces of the side walls (|y|)
TOP_FROM = 0.11                         # the fixed top panel starts this far behind the front face
TOP_UNDER = 0.215                       # underside of the top panel
HINGE_Z = 0.055                         # door hinge axis height (printer frame)
REAR_FROM = 0.16                        # print engine block behind the bay
TAN = math.tan(math.radians(15.0))      # guide rails fall 15 degrees toward the back
RAIL_Y = (0.176, WALL_IN)               # rails stand out from the side walls' inner faces
RAIL_T = 0.008                          # rail thickness
LIP_X = 0.095                           # rails end here (printer frame); the seat notch is behind
LIP_Z = 0.117                           # top of the rails' end (the notch's front lip)
NOTCH = (LIP_X, LIP_X + 0.014)          # seat notch, along x
SEAT_Z = 0.105                          # notch floor (where a seated boss bottom rests)
RAIL_FROM = 0.017                       # rails start just inside the door
FLARE = 0.015                           # lead-in flares in front of the rails, 8 mm open each way
SLOT = {"right": 0.016, "left": 0.024}  # rail slot heights per side (the bosses are 10 and 18 mm tall: keyed)
SUPPORT_X = (0.022, 0.040)              # support ribs on the bay floor (printer frame x) ...
SUPPORT_TOP = 0.0645                    # ... 0.5 mm under a seated cartridge's bottom
ROLLER = (0.06, 0.004, 0.006, 0.15)     # transfer roller: x, top below the bay floor, radius, half length


def rail_top(x: float) -> float:
    """Top of the lower guide rails at printer-frame ``x`` (on the slope)."""
    return LIP_Z + (LIP_X - x) * TAN


# -- door (door frame; hinge at printer (0, 0, HINGE_Z)) ------------------------------------------------------------
DOOR_HALF_W = 0.183
DOOR_T = 0.012
PANEL_H = TOP_UNDER + 0.003 - HINGE_Z   # front panel, 16.3 cm tall
FLANGE_L = TOP_FROM - 0.003             # the door's top flange covers the front of the bay from above
RIB = ((DOOR_T, 0.032), (0.105, 0.113), 0.12)   # pressure rib on the panel's inside: x, z, half width
DOOR_MASS = 0.35
OPEN = math.radians(80.0)               # start: fully open (its stop), leaning toward the robot
DOOR_DAMPING = 0.02                     # N m s / rad


def door_boxes():
    t, h, w = DOOR_T, PANEL_H, 2 * DOOR_HALF_W
    (rx0, rx1), (rz0, rz1), rw = RIB
    return [
        ("panel", (t / 2, 0.0, h / 2), (t, w, h), 0.0),
        ("flange", (FLANGE_L / 2, 0.0, h + t / 2), (FLANGE_L, w, t), 0.0),
        ("rib", ((rx0 + rx1) / 2, 0.0, (rz0 + rz1) / 2), (rx1 - rx0, 2 * rw, rz1 - rz0), 0.0),
    ]


def door_com() -> np.ndarray:
    """Door-frame centre of mass (uniform moulding)."""
    vols = [(np.prod(s), np.asarray(c)) for _, c, s, _ in door_boxes()]
    return sum(v * c for v, c in vols) / sum(v for v, _ in vols)


def door_rot(theta: float) -> np.ndarray:
    """Door-frame axes in the printer/station frame at angle ``theta`` (rotation about -y)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def hinge_world() -> np.ndarray:
    return np.array([X_FRONT, 0.0, TABLE + HINGE_Z])


def door_to_world(p, theta: float) -> np.ndarray:
    return hinge_world() + door_rot(theta) @ np.asarray(p, float)


# -- printer body boxes (printer frame) -----------------------------------------------------------------------------
def printer_boxes():
    """(name, centre, size, pitch) of every collider of the printer body, printer frame."""
    d, w, h = P_DEPTH, P_HALF_W, P_HEIGHT
    out = []
    for i, (fx, fy) in enumerate(((0.03, 0.17), (0.03, -0.17), (d - 0.03, 0.17), (d - 0.03, -0.17))):
        out.append((f"foot{i}", (fx, fy, FOOT / 2), (0.03, 0.03, FOOT), 0.0))
    out.append(("base", (d / 2, 0.0, (FOOT + BASE_TOP) / 2), (d, 2 * w, BASE_TOP - FOOT), 0.0))
    for side, sy in (("left", 1), ("right", -1)):
        out.append((f"wall_{side}", (d / 2, sy * (WALL_IN + w) / 2, (BASE_TOP + h) / 2),
                    (d, w - WALL_IN, h - BASE_TOP), 0.0))
    out.append(("top", ((TOP_FROM + d) / 2, 0.0, (TOP_UNDER + h) / 2), (d - TOP_FROM, 2 * WALL_IN, h - TOP_UNDER), 0.0))
    # two support ribs on the bay floor under the cartridge's front: seated, it rests on them and on its bosses
    for i, sy in enumerate((1, -1)):
        out.append((f"support{i}", ((SUPPORT_X[0] + SUPPORT_X[1]) / 2, sy * 0.11, (BASE_TOP + SUPPORT_TOP) / 2),
                    (SUPPORT_X[1] - SUPPORT_X[0], 0.04, SUPPORT_TOP - BASE_TOP), 0.0))
    out.append(("engine", ((REAR_FROM + d) / 2, 0.0, (BASE_TOP + TOP_UNDER) / 2),
                (d - REAR_FROM, 2 * WALL_IN, TOP_UNDER - BASE_TOP), 0.0))
    pitch = math.atan(TAN)
    length = (LIP_X - RAIL_FROM) / math.cos(pitch)
    flare_pitch = math.atan(0.008 / FLARE)
    for side, sy in (("left", 1), ("right", -1)):
        yc, yw = sy * (RAIL_Y[0] + RAIL_Y[1]) / 2, RAIL_Y[1] - RAIL_Y[0]
        mid = (RAIL_FROM + LIP_X) / 2
        # lower rail: its top surface is rail_top(x); the box centre is RAIL_T/2 below it, square to the slope
        nrm = np.array([math.sin(pitch), 0.0, math.cos(pitch)])     # the slope's upward normal
        top_mid = np.array([mid, yc, rail_top(mid)])
        out.append((f"rail_{side}_low", tuple(top_mid - nrm * RAIL_T / 2), (length, yw, RAIL_T), pitch))
        # upper rail: SLOT above the lower one, all the way to the lip ...
        out.append((f"rail_{side}_up", tuple(top_mid + nrm * (SLOT[side] * math.cos(pitch) + RAIL_T / 2)),
                    (length, yw, RAIL_T), pitch))
        # ... then a flat ceiling over the lip and the notch to the back stop, SLOT above the lip: a boss can rise
        # out of its notch only to the slot's height, and can reach the notch only along the slot
        cz = LIP_Z + SLOT[side]
        out.append((f"ceiling_{side}", ((LIP_X - 0.001 + NOTCH[1] + 0.015) / 2, yc, cz + RAIL_T / 2),
                    (NOTCH[1] + 0.016 - LIP_X, yw, RAIL_T), 0.0))
        # lead-in flares: 15 mm long, opening 8 mm down and up toward the front
        fx = RAIL_FROM - FLARE / 2
        z0 = rail_top(RAIL_FROM)
        fl = math.hypot(FLARE, 0.008)
        out.append((f"flare_{side}_low", (fx, yc, z0 - 0.004 - RAIL_T / 2), (fl, yw, RAIL_T), -flare_pitch))
        out.append((f"flare_{side}_up", (fx, yc, z0 + SLOT[side] + 0.004 + RAIL_T / 2), (fl, yw, RAIL_T),
                    flare_pitch))
        # seat notch floor and back stop
        out.append((f"seat_{side}", ((NOTCH[0] + NOTCH[1]) / 2, yc, (BASE_TOP + SEAT_Z) / 2),
                    (NOTCH[1] - NOTCH[0], yw, SEAT_Z - BASE_TOP), 0.0))
        out.append((f"stop_{side}", (NOTCH[1] + 0.0075, yc, (BASE_TOP + 0.16) / 2), (0.015, yw, 0.16 - BASE_TOP),
                    0.0))
    return out


ROLLER_C = np.array([ROLLER[0], 0.0, BASE_TOP + ROLLER[1] - ROLLER[2]])   # roller axis centre, printer frame


def printer_to_world(p) -> np.ndarray:
    return np.asarray(p, float) + np.array([X_FRONT, 0.0, TABLE])


# -- cartridges (cartridge frame) -----------------------------------------------------------------------------------
C_SIZE = (0.10, 0.34, 0.09)             # main body: depth (x), length (y), height (z)
FIN = ((-0.02, 0.02), 0.008, 0.045)     # handle on top: x range, half thickness (y), height above the body
BOSS_X = 0.035                          # side bosses, toward the back of the body
BOSS = {"right": (0.010, 0.010), "left": (0.010, 0.018)}   # (x, z) size; bottoms level with each other
BOSS_OUT = 0.0135                       # how far each boss stands out from the body's end (1.5 mm from the wall)
MASS = {"new": 0.70, "old": 0.55}       # kg (the used one has less toner left)
FIN_CENTRE = np.array([0.0, 0.0, C_SIZE[2] / 2 + FIN[2] / 2])


def cartridge_boxes():
    dx, dy, dz = C_SIZE
    (fx0, fx1), fy, fh = FIN
    out = [("body", (0.0, 0.0, 0.0), (dx, dy, dz), 0.0),
           ("handle", ((fx0 + fx1) / 2, 0.0, dz / 2 + fh / 2), (fx1 - fx0, 2 * fy, fh), 0.0)]
    for side, sy in (("left", 1), ("right", -1)):
        bx, bz = BOSS[side]
        out.append((f"boss_{side}", (BOSS_X, sy * (dy / 2 + BOSS_OUT / 2), -0.005 + bz / 2), (bx, BOSS_OUT, bz), 0.0))
    return out


# Seated: both bosses down in the middle of their seat notches; the body's centre, printer frame.
SEAT_C = np.array([(NOTCH[0] + NOTCH[1]) / 2 - BOSS_X, 0.0, SEAT_Z + 0.005])
FRONT_GAP = SEAT_C[0] - C_SIZE[0] / 2 - DOOR_T     # seated cartridge's front face to the shut door's inside


def line_centre(boss_x: float, lift: float = 0.0) -> np.ndarray:
    """Cartridge centre (printer frame) with its bosses' bottoms on the rail line at printer-frame ``boss_x`` (the
    line is continued in front of the rails), raised ``lift``."""
    return np.array([boss_x - BOSS_X, 0.0, rail_top(boss_x) + 0.005 + lift])


TABLE_C = {"old": np.array([0.205, 0.40]), "new": np.array([0.205, -0.40])}   # where each one rests on the table
NEW_START = np.array([*TABLE_C["new"], TABLE + C_SIZE[2] / 2])                 # station
OLD_START = printer_to_world(SEAT_C + [0.0, 0.0, 0.0005])                      # station: seated in the printer


# -- simple box maths -----------------------------------------------------------------------------------------------
def rot_y(pitch: float) -> np.ndarray:
    c, s = math.cos(pitch), math.sin(pitch)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def world_boxes(boxes, origin, rot):
    """Boxes (name, centre, size, pitch) in a frame placed at ``origin`` with axes ``rot``: (name, centre, rotation,
    half sizes) in the station frame."""
    return [(n, np.asarray(origin, float) + rot @ np.asarray(c, float), rot @ rot_y(p), np.asarray(s, float) / 2)
            for n, c, s, p in boxes]


def printer_world():
    return world_boxes(printer_boxes(), printer_to_world((0, 0, 0)), np.eye(3))


def door_world(theta: float):
    return world_boxes(door_boxes(), hinge_world(), door_rot(theta))


def cartridge_world(centre, rot=np.eye(3)):
    return world_boxes(cartridge_boxes(), centre, rot)


def box_points(c, r, h, n: int = 5) -> np.ndarray:
    t = np.linspace(-1.0, 1.0, n)
    g = np.array(np.meshgrid(t, t, t)).reshape(3, -1).T
    return c + (g * h) @ r.T


def point_box_dist(points, c, r, h) -> np.ndarray:
    """Distance from each point to a box (0 inside)."""
    local = (np.atleast_2d(points) - c) @ r
    return np.linalg.norm(np.maximum(np.abs(local) - h, 0.0), axis=1)


def point_box_depth(points, c, r, h) -> np.ndarray:
    """How deep each point is inside a box (0 outside)."""
    local = (np.atleast_2d(points) - c) @ r
    inside = np.all(np.abs(local) < h, axis=1)
    return np.where(inside, np.min(h - np.abs(local), axis=1), 0.0)


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
