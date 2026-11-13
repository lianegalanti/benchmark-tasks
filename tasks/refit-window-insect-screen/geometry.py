"""Dimensions, layout and box maths for the insect-screen task. Pure NumPy; shared by make_assets.py, task.py, demo.py
and the tests.

Station frame: x forward, y to the robot's left, z up (m). The window frame W is the station frame shifted to
``WINDOW``: its origin is the centre of the bottom channel's floor, in the plane the screen's mid-thickness takes when
it is fitted. The window faces the robot: the room (where the robot is) is at x < 0, the glass at x > 0.

The screen frame S has its origin at the centre of the screen's bottom edge (bottom face of the bottom rail, mid
thickness); x is the screen's normal (+x is its back, toward the glass), y along its width, z up its stiles. Fitted,
S coincides with W. A screen pose is written (o, R): S origin and rotation in the W (or station) frame.

The screen is an articulation with a floating root ``frame`` (four aluminium rails and the mesh) and two spring
plungers ``plunger_left`` / ``plunger_right`` on its top rail, each a prismatic joint along S's z axis: 0 = at rest
(pin standing ``PIN_PROUD`` above the rail), negative = pushed in, down to ``-PIN_TRAVEL`` (flush).
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import station

TABLE = float(station.TABLE_TOP)
MM = 0.001

# -- the screen (a made-to-measure aluminium insect screen for a small window) ------------------------------------------
SCREEN_W, SCREEN_H = 0.450, 0.600      # m: overall width (y) and height (z)
FRAME_T = 0.010                        # m: frame profile thickness (x, normal to the screen)
RAIL_B = 0.020                         # m: frame profile face width (in the screen's plane)
MESH_T = 0.001                         # m: the mesh panel (fibreglass mesh, rigid in the sim) at mid-thickness
FRAME_MASS = 0.43                      # kg: rails, corner keys and mesh
PIN_MASS = 0.01                        # kg each
SCREEN_MASS = FRAME_MASS + 2 * PIN_MASS
PIN_Y = 0.150                          # m: the plungers sit at y = +/-PIN_Y on the top rail
PIN_X, PIN_YW = 0.008, 0.010           # m: pin section (x by y)
PIN_PROUD = 0.019                      # m: pin stands this far above the rail at rest
PIN_TRAVEL = 0.019                     # m: and can be pushed in this far (flush with the rail)
PIN_LEN = 0.030                        # m: pin length (the rest is inside the rail's plunger housing)
PIN_RATE = 120.0                       # N/m each
PIN_PRELOAD = 0.5                      # N at rest (against its stop)
PIN_TARGET = PIN_PRELOAD / PIN_RATE    # spring rest point, beyond the stop (positive = further out)
PIN_DAMPING = 1.0                      # N s/m

# root boxes (lo, hi) in S
_hx, _hy = FRAME_T / 2, SCREEN_W / 2
RAILS = {
    "bottom_rail": ((-_hx, -_hy, 0.0), (_hx, _hy, RAIL_B)),
    "top_rail": ((-_hx, -_hy, SCREEN_H - RAIL_B), (_hx, _hy, SCREEN_H)),
    "stile_left": ((-_hx, _hy - RAIL_B, RAIL_B), (_hx, _hy, SCREEN_H - RAIL_B)),
    "stile_right": ((-_hx, -_hy, RAIL_B), (_hx, -_hy + RAIL_B, SCREEN_H - RAIL_B)),
}
MESH = ((-MESH_T / 2, -_hy + RAIL_B, RAIL_B), (MESH_T / 2, _hy - RAIL_B, SCREEN_H - RAIL_B))


def pin_box(sign: float, q: float = 0.0):
    """A plunger pin (lo, hi) in S at joint position ``q`` (<= 0 pushed in)."""
    y = sign * PIN_Y
    top = SCREEN_H + PIN_PROUD + q
    return (-PIN_X / 2, y - PIN_YW / 2, top - PIN_LEN), (PIN_X / 2, y + PIN_YW / 2, top)


# -- the window frame section (W frame; fixed) -----------------------------------------------------------------------
WINDOW = np.array([0.36, 0.0, TABLE + 0.050])   # W origin in the station frame (the sill stands 50 mm tall)
SILL_H = float(WINDOW[2] - TABLE)
LENGTH = SCREEN_W / 2 + 0.010           # head and sill run to y = +/-LENGTH (cropped there)
BOT_HALF = 0.0075                       # bottom channel: 15 mm wide inside (2.5 mm play each side of the frame) ...
BOT_LIP_FRONT = 0.008                   # ... front (room-side) lip 8 mm tall, ...
BOT_LIP_BACK = 0.015                    # ... back lip 15 mm tall
LIP_T = 0.003                           # lips 3 mm thick (aluminium extrusion)
TOP_HALF = 0.011                        # top channel: 22 mm wide inside ...
TOP_DEPTH = 0.024                       # ... and 24 mm deep
GAP = 0.016                             # fitted, the top rail stands this far below the channel's ceiling
CEIL = SCREEN_H + GAP                   # the top channel's ceiling (W z)
MOUTH = CEIL - TOP_DEPTH                # the top channel's lips end here (W z)
SEAT_PRESS = PIN_PROUD - GAP            # fitted, each pin is pushed in this far (3 mm): the plungers hold it down
FRONT_X, BACK_X = -0.025, 0.075         # head and sill run from the room-side face to the back of the frame
JAMB_X = 0.040                          # the jambs' (and the glazing's) room-side face, 35 mm behind the screen's back
JAMB_Y = (SCREEN_W / 2 - 0.045, SCREEN_W / 2 - 0.005)   # jamb |y| span: the screen overlaps each jamb by 35 mm
GLASS_X = (0.052, 0.058)
STOP_Y = (SCREEN_W / 2 + 0.003, LENGTH)   # end stops closing both channels' ends, 3 mm beyond the screen's sides
STOP_H = 0.015                          # m: each stop stands this far above the channel floor / below the ceiling

FRAME_BOXES = {      # name: (lo, hi) in W
    "sill": ((FRONT_X, -LENGTH, -SILL_H), (BACK_X, LENGTH, 0.0)),
    "sill_lip_front": ((-BOT_HALF - LIP_T, -LENGTH, 0.0), (-BOT_HALF, LENGTH, BOT_LIP_FRONT)),
    "sill_lip_back": ((BOT_HALF, -LENGTH, 0.0), (BOT_HALF + LIP_T, LENGTH, BOT_LIP_BACK)),
    "head": ((FRONT_X, -LENGTH, CEIL), (BACK_X, LENGTH, CEIL + 0.030)),
    "head_lip_front": ((-TOP_HALF - LIP_T, -LENGTH, MOUTH), (-TOP_HALF, LENGTH, CEIL)),
    "head_lip_back": ((TOP_HALF, -LENGTH, MOUTH), (TOP_HALF + LIP_T, LENGTH, CEIL)),
    "jamb_left": ((JAMB_X, JAMB_Y[0], 0.0), (BACK_X, JAMB_Y[1], CEIL)),
    "jamb_right": ((JAMB_X, -JAMB_Y[1], 0.0), (BACK_X, -JAMB_Y[0], CEIL)),
    "glass": ((GLASS_X[0], -JAMB_Y[0], 0.0), (GLASS_X[1], JAMB_Y[0], CEIL)),
    # end stops: each channel closed at both ends, across its full width (lip to lip)
    "stop_bottom_left": ((-BOT_HALF - LIP_T, STOP_Y[0], 0.0), (BOT_HALF + LIP_T, STOP_Y[1], STOP_H)),
    "stop_bottom_right": ((-BOT_HALF - LIP_T, -STOP_Y[1], 0.0), (BOT_HALF + LIP_T, -STOP_Y[0], STOP_H)),
    "stop_top_left": ((-TOP_HALF - LIP_T, STOP_Y[0], MOUTH), (TOP_HALF + LIP_T, STOP_Y[1], CEIL)),
    "stop_top_right": ((-TOP_HALF - LIP_T, -STOP_Y[1], MOUTH), (TOP_HALF + LIP_T, -STOP_Y[0], CEIL)),
}


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def to_station(p) -> np.ndarray:
    return np.asarray(p, float) + WINDOW


def quat_xyzw(r: np.ndarray) -> tuple:
    from yam_arena.ik import matrix_to_quat_xyzw

    return tuple(float(v) for v in matrix_to_quat_xyzw(r))


TOP = np.array([0.0, 0.0, SCREEN_H])     # the pivot: centre of the screen's top edge (S)


def pose_from_pivot(pivot, lean: float):
    """Screen pose (o, R) in W with its top edge's centre at ``pivot`` (W) and leaning ``lean`` rad (top toward the
    glass, bottom toward the room)."""
    r = rot_y(lean)
    return np.asarray(pivot, float) - r @ TOP, r


# -- the screen's start: standing on the table in front of the sill, leaning back on the jambs --------------------------
START_BOTTOM_X = -0.070       # W x of the bottom edge's back (lowest) corner, 45 mm in front of the sill
REST_GAP = 0.0003


def start_pose():
    """(o, R) in W: the screen's lowest edge (bottom back) REST_GAP above the table at START_BOTTOM_X, its top back edge
    REST_GAP off the jambs' faces."""
    reach = (JAMB_X - REST_GAP) - START_BOTTOM_X          # x from the bottom back corner to the top back corner
    lean = math.asin(reach / SCREEN_H)
    r = rot_y(lean)
    corner = np.array([FRAME_T / 2, 0.0, 0.0])            # bottom back edge (S)
    target = np.array([START_BOTTOM_X, 0.0, -SILL_H + REST_GAP])
    return target - r @ corner, r


START_LEAN = math.asin(((JAMB_X - REST_GAP) - START_BOTTOM_X) / SCREEN_H)


# -- box maths ------------------------------------------------------------------------------------------------------
def lohi(lo, hi, offset=(0.0, 0.0, 0.0)):
    """A (centre, rotation, half sizes) box from corners, shifted by ``offset``."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return ((lo + hi) / 2 + np.asarray(offset, float), np.eye(3), (hi - lo) / 2)


def posed(box, o, r):
    """A box given in S, placed by the screen pose (o, R)."""
    c, rr, h = box
    return (np.asarray(o, float) + np.asarray(r) @ c, np.asarray(r) @ rr, h)


def box_points(c, r, h, n: int = 5) -> np.ndarray:
    """Points on a box's surface grid (centre, rotation, half sizes)."""
    u = np.linspace(-1, 1, n)
    g = np.array([[a, b, d] for a in u for b in u for d in u])
    g = g[np.abs(g).max(axis=1) == 1.0]
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def box_corners(c, r, h) -> np.ndarray:
    g = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    return np.asarray(c) + (g * np.asarray(h)) @ np.asarray(r).T


def point_box_dist(points, c, r, h) -> np.ndarray:
    """Distance from points to a solid box (0 inside)."""
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    d = np.maximum(np.abs(local) - np.asarray(h), 0.0)
    return np.linalg.norm(d, axis=-1)


def point_box_depth(points, c, r, h) -> np.ndarray:
    """How far points lie inside a solid box (0 outside)."""
    local = (np.asarray(points) - np.asarray(c)) @ np.asarray(r)
    return np.maximum((np.asarray(h) - np.abs(local)).min(axis=-1), 0.0)


def boxes_gap(a, b, n: int = 7) -> float:
    """Approximate gap between two boxes (sampled; 0 if they touch or overlap)."""
    return float(min(point_box_dist(box_points(*a, n), *b).min(), point_box_dist(box_points(*b, n), *a).min()))


def frame_boxes_w():
    """The window frame's boxes in W."""
    return {k: lohi(lo, hi) for k, (lo, hi) in FRAME_BOXES.items()}


def screen_boxes(o, r, q=(0.0, 0.0), pins: bool = True, mesh: bool = True):
    """The screen's boxes (rails, mesh, pins) placed at pose (o, R), pins at joint positions q (left, right)."""
    out = {k: posed(lohi(lo, hi), o, r) for k, (lo, hi) in RAILS.items()}
    if mesh:
        out["mesh"] = posed(lohi(*MESH), o, r)
    if pins:
        for name, sign, qq in (("pin_left", 1.0, q[0]), ("pin_right", -1.0, q[1])):
            out[name] = posed(lohi(*pin_box(sign, qq)), o, r)
    return out


def edge_corners(o, r, z: float) -> np.ndarray:
    """The four corners of the screen's top (z = SCREEN_H) or bottom (z = 0) face, in the pose's frame."""
    pts = np.array([[x, y, z] for x in (-FRAME_T / 2, FRAME_T / 2) for y in (-SCREEN_W / 2, SCREEN_W / 2)])
    return np.asarray(o) + pts @ np.asarray(r).T


def pin_press(o, r) -> float:
    """How far the pins must be pushed in for the screen at (o, R) to fit under the ceiling (0 if they are free):
    the highest pin tip at rest above the ceiling."""
    tips = np.array([[x, s * PIN_Y + d, SCREEN_H + PIN_PROUD] for s in (1, -1) for x in (-PIN_X / 2, PIN_X / 2)
                     for d in (-PIN_YW / 2, PIN_YW / 2)])
    z = (np.asarray(o) + tips @ np.asarray(r).T)[:, 2]
    return float(max(0.0, z.max() - CEIL))
