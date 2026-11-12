"""Geometry of the close-door-past-hotplate scene, in the station frame. Pure NumPy, so the task file, the asset
generator and the tests can import it before Isaac Sim starts.

Station frame: x forward from the arms, y to the robot's left, z up (m); the arm bases stand at (0, +-0.305, 0.75).

The cabinet faces the robot. Its door hangs on a vertical hinge at the cabinet's right front corner and is open by
60 degrees: it runs from the hinge toward the robot and to the left. A hot plate stands in front of the door's hinge
half, between the door and the right arm. The last few centimetres of the door, at its free edge, stick out to the
left of the hot plate: that is the only part of the door's outer face a gripper can push without being over the
plate.

Door angle: 0 is closed, positive is open (the free edge toward the robot). At angle ``a`` the door runs from the
hinge axis along ``u(a) = (-sin a, cos a)``; its outer face (the cabinet's front when closed) faces
``n(a) = (-cos a, -sin a)``, and pushing on it along ``-n`` closes the door.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                       # yam_arena.task_file.TABLE_TOP["station"]

# -- cabinet and door ------------------------------------------------------------------------------------------------
DOOR_W, DOOR_T = 0.22, 0.015       # door width (hinge axis to free edge) and thickness
DOOR_Z = (0.005, 0.175)            # door bottom and top above the table
DOOR_MASS = 0.30                   # kg: 15 mm plywood
CAB_D, CAB_H, PANEL = 0.20, 0.18, 0.012   # carcass depth (x), height, panel thickness; its width is DOOR_W
FRONT_X = 0.40                     # the carcass's front face
HINGE = np.array([FRONT_X - DOOR_T / 2, -0.025])   # hinge axis (x, y): the door's right edge, mid-thickness
OPEN = math.radians(60.0)          # the door's angle at the start
MAX_OPEN = math.radians(90.0)      # the hinge's stop
HINGE_DAMPING = 0.05               # N m s/rad

# -- hot plate: a compact lab hot plate, a fixed box with a red disc marking the heated area on its 15 cm square top --
# Its right edge stands 3.5 cm (capsule to column) left of the right arm's ready pose, so a centimetre or two of
# placement error on a real table can't put the resting arm in the hot zone.
PLATE_LO = np.array([0.21, -0.21])    # footprint corners (x, y): 15 x 15 cm
PLATE_HI = np.array([0.36, -0.06])
PLATE_H = 0.08
BURNER_R = 0.06
PLATE_TOP = TOP + PLATE_H
# The hot zone is the whole column over the plate's footprint, from the table up, with no height limit: any part of
# either arm above the plate, however high, is in it.
ZONE_LO = np.array([*PLATE_LO, TOP])
ZONE_HI = np.array([*PLATE_HI, math.inf])

# -- scoring thresholds (door angle) -------------------------------------------------------------------------------
CLOSED = math.radians(4.0)         # closed: the free edge within 1.5 cm of the cabinet's front
HALF = OPEN / 2                    # pushed at least halfway closed: 30 degrees or less
MOVED = OPEN - math.radians(10.0)  # pushed at least 10 degrees closed: 50 degrees or less


def u(angle: float) -> np.ndarray:
    """Unit vector (x, y) along the door from the hinge axis toward the free edge."""
    return np.array([-math.sin(angle), math.cos(angle)])


def n(angle: float) -> np.ndarray:
    """Unit vector (x, y) out of the door's outer face (the side that is pushed to close it)."""
    return np.array([-math.cos(angle), -math.sin(angle)])


def door_point(angle: float, r: float, off: float = 0.0) -> np.ndarray:
    """(x, y) of the point ``r`` along the door from the hinge axis and ``off`` out from its outer face."""
    return HINGE + r * u(angle) + (DOOR_T / 2 + off) * n(angle)


def door_corners(angle: float) -> np.ndarray:
    """The door's footprint (4 x 2) at ``angle``."""
    a, b = u(angle), n(angle)
    return np.array([HINGE + s * DOOR_W * a + t * DOOR_T / 2 * b for s, t in ((0, -1), (1, -1), (1, 1), (0, 1))])


def free_edge_gap(angle: float) -> float:
    """How far the door's free edge (its outer corner) stands out in front of the closed door's face, as a tape
    measure along the cabinet's side reads it."""
    return float(door_point(0.0, DOOR_W)[0] - door_point(angle, DOOR_W)[0])


def _seg_dist(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def polygon_gap(p, q) -> float:
    """Distance between two convex polygons (rows of (x, y) corners); 0 if they overlap."""

    def inside(pt, poly):
        def cross(a, b):
            return a[0] * b[1] - a[1] * b[0]

        signs = [cross(poly[(i + 1) % len(poly)] - poly[i], pt - poly[i]) for i in range(len(poly))]
        return all(s >= 0 for s in signs) or all(s <= 0 for s in signs)

    if any(inside(x, q) for x in p) or any(inside(x, p) for x in q):
        return 0.0
    return min(_seg_dist(x, poly[i], poly[(i + 1) % len(poly)])
               for x, poly in [(x, q) for x in p] + [(x, p) for x in q] for i in range(len(poly)))


def plate_corners() -> np.ndarray:
    (x0, y0), (x1, y1) = PLATE_LO, PLATE_HI
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
