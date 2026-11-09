"""The tray, the cart and its nest (metres), shared by task.py and make_tray.py. Pure Python.

A tray is one rigid body: an open square box (a floor and four walls) with a handle rib standing across its middle,
taller than the walls, that splits it into two compartments. A gripper pinches the rib above the walls. The tray's
origin is the centre of its bounding box, so a level tray's underside is ``UNDER`` below its origin.

The cart is a flat platform; its nest is a lead-in funnel of four guide plates standing on it round a square pocket,
1.25 cm wider than a tray on every side at the bottom. Each plate leans outward at 55 degrees from the horizontal, so
the pocket widens to 2.6 cm of play on every side at the top, and a tray let go off centre slides down the plate into
the pocket, square (a plate is steeper than the tray's friction angle). The plates are 2 mm short of the pocket's
sides, so they stand 1 mm apart at the corners and 1 mm above the cart.
"""

from __future__ import annotations

import math

# -- the tray (origin: the centre of its bounding box) ---------------------------------------------------------------
TRAY = 0.11  # outside, square
TRAY_WALL = 0.005
TRAY_FLOOR = 0.005
TRAY_H = 0.035  # underside to rim
RIB_T = 0.012  # the handle rib's thickness (along y, the cart's travel and the fingers' closing direction)
RIB_TOP = 0.075  # underside to the rib's top: 4 cm above the rim
TRAY_MASS = 0.15  # kg, polypropylene

UNDER = -RIB_TOP / 2  # the underside, from the origin
RIM = UNDER + TRAY_H  # the rim, from the origin
FLOOR_TOP = UNDER + TRAY_FLOOR
INNER = TRAY - 2 * TRAY_WALL  # 0.10: inside, square
POCKET_Y = (INNER - RIB_T) / 2  # 0.044: a compartment's inside width along y
POCKET_C = RIB_T / 2 + POCKET_Y / 2  # 0.028: a compartment's centre from the rib's centre plane
GRIP_Z = RIM + 0.0275  # the rib's grip point (between the pad tips and the rib's top), from the origin

# -- the parts (two aluminium blocks, one in each compartment of the full tray) ---------------------------------------
PART = (0.06, 0.03, 0.02)  # long along x, the rib's length
PART_MASS = 0.10

def tray_parts() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, size) of each box of a tray, in the tray's frame."""
    off = TRAY / 2 - TRAY_WALL / 2
    wz = UNDER + TRAY_H / 2
    side = TRAY - 2 * TRAY_WALL  # the side walls fit between the near and far walls
    rib_h = RIB_TOP - TRAY_FLOOR
    return [
        ("floor", (0.0, 0.0, UNDER + TRAY_FLOOR / 2), (TRAY, TRAY, TRAY_FLOOR)),
        ("wall_near", (-off, 0.0, wz), (TRAY_WALL, TRAY, TRAY_H)),
        ("wall_far", (off, 0.0, wz), (TRAY_WALL, TRAY, TRAY_H)),
        ("wall_right", (0.0, -off, wz), (side, TRAY_WALL, TRAY_H)),
        ("wall_left", (0.0, off, wz), (side, TRAY_WALL, TRAY_H)),
        ("rib", (0.0, 0.0, FLOOR_TOP + rib_h / 2), (INNER, RIB_T, rib_h)),
    ]


# -- the cart and its nest ---------------------------------------------------------------------------------------------
CART = (0.18, 0.18, 0.04)
NEST_IN = TRAY + 2 * 0.0125  # 0.135: the pocket at the bottom, square
GUIDE_L, GUIDE_T = 0.024, 0.004  # each guide plate: its width up the slope, its thickness
GUIDE_DEG = 55.0  # the plates' slope from the horizontal
GAP = 0.001  # driven parts never touch: the plates stand 1 mm above the cart and 1 mm apart at the corners
_A = math.radians(GUIDE_DEG)
GUIDE_LOW = GAP + GUIDE_T * math.cos(_A)  # the pocket's edge (the plate's inner lower edge), above the cart's top
GUIDE_TOP = GUIDE_LOW + GUIDE_L * math.sin(_A)  # 0.023: the funnel's top, above the cart's top
FLARE = GUIDE_L * math.cos(_A)  # 0.0138: how much wider the pocket is at the top, each side
GUIDE_OUT = NEST_IN / 2 + FLARE + GUIDE_T * math.sin(_A)  # the plates' outer reach from the cart's centre


def nest_guides() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float],
                                tuple[float, float, float, float]]]:
    """(name, centre from the centre of the cart's top, size, rotation xyzw) of each guide plate. A plate's local x runs
    up its slope, away from the pocket; its local z is the normal of its face toward the pocket."""
    out = []
    for name, phi in (("far", 0.0), ("left", math.pi / 2), ("near", math.pi), ("right", -math.pi / 2)):
        u = (math.cos(phi), math.sin(phi))
        d = (u[0] * math.cos(_A), u[1] * math.cos(_A), math.sin(_A))  # up the slope
        n = (-u[0] * math.sin(_A), -u[1] * math.sin(_A), math.cos(_A))  # the face toward the pocket
        p = (u[0] * NEST_IN / 2, u[1] * NEST_IN / 2, GUIDE_LOW)  # the face's lower edge: the pocket's edge
        c = tuple(p[i] + d[i] * GUIDE_L / 2 - n[i] * GUIDE_T / 2 for i in range(3))
        s1, c1 = math.sin(phi / 2), math.cos(phi / 2)  # about z by phi, after about y by -GUIDE_DEG
        y2, c2 = -math.sin(_A / 2), math.cos(_A / 2)
        quat = (-s1 * y2, c1 * y2, c2 * s1, c1 * c2)
        out.append((name, c, (GUIDE_L, NEST_IN - 2 * GAP, GUIDE_T), quat))
    return out
