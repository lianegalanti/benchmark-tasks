"""The tub and lid shapes (metres), shared by task.py and make_lid.py. Pure Python.

A tub is an open square box: four walls standing on its carrier plate, which is its floor. A lid is one rigid body:
a square top plate, a skirt (four strips) hanging down round its edge, and a grip fin standing along its middle.
A seated lid rests with its plate's underside on the tub's rim, its skirt round the outside of the tub's walls, 1 cm
clear of them on every side, so the skirt keeps a seated lid within 1 cm of the tub's axis.
"""

from __future__ import annotations

# -- the tub ---------------------------------------------------------------------------------------------------------
TUB = 0.08  # outside, square
TUB_WALL = 0.006
TUB_H = 0.05  # wall height, from the carrier's top (+ the 1 mm gap below) to the rim
WALL_GAP = 0.001  # driven parts never touch: walls stand 1 mm above the carrier and 1 mm apart at the corners

# -- the lid (origin: the centre of the top plate) -------------------------------------------------------------------
CLEAR = 0.010  # skirt to tub wall, each side, when centred
SKIRT_T = 0.004
SKIRT_IN = TUB / 2 + CLEAR  # 0.05: half the skirt's inside width
LID = 2 * (SKIRT_IN + SKIRT_T)  # 0.108: the plate's width (the skirt's outside)
PLATE_T = 0.005
SKIRT_D = 0.015  # how far the skirt hangs below the plate's underside
TAB = (0.08, 0.024, 0.055)  # the grip fin on top: 8 cm long front to back (x), 2.4 cm thick (y), 5.5 cm tall;
# long and tall, so a pinch across its thickness takes hold with the fingers well off its centre along it, and the
# pads (whose faces reach 5 cm up from their tips) hold 4 cm of it, which keeps the lid from pivoting in the grip
LID_MASS = 0.04  # kg, polypropylene

PLATE_UNDER = -PLATE_T / 2  # the plate's underside, from the origin
SKIRT_BOTTOM = PLATE_UNDER - SKIRT_D  # -0.0175: a lid lying on the table has its origin this far above it
TAB_TOP = PLATE_T / 2 + TAB[2]
TAB_CENTRE_Z = PLATE_T / 2 + TAB[2] / 2


def lid_parts() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, size) of each box of the lid, in the lid's frame."""
    sz = PLATE_UNDER - SKIRT_D / 2
    sx = SKIRT_IN + SKIRT_T / 2
    return [
        ("plate", (0.0, 0.0, 0.0), (LID, LID, PLATE_T)),
        ("skirt_front", (sx, 0.0, sz), (SKIRT_T, LID, SKIRT_D)),
        ("skirt_back", (-sx, 0.0, sz), (SKIRT_T, LID, SKIRT_D)),
        ("skirt_left", (0.0, sx, sz), (2 * SKIRT_IN, SKIRT_T, SKIRT_D)),
        ("skirt_right", (0.0, -sx, sz), (2 * SKIRT_IN, SKIRT_T, SKIRT_D)),
        ("tab", (0.0, 0.0, TAB_CENTRE_Z), TAB),
    ]


def tub_walls() -> list[tuple[str, tuple[float, float], tuple[float, float, float]]]:
    """(name, horizontal offset from the carrier's centre, size) of each tub wall."""
    off = TUB / 2 - TUB_WALL / 2
    side = TUB - 2 * TUB_WALL - 2 * WALL_GAP  # the side walls fit between the front and back walls
    return [
        ("front", (off, 0.0), (TUB_WALL, TUB, TUB_H)),
        ("back", (-off, 0.0), (TUB_WALL, TUB, TUB_H)),
        ("left", (0.0, off), (side, TUB_WALL, TUB_H)),
        ("right", (0.0, -off), (side, TUB_WALL, TUB_H)),
    ]
