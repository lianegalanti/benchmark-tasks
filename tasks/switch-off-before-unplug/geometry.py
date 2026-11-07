"""Geometry of the switch-off-before-unplug scene, in the station frame. Pure Python and NumPy, so the task file,
the asset script and the tests can import it before Isaac Sim starts.

Station frame: x forward from the arms, y to the robot's left, z up (m); the arm bases stand at x 0, y +-0.305.
The power strip lies along y in front of the arms, its three sockets toward the right arm and its rocker switch toward
the left arm. The heater's plug stands in the rightmost socket. The heater stands behind the strip, the tray in front
of the right arm.

The strip asset (assets/strip.usda) is built in the strip's own frame: origin at the centre of its footprint on the
table, axes as the station's. The plug asset (assets/plug.usda) has its origin at the centre of the plug body's bottom
face, z up, pins pointing down.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475                            # yam_arena.task_file.TABLE_TOP["station"]

# -- power strip (strip frame) --------------------------------------------------------------------------------------
STRIP_XY = (0.30, 0.0)                  # where the strip's footprint centre stands on the station
STRIP_HX, STRIP_HY = 0.03, 0.145        # 6 x 29 cm footprint
STRIP_H = 0.042                         # housing top above the table
FLOOR = 0.008                           # the sockets' floor above the table (the housing's bottom slab)
RAIL = 0.010                            # side walls along the long edges
WELL = 0.040                            # square socket wells, 4 x 4 cm, STRIP_H - FLOOR = 3.4 cm deep
WELLS_Y = (-0.105, -0.050, 0.005)       # well centres (strip frame y); the plug is in the first one
PLUG_WELL = 0
BEZEL = ((-0.016, 0.055, STRIP_H), (0.016, 0.115, STRIP_H + 0.008))   # raised switch surround, corners
PIVOT = np.array([0.0, 0.085, STRIP_H + 0.014])                        # rocker hinge axis (along x) point
PADDLE_HALF = (0.012, 0.025, 0.004)     # rocker paddle: 2.4 (x) by 5.0 (y) by 0.8 cm, centred on the hinge
ROCK = math.radians(12.0)               # the paddle tilts 12 degrees either side of centre
LAMP_Y = 0.130                          # the red pilot lamp's centre, beside the switch

# The rocker's hinge angle q: 0 is ON (the paddle's "I" end, toward the lamp, pressed down), 24 degrees is OFF (its
# "O" end, toward the sockets, pressed down). The hinge stops at both; the paddle is level at 12 degrees.
Q_ON, Q_OFF = 0.0, 2 * ROCK
Q_CENTRE = ROCK
OFF_AT = math.radians(18.0)             # switched off: the O end down, at least 6 degrees past centre
DETENT = 0.01                           # N m: the over-centre spring, toward the nearer end
PADDLE_MASS = 0.01
HINGE_DAMPING = 0.002                   # N m s/rad

# -- plug (plug frame) ----------------------------------------------------------------------------------------------
BODY_HALF = (0.018, 0.018)              # 3.6 x 3.6 cm body
BODY_H = 0.055
PIN_R, PIN_LEN, PIN_DY = 0.0024, 0.019, 0.0095   # two 4.8 mm pins, 19 mm long, 19 mm apart (Type F)
CORD_R, CORD_LEN, CORD_Z = 0.004, 0.030, 0.045   # strain relief and cord stub out of the body's far (+x) face
PLUG_MASS = 0.08
PLUG_FRICTION = (1.0, 0.9)
SEATED_Z = TOP + FLOOR + PIN_LEN        # plug origin height fully pushed in (pins on the socket floor)
PLUG_START = (STRIP_XY[0], STRIP_XY[1] + WELLS_Y[PLUG_WELL], SEATED_Z + 0.001)
PULLED = 0.010                          # m raised (or off the well's centre) at which the plug has left the socket

# -- tray, heater ---------------------------------------------------------------------------------------------------
TRAY_XY = (0.20, -0.23)
TRAY_IN = 0.12                          # inside, square
TRAY_WALL, TRAY_FLOOR, TRAY_H = 0.004, 0.004, 0.020
HEATER_LO = (0.47, -0.14, TOP)          # white ceramic fan heater, 12 x 24 x 22 cm
HEATER_HI = (0.59, 0.10, TOP + 0.22)
GRILLE_Y = (-0.11, 0.07)                # the front heating grille, which glows orange while the heater runs
GRILLE_Z = (TOP + 0.03, TOP + 0.18)


def strip(p) -> np.ndarray:
    """A strip-frame point on the station."""
    return np.asarray(p, float) + np.array([STRIP_XY[0], STRIP_XY[1], TOP])


def well_centre(i: int = PLUG_WELL) -> np.ndarray:
    """A well's centre on the station, at its floor."""
    return strip((0.0, WELLS_Y[i], FLOOR))


def housing_boxes():
    """The housing as boxes (name, low corner, high corner), strip frame: a bottom slab, two side walls, infill
    between the wells, and the switch's raised bezel."""
    boxes = [("slab", (-STRIP_HX, -STRIP_HY, 0.0), (STRIP_HX, STRIP_HY, FLOOR)),
             ("wall_near", (-STRIP_HX, -STRIP_HY, FLOOR), (-STRIP_HX + RAIL, STRIP_HY, STRIP_H)),
             ("wall_far", (STRIP_HX - RAIL, -STRIP_HY, FLOOR), (STRIP_HX, STRIP_HY, STRIP_H))]
    edges = [-STRIP_HY]
    for c in WELLS_Y:
        edges += [c - WELL / 2, c + WELL / 2]
    edges.append(STRIP_HY)
    xi = STRIP_HX - RAIL
    for k in range(0, len(edges), 2):
        boxes.append((f"infill{k // 2}", (-xi, edges[k], FLOOR), (xi, edges[k + 1], STRIP_H)))
    boxes.append(("bezel", BEZEL[0], BEZEL[1]))
    return boxes


def paddle_point(q: float, y: float, z: float) -> np.ndarray:
    """A point (paddle frame y, z about the hinge) of the paddle at hinge angle ``q``, strip frame. The paddle's
    tilt is q - 12 degrees about +x: at q = 0 its +y ("I") end is down."""
    a = q - ROCK
    return PIVOT + np.array([0.0, y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)])


def switched_off(q: float) -> bool:
    return q >= OFF_AT
