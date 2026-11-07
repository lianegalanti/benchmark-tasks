"""Geometry of the stop-and-brake-rolling-cart scene: the tilted track, the cart and its brake lever, and where they
stand on the station table. Pure NumPy, so the task file and the tests can import it before Isaac Sim starts.

Station frame: x forward (away from the arms), y to the robot's left, z up; metres.

The track runs along y in front of the arms and slopes down toward the left (+y), where it ends at the table's left
edge. "Track frame" (also the cart's own frame, since the cart sits square on it): x across the track (+x away
from the arms), y down the slope, z normal to it. Its origin, for a cart at ``q`` metres up the slope from the
track's end, is on the plane of the rail tops, midway between the rails, under the cart's centre.

The brake lever turns on a hinge whose axis runs along the track (the cart's y), on the cart's near side (the side
facing the arms). Its angle ``phi`` is the hinge's joint position: positive ``phi`` tips the handle toward the cart
(+x) and swings the foot out (-x) and down. Stowed, the handle leans out toward the arms at ``STOWED``; engaged, it
leans toward the cart at ``ENGAGED``, past straight up, and the rubber pad at its foot presses on the deck.
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81
TOP = 0.7475                     # yam_arena.task_file.TABLE_TOP["station"]
TABLE_LEFT_Y = 0.65              # the station table's left edge (yam_arena.station: centre y 0, 1.30 m wide)

# -- the track -----------------------------------------------------------------------------------------------------
THETA = math.radians(1.2)        # slope, down toward +y
LANE_X = 0.30                    # the track's centre line, in front of the arm bases
END_Y = 0.64                     # the track's low end: 1 cm short of the table's left edge
START_END_Y = -0.44              # the track's high end
END_H = 0.012                    # deck top above the table at the low end
DECK_T = 0.05                    # deck thickness (sunk into the table at its low end, like a wedge shimmed level)
DECK_X = (-0.13, 0.09)           # deck across the track, track frame (wider on the near side, for the brake pad)
RAIL_H = 0.010                   # rails stand this high on the deck
RAIL_X, RAIL_W = 0.04, 0.015     # rail centres at +-RAIL_X
GUIDE_W, GUIDE_H = 0.020, 0.016  # central guide bar: width, height above the deck (6 mm above the rail tops)
RAIL_MU = 0.006                  # rails and guide bar: friction, "min" combined (the cart's ball-bearing wheels)
DECK_MU = (0.6, 0.5)             # the deck's painted plywood top, averaged with the cart's (0.8, 0.7)
CART_MU = (0.8, 0.7)             # the cart's plastic and its brake pad's rubber

DOWN = np.array([0.0, math.cos(THETA), -math.sin(THETA)])     # down the slope
NORMAL = np.array([0.0, math.sin(THETA), math.cos(THETA)])    # the deck's upward normal
ACROSS = np.array([1.0, 0.0, 0.0])
R_TRACK = np.column_stack([ACROSS, DOWN, NORMAL])             # track frame -> station frame
ROT_XYZW = (math.sin(-THETA / 2), 0.0, 0.0, math.cos(-THETA / 2))  # the same rotation, about +x
DECK_LEN = (END_Y - START_END_Y) / math.cos(THETA)            # along the slope

END_POINT = np.array([LANE_X, END_Y, TOP + END_H])            # deck top, on the centre line, at the low end


def deck_point(q: float, x: float = 0.0) -> np.ndarray:
    """The point on the deck top ``q`` metres up the slope from the low end, ``x`` across the track."""
    return END_POINT - q * DOWN + x * ACROSS


def origin_at(q: float) -> np.ndarray:
    """The track-frame origin (rail-top plane, centre line) under a cart centred ``q`` up the slope."""
    return deck_point(q) + RAIL_H * NORMAL


def q_of_y(y: float) -> float:
    """``q`` of the track-frame origin whose station y is ``y``."""
    return (END_Y - y) / math.cos(THETA)


def to_station(q: float, local) -> np.ndarray:
    """A track-frame point (a cart centred at ``q``) in the station frame."""
    return origin_at(q) + R_TRACK @ np.asarray(local, float)


def to_track(p) -> tuple[float, np.ndarray]:
    """A station point as (q of the track frame it is the origin of, offset (x, 0, z) from that origin)."""
    rel = R_TRACK.T @ (np.asarray(p, float) - origin_at(0.0))
    return -rel[1], np.array([rel[0], 0.0, rel[2]])


# -- the cart (track frame, origin under its centre on the rail-top plane) -------------------------------------------
BODY_X, BODY_Y, BODY_Z = 0.06, 0.08, (0.010, 0.060)   # half width, half length, bottom/top
RUNNER_Y, RUNNER_H = 0.07, 0.010                      # runners: half length, height (under the body, on the rails)
SHOE_X, SHOE_W, SHOE_Y, SHOE_Z0 = 0.018, 0.008, 0.05, -0.006  # guide shoes either side of the guide bar
BODY_MASS = 0.30
CART_HALF_LEN = BODY_Y

# -- the brake lever (its own frame: origin on the hinge axis; at phi = 0 the lever points straight up) --------------
PIVOT_X = -0.095                 # 3.5 cm out from the body's near side
LEVER_Y = -0.03                  # along the cart: on its uphill half
LEVER_HX = 0.004                 # half thickness (across the track)
PAD_A = 0.050                    # hinge axis to the pad's underside
PAD_H, PAD_HY = 0.008, 0.012     # pad height, half length
BAR_HY = 0.005
PADDLE_Z, PADDLE_HY = (0.035, 0.075), 0.015   # the handle's paddle, above the axis
LEVER_MASS = 0.03
STOWED = math.radians(-40.0)     # the hinge's lower stop: handle leaning out toward the arms
ENGAGED = math.radians(12.0)     # its upper stop: handle leaning toward the cart, the pad on the deck
INTERFERENCE = 0.0025            # engaged on an unlifted cart, the pad would reach this far below the deck top


def pad_depth(phi: float) -> float:
    """How far the pad's lowest corner is below the hinge axis (normal to the deck) at lever angle ``phi``."""
    return PAD_A * math.cos(phi) + LEVER_HX * abs(math.sin(phi))


PIVOT_H = pad_depth(ENGAGED) - INTERFERENCE         # hinge axis above the deck top
PIVOT_Z = PIVOT_H - RAIL_H                          # ... in the track frame


def lever_to_track(phi: float, p) -> np.ndarray:
    """A point given in the lever's frame, in the track frame, at lever angle ``phi`` (rotation about +y)."""
    x, y, z = p
    c, s = math.cos(phi), math.sin(phi)
    return np.array([PIVOT_X + x * c + z * s, LEVER_Y + y, PIVOT_Z - x * s + z * c])


def pad_contact(phi: float) -> np.ndarray:
    """The pad's lowest corner at ``phi`` (track frame): tipped outward (phi > 0), its inner corner."""
    corner_x = LEVER_HX if phi >= 0 else -LEVER_HX
    return lever_to_track(phi, (corner_x, 0.0, -PAD_A))


def lever_parts() -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """The lever's boxes (low corner, high corner) in its own frame."""
    return {
        "pad": ((-LEVER_HX, -PAD_HY, -PAD_A), (LEVER_HX, PAD_HY, -PAD_A + PAD_H)),
        "bar": ((-LEVER_HX, -BAR_HY, -PAD_A + PAD_H), (LEVER_HX, BAR_HY, PADDLE_Z[0])),
        "paddle": ((-LEVER_HX, -PADDLE_HY, PADDLE_Z[0]), (LEVER_HX, PADDLE_HY, PADDLE_Z[1])),
    }


def lever_com() -> np.ndarray:
    """The lever's centre of mass in its own frame (uniform density)."""
    vols, cents = [], []
    for lo, hi in lever_parts().values():
        lo, hi = np.asarray(lo), np.asarray(hi)
        vols.append(np.prod(hi - lo))
        cents.append((lo + hi) / 2)
    return np.average(np.array(cents), axis=0, weights=vols)


def lever_corners(phi: float) -> np.ndarray:
    """Every corner of the lever's boxes at ``phi`` (track frame), as rows."""
    out = []
    for lo, hi in lever_parts().values():
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    out.append(lever_to_track(phi, (x, y, z)))
    return np.array(out)


def handle_face_x(phi: float, height: float) -> float:
    """Track-frame x of the lever's outer face (the side facing the arms) at ``height`` above the hinge axis."""
    s, c = math.sin(phi), math.cos(phi)
    r = (height - LEVER_HX * s) / c  # along the lever, from the axis, where its outer face is at that height
    return float(PIVOT_X - LEVER_HX * c + r * s)


def body_parts() -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """The cart body's boxes (low corner, high corner), track frame."""
    bx, by, (z0, z1) = BODY_X, BODY_Y, BODY_Z
    sx0, sx1 = SHOE_X - SHOE_W / 2, SHOE_X + SHOE_W / 2
    return {
        "body": ((-bx, -by, z0), (bx, by, z1)),
        "runner_near": ((-RAIL_X - RAIL_W / 2, -RUNNER_Y, 0.0), (-RAIL_X + RAIL_W / 2, RUNNER_Y, RUNNER_H)),
        "runner_far": ((RAIL_X - RAIL_W / 2, -RUNNER_Y, 0.0), (RAIL_X + RAIL_W / 2, RUNNER_Y, RUNNER_H)),
        "shoe_near": ((-sx1, -SHOE_Y, SHOE_Z0), (-sx0, SHOE_Y, z0)),
        "shoe_far": ((sx0, -SHOE_Y, SHOE_Z0), (sx1, SHOE_Y, z0)),
        # the lever's bracket: one cheek beside the lever (downhill of it), from the body's side out to the axis
        "bracket": ((PIVOT_X + LEVER_HX + 0.002, LEVER_Y + PADDLE_HY - 0.008, PIVOT_Z - 0.008),
                    (-bx, LEVER_Y + PADDLE_HY + 0.004, PIVOT_Z + 0.008)),
    }


# the cart without its lever, as one box (track frame), for "arms clear of the cart": body, runners, shoes, bracket
BODY_BOX = (np.array([PIVOT_X, -BODY_Y, SHOE_Z0]), np.array([BODY_X, BODY_Y, BODY_Z[1]]))


def lever_box(phi: float) -> tuple[np.ndarray, np.ndarray]:
    """The box (track frame) around the lever at ``phi``."""
    c = lever_corners(phi)
    return c.min(axis=0), c.max(axis=0)


# -- the roll, untouched (analytic) ----------------------------------------------------------------------------------
START_Y = -0.30                  # the cart's centre at the start, at rest
Q_START = q_of_y(START_Y)


def accel(mu: float = RAIL_MU) -> float:
    """Down-slope acceleration of the cart sliding on the rails (m/s^2)."""
    return G * (math.sin(THETA) - mu * math.cos(THETA))


def roll_time(dq: float, mu: float = RAIL_MU) -> float:
    """Seconds for the cart, released at rest, to roll ``dq`` metres down the slope."""
    return math.sqrt(2 * dq / accel(mu))


