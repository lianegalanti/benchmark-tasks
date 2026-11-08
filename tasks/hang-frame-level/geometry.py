"""Geometry of the hang-frame-level scene: the frame and its sawtooth hanger, the board and its pin, the stand, and
the tests the scorer applies to the frame's pose. Pure NumPy, so the task file can import it before Isaac Sim starts.

Frame frame ("local"): origin at the centre of the frame's outer box; x points out of its back (toward the board when
it hangs), y along its width to the robot's left, z up. Hanging level and square to the board, local axes are the
station's axes.

The frame is a 30 x 15 cm moulding, 3.5 cm deep. Its glass, picture and backing fill the front 2 cm; behind them the
moulding's four rails leave a 1.5 cm deep recess. The hanger, a 1 cm thick bar with a row of 90-degree teeth along
its lower edge, sits in the recess under the top rail. The pin sticks 13 mm out of the board, so with the frame's
rails against the board it reaches 8 mm under the hanger and stops 2 mm short of the backing.
"""

from __future__ import annotations

import math

import numpy as np

TABLE = 0.7475                          # yam_arena.task_file.TABLE_TOP["station"]

# -- the frame (local, metres) --------------------------------------------------------------------------------------
W, H, D = 0.30, 0.15, 0.035             # outer width (y), height (z), depth (x)
HALF = np.array([D / 2, W / 2, H / 2])
RAIL = 0.02                             # moulding rail width, seen from the back
FRONT_D = 0.020                         # glass, picture and backing: the front 2 cm
BACKING_X = -D / 2 + FRONT_D            # the backing's rear face (the recess floor)
FRAME_MASS = 0.30                       # moulding, glass and backing
MEDAL_MASS = 0.20                       # the bronze medal in the shadow box
MEDAL_AT = np.array([-0.008, 0.040, -0.045])   # its centre: lower left as seen from the front
MEDAL_R = 0.020
MASS = FRAME_MASS + MEDAL_MASS
COM = MEDAL_MASS * MEDAL_AT / MASS      # the frame's own centre of mass is its box centre

# the hanger: a bar on the backing under the top rail, with teeth hanging from its lower edge
PITCH = 0.008                           # tooth pitch (m); notches 90 degrees, PITCH / 2 deep
HANGER_T = 0.010                        # thickness (x), from the backing toward the board
HANGER_X = (BACKING_X, BACKING_X + HANGER_T)
APEX_Z = H / 2 - RAIL - 0.008           # the notches' tops: the bar is 8 mm tall above them
TIP_Z = APEX_Z - PITCH / 2              # the teeth's tips
NOTCHES = np.arange(-5, 6)              # notch k is at local y = k * PITCH
HANGER_HALF_W = 6 * PITCH               # the bar runs from -48 to +48 mm (12 teeth)

# -- the board and its pin (station frame) --------------------------------------------------------------------------
BOARD_X = 0.335                         # the board's front face (toward the robot)
BOARD_T, BOARD_W, BOARD_H = 0.020, 0.10, 0.22
PIN_Y = 0.01                            # the pin and the board's centre line
PIN_Z = TABLE + 0.150                   # the pin's axis
PIN_R = 0.002                           # 4 mm steel pin
PIN_OUT = 0.013                         # how far it sticks out of the board, horizontally
PIN_REST = PIN_R / math.sin(math.radians(45.0))   # the pin's axis sits this far below a notch's top

# -- the stand the frame starts in (station frame) ------------------------------------------------------------------
STAND_X = 0.22                          # the frame's centre, standing in the slot
STAND_BASE_T = 0.012                    # base plate thickness
STAND_RAIL_H, STAND_RAIL_T = 0.015, 0.015
STAND_SLOT = D + 0.006                  # 3 mm play either side of the frame
STAND_LEN = 0.34
START = np.array([STAND_X, 0.0, TABLE + STAND_BASE_T + H / 2])   # the frame's centre at the start

# -- how the demo holds it --------------------------------------------------------------------------------------------
GRIP_Y = 0.12                           # each gripper takes the top rail this far either side of the centre
GRIP_BELOW_TOP = 0.010                  # tool point below the frame's top: the pads cover its top 2.3 cm
APPROACH_GAP = 0.003                    # the rails' backs 3 mm off the board while the frame is lowered on
FINGER_HALF_W = 0.034                   # a YAM finger is 6.8 cm wide across its closing direction


def rest_z_local() -> float:
    """Local z of the pin's axis when it sits in a notch."""
    return APEX_Z - PIN_REST


LEVEL_NOTCH = int(round(COM[1] / PITCH))   # the notch straight above the centre of mass


def hang_centre(notch: int, gap: float = 0.0) -> np.ndarray:
    """Station position of the frame's centre, hanging level and square to the board with ``notch`` on the pin and its
    rails ``gap`` off the board."""
    return np.array([BOARD_X - gap - D / 2, PIN_Y - notch * PITCH, PIN_Z - rest_z_local()])


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def tilt_deg(rot) -> float:
    """How far the frame's top edge (its local y axis) is from horizontal, in degrees."""
    return math.degrees(math.asin(min(1.0, abs(float(np.asarray(rot)[2, 1])))))


def roll(rot) -> float:
    """Signed rotation about the station's x axis (rad): positive when the frame's left end (+y) is up."""
    rot = np.asarray(rot)
    return math.atan2(rot[2, 1], rot[1, 1])


def equilibrium_tilt(notch: int) -> float:
    """The roll (rad) at which the frame hangs still from ``notch``, frictionless: its centre of mass straight under
    the pin. Positive (left end up) when the notch is left of the centre of mass."""
    dy, dz = notch * PITCH - COM[1], rest_z_local() - COM[2]
    return math.atan2(dy, dz)


def corners(pos, rot) -> np.ndarray:
    """The frame box's eight corners, station frame."""
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return np.asarray(pos, float) + (signs * HALF) @ np.asarray(rot).T


def lowest_z_hanging(notch: int, tilt: float) -> float:
    """Height of the frame's lowest corner when it hangs from ``notch`` at roll ``tilt``."""
    centre_local = np.array([0.0, -notch * PITCH, -rest_z_local()])       # from the pin, level
    pos = np.array([0.0, PIN_Y, PIN_Z]) + rot_x(tilt) @ centre_local
    return float(corners(pos, rot_x(tilt))[:, 2].min())


def pin_points(n: int = 14) -> np.ndarray:
    """Points along the pin's axis, from the board face to its tip (station frame)."""
    return np.array([[BOARD_X - s, PIN_Y, PIN_Z] for s in np.linspace(0.0, PIN_OUT - PIN_R, n)])


def on_pin(pos, rot, tol: float = 0.0025) -> bool:
    """Whether the frame hangs on the pin by its hanger: some point of the pin's axis that lies within the hanger's
    thickness is within ``tol`` of the notches' rest line (``PIN_REST`` under the notch tops) and inside the row of
    notches. Resting on the pin anywhere else (the top rail, the ends of the bar) does not count."""
    local = (pin_points() - np.asarray(pos, float)) @ np.asarray(rot)
    inside = (local[:, 0] >= HANGER_X[0] - 0.001) & (local[:, 0] <= HANGER_X[1] + 0.001)
    near = (np.abs(local[:, 2] - rest_z_local()) <= tol) & (np.abs(local[:, 1]) <= NOTCHES[-1] * PITCH + 0.004)
    return bool(np.any(inside & near))


def back_gap(pos, rot) -> float:
    """How far the frame's rearmost corner is from the board's face (m); negative would be inside the board."""
    return float(BOARD_X - corners(pos, rot)[:, 0].max())


# -- the frame's parts (local): (name, centre, size); the collision shapes of assets/frame.usda ----------------------
def collision_boxes() -> list[tuple[str, np.ndarray, np.ndarray]]:
    rear = D - FRONT_D
    rx = BACKING_X + rear / 2
    inner_h = H - 2 * RAIL
    return [
        ("front", np.array([-D / 2 + FRONT_D / 2, 0.0, 0.0]), np.array([FRONT_D, W, H])),
        ("rail_top", np.array([rx, 0.0, H / 2 - RAIL / 2]), np.array([rear, W, RAIL])),
        ("rail_bottom", np.array([rx, 0.0, -H / 2 + RAIL / 2]), np.array([rear, W, RAIL])),
        ("rail_left", np.array([rx, W / 2 - RAIL / 2, 0.0]), np.array([rear, RAIL, inner_h])),
        ("rail_right", np.array([rx, -W / 2 + RAIL / 2, 0.0]), np.array([rear, RAIL, inner_h])),
        ("hanger_bar", np.array([BACKING_X + HANGER_T / 2, 0.0, (APEX_Z + H / 2 - RAIL) / 2]),
         np.array([HANGER_T, 2 * HANGER_HALF_W, H / 2 - RAIL - APEX_Z])),
    ]


def teeth() -> list[np.ndarray]:
    """Centres of the hanger's teeth: squares turned 45 degrees about x, diagonal PITCH, centred on the notch tops'
    line, so their lower halves are the teeth and two neighbours meet at each notch's top."""
    return [np.array([BACKING_X + HANGER_T / 2, (k + 0.5) * PITCH, APEX_Z]) for k in range(-6, 6)]


def inertia() -> tuple[np.ndarray, np.ndarray]:
    """Inertia about the centre of mass (local axes): the frame as a uniform box plus the medal as a point."""
    i_box = FRAME_MASS / 12 * np.diag([W * W + H * H, D * D + H * H, D * D + W * W])
    out = np.zeros((3, 3))
    for m, c, i0 in ((FRAME_MASS, np.zeros(3), i_box), (MEDAL_MASS, MEDAL_AT, np.zeros((3, 3)))):
        r = c - COM
        out += i0 + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return out, COM
