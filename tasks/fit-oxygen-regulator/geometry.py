"""Geometry of the oxygen cylinder, its pin-index post valve, the yoke regulator and the cradle, and the fit tests.

Pure NumPy, so the task file and the tests can import it before Isaac Sim starts. Station frame: x forward from the
arms, y to the robot's left, z up (metres). The valve and pin-index dimensions are the basic dimensions of ISO 407
(the international form of the CGA 870 medical oxygen post-valve connection): pins 4 mm in diameter standing 5.5 mm
proud, holes 4.75 mm across, the oxygen positions 2 and 5 on a 14.3 mm radius round the outlet, a 25 mm thick valve
clamped between the yoke's inlet and its screw, and a 1.6 mm sealing washer.

Frames:
- **C** (cylinder part): origin on the cylinder's axis at the post valve's free end, axes as the station's. The
  cylinder lies along +x in its cradle; the valve post sticks out toward the robot (-x). The outlet port faces the
  robot's right (-y), the conical recess faces its left (+y).
- **R** (regulator part): origin on the inlet's axis, in the plane midway between the port face and the recess face
  when it is seated; axes as the station's when it is mounted. The inlet and its pins face +y; the T-screw comes in
  from +y along -y.
"""

from __future__ import annotations

import math

import numpy as np

# -- station placement ------------------------------------------------------------------------------------------------
TABLE = 0.7475                       # yam_arena.task_file.TABLE_TOP["station"]
AXIS_Z = 0.870                       # height of the cylinder's (and the valve outlet's) axis, 12.25 cm above the table
PORT_X = 0.290                       # outlet centre, along x
POST_TOP_X = PORT_X - 0.016          # the post's free end
C_ORIGIN = (POST_TOP_X, 0.0, AXIS_Z)

# -- the valve post (C frame), ISO 407 ----------------------------------------------------------------------------------
POST_LEN = 0.060                     # x: free end to the neck
POST_HY = 0.0125                     # half of a = 25 mm (port face to recess face)
POST_HZ = 0.014                      # half of its 28 mm width
PORT_U = 0.016                       # outlet centre from the free end (l8 >= 15 mm)
PIN_R_CIRCLE = 0.0143                # R: pin positions on this radius about the outlet
PIN_ANGLE = math.radians(18.0)       # positions 2 and 5: 18 degrees either side of the line toward the cylinder
PIN_D = 0.004                        # d1
PIN_LEN = 0.0055                     # l6: how far the pins stand proud of the yoke's inlet face
HOLE_W = 0.00475                     # D2 (the holes are modelled as 4.75 mm square pockets)
HOLE_DEPTH = 0.006                   # l5 >= 5.5 mm
COUNTERSINK = 0.0006                 # 45-degree lead-in at each hole's mouth, 0.6 mm (the mouth is 5.95 mm across)
END_CHAMFER = 0.003                  # 45-degree chamfer on the post end's top and bottom edges, 3 mm
RECESS_W = 0.007                     # D6: the conical recess, 7 mm across at the top (modelled as a square pocket)
RECESS_DEPTH = 0.002
STEM = 0.006                         # the valve spindle's square end on the free end, 6 mm across, 5 mm long
NECK_R, NECK_LEN = 0.0125, 0.025     # valve stem and cylinder neck ring between the post and the shoulder

PIN_DX = PIN_R_CIRCLE * math.cos(PIN_ANGLE)   # 13.6 mm from the outlet toward the cylinder
PIN_DZ = PIN_R_CIRCLE * math.sin(PIN_ANGLE)   # 4.42 mm either side of the axis
HOLES_C = [np.array([PORT_U + PIN_DX, -POST_HY, s * PIN_DZ]) for s in (-1.0, 1.0)]  # hole mouths, C frame
RECESS_C = np.array([PORT_U, POST_HY, 0.0])                                         # recess mouth, C frame

# -- the cylinder (C frame): a size D aluminium medical oxygen cylinder, empty ------------------------------------------
CYL_R = 0.0555                       # 111 mm (4.38 in) diameter
CYL_LEN = 0.419                      # 16.5 in from the neck ring to the base
CYL_MASS = 2.4                       # kg, empty, with its valve
SHOULDER = ((0.035, 0.012), (0.048, 0.012))   # (radius, length) steps of the domed shoulder after the neck
CYL_X0 = POST_LEN + NECK_LEN         # where the shoulder starts (C frame x)
CYL_END = CYL_X0 + CYL_LEN           # the cylinder's base (C frame x)

# -- the regulator (R frame) --------------------------------------------------------------------------------------------
WASHER_T = 0.0016                    # g: sealing washer thickness
WASHER_R = 0.008                     # d3 = 16 mm
INLET_Y = -POST_HY - WASHER_T        # the yoke's inlet face (the washer's back)
PLATE_T = 0.010                      # inlet plate thickness (y)
PLATE_X = (-0.012, 0.020)            # inlet plate extent along x: covers the outlet and both pins
FRAME_HZ = 0.023                     # the yoke frame's outer half height (z)
WINDOW_HZ = 0.0145                   # its window's inner half height: 0.5 mm either side of the 28 mm post
SLIDE_GAP = 0.004                    # total y play in the window with the pins and screw clear of the post
BAR_IN_Y = INLET_Y + 2 * POST_HY + WASHER_T + PIN_LEN + SLIDE_GAP   # inner face of the screw bar
BAR_T = 0.010
BAR_X = (-0.010, 0.010)              # top, bottom and screw bars along x
PINS_R = [np.array([PIN_DX, INLET_Y + PIN_LEN / 2, s * PIN_DZ]) for s in (-1.0, 1.0)]   # pin centres
PIN_TIP_Y = INLET_Y + PIN_LEN
BODY_W, BODY_HZ = 0.040, 0.023         # regulator body: a 40 x 65 x 46 mm block (x, y, z), flat sides for the grip
BODY_Y = (INLET_Y - PLATE_T - 0.065, INLET_Y - PLATE_T)
GAUGE_R, GAUGE_T = 0.020, 0.012                                             # contents gauge on the body's end
BARB = (0.003, 0.015, -0.012)        # hose barb: radius, length, height below the axis, out of the gauge end
REG_MASS = 0.36                      # kg without the T-screw
GRIP_Y = (BODY_Y[0] + BODY_Y[1]) / 2       # where the right hand grips: the middle of the body (the fingers are 5 cm wide)

# T-screw: a 1.75 mm pitch (M12 coarse) screw through the screw bar, with a 60 mm T-bar
PITCH = 0.00175
TIP_R = 0.0025                       # flat tip, 5 mm across
SHAFT_R = 0.004
TIP0_Y = BAR_IN_Y - 0.0005           # tip face at the start, 0.5 mm inside the window
TBAR_R, TBAR_HALF = 0.005, 0.030     # T-bar: 10 mm round, 60 mm long, lying along x at the start
TBAR0_Y = BAR_IN_Y + BAR_T + 0.021 + TBAR_R   # T-bar centre at the start
SCREW_MASS = 0.030
TRAVEL = 0.0125                      # the screw can advance this far (past the recess floor)
TIP_STIFFNESS = 2.0e5                # N/m: how fast the clamp force rises past first contact (28 N at 29 degrees)
TIP_FORCE = 60.0                     # N: clamp force cap
TURN_TORQUE0 = 0.05                  # N m: torque to turn the T-screw when it is not loaded
TORQUE_PER_N = 1.5 / 60.0            # N m per N of clamp force: 1.55 N m to turn it at the 60 N cap
HAND_TIGHT = 0.75                    # N m: hand-tight, it takes at least this much to turn it further (28 N clamp)
TURN_SPEED = math.radians(100.0)     # rad/s: the turning speed at which the simulated damping gives that torque

# -- the regulator's start: on the table in front of the right arm, as it will be mounted -------------------------------
REG_START = (PORT_X, -0.130, TABLE + FRAME_HZ + 0.001)
SEATED = (PORT_X, 0.0, AXIS_Z)       # the R origin when seated on an unmoved cylinder

# -- the cradle: two wooden V saddles and an end board ------------------------------------------------------------------
SADDLE_X = (0.44, 0.72)
SADDLE_W = 0.030                     # along x
BOARD = (0.070, 0.012)               # V board: length down the slope, thickness
V_APEX_Z = AXIS_Z - CYL_R / math.sin(math.radians(45.0))
END_BOARD_X = POST_TOP_X + CYL_END + 0.001   # face of the end board, 1 mm behind the cylinder's base


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def quat_matrix(xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(xyzw, float) / np.linalg.norm(xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angle_between(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.acos(max(-1.0, min(1.0, c)))


def turn_torque(force: float) -> float:
    """The torque that turns the T-screw further, with the tip pressing on the recess with ``force``."""
    return TURN_TORQUE0 + TORQUE_PER_N * force


def tip_y(d: float) -> float:
    """The screw tip's face (R frame y) at slide position ``d`` (0 at the start, negative as it advances)."""
    return TIP0_Y + d


def tip_target(theta: float) -> float:
    """Where the thread puts the tip for a T-handle angle ``theta`` about +y (tightening turns it negative)."""
    return float(np.clip(PITCH * theta / (2 * math.pi), -TRAVEL, 0.0))


# -- the fit (C frame) --------------------------------------------------------------------------------------------------
SEAT_FACE = 0.001                    # washer within 1 mm of the port face all round
SEAT_TILT = math.radians(3.0)
PIN_IN = 0.003                       # both pins at least 3 mm into their holes
HOLE_TOL = HOLE_W / 2 + 0.0005       # a pin's axis inside its hole (half width plus 0.5 mm)
TIP_TOL = 0.0012                     # the tip's axis within 1.2 mm of the recess's centre
ON_POST_TILT = math.radians(15.0)


def fit(p_rc, R_rc, d: float, theta: float) -> dict:
    """How the regulator sits on the valve, from its pose in the cylinder frame (``p_rc``, ``R_rc``: the R frame's
    origin and rotation in C) and its screw (slide ``d``, T-handle angle ``theta``)."""
    p_rc, R_rc = np.asarray(p_rc, float), np.asarray(R_rc, float)

    def to_c(q):
        return p_rc + R_rc @ np.asarray(q, float)

    turn = angle_between(np.eye(3), R_rc)
    # squareness: the angle between the inlet's axis and the outlet's (turning about that axis doesn't unseat it)
    tilt = math.acos(max(-1.0, min(1.0, float(R_rc[1, 1]))))
    # the yoke over the post: its near edge past the post's free end, its far edge short of the neck, and the post
    # inside its window
    near, far = to_c((PLATE_X[0], 0.0, 0.0))[0], to_c((PLATE_X[1], 0.0, 0.0))[0]
    on_post = bool(near >= 0.003 and far <= POST_LEN + 0.003 and abs(p_rc[2]) < 0.004 and abs(p_rc[1]) < 0.012
                   and turn < ON_POST_TILT)
    # pins: depth past the port face and lateral offset from each hole's axis
    depths, lateral = [], []
    for pin, hole in zip(PINS_R, HOLES_C):
        tip = to_c((pin[0], PIN_TIP_Y, pin[2]))
        depths.append(float(tip[1] - (-POST_HY)))
        lateral.append(float(max(abs(tip[0] - hole[0]), abs(tip[2] - hole[2]))))
    pins_in = bool(min(depths) >= PIN_IN and max(lateral) <= HOLE_TOL)
    # the washer's face against the port face, all round
    rim = [to_c((WASHER_R * math.cos(a), -POST_HY, WASHER_R * math.sin(a))) for a in np.linspace(0, 2 * math.pi, 8,
                                                                                                endpoint=False)]
    face_gap = float(max(abs(q[1] + POST_HY) for q in rim))
    seated = bool(pins_in and face_gap <= SEAT_FACE and tilt <= SEAT_TILT)
    # the screw tip in the recess, tightened past first contact
    tip = to_c((0.0, tip_y(d), 0.0))
    tip_in = bool(math.hypot(tip[0] - RECESS_C[0], tip[2] - RECESS_C[2]) <= TIP_TOL and tip[1] <= POST_HY - 0.0005)
    overshoot = float(d - tip_target(theta))
    force = float(min(TIP_FORCE, max(0.0, TIP_STIFFNESS * overshoot)))
    torque = turn_torque(force)
    clamped = bool(seated and tip_in and torque >= HAND_TIGHT)
    return dict(on_post=on_post, pin_depth=min(depths), pin_lateral=max(lateral), pins_in=pins_in, face_gap=face_gap,
                tilt=tilt, seated=seated, tip_in=tip_in, overshoot=overshoot, clamped=clamped,
                force=force, torque=torque)


def seated_pose_c() -> tuple[np.ndarray, np.ndarray]:
    """The R frame in C when seated."""
    return np.array([PORT_U, 0.0, 0.0]), np.eye(3)


def turns_to_clamp() -> float:
    """Turns of the T-handle from the start until the tip touches the recess floor, seated."""
    travel = TIP0_Y - (POST_HY - RECESS_DEPTH)
    return travel / PITCH


# Regulator outline for clearance checks (R frame), with the T-bar's sweep
REG_BOX = (np.array([-TBAR_HALF, BODY_Y[0] - GAUGE_T - BARB[1], -max(FRAME_HZ, TBAR_HALF)]),
           np.array([TBAR_HALF, TBAR0_Y + TBAR_R, max(FRAME_HZ, TBAR_HALF)]))
