"""On the station table, a short grey ramp (a chute with two low side rails, tilted 3 degrees) stands in front of the
right arm, its foot 15 cm right of the midpoint between the arms, sloping down toward the left. An orange rubber ball
(5 cm across) waits at its top behind a grey gate. At 1 s the gate lifts and the ball rolls down the ramp and on
across the bare table toward the table's left edge, 65 cm left of the midpoint, at about 0.42 m/s. Untouched, it
falls off the edge about 3 s after the gate lifts. A blue cup (an open square box, 8 x 8 cm inside, 4 cm high)
stands on the table in front of the left arm, nearer the arms than the ball's path.

The robot must stop the ball before it falls off the table, then put it in the blue cup.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The ball is about to roll off the left edge of the table. Stop it before it falls, then put it in the blue cup." ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import motionkit  # noqa: E402
import numpy as np  # noqa: E402
from motionkit import DrivenBox, Shape, lift  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import TABLE_CENTER, TABLE_SIZE  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Sphere  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": a ball that falls off the table scores 0 even after it was stopped, which "best" can't express. The best
# milestone reached is kept by the MONITOR below instead, and cleared at every episode start (motionkit.on_reset).
SCORE_OVER = "end"

# The table's left edge (y is to the robot's left): where the ball falls off.
EDGE_Y = TABLE_CENTER[1] + TABLE_SIZE[1] / 2  # 0.65

# -- the ball ------------------------------------------------------------------------------------------------------
LINE_X = 0.30  # the ball rolls along this line, metres in front of the arm bases
R = 0.025  # a 5 cm rubber ball
BALL_MASS = 0.06

# -- the ramp ------------------------------------------------------------------------------------------------------
THETA = math.radians(3.0)  # tilt: rises toward the right (-y)
FOOT_Y = -0.15  # where the ramp's top surface meets the table
RAMP_LEN = 0.34  # along the slope
RAMP_W, RAMP_T = 0.10, 0.03  # thick enough that its underside stays inside the table along its whole length
START_S = 0.25  # the ball's contact point at the start, along the slope from the foot
RAIL_W, RAIL_H = 0.006, 0.012  # side rails, standing on the ramp's surface
RAIL_GAP = 0.006  # between the ball and each rail
GREY = (0.55, 0.56, 0.58)

DOWN = np.array([0.0, math.cos(THETA), -math.sin(THETA)])  # down the slope
NORMAL = np.array([0.0, math.sin(THETA), math.cos(THETA)])  # the ramp surface's upward normal
ROT = (math.sin(-THETA / 2), 0.0, 0.0, math.cos(-THETA / 2))  # about +x: turns +y onto DOWN and +z onto NORMAL


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


def on_ramp(s: float, x: float = LINE_X) -> np.ndarray:
    """The point on the ramp's top surface ``s`` metres up the slope from its foot."""
    return np.array([x, FOOT_Y, TOP]) - s * DOWN


def _slab(name: str, x: float, width: float, below: float, above: float, s0: float, s1: float, color) -> Shape:
    """A tilted box spanning ``s0..s1`` along the slope, from ``below`` under the ramp surface to ``above`` over it."""
    mid = on_ramp((s0 + s1) / 2, x) + (above - below) / 2 * NORMAL
    return Shape(name, pos=_floats(mid), size=(width, s1 - s0, above + below), color=color, rot_xyzw=ROT)


# The ramp's top surface runs 1 cm past the foot, into the table, so the ball rolls off it onto the table without a step.
RAMP = _slab("ramp", LINE_X, RAMP_W, RAMP_T, 0.0, -0.01, RAMP_LEN, GREY)
RAILS = [
    _slab(f"ramp_rail_{side}", LINE_X + sign * (R + RAIL_GAP + RAIL_W / 2), RAIL_W, 0.002, RAIL_H, 0.0, RAMP_LEN,
          (0.3, 0.3, 0.32))
    for side, sign in (("near", -1), ("far", 1))
]

START = on_ramp(START_S) + (R + 0.001) * NORMAL  # the ball's centre, 1 mm off the surface: it settles onto it
BALL = Sphere("ball", pos=_floats(START), radius=R, mass=BALL_MASS, color=(0.95, 0.45, 0.1))

# -- the gate ------------------------------------------------------------------------------------------------------
GATE_AT_S, GATE_OVER_S, GATE_RISE = 1.0, 0.3, 0.06  # it lifts 6 cm, from 1.0 to 1.3 s
GATE = (2 * (R + RAIL_GAP), 0.01, 0.05)  # fits between the rails
_gate_face = START[1] + R + 0.001  # its uphill face, 1 mm downhill of the ball
_gate_y = _gate_face + GATE[1] / 2
_gate_floor = TOP + (FOOT_Y - (_gate_face + GATE[1])) * math.tan(THETA)  # ramp surface under its downhill face
GATE_POS = _floats((LINE_X, _gate_y, _gate_floor + 0.002 + GATE[2] / 2))  # 2 mm clear of the ramp
GATE_BOX = DrivenBox("gate", pos=GATE_POS, size=GATE, color=(0.4, 0.4, 0.42),
                     path=lift(GATE_POS, GATE_RISE, GATE_AT_S, GATE_OVER_S))

# -- the cup -------------------------------------------------------------------------------------------------------
CUP_XY = (0.17, 0.22)
CUP_IN = 0.08  # inner width and depth
WALL, FLOOR, WALL_H = 0.01, 0.01, 0.04
CUP_FLOOR_TOP = TOP + FLOOR
CUP_RIM = TOP + WALL_H
BLUE = (0.15, 0.32, 0.8)


def _cup() -> list[Box]:
    x, y = CUP_XY
    out = CUP_IN / 2 + WALL / 2
    span = CUP_IN + 2 * WALL
    z = TOP + WALL_H / 2
    return [
        Box("cup_floor", pos=(x, y, TOP + FLOOR / 2), size=(span, span, FLOOR), color=BLUE),
        Box("cup_wall_near", pos=(x - out, y, z), size=(WALL, span, WALL_H), color=BLUE),
        Box("cup_wall_far", pos=(x + out, y, z), size=(WALL, span, WALL_H), color=BLUE),
        Box("cup_wall_right", pos=(x, y - out, z), size=(CUP_IN, WALL, WALL_H), color=BLUE),
        Box("cup_wall_left", pos=(x, y + out, z), size=(CUP_IN, WALL, WALL_H), color=BLUE),
    ]


OBJECTS = [RAMP, *RAILS, GATE_BOX, BALL, *_cup()]

# -- scoring -------------------------------------------------------------------------------------------------------
FALLEN_Z = TOP - 0.05  # the ball's centre this far below the table top: it has gone over an edge
LIFT_M = 0.03  # the ball's underside this far above the table top counts as picked up
STILL_SPEED, STILL_S = 0.02, 0.5  # stopped: under 2 cm/s for 0.5 s in a row ...
MOVED_M = 0.02  # ... at least 2 cm from where it waited behind the gate


def _in_cup(ball) -> bool:
    """The ball's centre is inside the cup's walls and below its rim."""
    dx, dy = ball.pos[0] - CUP_XY[0], ball.pos[1] - CUP_XY[1]
    return bool(abs(dx) < CUP_IN / 2 and abs(dy) < CUP_IN / 2 and ball.pos[2] < CUP_RIM)


class Monitor:
    """What the ball has done so far this episode: whether it has fallen off the table, the best milestone it
    reached, and since when it has been still."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.fallen, self.best, self.still_since = False, 0.0, None

    def update(self, ball, t: float) -> float:
        """Take one control step's ball state at episode time ``t``; return the score now."""
        if ball.pos[2] < FALLEN_Z:
            self.fallen = True
        moved = np.linalg.norm(ball.pos - START) >= MOVED_M
        if moved and np.linalg.norm(ball.lin_vel) < STILL_SPEED:
            self.still_since = t if self.still_since is None else self.still_since
        else:
            self.still_since = None
        now = 0.0
        if _in_cup(ball):
            now = 0.75
        elif ball.low[2] >= TOP + LIFT_M:
            now = 0.5
        elif self.still_since is not None and t - self.still_since >= STILL_S - 1e-9:
            now = 0.25
        self.best = max(self.best, now)
        return 0.0 if self.fallen else self.best


MONITOR = Monitor()
motionkit.on_reset(MONITOR.reset)


def success(objects):
    """In the cup: centre inside the walls, underside within 1 cm of the cup floor, and still."""
    ball = objects["ball"]
    return bool(_in_cup(ball) and ball.low[2] < CUP_FLOOR_TOP + 0.01 and ball.resting())


def distance(objects):
    """Metres from the ball's centre to where it sits on the middle of the cup floor."""
    goal = np.array([*CUP_XY, CUP_FLOOR_TOP + R])
    return float(np.linalg.norm(objects["ball"].pos - goal))


def score(objects):
    """0 once the ball has fallen off the table (its centre 5 cm below the table top), for the rest of the trial.
    Otherwise the best reached so far: 0.75 once its centre was inside the cup's walls below the rim; 0.5 once its
    underside was 3 cm or more above the table top (picked up); 0.25 once it was still (under 2 cm/s) for 0.5 s,
    at least 2 cm from where it waited behind the gate (stopped)."""
    return MONITOR.update(objects["ball"], motionkit.clock())


# -- the ball's roll, untouched (analytic) ---------------------------------------------------------------------------
G = 9.81
ACCEL = 5 / 7 * G * math.sin(THETA)  # a solid ball rolling without slipping down the slope
# The ball's centre runs parallel to the ramp, then to the table, R above each; the two lines meet at the "knee".
KNEE_Y = FOOT_Y - R * math.tan(THETA / 2)
SLOPE_RUN = (KNEE_Y - START[1]) / math.cos(THETA)  # metres along the slope from the start to the knee
SPEED = math.sqrt(2 * ACCEL * SLOPE_RUN)  # on the table
KNEE_S = GATE_AT_S + SPEED / ACCEL  # when it rolls onto the table


def roll_y(t: float, damping: float = 0.0) -> float:
    """The untouched ball's centre y at episode time ``t``, released when the gate starts to lift. ``damping`` is
    PhysX's angular damping (1/s): rolling, it slows the ball by 2/7 of that rate, v' = -(2/7) c v."""
    if t <= GATE_AT_S:
        return float(START[1])
    if t <= KNEE_S:
        return float(START[1] + 0.5 * ACCEL * (t - GATE_AT_S) ** 2 * math.cos(THETA))
    k = 2 / 7 * damping
    dt = t - KNEE_S
    run = SPEED * dt if k == 0 else SPEED * (1 - math.exp(-k * dt)) / k
    return float(KNEE_Y + run)


def roll_time(y: float, damping: float = 0.0) -> float:
    """When the untouched ball's centre reaches ``y`` (by bisection; inf if it never does)."""
    lo, hi = GATE_AT_S, 60.0
    if roll_y(hi, damping) < y:
        return math.inf
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if roll_y(mid, damping) < y else (lo, mid)
    return hi


# -- the scripted demo ---------------------------------------------------------------------------------------------
# The left arm stops the ball with a fork: its fingers, turned to close across the ball's path (along x), stand
# FORK_GAP apart, narrower than the ball, so it wedges against both fingers' leading edges at once and stays centred
# on the path (a single closed finger, square to the path, is only 3 cm wide and glances the ball aside). Then the
# fingers open, the hand slides back over the ball and closes on it.
STOP_Y = 0.38  # the fork's TCP, across the ball's path in front of the left arm
GRIP_YAW = math.pi / 2  # the fingers close along x, across the path
FORK_GAP = 0.044  # between the finger pads; the ball is 5 cm across
FORK = -FORK_GAP / 2  # finger joint target: each finger stands FORK_GAP / 2 off the TCP
FINGER_HALF_W = 0.016  # a finger's half-width along the path at the ball's centre height (YAM tip mesh, 2 cm up)
REST_Y = STOP_Y - FINGER_HALF_W - math.sqrt(R**2 - (FORK_GAP / 2) ** 2)  # the stopped ball's centre
BARRIER_Z = TOP + 0.005 + PAD_TIP_PAST_TCP  # pad tips 5 mm above the table
ARRIVE_S = roll_time(REST_Y)
TRAVEL_Z = TOP + 0.10
CLEAR_Z = TOP + 2 * R + 0.012 + PAD_TIP_PAST_TCP  # pad tips 12 mm above the ball's top
GRASP_Z = TOP + R  # TCP at the ball's centre
DROP_Z = CUP_RIM + 0.01 + R  # ball's underside 1 cm above the cup's rim


def demo():
    """Left arm: the finger fork down across the ball's path in front of the left arm by 2.0 s (the ball comes at
    about 3.3 s), hold 1.0 s past its arrival while it stops, open the fingers around it, slide over it, grasp, and
    drop it into the cup from just above the rim."""
    x, y, yaw = LINE_X, STOP_Y, GRIP_YAW
    cx, cy = CUP_XY
    plan = [
        Waypoint(1.2, (x, y, TRAVEL_Z), FORK, "above the stop", yaw),
        Waypoint(0.8, (x, y, BARRIER_Z), FORK, "barrier down", yaw),
    ]
    plan.append(Waypoint(ARRIVE_S + 1.0 - sum(w.duration for w in plan), (x, y, BARRIER_Z), FORK,
                         "hold the barrier", yaw))
    plan += [
        Waypoint(0.4, (x, y, BARRIER_Z), FINGER_OPEN, "open around the ball", yaw),
        Waypoint(0.6, (x, REST_Y, GRASP_Z), FINGER_OPEN, "over the ball", yaw),
        Waypoint(0.6, (x, REST_Y, GRASP_Z), FINGER_CLOSED, "close", yaw),
        Waypoint(0.8, (x, REST_Y, TRAVEL_Z), FINGER_CLOSED, "lift", yaw),
        Waypoint(1.5, (cx, cy, TRAVEL_Z), FINGER_CLOSED, "carry", yaw),
        Waypoint(0.8, (cx, cy, DROP_Z), FINGER_CLOSED, "lower", yaw),
        Waypoint(0.5, (cx, cy, DROP_Z), FINGER_OPEN, "release", yaw),
        Waypoint(0.8, (cx, cy, TRAVEL_Z), FINGER_OPEN, "retreat", yaw),
    ]
    return {"left": plan}
