"""On the station table, a grey chute (a board with two side rails, tilted 10 degrees, covered by a lid over all but
its top 11 cm) stands on two legs 34 cm in front of the arms, sloping down toward the left. Its open end is 20 cm left
of the midpoint between the arms, 15 cm above the table. An orange rubber ball (5 cm across) waits at its uncovered
top, in front of the right arm, behind a grey gate. At 7.5 s the gate lifts: the ball rolls down inside the covered
chute and flies out of its end at about 0.85 m/s. Untouched, it lands on the bare table about 13 cm past the chute's
end and rolls on off the table's left edge. A light blue plastic basket (an open box, 11 x 16 cm inside, 6.5 cm deep,
with a handle block standing up from its near wall) sits on the table in front of the left arm, nearer the arms.

The robot must catch the ball in the basket before the ball touches the table.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "In a few seconds a ball will fly out of the end of the covered chute. Catch it in the blue basket before it touches the table." ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import motionkit  # noqa: E402
import numpy as np  # noqa: E402
from motionkit import DrivenBox, Shape, UsdBody, lift  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import TABLE_CENTER, TABLE_SIZE  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box, Sphere  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the score keeps what the ball did over the whole trial (caught, touched the table, fell off), which the
# MONITOR below remembers and clears at every episode start (motionkit.on_reset); "best" couldn't express that a
# ball that touched the table first is no longer a catch.
SCORE_OVER = "end"

EDGE_Y = TABLE_CENTER[1] + TABLE_SIZE[1] / 2  # 0.65: the table's left edge, where an untouched ball goes off
G = 9.81


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the ball ------------------------------------------------------------------------------------------------------
LINE_X = 0.34  # the chute's centre line and the ball's flight, metres in front of the arm bases
R = 0.025  # a 5 cm rubber ball
BALL_MASS = 0.05

# -- the chute -----------------------------------------------------------------------------------------------------
THETA = math.radians(10.0)  # tilt: rises toward the right (-y)
MOUTH_Y = 0.20  # where the chute's floor ends (its open end), left of the midpoint
MOUTH_H = 0.15  # the floor's height above the table there
CHUTE_LEN = 0.36  # along the slope
BOARD_T = 0.012  # the floor board's thickness
START_S = 0.30  # the ball's contact point at the start, up the slope from the open end
RAIL_W, RAIL_GAP = 0.006, 0.006  # side rails, and the gap between the ball and each rail
ROOF_GAP, ROOF_T = 0.012, 0.006  # the cover: this far above the ball's top, this thick
RAIL_H = 2 * R + ROOF_GAP  # the rails stand this high above the floor; the cover lies on them
INNER_HALF = R + RAIL_GAP  # half the channel's inside width
OUTER_HALF = INNER_HALF + RAIL_W
GREY, DARK = (0.55, 0.56, 0.58), (0.3, 0.3, 0.32)

DOWN = np.array([0.0, math.cos(THETA), -math.sin(THETA)])  # down the slope
NORMAL = np.array([0.0, math.sin(THETA), math.cos(THETA)])  # the floor's upward normal
ROT = (math.sin(-THETA / 2), 0.0, 0.0, math.cos(-THETA / 2))  # about +x: turns +y onto DOWN and +z onto NORMAL
MOUTH = np.array([LINE_X, MOUTH_Y, TOP + MOUTH_H])  # the middle of the floor's lower edge


def on_chute(s: float, x: float = LINE_X) -> np.ndarray:
    """The point on the chute's floor ``s`` metres up the slope from its open end."""
    return np.array([x, MOUTH_Y, TOP + MOUTH_H]) - s * DOWN


def _slab(name: str, x: float, width: float, below: float, above: float, s0: float, s1: float, color) -> Shape:
    """A tilted box spanning ``s0..s1`` along the slope, from ``below`` under the floor's surface to ``above`` over it."""
    mid = on_chute((s0 + s1) / 2, x) + (above - below) / 2 * NORMAL
    return Shape(name, pos=_floats(mid), size=(width, s1 - s0, above + below), color=color, rot_xyzw=ROT)


START = on_chute(START_S) + (R + 0.001) * NORMAL  # the ball's centre, 1 mm off the floor: it settles onto it
BALL = Sphere("ball", pos=_floats(START), radius=R, mass=BALL_MASS, color=(0.95, 0.45, 0.1))

# -- the gate ------------------------------------------------------------------------------------------------------
GATE_AT_S, GATE_OVER_S, GATE_RISE = 7.5, 0.3, 0.08  # it lifts 8 cm, from 7.5 to 7.8 s
GATE = (2 * INNER_HALF - 0.004, 0.01, 0.055)  # 2 mm clear of each rail
GATE_FACE_Y = float(START[1] + R + 0.001)  # its uphill face, 1 mm downhill of the ball
_gate_y = GATE_FACE_Y + GATE[1] / 2
_gate_floor = TOP + MOUTH_H + (MOUTH_Y - (GATE_FACE_Y + GATE[1])) * math.tan(THETA)  # floor under its downhill face
GATE_POS = _floats((LINE_X, _gate_y, _gate_floor + 0.002 + GATE[2] / 2))  # 2 mm clear of the floor
GATE_BOX = DrivenBox("gate", pos=GATE_POS, size=GATE, color=(0.4, 0.4, 0.42),
                     path=lift(GATE_POS, GATE_RISE, GATE_AT_S, GATE_OVER_S))

# The cover runs from the open end up to 1 cm (along the slope) short of the gate's downhill face, so once the ball
# has passed the gate nothing can reach it until it leaves the chute; the gate and the waiting ball stay in view.
ROOF_TOP_S = (MOUTH_Y - (GATE_FACE_Y + GATE[1])) / math.cos(THETA) - 0.01

CHUTE = [
    _slab("chute_floor", LINE_X, 2 * OUTER_HALF, BOARD_T, 0.0, 0.0, CHUTE_LEN, GREY),
    *(_slab(f"chute_rail_{side}", LINE_X + sign * (INNER_HALF + RAIL_W / 2), RAIL_W, 0.002, RAIL_H, 0.0, CHUTE_LEN,
            DARK) for side, sign in (("near", -1), ("far", 1))),
    _slab("chute_cover", LINE_X, 2 * OUTER_HALF, -RAIL_H, RAIL_H + ROOF_T, 0.0, ROOF_TOP_S, DARK),
]

# Two legs (6 x 3 cm, wood) carry the floor board, from the table up to its underside at their middles.
LEG_S = (0.06, 0.32)
LEG = (0.06, 0.03)


def _underside(s: float) -> float:
    return float(on_chute(s)[2] - BOARD_T / math.cos(THETA))


LEGS = [
    Box(f"chute_leg_{i}", pos=_floats((LINE_X, on_chute(s)[1], (TOP + _underside(s)) / 2)),
        size=(LEG[0], LEG[1], _underside(s) - TOP), color=(0.45, 0.36, 0.25))
    for i, s in enumerate(LEG_S)
]

# -- the basket ----------------------------------------------------------------------------------------------------
# One rigid body (assets/basket.usda, written by make_basket.py from these numbers). Its origin is the middle of the
# floor's underside; x points away from the arms, y to the left, like the station's.
IN_X, IN_Y = 0.11, 0.16  # inside, across and along the ball's flight
WALL, FLOOR, RIM = 0.008, 0.008, 0.065  # wall and floor thickness; the rim's height above the underside
# The handle: a block 3 cm thick (x) and 6 cm wide (y) on the near wall's centre line, from 2 cm down the wall to
# 6 cm above the rim. The pads close on its 3 cm (so they squeeze at the gripper's full 50 N) over 4.7 cm of its
# height; the basket's weight, 6 cm out from it, only tries to tilt it between the pads, which their height resists.
# (A 1.2 cm tab gripped over 2.5 cm lost the basket in the first sim run.)
TAB = (0.03, 0.06, 0.08)  # thickness (x), width (y), height (z)
TAB_X = -(IN_X / 2 + WALL / 2)
TAB_Z = (RIM - 0.02, RIM + 0.06)
BASKET_MASS = 0.10
BLUE = (0.15, 0.32, 0.8)
BASKET_START = (0.20, 0.52)  # its origin's x, y on the table at the start
BASKET = UsdBody("basket", usd="assets/basket.usda", pos=_floats((*BASKET_START, TOP + 0.001)),
                 size=(IN_X + 2 * WALL, IN_Y + 2 * WALL, RIM), mass=BASKET_MASS, color=BLUE, friction=(0.8, 0.7),
                 torsional_patch=0.01)


def basket_boxes() -> dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """The basket's parts as (centre, size) in its own frame: what make_basket.py writes to the USD file."""
    out_x, out_y = IN_X + 2 * WALL, IN_Y + 2 * WALL
    wz = (FLOOR + RIM) / 2
    return {
        "floor": ((0.0, 0.0, FLOOR / 2), (out_x, out_y, FLOOR)),
        "wall_near": ((-(IN_X + WALL) / 2, 0.0, wz), (WALL, out_y, RIM - FLOOR)),
        "wall_far": (((IN_X + WALL) / 2, 0.0, wz), (WALL, out_y, RIM - FLOOR)),
        "wall_right": ((0.0, -(IN_Y + WALL) / 2, wz), (IN_X, WALL, RIM - FLOOR)),
        "wall_left": ((0.0, (IN_Y + WALL) / 2, wz), (IN_X, WALL, RIM - FLOOR)),
        "tab": ((TAB_X, 0.0, sum(TAB_Z) / 2), (TAB[0], TAB[1], TAB_Z[1] - TAB_Z[0])),
    }


OBJECTS = [*CHUTE, *LEGS, GATE_BOX, BALL, BASKET]

# -- the ball's run, untouched (analytic) ----------------------------------------------------------------------------
ACCEL = 5 / 7 * G * math.sin(THETA)  # a solid ball rolling without slipping down the slope
EXIT_SPEED = math.sqrt(2 * ACCEL * START_S)  # when its contact point reaches the floor's end
RAMP_S = EXIT_SPEED / ACCEL  # from release to leaving the chute
# The gate lets the ball go somewhere while it lifts: from its first movement to the end of the lift.
EXIT_EARLIEST = GATE_AT_S + RAMP_S
EXIT_LATEST = GATE_AT_S + GATE_OVER_S + RAMP_S
# Faster than sqrt(g R cos(theta)) the ball can't follow the floor round its edge: it flies off at once.
assert EXIT_SPEED**2 / R > G * math.cos(THETA)


def flight(t: float, speed: float = EXIT_SPEED) -> np.ndarray:
    """The ball's centre ``t`` seconds after it leaves the chute at ``speed`` (m/s), flying freely."""
    p0 = MOUTH + R * NORMAL  # its contact point on the floor's end
    return p0 + speed * DOWN * t + np.array([0.0, 0.0, -0.5 * G * t * t])


def fall_time(z: float, speed: float = EXIT_SPEED) -> float:
    """When the flying ball's centre comes down to height ``z``."""
    z0, vz = float((MOUTH + R * NORMAL)[2]), -speed * math.sin(THETA)
    return (vz + math.sqrt(vz * vz + 2 * G * (z0 - z))) / G


LAND_Y = float(flight(fall_time(TOP + R))[1])  # where an untouched ball lands on the table


def _hits_box(path: np.ndarray, lo, hi) -> bool:
    """Whether a ball (radius R) whose centre runs along ``path`` (N x 3) touches the box ``lo..hi``."""
    d = np.maximum(0.0, np.maximum(np.asarray(lo) - path, path - np.asarray(hi)))
    return bool(np.any(np.linalg.norm(d, axis=1) < R))


def caught_by_basket_at(y: float, speed: float = EXIT_SPEED) -> bool:
    """Whether a basket standing on the table at (LINE_X, ``y``) catches the flying ball cleanly: the ball passes over
    its right wall without touching it and comes down onto its floor at least a radius inside both end walls."""
    t_floor = fall_time(TOP + FLOOR + R, speed)
    path = np.array([flight(t, speed) for t in np.linspace(0.0, t_floor, 400)])
    x0, x1 = LINE_X - IN_X / 2 - WALL, LINE_X + IN_X / 2 + WALL
    right_wall = ((x0, y - IN_Y / 2 - WALL, TOP), (x1, y - IN_Y / 2, TOP + RIM))
    land = path[-1][1]
    return (not _hits_box(path, *right_wall)) and y - IN_Y / 2 + R <= land <= y + IN_Y / 2 - R


LEG_CLEAR = 0.015  # the basket stays this far from the chute's leg under its open end


def catch_window(speed: float = EXIT_SPEED, step: float = 0.001) -> tuple[float, float]:
    """The basket positions along y (its centre on the ball's line, standing on the table, square to the table) that
    catch the ball cleanly and keep the basket LEG_CLEAR from the leg under the chute's end."""
    leg_face = on_chute(LEG_S[0])[1] + LEG[1] / 2
    ys = [y for y in np.arange(0.15, 0.55, step)
          if y - IN_Y / 2 - WALL >= leg_face + LEG_CLEAR and caught_by_basket_at(y, speed)]
    return float(min(ys)), float(max(ys))


# -- scoring -------------------------------------------------------------------------------------------------------
TOUCH_M = 0.003  # the ball's underside within 3 mm of the table top: it touched the table
FALLEN_Z = TOP - 0.05  # the ball's centre this far below the table top: it went over an edge
CATCH_S = 1.0  # a catch: in the basket within this long after the ball was last inside the covered chute
UPRIGHT_COS = math.cos(math.radians(30.0))  # the basket counts while tilted at most 30 degrees


def _rotation(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def in_basket(ball, basket) -> bool:
    """The basket is upright (tilted 30 degrees at most) and the ball's centre is inside its walls, above its floor's
    underside and below its rim."""
    rot = _rotation(basket.quat_xyzw)
    if rot[2, 2] < UPRIGHT_COS:
        return False
    p = rot.T @ (np.asarray(ball.pos, float) - np.asarray(basket.pos, float))
    return bool(abs(p[0]) < IN_X / 2 and abs(p[1]) < IN_Y / 2 and 0.0 < p[2] < RIM)


def over_basket(ball, basket) -> bool:
    """The ball's centre is over the basket's outer footprint (in the basket's own frame, walls included)."""
    p = _rotation(basket.quat_xyzw).T @ (np.asarray(ball.pos, float) - np.asarray(basket.pos, float))
    return bool(abs(p[0]) <= IN_X / 2 + WALL and abs(p[1]) <= IN_Y / 2 + WALL)


def touches_table(ball, basket) -> bool:
    """The ball's underside is within 3 mm of the table top, outside the basket's footprint. The underside is the
    centre's height less the radius: the reported bounds of a spinning sphere are its turned box's, up to R*sqrt(3)
    below the centre, which put a ball lying on the basket's floor (centre 3.3 cm up) "below" the table top. Over
    the basket the ball lies on its 8 mm floor and can't touch the table."""
    return bool(ball.pos[2] - R < TOP + TOUCH_M and not over_basket(ball, basket))


def in_cover(ball) -> bool:
    """The ball's centre is inside the covered part of the chute."""
    rel = np.asarray(ball.pos, float) - MOUTH
    s, n = -float(rel @ DOWN), float(rel @ NORMAL)
    return bool(0.0 <= s <= ROOF_TOP_S and 0.0 < n < RAIL_H and abs(rel[0]) < INNER_HALF)


class Monitor:
    """What the ball has done so far this episode."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.last_covered = None  # episode time the ball was last seen inside the covered chute
        self.out = False  # it came out of the chute's open end
        self.touched = False  # it touched the table
        self.fallen = False  # it went over an edge
        self.caught = False  # it went into the basket straight from the chute

    def update(self, ball, basket, t: float) -> float:
        """Take one control step's states at episode time ``t``; return the score now (repeating a step is harmless)."""
        if in_cover(ball):
            self.last_covered = t
        if self.last_covered is not None and ball.pos[1] > MOUTH_Y:
            self.out = True
        if touches_table(ball, basket):
            self.touched = True
        if ball.pos[2] < FALLEN_Z:
            self.fallen = True
        inside = in_basket(ball, basket)
        if (inside and self.out and not self.touched and t - self.last_covered <= CATCH_S + 1e-9):
            self.caught = True
        if self.caught:
            return 0.75
        if self.out and inside:
            return 0.5
        if self.out and not self.fallen:
            return 0.25
        return 0.0


MONITOR = Monitor()
motionkit.on_reset(MONITOR.reset)


DEBUG = os.environ.get("BALL_DEBUG") == "1"  # BALL_DEBUG=1: print the basket, the ball and the left pads
_DEBUG_EVERY = 0.5
_debug = {"next": 0.0}
motionkit.on_reset(lambda: _debug.update(next=0.0))


def _debug_print(objects, t: float) -> None:
    """Every 0.5 s: the basket's origin and tilt and its offset from the catch spot; after the gate lifts, the ball
    (every step while it flies); the left pads' gap and where the left fingers are (the midpoint of their link
    origins)."""
    ball, basket = objects["ball"], objects["basket"]
    flying = GATE_AT_S <= t <= EXIT_LATEST + 1.5
    if t + 1e-9 < _debug["next"] and not flying:
        return
    _debug["next"] = max(_debug["next"], math.floor(t / _DEBUG_EVERY + 1e-9) * _DEBUG_EVERY + _DEBUG_EVERY)
    rot = _rotation(basket.quat_xyzw)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rot[2, 2]))))
    yaw = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    bp = np.asarray(basket.pos)
    line = (f"[ball-debug] t={t:5.2f} basket=({bp[0]:.3f},{bp[1]:.3f},{bp[2] - TOP:+.3f}) tilt={tilt:4.1f} "
            f"yaw={yaw:+5.1f} off_spot=({bp[0] - LINE_X:+.3f},{bp[1] - CATCH_Y:+.3f})")
    fingers = motionkit.finger_gap("left")
    if fingers is not None:
        gap, tips = fingers
        line += f" left_gap={gap * 1000:5.1f}mm left_fingers=({tips[0]:.3f},{tips[1]:.3f},{tips[2] - TOP:+.3f})"
    if t >= GATE_AT_S:
        p, v = np.asarray(ball.pos), np.asarray(ball.lin_vel)
        line += (f" ball=({p[0]:.3f},{p[1]:.3f},{p[2] - TOP:+.3f}) v=({v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}) "
                 f"in_basket={in_basket(ball, basket)}")
    line += f" out={MONITOR.out} touched={MONITOR.touched} caught={MONITOR.caught} fallen={MONITOR.fallen}"
    print(line, flush=True)


def _observe(objects) -> float:
    t = motionkit.clock()
    out = MONITOR.update(objects["ball"], objects["basket"], t)
    if DEBUG:
        _debug_print(objects, t)
    return out


def success(objects):
    """Caught (straight from the chute into the basket, without touching the table) and lying still in the upright
    basket."""
    _observe(objects)
    return bool(MONITOR.caught and in_basket(objects["ball"], objects["basket"]) and objects["ball"].resting())


def distance(objects):
    """Metres from the ball's centre to the middle of the basket's floor (where it would lie)."""
    basket = objects["basket"]
    goal = np.asarray(basket.pos, float) + _rotation(basket.quat_xyzw) @ np.array([0.0, 0.0, FLOOR + R])
    return float(np.linalg.norm(np.asarray(objects["ball"].pos, float) - goal))


def score(objects):
    """At the end of the trial (success is 1.0): 0.75 if the ball was caught (it went into the upright basket within
    1 s of leaving the covered chute, without touching the table) but did not stay; else 0.5 if it came out of the
    chute and is in the upright basket now (it touched the table on the way); else 0.25 if it came out of the chute and
    has not gone over an edge of the table; else 0 (still in the chute, or fallen off the table)."""
    return _observe(objects)


# -- the scripted demo ---------------------------------------------------------------------------------------------
# The left arm takes the basket by its tab (fingers closing along x, across the tab), carries it to the middle of the
# catch window on the ball's line, sets it down, lets go and moves back out of the ball's way, all before the gate
# lifts. The ball then lands in the standing basket.
GRIP_YAW = math.pi / 2  # the fingers close along x, across the tab
GRASP_DZ = RIM + 0.010 + PAD_TIP_PAST_TCP  # TCP above the basket's underside: pad tips 1 cm over the rim
PLACE_GAP = 0.004  # the basket's underside this far above the table when it is let go
TRAVEL_Z = TOP + 0.17
CLEAR_Z = TOP + 0.16  # pad tips 2 cm over the handle's top
CATCH_Y = float(np.mean(catch_window()))
RETREAT_XY = (0.15, CATCH_Y)  # back toward the arms, the gripper's sides 10 cm and more off the ball's line


def demo():
    """Left arm: the basket from the side to the catch spot on the ball's line, set down and let go by 6.4 s; the
    hand back out of the way by 7.4 s. The ball comes out of the chute at 8.2 to 8.5 s. The lift and carry are slow
    (joint 1 swings 0.9 rad while the wrist turns back the other way) so the basket doesn't swing in the grip."""
    bx, by = BASKET_START
    pick = (bx + TAB_X, by, TOP + GRASP_DZ)
    place = (LINE_X + TAB_X, CATCH_Y, TOP + PLACE_GAP + GRASP_DZ)
    yaw = GRIP_YAW
    return {"left": [
        Waypoint(1.4, (pick[0], pick[1], TRAVEL_Z), FINGER_OPEN, "above the tab", yaw),
        Waypoint(0.8, pick, FINGER_OPEN, "down around the tab", yaw),
        Waypoint(0.6, pick, FINGER_CLOSED, "grip the tab", yaw),
        Waypoint(0.8, (pick[0], pick[1], TRAVEL_Z), FINGER_CLOSED, "lift", yaw),
        Waypoint(1.5, (place[0], place[1], TRAVEL_Z), FINGER_CLOSED, "carry", yaw),
        Waypoint(0.8, place, FINGER_CLOSED, "set down", yaw),
        Waypoint(0.5, place, FINGER_OPEN, "let go", yaw),
        Waypoint(0.4, (place[0], place[1], CLEAR_Z), FINGER_OPEN, "up off the tab", yaw),
        Waypoint(0.6, (*RETREAT_XY, TRAVEL_Z), FINGER_OPEN, "back out of the way", yaw),
        Waypoint(4.0, (*RETREAT_XY, TRAVEL_Z), FINGER_OPEN, "wait", yaw),
    ]}
