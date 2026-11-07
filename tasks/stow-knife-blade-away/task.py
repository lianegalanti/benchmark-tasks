"""On the station table, in front of the robot and a little to its right: a wooden cutting board (30 x 30 cm). A kitchen
knife (22 cm: a 10 cm brown wooden handle and a 12 cm steel blade) lies flat on its side on the board, along the
robot's forward direction, its blade pointing at the robot and its handle away from it. In front of the left arm: an
open, shallow wooden drawer whose front panel, with a pull, faces the robot. It holds a knife tray with three
slots running away from the robot, separated by low dividers. The two left slots already hold a knife each, lying
flat with the handle toward the drawer front; the right slot, next to the board, is empty.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 \\
           --instruction "Put the knife away in the empty slot of the knife tray in the drawer." ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 2.0

INSTRUCTION = "Put the knife away in the empty slot of the knife tray in the drawer."

WOOD = (0.62, 0.45, 0.28)       # board
DRAWER = (0.80, 0.70, 0.52)     # light birch drawer
INSERT = (0.55, 0.40, 0.25)     # darker tray dividers
HANDLE = (0.20, 0.20, 0.22)     # the stored knives' black handles
STEEL = (0.78, 0.80, 0.83)
KNOB = (0.25, 0.25, 0.27)


def _f(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


def _box(name, lo, hi, color) -> Box:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Box(name, pos=_f((lo + hi) / 2), size=_f(hi - lo), color=color)


# -- objects ---------------------------------------------------------------------------------------------------------
KNIFE = UsdBody("knife", usd="assets/knife.usda", pos=_f(g.START), rot_xyzw=g.yaw_xyzw(g.START_YAW),
                size=(g.KNIFE_LEN, 0.03, g.HANDLE_T), mass=g.KNIFE_MASS, friction=(1.2, 1.0), torsional_patch=0.01)

BOARD = _box("cutting_board", (*g.BOARD_LO, g.TOP), (*g.BOARD_HI, g.BOARD_TOP), WOOD)


def _drawer() -> list[Box]:
    lo_y, hi_y = g.RIGHT_IN - g.WALL_T, g.LEFT_IN + g.WALL_T
    front, back = g.FRONT_IN - g.FRONT_T, g.BACK_IN + g.WALL_T
    mid_y = (g.RIGHT_IN + g.LEFT_IN) / 2
    parts = [
        _box("drawer_floor", (g.FRONT_IN, g.RIGHT_IN, g.TOP), (g.BACK_IN, g.LEFT_IN, g.FLOOR), DRAWER),
        _box("drawer_front", (front, lo_y, g.TOP), (g.FRONT_IN, hi_y, g.WALL_TOP), DRAWER),
        _box("drawer_back", (g.BACK_IN, lo_y, g.TOP), (back, hi_y, g.WALL_TOP), DRAWER),
        _box("drawer_right", (g.FRONT_IN, lo_y, g.TOP), (g.BACK_IN, g.RIGHT_IN, g.WALL_TOP), DRAWER),
        _box("drawer_left", (g.FRONT_IN, g.LEFT_IN, g.TOP), (g.BACK_IN, hi_y, g.WALL_TOP), DRAWER),
        _box("drawer_pull", (front - 0.015, mid_y - 0.04, g.TOP + 0.022), (front, mid_y + 0.04, g.TOP + 0.034), KNOB),
    ]
    for i, (a, b) in enumerate(zip(g.SLOTS[:-1], g.SLOTS[1:])):
        parts.append(_box(f"tray_divider{i}", (g.FRONT_IN, a[1], g.FLOOR), (g.BACK_IN, b[0], g.FLOOR + g.DIVIDER_H),
                          INSERT))
    return parts


def _stored_knife(name, slot, handle_len, blade_len, handle_w, blade_w) -> list[Box]:
    """A knife already in the tray, handle toward the drawer front. Fixed in place (decor and the house convention)."""
    y = (slot[0] + slot[1]) / 2
    x0 = g.FRONT_IN + 0.025
    return [
        _box(f"{name}_handle", (x0, y - handle_w / 2, g.FLOOR), (x0 + handle_len, y + handle_w / 2, g.FLOOR + 0.022),
             HANDLE),
        _box(f"{name}_blade", (x0 + handle_len, y - handle_w / 2, g.FLOOR),
             (x0 + handle_len + blade_len, y - handle_w / 2 + blade_w, g.FLOOR + 0.003), STEEL),
    ]


STORED = [*_stored_knife("paring_knife", g.SLOTS[1], 0.10, 0.11, 0.024, 0.025),
          *_stored_knife("chef_knife", g.SLOTS[2], 0.12, 0.14, 0.026, 0.040)]
OBJECTS = [KNIFE, BOARD, *_drawer(), *STORED]

# -- scoring -----------------------------------------------------------------------------------------------------------
FALLEN_Z = g.TOP - 0.05          # any part of the knife this far below the table top: it went off the table
LIFT_M = 0.03                    # the knife's lowest point this far above the board's top counts as lifted
ON_FLOOR = 0.005                 # resting on the tray floor: its lowest point within this of the floor's top
FLAT = math.radians(10.0)        # the knife's long axis within this of horizontal
HEADING = math.radians(20.0)     # handle to tip, seen from above, within this of straight back (+x)
HOLD_R = 0.03                    # a gripper holds the knife when its tool point is this close to the centre line ...
HOLD_OPEN = 0.030                # ... and its finger pads are less than this far apart (the handle is 2.4 cm)
STILL = 0.01                     # m/s


def holds(tcp, opening: float, pos, rot) -> bool:
    d = np.linalg.norm(g.knife_axis(pos, rot, 45) - np.asarray(tcp), axis=1).min()
    return bool(d < HOLD_R and opening < HOLD_OPEN)


def classify(pos, rot) -> dict:
    """Where a knife posed at ``pos``/``rot`` (origin, rotation; world) lies, by geometry alone."""
    pos, rot = np.asarray(pos, float), np.asarray(rot, float)
    c = g.knife_corners(pos, rot)
    low = float(c[:, 2].min())
    axis = rot[:, 0]
    flat = abs(axis[2]) <= math.sin(FLAT)
    heading = math.atan2(axis[1], axis[0])  # 0: tip toward the back of the drawer, handle toward its front
    # in the tray: every corner inside the drawer's inner walls, its lowest point below their top
    in_tray = bool(np.all(c[:, 0] > g.TRAY_LO[0]) and np.all(c[:, 0] < g.TRAY_HI[0])
                   and np.all(c[:, 1] > g.TRAY_LO[1]) and np.all(c[:, 1] < g.TRAY_HI[1])
                   and low < g.WALL_TOP)
    in_slot = bool(in_tray and np.all(c[:, 1] > g.EMPTY[0]) and np.all(c[:, 1] < g.EMPTY[1])
                   and low < g.FLOOR + ON_FLOOR)
    return {"low": low, "fallen": low < FALLEN_Z, "lifted": low >= g.BOARD_TOP + LIFT_M, "in_tray": in_tray,
            "in_slot": in_slot, "flat": bool(flat), "handle_front": abs(heading) <= HEADING,
            "stowed": bool(in_slot and flat and abs(heading) <= HEADING)}


class Monitor:
    """The knife over the episode: whether it ever fell off the table or was lifted, and where it is now."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.fallen, self.lifted = False, False
        self.now: dict = {"stowed": False, "in_tray": False}
        self.held, self.still, self.score = False, False, 0.0

    def update(self, pos, rot, lin_vel=(0.0, 0.0, 0.0), grippers=()) -> float:
        """One control step: the knife origin's pose and velocity, and each gripper's (tool point, pad opening)."""
        k = classify(pos, rot)
        self.fallen |= k["fallen"]
        self.lifted |= k["lifted"]
        self.held = any(holds(t, o, pos, rot) for t, o in grippers)
        self.still = bool(np.linalg.norm(lin_vel) < STILL)
        self.now = k
        if self.fallen:
            self.score = 0.0
        elif k["stowed"] and not self.held:
            self.score = 1.0
        elif k["in_tray"] and not self.held:
            self.score = 0.5
        else:
            self.score = 0.25 if self.lifted else 0.0
        return self.score

    def success(self) -> bool:
        return bool(not self.fallen and self.now["stowed"] and not self.held and self.still)


MONITOR = Monitor()
on_reset(lambda environment=None: MONITOR.reset())


@on_step
def _monitor(environment) -> None:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.scene import object_states

    from scenekit import arm_joints

    knife = object_states(environment, ["knife"])["knife"]
    grippers = []
    for side in ("left", "right"):
        tcp, _ = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        grippers.append((tcp, float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())))
    MONITOR.update(knife.pos, g.quat_to_matrix(knife.quat_xyzw), knife.lin_vel, grippers)


def success(objects) -> bool:
    """The knife lies flat in the empty slot, handle toward the drawer front, let go and still (held 2 s)."""
    return MONITOR.success()


def score(objects) -> float:
    """0 if the knife ever went off the table. Otherwise, at the end: 1.0 stowed (flat in the empty slot, handle
    toward the drawer front within 20 degrees, let go); 0.5 in the tray, let go, but blade toward the front, out of
    its slot or not flat; 0.25 if it was ever lifted 3 cm off the board; else 0."""
    return MONITOR.score


# -- the scripted demo -----------------------------------------------------------------------------------------------
# The right arm takes the handle, turns the knife a quarter turn and lays it back on the board, blade to the robot's
# right. The left arm takes the handle from the other side (its gripper turned half a turn from the right arm's),
# turns the knife another quarter turn on the way, and lays it in the empty slot, handle toward the drawer front.
# Neither arm can turn a held knife half a turn in one go (joint 6 stops at +-120 degrees), hence the regrasp.
TRAVEL_Z = g.TOP + 0.10          # knife underside 8.3 cm up: clears the drawer walls by 3.3 cm
RELEASE_GAP = 0.003              # the knife is let go this far above where it will rest
RELEASE = -0.018                 # finger joint at release: pads 3.6 cm apart, 6 mm clear of the handle
PARK = {"left": (0.22, 0.30, g.TOP + 0.14), "right": (0.22, -0.30, g.TOP + 0.14)}
TURN = math.pi / 2


def _tcp_z(floor: float) -> float:
    return floor + g.GRIP[2]


def right_plan() -> list[Waypoint]:
    sx, sy = g.START_GRIP_XY
    mx, my = g.MID_GRIP_XY
    pick_z = _tcp_z(g.START[2])
    return [
        Waypoint(2.0, (sx, sy, TRAVEL_Z), FINGER_OPEN, "above the handle", 0.0),
        Waypoint(1.5, (sx, sy, pick_z), FINGER_OPEN, "down to the handle", 0.0),
        Waypoint(0.8, (sx, sy, pick_z), 0.0, "close", 0.0),
        Waypoint(1.2, (sx, sy, TRAVEL_Z), 0.0, "lift", 0.0),
        Waypoint(2.5, (mx, my, TRAVEL_Z), 0.0, "turn a quarter", TURN),
        Waypoint(1.5, (mx, my, _tcp_z(g.BOARD_TOP + RELEASE_GAP)), 0.0, "lay it down", TURN),
        Waypoint(0.6, (mx, my, _tcp_z(g.BOARD_TOP + RELEASE_GAP)), FINGER_OPEN, "let go", TURN),
        Waypoint(1.0, (mx, my, TRAVEL_Z), FINGER_OPEN, "up", TURN),
        Waypoint(1.8, PARK["right"], FINGER_OPEN, "park", 0.0),
    ]


def left_plan(wait_s: float) -> list[Waypoint]:
    mx, my = g.MID_GRIP_XY
    gx, gy = g.GOAL_GRIP_XY
    return [
        Waypoint(2.0, PARK["left"], FINGER_OPEN, "park", 0.0),
        Waypoint(wait_s - 2.0, PARK["left"], FINGER_OPEN, "wait for the right arm", 0.0),
        Waypoint(2.0, (mx, my, TRAVEL_Z), FINGER_OPEN, "above the handle", -TURN),
        Waypoint(1.5, (mx, my, _tcp_z(g.BOARD_TOP)), FINGER_OPEN, "down to the handle", -TURN),
        Waypoint(0.8, (mx, my, _tcp_z(g.BOARD_TOP)), 0.0, "close", -TURN),
        Waypoint(1.2, (mx, my, TRAVEL_Z), 0.0, "lift", -TURN),
        Waypoint(3.0, (gx, gy, TRAVEL_Z), 0.0, "turn the blade to the back", 0.0),
        Waypoint(1.5, (gx, gy, _tcp_z(g.FLOOR + RELEASE_GAP)), 0.0, "into the slot", 0.0),
        Waypoint(0.6, (gx, gy, _tcp_z(g.FLOOR + RELEASE_GAP)), RELEASE, "let go", 0.0),
        Waypoint(1.0, (gx, gy, TRAVEL_Z), RELEASE, "up", 0.0),
        Waypoint(1.8, PARK["left"], FINGER_OPEN, "park", 0.0),
    ]


def demo():
    right = right_plan()
    return {"right": right, "left": left_plan(sum(w.duration for w in right))}
