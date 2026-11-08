"""On the station table: a dark wooden picture frame (a 30 x 15 cm shadow box, 3.5 cm deep, 0.5 kg) stands upright in a
grey slotted stand in front of the robot, its glass facing the robot. Behind the glass, a bronze medal on a red ribbon
is mounted in its lower left corner, so the frame is heavier on its left. On the frame's back, centred under the top
rail, is a steel sawtooth hanger: a bar with a row of 11 V-notches (8 mm apart) along its lower edge. Further out, at
the far edge of the arms' reach, a white wooden board (10 cm wide, 22 cm tall) stands upright across the table with a
steel pin sticking 13 mm out of its face toward the robot, 15 cm above the table.

The robot must hang the frame on the pin by its hanger so that it hangs level by itself. Hung from the notch at the
middle of the hanger, the off-centre medal pulls the frame about 13 degrees crooked (its low corner reaches the table);
it hangs level only from the notch two teeth toward the medal's side, which the robot finds by looking at how the
frame hangs and moving it along the teeth.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py ...   (or python demo.py)
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import Shape, UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
assert TABLE_TOP[SCENE] == g.TABLE
# "end": the scorer keeps its own milestones (lifted, dropped) for the whole episode; see Monitor.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Hang the picture frame on the pin in the white board by the sawtooth hanger on its back, so that it "
               "hangs level by itself.")

WHITE = (0.92, 0.92, 0.90)
STEEL = (0.62, 0.63, 0.66)
GREY = (0.45, 0.47, 0.50)


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects at reset."""
    return tuple(float(x) for x in v)


# -- objects ---------------------------------------------------------------------------------------------------------
FRAME = UsdBody("frame", usd="assets/frame.usda", pos=_floats(g.START), size=_floats(2 * g.HALF), mass=g.MASS,
                friction=(0.8, 0.7), torsional_patch=0.01)

# The board's painted face is smooth (friction 0.1, and the lower of the two surfaces rules), so the frame's rails
# barely hold it crooked; the steel pin has friction 0.3.
BOARD = Shape("board", pos=_floats((g.BOARD_X + g.BOARD_T / 2, g.PIN_Y, g.TABLE + g.BOARD_H / 2)),
              size=(g.BOARD_T, g.BOARD_W, g.BOARD_H), color=WHITE, friction=(0.1, 0.1), friction_min=True)
_PIN_LEN = g.PIN_OUT + 0.010                       # 1 cm of it is inside the board
PIN = Shape("pin", kind="capsule", radius=g.PIN_R, height=_PIN_LEN - 2 * g.PIN_R, axis="X",
            pos=_floats((g.BOARD_X - g.PIN_OUT + _PIN_LEN / 2, g.PIN_Y, g.PIN_Z)),
            size=(_PIN_LEN, 2 * g.PIN_R, 2 * g.PIN_R), color=STEEL, friction=(0.3, 0.3), friction_min=True)


def _stand() -> list[Box]:
    top = g.TABLE + g.STAND_BASE_T
    rail_x = g.STAND_SLOT / 2 + g.STAND_RAIL_T / 2
    base_d = g.STAND_SLOT + 2 * g.STAND_RAIL_T
    out = [Box("stand_base", pos=_floats((g.STAND_X, 0.0, g.TABLE + g.STAND_BASE_T / 2)),
               size=(base_d, g.STAND_LEN, g.STAND_BASE_T), color=GREY)]
    for name, sx in (("stand_front", -1), ("stand_back", 1)):
        out.append(Box(name, pos=_floats((g.STAND_X + sx * rail_x, 0.0, top + g.STAND_RAIL_H / 2)),
                       size=(g.STAND_RAIL_T, g.STAND_LEN, g.STAND_RAIL_H), color=GREY))
    return out


STAND = _stand()
OBJECTS = [FRAME, BOARD, PIN, *STAND]

# -- scoring ---------------------------------------------------------------------------------------------------------
LEVEL_DEG = 2.0          # success: the top edge within this of horizontal
CROOKED_DEG = 10.0       # partial credit: hanging within this
CLEAR = 0.02             # m: let go means every part of both arms at least this far from the frame
ARM_R = 0.03             # m: arm links are judged as capsules of this radius between their joints
GRIP_R = 0.035           # m: the gripper as a capsule of this radius from the wrist joint to the fingertips
PAD_TIP = 0.013          # m: fingertips past the grasp point (yam_arena.constants.PAD_TIP_PAST_TCP)
OFF_TABLE = 0.005        # m: hanging means the frame's lowest corner at least this far above the table
LIFT_BOTTOM = 0.02       # m: lifted means the frame's lowest corner this far above the table ...
LIFT_NEAR = 0.03         # m: ... while its rearmost corner is within this of the board's face
DROP_SPEED = 0.6         # m/s: the frame's centre falling faster than this means it was dropped
STILL_SPEED = 0.01       # m/s
STILL_SPIN = math.radians(3.0)   # rad/s
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")


def _to_obb(points, centre, rot, half) -> np.ndarray:
    """Distance of each point (rows) to a box (centre, rotation, half sizes); 0 inside."""
    local = (np.atleast_2d(points) - centre) @ rot
    return np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1)


def capsule_gap(a, b, r, pos, rot) -> float:
    """Distance from a capsule (ends a, b, radius r) to the frame's outer box."""
    seg = np.asarray(a, float) + np.outer(np.linspace(0.0, 1.0, 24), np.asarray(b, float) - np.asarray(a, float))
    return float(_to_obb(seg, np.asarray(pos, float), np.asarray(rot), g.HALF).min() - r)


class Monitor:
    """What the frame has done so far this episode (lifted to the board, dropped) and how it hangs now."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.lifted = self.dropped = False
        self.hanging = self.still = False
        self.tilt = 90.0
        self.why: list[str] = []

    def update(self, pos, quat_xyzw, lin_vel, ang_vel, arm_caps) -> None:
        """One control step: the frame's pose and velocities (station frame) and the arms as capsules (a, b, r)."""
        pos, rot = np.asarray(pos, float), g.rotation(quat_xyzw)
        low = float(g.corners(pos, rot)[:, 2].min())
        if lin_vel[2] < -DROP_SPEED or pos[2] < g.TABLE:
            self.dropped = True
            if "dropped" not in self.why:
                self.why.append("dropped")
        if low >= g.TABLE + LIFT_BOTTOM and g.back_gap(pos, rot) <= LIFT_NEAR:
            self.lifted = True
        clear = all(capsule_gap(a, b, r, pos, rot) >= CLEAR for a, b, r in arm_caps)
        self.tilt = g.tilt_deg(rot)
        self.hanging = bool(clear and low >= g.TABLE + OFF_TABLE and g.on_pin(pos, rot))
        self.still = bool(np.linalg.norm(lin_vel) < STILL_SPEED and np.linalg.norm(ang_vel) < STILL_SPIN)

    def success(self) -> bool:
        return bool(not self.dropped and self.hanging and self.tilt <= LEVEL_DEG and self.still)

    def score(self) -> float:
        if self.dropped:
            return 0.0
        if self.hanging and self.tilt <= CROOKED_DEG:
            return 0.5
        return 0.25 if self.lifted else 0.0


MONITOR = Monitor()


@on_reset
def _reset(environment=None) -> None:
    MONITOR.reset()


def _arm_caps(environment) -> list:
    """Both arms as capsules along their links, and each gripper as a fat capsule out to its fingertips."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    from scenekit import arm_joints, robot_links

    links = robot_links(environment)
    caps = []
    for side in ("left", "right"):
        q = arm_joints(environment, side, ARM_JOINTS)
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        chain = [links[side][n] for n in CHAIN if n in links[side]]
        caps += [(a, b, ARM_R) for a, b in zip(chain[:-1], chain[1:])]
        caps.append((chain[-1], tcp + rot[:, 2] * PAD_TIP, GRIP_R))
    return caps


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    f = object_states(environment, ["frame"])["frame"]
    MONITOR.update(f.pos, f.quat_xyzw, f.lin_vel, f.ang_vel, _arm_caps(environment))


def success(objects) -> bool:
    """Hanging on the pin by its hanger (the pin in a notch), not touching the table, every part of both arms at
    least 2 cm away, its top edge within 2 degrees of horizontal, and still; held 1 s."""
    return MONITOR.success()


def score(objects) -> float:
    """Partial credit (success scores 1.0): 0 once the frame has been dropped (its centre falling faster than 0.6 m/s)
    or fallen below the table top. Otherwise 0.5 if at the end it hangs on the pin by itself within 10 degrees of
    level; else 0.25 if it was at some point lifted to the board (lowest corner 2 cm above the table, rearmost corner
    within 3 cm of the board's face); else 0."""
    return MONITOR.score()


# -- the scripted demo (both arms in step; it knows from the setup which notch hangs level) ---------------------------
GRIP_YAW = {"left": -math.pi / 2, "right": math.pi / 2}   # fingers close along x, on the top rail's front and back


def frame_path(notch: int = g.LEVEL_NOTCH) -> list[tuple[float, np.ndarray, float, str]]:
    """The frame's centre, held level, as (seconds, centre, finger, label) steps; the grippers follow it."""
    s = g.START
    hang = g.hang_centre(notch, g.APPROACH_GAP)
    up = np.array([0.0, 0.0, 0.010])
    return [
        (2.0, s + [0, 0, 0.05], FINGER_OPEN, "above the frame"),
        (1.5, s, FINGER_OPEN, "down to the top rail"),
        (1.0, s, FINGER_CLOSED, "grip"),
        (1.5, s + [0, 0, 0.04], FINGER_CLOSED, "lift out of the stand"),
        (2.5, hang + up - [0.04, 0, 0], FINGER_CLOSED, "carry to the board, hanger above the pin"),
        (1.5, hang + up, FINGER_CLOSED, "back against the board"),
        (1.5, hang - [0, 0, 0.001], FINGER_CLOSED, "lower the notch onto the pin"),
        (1.0, hang - [0, 0, 0.001], FINGER_OPEN, "let go"),
    ]


def grip_point(centre, side: str) -> np.ndarray:
    sign = 1.0 if side == "left" else -1.0
    return np.asarray(centre, float) + [0.0, sign * g.GRIP_Y, g.H / 2 - g.GRIP_BELOW_TOP]


def demo(notch: int = g.LEVEL_NOTCH):
    """Both arms take the top rail 12 cm either side of the centre, lift the frame out of its stand, carry it level to
    the board with the level notch 1 cm above the pin, lower it on, let go, lift the fingers over the rail and back
    away, out past the frame's ends."""
    plans = {}
    for side in ("left", "right"):
        yaw, out = GRIP_YAW[side], (0.10 if side == "left" else -0.10)
        plan = [Waypoint(t, _floats(grip_point(c, side)), f, label, yaw) for t, c, f, label in frame_path(notch)]
        last = np.asarray(plan[-1].pos)
        plan += [
            Waypoint(1.0, _floats(last + [-0.02, 0, 0.035]), FINGER_OPEN, "up over the rail", yaw),
            Waypoint(1.5, _floats(last + [-0.12, out, 0.05]), FINGER_OPEN, "back away, past the frame's end", yaw),
            Waypoint(3.0, _floats(last + [-0.12, out, 0.05]), FINGER_OPEN, "wait", yaw),
        ]
        plans[side] = plan
    return plans
