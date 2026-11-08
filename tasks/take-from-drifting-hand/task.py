"""A person across the table holds out a part for the robot to take: a yellow tool block (10 x 2.5 x 5.5 cm) lies
on their flat palm, between two low stops that keep it from sliding off their fingers or wrist. While they wait,
their hand drifts, as a held-out hand does: small loops that keep shifting, a few cm/s, never pausing and never
repeating, over an area about 7 x 10 cm and 1 cm up and down. A grey tray stands on the table at the right.

The robot must take the block from the hand, lifting it straight up out of the palm, without bumping the hand,
and put it in the tray.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The person is holding out the yellow block for you. ..." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, drift, offset_path, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # score() reads the monitor's flags (kept over the whole trial) and the end state

# -- the hand -------------------------------------------------------------------------------------------------------
# The hand's reference point is the middle of the palm's top face; every part of the hand keeps a fixed offset from
# it. The palm is a flat plate held out level, fingers toward the robot (-x), the forearm running away from it (+x).
PALM = (0.13, 0.09, 0.015)  # x (wrist to fingertips), y (across), thickness
STOP = (0.01, 0.05, 0.01)  # the two stops at the block's ends, standing on the palm
FOREARM = (0.22, 0.06, 0.03)
GAP = 0.001  # between the driven parts of the hand, so they never touch each other
BLOCK = (0.10, 0.025, 0.055)  # the tool block: long along x, gripped across its 2.5 cm width
BLOCK_MASS = 0.14
END_CLEAR = 0.002  # each stop stands 2 mm clear of the block's end
STOP_X = BLOCK[0] / 2 + END_CLEAR + STOP[0] / 2  # 5.7 cm either side of the palm's middle
SEAT_DZ = BLOCK[2] / 2  # the seated block's centre above the palm's top
SKIN = (0.87, 0.70, 0.58)
SLEEVE = (0.30, 0.36, 0.46)
YELLOW = (0.95, 0.75, 0.10)

# -- the drift --------------------------------------------------------------------------------------------------------
# A small loop (3.8 s) whose centre wanders on slower, unrelated periods, plus a bob. The periods (3.4, 3.8, 6.7, 6.8,
# 14.6 and 16.6 s) share no common multiple shorter than years, so the hand never retraces its path (within a trial,
# no delay brings a 40 s stretch of it back to within 2 cm); the loop keeps it moving at 2 cm/s or more, so it
# never pauses. (axis, amplitude m, period s, kind): see motionkit.drift.
HOME = (0.30, -0.17, TOP + 0.08)  # where the palm's top middle starts: 30 cm out, 13.5 cm inside the right arm
DRIFT_TERMS = (
    (0, 0.018, 3.8, "sin"), (1, 0.025, 3.8, "1-cos"),  # the loop: 3.6 cm deep, 5 cm across
    (0, 0.015, 14.6, "sin"),  # slow wander toward and away from the robot
    (1, 0.020, 16.6, "sin"), (1, 0.008, 6.7, "sin"),  # slow wander across
    (2, 0.003, 6.8, "sin"), (2, 0.002, 3.4, "sin"),  # bobbing up and down, 0.5 cm either way at most
)
RAMP_S = 2.0  # the hand starts still and reaches full pace over the first 2 s
PATH = drift(HOME, DRIFT_TERMS, ramp_s=RAMP_S)


def palm_top(t: float) -> np.ndarray:
    """The middle of the palm's top face at episode time ``t``."""
    return PATH(t)[0]


def seat(t: float) -> np.ndarray:
    """Where the seated block's centre is at episode time ``t``."""
    return palm_top(t) + np.array([0.0, 0.0, SEAT_DZ])


# Each part of the hand: (name, centre relative to the palm's top middle, size)
HAND_PARTS = (
    ("palm", (0.0, 0.0, -PALM[2] / 2), PALM),
    ("stop_fingers", (-STOP_X, 0.0, GAP + STOP[2] / 2), STOP),
    ("stop_wrist", (STOP_X, 0.0, GAP + STOP[2] / 2), STOP),
    ("forearm", (PALM[0] / 2 + GAP + FOREARM[0] / 2, 0.0, -0.005 - FOREARM[2] / 2), FOREARM),
)
HAND = [
    DrivenBox(f"hand_{name}", pos=tuple(float(v) for v in offset_path(PATH, off)(0.0)[0]),
              size=tuple(float(v) for v in size), color=SLEEVE if name == "forearm" else SKIN,
              path=offset_path(PATH, off))
    for name, off, size in HAND_PARTS
]
TOOL = Box("tool_block", pos=tuple(float(v) for v in seat(0.0)), size=BLOCK, mass=BLOCK_MASS, color=YELLOW)

# -- the tray -------------------------------------------------------------------------------------------------------
TRAY_XY = (0.20, -0.40)
TRAY_IN = (0.16, 0.14)  # inner length (x) and width (y)
T_WALL, T_FLOOR, T_WALL_H = 0.01, 0.01, 0.05
TRAY_FLOOR_TOP = TOP + T_FLOOR
GREY = (0.45, 0.47, 0.50)


def _tray() -> list[Box]:
    x, y = TRAY_XY
    ix, iy = TRAY_IN
    ox, oy = ix / 2 + T_WALL / 2, iy / 2 + T_WALL / 2
    z = TOP + T_WALL_H / 2
    return [
        Box("tray_floor", pos=(x, y, TOP + T_FLOOR / 2), size=(ix + 2 * T_WALL, iy + 2 * T_WALL, T_FLOOR), color=GREY),
        Box("tray_wall_near", pos=(x - ox, y, z), size=(T_WALL, iy + 2 * T_WALL, T_WALL_H), color=GREY),
        Box("tray_wall_far", pos=(x + ox, y, z), size=(T_WALL, iy + 2 * T_WALL, T_WALL_H), color=GREY),
        Box("tray_wall_right", pos=(x, y - oy, z), size=(ix, T_WALL, T_WALL_H), color=GREY),
        Box("tray_wall_left", pos=(x, y + oy, z), size=(ix, T_WALL, T_WALL_H), color=GREY),
    ]


OBJECTS = [*HAND, TOOL, *_tray()]

# -- the monitor ----------------------------------------------------------------------------------------------------
# Every control step the monitor reads the palm's measured position (so the hand is wherever it has drifted to) and
# checks two things.
#
# How the block leaves the palm. It is in its seat while its centre is within 1.5 cm sideways of where it sits
# seated and within 1.5 cm above or below that height. The first time it is out, the monitor records whether it was
# taken *up* (more than 1.5 cm above its seat, still within 1.5 cm sideways) or knocked *off* (pushed, dragged or
# tipped more than 1.5 cm sideways while low). Once taken up, it counts as *taken* when its centre has been 3 cm or
# more above its seat.
#
# Whether the robot bumps the hand. The hand is its four boxes (palm, stops, forearm); the robot is a conservative
# set of capsules (segment and radius): each arm link between consecutive link origins (radius 4 cm); the gripper
# housing from its link origin to where the fingers start (radius 5 cm); the wrist camera (a 2.5 cm ball); and each
# finger, from where it starts to its pad tip, at its measured opening (radius 8 mm). The robot is *at the hand*
# when any capsule comes within 1 cm of any hand box. The capsules are larger than the parts, so this errs toward
# flagging: a real gripper flagged here passes within a few cm of the hand, never closer than 1 cm.
OUT_M = 0.015
TAKEN_M = 0.03
CLEAR = 0.01
ARM_R = 0.04
HOUSING_R = 0.05
CAM_R = 0.025
FINGER_R = 0.008
FINGER_ROOT = 0.077  # m above the TCP, along the approach axis: where the fingers leave the housing
WRIST_CAM = (0.0, -0.08, 0.0)  # the wrist camera in the gripper link frame
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")


def hand_boxes(palm_pos) -> list[tuple[np.ndarray, np.ndarray]]:
    """The hand's boxes (low and high corners), the palm's top middle at ``palm_pos``. ``palm_pos`` is the measured
    centre of the palm box, as the monitor reads it."""
    ref = np.asarray(palm_pos, float) + np.array([0.0, 0.0, PALM[2] / 2])
    out = []
    for _, off, size in HAND_PARTS:
        c, half = ref + np.asarray(off), np.asarray(size) / 2
        out.append((c - half, c + half))
    return out


def gripper_capsules(grip_origin, tcp, rot, finger) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """One gripper as capsules (end, end, radius): ``grip_origin`` its link origin, ``tcp`` and ``rot`` its TCP pose
    (rotation columns: closing axis, -, approach axis) and ``finger`` its finger joint (0 closed .. -0.047 open)."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    closing, approach = rot[:, 0], rot[:, 2]
    root = tcp - FINGER_ROOT * approach
    tip = tcp + PAD_TIP_PAST_TCP * approach
    half = abs(float(finger)) + FINGER_R / 2  # each finger's middle line: the pad face plus half a finger
    cam = np.asarray(grip_origin, float) + rot @ np.asarray(WRIST_CAM)
    caps = [(np.asarray(grip_origin, float), root, HOUSING_R), (cam, cam, CAM_R)]
    for s in (-1.0, 1.0):
        caps.append((root + s * half * closing, tip + s * half * closing, FINGER_R))
    return caps


def arm_capsules(chain) -> list[tuple[np.ndarray, np.ndarray, float]]:
    pts = [np.asarray(p, float) for p in chain]
    return [(a, b, ARM_R) for a, b in zip(pts[:-1], pts[1:])]


def _to_box(points, lo, hi) -> np.ndarray:
    p = np.atleast_2d(points)
    return np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)


def gap_to_hand(palm_pos, capsules) -> float:
    """The smallest distance (m) from any capsule's surface to any hand box; negative inside."""
    best = np.inf
    for a, b, r in capsules:
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.004)) + 1)  # points every 4 mm along the segment
        seg = a + np.outer(np.linspace(0.0, 1.0, n), b - a)
        for lo, hi in hand_boxes(palm_pos):
            best = min(best, float(_to_box(seg, lo, hi).min()) - r)
    return best


def offset_from_seat(block, palm) -> tuple[float, float]:
    """(sideways distance from the seat, height above it), in m, with the palm box's measured state ``palm``."""
    seat_now = np.asarray(palm.pos, float) + np.array([0.0, 0.0, PALM[2] / 2 + SEAT_DZ])
    d = np.asarray(block.pos, float) - seat_now
    return float(np.hypot(d[0], d[1])), float(d[2])


_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.update(left=None, taken=False, bumped=False, closest=np.inf)


reset_state()


def update(block, palm, capsules) -> None:
    """One control step of the monitor, over the measured block and palm and the robot's capsules."""
    side, up = offset_from_seat(block, palm)
    if _STATE["left"] is None and (side > OUT_M or abs(up) > OUT_M):
        _STATE["left"] = "up" if (side <= OUT_M and up > OUT_M) else "off"
    if _STATE["left"] == "up" and up >= TAKEN_M:
        _STATE["taken"] = True
    gap = gap_to_hand(palm.pos, capsules)
    _STATE["closest"] = min(_STATE["closest"], gap)
    if gap < CLEAR:
        _STATE["bumped"] = True


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints, robot_links
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.scene import object_states

    st = object_states(environment, ["tool_block", "hand_palm"])
    links = robot_links(environment)
    caps = []
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        finger = float(arm_joints(environment, side, ["joint7"])[0])
        caps += arm_capsules([links[side][n] for n in CHAIN])
        caps += gripper_capsules(links[side]["gripper"], tcp, rot, finger)
    update(st["tool_block"], st["hand_palm"], caps)


# -- scoring --------------------------------------------------------------------------------------------------------


def in_tray(block) -> bool:
    """Resting in the tray: centre between the walls, underside within 1 cm of the floor, still."""
    dx, dy = block.pos[0] - TRAY_XY[0], block.pos[1] - TRAY_XY[1]
    inside = abs(dx) < TRAY_IN[0] / 2 and abs(dy) < TRAY_IN[1] / 2
    return bool(inside and block.low[2] < TRAY_FLOOR_TOP + 0.01 and block.resting())


def success(objects):
    """The block was taken straight up out of the palm, the robot never came within 1 cm of the hand, and the block
    rests in the tray."""
    return bool(_STATE["left"] == "up" and not _STATE["bumped"] and in_tray(objects["tool_block"]))


def distance(objects):
    """Metres from the block's centre to where it would rest in the middle of the tray."""
    goal = np.array([*TRAY_XY, TRAY_FLOOR_TOP + BLOCK[2] / 2])
    return float(np.linalg.norm(objects["tool_block"].pos - goal))


def score(objects):
    """Partial credit at the end of the trial (success scores 1.0), and 0 unless the block left the palm upward:
    0.5 if it rests in the tray but the robot came within 1 cm of the hand at some point; 0.25 if it was taken
    (3 cm above its seat) but doesn't end in the tray."""
    if _STATE["left"] != "up":
        return 0.0
    if in_tray(objects["tool_block"]):
        return 0.5
    return 0.25 if _STATE["taken"] else 0.0


# -- the scripted demo ----------------------------------------------------------------------------------------------
GRIP_BELOW_TOP = 0.005  # TCP this far below the block's top: the pad tips 3.7 cm above the palm, pads on 1.8 cm
GRASP_DZ = BLOCK[2] - GRIP_BELOW_TOP  # TCP above the palm's top while gripping
HOVER_DZ = BLOCK[2] + PAD_TIP_PAST_TCP + 0.025  # open pad tips 2.5 cm above the block's top
LIFT_DZ = GRASP_DZ + 0.045  # the held block lifted 4.5 cm out of the palm
FOLLOW_STEP = 0.05  # s, one control step at 20 Hz: a waypoint on the drift every step
LEAD_S = 0.05  # aim one control step ahead, where the hand will be when the arm gets there
MEET_S = 2.5  # over the block at hover height
DESCEND_S = (3.0, 4.2)
CLOSE_S = (4.4, 5.0)
LIFT_S = (5.2, 6.0)
LEAVE_S = 6.0
TRAVEL_Z = TOP + 0.18  # TCP while carrying: the block's underside 13 cm above the table
PLACE_Z = TRAY_FLOOR_TOP + 0.01 + GRASP_DZ  # the block's underside 1 cm above the tray floor
PARK = (0.20, -0.34, TOP + 0.18)


def _ease(a: float, b: float, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return a + (b - a) * (1 - np.cos(np.pi * u)) / 2


def follow_dz(t: float) -> float:
    """The TCP's height above the palm's top while following the hand."""
    if t < LIFT_S[0]:
        return _ease(HOVER_DZ, GRASP_DZ, (t - DESCEND_S[0]) / (DESCEND_S[1] - DESCEND_S[0]))
    return _ease(GRASP_DZ, LIFT_DZ, (t - LIFT_S[0]) / (LIFT_S[1] - LIFT_S[0]))


def follow_finger(t: float) -> float:
    return _ease(FINGER_OPEN, FINGER_CLOSED, (t - CLOSE_S[0]) / (CLOSE_S[1] - CLOSE_S[0]))


def _over_block(t: float, dz: float) -> tuple[float, float, float]:
    x, y, z = palm_top(t + LEAD_S)
    return (float(x), float(y), float(z + dz))


def demo():
    """Right arm: over the block by 2.5 s, then a waypoint on the hand's drift every control step: descend around
    the block (3.0 to 4.2 s), close (4.4 to 5.0 s), lift it 4.5 cm straight up out of the palm (5.2 to 6.0 s), all
    while following. Then carry it to the tray, lower it to 1 cm above the floor, let go and park."""
    tx, ty = TRAY_XY
    plan = [Waypoint(MEET_S, _over_block(MEET_S, HOVER_DZ), FINGER_OPEN, "over the block")]
    n = round((LEAVE_S - MEET_S) / FOLLOW_STEP)
    for k in range(1, n + 1):
        t = MEET_S + k * FOLLOW_STEP
        label = "follow" if t < DESCEND_S[0] else "descend" if t <= DESCEND_S[1] + 1e-9 else "close" \
            if t <= CLOSE_S[1] + 1e-9 else "hold" if t < LIFT_S[0] - 1e-9 else "lift"
        plan.append(Waypoint(FOLLOW_STEP, _over_block(t, follow_dz(t)), float(follow_finger(t)), label))
    x, y, _ = plan[-1].pos
    plan += [
        Waypoint(0.6, (x - 0.04, y, TRAVEL_Z), FINGER_CLOSED, "up and away from the hand"),
        Waypoint(1.6, (tx, ty, TRAVEL_Z), FINGER_CLOSED, "carry"),
        Waypoint(1.2, (tx, ty, PLACE_Z), FINGER_CLOSED, "lower"),
        Waypoint(0.6, (tx, ty, PLACE_Z), FINGER_OPEN, "release"),
        Waypoint(1.0, PARK, FINGER_OPEN, "park"),
    ]
    return {"right": plan}
