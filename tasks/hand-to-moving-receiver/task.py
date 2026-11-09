"""On the station table, a bare aluminium cylinder (4 cm across, 7 cm tall) stands 18 cm in front of the arms, between
them, toward the right arm. A blue open receiving cup (10 x 10 cm inside, 2.5 cm deep) is carried by a machine axis
in front of the right arm, like another machine's gripper or a coworker's hand held out for a part: it wanders slowly
and unevenly over an 8 x 18 cm area, at up to 6 cm/s, with brief stops of up to 0.8 s, and never waits for the robot.

The robot must hand the cylinder over into the moving cup: set it down inside, moving along with the cup, and let go
only when its base is no more than 2 cm above the cup's floor. Dropping or tossing it in counts only partly.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Hand the aluminium cylinder into the blue cup. ..." ...
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, offset_path, on_reset, on_step, stop_and_go  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Cylinder  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # score() keeps its own best partial credit (see _STATE), and checks the end state itself

# -- the receiving cup ---------------------------------------------------------------------------------------------
CUP_IN = 0.10  # inner width and depth
WALL, FLOOR, DEPTH, GAP = 0.01, 0.01, 0.025, 0.001  # DEPTH: rim above the floor's top; 1 mm gaps between driven parts
CUP_LOW = TOP + 0.002  # it glides 2 mm above the table: driven, it doesn't rub
FLOOR_TOP = CUP_LOW + FLOOR
RIM = FLOOR_TOP + DEPTH
FLOOR_Z = CUP_LOW + FLOOR / 2
START_XY = (0.30, -0.22)
# Where the floor's centre goes (x out from the bases, y to the robot's left), how long each move takes and how long
# the cup then stops there. Peak speeds 5.3 to 6.0 cm/s (cosine profile: 1.57 x the mean), one loop every 16.8 s.
LEGS = [
    ((0.34, -0.36), 4.0, 0.6),
    ((0.28, -0.30), 2.5, 0.0),
    ((0.36, -0.40), 3.5, 0.8),
    ((0.30, -0.22), 5.0, 0.4),
]
PATH = stop_and_go((*START_XY, FLOOR_Z), [((*xy, FLOOR_Z), travel, pause) for xy, travel, pause in LEGS])
CYCLE_S = PATH.period
BLUE = (0.15, 0.32, 0.8)


def _cup() -> list[DrivenBox]:
    """Floor and four walls, each a driven box following PATH with its own offset (all move as one)."""
    out = CUP_IN / 2 + GAP + WALL / 2
    span = CUP_IN + 2 * (GAP + WALL)
    wall_h = FLOOR + DEPTH
    dz = wall_h / 2 - FLOOR / 2  # a wall's centre above the floor's
    parts = [
        ("cup_floor", (0.0, 0.0, 0.0), (CUP_IN, CUP_IN, FLOOR)),
        ("cup_wall_near", (-out, 0.0, dz), (WALL, span, wall_h)),
        ("cup_wall_far", (out, 0.0, dz), (WALL, span, wall_h)),
        ("cup_wall_right", (0.0, -out, dz), (CUP_IN, WALL, wall_h)),
        ("cup_wall_left", (0.0, out, dz), (CUP_IN, WALL, wall_h)),
    ]
    boxes = []
    for name, off, size in parts:
        path = offset_path(PATH, off)
        pos = tuple(float(v) for v in path(0.0)[0])
        boxes.append(DrivenBox(name, pos=pos, size=size, color=BLUE, path=path))
    return boxes


CUP = _cup()

# -- the part ------------------------------------------------------------------------------------------------------
RADIUS, HEIGHT = 0.02, 0.07
PART_XY = (0.18, -0.10)
PART = Cylinder("part", pos=(*PART_XY, TOP + HEIGHT / 2), radius=RADIUS, height=HEIGHT, mass=0.24,
                color=(0.78, 0.8, 0.82))

OBJECTS = [PART, *CUP]

# -- the handover monitor ------------------------------------------------------------------------------------------
# Every control step the monitor below reads each gripper's TCP (forward kinematics of the measured joints) and the
# gap between its finger pads. A gripper holds the part while its pads are closed on it (gap 3.0 to 4.6 cm: the part
# is 4 cm across) and the part's centre is within 3.5 cm of its TCP. The step it stops holding is a release; the
# monitor then records, from the last step it held, the part's base height above the cup floor and whether its centre
# was over the cup's inside, and the gripper's speed relative to the cup over the last two control intervals.
HOLD_GAP = (0.030, 0.046)
HOLD_REACH = 0.035
GENTLE_SPEED = 0.03  # m/s: the gripper moves with the cup to within this at release
GENTLE_HEIGHT = 0.02  # m: the part's base at most this far above the cup floor at release
LIFT_M = 0.03  # the part's base this far above the table counts as picked up
DEBUG = bool(os.environ.get("HANDOVER_DEBUG"))  # print every held step and release (for checking a sim run)


def _over_cup(part_pos, cup_pos) -> bool:
    """The part's centre is over the inside of the cup (between its walls), wherever the cup is now."""
    dx, dy = part_pos[0] - cup_pos[0], part_pos[1] - cup_pos[1]
    return bool(abs(dx) < CUP_IN / 2 and abs(dy) < CUP_IN / 2)


def holding(tcp, gap, part_pos) -> bool:
    return bool(HOLD_GAP[0] <= gap <= HOLD_GAP[1] and np.linalg.norm(np.asarray(part_pos) - tcp) < HOLD_REACH)


_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.update(holder=None, last=None, prev=None, releases=[], lifted=False)


reset_state()


def update(t: float, part, cup, grips: dict) -> None:
    """One control step of the monitor. ``part`` and ``cup`` (the cup floor) are object states, ``grips`` maps each arm
    to (TCP position, pad gap in m), all measured after the step at episode time ``t``."""
    tcps = {arm: np.asarray(tcp, float) for arm, (tcp, _) in grips.items()}
    rel = {arm: tcp - cup.pos for arm, tcp in tcps.items()}  # each TCP relative to the moving cup
    if part.low[2] >= TOP + LIFT_M:
        _STATE["lifted"] = True
    holder = next((arm for arm, (tcp, gap) in grips.items() if holding(tcps[arm], gap, part.pos)), None)
    prev, last = _STATE["prev"], _STATE["last"]
    was = _STATE["holder"]
    if was is not None and holder != was and last is not None:
        # released this step: speed over (the step before the last held one, the last held one] and over (the last held
        # one, now]; base height and position at the last held step
        speeds = [np.linalg.norm(rel[was] - last["rel"][was]) / max(t - last["t"], 1e-6)]
        if prev is not None:
            speeds.append(np.linalg.norm(last["rel"][was] - prev["rel"][was]) / max(last["t"] - prev["t"], 1e-6))
        height = last["part_low"] - last["floor_top"]
        inside = last["over_cup"]
        speed = float(max(speeds))
        _STATE["releases"].append(dict(t=t, arm=was, speed=speed, height=float(height), over_cup=inside,
                                       gentle=bool(inside and speed < GENTLE_SPEED and height <= GENTLE_HEIGHT)))
        if DEBUG:
            print(f"handover release {_STATE['releases'][-1]}", flush=True)
    if DEBUG and holder is not None:
        r = rel[holder]
        print(f"handover t={t:.2f} {holder} gap={grips[holder][1]:.4f} tcp-cup={r[0]:+.4f},{r[1]:+.4f},{r[2]:+.4f} "
              f"base-floor={part.low[2] - cup.high[2]:+.4f}", flush=True)
    _STATE["prev"] = last
    _STATE["last"] = dict(t=t, rel=rel, part_low=float(part.low[2]), floor_top=float(cup.high[2]),
                          over_cup=_over_cup(part.pos, cup.pos))
    _STATE["holder"] = holder


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints, clock
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.scene import object_states

    st = object_states(environment, ["part", "cup_floor"])
    grips = {}
    for side in ("left", "right"):
        tcp, _ = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        gap = float(np.sum(np.abs(arm_joints(environment, side, FINGER_JOINTS))))
        grips[side] = (tcp, gap)
    update(clock(), st["part"], st["cup_floor"], grips)


def gentle_handover() -> bool:
    """The part's last release was gentle: over the cup's inside, its base 2 cm or less above the floor, the gripper
    moving with the cup to within 3 cm/s."""
    return bool(_STATE["releases"] and _STATE["releases"][-1]["gentle"])


# -- scoring -------------------------------------------------------------------------------------------------------


def in_cup(objects) -> bool:
    """The part rests in the cup, let go: its centre over the inside, its base within 1 cm of the floor, riding with
    the cup (velocity within 1 cm/s of the cup's, turning under 0.5 rad/s), and no gripper holding it."""
    part, cup = objects["part"], objects["cup_floor"]
    riding = np.linalg.norm(part.lin_vel - cup.lin_vel) < 0.01 and np.linalg.norm(part.ang_vel) < 0.5
    return bool(_STATE["holder"] is None and _over_cup(part.pos, cup.pos) and part.low[2] < cup.high[2] + 0.01
                and riding)


def success(objects):
    """In the cup, after a gentle handover."""
    return bool(in_cup(objects) and gentle_handover())


def distance(objects):
    """Metres from the part's centre to where it would stand in the middle of the cup, wherever the cup is."""
    cup = objects["cup_floor"]
    goal = np.array([cup.pos[0], cup.pos[1], cup.high[2] + HEIGHT / 2])
    return float(np.linalg.norm(objects["part"].pos - goal))


def score(objects):
    """Partial credit at the end of the trial (success scores 1.0): 0.5 if the part is in the cup but got there
    otherwise (dropped or tossed in, let go while moving differently from the cup, or over the wall); 0.25 if its base
    was ever 3 cm or more above the table (picked up)."""
    if in_cup(objects):
        return 0.5
    return 0.25 if _STATE["lifted"] else 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
# What it guards against in real physics:
# - lag: the arm's joint PD (no velocity feed-forward) trails a moving target by about damping / stiffness = 5 / 80 s
#   (60 ms) at the shoulder, so every follow waypoint aims at where the cup will be FOLLOW_LEAD later;
# - a spring-back at release: the arm's gravity compensation doesn't include the 240 g part, so holding it the TCP sags
#   about 5 mm (shoulder stiffness 80 Nm/rad), and springs back up within a few control steps when the part is let go
#   in mid-air (about 10 cm/s). The demo sets the part down on the cup floor first (commanded as if there were no sag,
#   so the floor, not the arm, carries the part), then opens: the arm has nothing left to spring back from;
# - a sudden let-go: the fingers open slowly (the grip force fades from 50 N to 0 over about 0.3 s, the pads have let go
#   0.15 s later), while the cup moves near its peak speed with little change of speed;
# - a grip lower or higher on the part than planned: set down on the floor, its base is at most a few mm up, far
#   below the 2 cm limit either way.
GRIP_BELOW_TOP = 0.016  # TCP this far below the part's top: the pad tips 2.9 cm below it, 4.1 cm above its base
GRASP_Z = TOP + HEIGHT - GRIP_BELOW_TOP
TRAVEL_Z = TOP + 0.16  # the held part's base 10.6 cm above the table, 6.9 cm above the rim
RELEASE_GAP = 0.0  # the held part's base on the cup floor when it is let go (as commanded, without the sag)
RELEASE_Z = FLOOR_TOP + RELEASE_GAP + HEIGHT - GRIP_BELOW_TOP
FOLLOW_STEP = 0.05  # s, one control step at 20 Hz: the TCP is over the cup at every step, so it moves with it
FOLLOW_LEAD = 0.05  # s: each follow waypoint aims at the cup this much later, to make up for the arm's tracking lag
MEET_S = 6.0  # over the cup at travel height (the cup is crossing from (0.34, -0.36) to (0.28, -0.30))
LOWER_S = (6.5, 7.8)  # lowered while following, the part's base onto the cup floor
OPEN_S = 8.5  # the fingers start to open, as the cup crosses toward (0.36, -0.40) at 5.6 cm/s (peak 5.7 at 8.85 s)
# The finger target (m, 0 closed .. FINGER_OPEN) from OPEN_S on, as (s after OPEN_S, target): the pads press the 4 cm
# part with the gripper's 50 N cap until the target is 1 cm per finger past its surface (0.01 of the 0.02 m they are
# open on it), then with 5000 N/m x the remaining overlap: 15 N at 0.3 s (still a firm grip), 0 at 0.45 s; the pads
# are 4.6 cm apart (the monitor's release) at about 0.6 s, and fully open at 1.0 s
FINGER_RAMP = [(0.0, FINGER_CLOSED), (0.3, -0.017), (0.8, -0.027), (1.0, FINGER_OPEN)]
RELEASED_S = OPEN_S + 0.6  # about when the monitor sees the pads 4.6 cm apart
RISE_S = (9.8, 10.4)  # up again, still following (the cup moves on until 10.6 s)
LEAVE_S = 10.4
PARK = (0.16, -0.30, TOP + 0.20)


def _over_cup_at(t: float, z: float) -> tuple[float, float, float]:
    x, y, _ = PATH(t + FOLLOW_LEAD)[0]
    return (float(x), float(y), float(z))


def _ease(a: float, b: float, u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return float(a + (b - a) * (1 - np.cos(np.pi * u)) / 2)


def follow_z(t: float) -> float:
    """The TCP height while following the cup."""
    if t < RISE_S[0]:
        return _ease(TRAVEL_Z, RELEASE_Z, (t - LOWER_S[0]) / (LOWER_S[1] - LOWER_S[0]))
    return _ease(RELEASE_Z, TRAVEL_Z, (t - RISE_S[0]) / (RISE_S[1] - RISE_S[0]))


def finger_at(t: float) -> float:
    """The finger target while following the cup: closed until OPEN_S, then opened slowly along FINGER_RAMP."""
    u = t - OPEN_S
    if u <= 0.0:
        return float(FINGER_CLOSED)
    for (u0, f0), (u1, f1) in zip(FINGER_RAMP, FINGER_RAMP[1:]):
        if u <= u1:
            return float(f0 + (f1 - f0) * (u - u0) / (u1 - u0))
    return float(FINGER_OPEN)


def demo():
    """Right arm: pick the cylinder from the top (lifted by 3.9 s), carry it over the cup by 6.0 s, then put the TCP
    over the cup at every control step: set the part down on the cup floor (6.5 to 7.8 s), ride along, open the
    fingers slowly from 8.5 s while the cup moves at 5.6 cm/s (let go about 9.1 s), ride along open until 9.8 s, rise
    while following, and park."""
    px, py = PART_XY
    plan = [
        Waypoint(1.5, (px, py, TRAVEL_Z), FINGER_OPEN, "above the part"),
        Waypoint(1.0, (px, py, GRASP_Z), FINGER_OPEN, "descend"),
        Waypoint(0.6, (px, py, GRASP_Z), FINGER_CLOSED, "close"),
        Waypoint(0.8, (px, py, TRAVEL_Z), FINGER_CLOSED, "lift"),
    ]
    plan.append(Waypoint(MEET_S - sum(w.duration for w in plan), _over_cup_at(MEET_S, TRAVEL_Z), FINGER_CLOSED,
                         "carry"))
    n = round((LEAVE_S - MEET_S) / FOLLOW_STEP)
    for k in range(1, n + 1):
        t = MEET_S + k * FOLLOW_STEP
        label = "follow" if t < LOWER_S[0] else "lower" if t <= LOWER_S[1] else "ride" if t <= OPEN_S + 1e-9 \
            else "release" if t < RELEASED_S + 0.2 else "ride open" if t < RISE_S[0] else "rise"
        plan.append(Waypoint(FOLLOW_STEP, _over_cup_at(t, follow_z(t)), finger_at(t), label))
    plan.append(Waypoint(1.0, PARK, FINGER_OPEN, "park"))
    return {"right": plan}
