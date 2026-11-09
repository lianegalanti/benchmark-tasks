"""On the station table, a powered roller conveyor (26 cm wide, 74 cm long) runs across the table in front of the arms,
from 20 to 46 cm in front of them, carrying parcels from the robot's left to its right. Its rollers are skewed 15
degrees, so they drive whatever rides on them diagonally: 5.8 cm/s toward the right and 1.6 cm/s toward the arms. The
conveyor's far side has a low rail; its near side, toward the arms, is open: a 4 cm drop to the table. Two white lines
run along the rollers 14 cm apart, marking the centre lane, 33 cm in front of the arms. Just right of the midpoint
between the arms the conveyor passes through a dark scanner arch whose opening is 20 cm wide, centred on the lane.

A brown cardboard parcel (10 x 10 x 6 cm, 150 g) rides on the conveyor in front of the left arm, 4 cm nearer the arms
than the lane's centre. Untouched, it drifts toward the open near side and falls off it about 6 s after the start,
10 cm before it reaches the arch.

The robot must get the parcel between the two white lines, so that it rides into the arch centred in the lane, without
letting it fall off. The rollers keep driving it toward the open side after it is let go. The parcel is wider than the
open gripper; it is meant to be pushed.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The parcel on the conveyor is drifting toward the open edge. Get it between the two white lines before it reaches the arch, without letting it fall off." ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import motionkit  # noqa: E402
import numpy as np  # noqa: E402
from motionkit import ConveyorBed, LaunchedBox, Marker, ramp_distance, ramp_speed  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end", with the outcome kept by the MONITOR below: the parcel's lateral offset when it crosses into the arch, or 0
# once it has fallen off. What happens after the crossing doesn't change the score.
SCORE_OVER = "end"


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the conveyor --------------------------------------------------------------------------------------------------
BED_X = (0.20, 0.46)  # near (open) and far edges, metres in front of the arm bases
BED_Y = (-0.20, 0.54)  # right and left ends (y is to the robot's left); parcels travel toward -y
BED_H = 0.04
BED_GAP = 0.002  # the bed rides 2 mm above the table: a moving rigid body, it must not rub on it
BED_TOP = TOP + BED_GAP + BED_H
LANE_X = 0.33  # the centre lane's centre line
LINE_HALF = 0.07  # each white line's centre, either side of it: a 10 cm parcel between them is within 2 cm of centre
SKEW = math.radians(15.0)  # the rollers' skew: their drive direction is 15 degrees off the conveyor's length
SPEED = 0.06  # m/s along the drive direction (the rollers' surface speed)
RAMP_S = 0.5  # the conveyor starts with the episode, reaching full speed in 0.5 s
DRIVE = np.array([-math.sin(SKEW), -math.cos(SKEW), 0.0])  # toward the right (-y) and toward the arms (-x)
SPEED_FN = ramp_speed(SPEED, RAMP_S)
FRICTION_BED = (0.8, 0.7)  # rubber-lagged rollers

BED = ConveyorBed("conveyor", pos=_floats((sum(BED_X) / 2, sum(BED_Y) / 2, TOP + BED_GAP + BED_H / 2)),
                  size=_floats((BED_X[1] - BED_X[0], BED_Y[1] - BED_Y[0], BED_H)), color=(0.16, 0.16, 0.18),
                  speed=SPEED_FN, heading=float(math.atan2(DRIVE[1], DRIVE[0])), friction=FRICTION_BED)

# The rollers, drawn as visual-only stripes on the bed (the bed's surface does the driving): each roller's axis is
# square to the drive direction, so it is turned SKEW from the across-the-conveyor (x) direction (clockwise, seen
# from above).
ROLLER_W, ROLLER_PITCH, ROLLER_LEN = 0.025, 0.035, 0.24
ROLLER_Y = tuple(float(y) for y in np.arange(BED_Y[0] + 0.05, BED_Y[1] - 0.045, ROLLER_PITCH))
PAINT = 0.0008  # the stripes' and lines' thickness (no colliders)
ROLLERS = [
    Marker(f"roller_{k:02d}", pos=_floats((LANE_X, y, BED_TOP)), size=(ROLLER_LEN, ROLLER_W, PAINT),
           color=(0.55, 0.56, 0.58), yaw=float(-SKEW))
    for k, y in enumerate(ROLLER_Y)
]

# -- the scanner arch: the gate line is its upstream face ----------------------------------------------------------
GATE_Y = 0.0  # the arch's upstream face; a parcel "crosses" when its centre passes it
ARCH_Y = (-0.15, GATE_Y)
OPENING = 0.20  # between its two inner guides, centred on the lane: fits a 10 cm parcel up to 5 cm off centre
GUIDE_T = 0.02
ARCH_TOP = BED_TOP + 0.14
ROOF_T = 0.01
HANG = 0.003  # the guides and the end stop hang 3 mm above the bed
LEG_X = {"near": (0.17, 0.19), "far": (0.48, 0.50)}
DARK = (0.12, 0.13, 0.17)


def _box(name, x, y, z, color=DARK) -> Box:
    """A fixed box spanning the given (lo, hi) ranges."""
    return Box(name, pos=_floats((sum(x) / 2, sum(y) / 2, sum(z) / 2)),
               size=_floats((x[1] - x[0], y[1] - y[0], z[1] - z[0])), color=color)


def _arch() -> list[Box]:
    lo, hi = LANE_X - OPENING / 2, LANE_X + OPENING / 2
    hang = (BED_TOP + HANG, ARCH_TOP - ROOF_T)
    return [
        _box("arch_leg_near", LEG_X["near"], ARCH_Y, (TOP, ARCH_TOP - ROOF_T)),
        _box("arch_leg_far", LEG_X["far"], ARCH_Y, (TOP, ARCH_TOP - ROOF_T)),
        _box("arch_roof", (LEG_X["near"][0], LEG_X["far"][1]), ARCH_Y, (ARCH_TOP - ROOF_T, ARCH_TOP)),
        _box("arch_guide_near", (lo - GUIDE_T, lo), ARCH_Y, hang),
        _box("arch_guide_far", (hi, hi + GUIDE_T), ARCH_Y, hang),
        _box("arch_end_stop", (lo, hi), (ARCH_Y[0], ARCH_Y[0] + 0.015), hang),
    ]


# the far rail: stands on the table beside the conveyor's far edge, 3 cm above the rollers, up to the arch
RAIL_X = (BED_X[1] + 0.003, BED_X[1] + 0.013)
RAIL = _box("far_rail", RAIL_X, (GATE_Y, BED_Y[1]), (TOP, BED_TOP + 0.03), color=(0.35, 0.36, 0.38))

LINES = [
    Marker(f"lane_line_{side}", pos=_floats((LANE_X + sign * LINE_HALF, (GATE_Y + BED_Y[1]) / 2, BED_TOP + PAINT / 2)),
           size=_floats((0.006, BED_Y[1] - GATE_Y, PAINT)), color=(0.97, 0.97, 0.97))
    for side, sign in (("near", -1), ("far", 1))
]

# -- the parcel ----------------------------------------------------------------------------------------------------
PARCEL = (0.10, 0.10, 0.06)
START = (0.29, 0.44, BED_TOP + PARCEL[2] / 2 + 0.0005)  # 4 cm nearer the open side than the lane's centre
# A free box with cardboard's friction, at rest at the start (the conveyor starts under it).
PARCEL_SPEC = LaunchedBox("parcel", pos=_floats(START), size=PARCEL, mass=0.15, color=(0.62, 0.45, 0.28),
                          velocity=(0.0, 0.0, 0.0), friction=(0.5, 0.4))

OBJECTS = [BED, *ROLLERS, *LINES, RAIL, *_arch(), PARCEL_SPEC]


def ride(t: float, start=START) -> np.ndarray:
    """Where the parcel's centre is at episode time ``t``, riding untouched from ``start`` (it moves with the rollers'
    surface: friction can give it 0.5 x 9.81 m/s², far more than the 0.19 m/s² the start-up asks)."""
    return np.asarray(start, float) + ramp_distance(SPEED, RAMP_S, t) * DRIVE


def time_to(y: float, start=START) -> float:
    """When the untouched parcel's centre reaches ``y`` along the conveyor."""
    run = (start[1] - y) / -DRIVE[1]  # metres along the drive direction
    return run / SPEED + RAMP_S / 2


# -- scoring -------------------------------------------------------------------------------------------------------
ON_LANE = 0.02  # within 2 cm of the lane's centre: wholly between the white lines
NEAR_LANE = 0.05  # within 5 cm
MAX_TILT_DEG = 15.0
FELL_DROP = 0.015  # its underside this far below the rollers' top: it has gone over an edge


def _over_bed(p) -> bool:
    return bool(BED_X[0] < p.pos[0] < BED_X[1] and BED_Y[0] < p.pos[1] < BED_Y[1])


def _upright(p) -> bool:
    x, y, _, _ = p.quat_xyzw
    return bool(1 - 2 * (x * x + y * y) > math.cos(math.radians(MAX_TILT_DEG)))


def _riding(p) -> bool:
    """Standing upright on the rollers: centre over the bed, underside within 1 cm of its top."""
    return bool(_over_bed(p) and abs(p.low[2] - BED_TOP) < 0.01 and _upright(p))


def _fallen(p) -> bool:
    """Off the conveyor: its centre past an edge, or its underside below the rollers' top."""
    return bool(not _over_bed(p) or p.low[2] < BED_TOP - FELL_DROP)


def crossing_score(offset: float) -> float:
    """The score for crossing into the arch ``offset`` metres off the lane's centre."""
    if abs(offset) <= ON_LANE:
        return 1.0
    if abs(offset) <= NEAR_LANE:
        return 0.5
    return 0.25


class Monitor:
    """The episode so far: whether the parcel has fallen off, and its offset when it crossed into the arch."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.fallen, self.past, self.offset, self.t = False, False, None, None

    def update(self, p, t: float) -> None:
        """Take one control step's parcel state at episode time ``t`` (once per step; repeats are ignored)."""
        if t == self.t:
            return
        self.t = t
        if self.fallen or self.offset is not None:
            return  # decided
        if _fallen(p):
            self.fallen = True
        elif p.pos[1] <= GATE_Y and not self.past:
            self.past = True
            if _riding(p):  # it rode in; carried past (lifted over or through the arch) earns no crossing
                self.offset = float(p.pos[0] - LANE_X)

    def score(self, p) -> float:
        if self.offset is not None:
            return crossing_score(self.offset)
        if self.fallen:
            return 0.0
        return 0.25 if _riding(p) else 0.0


MONITOR = Monitor()
motionkit.on_reset(MONITOR.reset)


def success(objects):
    """It rode into the arch upright on the rollers, its centre within 2 cm of the lane's centre."""
    MONITOR.update(objects["parcel"], motionkit.clock())
    return bool(MONITOR.offset is not None and abs(MONITOR.offset) <= ON_LANE)


def distance(objects):
    """Metres from the parcel's centre to the lane's centre line (across the conveyor)."""
    return float(abs(objects["parcel"].pos[0] - LANE_X))


def score(objects):
    """Decided when the parcel's centre crosses the arch's upstream face, upright on the rollers: 1.0 if it is then
    within 2 cm of the lane's centre, 0.5 within 5 cm, 0.25 farther (no crossing credit if it was not riding upright
    on the rollers as its centre passed the face). 0 if it fell off the conveyor before that.
    Otherwise, at the end of the trial: 0.25 if it stands upright on the rollers, 0 if not (tipped, or held up)."""
    p = objects["parcel"]
    MONITOR.update(p, motionkit.clock())
    return MONITOR.score(p)


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Left arm, closed fingers turned a quarter turn (they close along x, like a flat paddle facing across the conveyor),
# open loop against the parcel's ride: come down beside its near face, push it out to the lane, then ride along
# beside it as a moving side rail (the rollers press it against the fingers), and let go just before the arch,
# leaving it a little beyond the lane's centre so that the drift over the last few centimetres brings it to the centre.
DT = 0.05  # one control step at 20 Hz
YAW = -math.pi / 2  # (the other way round runs joint 6 into its stop)
FINGER_HALF = 0.018  # from the closed fingers' centre line to where they touch a box's face (measured in Isaac Sim)
GAP = 0.02  # between the fingers and the parcel's near face as they come down
TIP_Z = BED_TOP + PAD_TIP_PAST_TCP + 0.015  # TCP height with the pad tips 1.5 cm above the rollers
CLEAR_Z = BED_TOP + PARCEL[2] + PAD_TIP_PAST_TCP + 0.03  # pad tips 3 cm above the parcel's top
OVER_AT, DOWN_S, PUSH_S = 1.8, 0.8, 1.6
RELEASE_Y = GATE_Y + 0.09  # where the parcel's centre is when the fingers back off
BACK_S, UP_S = 0.3, 0.6
DRIFT_X = -DRIVE[0] / -DRIVE[1] * (RELEASE_Y - GATE_Y)  # how far it drifts toward the arms after it is let go
ESCORT_X = LANE_X + DRIFT_X - PARCEL[0] / 2 - FINGER_HALF  # the fingers' line while they ride beside it


def _near_face_tcp(t: float) -> float:
    """The TCP's x that leaves GAP between the fingers and the untouched parcel's near face at time ``t``."""
    return float(ride(t)[0] - PARCEL[0] / 2 - FINGER_HALF - GAP)


def demo():
    """Left arm: over the parcel's near side by 1.8 s, down beside it (riding along with it) by 2.6 s, push it out to
    the lane by 4.2 s, ride beside it until its centre is 9 cm from the arch (about 6.3 s), then back off and lift.
    It drifts 2.4 cm toward the arms on its own before it crosses into the arch, ending on the lane's centre."""
    plan = [Waypoint(OVER_AT, _floats((_near_face_tcp(OVER_AT), ride(OVER_AT)[1], CLEAR_Z)), FINGER_CLOSED,
                     "over its near side", YAW)]
    t = OVER_AT
    n = round(DOWN_S / DT)
    for k in range(1, n + 1):  # follow it down, one control step at a time
        s = k / n
        s = s * s * (3 - 2 * s)
        tk = t + k * DT
        plan.append(Waypoint(DT, _floats((_near_face_tcp(tk), ride(tk)[1], CLEAR_Z + s * (TIP_Z - CLEAR_Z))),
                             FINGER_CLOSED, "down beside it", YAW))
    t += n * DT
    x0 = _near_face_tcp(t)
    n = round(PUSH_S / DT)
    for k in range(1, n + 1):  # push it out to the lane while riding along
        s = k / n
        s = s * s * (3 - 2 * s)
        tk = t + k * DT
        plan.append(Waypoint(DT, _floats((x0 + s * (ESCORT_X - x0), ride(tk)[1], TIP_Z)), FINGER_CLOSED,
                             "push it to the lane", YAW))
    t += n * DT
    t_release = time_to(RELEASE_Y)
    n = round((t_release - t) / DT)
    for k in range(1, n + 1):  # ride beside it, a moving side rail
        tk = t + k * DT
        plan.append(Waypoint(DT, _floats((ESCORT_X, ride(tk)[1], TIP_Z)), FINGER_CLOSED, "ride beside it", YAW))
    t += n * DT
    y = ride(t)[1]
    plan += [
        Waypoint(BACK_S, _floats((ESCORT_X - 0.012, y, TIP_Z)), FINGER_CLOSED, "back off", YAW),
        Waypoint(UP_S, _floats((ESCORT_X - 0.012, y, CLEAR_Z + 0.03)), FINGER_CLOSED, "up and away", YAW),
    ]
    return {"left": plan}
