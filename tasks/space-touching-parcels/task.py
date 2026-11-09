"""On the station table, a belt conveyor runs straight across the table from the robot's right to its left, 32 cm in
front of the arm bases: a dark rubber belt 16 cm wide on a grey steel bed, running at 5 cm/s. Three brown cardboard
parcels (9 cm long along the belt, 6 cm across it, 5 cm tall) ride on it nose to tail, touching: the front one's
leading face starts 5 cm right of the midpoint between the arms. 58 cm left of the midpoint, a scanner arch stands
over the belt (two posts beside it and a head across it, a red line under the head marking the scan line). Past it
the belt runs on, off the table's left edge, to an end stop.

The scanner needs at least 5 cm of clear belt in front of each parcel as it passes. The robot must open those gaps
before the parcels reach the arch, while the belt keeps running, and keep every parcel on the belt.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The three parcels on the belt are touching. Space them out ..." ...
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import Belt, episode_time, on_reset, on_step, speed_schedule  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import FLOOR_Z  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # the monitor below records each parcel's pass under the scanner; the score is read at the end


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the belt --------------------------------------------------------------------------------------------------------
# +y (to the robot's left) is the direction of flow. The belt starts with the trial and runs at one speed throughout.
SPEED = 0.05  # m/s
RAMP_S = 0.5  # from standing to full speed, smoothly
RUN = speed_schedule(((0.0, SPEED),), RAMP_S)

LINE_X = 0.32  # the belt's centre line, metres in front of the arm bases
BELT_W = 0.16
BELT_NEAR, BELT_FAR = LINE_X - BELT_W / 2, LINE_X + BELT_W / 2  # its edges (x)
START_Y, END_Y = -0.62, 1.06  # the conveyor's two ends: 3 cm inside the table's right edge, 41 cm past its left edge
BED = (BELT_W + 0.02, END_Y - START_Y, 0.015)  # grey steel bed, 1 cm wider than the belt on either side
BED_TOP = TOP + BED[2]
BELT_T = 0.01
BELT_Z = BED_TOP + 0.002 + BELT_T / 2  # 2 mm above the bed: the belt never rubs on it
BELT_TOP = BELT_Z + BELT_T / 2
MID_Y = (START_Y + END_Y) / 2
TABLE_EDGE_Y = 0.65
GREY_STEEL = (0.62, 0.64, 0.66)
DARK = (0.18, 0.18, 0.2)
RUBBER = (0.12, 0.12, 0.13)
RED = (0.85, 0.1, 0.08)
CARDBOARD = (0.70, 0.53, 0.34)

BELT = Belt("belt", pos=_floats((LINE_X, MID_Y, BELT_Z)), size=_floats((BELT_W, END_Y - START_Y, BELT_T)),
            color=RUBBER, run=RUN, direction=(0.0, 1.0, 0.0))
CONVEYOR = [
    Box("conveyor_bed", pos=_floats((LINE_X, MID_Y, TOP + BED[2] / 2)), size=_floats(BED), color=GREY_STEEL),
    # past the table's left edge the bed stands on a post to the floor
    Box("conveyor_post", pos=_floats((LINE_X, END_Y - 0.05, (TOP + FLOOR_Z) / 2)),
        size=_floats((0.06, 0.04, TOP - FLOOR_Z)), color=DARK),
    # an end stop across the belt's far end: parcels that have passed the scanner gather against it
    Box("end_stop", pos=_floats((LINE_X, END_Y + 0.002 + 0.005, (TOP + BELT_TOP + 0.06) / 2)),
        size=_floats((BED[0], 0.01, BELT_TOP + 0.06 - TOP)), color=GREY_STEEL),
]

# -- the scanner arch ------------------------------------------------------------------------------------------------
SCAN_Y = 0.58  # the scan line (the red line under the scanner's head), 58 cm left of the midpoint between the arms
POST = 0.02  # square posts
POST_IN = BED[0] / 2 + 0.005  # the posts' inner faces, 5 mm outside the bed
HEAD_LOW = BELT_TOP + 0.13  # the head's underside, 8 cm over the parcels' tops
HEAD = (2 * (POST_IN + POST), 0.03, 0.03)
MARK = 0.004  # the red line: 4 mm wide and thick, under the head, across the belt


def _scanner() -> list[Box]:
    post_h = HEAD_LOW - TOP
    return [
        Box("scanner_post_near", pos=_floats((LINE_X - POST_IN - POST / 2, SCAN_Y, TOP + post_h / 2)),
            size=_floats((POST, POST, post_h)), color=DARK),
        Box("scanner_post_far", pos=_floats((LINE_X + POST_IN + POST / 2, SCAN_Y, TOP + post_h / 2)),
            size=_floats((POST, POST, post_h)), color=DARK),
        Box("scanner_head", pos=_floats((LINE_X, SCAN_Y, HEAD_LOW + HEAD[2] / 2)), size=_floats(HEAD), color=DARK),
        Box("scan_line", pos=_floats((LINE_X, SCAN_Y, HEAD_LOW - MARK / 2)), size=_floats((2 * POST_IN, MARK, MARK)),
            color=RED),
    ]


# -- the parcels -----------------------------------------------------------------------------------------------------
PARCEL = (0.06, 0.09, 0.05)  # across the belt (x), along it (y), tall
PARCEL_MASS = 0.15
TOUCH = 0.002  # "touching": 2 mm apart at the start, so the boxes don't spawn into each other
FRONT_Y0 = -0.05  # the front parcel's leading face at the start
NAMES = ("parcel_1", "parcel_2", "parcel_3")  # from the front of the train back
PARCEL_Z = BELT_TOP + PARCEL[2] / 2
PARCEL_TOP = BELT_TOP + PARCEL[2]
START = {n: FRONT_Y0 - PARCEL[1] / 2 - i * (PARCEL[1] + TOUCH) for i, n in enumerate(NAMES)}  # centres (y)
PARCELS = [Box(n, pos=_floats((LINE_X, START[n], PARCEL_Z)), size=_floats(PARCEL), mass=PARCEL_MASS, color=CARDBOARD)
           for n in NAMES]

OBJECTS = [*CONVEYOR, BELT, *_scanner(), *PARCELS]


def riding_y(name: str, t: float, shift: float = 0.0) -> float:
    """Where parcel ``name``'s centre is along the line at ``t`` if it rides the belt from its start, moved ``shift``
    along the belt (+ downstream)."""
    return float(START[name] + shift + RUN(t)[0])


# -- the scanner's record ----------------------------------------------------------------------------------------------
# Every control step the monitor below follows each parcel's leading face (its highest y) and trailing face (its
# lowest y). A parcel's pass under the scanner runs from the moment its leading face reaches the scan line to the
# moment its trailing face leaves it (the times are interpolated between control steps). Throughout its pass it must
# lie on the belt in its lane; only its first pass counts. The gap in front of a parcel is what a scanner's photo-eye
# measures: the length of belt that ran past the line between the parcel ahead (the one that reached the line before
# it) leaving the line and this parcel reaching it.
MIN_GAP = 0.05  # the gap the scanner needs
HALF_GAP = 0.025  # half credit for a gap at least this long
ON_BELT = 0.01  # lying on the belt: the parcel's underside within 1 cm of the belt's top
LANE_TOL = 0.0  # in the lane: the parcel's footprint within the belt's edges

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update({n: dict(last=None, front_t=None, rear_t=None, fault=None) for n in NAMES})


reset_state()


def on_belt(p) -> str | None:
    """Why the parcel (an ObjectState) is not lying on the belt in its lane, or None if it is."""
    if not BELT_TOP - ON_BELT <= p.low[2] <= BELT_TOP + ON_BELT:
        return "not lying on the belt"
    if p.low[0] < BELT_NEAR - LANE_TOL or p.high[0] > BELT_FAR + LANE_TOL:
        return "over the belt's edge"
    return None


def _crossed(last_t, last_face, t, face) -> float | None:
    """When a face (y) reached the scan line between the last control step and this one, or None if it didn't."""
    if face < SCAN_Y:
        return None
    if last_face is None or last_face >= SCAN_Y or t <= last_t:
        return float(t)
    return float(last_t + (t - last_t) * (SCAN_Y - last_face) / (face - last_face))


def update(t: float, parcels) -> None:
    """One control step of the scanner's record at episode time ``t``: ``parcels`` maps each name to its state."""
    for n in NAMES:
        rec, p = _STATE[n], parcels[n]
        front, rear = float(p.high[1]), float(p.low[1])
        last_t, last_front, last_rear = rec["last"] if rec["last"] is not None else (t, None, None)
        if rec["front_t"] is None:
            rec["front_t"] = _crossed(last_t, last_front, t, front)
        if rec["front_t"] is not None and rec["rear_t"] is None:
            why = on_belt(p)
            if why is not None and rec["fault"] is None:
                rec["fault"] = f"{n} {why} under the scanner"
            rec["rear_t"] = _crossed(last_t, last_rear, t, rear)
        rec["last"] = (t, front, rear)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    t = episode_time()
    states = object_states(environment, list(NAMES))
    passed = {n: _STATE[n]["front_t"] for n in NAMES}
    update(t, states)
    if DEBUG:
        _debug(t, states, passed)


DEBUG = os.environ.get("PARCEL_DEBUG") == "1"  # PARCEL_DEBUG=1: print the parcels every 0.5 s and each gap measured


def _debug(t: float, states, passed) -> None:
    if round(t / 0.5, 6) % 1 == 0:
        where = "  ".join(f"{n}: x {states[n].pos[0]:.3f} y {states[n].pos[1]:+.3f} z {states[n].low[2] - BELT_TOP:+.3f}"
                          for n in NAMES)
        print(f"[parcels] t {t:5.2f}  belt {RUN(t)[0]:.3f}  {where}", flush=True)
    for n in NAMES:
        rec = _STATE[n]
        if passed[n] is None and rec["front_t"] is not None:
            ahead = [m for m in NAMES if m != n and _STATE[m]["front_t"] is not None and _STATE[m]["front_t"] < rec["front_t"]]
            if ahead:
                a = max(ahead, key=lambda m: _STATE[m]["front_t"])
                left = _STATE[a]["rear_t"]
                gap = RUN(rec["front_t"])[0] - RUN(left)[0] if left is not None else float("nan")
                print(f"[parcels] {n} reached the scan line at {rec['front_t']:.2f} s, gap to {a}: {gap * 100:.1f} cm",
                      flush=True)
            else:
                print(f"[parcels] {n} reached the scan line first at {rec['front_t']:.2f} s", flush=True)
        if rec["fault"] is not None and not rec.get("printed"):
            rec["printed"] = True
            print(f"[parcels] fault: {rec['fault']}", flush=True)
    if all(_STATE[n]["rear_t"] is not None for n in NAMES) and not _STATE.get("_done"):
        _STATE["_done"] = True
        print(f"[parcels] all passed: score {result()[0]} ({result()[1]})", flush=True)


def gap_credit(gap: float) -> float:
    return 1.0 if gap >= MIN_GAP else 0.5 if gap >= HALF_GAP else 0.0


def result() -> tuple[float, str]:
    """The score so far and why: 0 until every parcel has passed under the scanner lying on the belt in its lane;
    then 0.5 for each of the two gaps (in front of the second and the third parcel to pass) of 5 cm or more, 0.25 for
    one of 2.5 to 5 cm."""
    for n in NAMES:
        rec = _STATE[n]
        if rec["fault"] is not None:
            return 0.0, rec["fault"]
        if rec["rear_t"] is None:
            return 0.0, f"{n} has not passed the scanner"
    found = gaps()
    return 0.5 * sum(gap_credit(g) for g in found), "gaps at the scanner: " + ", ".join(
        f"{g * 100:.1f} cm" for g in found)


def gaps() -> list[float]:
    """The two gaps at the scanner (m), in front of the second and the third parcel to reach the line; negative if a
    parcel reached the line before the one ahead had left it. Only once every parcel has reached the line."""
    order = sorted(NAMES, key=lambda n: _STATE[n]["front_t"])
    return [RUN(_STATE[b]["front_t"])[0] - RUN(_STATE[a]["rear_t"])[0] for a, b in zip(order, order[1:])]


def score(objects):
    """See :func:`result`. The record is kept by the monitor, which runs before every scoring."""
    return result()[0]


def success(objects):
    """Every parcel passed the scanner lying on the belt in its lane, with 5 cm or more in front of the second and
    the third."""
    return result()[0] == 1.0


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop, timed against the belt. The right arm first: it takes the last parcel, lifts it 5 cm off the belt and
# sets it down 13 cm farther back; then the left arm takes the front parcel and sets it down 10 cm farther ahead, so
# the two wrists are never down at the belt together. Each arm follows its parcel along the belt from the moment it
# comes down onto it until it has let go and come back up: the parcel is gripped, and set down, moving with the belt
# (no relative velocity). Fingers close across the belt, on the parcels' free side faces, never between two parcels
# (the fingers reach 3.4 cm along the belt either side of the grip, 1.1 cm short of the neighbour 2 mm away): the left
# gripper turned -90 degrees, the right one +90 (each arm's joints stay inside their limits that way). The pads grip
# the parcel around its centre of mass, over 3.5 cm of its 5 cm height, so it doesn't swing in the fingers as it is
# carried, and the fingers hold it 0.3 s before lifting it.
DT = 0.05  # one control step at 20 Hz
YAW = {"left": float(-np.pi / 2), "right": float(np.pi / 2)}
GRASP_Z = PARCEL_Z + 0.003  # TCP 3 mm above the parcel's centre: the pad tips stop 1.5 cm above the belt
PRE_Z = PARCEL_TOP + PAD_TIP_PAST_TCP + 0.03  # over the belt: the open pad tips 3 cm above the parcels' tops
OFF_Z = GRASP_Z + 0.05  # lifted: the parcel's underside 5 cm above the belt
PLACE_Z = GRASP_Z + 0.003  # let go with the parcel's underside 3 mm above the belt
PARK_Z = TOP + 0.15
LEFT_MEET = 4.8
DESCEND_S, CLOSE_S, GRIP_S, LIFT_S, LOWER_S, RELEASE_S, UP_S = 0.6, 0.5, 0.3, 0.5, 0.6, 0.4, 0.5
# (arm, parcel, how far it goes along the belt, when the arm starts down onto it, how long it carries it, where the
# arm waits afterwards). The left arm comes down onto its parcel once the right arm has carried its own 13 cm back,
# away from it: the two wrists, which lean toward each other when both are down at the belt, stay 8 cm or more apart.
MOVES = (("right", "parcel_3", -0.13, 2.0, 1.5, (0.16, -0.30)), ("left", "parcel_1", 0.10, LEFT_MEET, 1.0, (0.16, 0.30)))
END_S = 23.5  # the plans hold their park pose until the last parcel has passed (20.9 s) and success has held 1 s


def _track(arm: str, name: str, t0: float, seconds: float, z0: float, z1: float, f0: float, f1: float, s0: float, s1: float,
           label: str) -> list[Waypoint]:
    """Waypoints one control step apart that follow parcel ``name`` along the belt from ``t0``, while the TCP goes
    from ``z0`` to ``z1``, the fingers from ``f0`` to ``f1`` and the parcel's place on the belt from ``s0`` to ``s1``
    ahead of where it started (each smoothly)."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        y = riding_y(name, t0 + k * DT, s0 + s * (s1 - s0))
        out.append(Waypoint(DT, (LINE_X, y, z0 + s * (z1 - z0)), f0 + s * (f1 - f0), label, YAW[arm]))
    return out


def move_plan(arm: str, name: str, shift: float, meet: float, shift_s: float, park) -> list[Waypoint]:
    """Arm ``arm``: go over parcel ``name`` where it will be at ``meet``, follow it down, close, lift it, carry it
    ``shift`` along the belt in ``shift_s``, set it down, let go, come back up, and wait at ``park`` (x, y)."""
    y = riding_y(name, meet)
    if meet > 3.0:  # time to spare: wait high over the meeting point, then come down to just over the belt
        plan = [Waypoint(2.0, (LINE_X, y, PARK_Z), FINGER_OPEN, f"high over {name}", YAW[arm]),
                Waypoint(meet - 3.0, (LINE_X, y, PARK_Z), FINGER_OPEN, f"wait for {name}", YAW[arm]),
                Waypoint(1.0, (LINE_X, y, PRE_Z), FINGER_OPEN, f"over {name}", YAW[arm])]
    else:
        plan = [Waypoint(meet, (LINE_X, y, PRE_Z), FINGER_OPEN, f"over {name}", YAW[arm])]
    t = meet
    for seconds, z, f, s, label in (
        (DESCEND_S, (PRE_Z, GRASP_Z), (FINGER_OPEN, FINGER_OPEN), (0.0, 0.0), "follow it down"),
        (CLOSE_S, (GRASP_Z, GRASP_Z), (FINGER_OPEN, FINGER_CLOSED), (0.0, 0.0), "close"),
        (GRIP_S, (GRASP_Z, GRASP_Z), (FINGER_CLOSED, FINGER_CLOSED), (0.0, 0.0), "grip"),
        (LIFT_S, (GRASP_Z, OFF_Z), (FINGER_CLOSED, FINGER_CLOSED), (0.0, 0.0), "lift"),
        (shift_s, (OFF_Z, OFF_Z), (FINGER_CLOSED, FINGER_CLOSED), (0.0, shift), "carry along the belt"),
        (LOWER_S, (OFF_Z, PLACE_Z), (FINGER_CLOSED, FINGER_CLOSED), (shift, shift), "set down"),
        (RELEASE_S, (PLACE_Z, PLACE_Z), (FINGER_CLOSED, FINGER_OPEN), (shift, shift), "let go"),
        (UP_S, (PLACE_Z, PRE_Z), (FINGER_OPEN, FINGER_OPEN), (shift, shift), "up"),
    ):
        plan += _track(arm, name, t, seconds, *z, *f, *s, label)
        t += seconds
    plan.append(Waypoint(1.0, (*park, PARK_Z), FINGER_OPEN, "wait clear of the belt", YAW[arm]))
    # yam_scripted ends the rollout 3 s after the plan does: hold the park pose until the parcels have passed
    plan.append(Waypoint(END_S - sum(w.duration for w in plan), (*park, PARK_Z), FINGER_OPEN,
                         "wait while the parcels pass", YAW[arm]))
    return plan


def demo():
    """Right arm: parcel_3 13 cm back, starting down at 2.0 s, let go by 6.4 s. Left arm: parcel_1 10 cm ahead,
    starting down at 4.8 s, let go by 8.7 s; parcel_1 reaches the scan line at 10.85 s, and the last parcel
    (parcel_3) has passed it at 20.9 s, with 10.2 cm of belt in front of parcel_2 and 13.2 cm in front of parcel_3."""
    return {arm: move_plan(arm, *move) for arm, *move in MOVES}
