"""On the station table, a belt conveyor runs straight across the table from the robot's right to its left, 32 cm in
front of the arm bases: a dark rubber belt 16 cm wide on a grey steel bed, running at 6 cm/s. Four brown cardboard
parcels (9 cm long along the belt, 6 cm across it, 5 cm tall) ride on it in a line, 6 cm apart, except for one empty
24 cm gap between the second and the third. A fifth parcel waits on a small grey steel infeed table beside the belt,
between the belt and the left arm, its top level with the belt. 52 cm left of the midpoint between the arms, a scanner
arch stands over the belt (two posts beside it and a head across it, a red line under the head marking the scan line).
Past it the belt runs on, off the table's left edge, to an end stop.

The robot must merge the waiting parcel into the gap while the belt runs: set it onto the belt between the second and
the third parcel, square to the belt, so that it has at least 4 cm of clear belt on both sides when it passes the
scanner, without touching the other parcels: neither the arms nor the merged parcel may touch any of the four at
any time. It must be let go before it reaches the scanner: no arm may touch it during its pass.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Put the parcel from the side table into the empty gap ..." ...
"""

from __future__ import annotations

import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import (AwakeBox, Belt, arm_joints, episode_time, on_reset, on_step, robot_link_poses,  # noqa: E402
                       robot_links, speed_schedule)
from yam_arena.analysis import box_signed_distance, link_hulls, world_hulls  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import FLOOR_Z  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402
from yam_arena.urdf import quat_xyzw_to_matrix  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # the monitor below records each parcel's pass under the scanner and every touch of the four
#                     parcels of the line, over the whole trial; the score is read at the end


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the belt --------------------------------------------------------------------------------------------------------
# +y (to the robot's left) is the direction of flow. The belt starts with the trial and runs at one speed throughout.
SPEED = 0.06  # m/s
RAMP_S = 0.5  # from standing to full speed, smoothly
RUN = speed_schedule(((0.0, SPEED),), RAMP_S)

LINE_X = 0.32  # the belt's centre line, metres in front of the arm bases
BELT_W = 0.16
BELT_NEAR, BELT_FAR = LINE_X - BELT_W / 2, LINE_X + BELT_W / 2  # its edges (x)
START_Y, END_Y = -0.62, 1.10  # the conveyor's two ends: 3 cm inside the table's right edge, 45 cm past its left edge
# (room for all five parcels to queue against the end stop with the last one 13 cm past the scan line)
BED = (BELT_W + 0.02, END_Y - START_Y, 0.015)  # grey steel bed, 1 cm wider than the belt on either side
BED_TOP = TOP + BED[2]
BED_NEAR = LINE_X - BED[0] / 2
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

# Measured in Isaac Sim (MERGE_DEBUG runs): parcels on this belt model ride a steady ~1.1 cm/s slower than the surface
# it is given (4.9 cm/s on a 6 cm/s surface; 6.3 on 7.4). The plate is driven at 7.1 cm/s (6 / 0.845) so they ride at
# the belt's 6 cm/s, which the demo and the descriptions assume. Scoring doesn't depend on it (it measures lengths).
CARRY = 0.845
BELT = Belt("belt", pos=_floats((LINE_X, MID_Y, BELT_Z)), size=_floats((BELT_W, END_Y - START_Y, BELT_T)),
            color=RUBBER, run=RUN, direction=(0.0, 1.0, 0.0), carry=CARRY)
CONVEYOR = [
    Box("conveyor_bed", pos=_floats((LINE_X, MID_Y, TOP + BED[2] / 2)), size=_floats(BED), color=GREY_STEEL),
    # past the table's left edge the bed stands on a post to the floor
    Box("conveyor_post", pos=_floats((LINE_X, END_Y - 0.05, (TOP + FLOOR_Z) / 2)),
        size=_floats((0.06, 0.04, TOP - FLOOR_Z)), color=DARK),
    # an end stop across the belt's far end: parcels that have passed the scanner gather against it
    Box("end_stop", pos=_floats((LINE_X, END_Y + 0.002 + 0.005, (TOP + BELT_TOP + 0.06) / 2)),
        size=_floats((BED[0], 0.01, BELT_TOP + 0.06 - TOP)), color=GREY_STEEL),
]

# -- the infeed table ------------------------------------------------------------------------------------------------
# A small steel table on the near side of the belt, in front of the left arm, its top level with the belt's top: the
# side table of a merge station, from which parcels are put onto the running line.
INFEED = (0.12, 0.16, BELT_TOP - TOP)  # across the belt (x), along it (y), tall
INFEED_XY = (0.155, 0.05)
INFEED_TABLE = Box("infeed_table", pos=_floats((*INFEED_XY, TOP + INFEED[2] / 2)), size=_floats(INFEED),
                   color=GREY_STEEL)

# -- the scanner arch ------------------------------------------------------------------------------------------------
SCAN_Y = 0.52  # the scan line (the red line under the scanner's head), 52 cm left of the midpoint between the arms
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
SPACING = 0.06  # empty belt between neighbouring parcels of the line
GAP = 0.24  # the empty gap between the second and the third parcel: room for one parcel with 7.5 cm either side
REAR_Y0 = -0.58  # the last parcel's trailing face at the start, 4 cm inside the belt's upstream end
TRAIN = ("parcel_1", "parcel_2", "parcel_3", "parcel_4")  # on the belt, from the front of the line back
MERGE = "merge_parcel"  # on the infeed table
NAMES = (*TRAIN, MERGE)
PARCEL_Z = BELT_TOP + PARCEL[2] / 2  # also on the infeed table, whose top is level with the belt's
PARCEL_TOP = BELT_TOP + PARCEL[2]


def _train_start() -> dict[str, float]:
    out, rear = {}, REAR_Y0
    for n in reversed(TRAIN):
        out[n] = rear + PARCEL[1] / 2
        rear += PARCEL[1] + (GAP if n == "parcel_3" else SPACING)
    return dict(sorted(out.items()))


START = _train_start()  # centres (y) at the start
GAP0 = (START["parcel_2"] - PARCEL[1] / 2 + START["parcel_3"] + PARCEL[1] / 2) / 2  # the gap's middle at the start
# The parcels never sleep: they settle while the belt stands through the reset, and a sleeping parcel would not be
# woken by the belt starting under it (a first sim run suggested the line stayed put while the belt ran).
PARCELS = [AwakeBox(n, pos=_floats((LINE_X, START[n], PARCEL_Z)), size=_floats(PARCEL), mass=PARCEL_MASS,
                    color=CARDBOARD) for n in TRAIN]
MERGE_PARCEL = AwakeBox(MERGE, pos=_floats((*INFEED_XY, PARCEL_Z)), size=_floats(PARCEL), mass=PARCEL_MASS,
                   color=CARDBOARD)

OBJECTS = [*CONVEYOR, BELT, INFEED_TABLE, *_scanner(), *PARCELS, MERGE_PARCEL]


def riding_y(name: str, t: float, shift: float = 0.0) -> float:
    """Where train parcel ``name``'s centre is along the line at ``t`` if it rides the belt from its start, moved
    ``shift`` along the belt (+ downstream)."""
    return float(START[name] + shift + RUN(t)[0])


def gap_y(t: float) -> float:
    """Where the middle of the gap is along the line at ``t``."""
    return float(GAP0 + RUN(t)[0])


# -- the scanner's record ----------------------------------------------------------------------------------------------
# Every control step the monitor below follows each parcel's leading face (the highest y of its bounding box) and
# trailing face (its lowest y). A parcel's pass under the scanner runs from the moment its leading face reaches the
# scan line to the moment its trailing face leaves it (times interpolated between control steps). Throughout its pass
# it must lie on the belt in its lane; only its first pass counts. The clear belt between two parcels is measured at
# the moment the parcel ahead leaves the line: from the line to the next one's leading face (interpolated between
# control steps). It is a length along the belt, so it doesn't depend on how fast the parcels ride. During each pass the monitor also keeps the parcel's worst turn about the vertical
# (from square to the belt) and worst tilt (from lying on its 9 x 6 cm face), and where it was across the belt when
# it reached the line.
CLEAR = 0.04  # clear belt the merged parcel needs ahead of and behind it
SQUARE_DEG = 15.0  # the merged parcel turned or tilted no more than this
KEPT = 0.015  # undisturbed: the line's spacing at the scan line within this of how it started, each parcel within
#               this of the belt's centre line
KEPT_DEG = 5.0  # ... and turned or tilted no more than this
ON_BELT = 0.01  # lying on the belt: the parcel's underside within 1 cm of the belt's top
LANE_TOL = 0.0  # in the lane: the parcel's footprint within the belt's edges

# Contact with the four parcels of the line, watched every control step of the whole trial. A body touches a parcel
# when it comes within TOUCH of the parcel's box (PhysX starts making contacts 5 mm out): an arm when any of its
# links' collision hulls (the URDF's, placed where the simulator has the links) does, the merged parcel when its box
# does. Two parcels that have both passed under the scanner may touch (they queue against the end stop). The merged
# parcel itself must be let go before it reaches the scanner: no arm may touch it during its pass (no holding it on the
# belt or steering it through).
TOUCH = 0.005

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update({n: dict(last=None, front_t=None, rear_t=None, fault=None, x=None, turn=0.0, tilt=0.0, fronts=None)
                   for n in NAMES})
    _STATE["touch"] = None  # the first touch of the line's parcels, or of the merged parcel in its pass: (t, what)
    # closest approach (m): of the arms and of the merged parcel to the line's parcels, of the arms to the merged
    # parcel during its pass
    _STATE["closest"] = {"arms": np.inf, "merged": np.inf, "held": np.inf}


reset_state()


def on_belt(p) -> str | None:
    """Why the parcel (an ObjectState) is not lying on the belt in its lane, or None if it is."""
    if not BELT_TOP - ON_BELT <= p.low[2] <= BELT_TOP + ON_BELT:
        return "not lying on the belt"
    if p.low[0] < BELT_NEAR - LANE_TOL or p.high[0] > BELT_FAR + LANE_TOL:
        return "over the belt's edge"
    return None


def turn_and_tilt(quat_xyzw) -> tuple[float, float]:
    """A parcel's turn about the vertical away from square to the belt (its 9 cm edges along the belt, either way
    round: 0 to 90 degrees) and its tilt from lying on its 9 x 6 cm face (0 to 90 degrees)."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    long_axis = np.array([2 * (x * y - w * z), 1 - 2 * (x * x + z * z)])  # the parcel's own y axis, in plan (x, y)
    up_z = 1 - 2 * (x * x + y * y)  # the vertical component of its own z axis
    turn = np.degrees(np.arctan2(abs(long_axis[0]), abs(long_axis[1])))
    tilt = np.degrees(np.arccos(np.clip(abs(up_z), 0.0, 1.0)))
    return float(turn), float(tilt)


def _box(p) -> tuple[np.ndarray, np.ndarray]:
    """A parcel's (ObjectState) box: its centre and rotation matrix (its size is PARCEL)."""
    return np.asarray(p.pos, float), quat_xyzw_to_matrix(np.asarray(p.quat_xyzw, float))


_SIGNS = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)


def _corners(p) -> np.ndarray:
    centre, rot = _box(p)
    return centre + (_SIGNS * np.asarray(PARCEL) / 2) @ rot.T


def parcel_gap(a, b) -> float:
    """How far apart two parcels' boxes are (m; negative if they overlap, by as much): the widest separation along
    any of the 15 separating axes of two boxes. Exact for boxes facing each other; never more than the true distance
    otherwise, so it errs towards touching."""
    (ca, ra), (cb, rb) = _box(a), _box(b)
    half = np.asarray(PARCEL) / 2
    axes = [*ra.T, *rb.T]
    axes += [n / np.linalg.norm(n) for u in ra.T for v in rb.T if np.linalg.norm(n := np.cross(u, v)) > 1e-6]
    return float(max(abs(np.dot(cb - ca, n)) - np.abs(ra.T @ n) @ half - np.abs(rb.T @ n) @ half for n in axes))


def arm_gap(hulls, p) -> tuple[float, str | None]:
    """How far an arm's collision hulls (world ``(link, Hull)`` pairs) are from a parcel's box (m; negative if they
    overlap), and the nearest link: the nearest hull vertex to the box, or box corner to a hull (a hull's face
    planes give a lower bound for a point outside it, so this errs towards touching)."""
    centre, rot = _box(p)
    corners = _corners(p)
    lo, hi = corners.min(0) - 0.05, corners.max(0) + 0.05
    best, link = np.inf, None
    for name, h in hulls:
        if np.any(h.vertices.min(0) > hi) or np.any(h.vertices.max(0) < lo):
            continue  # more than 5 cm apart
        d = min(float(box_signed_distance(h.vertices, centre, rot, PARCEL).min()),
                float((corners @ h.planes[:, :3].T - h.planes[:, 3]).max(axis=1).min()))
        if d < best:
            best, link = d, name
    return best, link


def _link_corners() -> dict[str, np.ndarray]:
    """The corners of each link's collision hulls' bounding box, in the link's frame (homogeneous, 8 x 4)."""
    if "corners" not in _HULL_BOXES:
        out = {}
        for name, hulls in link_hulls().items():
            v = np.concatenate([h.vertices for h in hulls])
            lo, hi = v.min(0), v.max(0)
            out[name] = np.c_[np.where(_SIGNS > 0, hi, lo), np.ones(8)]
        _HULL_BOXES["corners"] = out
    return _HULL_BOXES["corners"]


_HULL_BOXES: dict = {}


def near_hulls(poses, lo, hi):
    """The collision hulls (world ``(link, Hull)`` pairs) of those links (``poses``: 4x4 by link name) whose bounding
    box comes within 5 cm of the box from ``lo`` to ``hi``."""
    corners, near = _link_corners(), {}
    if not set(poses) & set(corners) and "named" not in _HULL_BOXES:
        _HULL_BOXES["named"] = True
        print(f"[merge] WARNING: no arm link has a collision hull: links {sorted(poses)}", flush=True)
    for name, pose in poses.items():
        if name not in corners:
            continue
        w = (corners[name] @ pose.T)[:, :3]
        if np.all(w.min(0) <= hi + 0.05) and np.all(w.max(0) >= lo - 0.05):
            near[name] = pose
    return world_hulls(near) if near else []


def watch_contacts(t: float, parcels, link_poses=None) -> None:
    """One control step of the contact watch: whether an arm (``link_poses``: each arm's link poses as 4x4 matrices,
    by side and link name) or the merged parcel touches any of the four parcels of the line, and whether an arm
    touches the merged parcel during its pass under the scanner. Latches the first touch for the rest of the
    trial."""
    closest = _STATE["closest"]
    every = np.concatenate([_corners(parcels[n]) for n in TRAIN])
    for side, poses in (link_poses or {}).items():
        hulls = near_hulls(poses, every.min(0), every.max(0))
        for n in TRAIN:
            d, link = arm_gap(hulls, parcels[n])
            closest["arms"] = min(closest["arms"], d)
            if d < TOUCH and _STATE["touch"] is None:
                _STATE["touch"] = (float(t), f"the {side} arm's {link} touched {n}")
        if _STATE[MERGE]["front_t"] is not None and _STATE[MERGE]["rear_t"] is None:  # the merged parcel's pass
            mine = _corners(parcels[MERGE])
            d, link = arm_gap(near_hulls(poses, mine.min(0), mine.max(0)), parcels[MERGE])
            closest["held"] = min(closest["held"], d)
            if d < TOUCH and _STATE["touch"] is None:
                _STATE["touch"] = (float(t), f"the {side} arm's {link} touched the merged parcel under the scanner")
    for n in TRAIN:
        if _STATE[n]["rear_t"] is not None and _STATE[MERGE]["rear_t"] is not None:
            continue  # both past the scanner: they may queue against the end stop
        d = parcel_gap(parcels[MERGE], parcels[n])
        closest["merged"] = min(closest["merged"], d)
        if d < TOUCH and _STATE["touch"] is None:
            _STATE["touch"] = (float(t), f"the merged parcel touched {n}")


def _crossed(last_t, last_face, t, face) -> float | None:
    """When a face (y) reached the scan line between the last control step and this one, or None if it didn't."""
    if face < SCAN_Y:
        return None
    if last_face is None or last_face >= SCAN_Y or t <= last_t:
        return float(t)
    return float(last_t + (t - last_t) * (SCAN_Y - last_face) / (face - last_face))


def update(t: float, parcels) -> None:
    """One control step of the scanner's record at episode time ``t``: ``parcels`` maps each name to its state."""
    fronts = {n: float(parcels[n].high[1]) for n in NAMES}
    for n in NAMES:
        rec, p = _STATE[n], parcels[n]
        front, rear = fronts[n], float(p.low[1])
        last_t, last_front, last_rear = rec["last"] if rec["last"] is not None else (t, None, None)
        if rec["front_t"] is None:
            rec["front_t"] = _crossed(last_t, last_front, t, front)
            if rec["front_t"] is not None:
                rec["x"] = float(p.pos[0])
        if rec["front_t"] is not None and rec["rear_t"] is None:
            why = on_belt(p)
            if why is not None and rec["fault"] is None:
                rec["fault"] = f"{n} {why} under the scanner"
            turn, tilt = turn_and_tilt(p.quat_xyzw)
            rec["turn"], rec["tilt"] = max(rec["turn"], turn), max(rec["tilt"], tilt)
            rec["rear_t"] = _crossed(last_t, last_rear, t, rear)
            if rec["rear_t"] is not None:  # where every other parcel's leading face was as this one left the line
                a = 1.0 if t <= last_t else (rec["rear_t"] - last_t) / (t - last_t)
                rec["fronts"] = {m: f if _STATE[m]["last"] is None else _STATE[m]["last"][1] + a * (
                    f - _STATE[m]["last"][1]) for m, f in fronts.items() if m != n}
    for n in NAMES:
        p = parcels[n]
        _STATE[n]["last"] = (t, fronts[n], float(p.low[1]))


DEBUG = os.environ.get("MERGE_DEBUG") == "1"  # MERGE_DEBUG=1: print the merged parcel and the left gripper


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    t = episode_time()
    states = object_states(environment, list(NAMES))
    update(t, states)
    watch_contacts(t, states, robot_link_poses(environment))
    if DEBUG:
        _debug(environment, t, states)


def _debug(environment, t: float, states) -> None:
    """Every 0.5 s: the merged parcel's pose and where the gap is, and how far each parcel of the line is from where
    riding the belt puts it; every control step while the left gripper is within 10 cm of the merged parcel: the pad
    gap and the fingers' midpoint relative to the parcel's centre."""
    p = states[MERGE]
    turn, tilt = turn_and_tilt(p.quat_xyzw)
    k = round(t / 0.05)
    if k % 10 == 0:
        line = " ".join(f"{n[-1]}:y={states[n].pos[1]:.3f}({100 * (states[n].pos[1] - riding_y(n, t)):+.1f})"
                        for n in TRAIN)
        print(f"[merge] t={t:5.2f} belt run={RUN(t)[0]:.3f} m line (cm off riding) {line}", flush=True)
        print(f"[merge] t={t:5.2f} parcel pos=({p.pos[0]:.3f}, {p.pos[1]:.3f}, {p.pos[2]:.3f}) "
              f"underside-belt={100 * (p.low[2] - BELT_TOP):+.1f} cm turn={turn:.0f} tilt={tilt:.0f} "
              f"gap middle y={gap_y(t):.3f} parcel-gap={100 * (p.pos[1] - gap_y(t)):+.1f} cm", flush=True)
    links = robot_links(environment)["left"]
    mid = (np.asarray(links["tip_left"]) + np.asarray(links["tip_right"])) / 2
    if np.linalg.norm(mid - p.pos) < 0.10:
        j7, j8 = arm_joints(environment, "left", ["joint7", "joint8"])
        rel = 100 * (mid - p.pos)
        print(f"[merge]   t={t:5.2f} finger joints=({j7:+.4f}, {j8:+.4f}) pad gap~{1000 * (abs(j7) + abs(j8)):.0f} mm "
              f"tips' midpoint - parcel = ({rel[0]:+.1f}, {rel[1]:+.1f}, {rel[2]:+.1f}) cm turn={turn:.0f} "
              f"tilt={tilt:.0f} underside-belt={100 * (p.low[2] - BELT_TOP):+.1f} cm "
              f"parcel-gap={100 * (p.pos[1] - gap_y(t)):+.1f} cm", flush=True)


def order() -> list[str]:
    """The parcels in the order they reached the scan line (only those that did)."""
    return sorted((n for n in NAMES if _STATE[n]["front_t"] is not None), key=lambda n: _STATE[n]["front_t"])


def clear_between(ahead: str, behind: str) -> float:
    """The clear belt between two parcels (m): from the scan line to ``behind``'s leading face at the moment
    ``ahead``'s trailing face left the line (what a photo at that moment shows, or a photo-eye's dark time times the
    line's speed); negative if ``behind`` had already reached the line. Measured along the belt, so it doesn't depend
    on how fast the parcels ride."""
    return SCAN_Y - _STATE[ahead]["fronts"][behind]


def disturbed() -> list[str]:
    """Why the four other parcels count as disturbed (an empty list if they don't): the line's spacing at the scan line
    changed by more than 1.5 cm (6 cm between the first two and the last two, 24 cm from the second to the third),
    or one of them reached the line more than 1.5 cm across the belt from its centre line, or turned or tilted by
    more than 5 degrees."""
    out = []
    for a, b, was in (("parcel_1", "parcel_2", SPACING), ("parcel_2", "parcel_3", GAP), ("parcel_3", "parcel_4", SPACING)):
        now = clear_between(a, b)
        if abs(now - was) > KEPT:
            out.append(f"{a} to {b} {now * 100:.1f} cm apart (was {was * 100:.0f})")
    for n in TRAIN:
        rec = _STATE[n]
        if abs(rec["x"] - LINE_X) > KEPT:
            out.append(f"{n} {(rec['x'] - LINE_X) * 100:+.1f} cm across the belt")
        if max(rec["turn"], rec["tilt"]) > KEPT_DEG:
            out.append(f"{n} turned or tilted {max(rec['turn'], rec['tilt']):.0f} degrees")
    return out


def result() -> tuple[float, str]:
    """The score so far and why: 0 if an arm or the merged parcel touched any of the four parcels of the line at any
    time, or an arm touched the merged parcel during its pass (see :func:`watch_contacts`); 0 until every parcel has passed under the scanner lying on the belt in its
    lane; 0 if the four were disturbed (see :func:`disturbed`: what a touch between two control steps would leave);
    and 0 unless the merged parcel passed between parcel_2 and parcel_3 (in the gap). Then a third for each of: 4 cm
    or more clear belt ahead of it, 4 cm or more behind it, and square (turned and tilted 15 degrees or less)."""
    if _STATE["touch"] is not None:
        t, what = _STATE["touch"]
        return 0.0, f"{what} at {t:.2f} s"
    for n in NAMES:
        rec = _STATE[n]
        if rec["fault"] is not None:
            return 0.0, rec["fault"]
        if rec["rear_t"] is None:
            return 0.0, f"{n} has not passed the scanner"
    seq = order()
    i = seq.index(MERGE)
    if not 0 < i < len(seq) - 1 or (seq[i - 1], seq[i + 1]) != ("parcel_2", "parcel_3"):
        return 0.0, "the merged parcel did not pass in the gap: order " + ", ".join(seq)
    spacing = "/".join(f"{clear_between(a, b) * 100:.1f}" for a, b in zip(TRAIN, TRAIN[1:]))
    moved = disturbed()
    if moved:
        return 0.0, f"the other parcels were disturbed (line spacing {spacing} cm; " + "; ".join(moved) + ")"
    ahead, behind = clear_between("parcel_2", MERGE), clear_between(MERGE, "parcel_3")
    turn, tilt = _STATE[MERGE]["turn"], _STATE[MERGE]["tilt"]
    checks = (ahead >= CLEAR, behind >= CLEAR, max(turn, tilt) <= SQUARE_DEG)
    parts = (f"clear ahead {ahead * 100:.1f} cm", f"clear behind {behind * 100:.1f} cm",
             f"square (turned {turn:.0f}, tilted {tilt:.0f} degrees)")
    closest = _STATE["closest"]
    near = {k: "over 5 cm" if v == np.inf else f"{v * 100:.1f} cm" for k, v in closest.items()}
    why = "; ".join(("ok " if ok else "FAIL ") + part for ok, part in zip(checks, parts)) + (
        f"; no touch (closest to the line's parcels: arms {near['arms']}, merged parcel {near['merged']}; arms to "
        f"the merged parcel in its pass: {near['held']}; line spacing {spacing} cm)")
    return sum(checks) / 3, why


_SAID: dict = {}


def score(objects):
    """See :func:`result`. The record is kept by the monitor, which runs before every scoring. With MERGE_DEBUG=1 it
    prints the breakdown whenever it changes."""
    value, why = result()
    if DEBUG and _SAID.get("why") != why:
        _SAID["why"] = why
        print(f"[merge] score {value:.2f}: {why}", flush=True)
    return value


def success(objects):
    """The merged parcel passed the scanner in the gap, square, with 4 cm or more clear on both sides, nothing
    touched the other four and they passed undisturbed."""
    return result()[0] == 1.0


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop, timed against the belt. The left arm takes the parcel off the infeed table and lifts it 1.5 cm higher
# than the other parcels' tops. As the gap comes abreast it carries it over the belt while speeding up to the belt,
# ending over the gap and moving with it, follows the gap down slowly until the parcel's underside is at the belt,
# holds it there a moment, opens the fingers and keeps following the gap while they open, then comes back up still
# following it, so the parcel is set down moving with the belt. Fingers close across the belt, on the parcel's long
# side faces, never between two parcels: the gripper turned -90 degrees. The right arm stays in its ready pose.
DT = 0.05  # one control step at 20 Hz
ARM = "left"
YAW = float(-np.pi / 2)
GRASP_Z = PARCEL_Z + 0.003  # as deep as the 1.5 cm floor clearance allows: pad tips 1 cm below the parcel's centre
PRE_Z = PARCEL_TOP + PAD_TIP_PAST_TCP + 0.03  # the open pad tips 3 cm above the parcels' tops
CARRY_Z = GRASP_Z + PARCEL[2] + 0.015  # carried: its underside (and the pad tips) 1.5 cm or more over the others' tops
PLACE_Z = GRASP_Z  # set down: the parcel's underside commanded at the belt's top (the arm sags ~3 mm under load)
PARK = (0.16, 0.30, TOP + 0.15)
# Measured in Isaac Sim: the fingers take about 1 s to close from fully open (and open about as slowly), so the arm
# holds still 1.4 s after the close command before lifting, and follows the gap for 1 s after the open command before
# rising. Both finger drives saturate on the parcel, so the pair can slide across (x) as one: the parcel may end up
# held up to 1.7 cm either side of the TCP, which leaves it 3.3 cm or more inside the belt's edges.
AIM = 0.0  # set down on the gap's middle: 3.5 cm of slack either way, for parcels riding up to 7 % off the belt speed
OVER_S, DESCEND_S, CLOSE_S, GRIP_S, LIFT_S = 1.2, 0.7, 0.5, 1.4, 0.9
MOVE_T = 4.7  # when the carry over the belt starts (as soon as it is lifted): the gap's middle is 1.8 cm short of it
CARRY_S, LOWER_S, SETTLE_S, RELEASE_S, OPEN_S, UP_S = 1.6, 1.2, 0.3, 0.4, 1.0, 0.5
END_S = 21.0  # hold the park pose until the last parcel has passed and success has held its 1 s


def place_xy(t: float) -> tuple[float, float]:
    """Where the TCP goes at ``t`` to hold the parcel over its spot in the gap (AIM ahead of the gap's middle)."""
    return LINE_X, gap_y(t) + AIM


def _blend(t0: float, seconds: float, xy0, z0: float, z1: float, f0: float, f1: float, label: str) -> list[Waypoint]:
    """Waypoints one control step apart from ``t0``: the TCP goes from ``xy0`` (fixed) to :func:`place_xy`, moving
    with the gap (``xy0`` None: there throughout), from ``z0`` to ``z1``, the fingers from ``f0`` to ``f1`` (each
    smoothly, so it arrives moving with the belt)."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        t = t0 + k * DT
        goal = np.array(place_xy(t))
        xy = goal if xy0 is None else (1 - s) * np.asarray(xy0, float) + s * goal
        out.append(Waypoint(DT, _floats((*xy, z0 + s * (z1 - z0))), f0 + s * (f1 - f0), label, YAW))
    return out


def demo():
    """Left arm: take the parcel off the infeed table (down at 1.9 s, close 1.9 to 2.4 s, hold until 3.8 s, lifted by
    4.7 s), carry it over the gap from 4.7 s, follow the gap down until 7.5 s and hold the parcel on the belt until
    7.8 s, open the fingers at 7.8 to 8.2 s and keep following the gap until 9.2 s, then rise. The parcel is set down
    on the gap's middle: 7.5 cm of clear belt either side of it. The last parcel has passed at
    18.6 s; the plan holds until 21 s."""
    x, y = INFEED_XY
    plan = [Waypoint(OVER_S, (x, y, PRE_Z), FINGER_OPEN, "over the waiting parcel", YAW),
            Waypoint(DESCEND_S, (x, y, GRASP_Z), FINGER_OPEN, "down", YAW),
            Waypoint(CLOSE_S, (x, y, GRASP_Z), FINGER_CLOSED, "close", YAW),
            Waypoint(GRIP_S, (x, y, GRASP_Z), FINGER_CLOSED, "let the fingers close", YAW),
            Waypoint(LIFT_S, (x, y, CARRY_Z), FINGER_CLOSED, "lift", YAW)]
    t = sum(w.duration for w in plan)
    if MOVE_T - t > 1e-9:
        plan.append(Waypoint(MOVE_T - t, (x, y, CARRY_Z), FINGER_CLOSED, "wait for the gap", YAW))
    t = MOVE_T
    for seconds, xy0, z, f, label in (
        (CARRY_S, (x, y), (CARRY_Z, CARRY_Z), (FINGER_CLOSED, FINGER_CLOSED), "carry over the gap"),
        (LOWER_S, None, (CARRY_Z, PLACE_Z), (FINGER_CLOSED, FINGER_CLOSED), "set down"),
        (SETTLE_S, None, (PLACE_Z, PLACE_Z), (FINGER_CLOSED, FINGER_CLOSED), "hold it on the belt"),
        (RELEASE_S, None, (PLACE_Z, PLACE_Z), (FINGER_CLOSED, FINGER_OPEN), "let go"),
        (OPEN_S, None, (PLACE_Z, PLACE_Z), (FINGER_OPEN, FINGER_OPEN), "let the fingers open"),
        (UP_S, None, (PLACE_Z, PRE_Z), (FINGER_OPEN, FINGER_OPEN), "up"),
    ):
        plan += _blend(t, seconds, xy0, *z, *f, label)
        t += seconds
    plan.append(Waypoint(1.0, PARK, FINGER_OPEN, "clear of the belt", YAW))
    plan.append(Waypoint(END_S - t - 1.0, PARK, FINGER_OPEN, "wait while the parcels pass", YAW))
    return {ARM: plan}
