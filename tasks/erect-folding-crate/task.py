"""On the station table, between the arms: a folded green polypropylene folding produce crate (400 x 300 x 174 mm
when up, 0.88 kg), lying flat (7.8 cm tall), its long sides facing the robot. Its four walls are folded in: the two
short end walls lowest, the two long side walls on top. Each wall has a handhold slot under a round top rail. Each
long wall has an orange sprung catch near each end. The robot must put the crate up: raise the long walls, then swing
each short wall up until both its corner catches click, and let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from cratekit import Crate, on_reset, on_step, write_start_joints  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Set up the folded crate so it is ready to fill, then let go of it."

CRATE = Crate("crate", pos=(g.XC, g.YC, g.TOP), size=(g.WIDTH, g.LENGTH, g.HEIGHT), mass=g.MASS,
              color=g.CRATE_GREEN)
OBJECTS = [CRATE]

# -- judging ----------------------------------------------------------------------------------------------------------
LONG_CREDIT = 0.1        # without success: per long wall standing up on its own (within 3 degrees, no gripper near)
CORNER_CREDIT = 0.15     # ... plus per latched catch: at most 0.2 + 0.6 = 0.8
GRIP_R = 0.05            # m: each gripper is a capsule this thick from 10 cm behind its grasp point to its fingertips
CLEAR = 0.02             # m: released means both grippers at least this far from every part of the crate
STILL_S = 1.0            # s: still means over the last second ...
STILL_MOVE = 0.002       # ... the base moved less than 2 mm ...
STILL_TURN = math.radians(0.5)   # ... and no hinge turned more than 0.5 degree, and no catch moved 0.5 mm
DROP = 0.10              # violation: the base more than 10 cm below the table top (fallen off); lifting or tilting is fine
PAD_TIP = 0.013          # fingertips past the grasp point (yam_arena.constants.PAD_TIP_PAST_TCP)
DT = 0.05                # s per control step at 20 Hz; replaced by the environment's step_dt when known

_STATE: dict = {}


def reset_state() -> None:
    _STATE.update(violation=False, why=[], latched={p: False for p in g.PAWLS}, standing={w: False for w in g.LONG},
                  upright=False, flat=False, released=False, still=False, history=[], steps=0)


@on_reset
def _on_reset(environment=None) -> None:
    if environment is not None:
        write_start_joints(environment)
    reset_state()


reset_state()


def _rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def crate_boxes(pos, rot, joints, only: str | None = None) -> list:
    """Every part of the crate as oriented boxes (centre, rotation, half sizes), station frame. The rails are taken
    and hooks as their bounding boxes; the catches are inside the long walls' boxes plus 6 mm, so they are left out."""
    out = [] if only else [(pos + rot @ np.asarray(c), rot, np.asarray(s) / 2) for c, s, _, _ in g.base_boxes()]
    for name, wall in g.WALLS.items():
        if only and name != only:
            continue
        q = joints.get(g.hinge_joint(name), 0.0)
        frame = rot @ wall.frame(q)
        origin = pos + rot @ np.asarray(wall.hinge)
        boxes, caps = g.wall_shapes(wall)
        for c, s, _, _ in boxes:
            out.append((origin + frame @ np.asarray(c), frame, np.asarray(s) / 2))
        for c, r, length, _, axis in caps:
            half = np.array([length / 2 + r, r, r]) if axis == "X" else np.array([r, r, length / 2 + r])
            out.append((origin + frame @ np.asarray(c), frame, half))
    return out


def _dist_to_boxes(points, boxes) -> float:
    best = np.inf
    for c, r, half in boxes:
        local = (np.atleast_2d(points) - c) @ r
        best = min(best, float(np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=1).min()))
    return best


def gripper_segments(environment) -> list:
    """Each gripper's axis (from 10 cm behind its grasp point to its fingertips), from the measured arm joints."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    from cratekit import arm_joints

    segs = []
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        segs.append((tcp - 0.10 * rot[:, 2], tcp + PAD_TIP * rot[:, 2]))
    return segs


def update(pos, quat_xyzw, joints, segments, dt: float = DT) -> None:
    """One control step of the monitor: the crate's base pose and joints, and the grippers' axes (station frame)."""
    pos = np.asarray(pos, float)
    rot = _rot(quat_xyzw)
    _STATE["steps"] += 1

    # Whole-trial violation: the crate falls off the table. Lifting or tilting it on the way is allowed.
    tilt = math.acos(max(-1.0, min(1.0, rot[2, 2])))
    if pos[2] < g.TOP - DROP:
        _flag("crate fell off the table")

    _STATE["latched"] = {p: g.corner_latched(joints, p) for p in g.PAWLS}
    _STATE["upright"] = all(abs(joints.get(g.hinge_joint(w), 0.0) - g.UP) <= g.VERTICAL_TOL for w in g.WALLS)
    _STATE["flat"] = tilt <= g.FLAT_TOL
    boxes = crate_boxes(pos, rot, joints)
    axes = [a + np.outer(np.linspace(0, 1, 12), b - a) for a, b in segments]
    _STATE["released"] = all(_dist_to_boxes(ax, boxes) - GRIP_R >= CLEAR for ax in axes)
    for w in g.LONG:
        wall_boxes = crate_boxes(pos, rot, joints, only=w)
        _STATE["standing"][w] = bool(abs(joints.get(g.hinge_joint(w), 0.0) - g.UP) <= g.VERTICAL_TOL and all(
            _dist_to_boxes(ax, wall_boxes) - GRIP_R >= CLEAR for ax in axes))

    # Still: base origin and every joint over the last second.
    hist = _STATE["history"]
    hist.append((pos.copy(), dict(joints)))
    n = max(1, round(STILL_S / dt)) + 1
    del hist[:-n]
    still = len(hist) == n
    if still:
        p0, j0 = hist[0]
        still = bool(max(np.linalg.norm(p - p0) for p, _ in hist) < STILL_MOVE)
        for name in j0:
            lim = STILL_TURN if name.endswith("_hinge") else 0.0005
            still = still and all(abs(j.get(name, 0.0) - j0[name]) < lim for _, j in hist)
    _STATE["still"] = still


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    global DT
    DT = float(getattr(environment.unwrapped, "step_dt", DT))
    c = object_states(environment, ["crate"])["crate"]
    update(c.pos, c.quat_xyzw, c.joints, gripper_segments(environment), DT)


def success(objects) -> bool:
    """All four catches latched, every wall within 3 degrees of vertical, the base within 3 degrees of flat, both
    grippers at least 2 cm from the crate and the crate still over the last second, with no violation; held 1 s."""
    return bool(not _STATE["violation"] and all(_STATE["latched"].values()) and _STATE["upright"]
                and _STATE["flat"] and _STATE["released"] and _STATE["still"])


def score(objects) -> float:
    """Without success: 0 after a violation (the crate fell off the table). Otherwise, at the end, 0.1 for each long wall standing up on its own (within 3 degrees of vertical,
    no gripper within 2 cm of it) plus 0.15 for each latched catch: at most 0.8."""
    if _STATE["violation"]:
        return 0.0
    return LONG_CREDIT * sum(_STATE["standing"].values()) + CORNER_CREDIT * sum(_STATE["latched"].values())
