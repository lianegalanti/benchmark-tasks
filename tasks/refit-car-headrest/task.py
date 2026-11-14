"""On the station table, straight ahead of the arms: the top of a car seat back (cropped), fixed to the table, with
the two guide sleeves of its headrest. The right sleeve's bezel carries the sprung release button on top. The headrest
(260 x 170 x 105 mm cushion, two 14 mm steel posts with four notches each) leans against the seat back, its post tips
on the table, as left after it was taken out.

The robot must put the headrest back, locked at its lowest position: line both posts up with both sleeves and push
them in (the pawl clicks into the first notch and stops it there), then hold the button down while pushing the
headrest down to the bottom, and let go of the button so the pawl drops into the lowest notch; then let go of the
headrest.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Put the headrest back into the seat, locked at its lowest position, then let go of it."

# -- objects -------------------------------------------------------------------------------------------------------
_O0, _R0 = g.start_pose()
HEADREST = UsdBody("headrest", usd="assets/headrest.usda", pos=tuple(float(v) for v in g.to_station(_O0)),
                   rot_xyzw=g.quat_xyzw(_R0), size=(2 * g.CUSH_T, 2 * g.CUSH_W, g.POST_LEN + g.CUSH_H), mass=g.MASS,
                   usd_materials=True, solver_iters=32, contact_offset=0.001, torsional_patch=0.01)
SEAT = UsdBody("seat", usd="assets/seat.usda", pos=tuple(float(v) for v in g.SEAT), size=(0.10, 0.44, 0.105),
               articulation=True, joint_pos=(("lock", 0.0),),
               drives=(("lock", g.LOCK_RATE, g.LOCK_DAMPING),), drive_targets=(("lock", g.LOCK_TARGET),),
               usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001)
# the release button: its own fixed-base articulation in the seat's frame (so that it collides with the seat's latch)
BUTTON = UsdBody("button", usd="assets/button.usda", pos=tuple(float(v) for v in g.SEAT), size=(0.014, 0.016, 0.03),
                 articulation=True, joint_pos=(("button", 0.0),),
                 drives=(("button", g.BUTTON_RATE, g.BUTTON_DAMPING),), drive_targets=(("button", g.BUTTON_TARGET),),
                 usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001)
OBJECTS = [HEADREST, SEAT, BUTTON]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
ENTERED = 0.020                 # m: 0.25: both post tips at least this far down their sleeves ...
IN_BORE = g.BORE + 0.001        # m: ... with their axes within this of the sleeves' axes
ENGAGED = 0.0015                # m: the lock's pawl in (button out) to within this of its rest stop ...
NOTCH_TOL = g.WINDOW + 0.001    # m: ... with the right post's tips within this of a notch's depth (pawl in the notch)
UPRIGHT = math.radians(3)       # fitted: the posts within 3 degrees of the sleeves' axes
ENGAGED_TILT = math.radians(6)  # 0.5
CLEAR = 0.02                    # m: every part of both grippers this far from the headrest (let go)
STILL_V, STILL_W = 0.01, 0.1    # m/s and rad/s: the headrest still
STILL_LOCK = 0.0005             # m per control step (1 cm/s): the lock still
OFF_TABLE = 0.02                # m: any corner of the headrest this far below the table top: it left the table
FORCED = 0.004                  # m: any part of the headrest this far inside the seat: forced
PAWL_FORCED = 0.0015            # m: any part of a post this far inside the pawl (as measured), or the right post
                                #    carried this far past a notch's flat face in one step with the button out at both
                                #    ends of the step: forced past the pawl
BUTTON_OUT = 0.002              # m: the button pressed less than this: out (the pawl still 1.3 mm or more in a notch)
FALL_V = 0.5                    # m/s: the headrest's centre falling this fast with no gripper within CLEAR: dropped

_STATE: dict = {}


# -- the grippers as boxes, from each arm's encoders --------------------------------------------------------------------
def gripper_boxes(tcp, rot, opening: float):
    """The gripper as boxes (centre, rotation, half sizes) in the station frame: each finger's thin pad (from 3 cm
    behind the TCP to its tip 13 mm past it), each finger's thicker root and the finger carriage / housing. The
    fingers close along the TCP frame's x axis; ``opening`` is the gap between the pads."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    half = opening / 2
    local = []
    for s in (-1.0, 1.0):
        local.append((np.array([s * (half + 0.005), 0.0, -0.0085]), np.array([0.005, 0.020, 0.0215])))
        local.append((np.array([s * (half + 0.010), 0.0, -0.045]), np.array([0.010, 0.020, 0.015])))
    local.append((np.array([0.0, 0.0, -0.1025]), np.array([max(0.051, half + 0.024), 0.034, 0.0425])))
    return [(tcp + rot @ c, rot, h) for c, h in local]


def _rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _arms(environment):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    from scenekit import arm_joints

    out = {}
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        opening = float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())
        out[side] = {"tcp": tcp, "rot": rot, "opening": opening}
    return out


def _radius(box) -> float:
    return float(np.linalg.norm(box[2]))


def gap_to(boxes_a, boxes_b, far: float = 0.05) -> float:
    """Smallest gap between two sets of boxes (pairs whose bounding spheres are ``far`` apart are skipped)."""
    best = far
    for a in boxes_a:
        for b in boxes_b:
            d = float(np.linalg.norm(np.asarray(a[0]) - np.asarray(b[0]))) - _radius(a) - _radius(b)
            if d < best:
                best = min(best, g.boxes_gap(a, b))
    return best


# -- the headrest in the seat's frame -----------------------------------------------------------------------------------
def tilt(r) -> float:
    """Angle between the posts and the sleeves' axes (F's z)."""
    return math.acos(min(1.0, max(-1.0, float(np.asarray(r, float)[2, 2]))))


def posts_in(o, r) -> bool:
    """Both post tips at least ENTERED down their sleeves, their axes within IN_BORE of the sleeves' axes."""
    for side, sign in (("left", 1.0), ("right", -1.0)):
        tip = np.asarray(o) + np.asarray(r) @ g.TIPS_H[side]
        if -tip[2] < ENTERED or math.hypot(float(tip[0]), float(tip[1]) - sign * g.POST_Y) > IN_BORE:
            return False
    return True


def notch_at(o, r) -> int | None:
    """The notch (0 = first/highest .. 3 = lowest) level with the pawl, from the right post's depth, or None."""
    d = g.depth_of(o, r, "right")
    for k, dk in enumerate(g.DEPTHS):
        if abs(d - dk) <= NOTCH_TOL:
            return k
    return None


def stage(o, r, q) -> tuple[bool, int | None, bool]:
    """(entered, notch the pawl is engaged in or None, fitted at the lowest notch) for the headrest pose (o, R) in F
    and the lock's joint q (m, positive = button pressed in)."""
    inside = posts_in(o, r)
    t = tilt(r)
    k = notch_at(o, r) if inside and q <= ENGAGED and t <= ENGAGED_TILT else None
    fitted = k == len(g.DEPTHS) - 1 and t <= UPRIGHT
    return inside, k, fitted


def level_of(inside: bool, k, fitted: bool) -> float:
    return 0.75 if fitted else 0.5 if k is not None else 0.25 if inside else 0.0


def forced_depth(o, r, q: float) -> tuple[float, float]:
    """How far any part of the headrest lies inside the seat, and any part of a post inside the pawl (the deepest
    sample points)."""
    fixed = list(g.SEAT_F.values())
    hb = g.headrest_boxes(o, r)
    pawl = g.lock_boxes_f(q)["pawl"]
    into_pawl = max(float(g.point_box_depth(g.box_points(*b, 9), *pawl).max()) for k, b in hb.items()
                    if k.startswith("post_right"))
    deepest = 0.0
    for box in hb.values():
        rb = _radius(box)
        pts = None
        for fb in fixed:
            if float(np.linalg.norm(np.asarray(box[0]) - np.asarray(fb[0]))) > rb + _radius(fb):
                continue
            if pts is None:
                pts = g.box_points(*box, 7)
            deepest = max(deepest, float(g.point_box_depth(pts, *fb).max()))
    return deepest, into_pawl


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, level=0.0, fitted=False, clear=False, still=False, q=0.0, qb=0.0,
                  prev=None, info={})


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def forced_past(prev, now) -> bool:
    """Between two steps (each: right post's depth, its tip within its bore, the button's travel), the right post went
    PAWL_FORCED or more past a notch's flat face (its locked depth) while the button was out at both ends: the pawl
    can only have been forced, whatever happened between the samples."""
    if prev is None:
        return False
    (d0, in0, b0), (d1, in1, b1) = prev, now
    if not (in0 and in1 and b0 < BUTTON_OUT and b1 < BUTTON_OUT):
        return False
    return any(d0 <= lk + 0.0005 and d1 > lk + PAWL_FORCED for lk in g.LOCKED)


def update(pos, rot, q: float, vel, w, arms: dict, qb: float = 0.0) -> None:
    """One control step of the monitor. ``pos``/``rot``: the headrest frame's origin and rotation in the station
    frame; ``q``: the latch's joint (m, positive = drawn back); ``qb``: the button's (m, positive = pressed down);
    ``arms``: side -> {"tcp", "rot", "opening"} from the arm's encoders. Everything is judged in the seat's frame (it
    is fixed to the table)."""
    S = _STATE
    o = np.asarray(pos, float) - g.SEAT
    r = np.asarray(rot, float)
    q, qb = float(q), float(qb)

    boxes = list(g.headrest_boxes(np.asarray(pos, float), r).values())
    gaps = [gap_to(gripper_boxes(a["tcp"], a["rot"], a["opening"]), boxes) for a in arms.values()]
    clear = all(d >= CLEAR for d in gaps)

    # -- violations (latched for the whole trial) ---------------------------------------------------------------
    low = min(float(g.box_corners(*b)[:, 2].min()) for b in g.headrest_boxes(o, r).values())
    if low < -g.C - OFF_TABLE:
        _flag("the headrest fell off the table")
    depth, into_pawl = forced_depth(o, r, q)
    tip = o + r @ g.TIPS_H["right"]
    now = (g.depth_of(o, r, "right"), math.hypot(float(tip[0]), float(tip[1]) + g.POST_Y) <= IN_BORE, qb)
    if depth > FORCED or into_pawl > PAWL_FORCED or forced_past(S["prev"], now):
        _flag("the headrest was forced into the seat or past the pawl")
    S["prev"] = now
    # dropped: the centre of mass falling faster than FALL_V (a free fall of about 1.3 cm) while neither gripper is
    # within CLEAR of the headrest; carrying it down or setting it down gently is fine
    com_v = np.asarray(vel, float) + np.cross(np.asarray(w, float), r @ g.COM_H)
    if clear and com_v[2] < -FALL_V:
        _flag("the headrest was dropped")

    # -- progress ------------------------------------------------------------------------------------------------
    inside, k, fitted = stage(o, r, q)
    lvl = level_of(inside, k, fitted)
    still = (float(np.linalg.norm(vel)) < STILL_V and float(np.linalg.norm(w)) < STILL_W
             and abs(q - S["q"]) < STILL_LOCK)
    S.update(level=lvl, fitted=fitted, clear=clear, still=still, q=q, qb=qb,
             info={"depth_mm": [round(g.depth_of(o, r, s) * 1000, 1) for s in ("left", "right")],
                   "lock_mm": round(q * 1000, 2), "button_mm": round(qb * 1000, 2), "notch": k, "tilt_deg": round(math.degrees(tilt(r)), 2),
                   "gaps_mm": [round(d * 1000) for d in gaps], "forced_mm": [round(depth * 1000, 1), round(into_pawl * 1000, 1)],
                   "o_mm": [round(float(v) * 1000, 1) for v in o],
                   "axes": [round(float(v), 3) for v in (r[0, 2], r[1, 2], r[1, 0])]})


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["headrest", "seat", "button"])
    h = st["headrest"]
    update(h.pos, _rot(h.quat_xyzw), st["seat"].joints.get("lock", 0.0), h.lin_vel, h.ang_vel, _arms(environment),
           st["button"].joints.get("button", 0.0))


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """No violation so far; the headrest fitted (both posts in their sleeves, upright within 3 degrees, the pawl in
    the lowest notch with the button out); every part of both grippers at least 2 cm from the headrest; the headrest
    and the lock still. Held for 1 s."""
    S = _STATE
    return bool(not violation() and S["fitted"] and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: the headrest left the table (a corner 2 cm below its top), was dropped (its centre
    falling faster than 0.5 m/s with no gripper within 2 cm), was forced into the seat (4 mm) or a post was forced past
    the pawl (1.5 mm past a notch's flat face with the button out). Otherwise the stage at the end: 0.25 both posts
    20 mm or more down their sleeves; 0.5 the pawl engaged in a notch (post tips within 2.5 mm of its depth: 50, 65,
    80 or 95 mm); 0.75 locked at the lowest notch (95 +/- 2.5 mm), upright, but still touched or held. Success (locked
    at the lowest notch, released, still, held 1 s) scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
