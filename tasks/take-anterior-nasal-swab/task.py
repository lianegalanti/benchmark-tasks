"""On the station table, ahead of the robot: a life-size head-and-shoulders patient manikin facing the robot, its head
on a sprung neck, its lower nose soft. Its nostrils are 21 cm above the table. To the robot's right, a swab transport
tube stands in a small rack; the sterile foam-tipped anterior nasal swab is fixed in the tube's cap, which is its
handle.

The robot must take the swab out of its tube, put the whole foam tip 1.5 to 2.2 cm into one nostril and rub it round
the nostril's wall in four full circles with light pressure (0.3 to 2 N), then do the same in the other nostril, and
put the swab back in its tube with the cap seated. The tip must touch nothing but the linings of the nostrils (and the
inside of its own tube), and the swab may be let go of only in its tube.

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
CONTROL_HZ = 20

INSTRUCTION = ("Take an anterior nasal swab from the manikin: four circles against the wall of each nostril, "
               "then put the swab back in its tube and let go.")

# -- objects -------------------------------------------------------------------------------------------------------
MANIKIN = UsdBody("manikin", usd="assets/manikin.usda", pos=tuple(float(v) for v in g.M0), size=(0.2, 0.4, 0.38),
                  articulation=True,
                  joint_pos=(("neck_pitch", 0.0), ("neck_yaw", 0.0), ("nose_f", 0.0), ("nose_y", 0.0)),
                  drives=(("neck_.*", g.NECK_K, g.NECK_D, g.NECK_CAP), ("nose_.*", g.NOSE_K, g.NOSE_D, 20.0)),
                  usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001)
SWAB = UsdBody("swab", usd="assets/swab.usda", pos=tuple(float(v) for v in g.SWAB_START),
               rot_xyzw=tuple(float(v) for v in g.quat_xyzw(g.R_SWAB0)), size=(g.SWAB_LEN, 0.018, 0.020),
               mass=g.SWAB_MASS, usd_materials=True, torsional_patch=0.008, contact_offset=0.001, solver_iters=32)
TUBE = UsdBody("tube", usd="assets/tube.usda", pos=tuple(float(v) for v in g.TUBE_POS), size=(0.05, 0.05, 0.153),
               mass=None, usd_materials=True, contact_offset=0.001)
OBJECTS = [MANIKIN, SWAB, TUBE]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
OFF_TABLE = 0.05            # m: the swab's centre this far below the table top: it fell off
CLEAR = 0.02                # m: let go: every part of both grippers this far from the swab, tube and manikin
STILL_V = 0.01              # m/s: the swab still ...
STILL_NECK = math.radians(0.1)   # ... and the head turning less than this per control step (2 deg/s)
LET_GO = 0.01               # m: both grippers this far from the swab: it has been let go of
PER_NOSTRIL = 0.35
CAPPED = 0.2
RELEASED = 0.1

_STATE: dict = {}


def gripper_boxes(tcp, rot, opening: float):
    """The gripper as boxes (centre, rotation, half sizes), station frame: each finger's pad (from 3 cm behind the TCP
    to its tip 13 mm past it), each finger's root and the housing. The fingers close along the TCP frame's x."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    half = opening / 2
    local = []
    for s in (-1.0, 1.0):
        local.append((np.array([s * (half + 0.005), 0.0, -0.0085]), np.array([0.005, 0.020, 0.0215])))
        local.append((np.array([s * (half + 0.010), 0.0, -0.045]), np.array([0.010, 0.034, 0.015])))
    local.append((np.array([0.0, 0.0, -0.1025]), np.array([max(0.051, half + 0.024), 0.034, 0.0425])))
    return [(tcp + rot @ c, rot, h) for c, h in local]


# -- the patient's parts in the station frame, from measured link poses ---------------------------------------------
class Pose:
    """Measured poses: the manikin's root, head and lower nose, the tube; joint values."""

    def __init__(self, m_pos, m_rot, head_pos, head_rot, nose_pos, nose_rot, tube_pos, tube_rot, joints):
        self.m_pos, self.m_rot = np.asarray(m_pos, float), np.asarray(m_rot, float)
        self.h_pos, self.h_rot = np.asarray(head_pos, float), np.asarray(head_rot, float)
        self.n_pos, self.n_rot = np.asarray(nose_pos, float), np.asarray(nose_rot, float)
        self.t_pos, self.t_rot = np.asarray(tube_pos, float), np.asarray(tube_rot, float)
        self.joints = dict(joints)


def pose_from_joints(pitch=0.0, yaw=0.0, f=0.0, y=0.0) -> Pose:
    """The poses a straight-standing manikin has with these joint values (for the tests and the demo's plan)."""
    hp, hr = g.head_pose(pitch, yaw)
    npos, nrot = g.nose_pose(hp, hr, f, y)
    return Pose(g.M0, np.eye(3), hp, hr, npos, nrot, g.TUBE_POS, np.eye(3),
                {"neck_pitch": pitch, "neck_yaw": yaw, "nose_f": f, "nose_y": y})


def tip_in_nostril(p: Pose, end) -> tuple[str | None, float]:
    """Which nostril the tip's end is in (its centre inside the lumen's ellipse, between 3 mm below the opening and
    35 mm up), and its depth past the opening along the nostril's axis."""
    local = p.n_rot.T @ (np.asarray(end, float) - p.n_pos)
    for side, yc in g.NOSTRIL_Y.items():
        rho = math.hypot(local[2] / g.SEMI_F, (local[1] - yc) / g.SEMI_Y)
        if rho <= 1.0 and -0.003 <= local[0] <= 0.035:
            return side, float(local[0])
    return None, float(local[0])


def in_lumen(p: Pose, pts) -> np.ndarray:
    """Which points (station frame) lie inside a nostril: within its oval and between its opening and its top."""
    local = (np.atleast_2d(pts) - p.n_pos) @ p.n_rot
    inside = np.zeros(len(local), bool)
    for yc in g.NOSTRIL_Y.values():
        rho = np.hypot(local[:, 2] / g.SEMI_F, (local[:, 1] - yc) / g.SEMI_Y)
        inside |= (rho <= 1.0) & (local[:, 0] >= 0.0) & (local[:, 0] <= g.LUMEN_DEPTH)
    return inside


def tip_in_bore(p: Pose, end) -> bool:
    local = p.t_rot.T @ (np.asarray(end, float) - p.t_pos)
    return bool(math.hypot(local[0], local[1]) < g.TUBE_R_IN and -g.TUBE_LEN < local[2] < 0.005)


def tip_gaps(p: Pose, swab_pos, swab_rot, grips) -> dict:
    """Gaps (m) from the foam tip to everything it must not touch: everything but the nostrils' linings (and its own
    tube's bore)."""
    a, b = g.tip_segment(swab_pos, swab_rot)
    pts = g.seg_points(a, b, 7)
    r = g.TIP[0]
    out = {"table": float(pts[:, 2].min() - r - g.TABLE)}
    # the patient's skin: only from the parts of the tip outside the nostrils (inside one, the lining encloses it)
    skin = pts[~in_lumen(p, pts)]
    if len(skin) == 0:
        out.update(face=1.0, lips=1.0, nose=1.0)
    else:
        out["face"] = float(g.hull_dist((skin - p.h_pos) @ p.h_rot).min() - r)
        lips = []
        for _, c, rl, length in g.LIPS:
            cw = p.h_pos + p.h_rot @ c
            ax = p.h_rot @ np.array([0.0, length / 2, 0.0])
            lips.append(g.point_seg_dist(skin, cw - ax, cw + ax).min() - rl - r)
        out["lips"] = float(min(lips))
        out["nose"] = float(min(g.point_box_dist(skin, p.n_pos + p.n_rot @ c, p.n_rot, s / 2).min()
                                for _, c, s in g.nose_outer_boxes()) - r)
    body = [g.point_box_dist(pts, p.m_pos + p.m_rot @ c, p.m_rot, s / 2).min() for _, c, s in g.bust_boxes()]
    nc = p.m_pos + p.m_rot @ (g.NECK_C + [0.0, 0.0, sum(g.NECK_Z) / 2])
    body.append(g.point_cyl_dist(pts, nc, p.m_rot @ g.AXES["z"], g.NECK_R, (g.NECK_Z[1] - g.NECK_Z[0]) / 2).min())
    sc = p.m_pos + p.m_rot @ g.SHOULDER_C
    sax = p.m_rot @ np.array([0.0, g.SHOULDER_LEN / 2, 0.0])
    body.append(g.point_seg_dist(pts, sc - sax, sc + sax).min() - g.SHOULDER_R)
    out["body"] = float(min(body) - r)
    rack = [g.point_box_dist(pts, p.t_pos + p.t_rot @ c, p.t_rot, s / 2).min() for _, c, s in g.rack_boxes()]
    if not tip_in_bore(p, b + (b - a) / np.linalg.norm(b - a) * r):
        tc = p.t_pos + p.t_rot @ np.array([0.0, 0.0, -g.TUBE_LEN / 2])
        rack.append(g.point_cyl_dist(pts, tc, p.t_rot @ g.AXES["z"], g.TUBE_R_OUT, g.TUBE_LEN / 2).min())
    out["tube"] = float(min(rack) - r)
    out["grippers"] = float(min(g.point_box_dist(pts, *bx).min() for grip in grips for bx in grip) - r)
    return out


def seated(p: Pose, swab_pos, swab_rot) -> bool:
    """The swab in its tube with the cap seated: the flange's underside within 2 mm of the rim and 3 mm of the
    tube's axis, the swab within 8 degrees of the axis, tip down."""
    rot = np.asarray(swab_rot, float)
    flange = np.asarray(swab_pos, float) + rot @ np.array([g.FLANGE[2], 0.0, 0.0])
    local = p.t_rot.T @ (flange - p.t_pos)
    axis = p.t_rot.T @ rot[:, 0]
    tilt = math.acos(max(-1.0, min(1.0, -float(axis[2]))))
    return bool(-0.001 <= local[2] <= g.SEAT_TOL and math.hypot(local[0], local[1]) <= g.SEAT_OFF
                and tilt <= g.SEAT_TILT)


def release_gaps(p: Pose, swab_pos, swab_rot, grips) -> list[float]:
    """Each gripper's gap (m) to the swab, the tube and rack, and the manikin's head and body."""
    out = []
    sb = g.swab_boxes(swab_pos, swab_rot)
    tc = p.t_pos + p.t_rot @ np.array([0.0, 0.0, -(g.TUBE_LEN + 0.003) / 2])
    for grip in grips:
        pts = np.vstack([g.box_points(*bx, 4) for bx in grip])
        gaps = [min(g.boxes_gap(bx, s, 4) for bx in grip for s in sb),
                float(g.point_cyl_dist(pts, tc, p.t_rot @ g.AXES["z"], g.RACK[0] / 2 * 1.42,
                                       (g.TUBE_LEN + 0.003) / 2).min()),
                float(g.ellipsoid_dist(pts, p.h_pos + p.h_rot @ g.HEAD_C, p.h_rot, g.HEAD_AX).min()),
                float(min(g.point_box_dist(pts, p.n_pos + p.n_rot @ c, p.n_rot, s / 2).min()
                          for _, c, s in g.nose_outer_boxes()))]
        out.append(min(gaps))
    return out


# -- the monitor -----------------------------------------------------------------------------------------------------
def reset_state() -> None:
    _STATE.clear()
    _STATE.update(step=0, why=[], violation=False, tare=None, circles={"left": 0, "right": 0},
                  turned={"left": 0.0, "right": 0.0}, last=None, used=False, seated=False, released=False,
                  still=False, level=0.0, side=None, depth=None, disp_mm=0.0, on_wall=None, gaps=None, pitch=0.0,
                  yaw=0.0, max_neck_deg=0.0, max_depth=0.0)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def wall_gap(p: Pose, side: str, swab_pos, swab_rot, push) -> float:
    """Gap (m) between the foam tip's deep end and nostril ``side``'s lining, on the side the nose is pushed toward
    (``push``: unit (f, y) direction): how far the tip is from the wall it should be pressing."""
    _, b = g.tip_segment(swab_pos, swab_rot)
    local = p.n_rot.T @ (b - p.n_pos)
    o = np.array([local[2], local[1] - g.NOSTRIL_Y[side]])
    reach = math.hypot(g.SEMI_F * push[0], g.SEMI_Y * push[1])      # the oval's extent in that direction
    return float(reach - g.TIP[0] - o @ push)


def level(circles: dict, capped: bool, released: bool, still: bool) -> float:
    """Score from the circles counted in each nostril (0-4), the swab back in its tube with the cap seated, and both
    hands let go."""
    n = sum(min(g.CIRCLES, c) for c in circles.values())
    s = PER_NOSTRIL * n / g.CIRCLES
    if capped and n >= 1:
        s += CAPPED + (RELEASED if released else 0.0)
    if all(c >= g.CIRCLES for c in circles.values()) and capped and released and still:
        return 1.0
    return round(min(s, 0.9), 4)


def update(p: Pose, swab_pos, swab_pos_vel, swab_rot, grips) -> None:
    """One control step of the monitor, on measured values (station frame). ``grips``: each gripper as boxes."""
    S = _STATE
    S["step"] += 1
    sp, sr = np.asarray(swab_pos, float), np.asarray(swab_rot, float)
    jf, jy = float(p.joints.get("nose_f", 0.0)), float(p.joints.get("nose_y", 0.0))
    if S["tare"] is None:
        S["tare"] = (jf, jy)                    # zero the soft nose's gauge with nothing touching it
    df, dy = jf - S["tare"][0], jy - S["tare"][1]
    disp = math.hypot(df, dy)
    S["disp_mm"] = round(disp * 1000, 2)
    pitch, yaw = float(p.joints.get("neck_pitch", 0.0)), float(p.joints.get("neck_yaw", 0.0))
    neck = math.hypot(pitch, yaw)
    S["max_neck_deg"] = max(S["max_neck_deg"], math.degrees(neck))
    if neck > g.HEAD_PUSHED:
        _flag("the patient's head was pushed more than 15 degrees")

    end = g.tip_end(sp, sr)
    side, depth = tip_in_nostril(p, end)
    S["side"], S["depth"] = side, round(depth * 1000, 1) if side else None
    local_t = p.t_rot.T @ (end - p.t_pos)
    if local_t[2] > 0.005:
        S["used"] = True                        # the tip has been out of its tube
    if sp[2] < g.TABLE - OFF_TABLE:
        _flag("the swab fell off the table")
    S["seated"] = seated(p, sp, sr)

    # the tip: nothing but the linings (until it is back in its tube with the cap seated)
    gaps = tip_gaps(p, sp, sr, grips)
    S["gaps"] = {k: round(v * 1000, 1) for k, v in gaps.items()}
    if not S["seated"]:
        for k, v in gaps.items():
            if v < g.TOUCH:
                _flag(f"the swab's tip touched something other than a nostril's lining ({k})")

    # circles: the push on the nose turning round, with the tip on the lining it pushes, deep enough, in the band
    S["on_wall"] = None
    if side is not None:
        S["max_depth"] = max(S["max_depth"], depth)
        if depth > g.TOO_DEEP:
            _flag("the swab went more than 2.5 cm up the nostril")
        band = g.DEPTH_BAND[0] <= depth <= g.DEPTH_BAND[1] and g.DISP_BAND[0] <= disp <= g.DISP_BAND[1]
        if band:
            push = np.array([df, dy]) / disp
            gap = wall_gap(p, side, sp, sr, push)
            S["on_wall"] = round(gap * 1000, 2)
            band = gap <= g.ON_WALL
        if band:
            ang = math.atan2(dy, df)
            if S["last"] is not None and S["last"][0] == side:
                d = (ang - S["last"][1] + math.pi) % (2 * math.pi) - math.pi
                if abs(d) < math.pi / 2:
                    S["turned"][side] += d
            S["last"] = (side, ang)
            S["circles"][side] = max(S["circles"][side],
                                     min(g.CIRCLES, int(abs(S["turned"][side]) // (2 * math.pi))))
        else:
            S["last"] = None
    else:
        S["last"] = None

    # let go of only in its tube
    sb = g.swab_boxes(sp, sr)
    hold = min(g.boxes_gap(bx, s, 4) for grip in grips for bx in grip for s in sb)
    if S["used"] and not S["seated"] and hold >= LET_GO:
        _flag("the swab was let go of outside its tube")
    S["released"] = all(x >= CLEAR for x in release_gaps(p, sp, sr, grips))
    still_neck = abs(pitch - S["pitch"]) < STILL_NECK and abs(yaw - S["yaw"]) < STILL_NECK
    S["still"] = bool(float(np.linalg.norm(swab_pos_vel)) < STILL_V and still_neck)
    S["pitch"], S["yaw"] = pitch, yaw
    S["level"] = 0.0 if S["violation"] else level(S["circles"], S["seated"] and S["used"], S["released"],
                                                  S["still"])


def _grips(environment):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    from scenekit import arm_joints

    out = []
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        opening = float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())
        out.append(gripper_boxes(tcp, rot, opening))
    return out


@on_reset
def _reset(environment=None) -> None:
    reset_state()


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["manikin", "swab", "tube"])
    m, sw, tb = st["manikin"], st["swab"], st["tube"]
    hp, hq = m.links["head"]
    npos, nq = m.links["nose"]
    p = Pose(m.pos, g.rot_from_xyzw(m.quat_xyzw), hp, g.rot_from_xyzw(hq), npos, g.rot_from_xyzw(nq), tb.pos,
             g.rot_from_xyzw(tb.quat_xyzw), m.joints)
    update(p, sw.pos, sw.lin_vel, g.rot_from_xyzw(sw.quat_xyzw), _grips(environment))


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """Four counted circles in each nostril, the swab back in its tube with the cap seated, both grippers 2 cm from
    the swab, the tube and the manikin, the swab and the head still; no violation. Held for 1 s."""
    return bool(not violation() and _STATE["level"] >= 1.0)


def score(objects) -> float:
    """0 after any violation (the tip touched anything but a nostril's lining before the swab was back in its tube
    with the cap seated; the tip more than 2.5 cm up a nostril; the head pushed more than 15 degrees; the swab let go
    of anywhere but seated in its tube, or off the table). Otherwise 0.35 x counted circles / 4 for each nostril (up
    to 4 each), plus 0.2 for the used swab back in its tube with the cap seated, plus 0.1 if then both grippers have let
    go. Success scores 1."""
    return 0.0 if violation() else min(0.9, float(_STATE["level"]))
