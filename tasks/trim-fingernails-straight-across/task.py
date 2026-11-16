"""On the station table: a resident's right hand lies palm down on a foam hand rest, reaching in from the robot's far
left, fingers spread and pointing toward the robot's right arm, the fingers held out past the rest's edge. The index
and middle fingernails are long: 3-4 mm of white free edge, rounded. A lever nail clipper (58 mm, 25 g) stands jaws
down in the slot of a nail-care caddy in front of the right arm, its lever already swung round into cutting position.

The robot must trim both nails straight across, leaving 1-2 mm of white edge on every part of each nail, without the
jaws ever closing on the skin at the quick, holding each finger steady while it cuts (the knuckles are relaxed and
springy: 0.7 N at the nail turns a finger 10 degrees); then put the clipper down and let go.

Cutting is the clipper's own mechanism: the gripper squeezes the lever's end toward the body; the lever rocks on the
pin's head and its heel pushes the upper arm down until the jaws meet (a gear joint ties the upper arm's hinge to the
lever's). Each time the jaws close on the nail (their measured opening at most the nail's thickness plus 0.3 mm, with at
least 6 N on the lever's pivot, for two control steps, after opening at least 1.5 mm), the free-edge strips the cutting
line crosses are cut back to it, where the nail lies inside the shut jaws' blades (from 2 mm below the line to 1.5 mm
above it); nothing else cuts. A finger counts as steadied only if the other hand holds it at every cut.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import Shape, UsdBody, link_force, on_reset, on_step, set_targets  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Trim the resident's index and middle fingernails straight across with the nail clipper, leaving 1 to 2 mm "
               "of white nail, without nicking the skin and keeping each finger steady; then put the clipper down and "
               "let go of everything.")

BIN_JOINTS = {n: [f"{n}_b{b}" for b in range(g.N_BINS)] for n in g.NAILS}

# -- objects -------------------------------------------------------------------------------------------------------
HAND = UsdBody("hand", usd="assets/hand.usda", pos=tuple(float(v) for v in g.HAND0), size=(0.30, 0.15, 0.06),
               articulation=True,
               joint_pos=tuple((j, 0.0) for j in ("index", "middle", *BIN_JOINTS["index"], *BIN_JOINTS["middle"])),
               drives=(("(index|middle)", float(g.FINGER_K), float(g.FINGER_D)),
                       (".*_b[0-9]", float(g.BIN_K), float(g.BIN_D), 100.0)),
               usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.0005,
               disable_gravity=True)
CLIPPER = UsdBody("clipper", usd="assets/clipper.usda", pos=tuple(float(v) for v in g.CLIPPER_START),
                  rot_xyzw=tuple(float(v) for v in g.quat_xyzw(g.R_STAND)), size=(0.06, 0.014, 0.026),
                  articulation=True, joint_pos=(("jaw", 0.0), ("lever", 0.0)),
                  drives=(("jaw", 0.0, float(g.JAW_D)),
                          ("lever", float(g.LEVER_K_EQ), float(g.LEVER_D), 5.0)),
                  drive_targets=(("lever", float(g.LEVER_PRE)),),
                  usd_materials=True, self_collisions=True, solver_iters=32, contact_offset=0.0005)
_rest_c, _rest_r = g.rest_pose()
REST = Shape("hand_rest", pos=tuple(float(v) for v in _rest_c), size=tuple(float(v) for v in g.REST_SIZE),
             rot_xyzw=tuple(float(v) for v in g.quat_xyzw(_rest_r)), color=(0.62, 0.74, 0.80))
CADDY = [Shape(f"caddy_{n}", pos=tuple(float(v) for v in c), size=tuple(float(v) for v in s), color=(0.30, 0.55, 0.62))
         for n, c, s in g.caddy_boxes()]
_towel_c, _towel_r, _towel_s = g.towel_pose()
TOWEL = Shape("towel_roll", pos=tuple(float(v) for v in _towel_c), size=tuple(float(v) for v in _towel_s),
              rot_xyzw=tuple(float(v) for v in g.quat_xyzw(_towel_r)), color=(0.93, 0.93, 0.90))
OBJECTS = [HAND, CLIPPER, REST, TOWEL, *CADDY]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
OFF_TABLE = 0.05                # m: the clipper's centre this far below the table top: it fell off the table
DOWN_Z = 0.04                   # m: put down: its lowest point within this of the table top (on the table or caddy)
CLEAR = 0.02                    # m: let go: every part of both grippers this far from the clipper and both fingers
STILL_V = 0.01                  # m/s: the clipper still
W_BAND, W_STRAIGHT, W_STEADY = 0.3, 0.1, 0.1   # per nail
CAP_NOT_DONE = 0.9              # best score while the clipper is still held or not put down

_STATE: dict = {}


def gripper_boxes(tcp, rot, opening: float):
    """The gripper as boxes (centre, rotation, half sizes), station frame: each finger's pad (from 3 cm behind the TCP
    to its tip 13 mm past it), each finger's root and the housing. The fingers close along the TCP frame's x."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    half = opening / 2
    local = []
    for s in (-1.0, 1.0):
        local.append((np.array([s * (half + 0.005), 0.0, -0.0085]), np.array([0.005, 0.020, 0.0215])))
        local.append((np.array([s * (half + 0.010), 0.0, -0.045]), np.array([0.010, 0.020, 0.015])))
    local.append((np.array([0.0, 0.0, -0.1025]), np.array([max(0.051, half + 0.024), 0.034, 0.0425])))
    return [(tcp + rot @ c, rot, h) for c, h in local]


def _rot(quat_xyzw) -> np.ndarray:
    return g.rot_from_xyzw(quat_xyzw)


HOLD_GAP = 0.01                 # m: a gripper this close to the finger's middle segment, and not the one holding the
                                # clipper (touching it: within GRIP_TOUCH), holds the finger
GRIP_TOUCH = 0.003


def finger_held(n: str, f_pose, grips, c_pos, c_rot, theta: float, beta: float) -> bool:
    """Some gripper not holding the clipper is within HOLD_GAP of the finger's middle segment."""
    seg = g.finger_capsules_world(n, *f_pose)[1]
    boxes = g.clipper_boxes(c_pos, c_rot, theta, beta)
    for grip in grips:
        if min(g.boxes_gap(a, b, 5) for a in grip for b in boxes) <= GRIP_TOUCH:
            continue
        pts = np.vstack([g.box_points(*a, 5) for a in grip])
        if float(g.point_capsule_dist(pts, *seg).min()) <= HOLD_GAP:
            return True
    return False


def nail_score(n: str) -> float:
    """One nail (0 .. 0.5): 0.3 x the fraction of its 8 bins with 1-2 mm of white edge; 0.1 if every bin was cut and
    the edge is straight (longest minus shortest at most 0.75 mm); 0.1 if it was cut at all and its finger was never
    turned more than 10 degrees from rest, and the other hand held it (within 1 cm of its middle segment) at every
    closure that cut it."""
    S = _STATE
    length, cut = S["L"][n], S["cut"][n]
    band = float(np.mean((length >= g.BAND[0] - 1e-9) & (length <= g.BAND[1] + 1e-9)))
    straight = bool(cut.all() and length.max() - length.min() <= g.STRAIGHT + 1e-9)
    steady = bool(cut.any() and S["maxdef"][n] <= g.STEADY_DEG and not S["unheld"][n])
    return W_BAND * band + W_STRAIGHT * straight + W_STEADY * steady


def reset_state() -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, L={n: g.free_edge0(n).copy() for n in g.NAILS},
                  cut={n: np.zeros(g.N_BINS, bool) for n in g.NAILS}, maxdef={n: 0.0 for n in g.NAILS},
                  unheld={n: False for n in g.NAILS},
                  armed=True, shut_steps=0, closures=[], lever_force=0.0, gap=g.GAP0, down=False, clear=False, still=False,
                  gaps=None, total=0.0, level=0.0)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def bin_targets(n: str) -> list[float]:
    """Slide of each free-edge strip (m, <= 0): back into the plate by what has been cut off it."""
    return [float(v) for v in _STATE["L"][n] - g.free_edge0(n)]


def update(fingers: dict, flex: dict, c_pos, c_rot, theta: float, beta: float, c_vel, grips,
           lever_force: float = math.inf) -> dict:
    """One control step of the monitor, on measured values (station frame). ``fingers``: each trimmed finger's link
    pose (pos, rotation); ``flex``: its knuckle angle (rad); ``c_pos``/``c_rot``: the clipper body's pose; ``theta``,
    ``beta``: its jaw angle and lever angle (rad, as geometry.py); ``grips``: each gripper as boxes;
    ``lever_force``: the force (N) the lever's pivot joint carries (the squeeze on the lever). Returns the nails
    whose strips must move (name -> slide targets)."""
    S = _STATE
    cp, cr = np.asarray(c_pos, float), np.asarray(c_rot, float)
    for n in g.NAILS:
        S["maxdef"][n] = max(S["maxdef"][n], abs(math.degrees(flex[n])))

    gap = g.jaw_gap(theta)
    S["gap"] = gap
    S["lever_force"] = round(float(lever_force), 2)
    out = {}
    if gap >= g.REARM_GAP:
        S["armed"], S["shut_steps"] = True, 0
    if gap < g.PINCH_GAP:
        lo, up = g.edge_segments_world(cp, cr, theta)
        for n in g.NAILS:
            if g.pinches_skin(n, lo, up, *fingers[n]):
                _flag(f"the jaws closed on the skin of the {n} finger")
    if S["armed"] and gap <= g.CUT_GAP and lever_force >= g.F_CUT:
        S["shut_steps"] += 1
        if S["shut_steps"] >= g.CUT_STEPS:
            S["armed"] = False
            line = g.edge_points_world(cp, cr, theta)
            for n in g.NAILS:
                uvw = g.cut_line_in_nail(n, line, *fingers[n])
                new, cut, nick, info = g.closure(n, uvw, S["L"][n])
                if nick:
                    _flag(f"nicked the {n} finger: the jaws closed within 0.5 mm of the quick")
                if cut.any():
                    S["L"][n], S["cut"][n] = new, S["cut"][n] | cut
                    out[n] = bin_targets(n)
                    if not finger_held(n, fingers[n], grips, cp, cr, theta, beta):
                        S["unheld"][n] = True
                if info["u_mid"] is not None and abs(info["u_mid"]) < 0.02:     # every closure at a nail
                    S["closures"].append({"nail": n, "u_mid_mm": None if info["u_mid"] is None else
                                          round(info["u_mid"] * 1000, 2),
                                          "w_mid_mm": None if info["w_mid"] is None else round(info["w_mid"] * 1000, 2),
                                          "white_mm": [round(x * 1000, 2) for x in S["L"][n]], "nick": nick})
    else:
        S["shut_steps"] = 0

    boxes = g.clipper_boxes(cp, cr, theta, beta)
    if cp[2] < g.TABLE - OFF_TABLE:
        _flag("the clipper fell off the table")
    low = min(float(g.box_points(*b, 2)[:, 2].min()) for b in boxes)
    S["down"] = low <= g.TABLE + DOWN_Z
    gaps = []
    for grip in grips:
        to_clip = min(g.boxes_gap(a, b, 5) for a in grip for b in boxes)
        pts = np.vstack([g.box_points(*a, 5) for a in grip])
        to_hand = min(float(g.point_capsule_dist(pts, *c).min()) for n in g.NAILS
                      for c in g.finger_capsules_world(n, *fingers[n]))
        gaps.append((to_clip, to_hand))
    S["gaps"] = [tuple(round(x * 1000, 1) for x in gp) for gp in gaps]
    S["clear"] = all(min(gp) >= CLEAR for gp in gaps)
    S["still"] = bool(float(np.linalg.norm(c_vel)) < STILL_V)
    total = sum(nail_score(n) for n in g.NAILS)
    S["total"] = round(total, 6)
    done = S["down"] and S["clear"] and S["still"]
    S["level"] = 0.0 if S["violation"] else (total if done else min(total, CAP_NOT_DONE))
    return out


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

    st = object_states(environment, ["hand", "clipper"])
    hd, cl = st["hand"], st["clipper"]
    fingers = {n: (np.asarray(hd.links[n][0], float), _rot(hd.links[n][1])) for n in g.NAILS}
    flex = {n: float(hd.joints.get(n, 0.0)) for n in g.NAILS}
    body_pos, body_q = cl.links.get("body", (cl.pos, cl.quat_xyzw))
    theta = float(cl.joints.get("jaw", 0.0))
    beta = g.PHI0 + float(cl.joints.get("lever", 0.0))
    out = update(fingers, flex, body_pos, _rot(body_q), theta, beta, cl.lin_vel, _grips(environment),
                 link_force(environment, "clipper", "lever"))
    for n, targets in out.items():
        set_targets(environment, "hand", BIN_JOINTS[n], targets)


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """Both nails: every bin's white edge 1-2 mm, straight across (spread at most 0.75 mm), each finger held within 10
    degrees of rest; no nick, ever; the clipper put down (lowest point within 4 cm of the table top), still, and every
    part of both grippers 2 cm from it and from both fingers. Held for 1 s."""
    return bool(not violation() and _STATE["level"] >= 1.0 - 1e-9)


def score(objects) -> float:
    """0 after any violation (the jaws closing within 0.5 mm of the quick or on the skin; the clipper off the table).
    Otherwise the sum over the two nails (0.5 each: 0.3 x fraction of bins at 1-2 mm, 0.1 straight, 0.1 steady),
    capped at 0.9 until the clipper is put down and let go. Success scores 1."""
    return 0.0 if violation() else min(0.99, float(_STATE["level"]))
