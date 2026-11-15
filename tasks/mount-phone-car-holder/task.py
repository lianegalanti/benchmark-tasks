"""On the station table, straight ahead of the arms: a car dashboard section (60 cm wide, 9 cm tall) with a spring-clamp
phone holder clipped to its air vent, the cradle facing the robot. The holder's two side arms are closed and empty, as
left after the last phone was taken out. A phone (iPhone 15 size) leans back against the dashboard's top front edge,
to the right of the holder, screen toward the robot.

The robot must mount the phone. The holder's arms are geared together and sprung shut, with no catch: they only stay
open while something holds them. So the robot has to hold them open (by a pull tab on an arm's front, or with the
phone's own edge), bring the phone in between them onto the cradle's two feet, and let the arms close on its long
edges; then let go. The holder, not the robot, must end up holding the phone.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import Shape, UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Mount the phone in the car phone holder so that the holder's arms grip it, then let go of the phone."

# -- objects -------------------------------------------------------------------------------------------------------
HOLDER = UsdBody("holder", usd="assets/holder.usda", pos=tuple(float(v) for v in g.HOLDER), size=(0.045, 0.088, 0.10),
                 articulation=True, joint_pos=(("arm_left", 0.0), ("arm_right", 0.0)),
                 drives=(("arm_.*", g.ARM_RATE, g.ARM_DAMPING),),
                 drive_targets=(("arm_left", g.ARM_TARGET), ("arm_right", g.ARM_TARGET)),
                 usd_materials=True, self_collisions=True, solver_iters=32, contact_offset=0.001)
PHONE = UsdBody("phone", usd="assets/phone.usda", pos=tuple(float(v) for v in g.phone_start()),
                rot_xyzw=g.quat_xyzw(g.PHONE_R0), size=(g.PHONE_T, g.PHONE_W, g.PHONE_L), mass=g.PHONE_MASS,
                friction=(0.5, 0.4), torsional_patch=0.008)
DASH = Shape("dashboard", pos=(g.DASH_X + g.DASH_D / 2, 0.0, g.TABLE + g.DASH_H / 2), size=(g.DASH_D, g.DASH_W, g.DASH_H),
             color=(0.18, 0.18, 0.19))
VENT = Shape("vent_grille", pos=(g.DASH_X - 0.0005, 0.0, g.TABLE + 0.055), size=(0.001, 0.16, 0.05),
             color=(0.05, 0.05, 0.05), collide=False)
OBJECTS = [HOLDER, PHONE, DASH, VENT]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
FALL_V = 0.5                    # m/s: a phone no gripper holds, falling this fast, was dropped (a 1.3 cm free fall)
OFF_TABLE = 0.02                # m: the phone's centre this far below the table top has left it
CLEAR = 0.02                    # m: every part of both grippers this far from the phone and the holder's arms
GRIP_TOL = 0.002                # m: each arm within this of the travel at which its pad meets the phone's edge
CENTRE_TOL = 0.002              # m: clamped: the phone centred within this (so both pads are on its edges)
GRIP_CENTRE = 0.004             # m: gripped (0.5): looser
UPRIGHT = math.radians(6)       # clamped: phone's long axis within this of the cradle's vertical, face square to it
GRIP_TILT = math.radians(20)    # gripped (0.5): looser
CLAMP_FEET = (-0.002, 0.010)    # m: clamped: phone's lowest corner from 2 mm below to 1 cm above the feet's tops
BACK_X = (-0.006, 0.001)        # m: clamped: phone's back this far from the back plate's face (inside the lips)
STILL_V, STILL_W = 0.01, 0.1    # m/s and rad/s: phone still
STILL_ARM = 0.0005              # m per control step (1 cm/s): arms still
TOUCH = 0.001                   # m: a holder arm's pad this close to the phone touches it
HOLD_R = 0.01                   # m: a gripper may be holding the phone when any finger is this close to it
ARM_R = 0.035                   # m: arm links (not the gripper) checked as capsules, for the tests' clearances
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")

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


def gap_to(boxes_a, boxes_b) -> float:
    return min(g.boxes_gap(a, b) for a in boxes_a for b in boxes_b)


# -- the phone in the holder's frame --------------------------------------------------------------------------------
def phone_in_holder(pos, rot, holder_pos, holder_rot):
    """The phone's centre and rotation in the holder frame H (measured from the holder's own pose)."""
    hr = np.asarray(holder_rot, float)
    return hr.T @ (np.asarray(pos, float) - np.asarray(holder_pos, float)), hr.T @ np.asarray(rot, float)


def lowest_z(pos, rot) -> float:
    corners = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * g.PHONE_HALF
    return float((np.asarray(pos) + corners @ np.asarray(rot).T)[:, 2].min())


def tilt(rot) -> float:
    """Largest angle between the phone's axes and the upright cradle's (long axis vertical, back toward +x)."""
    rot = np.asarray(rot, float)
    return max(math.acos(min(1.0, max(-1.0, float(rot[2, 2])))), math.acos(min(1.0, max(-1.0, float(rot[0, 0])))))


def gripped(p, r, d_left: float, d_right: float, pad_gaps) -> bool:
    """0.5: both arms closed onto the phone's long edges: each arm stopped by the phone (within 2 mm of the travel at
    which its pad meets the edge, not at its closed stop) and its pad touching the phone; the phone roughly centred
    and upright. Nothing about where it sits on the cradle."""
    return bool(abs(d_left - g.PHONE_D) <= GRIP_TOL and abs(d_right - g.PHONE_D) <= GRIP_TOL
                and max(pad_gaps) < TOUCH and abs(p[1]) <= GRIP_CENTRE and tilt(r) <= GRIP_TILT)


def clamped(p, r, d_left: float, d_right: float, pad_gaps) -> bool:
    """0.75: gripped, and seated: the phone centred between the pads, upright and square, on or up to 1 cm above the
    feet, its back against the plate within the lips."""
    back = p[0] + g.PHONE_T / 2
    return bool(gripped(p, r, d_left, d_right, pad_gaps) and abs(p[1]) <= CENTRE_TOL and tilt(r) <= UPRIGHT
                and CLAMP_FEET[0] <= lowest_z(p, r) <= CLAMP_FEET[1] and BACK_X[0] <= back <= BACK_X[1])


def level_of(gripped_: bool, clamped_: bool) -> float:
    return 0.75 if clamped_ else 0.5 if gripped_ else 0.0


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, level=0.0, gripped=False, clamped=False, clear=False, still=False,
                  arms=(0.0, 0.0), held=False, max_open=0.0, info={})


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(holder_pos, holder_rot, joints: dict, phone_pos, phone_rot, phone_vel, phone_w, arms: dict) -> None:
    """One control step of the monitor. ``arms``: side -> {"tcp", "rot", "opening"} from the arm's encoders.
    Everything about the cradle is judged in the holder's measured frame; falls in the station frame."""
    S = _STATE
    hp, hr = np.asarray(holder_pos, float), np.asarray(holder_rot, float)
    pos, rot = np.asarray(phone_pos, float), np.asarray(phone_rot, float)
    vel = np.asarray(phone_vel, float)
    dl, dr = float(joints.get("arm_left", 0.0)), float(joints.get("arm_right", 0.0))
    p, r = phone_in_holder(pos, rot, hp, hr)

    grips = {s: gripper_boxes(a["tcp"], a["rot"], a["opening"]) for s, a in arms.items()}
    phone_box = [g.phone_box(pos, rot)]
    jaws = [(hp + hr @ c, hr @ rr, h) for c, rr, h in g.jaw_boxes(dl, dr)]
    tabs = [(hp + hr @ c, hr @ rr, h) for c, rr, h in g.tab_boxes(dl, dr)]
    held = any(gap_to(b[:4], phone_box) < HOLD_R for b in grips.values())     # b[:4]: the fingers
    S["held"] = held

    # -- violations (latched for the whole trial) ---------------------------------------------------------------
    if not held and vel[2] < -FALL_V:
        _flag("the phone was dropped")
    if pos[2] < g.TABLE - OFF_TABLE:
        _flag("the phone left the table")

    # -- the stage now (only the stage at the end counts) ----------------------------------------------------------
    pads = [(hp + hr @ c, hr @ rr, h) for c, rr, h in g.pad_boxes(dl, dr)]
    pad_gaps = [g.boxes_gap(q, phone_box[0]) for q in pads]
    gripped_ = gripped(p, r, dl, dr, pad_gaps)
    clamped_ = clamped(p, r, dl, dr, pad_gaps)
    lvl = level_of(gripped_, clamped_)
    clear = all(gap_to(b, phone_box + jaws + tabs) >= CLEAR for b in grips.values())
    pl, pr = S["arms"]
    still = (float(np.linalg.norm(vel)) < STILL_V and float(np.linalg.norm(phone_w)) < STILL_W
             and abs(dl - pl) < STILL_ARM and abs(dr - pr) < STILL_ARM)
    S.update(level=lvl, gripped=gripped_, clamped=clamped_, clear=clear, still=still, arms=(dl, dr),
             max_open=max(S["max_open"], dl, dr),
             info={"arms_mm": [round(dl * 1000, 1), round(dr * 1000, 1)],
                   "phone_mm": np.round(p * 1000, 1).tolist(), "tilt_deg": round(math.degrees(tilt(r)), 1),
                   "low_mm": round(lowest_z(p, r) * 1000, 1), "pad_gap_mm": [round(x * 1000, 1) for x in pad_gaps],
                   "held": held, "clear": clear})


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["holder", "phone"])
    h, ph = st["holder"], st["phone"]
    update(h.pos, _rot(h.quat_xyzw), h.joints, ph.pos, _rot(ph.quat_xyzw), ph.lin_vel, ph.ang_vel,
           _arms(environment))


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """No violation so far; the holder's arms clamp the phone's long edges, the phone centred and upright, on or
    within 1 cm above the feet; every part of both grippers at least 2 cm from the phone and the holder's arms; the
    phone and the arms still. Held for 1 s."""
    S = _STATE
    return bool(not violation() and S["clamped"] and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: the phone dropped (falling faster than 0.5 m/s with no gripper holding it) or off the
    table. Otherwise the stage at the end, and nothing unless the holder's arms are closed on the phone: 0.5 the arms
    grip its long edges; 0.75 they clamp it seated on the cradle. Success (released, still, held 1 s) scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
