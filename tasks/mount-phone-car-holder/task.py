"""On the station table, straight ahead of the arms: a car dashboard section (60 cm wide, 9 cm tall) with a one-touch
phone holder clipped to its air vent, the cradle facing the robot. The holder's two side arms are closed and empty, as
left after the last phone was taken out. A phone (iPhone 15 size) leans back against the dashboard's top front edge,
to the right of the holder, screen toward the robot.

The robot must mount the phone: pull the holder's arms open until they click (the catch holds them open), stand the
phone on the cradle's two feet between them and press it onto the trigger in the back plate, which lets the sprung
arms close on the phone's long edges; then let go. The holder, not the robot, must end up holding the phone.

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
                 articulation=True, joint_pos=(("arm_left", 0.0), ("arm_right", 0.0), ("trigger", 0.0)),
                 drives=(("arm_.*", g.ARM_RATE, g.ARM_DAMPING), ("trigger", g.TRIG_RATE, g.TRIG_DAMPING)),
                 drive_targets=(("arm_left", g.ARM_TARGET), ("arm_right", g.ARM_TARGET), ("trigger", g.TRIG_TARGET)),
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
COCKED = 0.014                  # m: an arm opened this far (pads 86 mm apart or more) is open
CLEAR = 0.02                    # m: every part of both grippers this far from the phone and the holder's arms
GRIP_TOL = 0.002                # m: clamped: each arm within this of the travel at which its pad meets the phone ...
CENTRE_TOL = 0.002              # ... and the phone centred within this (so both pads are on its edges)
UPRIGHT = math.radians(6)       # clamped: phone's long axis within this of the cradle's vertical, face square to it
PLACE_TILT = math.radians(10)   # placed (0.5): same, looser
FEET_TOL = (-0.002, 0.003)      # m: placed: phone's lowest corner this close to the feet's top (below, above)
CLAMP_FEET = (-0.002, 0.010)    # m: clamped: the pads may hold it up to 1 cm above the feet
BACK_X = (-0.006, 0.001)        # m: clamped: phone's back this far from the back plate's face (lips hold it)
PLACE_BACK_X = (-0.012, 0.001)  # m: placed: phone's back between the plate and the arm lips
PLACE_Y = 0.015                 # m: placed: phone's centre within this of the cradle's centre line
STILL_V, STILL_W = 0.01, 0.1    # m/s and rad/s: phone still
STILL_ARM = 0.0005              # m per control step (1 cm/s): arms still
TOUCH = 0.001                   # m: a holder arm this close to the phone touches it
HOLD_R = 0.01                   # m: a gripper may be holding the phone when any finger is this close to it
FACE = math.radians(20)         # an arm stopping on a phone turned this far toward it closed on its face
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


def placed(p, r, d_left: float, d_right: float) -> bool:
    """0.5: the phone stands on both feet, upright and square, between the two open arms."""
    back = p[0] + g.PHONE_T / 2
    return bool(tilt(r) <= PLACE_TILT and FEET_TOL[0] <= lowest_z(p, r) <= FEET_TOL[1] and abs(p[1]) <= PLACE_Y
                and PLACE_BACK_X[0] <= back <= PLACE_BACK_X[1] and min(d_left, d_right) >= COCKED)


def clamped(p, r, d_left: float, d_right: float) -> bool:
    """0.75: both arms closed onto the phone's long edges (stopped by the phone, not by their closed stop), the phone
    centred between the pads, upright, on or up to 1 cm above the feet, its back against the plate within the lips."""
    back = p[0] + g.PHONE_T / 2
    return bool(abs(d_left - g.PHONE_D) <= GRIP_TOL and abs(d_right - g.PHONE_D) <= GRIP_TOL
                and abs(p[1]) <= CENTRE_TOL and tilt(r) <= UPRIGHT
                and CLAMP_FEET[0] <= lowest_z(p, r) <= CLAMP_FEET[1] and BACK_X[0] <= back <= BACK_X[1])


def level_of(cocked_: bool, placed_: bool, clamped_: bool) -> float:
    return 0.75 if clamped_ else 0.5 if placed_ else 0.25 if cocked_ else 0.0


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, level=0.0, closing=[False, False], clamped=False, clear=False, still=False,
                  arms=(0.0, 0.0), trigger=0.0, held=False, info={})


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
    held = any(gap_to(b[:4], phone_box) < HOLD_R for b in grips.values())     # b[:4]: the fingers
    S["held"] = held

    # -- violations (latched for the whole trial) ---------------------------------------------------------------
    if not held and vel[2] < -FALL_V:
        _flag("the phone was dropped")
    if pos[2] < g.TABLE - OFF_TABLE:
        _flag("the phone left the table")
    # an arm that has just been released from the catch closes until it stops; if it stopped on the phone, it must
    # be on the phone's edge: not on a face turned toward it, nor with its lip on the glass (stopped wider than the edge)
    gap_jaw = [g.boxes_gap(j, phone_box[0]) for j in jaws]
    near = [min(gap_to(b, [j]) for b in grips.values()) for j in jaws]
    pl, pr = S["arms"]
    for i, (side, d, d0) in enumerate((("left", dl, pl), ("right", dr, pr))):
        if d < COCKED <= d0 and near[i] >= CLEAR:
            S["closing"][i] = True               # it fell from the catch (no gripper is moving it)
        if S["closing"][i] and abs(d - d0) < STILL_ARM:
            S["closing"][i] = False              # it has stopped: judge where
            if gap_jaw[i] < TOUCH:
                face_on = abs(r[1, 0]) > math.sin(FACE)
                on_glass = g.PHONE_D + 0.0015 < d < COCKED
                if face_on or on_glass:
                    _flag(f"the holder's {side} arm closed on the phone's face, not its edge")

    # -- progress ------------------------------------------------------------------------------------------------
    cocked_ = min(dl, dr) >= COCKED and min(near) >= CLEAR
    placed_ = placed(p, r, dl, dr)
    clamped_ = clamped(p, r, dl, dr)
    lvl = level_of(cocked_, placed_, clamped_)
    clear = all(gap_to(b, phone_box + jaws) >= CLEAR for b in grips.values())
    still = (float(np.linalg.norm(vel)) < STILL_V and float(np.linalg.norm(phone_w)) < STILL_W
             and abs(dl - pl) < STILL_ARM and abs(dr - pr) < STILL_ARM)
    S.update(level=lvl, clamped=clamped_, clear=clear, still=still, arms=(dl, dr),
             trigger=float(joints.get("trigger", 0.0)),
             info={"arms_mm": [round(dl * 1000, 1), round(dr * 1000, 1)],
                   "trigger_mm": round(float(joints.get("trigger", 0.0)) * 1000, 1),
                   "phone_mm": np.round(p * 1000, 1).tolist(), "tilt_deg": round(math.degrees(tilt(r)), 1),
                   "low_mm": round(lowest_z(p, r) * 1000, 1), "held": held})


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
    within 1 cm above the feet; every part of both grippers at least 2 cm from the phone and the holder's arms; the phone
    and the arms still. Held for 1 s."""
    S = _STATE
    return bool(not violation() and S["clamped"] and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: the phone dropped (falling faster than 0.5 m/s with no gripper holding it) or off the
    table, or a holder arm, just released, stopped on its face instead of its edge. Otherwise the stage at the end:
    0.25 both arms cocked open (held by the catch, no gripper near them); 0.5 the phone standing on both feet between
    the open arms, upright; 0.75 the arms clamped on its edges. Success (released, still, held 1 s) scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
