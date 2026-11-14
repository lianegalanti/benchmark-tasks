"""On the station table, straight ahead of the arms: the motor base of a 9-cup food processor (white, 20 x 17 cm, its
coupling boss and drive shaft on top). In front of it, between the arms, its empty polycarbonate work bowl (18 cm
across, handle on its right) stands on its foot; to the right, its lid (feed tube up) stands on its skirt.

The robot must assemble the processor so it can run: set the bowl on the base over the drive shaft and turn it
clockwise until its three lugs lock under the boss's ledges, then put the lid on and turn it clockwise until its four
lugs lock under the rim's ledges, which brings the lid's interlock tab over the bowl's handle where it pushes the
sprung safety switch rod down. Then let go of both.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from fpkit import Slider, on_reset, on_step, part, reset_sliders  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Assemble the food processor so it is ready to run: lock the work bowl onto the base and the lid onto "
               "the bowl, then let go of both.")

# -- objects -------------------------------------------------------------------------------------------------------------
_BCYL, _BCONE = g.base_cylinders()
BASE = part("base", pos=tuple(float(v) for v in g.base_origin()), boxes=g.base_boxes(), cylinders=_BCYL, cones=_BCONE,
            mass=None, color=g.WHITE, friction=0.4)

BOWL = part("bowl", pos=(g.BOWL0[0], g.BOWL0[1], g.bowl_start_z()), boxes=g.bowl_boxes(), mass=g.BOWL_MASS,
            rot_xyzw=g.xyzw(g.rot_z(g.BOWL0_YAW * g.DEG)), color=g.CLEAR_PC, friction=0.4, torsional_patch=0.01,
            sliders=(Slider("rod", tuple(g.rod_boxes()), -g.ROD_TRAVEL, 0.0, g.ROD_RATE, g.ROD_DAMPING,
                            g.ROD_PRELOAD / g.ROD_RATE),
                     Slider("lock_detent", tuple(g.lock_plunger_boxes()), -g.DETENT_TRAVEL, 0.0, g.DETENT_RATE,
                            g.DETENT_DAMPING, g.DETENT_PRELOAD / g.DETENT_RATE),
                     Slider("lid_detent", tuple(g.lid_plunger_boxes()), 0.0, g.DETENT_TRAVEL, g.LID_DETENT_RATE,
                            g.DETENT_DAMPING, -g.LID_DETENT_PRELOAD / g.LID_DETENT_RATE)))

LID = part("lid", pos=(g.LID0[0], g.LID0[1], g.lid_start_z()), boxes=g.lid_boxes(), mass=g.LID_MASS,
           rot_xyzw=g.xyzw(g.rot_z(g.LID0_YAW * g.DEG)), color=g.CLEAR_PC, friction=0.4, torsional_patch=0.01)

OBJECTS = [BASE, BOWL, LID]

# -- rules (numbers shared by the scorer, the tests and the README) ------------------------------------------------------
OFF_TABLE = 0.05                # m: a part's origin this far below the table top has fallen off it
CENTRE_TOL = 0.004              # m: seated: the part's axis within this of the one it sits on
SEAT_TOL = (-0.002, 0.003)      # m: seated: its height within this of fully down
TILT_TOL = math.radians(3.0)    # seated: its axis within this of the other's
LOCK_WIN = (-2.0, 2.5)          # degrees from locked (the stop is at -0.5): every lug wholly under its ledge
FORCED = -3.5                   # degrees: turned 3 degrees past the stop: a lug or stop broken off
WRAP_LOW = -20.0                # degrees: turns are read in [-20, -20 + period): a lug in its gap (up to 92 degrees
#                                 counter-clockwise for the bowl, 62 for the lid) is never read as past its stop
OVER_LIMIT = 0.002              # m: a sprung part pushed this far past its end
CLEAR = 0.02                    # m: let go: every part of both grippers this far from the bowl and the lid
STILL_V, STILL_W = 0.01, 0.1    # m/s and rad/s: still

_STATE: dict = {}


# -- the grippers as boxes, from each arm's encoders --------------------------------------------------------------------
def gripper_boxes(tcp, rot, opening: float):
    """The gripper as boxes (centre, rotation, half sizes) in the station frame: each finger's pad (from 3 cm behind
    the TCP to its tip 13 mm past it) and root, and the finger carriage. The fingers close along the TCP's x axis;
    ``opening`` is the gap between the pads."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    half = opening / 2
    local = []
    for s in (-1.0, 1.0):
        local.append((np.array([s * (half + 0.005), 0.0, -0.0085]), np.array([0.005, 0.020, 0.0215])))
        local.append((np.array([s * (half + 0.010), 0.0, -0.045]), np.array([0.010, 0.020, 0.015])))
    local.append((np.array([0.0, 0.0, -0.1025]), np.array([max(0.051, half + 0.024), 0.034, 0.0425])))
    return [(tcp + rot @ c, rot, h) for c, h in local]


def gripper_points(arms: dict) -> np.ndarray:
    pts = [g.box_points(c, r, h) for a in arms.values() for c, r, h in gripper_boxes(a["tcp"], a["rot"], a["opening"])]
    return np.vstack(pts)


def _arms(environment):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    out = {}
    for side in ("left", "right"):
        robot = environment.unwrapped.scene[f"{side}_robot"]
        q = robot.data.joint_pos
        q = getattr(q, "torch", q)[0].detach().cpu().numpy().astype(float)
        ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
        fids = robot.find_joints(list(FINGER_JOINTS), preserve_order=True)[0]
        tcp, rot = ik.tcp_pose(q[ids], station.BASES[side])
        out[side] = {"tcp": tcp, "rot": rot, "opening": float(np.abs(q[fids]).sum())}
    return out


# -- the parts relative to each other ---------------------------------------------------------------------------------
def wrap(deg: float, period: float, low: float = WRAP_LOW) -> float:
    """``deg`` wrapped to [low, low + period): each lug measured from its own locked position, counter-clockwise
    through its gap (the drop positions) up to the next lug's stop, and clockwise only to WRAP_LOW past it."""
    return (deg - low) % period + low


def relative(pos, rot, ref_pos, ref_rot):
    """A body's origin and rotation in a reference body's frame, and its turn (degrees) and tilt (rad) there."""
    rr = np.asarray(ref_rot, float)
    p = rr.T @ (np.asarray(pos, float) - np.asarray(ref_pos, float))
    r = rr.T @ np.asarray(rot, float)
    return p, r, math.degrees(g.yaw_of(r)), g.tilt_of(r)


def seated(p, tilt: float, height: float) -> bool:
    return bool(math.hypot(p[0], p[1]) <= CENTRE_TOL and SEAT_TOL[0] <= p[2] - height <= SEAT_TOL[1]
                and tilt <= TILT_TOL)


def in_lock(turn: float) -> bool:
    return bool(LOCK_WIN[0] <= turn <= LOCK_WIN[1])


def level_of(bowl_seated: bool, bowl_locked: bool, lid_seated: bool) -> float:
    if bowl_locked and lid_seated:
        return 0.75
    if bowl_locked:
        return 0.5
    if bowl_seated:
        return 0.25
    return 0.0


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, level=0.0, ready=False, clear=False, still=False, info={})
    if environment is not None:
        reset_sliders(environment, "bowl")


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(base_pos, base_rot, bowl_pos, bowl_rot, bowl_v, bowl_w, joints: dict, lid_pos, lid_rot, lid_v, lid_w,
           arms: dict) -> None:
    """One control step of the monitor (station frame in; the bowl judged in the base's frame, the lid in the
    bowl's). ``arms``: side -> {"tcp", "rot", "opening"} from the arm's encoders."""
    S = _STATE
    bowl_pos, bowl_rot = np.asarray(bowl_pos, float), np.asarray(bowl_rot, float)
    lid_pos, lid_rot = np.asarray(lid_pos, float), np.asarray(lid_rot, float)
    pb, _, turn_b, tilt_b = relative(bowl_pos, bowl_rot, base_pos, base_rot)
    pl, _, turn_l, tilt_l = relative(lid_pos, lid_rot, bowl_pos, bowl_rot)
    # the bowl fits the boss three ways round and the lid the rim four ways: lock is judged at the nearest one
    tb, tl = wrap(turn_b, 120.0), wrap(turn_l, 90.0)
    rod = float(joints.get("rod", 0.0))
    lock_d, lid_d = float(joints.get("lock_detent", 0.0)), float(joints.get("lid_detent", 0.0))

    b_seated = seated(pb, tilt_b, g.SEAT_Z)
    l_seated = seated(pl, tilt_l, g.BOWL_H)

    # -- violations (latched for the whole trial): real failures only ----------------------------------------------
    if bowl_pos[2] < g.TABLE - OFF_TABLE:
        _flag("the bowl fell off the table")
    if lid_pos[2] < g.TABLE - OFF_TABLE:
        _flag("the lid fell off the table")
    if b_seated and tb < FORCED:
        _flag("the bowl was forced past its stops")
    if l_seated and tl < FORCED:
        _flag("the lid was forced past its stops")
    if (rod < -g.ROD_TRAVEL - OVER_LIMIT or lock_d < -g.DETENT_TRAVEL - OVER_LIMIT
            or lid_d > g.DETENT_TRAVEL + OVER_LIMIT):
        _flag("a sprung part was forced past its end")

    # -- progress ----------------------------------------------------------------------------------------------------
    b_locked = b_seated and in_lock(tb)
    lvl = level_of(b_seated, b_locked, l_seated)
    on_switch = b_locked and in_lock(turn_b)          # the handle (and the rod's foot) over the base's switch
    l_locked = l_seated and in_lock(tl)
    pressed = rod <= -g.ROD_ACTUATE
    ready = bool(on_switch and l_locked and pressed)

    pts = gripper_points(arms)
    bc, bb = g.bowl_outline()
    lc, lb = g.lid_outline()
    d_bowl = g.dist_to_outline((pts - bowl_pos) @ bowl_rot, bc, bb)
    d_lid = g.dist_to_outline((pts - lid_pos) @ lid_rot, lc, lb)
    clear = bool(min(d_bowl, d_lid) >= CLEAR)
    still = bool(max(np.linalg.norm(bowl_v), np.linalg.norm(lid_v)) < STILL_V
                 and max(np.linalg.norm(bowl_w), np.linalg.norm(lid_w)) < STILL_W)
    S.update(level=lvl, ready=ready, clear=clear, still=still,
             info={"bowl_turn": round(turn_b, 2), "tilts_deg": [round(math.degrees(tilt_b), 1), round(math.degrees(tilt_l), 1)], "bowl_xyz_mm": np.round(pb * 1000, 1).tolist(),
                   "lid_turn": round(turn_l, 2), "lid_xyz_mm": np.round(pl * 1000, 1).tolist(),
                   "rod_mm": round(-rod * 1000, 2), "detents_mm": [round(lock_d * 1000, 2), round(lid_d * 1000, 2)],
                   "seated": [b_seated, l_seated], "locked": [b_locked, l_locked],
                   "grip_gap_mm": [round(d_bowl * 1000, 1), round(d_lid * 1000, 1)]})


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["base", "bowl", "lid"])
    b, w, li = st["base"], st["bowl"], st["lid"]
    update(b.pos, b.rotation(), w.pos, w.rotation(), w.lin_vel, w.ang_vel, w.joints, li.pos, li.rotation(),
           li.lin_vel, li.ang_vel, _arms(environment))


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """No violation so far; the bowl locked with its handle over the base's switch, the lid locked, the switch rod
    pushed down at least 5 mm; every part of both grippers at least 2 cm from the bowl and the lid; both still.
    Held for 1 s."""
    S = _STATE
    return bool(not violation() and S["ready"] and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: the bowl or the lid fell off the table, either was forced more than 3 degrees past its
    stops, or a sprung part was forced 2 mm past its end. Otherwise the stage at the end: 0.25 the bowl seated over the
    drive shaft (its lugs down in the channel under the ledges' level); 0.5 the bowl locked (turned to within 3
    degrees of its stops, every lug wholly under its ledge); 0.75 the bowl locked and the lid seated on its rim.
    Success (lid locked too, the switch rod pressed 5 mm or more, both grippers let go, still for 1 s) scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
