"""On the station table, ahead of the robot: a cropped rear drum-brake corner of a car on a bench stand, its axle
running across the table (along y), 18 cm above it. The backing plate faces the robot's right arm; the 9 in brake
drum, with its five wheel studs, faces the left arm. Below the axle, the backing plate's adjusting slot is open (its
rubber plug is out); through it the brass star wheel of the self-adjuster is in reach. The shoes are new and the
adjuster is run in: they sit 1.5 mm inside the drum, so the brake would need a long pedal stroke. A brake adjusting
spoon (red handle, flat steel blade, 22.5 cm) lies on the table in front of the right arm, blade toward the corner.

The robot must put the spoon through the slot onto the star wheel's teeth and lever it round a click at a time,
turning the drum with the other hand to feel the drag, and stop when the drum turns with a light drag (shoes 0.1 to
0.4 mm from the drum); then put the spoon down. One click too many past that puts the shoes on the drum: they lock it,
and the self-adjuster's lever stops the wheel being turned back.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step, set_damping, set_target, write_joint  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Adjust the rear brake so the drum turns with a light drag, then put the spoon down and let go of everything."

# -- objects -------------------------------------------------------------------------------------------------------
CORNER = UsdBody("corner", usd="assets/corner.usda", pos=tuple(float(v) for v in g.B0), size=(0.29, 0.27, 0.27),
                 articulation=True, joint_pos=(("drum", 0.0), ("star", 0.0)),
                 drives=(("drum", 0.0, g.drum_damping(g.C0_MM)),
                         ("star", g.DETENT_K, g.DETENT_D, g.CLICK_TORQUE)),
                 usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001)
SPOON = UsdBody("spoon", usd="assets/spoon.usda", pos=tuple(float(v) for v in g.SPOON_START),
                rot_xyzw=tuple(float(v) for v in g.quat_xyzw(g.YAW90)),
                size=(g.SPOON_LEN, g.HANDLE[1], g.HANDLE[2]), mass=g.SPOON_MASS, usd_materials=True,
                torsional_patch=0.01, contact_offset=0.001, solver_iters=32)
OBJECTS = [CORNER, SPOON]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
OFF_TABLE = 0.05                # m: the spoon's centre this far below the table top: it left the table
DOWN_Z = 0.03                   # m: set down: every part of the spoon within this of the table top (it lies flat)
CLEAR = 0.02                    # m: let go: every part of both grippers this far from the spoon and the drum
STILL_V = 0.01                  # m/s: the spoon still ...
STILL_DRUM = math.radians(0.2)  # ... and the drum turning less than this per control step (4 deg/s)

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
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def gap_to_spoon(grip, spoon_pos, spoon_rot) -> float:
    return min(g.boxes_gap(a, b, 5) for a in grip for b in g.spoon_boxes(spoon_pos, spoon_rot))


def gap_to_drum(grip, corner_pos, corner_rot) -> float:
    """Gap from the gripper boxes to the drum (band and face, and the studs' ring, as cylinders)."""
    pts = np.vstack([g.box_points(*b, 5) for b in grip])
    return float(min(g.point_cyl_dist(pts, *c).min() for c in g.drum_world_cylinders(0.0, corner_pos, corner_rot)))


def spoon_top(spoon_pos, spoon_rot) -> float:
    return float(max(g.box_points(*b, 2)[:, 2].max() for b in g.spoon_boxes(spoon_pos, spoon_rot)))


# -- the adjuster ------------------------------------------------------------------------------------------------------
def adjuster_step(star_angle: float, clicks: int):
    """The self-adjuster, from the star wheel's measured angle (rad) and the clicks counted so far: returns (clicks,
    detent drive target (rad), angle to stop the wheel at or None). Turned past half a tooth in the tightening sense,
    the wheel's spring-loaded lever drops into the next notch (a click); turned back more than PAWL of a tooth behind
    its notch, the lever stops it (the self-adjuster's pawl)."""
    turned = g.TIGHTEN * star_angle / g.PITCH         # teeth turned in the tightening sense
    if turned > clicks + 0.5:
        clicks = int(math.floor(turned + 0.5))
    stop = None
    if turned < clicks - g.PAWL:
        stop = g.TIGHTEN * (clicks - g.PAWL) * g.PITCH
    return clicks, g.TIGHTEN * clicks * g.PITCH, stop


def level(clicks: int, done: bool) -> float:
    """Score from the clicks counted (clearance = 1.5 mm - 0.05 mm per click) and whether the spoon is down, let go
    and everything is still (``done``)."""
    if clicks >= g.N_LOCK or clicks <= 0:
        return 0.0
    if clicks > g.N_TIGHT:                    # 0 < clearance < 0.10 mm: a dragging brake
        return 0.25
    if clicks >= g.N_FREE:                    # 0.10 .. 0.40 mm: light drag
        return 1.0 if done else 0.75
    return round(0.5 * clicks / g.N_FREE, 4)  # partial: the clearance taken up so far, toward 0.40 mm


def reset_state() -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, clicks=0, damping=None, down=False, clear=False, still=False,
                  drum=0.0, level=0.0, gaps=None)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(corner_pos, corner_rot, drum_angle: float, star_angle: float, spoon_pos, spoon_rot, spoon_vel,
           grips) -> dict:
    """One control step of the monitor, on measured values (station frame). ``grips``: each gripper as boxes. Returns
    what to write to the sim: the star wheel's drive target, a stop angle (or None) and the drum's damping."""
    S = _STATE
    cp, cr = np.asarray(corner_pos, float), np.asarray(corner_rot, float)
    clicks, target, stop = adjuster_step(star_angle, S["clicks"])
    S["clicks"] = clicks
    if clicks >= g.N_LOCK:
        _flag("over-adjusted: the shoes were driven onto the drum and lock it")
    c = g.clearance_mm(clicks)
    damping = g.drum_damping(c)

    sp, sr = np.asarray(spoon_pos, float), np.asarray(spoon_rot, float)
    if sp[2] < g.TABLE - OFF_TABLE:
        _flag("the spoon left the table")
    S["down"] = spoon_top(sp, sr) <= g.TABLE + DOWN_Z
    gaps = [(gap_to_spoon(b, sp, sr), gap_to_drum(b, cp, cr)) for b in grips]
    S["gaps"] = [tuple(round(x * 1000, 1) for x in gp) for gp in gaps]
    S["clear"] = all(min(gp) >= CLEAR for gp in gaps)
    S["still"] = bool(float(np.linalg.norm(spoon_vel)) < STILL_V and abs(drum_angle - S["drum"]) < STILL_DRUM)
    S["drum"] = float(drum_angle)
    S["level"] = 0.0 if S["violation"] else level(clicks, S["down"] and S["clear"] and S["still"])
    return {"target": target, "stop": stop, "damping": damping}


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

    st = object_states(environment, ["corner", "spoon"])
    co, sp = st["corner"], st["spoon"]
    star = float(co.joints.get("star", 0.0))
    out = update(co.pos, _rot(co.quat_xyzw), float(co.joints.get("drum", 0.0)), star, sp.pos, _rot(sp.quat_xyzw),
                 sp.lin_vel, _grips(environment))
    set_target(environment, "corner", "star", out["target"])
    if out["stop"] is not None:
        write_joint(environment, "corner", "star", out["stop"])
    if _STATE["damping"] is None or abs(out["damping"] - _STATE["damping"]) > 1e-3 * out["damping"]:
        set_damping(environment, "corner", "drum", out["damping"])
        _STATE["damping"] = out["damping"]


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """The adjuster 22 to 28 clicks out (shoes 0.10 to 0.40 mm from the drum: it turns with a light drag), never
    over-adjusted; the spoon lying on the table, every part of both grippers 2 cm from it and from the drum, the spoon
    and the drum still. Held for 1 s."""
    return bool(not violation() and _STATE["level"] >= 1.0)


def score(objects) -> float:
    """0 after any violation (the shoes driven onto the drum: 30 clicks or more; the spoon off the table). Otherwise,
    at the end, from the clicks counted: 0 for none; 0.5 x clicks / 22 below 22 (shoes still more than 0.4 mm off:
    the drum turns freely); 0.75 at 22 to 28 (0.10 to 0.40 mm: light drag) with the spoon not yet down and let go;
    0.25 at 29 (0.05 mm: a dragging brake). Success scores 1."""
    return 0.0 if violation() else min(0.75, float(_STATE["level"]))
