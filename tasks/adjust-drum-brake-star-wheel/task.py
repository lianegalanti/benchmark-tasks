"""On the station table, ahead of the robot: a cropped rear drum-brake corner of a car on a bench stand, its axle
running across the table (along y), 18 cm above it. The backing plate faces the robot's right arm; the 9 in brake
drum, with its five wheel studs, faces the left arm. Below the axle, the backing plate's adjusting slot is open (its
rubber plug is out); through it the brass star wheel of the self-adjuster is in reach. The shoes are new and the
adjuster is run in: they sit about 1 mm inside the drum, so the brake would need a long pedal stroke. A brake adjusting
spoon (red handle, flat steel blade, 22.5 cm) lies on the table in front of the right arm, blade toward the corner.

The robot must put the spoon through the slot onto the star wheel's teeth and lever it round a click at a time,
turning the drum with the other hand to feel the drag, and stop when the drum turns with a light drag: the linings
pressed on the drum just enough that turning it takes 1.0 to 3.9 N m; then put the spoon down. Each click past that
presses the shoes harder, and the self-adjuster's lever stops the wheel being turned back.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step, set_target, write_joint  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Adjust the rear brake so the drum turns with a light drag, then put the spoon down and let go of everything."

# -- objects -------------------------------------------------------------------------------------------------------
CORNER = UsdBody("corner", usd="assets/corner.usda", pos=tuple(float(v) for v in g.B0), size=(0.29, 0.27, 0.27),
                 articulation=True,
                 joint_pos=(("drum", 0.0), ("star", 0.0), ("shoe_primary", 0.0), ("shoe_secondary", 0.0)),
                 drives=(("drum", 0.0, g.BEAR_DAMPING),
                         ("star", g.DETENT_K, g.DETENT_D, g.CLICK_TORQUE),
                         ("shoe_.*", g.K_SHOE, g.D_SHOE, g.SHOE_FORCE_MAX)),
                 usd_materials=True, self_collisions=True, solver_iters=32, contact_offset=0.001)
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
LOCK_STEPS = 10                 # over T_LOCK for 0.5 s: the brake was driven on hard (latched)

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


def level(torque: float, travel_mm: float, done: bool) -> float:
    """Score from the drum's turning torque (N m, measured), the shoes' mean outward travel (mm, measured) and whether
    the spoon is down, let go and everything is still (``done``)."""
    if torque > g.T_LOCK:                      # driven on (even if the trial ends before the 0.5 s latch)
        return 0.0
    if torque > g.T_HI:                        # a dragging brake
        return 0.25
    if torque >= g.T_LO:                       # light drag
        return 1.0 if done else 0.75
    # partial: how far the shoes have been set toward light drag: their measured travel plus, once on the drum, the
    # stretch of their springs that the drag shows; TRAVEL_LO_MM of it gives T_LO
    progress = max(0.0, travel_mm) + 1000 * (torque - g.T_BEAR) / (2 * g.MU_LINING * g.DRUM_R_IN * g.K_SHOE)
    return round(0.5 * min(1.0, progress / TRAVEL_LO_MM), 4)


TRAVEL_LO_MM = g.C0_MM + (g.T_LO - g.T_BEAR) / (2 * g.MU_LINING * g.DRUM_R_IN * g.K_SHOE) * 1000


def reset_state() -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, clicks=0, down=False, clear=False, still=False, drum=0.0, level=0.0,
                  gaps=None, torque=g.T_BEAR, forces=(0.0, 0.0), clear_mm=(g.C0_MM, g.C0_MM), travel_mm=0.0,
                  over=0, peak_torque=g.T_BEAR)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(corner_pos, corner_rot, drum_angle: float, star_angle: float, shoes, spoon_pos, spoon_rot, spoon_vel,
           grips) -> dict:
    """One control step of the monitor, on measured values (station frame). ``shoes``: each shoe joint's measured
    position (m, joint sign, in g.SHOES order); ``grips``: each gripper as boxes. Returns what to write to the sim: the
    star wheel's drive target, a stop angle (or None) and each shoe's drive target (m, joint sign)."""
    S = _STATE
    cp, cr = np.asarray(corner_pos, float), np.asarray(corner_rot, float)
    clicks, target, stop = adjuster_step(star_angle, S["clicks"])
    S["clicks"] = clicks
    # the brake: each lining's normal force on the drum is its drive spring's stretch (commanded minus measured
    # travel); the drum's turning torque follows from the forces
    t_m = g.target_mm(clicks) / 1000
    travel = [sign * float(q) for (_, _, sign, _), q in zip(g.SHOES, shoes)]
    forces = [g.shoe_force(t_m, u) for u in travel]
    torque = g.turning_torque(*forces)
    S["forces"] = tuple(round(f, 2) for f in forces)
    S["clear_mm"] = tuple(round(g.C0_MM - u * 1000, 4) for u in travel)
    S["travel_mm"] = float(np.mean(travel)) * 1000
    S["torque"] = torque
    S["peak_torque"] = max(S["peak_torque"], torque)
    S["over"] = S["over"] + 1 if torque > g.T_LOCK else 0
    if S["over"] >= LOCK_STEPS:
        _flag(f"over-adjusted: the shoes were driven onto the drum (over {g.T_LOCK} N m to turn it)")

    sp, sr = np.asarray(spoon_pos, float), np.asarray(spoon_rot, float)
    if sp[2] < g.TABLE - OFF_TABLE:
        _flag("the spoon left the table")
    S["down"] = spoon_top(sp, sr) <= g.TABLE + DOWN_Z
    gaps = [(gap_to_spoon(b, sp, sr), gap_to_drum(b, cp, cr)) for b in grips]
    S["gaps"] = [tuple(round(x * 1000, 1) for x in gp) for gp in gaps]
    S["clear"] = all(min(gp) >= CLEAR for gp in gaps)
    S["still"] = bool(float(np.linalg.norm(spoon_vel)) < STILL_V and abs(drum_angle - S["drum"]) < STILL_DRUM)
    S["drum"] = float(drum_angle)
    S["level"] = 0.0 if S["violation"] else level(torque, S["travel_mm"], S["down"] and S["clear"] and S["still"])
    return {"target": target, "stop": stop, "shoes": [sign * t_m for _, _, sign, _ in g.SHOES]}


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
    shoes = [float(co.joints.get(name, 0.0)) for name, _, _, _ in g.SHOES]
    out = update(co.pos, _rot(co.quat_xyzw), float(co.joints.get("drum", 0.0)), star, shoes, sp.pos,
                 _rot(sp.quat_xyzw), sp.lin_vel, _grips(environment))
    set_target(environment, "corner", "star", out["target"])
    for (name, _, _, _), t in zip(g.SHOES, out["shoes"]):
        set_target(environment, "corner", name, t)
    if out["stop"] is not None:
        write_joint(environment, "corner", "star", out["stop"])


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """Light drag: turning the drum takes 1.0 to 3.9 N m (from the linings' measured normal forces on it); never driven
    on over 4.8 N m; the spoon lying on the table, every part of both grippers 2 cm from it and from the drum, the spoon
    and the drum still. Held for 1 s."""
    return bool(not violation() and _STATE["level"] >= 1.0)


def score(objects) -> float:
    """0 after any violation (the shoes driven onto the drum: over 4.8 N m to turn it for 0.5 s; the spoon off the
    table). Otherwise, at the end, from the drum's turning torque: under 1.0 N m, 0.5 x the shoes' travel toward light
    drag (none: 0); 1.0 to 3.9 N m (light drag), 0.75 with the spoon not yet down and let go; 3.9 to 4.8 N m (a dragging
    brake), 0.25; over 4.8 N m, 0. Success scores 1."""
    return 0.0 if violation() else min(0.75, float(_STATE["level"]))
