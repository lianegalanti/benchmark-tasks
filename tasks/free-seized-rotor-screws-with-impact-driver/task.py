"""On the station table, ahead of the robot: a car's front hub and knuckle with its brake rotor, off the car for a
wheel-bearing job, standing hub-up on a bench stand, so the rotor lies flat and its hat faces up. The rotor is held
to the hub by two countersunk Phillips screws (M6, PH3 recess), one either side of the hub's pilot: one is already
out (its hole is open); the other is rusted in, as they nearly always are. A hand impact driver with a PH3 bit stands nose-down in a wooden block to the right of the
hub, and a 16 oz ball-peen hammer stands on its face on a stepped wooden stand to the left, its handle on the step.

The robot must break that screw loose with the impact driver: seat the bit fully in its recess, hold the driver
square to the screw, and strike its cap hard with the hammer. A blow drives the driver's body down over its anvil (a
3 mm stroke against a return spring), and a helical cam turns the bit 12 degrees as the stroke closes; 30 degrees
breaks the rust's hold. A hard blow on a bit that is not fully seated, or a driver held more than 5 degrees off
square, cams the bit out and damages the recess; a second cam-out strips it, and the screw then has to be drilled out.
Then it must put both tools down off the hub and let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import ContactPair, UsdBody, links_np, on_reset, on_step, on_substep, pose_np, set_target  # noqa: E402

import geometry as g  # noqa: E402
from make_assets import CAM_D, CAM_K, SCREW_CAP, SCREW_D, SCREW_K, SPRING_D, SPRING_REST  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Break the seized rotor screw loose with the impact driver and the hammer, then put both tools down "
               "off the hub and let go of them.")

# -- objects -------------------------------------------------------------------------------------------------------
HUB = UsdBody("hub", usd="assets/hub.usda", pos=tuple(float(v) for v in g.F0), size=(0.26, 0.26, 0.15),
              articulation=True, joint_pos=(("screw_a", 0.0),),
              drives=(("screw_.*", SCREW_K, SCREW_D, SCREW_CAP),),
              usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001)
BLOCK = UsdBody("block", usd="assets/block.usda", pos=(float(g.BLOCK_C[0]), float(g.BLOCK_C[1]), float(g.TABLE)),
                size=(g.BLOCK[0], g.BLOCK[0], g.BLOCK[1]), mass=None, usd_materials=True, contact_offset=0.001)
STAND = UsdBody("stand", usd="assets/stand.usda", pos=(float(g.STAND_C[0]), float(g.STAND_C[1]), float(g.TABLE)),
                size=(g.STAND[0], 0.15, g.STAND[1]), mass=None, usd_materials=True, contact_offset=0.001)
DRIVER = UsdBody("driver", usd="assets/driver.usda", pos=tuple(float(v) for v in g.DRIVER_START),
                 size=(0.039, 0.039, g.DRIVER_LEN), articulation=True, joint_pos=(("stroke", 0.0), ("cam", 0.0)),
                 drives=(("stroke", g.SPRING_K, SPRING_D, 1000.0), ("cam", CAM_K, CAM_D, 50.0)),
                 drive_targets=(("stroke", SPRING_REST),), usd_materials=True, self_collisions=False,
                 torsional_patch=0.02, contact_offset=0.001, solver_iters=32)
HAMMER = UsdBody("hammer", usd="assets/hammer.usda", pos=tuple(float(v) for v in g.HAMMER_START),
                 rot_xyzw=tuple(float(v) for v in g.quat_xyzw(g.HAMMER_STRIKE)), size=(0.1, 0.335, 0.03),
                 mass=g.HAMMER_MASS, usd_materials=True, torsional_patch=0.03, contact_offset=0.001, solver_iters=32)
OBJECTS = [HUB, BLOCK, STAND, DRIVER, HAMMER]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
OFF_TABLE = 0.05      # m: a tool's centre this far below the table top: it fell off the table
OFF_HUB = 0.005       # m: put down off the hub: every part of each tool this far from the hub, rotor, studs and screws
CLEAR = 0.02          # m: let go: every part of both grippers this far from both tools
STILL_V = 0.01        # m/s: both tools still

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


def tools_boxes(driver_pos, driver_rot, hammer_pos, hammer_rot):
    return g.driver_boxes(driver_pos, driver_rot) + g.hammer_parts(hammer_pos, hammer_rot)


# -- the blow (every physics substep) ----------------------------------------------------------------------------------
def face_vs_cap(body_pos, body_rot, hammer_pos, hammer_rot):
    """(axial gap from the hammer's face to the driver's cap along the driver's axis, allowing for the face's tilt;
    lateral offset of the face's centre from the axis; the face's tilt from square). ``body_pos``/``body_rot``: the
    driver's body frame."""
    a = np.asarray(body_rot, float)[:, 2]
    cap = np.asarray(body_pos, float) + a * g.DRIVER_LEN
    hr = np.asarray(hammer_rot, float)
    n = hr[:, 2]                                  # the face's outward normal
    face = np.asarray(hammer_pos, float) + n * g.FACE_TIP
    rel = face - cap
    h = float(rel @ a)
    lateral = float(np.linalg.norm(rel - h * a))
    tilt = float(math.acos(max(-1.0, min(1.0, float(-n @ a)))))
    return h - g.FACE_R * math.sin(tilt), lateral, tilt


def blow_step(body_pos, body_rot, tip, axis, stroke: float, hammer_pos, hammer_rot, impulse: float, dt: float):
    """One physics substep of the blow detector, on measured values: the driver's body frame, the bit's tip and axis
    (the anvil), the stroke (m), the hammer's pose, and the contact impulse between the hammer and the driver's body in
    this substep (N s, from PhysX). A blow is a contact passing at least J_MIN, after the face has been at least H_ARM
    off the cap; its speed is the fastest the gap closed over the substeps just before. It is judged once the stroke
    has had WINDOW substeps to bottom. Returns the blow once judged (and applies it), else None."""
    S = _STATE
    h, lateral, tilt = face_vs_cap(body_pos, body_rot, hammer_pos, hammer_rot)
    hist = S["hist"]
    if hist:
        hist[-1]["v"] = (hist[-1]["h"] - h) / dt
    hist.append({"h": h, "v": 0.0, "tip": np.asarray(tip, float), "axis": np.asarray(axis, float), "stroke": stroke})
    del hist[:-8]
    S["max_impulse"] = max(S["max_impulse"], impulse)
    pend = S["pending"]
    if pend is not None:
        pend["stroke"] = max(pend["stroke"], stroke)
        pend["impulse"] = max(pend["impulse"], impulse)
        pend["n"] += 1
        if pend["n"] < g.WINDOW:
            return None
        S["pending"] = None
        return _judge(pend)
    if h > g.H_ARM:
        S["armed"] = True
    if not (S["armed"] and impulse >= g.J_MIN):
        return None
    S["armed"] = False
    speed = max(e["v"] for e in hist[:-1]) if len(hist) > 1 else 0.0
    before = hist[-2] if len(hist) > 1 else hist[-1]     # the bit as it was just before the blow
    a = np.asarray(body_rot, float)[:, 2]
    rel = (np.asarray(hammer_pos, float) + np.asarray(hammer_rot, float)[:, 2] * g.FACE_TIP
           - np.asarray(body_pos, float) - a * g.DRIVER_LEN)
    S["pending"] = {"step": S["step"], "speed": speed, "lateral": lateral, "tilt": tilt, "tip": before["tip"],
                    "stroke0": before["stroke"], "held": bool(S["held"]),
                    "face_minus_cap_mm": [round(float(v) * 1000, 1) for v in rel],
                    "axis": before["axis"], "stroke": stroke, "impulse": impulse, "n": 0}
    return None


def _judge(pend: dict) -> dict:
    S = _STATE
    kind, name, det = g.blow_outcome(pend["speed"], pend["lateral"], pend["tilt"], pend["tip"], pend["axis"],
                                     S["screws"], S["ruined"], pend["stroke"] - pend["stroke0"], pend["held"])
    blow = {"step": pend["step"], "kind": kind, "screw": name, "speed": round(pend["speed"], 3),
            "impulse": round(pend["impulse"], 3), "lateral_mm": round(pend["lateral"] * 1000, 1),
            "face_tilt_deg": round(math.degrees(pend["tilt"]), 1), "stroke_mm": round(pend["stroke"] * 1000, 2),
            "stroke_before_mm": round(pend["stroke0"] * 1000, 2), "held": pend["held"],
            "face_minus_cap_mm": pend["face_minus_cap_mm"], **det}
    S["blows"].append(blow)
    if kind == "good":
        S["turned"][name] += g.TURN_PER_BLOW
    elif kind == "camout":
        S["camouts"][name] += 1
        S["ruined"][name] = S["camouts"][name] >= g.RUIN
    return blow


# -- the monitor (every control step) ----------------------------------------------------------------------------------
def level(turned: dict, camouts: dict, done: bool) -> float:
    """Score: the screw is worth 0.8 x the fraction of the 30 degree breakaway turned, less 0.1 per cam-out (0 once
    stripped by two); 0.2 more once it is loose and the tools are down off the hub, let go and still."""
    s = sum(g.screw_value(turned[k], camouts[k]) for k in g.SCREWS)
    loose = all(turned[k] >= g.LOOSE - 1e-9 and camouts[k] < g.RUIN for k in g.SCREWS)
    return round(s + (0.2 if loose and done else 0.0), 4)


def reset_state() -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, step=0, hist=[], armed=False, blows=[], pending=None, max_impulse=0.0,
                  held=False,
                  turned={k: 0.0 for k in g.SCREWS}, camouts={k: 0 for k in g.SCREWS},
                  ruined={k: False for k in g.SCREWS}, screws={k: g.screw_world(k) for k in g.SCREWS},
                  off_hub=False, clear=False, still=False, level=0.0, gaps=None)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(hub_pos, hub_rot, driver_pos, driver_rot, driver_vel, hammer_pos, hammer_rot, hammer_vel, grips) -> dict:
    """One control step of the monitor, on measured values (station frame). ``grips``: each gripper as boxes. Returns
    the screws' joint targets (rad) for the sim."""
    S = _STATE
    S["step"] += 1
    hp, hr = np.asarray(hub_pos, float), np.asarray(hub_rot, float)
    S["screws"] = {k: g.screw_world(k, hp, hr) for k in g.SCREWS}
    dp, dr = np.asarray(driver_pos, float), np.asarray(driver_rot, float)
    mp, mr = np.asarray(hammer_pos, float), np.asarray(hammer_rot, float)
    if dp[2] + dr[2, 2] * g.DRIVER_COM < g.TABLE - OFF_TABLE:
        _flag("the impact driver fell off the table")
    if (mp + mr @ g.HAMMER_COM)[2] < g.TABLE - OFF_TABLE:
        _flag("the hammer fell off the table")
    S["poses"] = {"driver_tip": [round(v, 4) for v in dp], "driver_axis": [round(v, 3) for v in dr[:, 2]],
                  "hammer": [round(v, 4) for v in mp], "hammer_face_dir": [round(v, 3) for v in mr[:, 2]],
                  "hammer_handle_dir": [round(v, 3) for v in mr[:, 1]]}
    tools = tools_boxes(dp, dr, mp, mr)
    S["off_hub"] = g.gap_to_fixture(tools, hp, hr) >= OFF_HUB
    body = g.driver_boxes(dp, dr)[-2:]             # the knurled body and the cap
    S["held"] = min(g.boxes_gap(a, b, 5) for grip in grips for a in grip for b in body) <= g.HELD
    gaps = [min(g.boxes_gap(a, b, 5) for a in grip for b in tools) for grip in grips]
    S["gaps"] = [round(x * 1000, 1) for x in gaps]
    S["clear"] = all(x >= CLEAR for x in gaps)
    S["still"] = bool(float(np.linalg.norm(driver_vel)) < STILL_V and float(np.linalg.norm(hammer_vel)) < STILL_V)
    done = S["off_hub"] and S["clear"] and S["still"]
    S["level"] = 0.0 if S["violation"] else level(S["turned"], S["camouts"], done)
    return {k: float(S["turned"][k]) for k in g.SCREWS}


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


_SIM: dict = {}


@on_reset
def _reset(environment=None) -> None:
    reset_state()
    if environment is not None and "contact" not in _SIM:
        try:
            _SIM["contact"] = ContactPair(environment, "hammer", "body")
            _SIM["cam"] = environment.unwrapped.scene["driver"].find_joints(["cam"], preserve_order=True)[0]
        except Exception as e:  # noqa: BLE001 - reported in the state (and the demo's log)
            _SIM["error"] = repr(e)
            _STATE["why"].append(f"contact view failed: {e!r}")


@on_substep
def _substep(environment) -> None:
    from scenekit import set_target_fast

    if "contact" not in _SIM:
        return
    links, joints = links_np(environment, "driver")
    b, a, m = links["body"], links["anvil"], pose_np(environment, "hammer")
    stroke = joints["stroke"]
    dt = float(environment.unwrapped.physics_dt)
    impulse = float(np.linalg.norm(_SIM["contact"].force())) * dt
    blow_step(b[:3], _rot(b[3:7]), a[:3], _rot(a[3:7])[:, 2], stroke, m[:3], _rot(m[3:7]), impulse, dt)
    # the helical cam: the anvil (and the bit) turns with the stroke, counter-clockwise seen from the cap
    set_target_fast(environment, "driver", _SIM["cam"], g.CAM * stroke)


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["hub", "driver", "hammer"])
    hb, dv, hm = st["hub"], st["driver"], st["hammer"]
    targets = update(hb.pos, _rot(hb.quat_xyzw), dv.pos, _rot(dv.quat_xyzw), dv.lin_vel, hm.pos, _rot(hm.quat_xyzw),
                     hm.lin_vel, _grips(environment))
    for k, v in targets.items():
        set_target(environment, "hub", f"screw_{k}", v)


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """The screw turned at least 30 degrees by good blows with no cam-out; both tools down off the hub, both grippers
    2 cm from them, the tools still. Held for 1 s."""
    return bool(not violation() and _STATE["level"] >= 1.0 - 1e-9)


def score(objects) -> float:
    """0 if a tool fell off the table. Otherwise, at the end: 0.8 x min(turned / 30 degrees, 1) less 0.1 per
    cam-out (0 if two cam-outs stripped it), plus 0.2 once the screw is loose and the tools are down off
    the hub and let go. At most 0.9 without success (1.0)."""
    return 0.0 if violation() else min(0.9, float(_STATE["level"]))
