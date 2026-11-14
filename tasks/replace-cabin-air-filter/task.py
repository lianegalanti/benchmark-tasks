"""On the station table, straight ahead of the arms: a section of a car's passenger-side dashboard, fixed to the table,
with its glovebox open and resting on its two stops. The glovebox (a 36 cm wide bin on a hinge along the bottom of
its opening) can only swing further down, past the stops, while both of its side walls are squeezed in. Behind it,
in the face of the HVAC housing, is the cabin air filter's slot holding the used filter. The new filter lies on the
table to the right.

The robot must replace the cabin air filter: lower the glovebox past its stops, take the old filter out and set it
down on the table, slide the new one fully into the slot, and swing the glovebox back up past its stops.

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

INSTRUCTION = ("Take the glovebox down, replace the cabin air filter behind it with the new one from the table, and put "
               "the glovebox back up.")

# -- objects -------------------------------------------------------------------------------------------------------
START_ANGLE = g.OPEN_STOP - math.radians(0.5)   # just short of the stops; the bin settles onto them
DASH = UsdBody("dash", usd="assets/dash.usda", pos=tuple(float(v) for v in g.dash_to_world((0, 0, 0))),
               size=(0.4, 2 * g.DASH_HALF_W, g.HINGE_Z + g.DASH_TOP), articulation=True,
               joint_pos=(("hinge", float(START_ANGLE)), ("pin_left", 0.0), ("pin_right", 0.0)),
               drives=(("hinge", 0.0, g.HINGE_DAMPING), ("pin_.*", g.PIN_RATE, g.PIN_DAMPING)),
               drive_targets=(("pin_left", float(g.PIN_TARGET)), ("pin_right", float(g.PIN_TARGET))),
               usd_materials=True, self_collisions=True, solver_iters=32, contact_offset=0.002)
OLD = UsdBody("old_filter", usd="assets/filter_old.usda", pos=tuple(float(v) for v in g.OLD_START),
              size=g.F_SIZE, mass=g.F_MASS, friction=(0.4, 0.35), torsional_patch=0.015)
NEW = UsdBody("new_filter", usd="assets/filter_new.usda",
              pos=tuple(float(v) for v in g.NEW_START + [0.0, 0.0, 0.0005]), size=g.F_SIZE, mass=g.F_MASS,
              friction=(0.4, 0.35), torsional_patch=0.015)
STAND = Shape("parts_bin", pos=(float(g.STAND[0][0]), float(g.STAND[0][1]), float(g.TABLE + g.STAND[1][2] / 2)),
              size=tuple(float(v) for v in g.STAND[1]), color=(0.15, 0.30, 0.55))
OBJECTS = [DASH, OLD, NEW, STAND]
FILTERS = {"old": "old_filter", "new": "new_filter"}

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
PASSED = math.radians(72.0)     # the bin is lowered past both stops: both bumps wholly in front of the fascia
UP = math.radians(1.5)          # back up: within this of the stops' angle (resting on them) or above ...
PIN_OUT = 0.002                 # ... with both flexing panels sprung back out (within 2 mm)
FORCED = 0.003                  # m: a bump this far into its stop block was forced through it (the stop broken)
HINGE_OVER = math.radians(3.0)  # the bin forced this far past its lowered hinge stop
SEAT_DEPTH = 0.003              # m: fully in: the filter's front face at most this far proud of its seated position
SEAT_SIDE = 0.004               # m: and within this of the slot's centre line and floor
SEAT_TILT = math.radians(3)     # and level and square within this
REST = 0.010                    # m: set down: the old filter's lowest point this close to the table top
OFF_TABLE = 0.05                # m: a filter's centre this far below the table top has fallen off it
LET_GO = 0.02                   # m: a filter is let go when every part of both grippers is at least this far from it
CLEAR = 0.02                    # m: released: every part of both grippers this far from the bin and both filters
STILL_V = 0.01                  # m/s: filters still; the bin turns less than STILL_BIN per control step
STILL_BIN = math.radians(0.2)
STILL_PIN = 0.0005              # m per control step

_STATE: dict = {}


# -- the grippers as boxes, from each arm's encoders --------------------------------------------------------------
def gripper_boxes(tcp, rot, opening: float):
    """The gripper as boxes (centre, rotation, half sizes) in the station frame: the two fingers (the bounds of their
    collision meshes, from the URDF: 13 mm past the TCP back to 81 mm behind it, 68 mm wide, spreading along the TCP
    frame's x axis as they open) and the housing behind them. ``opening`` is the gap between the pads."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    half = opening / 2
    local = [(np.array([0.012 + half, 0.0, -0.034]), np.array([0.036, 0.034, 0.047])),
             (np.array([-0.012 - half, 0.0, -0.034]), np.array([0.036, 0.034, 0.047])),
             (np.array([0.0, 0.0, -0.1075]), np.array([0.051, 0.0325, 0.037]))]
    return [(tcp + rot @ c, rot, h) for c, h in local]


def gap_to(boxes_a, boxes_b) -> float:
    return min(g.boxes_gap(a, b) for a in boxes_a for b in boxes_b)


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


def tilt_of(rot) -> float:
    return math.acos(max(-1.0, min(1.0, float(np.asarray(rot)[2, 2]))))


def yaw_of(rot) -> float:
    rot = np.asarray(rot)
    return math.atan2(rot[1, 0], rot[0, 0])


# -- the stops: how far each panel must be pushed in for its bump to clear its block, by hinge angle ------------------
def _needed(theta: float) -> float:
    """Smallest panel travel (m) at which the left bump clears its stop block at hinge angle ``theta`` (the right
    side is the mirror image)."""
    stop = [b for b in g.dash_world() if b[0] == "fascia_left"]

    def bump(d):
        return g.bump_parts(theta, d)

    if g.gap(bump(0.0), stop) >= 0:
        return 0.0
    lo, hi = 0.0, g.PIN_TRAVEL
    for _ in range(12):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if g.gap(bump(mid), stop) >= 0 else (mid, hi)
    return hi


NEED_GRID = np.arange(g.OPEN_STOP - math.radians(4.0), g.OPEN_STOP + math.radians(40.0), math.radians(0.5))
NEED = np.array([_needed(float(t)) for t in NEED_GRID])


def needed(theta: float) -> float:
    if theta <= NEED_GRID[0] or theta >= NEED_GRID[-1]:
        return 0.0
    return float(np.interp(theta, NEED_GRID, NEED))


# -- the monitor ----------------------------------------------------------------------------------------------------
def tab_point(pos, rot) -> np.ndarray:
    return np.asarray(pos, float) + np.asarray(rot, float) @ g.GRIP


def lowest_z(pos, rot) -> float:
    return min(float(g.box_points(c, r, h, 2)[:, 2].min()) for _, c, r, h in g.filter_world(pos, rot))


def out_of_dash(pos, rot) -> bool:
    """The filter's whole footprint lies outside the dash section's."""
    pts = np.vstack([g.box_points(c, r, h, 2) for _, c, r, h in g.filter_world(pos, rot)])
    x0 = g.X_HINGE + g.FASCIA[0]
    inside = (pts[:, 0] > x0) & (np.abs(pts[:, 1]) < g.DASH_HALF_W)
    return not bool(inside.any())


SEAT_WORLD = g.dash_to_world(g.SEAT_C)


def seated(pos, rot) -> bool:
    d = np.asarray(pos, float) - SEAT_WORLD
    return bool(-0.002 <= d[0] <= SEAT_DEPTH and abs(d[1]) <= SEAT_SIDE and abs(d[2]) <= SEAT_SIDE
                and tilt_of(rot) <= SEAT_TILT and abs(yaw_of(rot)) <= SEAT_TILT)


def level(lowered: bool, old_out: bool, new_in: bool) -> float:
    """Partial credit at the end: 0.25 the glovebox has been lowered past both stops; 0.5 also the old filter out on
    the table; 0.75 also the new filter fully in."""
    if not lowered:
        return 0.0
    if not old_out:
        return 0.25
    return 0.75 if new_in else 0.5


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, lowered=False, level=0.0, up=False, clear=False, still=False,
                  theta=float(START_ANGLE), pins=(0.0, 0.0), held={"old": False, "new": False}, info={})


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(theta: float, pins, filters: dict, arms: dict) -> None:
    """One control step of the monitor. ``theta``: hinge angle; ``pins``: (left, right) panel travel; ``filters``:
    name -> (pos, rot, velocity); ``arms``: side -> {"tcp", "rot", "opening"} from the arm's encoders. The dash
    section is fixed to the table, so its frame is the station frame shifted."""
    S = _STATE
    pl, pr = float(pins[0]), float(pins[1])
    # -- violations (latched for the whole trial) ---------------------------------------------------------------
    need = needed(theta)
    if need > 0 and min(pl, pr) < need - FORCED:
        _flag("the glovebox was forced past a stop it wasn't squeezed clear of")
    if theta > g.LOWERED + HINGE_OVER:
        _flag("the glovebox was forced past its hinge stop")
    status = {}
    for name, (pos, rot, vel) in filters.items():
        pos, rot, vel = np.asarray(pos, float), np.asarray(rot, float), np.asarray(vel, float)
        fb = [b[1:] for b in g.filter_world(pos, rot)]
        held = any(gap_to(gripper_boxes(a["tcp"], a["rot"], a["opening"]), fb) < LET_GO for a in arms.values())
        S["held"][name] = held
        if pos[2] < g.TABLE - OFF_TABLE:
            _flag(f"the {name} filter fell off the table")
        status[name] = (pos, rot, float(np.linalg.norm(vel)), held)

    # -- progress -------------------------------------------------------------------------------------------------
    if theta >= PASSED and not S["violation"]:
        S["lowered"] = True
    pos, rot, _, held = status["old"]
    old_out = out_of_dash(pos, rot) and lowest_z(pos, rot) <= g.TABLE + REST and not held
    pos, rot, _, held = status["new"]
    new_in = seated(pos, rot) and not held
    up = theta <= g.OPEN_STOP + UP and max(pl, pr) <= PIN_OUT
    parts = g.bin_world(theta, (pl, pr)) + [b for p, r, _, _ in status.values() for b in g.filter_world(p, r)]
    parts = [b[1:] for b in parts]
    clear = all(gap_to(gripper_boxes(a["tcp"], a["rot"], a["opening"]), parts) >= CLEAR for a in arms.values())
    still = (all(s < STILL_V for _, _, s, _ in status.values()) and abs(theta - S["theta"]) < STILL_BIN
             and abs(pl - S["pins"][0]) < STILL_PIN and abs(pr - S["pins"][1]) < STILL_PIN)
    new_d = (status["new"][0] - SEAT_WORLD) * 1000
    S.update(level=level(S["lowered"], old_out, new_in), up=up, clear=clear, still=still, theta=float(theta),
             pins=(pl, pr),
             info={"theta_deg": round(math.degrees(theta), 1), "pins_mm": [round(pl * 1000, 1), round(pr * 1000, 1)],
                   "new_mm": np.round(new_d, 1).tolist(),
                   "old_mm": np.round((status["old"][0] - SEAT_WORLD) * 1000, 1).tolist(), "old_out": bool(old_out), "new_in": bool(new_in),
                   "held": dict(S["held"])})


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["dash", *FILTERS.values()])
    j = st["dash"].joints
    filters = {k: (st[v].pos, _rot(st[v].quat_xyzw), st[v].lin_vel) for k, v in FILTERS.items()}
    update(float(j.get("hinge", START_ANGLE)), (float(j.get("pin_left", 0.0)), float(j.get("pin_right", 0.0))),
           filters, _arms(environment))


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """No violation so far; the glovebox was lowered past its stops; the old filter is out, set down on the table and
    let go; the new one is fully in and let go; the glovebox is back up on its stops; every part of both grippers is
    at least 2 cm from the glovebox and both filters; everything still. Held 1 s."""
    S = _STATE
    return bool(not violation() and S["level"] >= 0.75 and S["up"] and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: the glovebox forced past a stop it wasn't squeezed clear of or past its hinge stop, or a
    filter fallen off the table. Otherwise, at the end: 0.25 if the glovebox was lowered past both stops; 0.5 if the
    old filter is also out, set down on the table and let go (every gripper part 2 cm or more from it); 0.75 if the new one is also fully in and let go.
    Success scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
