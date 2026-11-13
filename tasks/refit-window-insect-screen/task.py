"""On the station table, straight ahead of the arms: a cropped section of a small window frame, fixed upright, with
a deep top channel in its head and a shallow bottom channel in its sill. Its insect screen (450 x 600 mm, aluminium
frame, two spring plungers on its top edge) stands on the table in front of it, leaning back against the jambs, as
left after cleaning.

The robot must put the screen back: push its top edge up into the top channel, compressing the plungers, swing the
bottom in over the sill's lip, and let it down into the bottom channel; then let go. The plungers then hold it.

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

INSTRUCTION = "Put the insect screen back in the window, then let go of it."

# -- objects -------------------------------------------------------------------------------------------------------
_O0, _R0 = g.start_pose()
SCREEN = UsdBody("screen", usd="assets/screen.usda", pos=tuple(float(v) for v in g.to_station(_O0)),
                 rot_xyzw=g.quat_xyzw(_R0), size=(g.FRAME_T, g.SCREEN_W, g.SCREEN_H), mass=g.SCREEN_MASS,
                 articulation=True, floating=True,
                 joint_pos=(("plunger_left", 0.0), ("plunger_right", 0.0)),
                 drives=(("plunger_.*", g.PIN_RATE, g.PIN_DAMPING),),
                 drive_targets=(("plunger_left", g.PIN_TARGET), ("plunger_right", g.PIN_TARGET)),
                 usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001)

_COLOURS = {"glass": (0.70, 0.82, 0.86)}
FRAME = [Shape(f"window_{name}", pos=tuple(float(v) for v in g.to_station((np.asarray(lo) + np.asarray(hi)) / 2)),
               size=tuple(float(b - a) for a, b in zip(lo, hi)), color=_COLOURS.get(name, (0.93, 0.93, 0.91)))
         for name, (lo, hi) in g.FRAME_BOXES.items()]
OBJECTS = [SCREEN, *FRAME]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
TOP_IN = 0.004                  # m: top edge at least this far above the top channel's mouth (inside it)
PUSHED = 0.008                  # m: 0.25: both plungers pushed in at least this far (fitted they are in 3 mm)
HOLDING = 0.0015                # m: fitted: both plungers pushed in at least this far (they bear on the ceiling)
SEAT_Z = (-0.003, 0.004)        # m: fitted: bottom edge's corners this close to the bottom channel's floor
OVER_Z = (-0.003, 0.024)        # m: 0.5: bottom edge's corners between the floor and 24 mm above it ...
BOT_X = g.BOT_HALF + 0.001      # m: ... and within this of the channel's centre line (inside its lips, 1 mm slack)
TOP_X = g.TOP_HALF + 0.001      # m: top edge's corners within this of the top channel's centre line
UPRIGHT = math.radians(3)       # fitted: the screen within 3 degrees of the window's plane
OVER_TILT = math.radians(6)     # 0.5
PUSH_TILT = math.radians(15)    # 0.25
CENTRE_Y = 0.015                # m: fitted: centred on the opening within 15 mm
CLEAR = 0.02                    # m: every part of both grippers this far from the screen (let go)
STILL_V, STILL_W = 0.01, 0.1    # m/s and rad/s: the screen still
STILL_PIN = 0.0005              # m per control step (1 cm/s): plungers still
OFF_TABLE = 0.02                # m: any corner of the screen this far below the table top: it left the table
FORCED = 0.004                  # m: any part of the screen's frame this far inside the window frame: forced into it
FALL_V = 0.5                    # m/s: the screen's centre falling this fast with no gripper holding it: dropped

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


# -- the screen in the window's frame -------------------------------------------------------------------------------
def tilt(r) -> float:
    """Largest angle between the screen's axes and the fitted screen's (stiles vertical, face square to the window,
    either face toward the room)."""
    r = np.asarray(r, float)
    # abs(r[0, 0]): the screen is the same either way round, so one turned front-to-back fits as well
    return max(math.acos(min(1.0, max(-1.0, float(r[2, 2])))), math.acos(min(1.0, abs(float(r[0, 0])))))


def top_in(o, r) -> bool:
    top = g.edge_corners(o, r, g.SCREEN_H)
    return bool(top[:, 2].min() >= g.MOUTH + TOP_IN and np.abs(top[:, 0]).max() <= TOP_X)


def bottom_ok(o, r, zr) -> bool:
    bot = g.edge_corners(o, r, 0.0)
    return bool(zr[0] <= bot[:, 2].min() and bot[:, 2].max() <= zr[1] and np.abs(bot[:, 0]).max() <= BOT_X)


def stage(o, r, press) -> tuple[bool, bool, bool]:
    """(pushed, over, fitted) for the screen pose (o, R) in W and the plungers' press (left, right; positive = in)."""
    t = tilt(r)
    inside = top_in(o, r)
    pushed = inside and t <= PUSH_TILT and min(press) >= PUSHED
    over = inside and t <= OVER_TILT and bottom_ok(o, r, OVER_Z)
    fitted = (inside and t <= UPRIGHT and bottom_ok(o, r, SEAT_Z) and abs(float(o[1])) <= CENTRE_Y
              and min(press) >= HOLDING)
    return pushed, over, fitted


def level_of(pushed: bool, over: bool, fitted: bool) -> float:
    return 0.75 if fitted else 0.5 if over else 0.25 if pushed else 0.0


def forced_depth(o, r) -> float:
    """How far any part of the screen's frame (rails) lies inside the window frame (the deepest sample point)."""
    frame = list(g.frame_boxes_w().values())
    deepest = 0.0
    for name, box in g.screen_boxes(o, r, pins=False, mesh=False).items():
        n = 21 if name in ("bottom_rail", "top_rail") or name.startswith("stile") else 7
        pts = _rail_points(box, n)
        for fb in frame:
            deepest = max(deepest, float(g.point_box_depth(pts, *fb).max()))
    return deepest


def _rail_points(box, n: int) -> np.ndarray:
    """Points on a long box's surface: n along its longest axis, 3 across the others."""
    c, r, h = box
    axes = [np.linspace(-1, 1, n if k == int(np.argmax(h)) else 3) for k in range(3)]
    grid = np.array([[a, b, d] for a in axes[0] for b in axes[1] for d in axes[2]])
    grid = grid[np.abs(grid).max(axis=1) == 1.0]
    return np.asarray(c) + (grid * np.asarray(h)) @ np.asarray(r).T


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, level=0.0, fitted=False, clear=False, still=False, press=(0.0, 0.0),
                  info={})


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(screen_pos, screen_rot, joints: dict, vel, w, arms: dict) -> None:
    """One control step of the monitor. ``screen_pos``/``screen_rot``: the screen frame's origin and rotation in the
    station frame; ``joints``: the plunger joints (m, negative = pushed in); ``arms``: side -> {"tcp", "rot",
    "opening"} from the arm's encoders. Everything is judged in the window's frame (it is fixed to the table)."""
    S = _STATE
    o = np.asarray(screen_pos, float) - g.WINDOW
    r = np.asarray(screen_rot, float)
    press = (max(0.0, -float(joints.get("plunger_left", 0.0))), max(0.0, -float(joints.get("plunger_right", 0.0))))

    screen = list(g.screen_boxes(o + g.WINDOW, r, (-press[0], -press[1])).values())
    grips = [gripper_boxes(a["tcp"], a["rot"], a["opening"]) for a in arms.values()]
    gaps = [gap_to(b, screen) for b in grips]
    clear = all(d >= CLEAR for d in gaps)

    # -- violations (latched for the whole trial) ---------------------------------------------------------------
    # dropped: the centre of mass falling faster than FALL_V (a free fall of about 1.3 cm) while neither gripper is
    # within CLEAR of the screen; carrying it down, setting it down or leaning it is fine
    com_v = np.asarray(vel, float) + np.cross(np.asarray(w, float), r @ np.array([0.0, 0.0, g.SCREEN_H / 2]))
    if clear and com_v[2] < -FALL_V:
        _flag("the screen was dropped")
    corners = g.box_corners(*g.lohi((-g.FRAME_T / 2, -g.SCREEN_W / 2, 0.0), (g.FRAME_T / 2, g.SCREEN_W / 2,
                                                                            g.SCREEN_H)))
    low = float((o + corners @ r.T)[:, 2].min())
    if low < -g.SILL_H - OFF_TABLE:
        _flag("the screen fell off the table")
    depth = forced_depth(o, r)
    if depth > FORCED:
        _flag("the screen was forced into the window frame")

    # -- progress ------------------------------------------------------------------------------------------------
    pushed, over, fitted = stage(o, r, press)
    lvl = level_of(pushed, over, fitted)
    p0 = S["press"]
    still = (float(np.linalg.norm(vel)) < STILL_V and float(np.linalg.norm(w)) < STILL_W
             and abs(press[0] - p0[0]) < STILL_PIN and abs(press[1] - p0[1]) < STILL_PIN)
    bot = g.edge_corners(o, r, 0.0)
    top = g.edge_corners(o, r, g.SCREEN_H)
    S.update(level=lvl, fitted=fitted, clear=clear, still=still, press=press,
             info={"press_mm": [round(p * 1000, 1) for p in press], "tilt_deg": round(math.degrees(tilt(r)), 2),
                   "top_mm": [round((top[:, 2].min() - g.MOUTH) * 1000, 1), round(np.abs(top[:, 0]).max() * 1000, 1)],
                   "bottom_mm": [round(bot[:, 2].min() * 1000, 1), round(bot[:, 2].max() * 1000, 1),
                                 round(np.abs(bot[:, 0]).max() * 1000, 1)],
                   "y_mm": round(float(o[1]) * 1000, 1), "forced_mm": round(depth * 1000, 1)})


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["screen"])["screen"]
    update(st.pos, _rot(st.quat_xyzw), st.joints, st.lin_vel, st.ang_vel, _arms(environment))


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """No violation so far; the screen fitted (top edge in the top channel, bottom edge down in the bottom channel,
    within 3 degrees of the window's plane, centred, both plungers bearing on the channel's ceiling); every part of
    both grippers at least 2 cm from the screen; the screen and its plungers still. Held for 1 s."""
    S = _STATE
    return bool(not violation() and S["fitted"] and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: the screen dropped (its centre falling faster than 0.5 m/s with no gripper within 2 cm),
    left the table (a corner 2 cm below its top) or was forced into the window frame (4 mm). Otherwise the stage at the end: 0.25 the top edge in the top channel with both plungers pushed in
    8 mm or more; 0.5 the top edge in the channel and the bottom edge over the bottom channel, inside its lips; 0.75
    fitted but still touched or held. Success (fitted, released, still, held 1 s) scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
