"""On the station table, a guide track runs across the table from right to left (toward the left arm), 33 cm in front
of the arms. A grey cart (18 x 18 cm, 4 cm tall) rides on it with a nest on its top: four black guide plates
round a square pocket, leaning outward like a funnel. In the nest sits a full tray: a blue square tray (11 x 11 cm, 3.5 cm deep) with a yellow handle rib
standing across its middle, 4 cm above its rim, and an orange aluminium block in each of its two compartments. The
cart starts at the table's right end with the trial and passes in front of both arms at a constant 6 cm/s without
stopping; a red end line is taped across the table on both sides of the track, 46 cm left of the midpoint between
the arms, and past the table's left edge the track carries the cart away. An identical empty tray stands on a low
grey stand in front of the left arm, and a low grey outfeed table stands in front of the right arm.

The robot must swap the trays while the cart goes by: take the full tray off the cart onto the outfeed table, with
both blocks still in it, and put the empty tray in the cart's nest, seated flat in the pocket and let go, before the
cart reaches the end line.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "A cart is passing in front of you with a full tray in its nest, and it won't stop. ..." ...
"""

from __future__ import annotations

import math
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import geometry as g  # noqa: E402
import numpy as np  # noqa: E402
from motionkit import DrivenBox, UsdBody, belt_run, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import TABLE_CENTER, TABLE_SIZE  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
# "end": the nest is judged when the cart reaches the end line, which the monitor below latches at that moment
# (cleared at every episode start); the outfeed table is judged at the end of the trial.
SCORE_OVER = "end"
TABLE_LEFT_Y = TABLE_CENTER[1] + TABLE_SIZE[1] / 2  # 0.65: the table's left edge (y is to the robot's left)


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the track and the cart ----------------------------------------------------------------------------------------
LINE_X = 0.33  # the track's centre line, metres in front of the arm bases
BED = (0.18, 1.24, 0.01)  # the fixed track bed: width (x), length (y), height; it ends 3 cm inside each table edge
CART_Z = TOP + BED[2] + 0.002 + g.CART[2] / 2  # 2 mm above the bed: the cart rides, it doesn't rub
CART_TOP = CART_Z + g.CART[2] / 2  # TOP + 0.052
NEST_TOP = CART_TOP + g.GUIDE_TOP  # the funnel's top: TOP + 0.075
SPEED = 0.06  # m/s, toward +y (the left arm)
RAMP_S = 0.5  # the cart speeds up from rest over this long at the start of the trial
START_Y = -0.53  # the cart's centre at the start: its right edge 3 cm inside the table's right edge, at the bed's end
END_Y = 0.46  # the end line: the nest is judged when the cart's centre reaches it
TURN_Y = TABLE_LEFT_Y + g.CART[1] / 2 + 0.02  # 0.75: past the table's edge, the track turns down out of the way
DROP, DROP_SPEED = 0.30, 0.15
PATH = belt_run((LINE_X, START_Y, CART_Z), SPEED, RAMP_S, TURN_Y, DROP, DROP_SPEED)


def _offset(path, d):
    """``path`` shifted by ``d``: a guide plate moves exactly with its cart."""
    d = np.asarray(d, float)
    return lambda t: (path(t)[0] + d, path(t)[1])


BED_BOX = Box("track_bed", pos=(LINE_X, 0.0, TOP + BED[2] / 2), size=BED, color=(0.08, 0.08, 0.09))
CART_OBJ = DrivenBox("cart", pos=_floats((LINE_X, START_Y, CART_Z)), size=g.CART, color=(0.55, 0.57, 0.6), path=PATH)
NEST = [
    DrivenBox(f"nest_{side}", pos=_floats((LINE_X + dx, START_Y + dy, CART_TOP + dz)), size=size,
              color=(0.1, 0.1, 0.1), path=_offset(PATH, (dx, dy, CART_TOP + dz - CART_Z)), rot_xyzw=_floats(quat))
    for side, (dx, dy, dz), size, quat in g.nest_guides()
]


def cart_y(t: float) -> float:
    """Where the cart's centre is along the track at episode time ``t``."""
    return float(PATH(t)[0][1])


def arrival(y: float) -> float:
    """When the cart's centre reaches ``y`` (after the ramp; it runs at constant speed then)."""
    return float((y - START_Y) / SPEED + RAMP_S / 2)


# -- the end line (red tape on the table, either side of the track) --------------------------------------------------
TAPE = (0.06, 0.012, 0.002)
TAPES = [
    Box(f"end_tape_{side}", pos=(LINE_X + sign * (BED[0] / 2 + 0.005 + TAPE[0] / 2), END_Y, TOP + TAPE[2] / 2),
        size=TAPE, color=(0.9, 0.05, 0.05))
    for side, sign in (("near", -1), ("far", 1))
]

# -- the stand (empty tray) and the outfeed table (full tray) --------------------------------------------------------
PAD = (0.15, 0.15, 0.03)  # both: low grey boxes
STAND_XY, OUT_XY = (0.14, 0.37), (0.14, -0.37)
PAD_TOP = TOP + PAD[2]
STAND = Box("stand", pos=(*STAND_XY, TOP + PAD[2] / 2), size=PAD, color=(0.42, 0.43, 0.45))
OUTFEED = Box("outfeed", pos=(*OUT_XY, TOP + PAD[2] / 2), size=PAD, color=(0.42, 0.43, 0.45))

# -- the trays and the parts -----------------------------------------------------------------------------------------
SPAWN_GAP = 0.0005  # everything free spawns this far above what it rests on, and settles during the reset
TRAY_SIZE = (g.TRAY, g.TRAY, g.RIB_TOP)
FULL_Z = CART_TOP - g.UNDER + SPAWN_GAP
EMPTY_Z = PAD_TOP - g.UNDER + SPAWN_GAP
FULL = UsdBody("tray_full", pos=_floats((LINE_X, START_Y, FULL_Z)), size=TRAY_SIZE, mass=g.TRAY_MASS,
               usd="assets/tray.usda", torsional_patch=0.008)
EMPTY = UsdBody("tray_empty", pos=_floats((*STAND_XY, EMPTY_Z)), size=TRAY_SIZE, mass=g.TRAY_MASS,
                usd="assets/tray.usda", torsional_patch=0.008)
PARTS = ("part_a", "part_b")  # a on the right of the rib (-y), b on its left
PART_Z = FULL_Z + g.FLOOR_TOP + g.PART[2] / 2 + SPAWN_GAP
PART_OBJS = [
    Box(n, pos=_floats((LINE_X, START_Y + s * g.POCKET_C, PART_Z)), size=g.PART, mass=g.PART_MASS,
        color=(0.95, 0.45, 0.1))
    for n, s in zip(PARTS, (-1, 1), strict=True)
]

OBJECTS = [BED_BOX, *TAPES, STAND, OUTFEED, CART_OBJ, *NEST, FULL, EMPTY, *PART_OBJS]

# -- the monitor -----------------------------------------------------------------------------------------------------
CENTRE_M = 0.015  # a seated tray's centre within this (horizontally) of the cart's centre
SEAT_M = 0.005  # its underside within this of the cart's top
TILT_DEG = 5.0  # seated: level within this
UPRIGHT_DEG = 10.0  # on the outfeed table, or carried: level within this
OUT_SEAT_M = 0.01  # on the outfeed table: its underside within this of the table's top
ON_CART_M = 0.03  # on the cart, not seated: its underside within this above the cart's top
RIDE_MPS = 0.02  # moving with the cart: velocity within this of the cart's
LET_GO_M = 0.05  # both grippers' grasp points at least this far from the rib's grip point
LIFT_M = 0.04  # the full tray's underside this far above the cart's top counts as taken off it

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(load=None, lifted=False, tcps=None)


reset_state()


def _rot(tray) -> np.ndarray:
    x, y, z, w = (float(v) for v in tray.quat_xyzw)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _up(tray) -> np.ndarray:
    """The tray's up axis (world frame)."""
    return _rot(tray)[:, 2]


def _under(tray) -> float:
    """The height of the tray's underside, under its centre."""
    return float((tray.pos + _up(tray) * g.UNDER)[2])


def level(tray, deg: float) -> bool:
    return bool(_up(tray)[2] >= math.cos(math.radians(deg)))


def let_go(tray, tcps) -> bool:
    """Neither gripper holds the tray: both grasp points 5 cm or more from its rib's grip point."""
    if tcps is None:
        return False
    grip = tray.pos + _up(tray) * g.GRIP_Z
    return all(np.linalg.norm(np.asarray(p, float) - grip) >= LET_GO_M for p in tcps)


def holds_parts(tray, parts) -> bool:
    """Every part's centre is inside the tray: between its walls and between its floor and its rim."""
    rot = _rot(tray)
    for part in parts:
        local = rot.T @ (np.asarray(part.pos, float) - tray.pos)
        if not (abs(local[0]) < g.INNER / 2 and abs(local[1]) < g.INNER / 2 and g.FLOOR_TOP < local[2] < g.RIM):
            return False
    return True


def seated(tray, cart) -> bool:
    """``tray`` sits in the cart's nest: centre within 1.5 cm of the cart's centre, underside within 5 mm of its top,
    level within 5 degrees, moving with it."""
    top = cart.pos[2] + g.CART[2] / 2
    return bool(level(tray, TILT_DEG) and np.hypot(*(tray.pos[:2] - cart.pos[:2])) <= CENTRE_M
                and abs(_under(tray) - top) <= SEAT_M and np.linalg.norm(tray.lin_vel - cart.lin_vel) <= RIDE_MPS)


def on_cart(tray, cart) -> bool:
    """``tray`` rides on the cart, if not seated: centre over the cart's top, underside within 3 cm above it, moving
    with it."""
    top = cart.pos[2] + g.CART[2] / 2
    d = np.abs(tray.pos[:2] - cart.pos[:2])
    return bool(np.all(d <= g.CART[0] / 2) and -SEAT_M <= _under(tray) - top <= ON_CART_M
                and np.linalg.norm(tray.lin_vel - cart.lin_vel) <= RIDE_MPS)


def load_credit(objects, tcps) -> float:
    """The empty tray in the nest: 0.5 seated and let go; 0.25 on the cart and let go but not seated; else 0."""
    tray, cart = objects["tray_empty"], objects["cart"]
    if not let_go(tray, tcps):
        return 0.0
    if seated(tray, cart):
        return 0.5
    return 0.25 if on_cart(tray, cart) else 0.0


def on_outfeed(tray) -> bool:
    """Centre over the outfeed table's top, underside within 1 cm of it, level within 10 degrees, still."""
    d = np.abs(tray.pos[:2] - np.asarray(OUT_XY))
    return bool(np.all(d <= PAD[0] / 2) and abs(_under(tray) - PAD_TOP) <= OUT_SEAT_M and level(tray, UPRIGHT_DEG)
                and tray.resting())


def unload_credit(objects, tcps, lifted: bool) -> float:
    """The full tray: 0.5 on the outfeed table with both parts in it and let go; 0.25 if it was taken off the cart
    and is now upright with both parts in it, off the cart, but not (or not yet) set down on the outfeed table."""
    tray, cart = objects["tray_full"], objects["cart"]
    parts = [objects[n] for n in PARTS]
    if not (level(tray, UPRIGHT_DEG) and holds_parts(tray, parts)):
        return 0.0
    if on_outfeed(tray) and let_go(tray, tcps):
        return 0.5
    off_cart = bool(np.any(np.abs(tray.pos[:2] - cart.pos[:2]) > g.CART[0] / 2))
    return 0.25 if lifted and off_cart else 0.0


def update(objects: dict, tcps) -> None:
    """One control step of the monitor: note when the full tray is taken off the cart, and judge the nest, once, at
    the step the cart's centre reaches the end line."""
    tcps = None if tcps is None else [np.asarray(p, float) for p in tcps]
    _STATE["tcps"] = tcps
    cart = objects["cart"]
    if _under(objects["tray_full"]) >= cart.pos[2] + g.CART[2] / 2 + LIFT_M:
        _STATE["lifted"] = True
    if _STATE["load"] is None and cart.pos[1] >= END_Y:
        _STATE["load"] = load_credit(objects, tcps)


NAMES = ("cart", "tray_full", "tray_empty", *PARTS)


DEBUG = bool(os.environ.get("SWAP_DEBUG"))  # print tray poses and grasps (for checking a sim run)


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.scene import object_states

    tcps = [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0] for side in ("left", "right")]
    objects = object_states(environment, list(NAMES))
    update(objects, tcps)
    if DEBUG:
        gaps = [float(np.sum(np.abs(arm_joints(environment, side, FINGER_JOINTS)))) for side in ("left", "right")]
        _debug(objects, tcps, gaps)


def _debug(objects, tcps, gaps) -> None:
    """Every 0.5 s the cart, both trays (pose, tilt, offset from the cart's centre) and the parts; every step a
    gripper is within 6 cm of a tray's grip point, its TCP relative to it and its pad gap; the nest's judgment."""
    step = _STATE["step"] = _STATE.get("step", 0) + 1
    t = step * DT
    cart = objects["cart"]
    if step % 10 == 0:
        print(f"swap_debug t={t:.2f} cart y={cart.pos[1]:.4f} lifted={_STATE['lifted']}")
        for n in ("tray_full", "tray_empty"):
            o = objects[n]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, _up(o)[2]))))
            off = (o.pos[:2] - cart.pos[:2]).round(4).tolist()
            print(f"swap_debug t={t:.2f} {n} pos={o.pos.round(4).tolist()} tilt={tilt:.1f} under={_under(o):.4f} "
                  f"vs_cart_xy={off} vel={o.lin_vel.round(3).tolist()}")
        for n in PARTS:
            print(f"swap_debug t={t:.2f} {n} pos={objects[n].pos.round(4).tolist()} "
                  f"in_full_tray={holds_parts(objects['tray_full'], [objects[n]])}")
    for side, tcp, gap in zip(("left", "right"), tcps, gaps, strict=True):
        for n in ("tray_full", "tray_empty"):
            grip = objects[n].pos + _up(objects[n]) * g.GRIP_Z
            r = np.asarray(tcp) - grip
            if np.linalg.norm(r) < 0.06:
                print(f"swap_debug t={t:.2f} {side} near {n}: tcp-grip={r.round(4).tolist()} pad_gap={gap:.4f} "
                      f"tcp-cart_y={float(tcp[1] - cart.pos[1]):.4f}")
    if _STATE["load"] is not None and not _STATE.get("printed"):
        _STATE["printed"] = True
        print(f"swap_debug t={t:.2f} nest judged at the end line: {_STATE['load']}")


# -- scoring -------------------------------------------------------------------------------------------------------


def score(objects):
    """The nest's credit, judged when the cart reached the end line (0 before then), plus the full tray's credit now:
    each half 0.5 (done), 0.25 (partly) or 0."""
    load = _STATE["load"] or 0.0
    return load + unload_credit(objects, _STATE["tcps"], _STATE["lifted"])


def success(objects):
    """Both halves done: the empty tray was seated in the nest at the end line, the full tray is on the outfeed."""
    return bool(score(objects) >= 1.0)


def distance(objects):
    """Metres from the full tray to its spot on the outfeed table plus, until the cart reaches the end line, from the
    empty tray to its seat in the nest."""
    full = objects["tray_full"]
    d = float(np.linalg.norm(full.pos - np.array([*OUT_XY, PAD_TOP - g.UNDER])))
    if _STATE["load"] is None:
        seat = objects["cart"].pos + np.array([0.0, 0.0, g.CART[2] / 2 - g.UNDER])
        d += float(np.linalg.norm(objects["tray_empty"].pos - seat))
    return d


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Each arm pinches a tray's rib from above, fingers closing along y (along the track), the pad tips 1.5 cm above the
# rim. The gripper takes about 1 s to close or open after the command (measured in Isaac Sim), so each arm holds still
# (or keeps following the cart) for 1.3 s after its close command and 1.2 s after its open command. The right arm
# waits over the track, then follows the cart (one waypoint per control step) while it goes down to the full tray,
# closes, holds and lifts it clear of the nest, then sets it on the outfeed table. The left arm takes the empty tray
# from the stand, waits over the track further on, and follows the cart while it lowers the tray into the now-empty
# nest, lets go 3 mm above the cart's top, keeps following while the fingers open, and only then rises.
DT = 1 / 20
TCP_OVER_ORIGIN = g.RIM + 0.015 + PAD_TIP_PAST_TCP  # 0.0255: TCP above a held tray's origin
TCP_OVER_UNDER = TCP_OVER_ORIGIN - g.UNDER  # 0.063: TCP above a held tray's underside
DROP_M = 0.003  # a tray is let go this far above where it sets down
CARRY_Z = NEST_TOP + 0.026 + TCP_OVER_UNDER  # a held tray's underside 2.6 cm over the funnel's top: TOP + 0.164
CART_GRASP_Z = FULL_Z - SPAWN_GAP + TCP_OVER_ORIGIN  # TOP + 0.115
CART_PLACE_Z = CART_TOP + DROP_M + TCP_OVER_UNDER
STAND_GRASP_Z = EMPTY_Z - SPAWN_GAP + TCP_OVER_ORIGIN
PAD_PRE_Z = STAND_GRASP_Z + 0.06
OUT_PLACE_Z = PAD_TOP + DROP_M + TCP_OVER_UNDER


def _on_step(y: float) -> float:
    """The cart's position nearest ``y`` at a whole control step, so a plan meets the cart exactly on a step."""
    return cart_y(round(arrival(y) / (1 / 20)) / 20)


MEET_R, MEET_L = _on_step(-0.35), _on_step(0.12)  # where each arm meets the cart over the track
WAIT_X = 0.22  # the left arm waits with the empty tray here, off the track, until the right arm has left it
TO_LINE_S, DOWN_S, LIFT_S = 1.2, 0.8, 1.0
CLOSE_S, GRIP_S = 0.4, 1.3  # the close command's ramp, then a hold before lifting
OPEN_S, LET_GO_S = 0.4, 1.2  # the open command's ramp, then a hold (following the cart) before rising
LOWER_S, SETTLE_S, RISE_S = 1.0, 0.3, 0.6
END_S = 22.0  # the left arm's plan runs to here, past the end line (16.75 s) and the 1 s success hold


def _track(t0: float, seconds: float, z0: float, z1: float, f0: float, f1: float, label: str):
    """Waypoints one control step apart over the cart's centre from ``t0``, while the TCP goes from ``z0`` to ``z1``
    and the fingers from ``f0`` to ``f1`` (both smoothly)."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        out.append(Waypoint(DT, (LINE_X, cart_y(t0 + k * DT), z0 + s * (z1 - z0)), f0 + s * (f1 - f0), label))
    return out


EASE_S = 1.0  # each arm speeds up to the cart's speed (and back down) over this long, at constant acceleration
EASE_M = SPEED * EASE_S / 2  # 3 cm: how far it travels meanwhile


def _ease(t0: float, y0: float, z: float, finger: float, into: bool, label: str):
    """Waypoints one control step apart at height ``z``, from ``y0`` at ``t0``: speeding up from rest to the cart's
    speed over EASE_S (``into``), or slowing from it to rest."""
    n = round(EASE_S / DT)
    a = SPEED / EASE_S
    out = []
    for k in range(1, n + 1):
        tau = k * DT
        y = y0 + (0.5 * a * tau * tau if into else SPEED * tau - 0.5 * a * tau * tau)
        out.append(Waypoint(DT, (LINE_X, y, z), finger, label))
    return out


def _wait_until(plan, meet: float, pos, finger, label):
    """A waypoint holding ``pos`` until ``meet`` (in whole control steps), or nothing if already there."""
    t = sum(w.duration for w in plan)
    wait = math.ceil((meet - t) / DT - 1e-6) * DT
    assert wait >= 0.0, f"{label}: {-wait:.2f} s late"
    return [Waypoint(wait, pos, finger, label)] if wait > 0 else []


def right_plan() -> list[Waypoint]:
    """Over the track ahead of the cart; from the cart's arrival (3.25 s) follow it down to the full tray, close on
    the rib, hold 1.3 s and lift it clear; then over to the outfeed table, down, let go, wait for the fingers to open
    and back up."""
    over = (LINE_X, MEET_R - EASE_M, CARRY_Z)
    plan = [Waypoint(TO_LINE_S, over, FINGER_OPEN, "over the track")]
    plan += _wait_until(plan, arrival(MEET_R) - EASE_S, over, FINGER_OPEN, "wait for the cart")
    plan += _ease(sum(w.duration for w in plan), over[1], CARRY_Z, FINGER_OPEN, True, "match the cart's speed")
    t = sum(w.duration for w in plan)
    for seconds, z0, z1, f0, f1, label in (
        (DOWN_S, CARRY_Z, CART_GRASP_Z, FINGER_OPEN, FINGER_OPEN, "down to the full tray"),
        (CLOSE_S, CART_GRASP_Z, CART_GRASP_Z, FINGER_OPEN, FINGER_CLOSED, "close on the full tray"),
        (GRIP_S, CART_GRASP_Z, CART_GRASP_Z, FINGER_CLOSED, FINGER_CLOSED, "grip the full tray"),
        (LIFT_S, CART_GRASP_Z, CARRY_Z, FINGER_CLOSED, FINGER_CLOSED, "lift the full tray"),
    ):
        plan += _track(t, seconds, z0, z1, f0, f1, label)
        t += seconds
    plan += _ease(t, cart_y(t), CARRY_Z, FINGER_CLOSED, False, "slow down with the full tray")
    ox, oy = OUT_XY
    plan += [
        Waypoint(1.5, (ox, oy, CARRY_Z), FINGER_CLOSED, "over the outfeed table"),
        Waypoint(1.2, (ox, oy, OUT_PLACE_Z), FINGER_CLOSED, "lower onto the outfeed table"),
        Waypoint(OPEN_S, (ox, oy, OUT_PLACE_Z), FINGER_OPEN, "release on the outfeed table"),
        Waypoint(LET_GO_S, (ox, oy, OUT_PLACE_Z), FINGER_OPEN, "let go of the full tray"),
        Waypoint(0.8, (ox, oy, CARRY_Z), FINGER_OPEN, "rise off the full tray"),
        Waypoint(1.0, (0.20, -0.25, CARRY_Z), FINGER_OPEN, "clear of the trays"),
    ]
    return plan


def left_plan() -> list[Waypoint]:
    """Take the empty tray off the stand (holding 1.3 s after closing) and wait with it over the track; from the
    empty nest's arrival (10.75 s) follow it, lower the tray into the nest, let go 3 mm above the cart, keep following
    for 1.2 s while the fingers open, and rise; then back over the table and wait there past the end line."""
    sx, sy = STAND_XY
    over = (LINE_X, MEET_L - EASE_M, CARRY_Z)
    plan = [
        Waypoint(2.0, (sx, sy, PAD_PRE_Z), FINGER_OPEN, "over the empty tray"),
        Waypoint(1.0, (sx, sy, STAND_GRASP_Z), FINGER_OPEN, "down to the empty tray"),
        Waypoint(CLOSE_S, (sx, sy, STAND_GRASP_Z), FINGER_CLOSED, "close on the empty tray"),
        Waypoint(GRIP_S, (sx, sy, STAND_GRASP_Z), FINGER_CLOSED, "grip the empty tray"),
        Waypoint(1.0, (sx, sy, CARRY_Z), FINGER_CLOSED, "lift the empty tray"),
        Waypoint(1.8, (WAIT_X, over[1], CARRY_Z), FINGER_CLOSED, "toward the track with the empty tray"),
    ]
    plan += _wait_until(plan, arrival(MEET_L) - EASE_S - 1.5, (WAIT_X, over[1], CARRY_Z), FINGER_CLOSED,
                        "wait for the nest")
    plan.append(Waypoint(1.5, over, FINGER_CLOSED, "over the track with the empty tray"))
    plan += _ease(sum(w.duration for w in plan), over[1], CARRY_Z, FINGER_CLOSED, True, "match the cart's speed")
    t = sum(w.duration for w in plan)
    for seconds, z0, z1, f0, f1, label in (
        (LOWER_S, CARRY_Z, CART_PLACE_Z, FINGER_CLOSED, FINGER_CLOSED, "lower into the nest"),
        (SETTLE_S, CART_PLACE_Z, CART_PLACE_Z, FINGER_CLOSED, FINGER_CLOSED, "settle in the nest"),
        (OPEN_S, CART_PLACE_Z, CART_PLACE_Z, FINGER_CLOSED, FINGER_OPEN, "release in the nest"),
        (LET_GO_S, CART_PLACE_Z, CART_PLACE_Z, FINGER_OPEN, FINGER_OPEN, "let go in the nest"),
        (RISE_S, CART_PLACE_Z, CARRY_Z + 0.02, FINGER_OPEN, FINGER_OPEN, "rise off the nest"),
    ):
        plan += _track(t, seconds, z0, z1, f0, f1, label)
        t += seconds
    plan += _ease(t, cart_y(t), CARRY_Z + 0.02, FINGER_OPEN, False, "slow down off the nest")
    t += EASE_S
    back = (0.20, 0.25, CARRY_Z)
    plan.append(Waypoint(1.0, back, FINGER_OPEN, "back over the table"))
    plan.append(Waypoint(END_S - t - 1.0, back, FINGER_OPEN, "wait"))
    return plan


def demo():
    """Right arm: take the full tray off the passing cart and set it on the outfeed table. Left arm: put the empty
    tray from the stand into the cart's nest as it passes, before the end line."""
    return {"left": left_plan(), "right": right_plan()}
