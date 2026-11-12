"""On the station table, straight ahead of the arms: a compact desktop laser printer (40 cm wide, 36 cm deep, 23 cm
tall), its front 27 cm in front of the arm bases. Its front door, an L-shaped cover hinged along the bottom of the
toner bay's opening, stands fully open (80 degrees), leaning toward the robot. In the bay sits the used toner
cartridge (black, grey handle): a 34 x 10 x 9 cm cartridge whose two side bosses are down in the seat notches at the
end of the bay's guide rails. The new cartridge (black, orange handle) lies on the table to the right of the printer;
the table to its left is free.

The robot must take the old cartridge out along the rails and put it down on the table, slide the new one in along the
rails until its bosses drop into the seat notches, and shut the door, which an unseated cartridge blocks.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Replace the toner cartridge in the printer with the new one from the table, put the old one down on "
               "the table, and shut the printer's front door. Don't touch the black roller inside the printer.")

# -- objects -------------------------------------------------------------------------------------------------------
# The printer starts exactly at rest on the table (no drop). Its materials come from its USD file: rubber feet, ABS
# everywhere else.
PRINTER = UsdBody("printer", usd="assets/printer.usda", pos=(float(g.X_FRONT), 0.0, float(g.TABLE)),
                  size=(g.P_DEPTH, 2 * g.P_HALF_W, g.P_HEIGHT), articulation=True,
                  joint_pos=(("hinge", float(g.OPEN)),), joint_damping=g.DOOR_DAMPING, friction=None,
                  self_collisions=False, fixed_base=False)
OLD = UsdBody("old_cartridge", usd="assets/cartridge_old.usda", pos=tuple(float(v) for v in g.OLD_START),
              size=g.C_SIZE, mass=g.MASS["old"], friction=(1.0, 0.9), torsional_patch=0.01)
NEW = UsdBody("new_cartridge", usd="assets/cartridge_new.usda",
              pos=tuple(float(v) for v in g.NEW_START + [0.0, 0.0, 0.0005]), size=g.C_SIZE, mass=g.MASS["new"],
              friction=(1.0, 0.9), torsional_patch=0.01)
OBJECTS = [PRINTER, OLD, NEW]
CARTS = {"old": "old_cartridge", "new": "new_cartridge"}

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
PRINTER_SHIFT = 0.01            # m: the printer moved this far on the table ...
PRINTER_TILT = math.radians(2)  # ... or tilted or turned this much
ROLLER_TOUCH = 0.003            # m: any arm part this close to the transfer roller touches it
FALL_V = 0.5                    # m/s: a cartridge no gripper holds falling this fast was dropped (a 1.3 cm free fall)
TIP = math.radians(30)          # a cartridge tilted this far (toner spills, or it fell)
OFF_TABLE = 0.02                # m: a cartridge's centre this far below the table top has left it
HOLD_R = 0.03                   # m: a gripper holds a cartridge when its grasp point is this close to the handle's
HOLD_OPEN = 0.03                # centre and its fingers are less than this far apart (on the 16 mm handle: 16 mm)
REST = 0.008                    # m: on the table: the cartridge's lowest corner this close to the table top ...
REST_TILT = math.radians(5)     # ... and level within this
SEAT_POS = 0.004                # m: seated: the new cartridge's body within this of its seated position ...
SEAT_TILT = math.radians(2)     # ... level and square to the printer within this
SHUT = math.radians(2)          # door shut within this (its front panel's top within 6 mm of the case)
CLEAR = 0.03                    # m: at the end, every part of both arms this far from the door and both cartridges
STILL_V = 0.01                  # m/s: cartridges still; the door moves less than STILL_DOOR per control step
STILL_DOOR = math.radians(0.2)
ARM_R, GRIP_R = 0.035, 0.045    # m: arm links and the gripper (housing and fingers) are checked as capsules
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")
PAD_TIP = 0.013

_STATE: dict = {}


def tilt_of(rot: np.ndarray) -> float:
    """Angle between a body's z axis and the vertical."""
    return math.acos(max(-1.0, min(1.0, float(rot[2, 2]))))


def yaw_of(rot: np.ndarray) -> float:
    return math.atan2(rot[1, 0], rot[0, 0])


def _rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _seg_dist(a0, a1, b0, b1, n: int = 16) -> float:
    """Approximate distance between two segments."""
    sa = np.asarray(a0) + np.outer(np.linspace(0, 1, n), np.asarray(a1) - np.asarray(a0))
    sb = np.asarray(b0) + np.outer(np.linspace(0, 1, n), np.asarray(b1) - np.asarray(b0))
    return float(np.linalg.norm(sa[:, None] - sb[None], axis=2).min())


def capsules_gap(caps, boxes) -> float:
    """Smallest distance from arm capsules (end, end, radius) to boxes (centre, rotation, half sizes)."""
    best = math.inf
    for a, b, r in caps:
        seg = np.asarray(a) + np.outer(np.linspace(0, 1, 12), np.asarray(b) - np.asarray(a))
        for c, rot, h in boxes:
            best = min(best, float(g.point_box_dist(seg, c, rot, h).min()) - r)
    return best


def arm_capsules(chain, tcp, approach) -> list:
    pts = [np.asarray(p, float) for p in chain]
    caps = [(a, b, ARM_R) for a, b in zip(pts[:-1], pts[1:])]
    caps.append((pts[-1], np.asarray(tcp, float) + PAD_TIP * np.asarray(approach, float), GRIP_R))
    return caps


def _arms(environment):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    from scenekit import arm_joints, robot_links

    links = robot_links(environment)
    out = {}
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        opening = float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())
        out[side] = {"caps": arm_capsules([links[side][n] for n in CHAIN], tcp, rot[:, 2]), "grasp": tcp,
                     "opening": opening}
    return out


def cart_boxes(pos, rot):
    return [(c, r, h) for _, c, r, h in g.cartridge_world(pos, rot)]


def handle_centre(pos, rot) -> np.ndarray:
    return np.asarray(pos) + rot @ g.FIN_CENTRE


def lowest_z(pos, rot) -> float:
    c, r, h = cart_boxes(pos, rot)[0]
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h
    return float((c + corners @ r.T)[:, 2].min())


def out_of_printer(pos, rot) -> bool:
    """The cartridge's whole footprint (bosses included) is outside the printer's."""
    pts = np.array([c + r @ (np.array([sx, sy, 0.0]) * h) for c, r, h in cart_boxes(pos, rot)
                    for sx in (-1, 1) for sy in (-1, 1)])
    inside = (pts[:, 0] > g.X_FRONT) & (pts[:, 0] < g.X_FRONT + g.P_DEPTH) & (np.abs(pts[:, 1]) < g.P_HALF_W)
    return not bool(inside.any())


SEAT_WORLD = g.printer_to_world(g.SEAT_C)


def seated(pos, rot) -> bool:
    return bool(np.linalg.norm(np.asarray(pos) - SEAT_WORLD) <= SEAT_POS and tilt_of(rot) <= SEAT_TILT
                and abs(yaw_of(rot)) <= SEAT_TILT)


def on_table(pos, rot) -> bool:
    return bool(lowest_z(pos, rot) <= g.TABLE + REST and tilt_of(rot) <= REST_TILT)


def level(old_out: bool, new_seated: bool, shut: bool) -> float:
    """Partial credit at the end: 0.2 old cartridge out and set down; 0.5 also the new one seated; 0.75 also the
    door shut."""
    if not old_out:
        return 0.0
    if not new_seated:
        return 0.2
    return 0.75 if shut else 0.5


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, held={"old": False, "new": False},
                  level=0.0, clear=False, still=False, door=float(g.OPEN))


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(printer_pos, printer_rot, door: float, carts: dict, arms: dict) -> None:
    """One control step of the monitor. ``carts``: name -> (pos, rot, velocity); ``arms``: side -> caps/grasp/opening.
    Seating, the printer's footprint, the door and the roller are judged in the printer's measured frame, so a printer
    nudged within its limit costs nothing; the table and falls in the table's frame."""
    S = _STATE
    pp, rp = np.asarray(printer_pos, float), np.asarray(printer_rot, float)
    nominal = g.printer_to_world((0.0, 0.0, 0.0))

    def to_printer(pos, rot):        # a pose as if the printer stood exactly at its marked place
        return nominal + rp.T @ (np.asarray(pos, float) - pp), rp.T @ rot

    def from_printer(boxes):         # boxes placed for the printer at its marked place, moved with it
        return [(pp + rp @ (c - nominal), rp @ r, h) for c, r, h in boxes]

    # against its marked place on the table (on a real table: a taped outline)
    if (np.linalg.norm(pp[:2] - nominal[:2]) > PRINTER_SHIFT or tilt_of(rp) > PRINTER_TILT
            or abs(yaw_of(rp)) > PRINTER_TILT):
        _flag("the printer was moved or tipped")
    half = np.array([0.0, g.ROLLER[3], 0.0])
    roller = [pp + rp @ (g.printer_to_world(g.ROLLER_C + k * half) - nominal) for k in (-1, 1)]
    for side, arm in arms.items():
        if any(_seg_dist(a, b, *roller) - r - g.ROLLER[2] < ROLLER_TOUCH for a, b, r in arm["caps"]):
            _flag(f"the {side} arm touched the transfer roller")

    status = {}
    for name, (pos, rot, vel) in carts.items():
        vel = np.asarray(vel, float)
        held = any(np.linalg.norm(a["grasp"] - handle_centre(pos, rot)) < HOLD_R and a["opening"] < HOLD_OPEN
                   for a in arms.values())
        S["held"][name] = held
        if not held and vel[2] < -FALL_V:
            _flag(f"the {name} cartridge was dropped")
        if tilt_of(rot) > TIP:
            _flag(f"the {name} cartridge was tipped over {math.degrees(TIP):.0f} degrees")
        if pos[2] < g.TABLE - OFF_TABLE:
            _flag(f"the {name} cartridge left the table")
        status[name] = (np.asarray(pos, float), rot, float(np.linalg.norm(vel)), held)

    pos, rot, _, held = status["old"]
    old_out = out_of_printer(*to_printer(pos, rot)) and on_table(pos, rot) and not held
    pos, rot, _, held = status["new"]
    new_seated = seated(*to_printer(pos, rot)) and not held
    shut = door <= SHUT
    door_boxes = from_printer([(c, r, h) for _, c, r, h in g.door_world(door)])
    boxes = door_boxes + [b for p, r, _, _ in status.values() for b in cart_boxes(p, r)]
    clear = all(capsules_gap(a["caps"], boxes) >= CLEAR for a in arms.values())
    still = (all(s < STILL_V for _, _, s, _ in status.values()) and abs(door - S["door"]) < STILL_DOOR)
    new_p, new_r = to_printer(*status["new"][:2])
    S["printer_off_mm"] = np.round((pp - nominal) * 1000, 1).tolist()
    S.update(level=level(old_out, new_seated, shut), clear=clear, still=still, door=float(door),
             seated_err=(np.round((new_p - SEAT_WORLD) * 1000, 1).tolist(), round(math.degrees(tilt_of(new_r)), 2),
                         round(math.degrees(yaw_of(new_r)), 2), np.round(status["old"][0], 3).tolist()),
             old_info=(bool(out_of_printer(*to_printer(*status["old"][:2]))), bool(on_table(*status["old"][:2])),
                       bool(status["old"][3])))


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["printer", *CARTS.values()])
    pr = st["printer"]
    carts = {k: (st[v].pos, _rot(st[v].quat_xyzw), np.asarray(st[v].lin_vel, float)) for k, v in CARTS.items()}
    update(pr.pos, _rot(pr.quat_xyzw), float(pr.joints.get("hinge", g.OPEN)), carts, _arms(environment))


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """No violation so far; the old cartridge out of the printer and resting upright on the table, let go; the new
    one seated, let go; the door shut; every part of both arms 3 cm clear of the door and both cartridges; the
    cartridges and the door still. Held 1 s."""
    S = _STATE
    return bool(not violation() and S["level"] >= 0.75 and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: the printer moved or tipped, the transfer roller touched, a cartridge dropped (falling
    faster than 0.5 m/s with no gripper holding it), tipped past 30 degrees or off the table. Otherwise, at the end:
    0.2 if the old cartridge is out of the printer, resting on the table and let go; 0.5 if the new one is also seated
    and let go; 0.75 if the door is also shut. Success scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
