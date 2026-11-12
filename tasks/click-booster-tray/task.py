"""On the station table, facing the robot: a toddler's booster seat (moulded plastic shell, about 40 x 33 x 23 cm,
2.5 kg) standing on its non-slip feet, as it would on a dining chair. Its feeding tray (42 x 25 cm, 0.45 kg) lies flat
on the table in front of it. The tray slides onto the two armrests from the front. A sprung catch in each tray end
ratchets over a row of three notches on the armrest's outer face (25 mm pitch); the tray stays on only where both
catches have dropped into a notch. The robot must click the tray on at the second notch and let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from partkit import Slider, on_reset, on_step, part  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Put the tray on the booster seat at its middle setting, locked at the second notch."

SHELL = (0.93, 0.93, 0.90)   # white shell
RACK = (0.35, 0.62, 0.85)    # light-blue armrest inserts (notches)
TRAY = (0.95, 0.62, 0.20)    # orange tray
RIM = (0.85, 0.50, 0.12)
CATCH = (0.20, 0.25, 0.30)   # dark grey catches


# -- objects ---------------------------------------------------------------------------------------------------------
def _catch(s: int) -> Slider:
    return Slider(f"catch_{'left' if s > 0 else 'right'}", rest=(g.CATCH_X, s * g.CATCH_REST_Y, sum(g.CATCH_Z) / 2),
                  boxes=g.catch_boxes(s, CATCH), cylinders=(g.catch_nose(s, CATCH),), travel=g.CATCH_TRAVEL, rate=g.CATCH_RATE, damping=5.0,
                  mass=g.CATCH_MASS, axis="Y", sign=s, friction=0.2)


BOOSTER = part("booster", pos=(*g.BOOSTER_XY, g.TABLE + 0.001), mass=g.BOOSTER_MASS,
               boxes=g.booster_boxes(SHELL, RACK), color=SHELL, friction=0.9)
TRAY_PART = part("tray", pos=(g.BOOSTER_XY[0] + g.START_OFFSET, g.BOOSTER_XY[1], g.TABLE + g.RAIL_DEPTH + 0.001),
                 mass=g.TRAY_MASS, boxes=g.tray_boxes(TRAY, RIM), sliders=(_catch(1), _catch(-1)), color=TRAY,
                 friction=0.4, friction_min=True)
OBJECTS = [BOOSTER, TRAY_PART]

# -- monitor and scoring ---------------------------------------------------------------------------------------------
_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(violation=False, why=[], booster0=None, notches=(None, None), seated=False, released=False,
                  still=False, tcp_prev=None)


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(booster, tray, fingers, tcps) -> None:
    """One control step's bookkeeping. ``booster``/``tray`` have ``pos`` (body origin), ``quat_xyzw``, ``lin_vel``;
    ``tray.sliders`` maps "catch_left"/"catch_right" to their positions (m, 0 at rest). ``fingers`` lists world
    points on both grippers (TCPs, pad tips, finger links); ``tcps`` the two TCPs."""
    bpos, tpos = np.asarray(booster.pos, float), np.asarray(tray.pos, float)
    if _STATE["booster0"] is None:
        _STATE["booster0"] = bpos.copy()
    if (np.linalg.norm(bpos[:2] - _STATE["booster0"][:2]) > g.BOOSTER_MOVE
            or g.tilt_deg(g.quat_matrix(booster.quat_xyzw)) > g.BOOSTER_TIP):
        _flag("the booster seat was knocked out of place")
    clearance = g.tray_clearance(fingers, tpos, tray.quat_xyzw)
    if float(tray.lin_vel[2]) < -g.DROP_SPEED and clearance >= g.RELEASE:   # falling, not being lowered
        _flag("the tray was dropped")
    if tpos[2] < g.TABLE - 0.05 or bpos[2] < g.TABLE - 0.05:
        _flag("something fell off the table")
    rel, rrot = g.tray_in_booster(bpos, booster.quat_xyzw, tpos, tray.quat_xyzw)
    q = tray.sliders
    notches = (g.catch_notch(rel, rrot, 1, q.get("catch_left", 1.0)),
               g.catch_notch(rel, rrot, -1, q.get("catch_right", 1.0)))
    tcps = [np.asarray(t, float) for t in tcps]
    prev = _STATE["tcp_prev"]
    arms_still = prev is not None and all(np.linalg.norm(a - b) < g.STILL_TCP for a, b in zip(tcps, prev))
    _STATE.update(notches=notches, seated=g.seated(rel, rrot),
                  released=clearance >= g.RELEASE,
                  still=bool(arms_still and float(np.linalg.norm(tray.lin_vel)) < g.STILL_TRAY),
                  tcp_prev=tcps, rel=rel)


@on_step
def _monitor(environment) -> None:
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP
    from yam_arena.scene import object_states

    from partkit import arm_joints, robot_links

    links = robot_links(environment)
    fingers, tcps = [], []
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        tcps.append(tcp)
        fingers += [tcp, tcp + rot[:, 2] * PAD_TIP_PAST_TCP]
        fingers += [p for name, p in links[side].items() if name.startswith("tip")]
    states = object_states(environment, ["booster", "tray"])
    update(states["booster"], states["tray"], fingers, tcps)


def locked_at() -> int | None:
    """The notch both catches are down in, or None."""
    a, b = _STATE["notches"]
    return a if a is not None and a == b else None


def success(objects) -> bool:
    """Both catches down in notch 2, every finger point 2 cm or more from the tray, the tray and both arms still, no
    violation so far. Held for 1 s."""
    return bool(not _STATE["violation"] and locked_at() == 2 and _STATE["released"] and _STATE["still"])


def score(objects) -> float:
    """0 after a violation (the booster moved over 3 cm or tipped over 10 degrees, the tray dropped, anything off the
    table). Otherwise at the end: 1.0 locked at notch 2, released and still (success); 0.75 locked at notch 2 but
    still held or moving; 0.5 locked with both catches in notch 1 or both in notch 3; 0.25 resting on both armrests,
    let go, not locked; else 0."""
    if _STATE["violation"]:
        return 0.0
    n = locked_at()
    if n == 2:
        return 1.0 if (_STATE["released"] and _STATE["still"]) else 0.75
    if n is not None:
        return 0.5
    return 0.25 if (_STATE["seated"] and _STATE["released"]) else 0.0


# -- scripted demo (open loop on the fixed layout; played by yam_scripted) --------------------------------------------
def demo_waypoints() -> dict:
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
    from yam_arena.demo import Waypoint

    table_z = g.RAIL_DEPTH                      # tray origin's booster z lying on the table
    seat_z = g.SEATED_Z
    o0 = g.START_OFFSET
    # The tray pivots about 7 degrees in the grips (back edge down), so it is carried high, its back edge over the
    # armrests' front ends, and set down back edge first; lowering the rim then brings it level.
    o_entry = -0.04                             # back edge 4 cm onto the armrests' flat tops, past their lead-in
    #                                             chamfer; catches still 1 cm short of the lip
    carry = 0.042                               # rim this far above its seated height while carrying
    # 6 mm past notch 2's seat: the catches' backs 6 mm past tooth 2's crest (the arms lag while they click over) and
    # 3.5 mm short of tooth 3's apex, so the springs push it back into notch 2 once the arms ease off
    o_push = g.offset_for_catch(g.notch_seat(2) + 0.006)
    o_back = g.offset_for_catch(g.notch_seat(2) - 0.001)    # eased back onto tooth 2's square face
    plans = {}
    for side, s in (("left", 1), ("right", -1)):
        def at(o, z, dz=0.0):
            x, y, zz = g.grip_tcp(o, z, s)
            return (x, y, zz + dz)

        yaw = g.GRIP_YAW
        plans[side] = [
            Waypoint(2.5, at(o0, table_z, 0.06), FINGER_OPEN, "above the front rim", yaw),
            Waypoint(1.5, at(o0, table_z), FINGER_OPEN, "fingers astride the rim", yaw),
            Waypoint(1.0, at(o0, table_z), FINGER_CLOSED, "grip", yaw),
            Waypoint(0.8, at(o0, table_z), FINGER_CLOSED, "hold", yaw),
            Waypoint(2.5, at(o0, seat_z, carry), FINGER_CLOSED, "lift above the armrests", yaw),
            Waypoint(2.0, at(o_entry, seat_z, carry), FINGER_CLOSED, "over the armrests' front ends", yaw),
            Waypoint(2.0, at(o_entry, seat_z), FINGER_CLOSED, "set down, back edge first", yaw),
            Waypoint(6.5, at(o_push, seat_z), FINGER_CLOSED, "slide on past notch 1 into notch 2", yaw),
            Waypoint(0.5, at(o_push, seat_z), FINGER_CLOSED, "settle", yaw),
            Waypoint(1.0, at(o_back, seat_z), FINGER_CLOSED, "ease back onto the catches", yaw),
            Waypoint(1.0, at(o_back, seat_z), FINGER_OPEN, "let go", yaw),
            Waypoint(0.5, at(o_back, seat_z), FINGER_OPEN, "wait", yaw),
            Waypoint(1.0, at(o_back - 0.008, seat_z, 0.012), FINGER_OPEN, "lift off the rim", yaw),
            Waypoint(1.5, at(o_back - 0.07, seat_z, 0.03), FINGER_OPEN, "back away", yaw),
        ]
    return plans


def demo():
    return demo_waypoints()
