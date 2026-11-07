"""Offline checks (no Isaac Sim): the layout, the knife asset, every scoring level, and that the demo is reachable and,
played kinematically with the knife riding in the gripper, ends with the knife stowed without hitting anything.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py      # from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import TABLE_TOP, Box, load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import geometry as g  # noqa: E402  (task.py put this folder on sys.path)

DT = 1 / 20
FINGER_T = 0.010        # a finger's thickness behind its pad (YAM tip mesh, lower 4 cm)
FINGER_HALF_W = 0.0192  # a finger's half-width across the closing direction, lower 2 cm above the tip


def pose(origin, yaw=0.0, tilt=0.0):
    """Knife origin and rotation: turned ``yaw`` about the vertical, its tip raised by ``tilt``."""
    c, s = math.cos(-tilt), math.sin(-tilt)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.asarray(origin, float), g.rz(yaw) @ ry


def resting_in_slot(yaw=0.0, dy=0.0):
    origin = g.origin_for_grip((g.GOAL_GRIP_XY[0], g.GOAL_GRIP_XY[1] + dy), yaw, g.FLOOR)
    return pose(origin, yaw)


def run(*steps):
    M.MONITOR.reset()
    out = None
    for st in steps:
        out = M.MONITOR.update(*st)
    return out


AWAY = [(np.array([0.2, 0.3, 0.95]), 0.0), (np.array([0.2, -0.3, 0.95]), 0.0)]  # both grippers parked, closed
START = pose(g.START, g.START_YAW)
LIFTED = pose(g.START + [0, 0, 0.05], g.START_YAW)


# -- layout --------------------------------------------------------------------------------------------------------
def test_it_loads_with_only_the_knife_free():
    assert TASK.free_objects == ("knife",) and TASK.score_over == "end" and TASK.success_hold_s == 2.0
    assert g.TOP == TABLE_TOP["station"]


def test_poses_are_plain_floats():
    """Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects at reset."""
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", (0.0,))), spec.name


def test_the_knife_asset_matches_the_geometry():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(HERE / M.KNIFE.usd))
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(g.KNIFE_MASS)
    for name, (centre, size) in g.KNIFE_PARTS.items():
        prim = stage.GetPrimAtPath(f"/Knife/{name}")
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        assert np.allclose(m[:3, 3], centre) and np.allclose(np.linalg.norm(m[:3, :3], axis=0), size, atol=1e-6)
    assert g.BLADE_T >= 0.004  # thick enough for stable contact


def test_the_knife_starts_flat_on_the_board_pointing_the_wrong_way():
    c = g.knife_corners(*START)
    assert np.all(c[:, 0] > g.BOARD_LO[0] + 0.01) and np.all(c[:, 0] < g.BOARD_HI[0] - 0.01)
    assert np.all(c[:, 1] > g.BOARD_LO[1] + 0.01) and np.all(c[:, 1] < g.BOARD_HI[1] - 0.01)
    assert c[:, 2].min() == pytest.approx(g.BOARD_TOP + 0.001)
    tip = g.knife_axis(*START)[-1]
    assert tip[0] < g.START[0] - 0.1  # the blade points at the robot
    assert run((*START, (0, 0, 0), AWAY)) == 0.0


def test_the_drawer_is_one_closed_tray_and_the_board_clears_it():
    assert g.SLOTS[0][0] == g.RIGHT_IN and np.allclose(np.diff([s[0] for s in g.SLOTS]), g.SLOT_W + g.DIVIDER_W)
    assert g.BOARD_HI[1] < g.RIGHT_IN - g.WALL_T  # a gap between the board and the drawer
    assert g.BACK_IN - g.FRONT_IN - g.KNIFE_LEN > 0.06  # the knife fits along the slot with 3 cm to spare each end
    # the stored knives sit inside their slots, handle toward the front
    for spec in M.STORED:
        lo, hi = np.asarray(spec.pos) - np.asarray(spec.size) / 2, np.asarray(spec.pos) + np.asarray(spec.size) / 2
        assert g.FRONT_IN < lo[0] and hi[0] < g.BACK_IN and g.SLOTS[1][0] < lo[1] and hi[1] < g.SLOTS[2][1]


# -- scoring ---------------------------------------------------------------------------------------------------------
def test_stowed_handle_front_and_let_go_is_full_score():
    assert run((*resting_in_slot(), (0, 0, 0), AWAY)) == 1.0
    assert M.MONITOR.success()
    assert run((*resting_in_slot(math.radians(15), dy=-0.01), (0, 0, 0), AWAY)) == 1.0  # a bit skew still counts


def test_still_held_or_moving_is_not_success():
    p = resting_in_slot()
    tcp = p[0] + p[1] @ g.GRIP
    assert run((*LIFTED, (0, 0, 0), AWAY), (*p, (0, 0, 0), [(tcp, 0.024), AWAY[1]])) == 0.25
    assert not M.MONITOR.success()
    assert run((*p, (0, 0, 0), [(tcp, 0.036), AWAY[1]])) == 1.0  # fingers opened to 3.6 cm: let go
    assert run((*p, (0.05, 0, 0), AWAY)) == 1.0 and not M.MONITOR.success()  # still sliding


def test_blade_toward_the_front_is_half():
    """The naive pick and place, without turning the knife: in the slot, flat, blade toward the drawer front."""
    p = pose(g.origin_for_grip(g.GOAL_GRIP_XY, math.pi, g.FLOOR), math.pi)
    x = g.knife_corners(*p)[:, 0]
    p = (p[0] - [(x.min() + x.max()) / 2 - (g.FRONT_IN + g.BACK_IN) / 2, 0, 0], p[1])  # centred along the slot
    k = M.classify(*p)
    assert k["in_slot"] and k["flat"] and not k["handle_front"]
    assert run((*LIFTED, (0, 0, 0), AWAY), (*p, (0, 0, 0), AWAY)) == 0.5


def test_the_angle_tolerance_is_20_degrees():
    assert M.classify(*resting_in_slot(math.radians(19)))["handle_front"]
    assert not M.classify(*resting_in_slot(math.radians(21)))["handle_front"]
    assert not M.classify(*resting_in_slot(math.pi / 2))["stowed"]


def test_out_of_slot_or_not_flat_but_in_the_tray_is_half():
    over_divider = resting_in_slot(dy=0.035)  # straddling the divider into the next slot
    assert M.classify(*over_divider)["in_tray"] and not M.classify(*over_divider)["in_slot"]
    assert run((*over_divider, (0, 0, 0), AWAY)) == 0.5
    on_top = pose(g.origin_for_grip((g.GOAL_GRIP_XY[0], (g.SLOTS[1][0] + g.SLOTS[1][1]) / 2), 0.0, g.FLOOR + 0.022))
    assert run((*on_top, (0, 0, 0), AWAY)) == 0.5  # on the paring knife in the next slot
    p = resting_in_slot()
    tilted = pose(p[0] + [0, 0, 0.012], 0.0, math.radians(14))  # propped up on something: not flat
    assert M.classify(*tilted)["in_tray"] and not M.classify(*tilted)["stowed"]
    assert run((*tilted, (0, 0, 0), AWAY)) == 0.5


def test_lifted_but_not_in_the_tray_is_a_quarter():
    assert run((*LIFTED, (0, 0, 0), AWAY)) == 0.25
    on_table = pose((0.25, -0.25, g.TOP), 0.0)
    assert run((*LIFTED, (0, 0, 0), AWAY), (*on_table, (0, 0, 0), AWAY)) == 0.25
    just_under = pose(g.START + [0, 0, M.LIFT_M - 0.002], g.START_YAW)
    assert run((*just_under, (0, 0, 0), AWAY)) == 0.0
    pushed_off_board = pose((0.25, -0.25, g.TOP), 0.0)  # slid onto the table without a lift
    assert run((*pushed_off_board, (0, 0, 0), AWAY)) == 0.0


def test_off_the_table_is_zero_for_good():
    floor = pose((0.5, -0.8, 0.02), 0.3)
    assert run((*LIFTED, (0, 0, 0), AWAY), (*floor, (0, 0, -1), AWAY)) == 0.0
    M.MONITOR.update(*resting_in_slot(), (0, 0, 0), AWAY)  # even if it were then found in the slot
    assert M.MONITOR.score == 0.0 and not M.MONITOR.success()


def test_the_monitor_resets_with_the_episode():
    import scenekit

    run((*pose((0.5, -0.8, 0.02)), (0, 0, 0), AWAY))
    for fn in scenekit._RESET:
        fn(None)
    assert not M.MONITOR.fallen and not M.MONITOR.lifted and M.MONITOR.score == 0.0


# -- the demo ------------------------------------------------------------------------------------------------------
def _play():
    """Play both arms' plans at 20 Hz. Returns per step: each arm's TCP, rotation, finger target and label."""
    plans = M.demo()
    arms = {s: ArmPlan(plans[s], station.BASES[s]) for s in ("left", "right")}
    n = round(max(a.duration for a in arms.values()) / DT) + 5
    steps, actions = [], {s: [] for s in arms}
    for k in range(n):
        t = (k + 1) * DT
        row = {}
        for s, a in arms.items():
            act, _ = a.action7(t)
            actions[s].append(act)
            row[s] = a.target(t)
        steps.append(row)
    return arms, steps, {s: np.array(v) for s, v in actions.items()}


ARMS, STEPS, ACTIONS = _play()


def test_the_demo_is_reachable_within_the_run_limit():
    for s, arm in ARMS.items():
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, s
        assert np.all(ACTIONS[s][:, :6] >= ik.LOWER) and np.all(ACTIONS[s][:, :6] <= ik.UPPER), s
        # every joint keeps 10 degrees from its stops (the wrist turn, joint 6, over 60), so nothing rides a limit
        q = ACTIONS[s][:, :6]
        assert np.all(q.min(0) - ik.LOWER > math.radians(10)) and np.all(ik.UPPER - q.max(0) > math.radians(10)), s
        assert np.abs(q[:, 5]).max() < ik.UPPER[5] - math.radians(60), s
    assert len(STEPS) + M.SUCCESS_HOLD_S / DT < 0.6 * 1200


def test_the_arms_never_work_close_together():
    d = min(np.linalg.norm(r["left"][0] - r["right"][0]) for r in STEPS)
    assert d > 0.20


def _fixed_boxes():
    for spec in M.OBJECTS:
        if isinstance(spec, Box) and spec.mass is None:
            yield spec.name, np.asarray(spec.pos) - np.asarray(spec.size) / 2, np.asarray(spec.pos) + np.asarray(
                spec.size) / 2


def _knife_samples(pos, rot, n=6):
    out = []
    for centre, size in g.KNIFE_PARTS.values():
        t = np.linspace(-0.5, 0.5, n)
        grid = np.array(np.meshgrid(t, t, t)).reshape(3, -1).T * np.asarray(size) + np.asarray(centre)
        out.append(grid @ rot.T + pos)
    return np.vstack(out)


def test_played_kinematically_the_demo_stows_the_knife_without_hitting_anything():
    """The knife rides rigidly in whichever gripper has closed on its grip point, drops RELEASE_GAP when let go,
    and must never enter the board or tray floor it lies on, nor come within 2 mm of another fixed box (drawer walls,
    dividers, stored knives) on the way."""
    pos, rot = START[0].copy(), START[1].copy()
    holder, in_tcp, turned = None, None, []
    boxes = list(_fixed_boxes())
    for row in STEPS:
        if holder is None:
            for s, (tcp, trot, finger, label) in row.items():
                grip = pos + rot @ g.GRIP
                if finger > -0.001 and np.linalg.norm(tcp - grip) < 1e-3:
                    holder, in_tcp = s, (trot.T @ (pos - tcp), trot.T @ rot)
                    turned.append(s)
        else:
            tcp, trot, finger, label = row[holder]
            if finger < -0.005:  # opening: let go; it settles RELEASE_GAP lower
                holder, pos = None, pos - [0, 0, M.RELEASE_GAP]
            else:
                pos, rot = tcp + trot @ in_tcp[0], trot @ in_tcp[1]
                pts = _knife_samples(pos, rot)
                for name, lo, hi in boxes:
                    m = 0.0 if name in ("cutting_board", "drawer_floor") else 0.002  # it lies on these
                    inside = np.all((pts > lo - m + 1e-9) & (pts < hi + m - 1e-9), axis=1)
                    assert not inside.any(), (name, label)
    assert turned == ["right", "left"]
    k = M.classify(pos, rot)
    assert k["stowed"], k
    assert pos[2] == pytest.approx(g.FLOOR) and abs(math.atan2(rot[1, 0], rot[0, 0])) < 1e-3
    # and after the left arm parks, it is let go
    final = [(row[s][0], abs(row[s][2]) * 2) for s in ("left", "right") for row in STEPS[-1:]]
    assert run((pos, rot, (0, 0, 0), final)) == 1.0


def test_the_fingers_clear_the_board_the_blade_and_the_slot():
    # pad tips stop 4 mm above whatever the knife lies on
    assert g.GRIP[2] - PAD_TIP_PAST_TCP >= 0.003
    # along the knife, the fingers' lower 2 cm stay 1.5 cm behind the bolster, on the handle
    assert -g.GRIP[0] - FINGER_HALF_W >= 0.015 and g.HANDLE_LEN + g.GRIP[0] - FINGER_HALF_W >= 0.015
    # in the slot, closed on the handle and opened to let go, each finger clears the dividers by 1.5 cm
    half = g.SLOT_W / 2
    assert half - (g.HANDLE_W / 2 + FINGER_T) >= 0.015
    assert half - (-M.RELEASE + FINGER_T) >= 0.015
    # opened to let go, the pads clear the handle and the fingers are open wide enough not to count as holding
    assert -M.RELEASE - g.HANDLE_W / 2 >= 0.005 and -2 * M.RELEASE > M.HOLD_OPEN
    # fully open over the board, each pad clears the handle by over 3 cm
    assert -FINGER_OPEN - g.HANDLE_W / 2 > 0.03
    assert g.GOAL_GRIP_XY[1] == pytest.approx(sum(g.EMPTY) / 2)


def test_carried_the_knife_clears_the_drawer_walls():
    assert M.TRAVEL_Z - g.GRIP[2] - g.WALL_TOP >= 0.03
