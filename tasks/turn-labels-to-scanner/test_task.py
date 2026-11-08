"""Offline checks (no Isaac Sim): the carton asset, the line's motion, the gate's clearances, the scanner's read rule at
every score level, and that the demo is reachable, turns each carton the right way and lets go before the gate.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
g = M.g
DT = 1 / 20
RUN_LIMIT_S = 700 * DT
FAR = [np.array([0.2, 0.3, 1.2]), np.array([0.2, -0.3, 1.2])]  # grasp points well away from the gate
C1, C2, C3 = M.NAMES


def quat(yaw_deg=0.0, roll_deg=0.0):
    """A carton's orientation: turned ``yaw_deg`` about the vertical, after tipping ``roll_deg`` about its own x."""
    y, r = math.radians(yaw_deg) / 2, math.radians(roll_deg) / 2
    qz = np.array([0.0, 0.0, math.sin(y), math.cos(y)])
    qx = np.array([math.sin(r), 0.0, 0.0, math.cos(r)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def carrier_at(name, t):
    return np.asarray(M.PATHS[name](t)[0], float), np.array([0.0, 0.0, 0.0, 1.0])


def seated(name, t, yaw_deg=0.0, dx=0.0, dz=0.0, roll_deg=0.0):
    c, _ = carrier_at(name, t)
    return np.array([c[0] + dx, c[1], g.CARTON_Z + dz]), quat(yaw_deg, roll_deg)


def run(frames, tcps=FAR):
    """Feed the monitor a list of {name: (pos, quat)} carton states at times t (carriers from their paths)."""
    M.reset_state()
    for t, cartons in frames:
        M.update(cartons, {n: carrier_at(n, t) for n in M.NAMES}, tcps)
    return M.score(None), M.success(None)


def through_gate(yaws, **kw):
    """Every carton riding its carrier at ``yaws`` (degrees) from the start until all have passed the gate."""
    ts = np.arange(0.0, M.READ_TIMES[C3] + 0.2, DT)
    return [(t, {n: seated(n, t, yaw, **kw.get(n, {})) for n, yaw in zip(M.NAMES, yaws, strict=True)}) for t in ts]


START_DEG = [math.degrees(y) for y in g.START_YAWS]


# -- the task file and the asset ---------------------------------------------------------------------------------------


def test_it_loads_with_the_carriers_driven_and_only_the_cartons_free():
    assert TASK.free_objects == M.NAMES
    assert TASK.score_over == "end"
    for n in M.NAMES:
        assert M.CARRIER_OF[n] in TASK.names
    for o in TASK.objects:  # plain floats for Isaac Lab
        assert all(type(v) is float for v in o.pos), o.name


def test_the_carton_asset_is_one_box_with_the_label_on_its_plus_x_face():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(HERE / "assets" / "carton.usda"))
    colliders, label_x, top_band = [], None, None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append(prim)
        if prim.GetName() == "label":
            label_x = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()[0]
        if prim.GetName() == "band_top":
            top_band = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
    assert len(colliders) == 1 and colliders[0].GetName() == "box"
    scale = colliders[0].GetAttribute("xformOp:scale").Get()
    assert tuple(round(v, 6) for v in scale) == pytest.approx(g.SIZE)
    assert label_x > g.SIZE[0] / 2  # on the +x face, outside the collider
    assert top_band[0] > 0 and top_band[2] > g.SIZE[2] / 2  # the band shows on top, on the label's side
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(g.MASS)


# -- the line and the gate ---------------------------------------------------------------------------------------------


def test_the_line_runs_at_4_cm_per_s_and_brings_a_carton_through_the_gate_every_5_5_s():
    for n in M.NAMES:
        path = M.PATHS[n]
        assert np.linalg.norm(path(0.0)[1]) == 0.0 and np.linalg.norm(path(1 / 240)[1]) < 1e-3
        assert path(3.0)[1][1] == pytest.approx(g.SPEED)
        assert M.carrier_y(n, M.arrival(n, 0.3)) == pytest.approx(0.3)
    assert np.diff(list(M.READ_TIMES.values())) == pytest.approx([5.5, 5.5])
    assert M.READ_TIMES[C3] + 1.5 < RUN_LIMIT_S  # the last read, and the 1 s success hold, inside the run limit
    for a, b in zip(M.NAMES, M.NAMES[1:]):
        assert g.STARTS[a] - g.STARTS[b] - g.CARRIER[1] > 0.1  # carriers never overlap
    assert g.STARTS[C3] - g.CARRIER[1] / 2 >= -g.BED[1] / 2 - 0.005  # the last carrier starts on the bed
    assert g.TURN_Y - g.CARRIER[1] / 2 > g.TABLE_LEFT_Y  # carriers turn down past the table's edge


def test_no_carton_starts_readable_and_doing_nothing_scores_zero():
    for yaw in g.START_YAWS:
        assert g.label_error(g.rot(g.yaw_quat(yaw))) >= math.radians(90) - 1e-9
    assert run(through_gate(START_DEG)) == (0.0, False)
    assert all(not r["read"] for r in M.reads().values()) and len(M.reads()) == 3


def test_the_gate_clears_the_cartons_and_carriers_with_room_to_spare():
    inner_far = M.SCANNER_X - g.SCANNER[0] / 2 - g.WINDOW[0] - g.LINE_X  # the scanner window's face, from the line
    inner_near = g.POST_X - g.POST[0] / 2
    corner = math.hypot(*g.HALF[:2])  # a carton turned 45 degrees
    assert inner_far - corner > 0.03 and inner_near - corner > 0.05
    assert inner_far > g.CARRIER[0] / 2 + 0.02 and inner_far > g.BED[0] / 2 + 0.01
    beam_bottom = g.TOP + g.POST[2]
    assert beam_bottom - (g.CARRIER_TOP + g.SIZE[2]) > 0.08
    assert g.GATE_Y + g.POST[1] / 2 < g.TABLE_LEFT_Y and g.TURN_Y > g.GATE_Y + 0.1
    # the scanner window looks straight across the line at the cartons' middle
    assert g.CARRIER_TOP < g.SCANNER_Z - g.WINDOW[2] / 2 and g.SCANNER_Z + g.WINDOW[2] / 2 < g.CARRIER_TOP + g.SIZE[2]


# -- scoring -----------------------------------------------------------------------------------------------------------


def test_each_carton_read_scores_a_third():
    assert run(through_gate([0, 0, 0])) == (1.0, True)
    assert run(through_gate([0, 0, START_DEG[2]]))[0] == pytest.approx(2 / 3)
    assert run(through_gate([0, START_DEG[1], START_DEG[2]]))[0] == pytest.approx(1 / 3)
    assert run(through_gate([START_DEG[0], START_DEG[1], 0]))[0] == pytest.approx(1 / 3)


def test_the_label_must_face_the_scanner_within_20_degrees():
    assert run(through_gate([19, -19, 15]))[0] == 1.0
    assert run(through_gate([25, 0, 0]))[0] == pytest.approx(2 / 3)
    assert run(through_gate([0, -25, 0]))[0] == pytest.approx(2 / 3)
    assert run(through_gate([0, 0, 180]))[0] == pytest.approx(2 / 3)  # facing the robot, not the scanner
    assert M.reads()[C3]["label_deg"] == pytest.approx(180)
    # a carton is square across, so 90 degrees off is as unreadable as the start
    assert run(through_gate([90, -90, 0]))[0] == pytest.approx(1 / 3)


def test_a_carton_lying_tipped_or_upside_down_is_not_read_even_if_its_label_faces_the_scanner():
    lying = {C1: dict(roll_deg=90, dz=g.SIZE[1] / 2 - g.SIZE[2] / 2)}  # on its side, label still facing +x
    assert run(through_gate([0, 0, 0], **lying))[0] == pytest.approx(2 / 3)
    assert M.reads()[C1]["upright"] is False and M.reads()[C1]["label_deg"] == pytest.approx(0, abs=1e-6)
    upside_down = {C2: dict(roll_deg=180)}  # label still on the +x side
    assert run(through_gate([0, 0, 0], **upside_down))[0] == pytest.approx(2 / 3)
    tipped = {C3: dict(roll_deg=15)}
    assert run(through_gate([0, 0, 0], **tipped))[0] == pytest.approx(2 / 3)
    assert run(through_gate([0, 0, 0], **{C3: dict(roll_deg=8)}))[0] == 1.0


def test_a_carton_off_its_carrier_or_held_through_the_gate_scores_nothing():
    assert run(through_gate([0, 0, 0], **{C1: dict(dx=0.04)}))[0] == pytest.approx(2 / 3)  # slid 4 cm across
    assert run(through_gate([0, 0, 0], **{C1: dict(dx=0.025)}))[0] == 1.0
    assert run(through_gate([0, 0, 0], **{C2: dict(dz=0.03)}))[0] == pytest.approx(2 / 3)  # lifted off it
    assert M.reads()[C2]["on_carrier"] is False
    # a gripper still at a carton as it passes the gate
    frames = through_gate([0, 0, 0])
    M.reset_state()
    for t, cartons in frames:
        near = cartons[C2][0] + np.array([0, 0, 0.012]) if abs(t - M.READ_TIMES[C2]) < 0.5 else FAR[0]
        M.update(cartons, {n: carrier_at(n, t) for n in M.NAMES}, [near, FAR[1]])
    assert M.score(None) == pytest.approx(2 / 3) and M.reads()[C2]["hands_clear"] is False


def test_a_carton_that_touched_the_table_never_counts_even_if_put_back():
    frames = through_gate([0, 0, 0])
    t0, cartons = frames[40]
    dropped = dict(cartons, **{C1: (np.array([0.25, 0.0, g.TOP + g.SIZE[2] / 2]), quat(0))})
    frames[40] = (t0, dropped)
    assert run(frames)[0] == pytest.approx(2 / 3)
    assert M.reads()[C1]["read"] is False and M._STATE["lost"] == {C1: "touched the table"}


def test_each_carton_is_read_once_at_the_gate_not_before_or_after():
    # turned the right way only after its carrier passed the gate: too late
    frames = through_gate(START_DEG)
    late = [(t, {n: seated(n, t, 0 if t > M.READ_TIMES[n] + 0.1 else d) for n, d in zip(M.NAMES, START_DEG)})
            for t, _ in frames]
    assert run(late)[0] == 0.0
    # right way round at the gate, turned away afterwards: still read
    after = [(t, {n: seated(n, t, d if t > M.READ_TIMES[n] + 0.1 else 0) for n, d in zip(M.NAMES, START_DEG)})
             for t, _ in frames]
    assert run(after) == (1.0, True)
    # nothing is read before its carrier reaches the gate
    M.reset_state()
    t, cartons = frames[100]
    M.update(cartons, {n: carrier_at(n, t) for n in M.NAMES}, FAR)
    assert M.reads() == {} and M.score(None) == 0.0


# -- the demo ----------------------------------------------------------------------------------------------------------


def _plans():
    return {side: ArmPlan(plan, station.BASES[side]) for side, plan in M.demo().items()}


def _spans(plan):
    t, out = 0.0, []
    for w in plan:
        out.append((t, t + w.duration, w))
        t += w.duration
    return out


def test_the_demo_is_reachable_and_within_the_joint_limits():
    for side, arm in _plans().items():
        n = round(arm.duration / DT) + 5
        actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(n)])
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, (side, arm.worst_ik_mm, arm.worst_ik_deg)
        assert np.all(actions[:, :6] >= ik.LOWER - 1e-9) and np.all(actions[:, :6] <= ik.UPPER + 1e-9), side
        # done moving before the last read; then it waits at rest until success (yam_scripted stops 3 s after)
        assert arm.duration - arm.waypoints[-1].duration < M.READ_TIMES[C3]
        assert M.READ_TIMES[C3] + 1.0 + 1.0 < arm.duration < 700 * DT - 3.0
        # no joint jumps between control steps (the IK stays on one branch)
        assert np.max(np.abs(np.diff(actions[:, :6], axis=0))) < 0.15, side


def test_the_demo_turns_every_carton_to_face_the_scanner():
    yaw = dict(zip(M.NAMES, START_DEG, strict=True))
    for _, name, _, a, b in M.TURNS:
        assert (a - yaw[name]) % 90 == pytest.approx(0)  # the fingers close across two of its sides
        yaw[name] += b - a
    for name, y in yaw.items():
        assert ((y + 180) % 360) - 180 == pytest.approx(0), name


def test_the_demo_follows_each_carton_and_lets_go_well_before_the_gate():
    plans = M.demo()
    for side, plan in plans.items():
        for t0, t1, w in _spans(plan):
            for name in M.NAMES:
                if w.label.endswith(name) and not w.label.startswith("over"):
                    assert w.pos[0] == g.LINE_X and w.pos[1] == pytest.approx(M.carrier_y(name, t1), abs=1e-9)
                    if w.label.startswith(("down", "close", "lift", "turn", "set down", "let go")):
                        assert M.carrier_y(name, t1) < g.GATE_Y - 0.12, (w.label, t1)
    # every carton's turns end before the next arm needs it, and in TURNS order for carton 2
    c2 = [(side, t1) for side, plan in plans.items() for _, t1, w in _spans(plan) if w.label == f"up {C2}"]
    starts = [(side, t0) for side, plan in plans.items() for t0, _, w in _spans(plan) if w.label == f"over {C2}"]
    right_done = max(t for s, t in c2 if s == "right")
    left_starts_down = min(t1 for _, t1, w in _spans(plans["left"]) if w.label == f"down {C2}") - M.DESCEND_S
    assert right_done < left_starts_down and len(starts) == 2


def test_the_demo_hands_are_clear_of_every_carton_as_it_passes_the_gate_and_of_the_gate_itself():
    plans = _plans()
    for name, t in M.READ_TIMES.items():
        centre = np.array([g.LINE_X, M.carrier_y(name, t), g.CARTON_Z])
        for side, arm in plans.items():
            p = arm.target(t)[0]
            assert np.linalg.norm(p - centre) > g.HAND_CLEAR + 0.05, (name, side)
    for side, arm in plans.items():
        for t in np.arange(0.0, arm.duration + 1.0, DT):
            assert arm.target(t)[0][1] < g.GATE_Y - g.POST[1] / 2 - 0.08, side


def test_the_demo_fingers_clear_the_cartons_and_carriers():
    carton_top = g.CARRIER_TOP + g.SIZE[2]
    assert M.PRE_Z - PAD_TIP_PAST_TCP - carton_top >= 0.02  # waiting over the line
    assert M.GRASP_Z - PAD_TIP_PAST_TCP - g.CARRIER_TOP > 0.03  # grasping: tips well above the carrier
    assert -FINGER_OPEN - g.SIZE[0] / 2 >= 0.015  # open fingers (9.4 cm apart) pass a 6 cm carton 1.7 cm clear
    # while moving between cartons, wherever the open hand is near a carton its tips are above its top
    plans = _plans()
    for side, arm in plans.items():
        for t0, t1, w in _spans(M.demo()[side]):
            if not w.label.startswith(("over", "up and away", "back")):
                continue
            for t in np.arange(t0, t1, DT / 2):
                p = arm.target(t)[0]
                for name in M.NAMES:
                    c = np.array([g.LINE_X, M.carrier_y(name, t)])
                    if np.hypot(*(p[:2] - c)) < 0.10:
                        assert p[2] - PAD_TIP_PAST_TCP - carton_top > 0.015, (side, w.label, name, t)


def test_the_arms_keep_apart():
    plans = _plans()
    end = max(a.duration for a in plans.values())
    for t in np.arange(0.0, end, DT):
        pl, pr = plans["left"].target(t)[0], plans["right"].target(t)[0]
        assert np.linalg.norm(pl - pr) > 0.2, t
