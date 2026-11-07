"""Offline checks (no Isaac Sim): the bar's path, the safety rule, the scoring, and that the demo is reachable, on time
and stays clear of the bar.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
PARKED = np.array([M.PARK_X, M.PAD_XY[1], M.BAR_Z])


def state(pos, size=M.PART, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def capsules(q, side):
    """One arm's capsules from the URDF's forward kinematics, as the monitor builds them from the simulator's links."""
    base = np.asarray(station.BASES[side])
    links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})
    tcp, rot = ik.tcp_pose(q, base)
    return M.arm_capsules([links[n][:3, 3] + base for n in M.CHAIN], tcp, rot[:, 2])


RIGHT_AT_READY = capsules(np.array(READY), "right")  # yam_scripted holds the arm without a plan at READY


def play(steps):
    """Run the monitor over (bar position, block, capsules) steps, from a reset; return the score after each."""
    M.reset_state()
    out = []
    for bar, block, caps in steps:
        M.update(bar, block, caps)
        out.append(M.score({"block": block}))
    return out


def test_it_loads_with_the_bar_driven_and_only_the_block_free():
    assert TASK.free_objects == ("block",)
    assert "sweep_bar" in TASK.names and TASK.score_over == "end"


def test_the_bar_strokes_then_parks_on_its_6_s_cycle():
    for t, x in ((0.0, M.PARK_X), (1.0, M.REACH_X), (2.0, M.PARK_X), (4.0, M.PARK_X), (5.99, M.PARK_X),
                 (7.0, M.REACH_X), (0.5, (M.PARK_X + M.REACH_X) / 2)):
        assert M.PATH(t)[0][0] == pytest.approx(x, abs=1e-9), t
        assert M.PATH(t)[0][1:] == pytest.approx([M.PAD_XY[1], M.BAR_Z]), t
    ts = np.arange(0.0, 18.0, 1 / 240)
    xs = np.array([M.PATH(t)[0][0] for t in ts])
    vx = np.array([M.PATH(t)[1][0] for t in ts])
    assert np.max(np.abs(np.diff(xs))) < 0.40 / 240  # continuous: peak speed 0.39 m/s
    assert np.allclose(np.gradient(xs, ts)[5:-5], vx[5:-5], atol=2e-3)  # the reported velocity is the motion's
    assert np.all(xs <= M.PARK_X + 1e-12) and np.all(xs >= M.REACH_X - 1e-12)


def test_the_block_cannot_be_reached_without_entering_the_sweep():
    # the pad and the block lie wholly under the bar's sweep, the block's top below its underside
    for lo, hi in ((np.array(M.BLOCK.pos[:2]) - 0.025, np.array(M.BLOCK.pos[:2]) + 0.025),
                   (np.array(M.PAD_XY) - M.PAD[0] / 2, np.array(M.PAD_XY) + M.PAD[0] / 2)):
        assert np.all(lo >= M.SWEPT_LO[:2]) and np.all(hi <= M.SWEPT_HI[:2])
    assert M.SWEPT_LO[2] - (M.BLOCK.pos[2] + M.PART[2] / 2) == pytest.approx(0.03)
    # and the bar passes 3 cm over the block resting on the pad: not a violation by itself
    for t in np.arange(0.0, M.CYCLE_S, DT):
        assert M.near_bar(M.PATH(t)[0], [], state(M.BLOCK.pos)) is None


def test_the_tray_and_the_waiting_place_are_outside_the_sweep():
    tray_far_x = M.TRAY_XY[0] + M.TRAY_IN / 2 + M.WALL
    assert M.SWEPT_LO[0] - tray_far_x > M.CLEAR
    assert M._clear_of_sweep(state((*M.TRAY_XY, M.TRAY_FLOOR_TOP + M.PART[2] / 2)))
    assert M.SWEPT_LO[0] - (M.WAIT_X + M.GRIP_R) >= M.CLEAR  # the gripper's capsule, even at the bar's closest


def test_the_safety_rule_catches_robot_parts_and_the_block():
    bar = PARKED
    lo, _ = M.bar_box(bar)
    above = bar + [0.0, 0.0, M.BAR[2] / 2]
    far = state((0.15, 0.40, M.TOP + 0.10))
    # a link capsule passing 1 cm (plus its radius) beside the bar's near face: too close; 3 cm: fine
    for gap, bad in ((0.01, True), (0.03, False)):
        x = lo[0] - M.ARM_R - gap
        cap = [(np.array([x, 0.2, 0.9]), np.array([x, 0.3, 0.9]), M.ARM_R)]
        assert (M.near_bar(bar, cap, far) is not None) == bad, gap
    # the gripper's capsule over the bar: its radius counts
    tip = above + [0.0, 0.0, M.GRIP_R + 0.015]
    assert M.near_bar(bar, [(tip + [0, 0, 0.13], tip, M.GRIP_R)], far) is not None
    # the block 1.5 cm from the bar, anywhere (held or not)
    near = state((lo[0] - 0.015 - M.PART[0] / 2, bar[1], bar[2]))
    assert M.near_bar(bar, [], near) is not None
    assert M.near_bar(bar, RIGHT_AT_READY, far) is None


def test_success_needs_the_block_still_on_the_tray_floor_and_no_violation():
    x, y = M.TRAY_XY
    z = M.TRAY_FLOOR_TOP + M.PART[2] / 2
    M.reset_state()
    assert M.success({"block": state((x + 0.03, y - 0.03, z))})
    assert not M.success({"block": state((x, y, z), vel=(0, 0, -0.1))})  # still falling
    assert not M.success({"block": state((x, y, z + 0.05))})  # held above the tray
    assert not M.success({"block": state((x + 0.1, y, M.TOP + M.PART[2] / 2))})  # beside the tray
    M.update(PARKED, state((M.PARK_X - 0.04, M.PAD_XY[1], M.BAR_Z)), [])  # the block brushed the bar earlier
    assert not M.success({"block": state((x, y, z))})
    M.reset_state()


def test_partial_credit_is_the_best_so_far():
    on_pad = state(M.BLOCK.pos)
    lifted = state((*M.PAD_XY, M.PAD_TOP + M.LIFT_M + M.PART[2] / 2 + 0.001))
    nudged = state((M.PAD_XY[0], M.PAD_XY[1], M.PAD_TOP + 0.02 + M.PART[2] / 2))  # up 2 cm only
    pulled_out = state((M.WAIT_X, M.PAD_XY[1], M.TOP + 0.15))
    dropped = state((0.20, 0.0, M.TOP + M.PART[2] / 2))
    in_tray = state((*M.TRAY_XY, M.TRAY_FLOOR_TOP + M.PART[2] / 2))
    caps = RIGHT_AT_READY
    assert play([(PARKED, s, caps) for s in (on_pad, nudged)]) == [0.0, 0.0]
    assert play([(PARKED, s, caps) for s in (on_pad, lifted, on_pad)]) == [0.0, 0.25, 0.25]  # put back: keeps 0.25
    assert play([(PARKED, s, caps) for s in (lifted, pulled_out, dropped)]) == [0.25, 0.5, 0.5]
    assert play([(PARKED, s, caps) for s in (lifted, pulled_out, in_tray)]) == [0.25, 0.5, 0.5]
    # pushed off the pad to the table without lifting it 3 cm: nothing
    assert play([(PARKED, s, caps) for s in (on_pad, dropped)]) == [0.0, 0.0]


def test_a_violation_scores_0_for_the_rest_of_the_trial():
    lifted = state((*M.PAD_XY, M.PAD_TOP + M.LIFT_M + M.PART[2] / 2 + 0.001))
    pulled_out = state((M.WAIT_X, M.PAD_XY[1], M.TOP + 0.15))
    in_tray = state((*M.TRAY_XY, M.TRAY_FLOOR_TOP + M.PART[2] / 2))
    bar_over_pad = np.array([M.PAD_XY[0], M.PAD_XY[1], M.BAR_Z])
    gripper = [(np.array([*M.PAD_XY, M.TOP + 0.25]), np.array([*M.PAD_XY, M.TOP + 0.12]), M.GRIP_R)]
    steps = [(PARKED, lifted, []), (PARKED, pulled_out, []), (bar_over_pad, pulled_out, gripper),
             (PARKED, in_tray, []), (PARKED, in_tray, [])]
    assert play(steps) == [0.25, 0.5, 0.0, 0.0, 0.0]
    assert M.violation() and not M.success({"block": in_tray})
    M.reset_state()  # a new trial starts clean
    assert not M.violation() and M.score({"block": in_tray}) == 0.0


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        spans[w.label] = (t, t + w.duration)
        t += w.duration
    return spans


def _bar_clear_of_gripper_over_pad(t):
    """The bar is far enough from a gripper (capsule) over the pad that the gripper may be down there."""
    x = M.PATH(t)[0][0] - M.BAR[0] / 2
    return x - (M.PAD_XY[0] + M.GRIP_R) >= M.CLEAR


def test_the_demo_is_reachable_and_works_while_the_bar_is_parked():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    spans = _spans(plan)
    reach_in, pull_out = spans["reach in"][0], spans["pull out"][1]
    assert reach_in >= M.STROKE_S and pull_out <= M.CYCLE_S - 0.2  # in and out within one park, 0.2 s to spare
    assert all(_bar_clear_of_gripper_over_pad(t) for t in np.arange(reach_in, pull_out + 0.3, 0.01))
    assert arm.duration < 30


def test_along_the_demo_the_tcp_and_the_held_block_stay_clear_of_the_bar():
    """The commanded TCP (a point) and, while the fingers hold it, the block under it, against the bar's box plus
    CLEAR, at both ends of every control step. This checks the TCP only; the next test checks the whole arm."""
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    spans = _spans(plan)
    held = (spans["close"][1], spans["release"][0])
    offset = M.BLOCK.pos[2] - M.GRASP_Z  # the block's centre above the TCP, as gripped
    for k in range(round(arm.duration / DT) + 5):
        t = (k + 1) * DT
        tcp = arm.target(t)[0]
        for tb in (k * DT, t):
            lo, hi = M.bar_box(M.PATH(tb)[0])
            assert M._to_box(tcp, lo, hi)[0] >= M.CLEAR, (t, tcp)
            if held[0] <= t <= held[1]:
                block = state(tcp + [0.0, 0.0, offset])
                assert M._box_gap(block.low, block.high, lo, hi) >= M.CLEAR, (t, tcp)


def test_along_the_demo_every_link_capsule_stays_clear_of_the_bar():
    """Every capsule the monitor checks, built from the URDF's forward kinematics on the demo's commanded joints
    (both arms), against the bar where it is at both ends of every control step: the monitor would not fire."""
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    for k in range(round(arm.duration / DT) + 5):
        t = (k + 1) * DT
        arm.action7(t)
        caps = capsules(arm.q, "left") + RIGHT_AT_READY
        for tb in (k * DT, t):
            assert M.near_bar(M.PATH(tb)[0], caps, None) is None, (t, arm.target(t)[3])


def test_the_demo_grips_the_block_and_lifts_it_high_enough():
    plan = {w.label: w for w in M.demo()["left"]}
    assert plan["descend"].pos[2] == pytest.approx(M.BLOCK.pos[2] + 0.004)
    hang = M.TRAVEL_Z - (M.GRASP_Z - (M.BLOCK.pos[2] - M.PART[2] / 2))  # the block's underside, lifted
    assert hang - M.PAD_TOP >= M.LIFT_M
    tips = M.GRASP_Z - PAD_TIP_PAST_TCP
    assert tips - M.PAD_TOP >= 0.0015  # the fingertips stop above the pad
