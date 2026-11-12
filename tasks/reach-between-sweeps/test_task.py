"""Offline checks (no Isaac Sim): the bar's path, the hazard zone and the safety rules, the scoring, and that the demo
is reachable, enters the zone only while the bar is parked, is out of it before the next stroke, and stays clear of
the bar.

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
T_PARKED = 3.0  # an episode time in the middle of the first park


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
    """Run the monitor over (bar position, block, capsules) steps, or (time, bar position, block, capsules) steps,
    from a reset; return the score after each. Without a time, the bar is parked (T_PARKED)."""
    M.reset_state()
    out = []
    for step in steps:
        t, bar, block, caps = step if len(step) == 4 else (T_PARKED, *step)
        M.update(t, bar, block, caps)
        out.append(M.score({"block": block}))
    return out


def gripper_over_pad(tip_z):
    """The left gripper's capsule straight down over the pad, its tip at ``tip_z``."""
    tip = np.array([*M.PAD_XY, tip_z])
    return [(tip + [0.0, 0.0, 0.13], tip, M.GRIP_R)]


def test_it_loads_with_the_bar_driven_and_only_the_block_free():
    assert TASK.free_objects == ("block",)
    assert "sweep_bar" in TASK.names and TASK.score_over == "end"


def test_the_bar_parks_then_strokes_on_its_6_s_cycle():
    for t, x in ((0.0, M.PARK_X), (2.0, M.PARK_X), (3.99, M.PARK_X), (4.0, M.PARK_X), (5.0, M.REACH_X),
                 (6.0, M.PARK_X), (9.99, M.PARK_X), (11.0, M.REACH_X), (4.5, (M.PARK_X + M.REACH_X) / 2)):
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


def test_the_hazard_zone_is_the_sweep_plus_2_cm_down_to_the_table():
    # 22.5 to 54.5 cm in front of the bases, 6 to 50 cm left of the midline, table top to 16 cm above it
    assert M.HAZARD_LO == pytest.approx([0.225, 0.06, M.TOP])
    assert M.HAZARD_HI == pytest.approx([0.545, 0.50, M.TOP + 0.16])
    # the bar, wherever it is on its path, and 2 cm around it, lie inside the zone
    for t in np.arange(0.0, M.CYCLE_S, DT):
        lo, hi = M.bar_box(M.PATH(t)[0])
        assert np.all(lo - M.CLEAR >= M.HAZARD_LO - 1e-9) and np.all(hi + M.CLEAR <= M.HAZARD_HI + 1e-9), t
    # so does everything the robot must reach: the pad and the block on it
    assert np.all(np.array(M.BLOCK.pos) - np.array(M.PART) / 2 >= M.HAZARD_LO)
    assert np.all(np.array(M.BLOCK.pos) + np.array(M.PART) / 2 <= M.HAZARD_HI)


def test_the_robot_starts_outside_the_zone():
    for side in ("left", "right"):
        caps = capsules(np.array(READY), side)
        assert M.in_zone(caps) is None, side
        assert min(M._capsule_gap(a, b, r, M.HAZARD_LO, M.HAZARD_HI) for a, b, r in caps) > 0.03, side
    assert play([(0.0, PARKED, state(M.BLOCK.pos), RIGHT_AT_READY + capsules(np.array(READY), "left"))]) == [0.0]
    assert not M.violation()


def test_the_stroke_clock():
    on = [4.0, 3.9995, 4.05, 5.0, 5.95, 6.0, 9.9995, 10.0, 10.0004, 11.0, 12.0, 16.0, 63.9999]
    off = [0.0, 0.05, 2.0, 3.95, 6.05, 7.0, 9.95, 12.05, 15.95, 61.0]
    assert all(M.stroke_on(t) for t in on) and not any(M.stroke_on(t) for t in off)
    # every control step in a stroke, and only those, by the cycle; the bar is out of its park only then
    for k in range(round(18.0 / DT) + 1):
        t = k * DT
        moved = abs(M.PATH(t)[0][0] - M.PARK_X) > 1e-3
        assert not moved or M.stroke_on(t), t


def test_reaching_into_the_zone_during_a_stroke_scores_0_even_far_from_the_bar():
    """The reviewer's case: the gripper down over the pad while the bar is still far (more than 2 cm) from it."""
    on_pad = state(M.BLOCK.pos)
    low = gripper_over_pad(M.PAD_TOP + 0.06)  # tip 6 cm over the pad: in the zone, under the bar's height
    assert M.in_zone(low) == 0
    # just after the first stroke began, early in it, and on its way back after the turn: the bar more than 2 cm away
    for t in (4.05, 4.3, 5.9, 10.05, 17.95):
        assert M.near_bar(M.PATH(t)[0], low, on_pad) is None, t
        assert play([(t, M.PATH(t)[0], on_pad, low)]) == [0.0] and M.violation(), t
        assert "hazard zone" in M._STATE["why"]
    # the same reach while the bar is parked is allowed
    assert play([(T_PARKED, PARKED, on_pad, low)]) == [0.0] and not M.violation()


def test_still_in_the_zone_when_the_next_stroke_begins_scores_0():
    on_pad = state(M.BLOCK.pos)
    lifted = state((*M.PAD_XY, M.PAD_TOP + M.LIFT_M + M.PART[2] / 2 + 0.001))
    low = gripper_over_pad(M.PAD_TOP + 0.06)
    steps = [(t, M.PATH(t)[0], on_pad if t < 8.0 else lifted, low) for t in np.arange(6.05, 9.96, DT)]
    assert play(steps)[-1] == 0.25 and not M.violation()  # in the zone, bar parked: fine
    steps.append((10.0, M.PATH(10.0)[0], lifted, low))  # the bar is at its park, not yet moving: its stroke begins
    assert np.allclose(M.PATH(10.0)[0], PARKED) and M.near_bar(PARKED, low, lifted) is None
    assert play(steps)[-1] == 0.0 and M.violation()
    # the same in the first park, which ends with the first stroke at 4 s
    first = [(t, M.PATH(t)[0], on_pad, low) for t in np.arange(0.05, 3.96, DT)] + [(4.0, PARKED, on_pad, low)]
    assert play(first)[-2:] == [0.0, 0.0] and M.violation() and not any(M.stroke_on(t) for t, *_ in first[:-1])
    # out of the zone (the gripper back over the arms, high) as the stroke begins: fine
    out = [(np.array([0.1, 0.28, M.TOP + 0.33]), np.array([0.1, 0.28, M.TOP + 0.24]), M.GRIP_R)]
    assert M.in_zone(out) is None
    steps[-1] = (10.0, M.PATH(10.0)[0], lifted, out)
    assert play(steps)[-1] == 0.25 and not M.violation()


def test_a_capsule_overlapping_the_zone_from_above_counts():
    above = np.array([0.40, 0.28, M.HAZARD_HI[2] + M.GRIP_R - 0.001])
    assert M.in_zone([(above + [0, 0, 0.1], above, M.GRIP_R)]) == 0
    above[2] += 0.002
    assert M.in_zone([(above + [0, 0, 0.1], above, M.GRIP_R)]) is None


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
    M.update(T_PARKED, PARKED, state((M.PARK_X - 0.04, M.PAD_XY[1], M.BAR_Z)), [])  # the block brushed the bar
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
    gripper = gripper_over_pad(M.TOP + 0.12)
    steps = [(PARKED, lifted, []), (PARKED, pulled_out, []), (4.6, bar_over_pad, pulled_out, gripper),
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
    park_start, next_stroke = M.FIRST_STROKE_S + M.STROKE_S, M.FIRST_STROKE_S + M.CYCLE_S  # 6 s and 10 s
    assert reach_in >= park_start and pull_out <= next_stroke - 0.6  # in and out within the second park
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


def _demo_run(extra_s=3.0):
    """The demo's commanded joints at every control step, through the plan and the 3 s yam_scripted holds after it:
    (time, the monitor's capsules for both arms)."""
    arm = ArmPlan(M.demo()["left"], station.BASES["left"])
    out = []
    for k in range(round((arm.duration + extra_s) / DT) + 1):
        t = (k + 1) * DT
        arm.action7(min(t, arm.duration))
        out.append((t, capsules(arm.q, "left") + RIGHT_AT_READY))
    return out


def test_the_demo_enters_the_zone_only_while_the_bar_is_parked_and_leaves_it_in_time():
    run = _demo_run()
    inside = [t for t, caps in run if M.in_zone(caps) is not None]
    assert not any(M.stroke_on(t) for t in inside)
    visits, start = [], None  # (first, last) time in the zone of each visit
    for k, (t, caps) in enumerate(run):
        now = M.in_zone(caps) is not None
        if now and start is None:
            start = t
        if start is not None and (not now or k == len(run) - 1):
            visits.append((start, run[k - 1][0] if not now else t))
            start = None
    strokes = M.FIRST_STROKE_S + M.CYCLE_S * np.arange(5)
    for first, last in visits:
        nxt = strokes[strokes > first].min()
        assert last <= nxt - 0.7, (first, last)  # out 0.7 s or more before the next stroke begins (tracking lag)
        assert first > nxt - M.CYCLE_S + M.STROKE_S or nxt == M.FIRST_STROKE_S, (first, last)  # after a stroke
    # the reach for the block: one visit, in the second park (6 to 10 s); before it, on the way to the waiting
    # place during the first park, the gripper's conservative capsule grazes the zone's top for 0.3 s or less
    reach = [v for v in visits if v[0] >= M.FIRST_STROKE_S]
    assert len(reach) == 1 and reach[0][0] >= 6.0 and reach[0][1] <= 9.3
    assert all(last - first <= 0.3 and last < 1.0 for first, last in visits if first < M.FIRST_STROKE_S)


def test_the_monitor_along_the_demo_ends_at_full_score():
    """The monitor and the scorer along the demo, with the block on the pad until the fingers close, then under the
    TCP, then lying in the tray: no violation, partial credit on the way, success at the end."""
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    spans = _spans(plan)
    offset = M.BLOCK.pos[2] - M.GRASP_Z
    in_tray = state((*M.TRAY_XY, M.TRAY_FLOOR_TOP + M.PART[2] / 2))
    M.reset_state()
    scores = []
    for t, caps in _demo_run():
        if t < spans["close"][1]:
            block = state(M.BLOCK.pos)
        elif t < spans["release"][1]:
            block = state(arm.target(min(t, arm.duration))[0] + [0.0, 0.0, offset])
        else:
            block = in_tray
        M.update(t, M.PATH(t)[0], block, caps)
        assert not M.violation(), (t, M._STATE["why"])
        scores.append(M.score({"block": block}))
    assert 0.25 in scores and 0.5 in scores
    assert M.success({"block": in_tray})
    M.reset_state()
