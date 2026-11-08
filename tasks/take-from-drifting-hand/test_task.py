"""Offline checks (no Isaac Sim): the hand's drift, its geometry, the monitor and every scoring level, and that the
demo is reachable, follows the hand and keeps clear of it.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
SPECS = {o.name: o for o in TASK.objects}
PALM_SPEC = SPECS["hand_palm"]
DOWN_ROT = grip_rotation()  # TCP pointing down, closing along y


def state(pos, size=M.BLOCK, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half,
                       pos + half)


def palm_at(t):
    pos, vel = PALM_SPEC.path(t)
    return state(pos, size=PALM_SPEC.size, vel=vel)


def seated(t, d=(0.0, 0.0, 0.0)):
    return state(M.seat(t) + np.asarray(d), vel=M.PATH(t)[1])


FAR = [(np.array([0.0, -0.3, 1.5]), np.array([0.0, -0.3, 1.6]), 0.01)]  # a robot nowhere near the hand
IN_TRAY = state((M.TRAY_XY[0] + 0.02, M.TRAY_XY[1] - 0.02, M.TRAY_FLOOR_TOP + M.BLOCK[2] / 2))


def play(steps):
    """Reset the monitor and feed it (block, t, capsules) control steps; return the last block."""
    M.reset_state()
    for block, t, caps in steps:
        M.update(block, palm_at(t), caps)
    return steps[-1][0]


def outcome(steps):
    last = play(steps)
    objs = {"tool_block": last}
    return 1.0 if M.success(objs) else M.score(objs)


def arm_capsules_at(q, side, finger):
    """The monitor's capsules for one arm at joints ``q``, from the URDF (as the sim reports its links)."""
    base = np.asarray(station.BASES[side], float)
    poses = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q)})
    chain = [poses[n][:3, 3] + base for n in M.CHAIN]
    tcp, rot = ik.tcp_pose(q, base)
    return M.arm_capsules(chain) + M.gripper_capsules(chain[-1], tcp, rot, finger)


# -- the drift ---------------------------------------------------------------------------------------------------------


def test_it_loads_with_the_hand_driven_and_only_the_block_free():
    assert set(TASK.free_objects) == {"tool_block"}
    assert {f"hand_{n}" for n, _, _ in M.HAND_PARTS} <= set(TASK.names)
    assert TASK.score_over == "end"
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos) and all(type(v) is float for v in o.size), o.name


def test_the_hand_drifts_smoothly_never_pauses_and_stays_in_a_small_area():
    ts = np.arange(0.0, 300.0, 1 / 40)
    ps = np.array([M.PATH(t)[0] for t in ts])
    vs = np.array([M.PATH(t)[1] for t in ts])
    assert np.allclose(ps[0], M.HOME) and np.allclose(vs[0], 0.0)  # starts at home, at rest
    for i in range(3):
        assert np.allclose(np.gradient(ps[:, i], ts)[5:-5], vs[5:-5, i], atol=2e-3)  # velocity is the motion's
    speed = np.linalg.norm(vs, axis=1)
    full = ts >= M.RAMP_S
    assert speed[full].min() > 0.02  # never pauses: 2 cm/s or more, all the way past the 2-minute trial
    assert 0.045 < speed.max() < 0.06 and 0.03 < speed[full].mean() < 0.045  # a few cm/s
    accel = np.linalg.norm(np.gradient(vs, ts, axis=0)[5:-5], axis=1)
    assert accel.max() < 0.1  # m/s^2; the seated block would need about 10 m/s^2 to slide on the palm
    span = ps.max(axis=0) - ps.min(axis=0)
    assert span[0] < 0.07 and span[1] < 0.11 and span[2] <= 0.0101
    assert span[0] > 0.05 and span[1] > 0.09 and span[2] > 0.008


def test_the_drift_never_repeats_within_a_trial():
    # for every delay from 0.5 s to the 2-minute run limit, the hand's path over a 40 s stretch differs somewhere by
    # 2 cm or more from the same stretch delayed: there is no cycle a policy could learn and replay
    ts = np.arange(2.0, 42.0, 0.05)
    base = np.array([M.palm_top(t) for t in ts])
    for lag in np.arange(0.5, 120.0, 0.05):
        shifted = np.array([M.palm_top(t + lag) for t in ts])
        assert np.linalg.norm(shifted - base, axis=1).max() > 0.02, lag


def test_tracking_is_needed():
    # across the closing direction (y) the open fingers forgive (9.4 - 2.5) / 2 = 3.5 cm; the hand drifts more than
    # that within many 2 s windows; and it bobs about 1 cm, so the grip height has to follow it too
    ts = np.arange(M.RAMP_S, 120.0, 0.25)
    moves = [np.ptp([M.palm_top(t + d)[1] for d in np.arange(0.0, 2.0, 0.05)]) for t in ts]
    assert np.mean(np.array(moves) > 0.035) > 0.3
    assert np.ptp([M.palm_top(t)[2] for t in np.arange(0.0, 120.0, 0.05)]) > 0.008


def test_the_hand_is_a_rigid_set_of_boxes_that_do_not_touch():
    names = [f"hand_{n}" for n, _, _ in M.HAND_PARTS]
    offsets = {n: np.subtract(SPECS[n].pos, M.palm_top(0.0)) for n in names}
    for t in (0.0, 1.3, 7.7, 55.5):
        for n in names:
            pos, vel = SPECS[n].path(t)
            assert np.allclose(vel, M.PATH(t)[1]) and np.allclose(pos - M.palm_top(t), offsets[n])
    boxes = [(np.asarray(SPECS[n].pos) - np.asarray(SPECS[n].size) / 2,
              np.asarray(SPECS[n].pos) + np.asarray(SPECS[n].size) / 2) for n in names]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            gap = np.max(np.maximum(boxes[i][0] - boxes[j][1], boxes[j][0] - boxes[i][1]))
            assert gap >= M.GAP - 1e-9, (names[i], names[j])
    # the monitor's hand boxes match the spawned ones
    for (lo, hi), n in zip(M.hand_boxes(PALM_SPEC.pos), names):
        assert np.allclose((lo + hi) / 2, SPECS[n].pos) and np.allclose(hi - lo, SPECS[n].size)


def test_the_block_sits_on_the_palm_between_the_stops_and_the_hand_clears_the_table_and_tray():
    b = SPECS["tool_block"]
    assert b.pos[2] - M.BLOCK[2] / 2 == pytest.approx(M.HOME[2])  # on the palm's top
    for s in ("stop_fingers", "stop_wrist"):
        gap = abs(SPECS[f"hand_{s}"].pos[0] - b.pos[0]) - M.STOP[0] / 2 - M.BLOCK[0] / 2
        assert gap == pytest.approx(M.END_CLEAR)
        assert SPECS[f"hand_{s}"].pos[0] + M.STOP[0] / 2 <= PALM_SPEC.pos[0] + M.PALM[0] / 2 + 1e-9
    tray_hi = np.array([M.TRAY_XY[0] + M.TRAY_IN[0] / 2 + M.T_WALL, M.TRAY_XY[1] + M.TRAY_IN[1] / 2 + M.T_WALL])
    for t in np.arange(0.0, 120.0, 0.1):
        lows = [lo for lo, _ in M.hand_boxes(PALM_SPEC.path(t)[0])]
        assert min(lo[2] for lo in lows) - M.TOP > 0.03  # the hand stays 3 cm or more above the table
        palm_lo = lows[0]
        assert palm_lo[1] - tray_hi[1] > 0.05  # and 5 cm or more to the side of the tray


# -- monitor and scoring ---------------------------------------------------------------------------------------------


def test_a_block_rocking_in_its_seat_is_still_in_the_palm():
    t = 9.0
    for d in ((0.008, 0.0, 0.002), (0.0, -0.01, 0.004), (0.0, 0.0, 0.012)):
        play([(seated(t, d), t, FAR)])
        assert M._STATE["left"] is None
    play([(seated(t, (0.0, 0.0, 0.02)), t, FAR)])
    assert M._STATE["left"] == "up"
    play([(seated(t, (0.0, 0.02, 0.005)), t, FAR)])
    assert M._STATE["left"] == "off"


def test_scores_for_every_outcome():
    t0, t1, t2 = 5.0, 6.0, 20.0
    up1 = seated(t0, (0.003, 0.004, 0.02))
    up2 = seated(t1, (0.0, 0.005, M.TAKEN_M + 0.01))
    on_table = state((0.25, -0.30, M.TOP + M.BLOCK[2] / 2))
    held = state((*M.TRAY_XY, M.TRAY_FLOOR_TOP + 0.10))
    pushed = seated(t0, (0.0, 0.03, 0.0))
    bump_tcp = M.palm_top(t0) + [0.0, 0.06, 0.02]  # open fingers dipped to 7 mm over the palm's edge
    bump = M.gripper_capsules(bump_tcp + [0, 0, 0.134], bump_tcp, DOWN_ROT, FINGER_OPEN)
    assert M.gap_to_hand(palm_at(t0).pos, bump) < M.CLEAR
    # 1.0: taken straight up, never near the hand, rests in the tray
    assert outcome([(up1, t0, FAR), (up2, t1, FAR), (IN_TRAY, t2, FAR)]) == 1.0
    # 0.5: as above, but the robot came within 1 cm of the hand once
    assert outcome([(seated(t0), t0, bump), (up2, t1, FAR), (IN_TRAY, t2, FAR)]) == 0.5
    # 0.25: taken up, then dropped on the table, or still held above the tray
    assert outcome([(up1, t0, FAR), (up2, t1, FAR), (on_table, t2, FAR)]) == 0.25
    assert outcome([(up1, t0, FAR), (up2, t1, FAR), (held, t2, FAR)]) == 0.25
    # 0: nothing done; lifted only 2 cm and put back; knocked off the palm, even if then put in the tray
    assert outcome([(seated(t0), t0, FAR), (seated(t2), t2, FAR)]) == 0.0
    assert outcome([(up1, t0, FAR), (seated(t2), t2, FAR)]) == 0.0
    assert outcome([(pushed, t0, FAR), (on_table, t1, FAR), (IN_TRAY, t2, FAR)]) == 0.0
    # still falling into the tray is not yet a success
    assert not M.success({"tool_block": state(IN_TRAY.pos, vel=(0, 0, -0.2))})


# -- the demo --------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        a, b = spans.get(w.label, (t, t))
        spans[w.label] = (min(a, t), t + w.duration)
        t += w.duration
    return spans


def _run(plan):
    arm = ArmPlan(plan, station.BASES["right"])
    steps = round(arm.duration / DT) + 5
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(steps)])
    return arm, actions


def test_the_demo_is_reachable():
    arm, actions = _run(M.demo()["right"])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 15  # far inside the 120 s run limit


def test_the_whole_drift_area_is_reachable_from_hover_to_lift_height():
    # wherever and whenever a policy chooses to take the block, a straight-down grip can reach it and lift it
    for t in np.arange(0.0, 120.0, 1.7):
        x, y, z = M.palm_top(t)
        for dz in (M.GRASP_DZ, M.HOVER_DZ, M.LIFT_DZ):
            _, pe, re = ik.solve((x, y, z + dz), ik.matrix_to_quat_xyzw(DOWN_ROT), base_pos=station.BASES["right"])
            assert pe < 5e-4 and np.degrees(re) < 0.1, (t, dz)


def test_the_demo_follows_the_hand_every_control_step_and_takes_the_block_straight_up():
    plan = M.demo()["right"]
    arm = ArmPlan(plan, station.BASES["right"])
    spans = _spans(plan)
    for t in np.arange(M.MEET_S + DT, M.LEAVE_S + 1e-9, DT):
        tcp, _, _, _ = arm.target(t)
        assert np.allclose(tcp, M.palm_top(t + M.LEAD_S) + [0, 0, M.follow_dz(t)], atol=1e-9), t
    assert spans["close"][0] >= M.DESCEND_S[1] - 1e-9
    for t in np.arange(M.DESCEND_S[1], M.CLOSE_S[0], DT):
        assert arm.target(t)[2] == pytest.approx(FINGER_OPEN)
    assert arm.target(M.LIFT_S[0])[2] == pytest.approx(FINGER_CLOSED)
    assert spans["lift"][0] >= M.CLOSE_S[1]
    # a block held at the TCP would be out of its seat only upward, then taken; and lifted clear when the arm leaves
    M.reset_state()
    for t in np.arange(M.CLOSE_S[1], M.LEAVE_S + 1e-9, DT):
        tcp = arm.target(t)[0]
        held = state(tcp - [0, 0, M.GRASP_DZ - M.BLOCK[2] / 2])
        M.update(held, palm_at(t), FAR)
    assert M._STATE["left"] == "up" and M._STATE["taken"]
    # the hand is moving throughout: no pause to wait for
    for t in np.arange(M.DESCEND_S[0], M.LIFT_S[1], DT):
        assert np.linalg.norm(M.PATH(t)[1]) > 0.02


def test_the_demo_keeps_both_arms_clear_of_the_hand():
    plan = M.demo()["right"]
    arm = ArmPlan(plan, station.BASES["right"])
    left = arm_capsules_at(np.array(READY), "left", FINGER_OPEN)
    worst = np.inf
    for k in range(round(arm.duration / DT) + 5):
        t = (k + 1) * DT
        a, _ = arm.action7(t)
        finger = arm.target(t)[2]
        if M.CLOSE_S[0] < t < M.LEAVE_S + 0.6:  # closed on the block, the fingers stop at its width
            finger = max(finger, -M.BLOCK[1] / 2)
        caps = arm_capsules_at(a[:6], "right", finger) + left
        worst = min(worst, M.gap_to_hand(PALM_SPEC.path(t)[0], caps))
    # the closest any capsule comes is the fingers' tips over the palm while gripping: 2.5 cm or more away, so 1.5 cm
    # or more beyond the 1 cm rule
    assert worst >= 0.025 and worst - M.CLEAR >= 0.015, worst


def test_the_fingers_clear_the_hand_and_grip_the_block():
    tips_over_palm = M.GRASP_DZ - PAD_TIP_PAST_TCP
    assert tips_over_palm - M.FINGER_R - M.CLEAR >= 0.015  # the grasp's margin under the 1 cm rule
    # more than the hand's bob (0.5 cm either way), so a lag of a whole bob in height still keeps 1 cm clear
    assert tips_over_palm - M.FINGER_R - M.CLEAR - 0.005 >= 0.01
    assert M.BLOCK[2] - tips_over_palm >= 0.015  # the pads hold 1.5 cm or more of the block's side
    assert tips_over_palm - M.STOP[2] - M.GAP > 0.015  # the tips pass 1.5 cm or more above the stops
    assert M.HOVER_DZ - PAD_TIP_PAST_TCP - M.BLOCK[2] >= 0.02
    # open fingers 9.4 cm apart around the 2.5 cm block: 3.45 cm either side
    assert (0.094 - M.BLOCK[1]) / 2 > 0.03
    # carried over the tray wall, and let go 1 cm above its floor
    assert M.TRAVEL_Z - M.GRASP_DZ - (M.TOP + M.T_WALL_H) > 0.075
    assert M.PLACE_Z - M.GRASP_DZ - M.TRAY_FLOOR_TOP == pytest.approx(0.01)
    # the open fingers (and the block, 10 cm long) fit inside the tray
    assert M.TRAY_IN[1] / 2 - (0.047 + 0.016) > 0.005 and M.TRAY_IN[0] - M.BLOCK[0] > 0.05
