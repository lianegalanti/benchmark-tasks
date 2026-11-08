"""Offline checks (no Isaac Sim): the lanes' timing, that taking parts in the order they come out fails while the
deadline order fits, the scoring and the straight-to-the-bin rule, and that the demo is reachable, on time and clear.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP
from yam_arena.constants import READY
from yam_arena.demo import ArmPlan, Waypoint, grip_rotation
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
RUN_LIMIT_S = 900 * DT
PAD_HALF_GAP = 0.047  # open pads' inner faces, either side of the TCP
PAD_T = 0.01  # a pad's thickness, generously
REACH_Y = (M.FEED_Y + 0.02, M.END_Y - 0.04)  # where any gripper could at best close on a part (generous to the robot)


def state(pos, size=(M.CUBE,) * 3, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def carrier_state(lane, k, t):
    pos, vel = M.carrier_path(lane, k)(t)
    return state(pos, M.CARRIER, vel)


def riding(lane, k, t):
    pos, vel = M.carrier_path(lane, k)(t)
    return state((pos[0], pos[1], M.CUBE_Z), vel=vel)


def demo_arm():
    plan = M.demo()["left"]
    return plan, ArmPlan(plan, station.BASES["left"])


def spans(plan):
    t, out = 0.0, {}
    for w in plan:
        out[w.label] = (t, t + w.duration)
        t += w.duration
    return out


# -- the scene and the lanes ---------------------------------------------------------------------------------------


def test_it_loads_with_only_the_four_cubes_free():
    assert sorted(TASK.free_objects) == sorted(f"part_{M.name(*p)}" for p in M.PARTS)
    assert TASK.score_over == "end"
    for spec in M.OBJECTS:  # Isaac Lab wants plain floats
        assert all(type(v) is float for v in spec.pos), spec.name


def test_each_lane_runs_at_its_speed_smoothly_and_the_cubes_cannot_slip():
    ts = np.arange(0.0, 60.0, 1 / 240)
    for lane, k in M.PARTS:
        path = M.carrier_path(lane, k)
        ys = np.array([path(t)[0][1] for t in ts])
        vy = np.array([path(t)[1][1] for t in ts])
        assert np.allclose(np.gradient(ys, ts)[5:-5], vy[5:-5], atol=2e-3)
        assert vy.max() == pytest.approx(M.SPEED[lane], abs=1e-9)
        assert np.max(np.abs(np.gradient(vy, ts)[5:-5])) < 0.5  # m/s^2; friction carries about 10
        assert ys[0] == pytest.approx(M.START_Y[k]) and ys[-1] == pytest.approx(M.STOP_Y[k])
    assert M.SPEED["near"] == pytest.approx(5 * M.SPEED["far"])


def test_parts_start_hidden_and_end_hidden_under_the_covers_on_the_table():
    for lane, k in M.PARTS:
        assert M.START_Y[k] + M.CARRIER[1] / 2 < M.FEED_Y and M.START_Y[k] - M.CARRIER[1] / 2 > M.FEED_BACK_Y
        assert M.STOP_Y[k] - M.CARRIER[1] / 2 > M.END_Y and M.STOP_Y[k] + M.CARRIER[1] / 2 < M.RUNOUT_BACK_Y
    for spec in M.OBJECTS:
        if spec.name.startswith(("cover", "rail")):
            assert abs(spec.pos[1]) + spec.size[1] / 2 <= station.TABLE_SIZE[1] / 2 + 1e-9, spec.name
    assert M.CUBE_TOP + 0.015 < M.ROOF_LOW  # cubes pass under the roofs


def test_carriers_on_a_lane_never_close_up():
    for lane in M.LANE_X:
        for t in np.arange(0.0, 60.0, 0.05):
            gap = M.carrier_path(lane, 1)(t)[0][1] - M.carrier_path(lane, 2)(t)[0][1] - M.CARRIER[1]
            assert gap > 0.02, (lane, t)


def test_the_slow_part_comes_out_first_and_ahead_but_the_fast_part_leaves_reach_much_sooner():
    for k in (1, 2):
        slow_out, fast_out = M.passes("far", k, M.FEED_Y - M.CUBE / 2), M.passes("near", k, M.FEED_Y - M.CUBE / 2)
        assert fast_out - slow_out > 1.5
        assert M.part_y("far", k, fast_out) > M.part_y("near", k, fast_out) + 0.02  # nearer the end line, too
        assert M.passes("far", k, M.END_Y) - M.passes("near", k, M.END_Y) > 9.0


# -- the order matters -----------------------------------------------------------------------------------------------


def test_taking_parts_as_they_come_out_loses_both_fast_parts():
    """Even a robot faster than the demo (4.5 s from closing on one part to closing on the next, the demo needs 5.2 s)
    that closes on each slow part at the earliest moment any gripper could, is still busy with it when the fast part
    behind it gets past the last point a gripper could reach."""
    for k in (1, 2):
        busy_until = M.passes("far", k, REACH_Y[0]) + 4.5
        assert M.passes("near", k, REACH_Y[1]) < busy_until - 0.8
    # and with the demo's own pace, every fast part would be past the end line when the arm is free for it
    for (lane, k), close in M.schedule(M.PARTS):
        if lane == "near":
            assert M.part_y(lane, k, close) > M.END_Y


def test_the_deadline_order_fits_with_room_to_spare():
    for (lane, k), close in M.schedule():
        assert close == pytest.approx(max(close, M.passes(lane, k, M.CATCH_Y[0])))
        y_closed = M.part_y(lane, k, close + M.CLOSE_S)
        assert M.CATCH_Y[0] - 1e-9 <= M.part_y(lane, k, close) and y_closed <= M.CATCH_Y[1]
        slack = M.passes(lane, k, M.CATCH_Y[1]) - (close + M.CLOSE_S)
        assert slack > (0.8 if lane == "near" else 4.0), (lane, k, slack)


def test_the_right_arm_cannot_reach_the_window():
    for x in M.LANE_X.values():
        for y in np.arange(M.FEED_Y, M.END_Y + 1e-9, 0.02):
            goal = ik.matrix_to_quat_xyzw(grip_rotation(M.YAW))
            _, pe, _ = ik.solve(np.array([x, y, M.GRASP_Z]), goal, np.array(READY), base_pos=station.BASES["right"],
                                iters=200)
            assert pe > 0.01, (x, y)


# -- scoring -------------------------------------------------------------------------------------------------------


def in_bin_state(i):
    x, y = M.DROP_XY[M.PARTS[i]]
    return state((x, y, M.BIN_FLOOR_TOP + M.CUBE / 2))


def objects_with(n_in_bin, t=40.0):
    objs = {}
    for i, (lane, k) in enumerate(M.PARTS):
        objs[f"part_{M.name(lane, k)}"] = in_bin_state(i) if i < n_in_bin else riding(lane, k, t)
    return objs


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_each_part_in_the_bin_is_worth_a_quarter(n):
    M.reset_state()
    objs = objects_with(n)
    assert M.score(objs) == pytest.approx(0.25 * n)
    assert M.success(objs) == (n == 4)


def test_a_part_must_lie_still_in_the_bin_below_its_rim():
    M.reset_state()
    x, y = M.BIN_XY
    assert M.in_bin(state((x, y, M.BIN_FLOOR_TOP + M.CUBE / 2 + M.CUBE)))  # on another part
    assert not M.in_bin(state((x, y, M.BIN_FLOOR_TOP + M.CUBE / 2), vel=(0, 0, -0.2)))  # still falling
    assert not M.in_bin(state((x, y, M.BIN_RIM + 0.03)))  # held over the bin
    assert not M.in_bin(state((x + 0.1, y, M.TOP + M.CUBE / 2)))  # beside it


def step_rule(part_pos, t0, t1, tcps=()):
    """Run the rule for part ('near', 1) at ``part_pos`` from t0 to t1, the other parts riding."""
    for t in np.arange(t0, t1, DT):
        parts = {p: riding(*p, t) for p in M.PARTS}
        parts["near", 1] = state(part_pos)
        carriers = {p: carrier_state(*p, t) for p in M.PARTS}
        M.update(parts, carriers, list(tcps), float(t))


def test_parts_riding_held_or_over_the_bin_are_never_lost():
    M.reset_state()
    for t in np.arange(0.0, 30.0, DT):
        M.update({p: riding(*p, t) for p in M.PARTS}, {p: carrier_state(*p, t) for p in M.PARTS}, [], float(t))
    held = np.array([0.3, 0.3, M.TOP + 0.15])
    step_rule(held + (0, 0, 0.004), 30.0, 35.0, tcps=[held])
    step_rule((*M.BIN_XY, M.BIN_RIM + 0.05), 35.0, 40.0)  # falling into the bin (here, slowly)
    assert not M._STATE["lost"]


def test_a_part_put_down_anywhere_but_the_bin_is_lost_for_good():
    on_table = (0.20, 0.42, M.TOP + M.CUBE / 2)
    on_roof = (M.LANE_X["near"], M.END_Y + 0.08, M.ROOF_TOP + M.CUBE / 2)
    for where in (on_table, on_roof):
        M.reset_state()
        step_rule(where, 10.0, 10.3)  # a moment: picked up again in time
        assert not M.lost("near", 1)
        step_rule(where, 11.0, 11.6)
        assert M.lost("near", 1)
        objs = objects_with(4)
        assert M.score(objs) == pytest.approx(0.75) and not M.success(objs)  # stays lost, even in the bin
    M.reset_state()  # a new trial starts clean
    assert M.score(objects_with(4)) == 1.0


def test_a_part_moved_onto_the_other_lane_is_lost():
    M.reset_state()
    for t in np.arange(8.0, 9.0, DT):
        parts = {p: riding(*p, t) for p in M.PARTS}
        far = riding("far", 1, t)
        parts["near", 1] = state(far.pos + (0, 0, M.CUBE))  # stacked on the far lane's part
        M.update(parts, {p: carrier_state(*p, t) for p in M.PARTS}, [], float(t))
    assert M.lost("near", 1) and not M.lost("far", 1)


# -- the demo ------------------------------------------------------------------------------------------------------


def test_the_demo_is_reachable_and_ends_well_within_the_run_limit():
    plan, arm = demo_arm()
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration + 1.0 < 0.6 * RUN_LIMIT_S


def test_the_demo_waits_open_at_grasp_height_and_closes_moving_with_each_part():
    plan, arm = demo_arm()
    s = spans(plan)
    for lane, k in M.ORDER:
        n = M.name(lane, k)
        wait0, close0 = s[f"wait for {n}"][0], s[f"close on {n}"][0]
        assert close0 - wait0 >= M.SETTLE_S - 1e-9
        for t in np.arange(close0, s[f"close on {n}"][1] + 1e-9, 0.01):
            tcp = arm.target(t)[0]
            part = M.carrier_path(lane, k)(t)[0]
            assert abs(tcp[1] - part[1]) < 0.006 and abs(tcp[0] - part[0]) < 1e-9, (n, t)
            assert tcp[2] == pytest.approx(M.GRASP_Z)
        # while the arm is down in front of the part, the part is not yet past the TCP
        t = s[f"down in front of {n}"][1]
        assert M.part_y(lane, k, t) <= arm.target(t)[0][1] + 1e-6


def test_the_open_pads_clear_the_bin_the_other_lane_and_the_covers():
    pads_near_lane_out = M.LANE_X["near"] - PAD_HALF_GAP - PAD_T
    assert pads_near_lane_out - (M.BIN_XY[0] + M.BIN_IN / 2 + M.WALL) > 0.015
    pads_far_lane_in = M.LANE_X["far"] - PAD_HALF_GAP - PAD_T
    assert pads_far_lane_in - (M.LANE_X["near"] + M.CARRIER[0] / 2) > 0.015
    assert PAD_HALF_GAP - M.CUBE / 2 > 0.015  # a cube rides in between the open pads
    plan, arm = demo_arm()
    for t in np.arange(0.0, arm.duration, 0.02):
        tcp = arm.target(t)[0]
        in_cover_x = M.COVER_X_OUT[0] - 0.03 < tcp[0] < M.COVER_X_OUT[1] + 0.03
        if in_cover_x and tcp[2] - PAD_TIP_PAST_TCP < M.ROOF_TOP + 0.03:
            assert M.FEED_Y + 0.06 < tcp[1] < M.END_Y - 0.06, t


def test_held_cubes_pass_well_above_the_lanes_and_drop_over_the_bin():
    plan, arm = demo_arm()
    s = spans(plan)
    for lane, k in M.ORDER:
        n = M.name(lane, k)
        for t in np.arange(*s[f"carry {n}"], 0.02):
            tcp = arm.target(t)[0]
            cube_low = tcp[2] + 0.004 - M.CUBE / 2
            over_lane = any(abs(tcp[0] - x) < M.CARRIER[0] / 2 + M.CUBE / 2 for x in M.LANE_X.values())
            if over_lane and t > s[f"carry {n}"][0] + 0.05:
                assert cube_low > M.CUBE_TOP + 0.03, (n, t)
            over_wall = abs(tcp[0] - M.BIN_XY[0]) > M.BIN_IN / 2 - M.CUBE / 2
            if abs(tcp[0] - M.BIN_XY[0]) < M.BIN_IN / 2 + M.WALL + M.CUBE / 2 and over_wall:
                assert cube_low > M.BIN_RIM + 0.01, (n, t)
        x, y, z = plan[[w.label for w in plan].index(f"drop {n}")].pos
        drop = state((x, y, z + 0.004))
        assert M.over_bin(drop) and abs(x - M.BIN_XY[0]) + M.CUBE / 2 < M.BIN_IN / 2
        assert abs(y - M.BIN_XY[1]) + M.CUBE / 2 < M.BIN_IN / 2
        assert drop.low[2] > M.BIN_RIM + 0.01


def test_the_demo_takes_every_part_in_the_deadline_order():
    order = [p for p, _ in M.schedule()]
    assert order == list(M.ORDER)
    assert [p[0] for p in order] == ["near", "far", "near", "far"]
    assert isinstance(M.demo()["left"][0], Waypoint)
