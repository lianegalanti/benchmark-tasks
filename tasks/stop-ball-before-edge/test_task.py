"""Offline checks (no Isaac Sim): the layout, the ball's untouched roll, the scoring, and that the demo is reachable
and has its barrier down in time.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

import motionkit
from yam_arena import ik, station
from yam_arena.constants import FINGER_CLOSED, READY
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20


def ball(pos, vel=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - M.R,
                       pos + M.R)


def run(states, t0=0.0):
    """Feed (state, ...) one control step apart from a fresh episode; return the scores."""
    M.MONITOR.reset()
    return [M.MONITOR.update(s, t0 + k * DT) for k, s in enumerate(states)]


def on_table(y, vel=(0.0, 0.0, 0.0)):
    return ball((M.LINE_X, y, M.TOP + M.R), vel)


def test_it_loads_with_only_the_ball_free_and_scores_at_the_end():
    assert TASK.free_objects == ("ball",)
    assert {"ramp", "gate", "cup_floor"} <= set(TASK.names) and TASK.score_over == "end"


def test_the_left_edge_is_the_tables():
    assert M.EDGE_Y == pytest.approx(station.TABLE_CENTER[1] + station.TABLE_SIZE[1] / 2) == pytest.approx(0.65)


def test_the_ramp_meets_the_table_at_its_foot_and_stays_inside_it_underneath():
    assert M.on_ramp(0.0)[2] == pytest.approx(M.TOP)
    assert M.on_ramp(M.RAMP_LEN)[2] - M.RAMP_T * math.cos(M.THETA) < M.TOP  # its underside is inside the table
    table_bottom = M.TOP - station.TABLE_SIZE[2]
    assert M.on_ramp(-0.01)[2] - M.RAMP_T - 0.002 > table_bottom
    # the ball starts on the ramp, resting on it (1 mm above), with the gate 1 mm downhill
    assert M.START_S < M.RAMP_LEN - M.R
    assert np.dot(M.START - M.on_ramp(M.START_S), M.NORMAL) == pytest.approx(M.R + 0.001)
    gate_face = M.GATE_POS[1] - M.GATE[1] / 2
    assert gate_face - (M.START[1] + M.R) == pytest.approx(0.001)


def test_the_gate_holds_then_lifts_clear_of_the_ball():
    assert np.allclose(M.GATE_BOX.path(0.0)[0], M.GATE_POS) and np.allclose(M.GATE_BOX.path(1.0)[0], M.GATE_POS)
    up = M.GATE_BOX.path(M.GATE_AT_S + M.GATE_OVER_S)[0]
    gate_bottom = up[2] - M.GATE[2] / 2
    assert gate_bottom > M.START[2] + M.R + 0.005  # 5 mm over the top of the ball
    ts = np.arange(0.0, 3.0, 1 / 240)
    zs = np.array([M.GATE_BOX.path(t)[0][2] for t in ts])
    vz = np.array([M.GATE_BOX.path(t)[1][2] for t in ts])
    assert np.allclose(np.gradient(zs, ts)[5:-5], vz[5:-5], atol=6e-3)  # central differences lag the start and stop of the 3.3 m/s^2 lift


def test_untouched_the_ball_rolls_off_the_left_edge_about_3_s_after_the_gate_lifts():
    assert 0.3 < M.SPEED < 0.5  # m/s on the table
    for damping in (0.0, 0.05):  # PhysX's default angular damping slows it by under 2 %
        t_edge = M.roll_time(M.EDGE_Y, damping)
        assert 2.0 < t_edge - M.GATE_AT_S < 4.0
    # it is never still once it has moved 2 cm: its speed there is already 6x the stillness threshold
    v_2cm = math.sqrt(2 * M.ACCEL * M.MOVED_M)
    assert v_2cm > 3 * M.STILL_SPEED


def test_the_ball_path_clears_the_cup_the_rails_and_the_idle_arms():
    cup_far_x = M.CUP_XY[0] + M.CUP_IN / 2 + M.WALL
    assert M.LINE_X - M.R - cup_far_x > 0.04
    assert M.RAILS[1].pos[0] - M.RAILS[1].size[0] / 2 - (M.LINE_X + M.R) == pytest.approx(M.RAIL_GAP)
    for side in ("left", "right"):
        tcp, _ = ik.tcp_pose(np.array(READY, float), np.asarray(station.BASES[side]))
        assert tcp[2] - (M.TOP + 2 * M.R) > 0.2  # the ready grippers are far above the rolling ball


def test_success_needs_the_ball_still_on_the_cup_floor():
    x, y = M.CUP_XY
    z = M.CUP_FLOOR_TOP + M.R
    assert M.success({"ball": ball((x + 0.01, y - 0.01, z))})
    assert not M.success({"ball": ball((x, y, z), vel=(0, 0, -0.1))})  # still falling
    assert not M.success({"ball": ball((x, y, z + 0.05))})  # held above the cup
    assert not M.success({"ball": on_table(0.34)})  # stopped on the table


def test_partial_credit_is_kept_unless_the_ball_falls():
    x, y = M.CUP_XY
    rolling = [ball(M.START)] * 5 + [on_table(yy, (0, 0.42, 0)) for yy in np.arange(-0.1, 0.6, 0.02)]
    assert max(run(rolling)) == 0.0  # untouched: nothing
    fallen = rolling + [ball((M.LINE_X, M.EDGE_Y + 0.03, M.TOP - 0.1), (0, 0.4, -1.0))]
    assert run(fallen)[-1] == 0.0
    waiting = [ball(M.START)] * 30  # still behind the gate: not "stopped"
    assert max(run(waiting)) == 0.0
    blip = [on_table(0.3)] * 5 + [on_table(0.3, (0, 0.1, 0))]  # still for only 0.25 s
    assert max(run(blip)) == 0.0
    stopped = [on_table(0.34)] * 12  # still for 0.55 s
    assert run(stopped)[-1] == 0.25
    lifted = stopped + [ball((M.LINE_X, 0.34, M.TOP + M.R + M.LIFT_M + 0.001))]
    assert run(lifted)[-1] == 0.5
    in_cup = lifted + [ball((x, y, M.CUP_RIM - 0.001))]
    assert run(in_cup)[-1] == 0.75
    dropped_on_table = in_cup + [on_table(0.1)] * 3  # bounced out, but still on the table: keeps 0.75
    assert run(dropped_on_table)[-1] == 0.75
    then_fell = dropped_on_table + [ball((M.LINE_X, M.EDGE_Y + 0.03, M.TOP - 0.2))] + [on_table(0.3)] * 20
    assert run(then_fell)[-1] == 0.0  # once fallen, 0 for the rest of the trial


def test_score_reads_the_episode_clock_and_resets_with_the_episode():
    motionkit._CLOCK.update(t=0.0, running=False)
    for fn in motionkit._RESET:
        fn()
    for k in range(12):
        motionkit._CLOCK["t"] = 2.0 + k * DT
        s = M.score({"ball": on_table(0.34)})
    assert s == 0.25
    for fn in motionkit._RESET:  # what the embodiment's reset hook does
        fn()
    assert M.MONITOR.best == 0.0 and not M.MONITOR.fallen


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        spans[w.label] = (t, t + w.duration)
        t += w.duration
    return spans


def test_the_demo_is_reachable():
    arm = ArmPlan(M.demo()["left"], station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 30


def test_the_barrier_is_down_half_a_second_before_the_ball_arrives_and_stays_while_it_stops():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    spans = _spans(plan)
    down = spans["barrier down"][1]
    assert down <= M.ARRIVE_S - 0.5
    barrier = np.array([M.LINE_X, M.STOP_Y, M.BARRIER_Z])
    for t in np.arange(down, M.ARRIVE_S + 1.0, DT):
        pos, _, finger, _ = arm.target(t)
        assert np.linalg.norm(pos - barrier) < 1e-6 and finger == M.FORK, t
    # the arm comes down onto the path well before the ball is anywhere near it
    assert M.roll_y(down) < M.STOP_Y - 0.4
    # it stands before the edge, with room to spare if it misses
    assert M.EDGE_Y - M.STOP_Y > 0.2


def test_the_fingers_meet_the_ball_and_clear_it():
    from yam_arena.constants import PAD_TIP_PAST_TCP

    tips = M.BARRIER_Z - PAD_TIP_PAST_TCP
    assert M.TOP < tips < M.TOP + M.R / 2  # the pad tips are below the ball's centre: it can't roll under them
    assert M.CLEAR_Z - PAD_TIP_PAST_TCP - (M.TOP + 2 * M.R) > 0.01  # lifted clear over the ball's top
    assert M.DROP_Z - PAD_TIP_PAST_TCP - M.CUP_RIM > 0.01  # releases with the pads above the rim


def test_poses_are_plain_floats():
    """Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects at reset."""
    for spec in TASK.module.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name


def test_the_fork_is_narrower_than_the_ball_and_opens_wide_of_it():
    assert 0.6 * 2 * M.R < M.FORK_GAP < 2 * M.R - 0.004
    assert -M.FINGER_OPEN - M.R > 0.015  # open, each pad clears the ball by 1.5 cm as the hand slides over it
