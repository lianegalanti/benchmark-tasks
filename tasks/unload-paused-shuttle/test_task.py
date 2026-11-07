"""Offline checks (no Isaac Sim): the shuttle's path, the scoring, and that the demo is reachable and on time.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20


def state(pos, size=M.PART, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def test_it_loads_with_the_shuttle_fixed_and_only_the_block_free():
    assert TASK.free_objects == ("block",)
    assert "shuttle" in TASK.names and TASK.score_over == "best"


def test_the_shuttle_waits_and_moves_on_its_10_s_cycle():
    left, right = M.LEFT_Y, M.RIGHT_Y
    for t, y in ((0.0, right), (2.4, right), (3.75, (left + right) / 2), (5.1, left), (7.4, left), (10.1, right)):
        assert M.PATH(t)[0][1] == pytest.approx(y, abs=1e-9), t
    ts = np.arange(0.0, 20.0, 1 / 240)
    ys = np.array([M.PATH(t)[0][1] for t in ts])
    assert np.max(np.abs(np.diff(ys))) < 0.3 / 2.5 * 1.6 / 240  # continuous: no step bigger than the peak speed allows
    vy = np.array([M.PATH(t)[1][1] for t in ts])
    assert np.allclose(np.gradient(ys, ts)[5:-5], vy[5:-5], atol=2e-3)  # the reported velocity is the motion's
    accel = np.max(np.abs(np.gradient(vy, ts)[5:-5]))
    assert accel < 0.5  # m/s^2, far below what the block's friction (about 7 m/s^2) can carry


def test_success_needs_the_block_still_on_the_tray_floor():
    x, y = M.TRAY_XY
    z = M.TRAY_FLOOR_TOP + M.PART[2] / 2
    assert M.success({"block": state((x + 0.03, y - 0.03, z))})
    assert not M.success({"block": state((x, y, z), vel=(0, 0, -0.1))})  # still falling
    assert not M.success({"block": state((x, y, z + 0.05))})  # held above the tray
    assert not M.success({"block": state((x + 0.1, y, M.TOP + M.PART[2] / 2))})  # beside the tray


def test_partial_credit():
    x, y = M.TRAY_XY
    on_shuttle = state(M.BLOCK.pos)
    lifted = state((M.LINE_X, M.LEFT_Y, M.PLATE_TOP + M.LIFT_M + M.PART[2] / 2 + 0.001))
    over_tray = state((x, y, M.TOP + M.WALL_H + 0.03))
    on_table = state((0.25, 0.0, M.TOP + M.PART[2] / 2))
    scores = [M.score({"block": s}) for s in (on_shuttle, lifted, over_tray, on_table)]
    assert scores == [0.0, 0.25, 0.5, 0.0]


def test_the_tray_and_the_shuttle_do_not_overlap():
    tray_far_x = M.TRAY_XY[0] + M.TRAY_IN / 2 + M.WALL
    shuttle_near_x = M.LINE_X - M.PLATE[0] / 2
    assert shuttle_near_x - tray_far_x > 0.03


def test_the_demo_is_reachable_and_grabs_while_the_shuttle_waits():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    t, spans = 0.0, {}
    for w in plan:
        spans[w.label] = (t, t + w.duration)
        t += w.duration
    stop, leave = M.AT_LEFT
    assert spans["descend"][0] >= stop and spans["lift"][0] < leave - 0.4  # 0.4 s of lifting before it moves off
    assert arm.duration < 30


def test_the_waiting_fingers_clear_the_arriving_block():
    from yam_arena.constants import PAD_TIP_PAST_TCP

    block_top = M.BLOCK.pos[2] + M.PART[2] / 2
    assert M.PRE_Z - PAD_TIP_PAST_TCP - block_top > 0.02
