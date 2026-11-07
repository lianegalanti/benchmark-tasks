"""Offline checks (no Isaac Sim): the tote's path, the scoring, and that the demo is reachable and lets go over the tote.

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
TOTE_PARTS = ("tote_floor", "tote_wall_near", "tote_wall_far", "tote_wall_right", "tote_wall_left")


def state(pos, size=M.PART, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.asarray(spin, float),
                       pos - half, pos + half)


def tote_at(t):
    pos, vel = M.PATH(t)
    return state(pos, size=(M.TOTE_IN, M.TOTE_IN, M.FLOOR), vel=vel)


def objects(block, t):
    return {"block": block, "tote_floor": tote_at(t)}


def test_it_loads_with_the_tote_driven_and_only_the_block_free():
    assert TASK.free_objects == ("block",)
    assert set(TOTE_PARTS) <= set(TASK.names) and TASK.score_over == "best"


def test_the_tote_runs_back_and_forth_without_pausing():
    left, right = M.LEFT_Y, M.RIGHT_Y
    for t, y in ((0.0, right), (M.LEG_S, left), (M.CYCLE_S, right), (M.CYCLE_S + M.LEG_S, left)):
        assert M.PATH(t)[0][1] == pytest.approx(y, abs=1e-9), t
    assert M.PATH(M.LEG_S / 2)[0][1] == pytest.approx((left + right) / 2, abs=1e-9)
    ts = np.arange(0.0, 3 * M.CYCLE_S, 1 / 240)
    ps = np.array([M.PATH(t)[0] for t in ts])
    vs = np.array([M.PATH(t)[1] for t in ts])
    assert np.allclose(ps[:, [0, 2]], ps[0, [0, 2]])  # only y changes
    assert ps[:, 1].min() >= right - 1e-9 and ps[:, 1].max() <= left + 1e-9
    assert np.allclose(np.gradient(ps[:, 1], ts)[5:-5], vs[5:-5, 1], atol=2e-3)  # the reported velocity is the motion's
    speed = np.abs(vs[:, 1])
    assert 0.05 <= speed.max() <= 0.15  # 10 cm/s cruise
    # no pause: under 1 cm/s for about 0.3 s per turn (6 turns in 3 cycles)
    assert np.mean(speed < 0.01) * 3 * M.CYCLE_S / 6 < 0.35
    accel = np.abs(np.gradient(vs[:, 1], ts)[5:-5])
    assert accel.max() < 0.3  # m/s^2, well under 1 and far below what the block's friction can carry
    assert np.max(np.abs(np.diff(accel))) < 0.01  # smooth: no jumps in acceleration, even at the turns


def test_the_tote_parts_move_as_one_without_touching():
    specs = {o.name: o for o in TASK.objects}
    for t in (0.0, 1.3, M.LEG_S, 8.7):
        floor = M.PATH(t)[0]
        for name in TOTE_PARTS:
            pos, vel = specs[name].path(t)
            assert np.allclose(vel, M.PATH(t)[1])
            assert np.allclose(pos - floor, np.subtract(specs[name].pos, specs["tote_floor"].pos))
    boxes = [(np.asarray(specs[n].pos) - np.asarray(specs[n].size) / 2, np.asarray(specs[n].pos) + np.asarray(specs[n].size) / 2)
             for n in TOTE_PARTS]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (lo1, hi1), (lo2, hi2) = boxes[i], boxes[j]
            assert np.any((lo1 >= hi2 - 1e-9) | (lo2 >= hi1 - 1e-9)), (TOTE_PARTS[i], TOTE_PARTS[j])  # disjoint
    assert specs["tote_floor"].pos[2] - M.FLOOR / 2 == pytest.approx(M.TOP + 0.002)  # 2 mm off the table


def test_the_tote_never_reaches_the_block_or_the_bases():
    tote_near_x = M.LINE_X - M.TOTE_IN / 2 - M.GAP - M.WALL
    block_far_x = M.BLOCK_XY[0] + M.PART[0] / 2
    assert tote_near_x - block_far_x > 0.03
    assert tote_near_x > 0.2


def test_success_needs_the_block_riding_on_the_tote_floor():
    t = M.LEG_S + 2.0  # mid-cruise, heading right
    x, y, _ = M.PATH(t)[0]
    v = M.PATH(t)[1]
    z = M.FLOOR_TOP + M.PART[2] / 2
    assert np.linalg.norm(v) > 0.09
    assert M.success(objects(state((x + 0.03, y - 0.03, z), vel=v), t))  # riding in the moving tote
    assert not M.success(objects(state((x, y, z)), t))  # in it but still: sliding against its floor
    assert not M.success(objects(state((x, y, z), vel=v + (0, 0, -0.1)), t))  # still falling
    assert not M.success(objects(state((x, y, z), vel=v, spin=(0, 0, 2.0)), t))  # spinning
    assert not M.success(objects(state((x, y, z + 0.05), vel=v), t))  # held above the floor
    assert not M.success(objects(state((x, y + 0.1, z), vel=v), t))  # beside the tote, carried along
    # where the tote was a second ago no longer counts
    assert not M.success(objects(state((*M.PATH(t - 1.0)[0][:2], z), vel=v), t))
    t0 = M.LEG_S  # at the turn, the tote is momentarily still
    x0, y0, _ = M.PATH(t0)[0]
    assert M.success(objects(state((x0, y0, z)), t0))


def test_partial_credit():
    t = 3.0
    x, y, _ = M.PATH(t)[0]
    on_table = state(M.BLOCK.pos)
    lifted = state((*M.BLOCK_XY, M.TOP + M.LIFT_M + M.PART[2] / 2 + 0.001))
    over_tote = state((x, y, M.RIM + 0.04 + M.PART[2] / 2))
    high_over_tote = state((x, y, M.RIM + 0.06 + M.PART[2] / 2))
    beside = state((x, y + 0.1, M.RIM + 0.02 + M.PART[2] / 2))
    pushed = state((0.20, 0.0, M.TOP + M.PART[2] / 2))
    scores = [M.score(objects(s, t)) for s in (on_table, lifted, over_tote, high_over_tote, beside, pushed)]
    assert scores == [0.0, 0.25, 0.5, 0.25, 0.25, 0.0]
    riding = state((x, y, M.FLOOR_TOP + M.PART[2] / 2), vel=M.PATH(t)[1])
    assert M.score(objects(riding, t)) == 0.5 and M.success(objects(riding, t))


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        a, b = spans.get(w.label, (t, t))
        spans[w.label] = (min(a, t), t + w.duration)
        t += w.duration
    return spans


def test_the_demo_is_reachable():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 30


def test_the_demo_lets_go_over_the_inside_of_the_moving_tote():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    start, end = _spans(plan)["release"]
    margin = M.TOTE_IN / 2 - M.PART[0] / 2  # the whole held block over the floor, not only its centre
    for t in np.arange(start, end + 1e-9, 0.01):
        tcp = arm.target(t)[0]
        tote = M.PATH(t)[0]
        assert abs(tcp[0] - tote[0]) < margin and abs(tcp[1] - tote[1]) < margin, t
        assert np.linalg.norm(M.PATH(t)[1]) > 0.05, t  # the tote is moving, not at a turn
    held_low = M.RELEASE_Z - 0.002 - M.PART[2] / 2
    assert M.RIM + 0.01 < held_low < M.RIM + M.ABOVE_RIM_M  # the block clears the rim and drops in


def test_the_held_block_and_fingers_clear_the_tote_walls():
    from yam_arena.constants import PAD_TIP_PAST_TCP

    assert M.RELEASE_Z - PAD_TIP_PAST_TCP > M.RIM + 0.01  # the open pads stay above the rim
    assert M.TRAVEL_Z - 0.002 - M.PART[2] / 2 > M.RIM + 0.04  # carried well above the rim
    assert M.GRASP_Z - PAD_TIP_PAST_TCP > M.TOP + 0.002  # the pads don't scrape the table
