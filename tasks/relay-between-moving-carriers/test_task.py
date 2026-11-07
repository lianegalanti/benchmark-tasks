"""Offline checks (no Isaac Sim): the carriers' paths, the scoring, that neither arm reaches the other's carrier, and
that the demo is reachable, on time and keeps the arms apart.

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


def state(pos, size=M.PART, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.asarray(spin, float),
                       pos - half, pos + half)


def carrier_b(t):
    pos, vel = M.PATH_B(t)
    return state(pos, M.PLATE, vel)


def in_nest(t, offset=(0.0, 0.0, 0.0), dv=(0.0, 0.0, 0.0)):
    """The block lying in B's nest at time t, moving with B (plus ``dv``)."""
    pos, vel = M.PATH_B(t)
    block = pos + np.array([0.0, 0.0, M.PLATE[2] / 2 + M.PART[2] / 2]) + offset
    return {"block": state(block, vel=vel + np.asarray(dv)), "carrier_b": carrier_b(t)}


def spans(plan):
    t, out = 0.0, {}
    for w in plan:
        out[w.label] = (t, t + w.duration)
        t += w.duration
    return out


def test_it_loads_with_the_carriers_fixed_and_only_the_block_free():
    assert TASK.free_objects == ("block",)
    assert {"carrier_a", "carrier_b", "handover_pad"} <= set(TASK.names) and TASK.score_over == "best"


@pytest.mark.parametrize("path, dwell, travel", [("PATH_A", M.A_DWELL_S, M.A_TRAVEL_S),
                                                 ("PATH_B", M.B_DWELL_S, M.B_TRAVEL_S)])
def test_each_carrier_waits_and_moves_on_its_own_cycle(path, dwell, travel):
    path = getattr(M, path)
    cycle = 2 * (dwell + travel)
    for t, x in ((0.0, M.FAR_X), (dwell - 0.1, M.FAR_X), (dwell + travel / 2, (M.FAR_X + M.NEAR_X) / 2),
                 (dwell + travel + 0.1, M.NEAR_X), (2 * dwell + travel - 0.1, M.NEAR_X), (cycle + 0.1, M.FAR_X)):
        assert path(t)[0][0] == pytest.approx(x, abs=1e-9), t
    ts = np.arange(0.0, 30.0, 1 / 240)
    ps = np.array([path(t)[0] for t in ts])
    vs = np.array([path(t)[1] for t in ts])
    assert np.ptp(ps[:, 1]) == 0 and np.ptp(ps[:, 2]) == 0  # straight along x only
    assert np.allclose(np.gradient(ps[:, 0], ts)[5:-5], vs[5:-5, 0], atol=2e-3)  # the reported velocity is the motion's
    peak = np.max(np.abs(vs[:, 0]))
    assert 0.05 <= peak <= 0.20  # m/s
    assert np.max(np.abs(np.gradient(vs[:, 0], ts)[5:-5])) < 0.5  # m/s^2, far below what friction can carry


def test_the_carriers_run_out_of_step():
    assert M.A_CYCLE_S != M.B_CYCLE_S
    ts = np.arange(0.0, 36.0, 0.05)
    xa = np.array([M.PATH_A(t)[0][0] for t in ts])
    xb = np.array([M.PATH_B(t)[0][0] for t in ts])
    assert np.max(np.abs(xa - xb)) > 0.1


def test_the_nest_lips_ride_with_carrier_b_without_touching_it():
    for t in np.arange(0.0, 30.0, 0.37):
        b = M.PATH_B(t)
        for lip in M.NEST:
            p = lip.path(t)
            assert np.allclose(p[0] - b[0], np.asarray(lip.pos) - np.asarray(M.CARRIER_B.pos), atol=1e-9)
            assert np.allclose(p[1], b[1], atol=1e-9)
    for lip in M.NEST:
        assert lip.pos[2] - lip.size[2] / 2 - M.PLATE_TOP == pytest.approx(M.LIP_GAP)
        assert abs(lip.pos[0] - M.FAR_X) + lip.size[0] / 2 <= M.PLATE[0] / 2  # on the plate
        assert abs(lip.pos[1] - M.B_Y) + lip.size[1] / 2 <= M.PLATE[1] / 2


def test_success_needs_the_block_riding_in_the_nest():
    for t in (7.0, 10.5, 16.5, 19.0):  # waiting and moving
        assert M.success(in_nest(t)), t
        assert M.success(in_nest(t, offset=(0.008, -0.008, 0.002))), t
    assert not M.success(in_nest(10.5, dv=(-0.05, 0, 0)))  # B moves on, the block doesn't (held, or sliding)
    assert not M.success(in_nest(19.0, offset=(0.0, 0.0, 0.03)))  # held above the nest
    assert not M.success(in_nest(19.0, offset=(0.02, 0.0, 0.0)))  # on a lip
    b = in_nest(19.0)
    b["block"] = state(b["block"].pos, spin=(0.0, 0.0, 1.0))
    assert not M.success(b)  # still spinning
    on_table = {"block": state((M.NEAR_X, M.B_Y, M.TOP + M.PART[2] / 2)), "carrier_b": carrier_b(30.0)}
    assert not M.success(on_table)


def test_partial_credit():
    on_a = state(M.BLOCK.pos)
    lifted_off_a = state((M.NEAR_X, M.A_Y, M.PLATE_TOP + M.LIFT_M + M.PART[2] / 2 + 0.001))
    on_pad = state((*M.PAD_XY, M.PAD_TOP + M.PART[2] / 2))
    lifted = M.TOP + M.LIFT_M + M.PART[2] / 2 + 0.001
    held_above_b = state((M.NEAR_X, M.B_Y, lifted))
    held_mid_left = state((0.3, 0.25, lifted))  # on the left arm's side, but still within the right arm's reach
    on_table_left = state((0.3, 0.3, M.TOP + M.PART[2] / 2))
    in_b = in_nest(19.0)["block"]
    cases = (on_a, lifted_off_a, on_pad, held_mid_left, held_above_b, on_table_left, in_b)
    scores = [M.score({"block": s}) for s in cases]
    assert scores == [0.0, 0.25, 0.0, 0.0, 0.5, 0.0, 0.5]
    assert M.distance(in_nest(10.5)) == pytest.approx(0.0, abs=1e-9)


def _max_reach_across(side: str, n: int = 60000) -> float:
    """The largest planar distance from the arm's base its TCP reaches 1 to 15 cm above the table, any orientation."""
    rng = np.random.default_rng(0)
    base = np.asarray(station.BASES[side])
    q = rng.uniform(ik.LOWER, ik.UPPER, size=(n, 6))
    p = np.array([ik.tcp_pose(qi, base)[0] for qi in q])
    low = (p[:, 2] > M.TOP + 0.01) & (p[:, 2] < M.TOP + 0.15)
    return float(np.max(np.linalg.norm(p[low, :2] - base[:2], axis=1)))


def test_neither_arm_reaches_the_other_arms_carrier():
    for side, line_y in (("right", M.B_Y), ("left", M.A_Y)):
        base = np.asarray(station.BASES[side])
        nearest = np.hypot(M.NEAR_X, line_y - base[1]) - M.NEST_TOL  # the closest a block on that carrier comes
        assert nearest - _max_reach_across(side) > 0.03, side


def test_half_credit_is_out_of_the_right_arms_reach():
    """0.5 needs the block's centre where the right arm's TCP can't take it: only the left arm earns it."""
    base = np.asarray(station.BASES["right"])
    assert tuple(base[:2]) == M.RIGHT_BASE_XY
    assert M.RIGHT_OUT_OF_REACH - _max_reach_across("right") >= 0.03
    above_b = np.hypot(M.NEAR_X - base[0], M.B_Y - base[1])  # where the demo's left arm waits with the block
    assert above_b - M.RIGHT_OUT_OF_REACH > 0.01


def test_the_handover_pad_and_the_carriers_do_not_overlap():
    pad_edge = M.PAD_XY[1] + M.PAD[1] / 2
    rail_edge = M.B_Y - 0.04 - 0.0075
    assert rail_edge - pad_edge > 0.3
    assert M.NEAR_X - M.PLATE[0] / 2 > 0.1  # the near stop leaves room in front of the arm bases


def _played(side):
    plan = M.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    n = round(max(ArmPlan(p, station.BASES[s]).duration for s, p in M.demo().items()) / DT) + 5
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(n)])
    tcp = np.array([ik.tcp_pose(a[:6], station.BASES[side])[0] for a in actions])
    return plan, arm, actions, tcp


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable(side):
    _, arm, actions, _ = _played(side)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 30


def test_the_right_arm_grabs_while_a_waits_at_its_near_stop():
    s = spans(M.demo()["right"])
    stop, leave = M.A_AT_NEAR
    assert s["descend"][0] >= stop and s["lift"][0] < leave - 0.4  # 0.4 s of lifting before A moves off


def test_the_left_arm_releases_while_b_waits_at_its_near_stop():
    s = spans(M.demo()["left"])
    stop, leave = (t + M.B_PLACE_CYCLE * M.B_CYCLE_S for t in M.B_AT_NEAR)
    assert s["down to just above the lips"][0] >= stop
    assert s["release"][1] < leave - 0.5 and s["up"][1] <= leave  # clear before B moves off


def test_the_left_arm_takes_the_block_only_after_the_right_arm_let_go():
    right, left = spans(M.demo()["right"]), spans(M.demo()["left"])
    assert left["above the pad"][0] >= right["up"][1]


def test_the_arms_stay_apart():
    _, _, _, left = _played("left")
    _, _, _, right = _played("right")
    gap = np.linalg.norm(left - right, axis=1)
    assert gap.min() > 0.08
    assert np.all(right[:, 1] < 0.06) and np.all(left[:, 1] > -0.06)  # each stays on its own side of the pad


def test_the_waiting_fingers_clear_the_arriving_block_and_the_lips():
    from yam_arena.constants import PAD_TIP_PAST_TCP

    block_top = M.BLOCK.pos[2] + M.PART[2] / 2
    assert M.PRE_Z - PAD_TIP_PAST_TCP - block_top > 0.02
    lip_top = M.PLATE_TOP + M.LIP_GAP + M.LIP_H
    assert M.NEST_SET_Z - PAD_TIP_PAST_TCP - lip_top > 0.005
