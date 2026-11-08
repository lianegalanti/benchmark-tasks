"""Offline checks (no Isaac Sim): the flags' timing, the pick monitor, the scoring of every outcome, the layout's
clearances, and that the demo is reachable and picks each part while its flag is up.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
CUBE = (M.PART,) * 3


def state(pos, size=CUBE, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def flags_at(t):
    return [state(path(t)[0], M.FLAG) for path in M.FLAG_PATHS]


def resting(i):
    return state(M.PARTS[i].pos)


def lifted(i, dz=M.LIFT_M + 0.005):
    return state(np.asarray(M.PARTS[i].pos) + (0, 0, dz))


def pushed_out(i):
    return state((M.BIN_X - M.BIN_IN / 2 - 0.03, M.BIN_Y[i], M.TOP + M.PART / 2))


def in_tote(k=0):
    return state((M.TOTE_XY[0], M.TOTE_XY[1] + 0.06 * (k - 1.5), M.TOTE_FLOOR_TOP + M.PART / 2))


def on_table(i):
    return state((0.22, M.BIN_Y[i], M.TOP + M.PART / 2))


def run(events, end):
    """Play the monitor from a reset. ``events`` maps a bin (0-based) to (time, part state when it leaves the bin);
    every other step the part rests in its bin. ``end`` maps a bin to its part's final state. Returns the score."""
    M.reset_state()
    for k in range(round(20 / DT)):
        t = k * DT
        parts = [events[i][1] if i in events and t >= events[i][0] else resting(i) for i in range(4)]
        M.update(parts, flags_at(t))
    final = {p.name: end.get(i, resting(i)) for i, p in enumerate(M.PARTS)}
    return M.score(final), M.success(final)


def good_time(i):
    return (M.raised_from(i) + M.raised_until(i)) / 2


def test_it_loads_with_the_flags_driven_and_only_the_parts_free():
    assert TASK.free_objects == tuple(f"part_{i}" for i in range(1, 5))
    assert all(f"flag_{i}" in TASK.names for i in range(1, 5)) and TASK.score_over == "end"


def test_each_flag_pops_up_once_for_4_s_in_the_stated_order():
    order = sorted(range(4), key=M.raised_from)
    assert [i + 1 for i in order] == [3, 1, 4, 2]
    for i in range(4):
        up_s = M.raised_until(i) - M.raised_from(i)
        assert 4.1 < up_s < 4.3  # counts as up for about 4.2 s
        ts = np.arange(0.0, 30.0, 1 / 240)
        up = np.array([M.flag_up(state(M.FLAG_PATHS[i](t)[0], M.FLAG)) for t in ts])
        rising = np.flatnonzero(np.diff(up.astype(int)) == 1)
        assert len(rising) == 1  # once
        assert ts[up][0] == pytest.approx(M.raised_from(i), abs=2 / 240)
        assert ts[up][-1] == pytest.approx(M.raised_until(i), abs=2 / 240)
    for t in np.arange(0.0, 20.0, 0.05):  # never two flags of the same arm up together
        for side in ("left", "right"):
            assert sum(M.flag_up(f) for f, s in zip(flags_at(t), M.ARM_OF) if s == side) <= 1


def test_the_flag_path_is_smooth_and_hidden_when_down():
    for path in M.FLAG_PATHS:
        ts = np.arange(0.0, 20.0, 1 / 240)
        z = np.array([path(t)[0][2] for t in ts])
        vz = np.array([path(t)[1][2] for t in ts])
        assert np.max(np.abs(np.diff(z))) < 0.08 / 0.3 * 1.6 / 240
        assert np.allclose(np.gradient(z, ts)[5:-5], vz[5:-5], atol=5e-3)
        assert path(0.0)[0][2] + M.FLAG[2] / 2 < M.SCREEN_TOP  # down: behind the screen
        assert z.max() + M.FLAG[2] / 2 > M.SCREEN_TOP + 0.07
        assert z.min() - M.FLAG[2] / 2 > M.TOP + 0.01  # never touches the table
    assert M.FLAG_X - M.FLAG[0] / 2 - (M.SCREEN_X + M.SCREEN[0] / 2) >= 0.01  # never touches the screen


def test_full_score_needs_all_four_taken_in_their_windows_and_in_the_tote():
    events = {i: (good_time(i), lifted(i)) for i in range(4)}
    end = {i: in_tote(i) for i in range(4)}
    assert run(events, end) == (1.0, True)
    assert M.outcomes() == [M.TAKEN] * 4


def test_partial_credit_per_pick():
    end = {i: in_tote(i) for i in range(4)}
    # three picked and delivered, one never taken
    assert run({i: (good_time(i), lifted(i)) for i in range(3)}, end) == (0.75, False)
    # all taken in their windows, two dropped on the table
    events = {i: (good_time(i), lifted(i)) for i in range(4)}
    assert run(events, {0: in_tote(0), 1: on_table(1), 2: in_tote(2), 3: on_table(3)}) == (0.75, False)
    # all taken in time, none delivered
    assert run(events, {i: on_table(i) for i in range(4)}) == (0.5, False)
    # one taken and delivered
    assert run({2: (good_time(2), lifted(2))}, {2: in_tote(2)}) == (0.25, False)
    # nothing touched
    assert run({}, {}) == (0.0, False)


def test_wrong_picks_cost_half_a_pick_each():
    end = {i: in_tote(i) for i in range(4)}
    early = {0: (M.raised_from(0) - 0.5, lifted(0))}  # bin 1 taken just before its flag counts as up
    late = {1: (M.raised_until(1) + 0.2, lifted(1))}  # bin 2 taken just after it drops
    good = {i: (good_time(i), lifted(i)) for i in (2, 3)}
    assert run({**early, **good}, end) == (pytest.approx((2 - 0.5) / 4), False)
    assert M.outcomes()[0] == M.WRONG
    assert run({**early, **late, **good}, end) == (pytest.approx((2 - 1) / 4), False)
    # taken during another bin's window is still wrong
    other = {0: (M.raised_from(2) + 0.5, lifted(0))}  # only bin 3's flag is up then
    assert run(other, end) == (0.0, False)
    # grabbing everything at the start: all wrong, clipped to 0
    assert run({i: (0.5, lifted(i)) for i in range(4)}, end) == (0.0, False)
    # pushed out of its bin while its flag is down is a wrong pick too
    assert run({0: (1.0, pushed_out(0)), 2: (good_time(2), lifted(2))}, end) == (0.125, False)


def test_pushing_a_part_out_while_its_flag_is_up_earns_nothing():
    assert run({2: (good_time(2), pushed_out(2))}, {2: on_table(2)}) == (0.0, False)
    assert M.outcomes()[2] == M.PUSHED_OUT


def test_a_part_jostled_inside_its_bin_has_not_left_it():
    nudged = state(np.asarray(M.PARTS[0].pos) + (0.03, -0.03, M.LIFT_M - 0.005))
    assert M.in_bin(0, nudged) and not M.in_bin(0, lifted(0)) and not M.in_bin(0, pushed_out(0))


def test_in_tote_needs_the_part_still_on_the_floor_inside_the_walls():
    assert M.in_tote(in_tote(0))
    assert not M.in_tote(state(in_tote(0).pos, vel=(0, 0, -0.2)))
    assert not M.in_tote(state(in_tote(0).pos + (0, 0, 0.05)))
    assert not M.in_tote(on_table(0))


def test_layout_clearances():
    tote_far = M.TOTE_XY[0] + M.TOTE_IN[0] / 2 + M.TOTE_WALL
    bin_near = M.BIN_X - M.BIN_IN / 2 - M.BIN_WALL
    bin_far = M.BIN_X + M.BIN_IN / 2 + M.BIN_WALL
    assert bin_near - tote_far >= 0.0249
    assert M.SCREEN_X - M.SCREEN[0] / 2 - bin_far >= 0.0249
    gaps = [abs(a - b) - M.BIN_IN - 2 * M.BIN_WALL for a, b in zip(M.BIN_Y[:-1], M.BIN_Y[1:])]
    assert min(gaps) >= 0.02
    # open fingers (94 mm apart, ~1 cm pads) stay 1.5 cm inside a bin's walls, and their tips 1.5 cm above them
    assert M.BIN_IN / 2 - (0.047 + 0.01) >= 0.015
    assert M.GRASP_Z - PAD_TIP_PAST_TCP - (M.TOP + M.BIN_H) >= 0.015 - 1e-9
    assert M.GRASP_Z < M.BIN_FLOOR_TOP + M.PART  # the TCP is still below the cube's top: a 2 cm deep grip
    # the camera stand's foot (11 cm square at x 0.016, y -0.009) is clear of the tote
    assert M.TOTE_XY[0] - M.TOTE_IN[0] / 2 - M.TOTE_WALL - (0.016 + 0.055) >= 0.005


def _spans(plan):
    t, out = 0.0, {}
    for w in plan:
        out[w.label] = (t, t + w.duration)
        t += w.duration
    return out


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable(side):
    arm = ArmPlan(M.demo()[side], station.BASES[side])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 25


def test_the_demo_lifts_each_part_out_while_its_flag_is_up_and_starts_only_after():
    plans = M.demo()
    for i, side in enumerate(M.ARM_OF):
        spans = _spans(plans[side])
        go = spans[f"over bin {i + 1}"][0]
        assert go >= M.raised_from(i) + 0.25  # reacts to the flag; nothing moves toward the bin before
        lift0, lift1 = spans[f"lift from bin {i + 1}"]
        # when the held cube's underside is 3 cm above the bin floor (cube rises with the TCP)
        arm = ArmPlan(plans[side], station.BASES[side])
        out = next(t for t in np.arange(lift0, lift1, DT)
                   if arm.target(t)[0][2] - M.GRASP_Z >= M.LIFT_M)
        assert out < M.raised_until(i) - 1.5, (i, out, M.raised_until(i))  # 1.5 s or more to spare


def test_the_demo_drops_every_part_by_the_run_limit_and_the_arms_keep_apart_at_the_tote():
    plans = M.demo()
    arms = {s: ArmPlan(plans[s], station.BASES[s]) for s in plans}
    end = max(a.duration for a in arms.values())
    assert end + 2 < 800 * DT
    for t in np.arange(0.0, end, DT):
        pl, pr = (arms[s].target(min(t, arms[s].duration - 1e-6))[0] for s in ("left", "right"))
        assert np.linalg.norm((pl - pr)[:2]) > 0.15, t


def test_each_cube_has_its_own_spot_in_the_tote_clear_of_walls_and_neighbours():
    half_len = M.TOTE_IN[1] / 2
    ys = sorted(M.SPOT_Y)
    assert len(set(ys)) == 4
    assert all(half_len - (abs(y) + M.PART / 2) >= 0.04 for y in ys)  # cube 4 cm or more from the end walls
    assert min(np.diff(ys)) - M.PART >= 0.04  # cubes 4 cm or more apart
    open_half = 0.047 + 0.01  # open fingers 94 mm apart, ~1 cm pads, closing along y
    assert min(np.diff(ys)) - M.PART / 2 - open_half >= 0.015  # open fingers clear a cube already set down
    assert all(half_len - (abs(y) + open_half) >= 0.015 for y in ys)  # ... and the end walls
    for i, side in enumerate(M.ARM_OF):  # each arm fills its own end of the tote
        assert np.sign(M.SPOT_Y[i]) == (1 if side == "left" else -1)
    # the tote stays clear of the bins and the arms' bases
    assert M.TOTE_XY[1] + half_len + M.TOTE_WALL < 0.305 - 0.07


def test_the_demo_sets_each_cube_down_gently_on_its_spot():
    plans = M.demo()
    for i, side in enumerate(M.ARM_OF):
        n = f"bin {i + 1}"
        ws = {w.label: w for w in plans[side]}
        lower, drop, up = ws[f"lower {n}"], ws[f"drop {n}"], ws[f"up from {n}"]
        assert lower.pos == drop.pos == (M.TOTE_XY[0], M.SPOT_Y[i], M.SET_Z)
        underside = M.SET_Z - 0.013 - M.PART / 2  # the TCP grips 1.3 cm above the cube's centre
        assert underside - M.TOTE_FLOOR_TOP == pytest.approx(0.01)  # let go 1 cm above the floor
        assert (M.HOVER_Z - M.SET_Z) / lower.duration < 0.12  # slow descent, under 12 cm/s on average
        assert drop.finger == M.FINGER_OPEN and up.pos[:2] == drop.pos[:2]  # opens, then rises straight up
        labels = [w.label for w in plans[side]]
        assert labels.index(f"drop {n}") < labels.index(f"up from {n}")


def test_the_debug_print_reports_outcomes_and_positions(capsys, monkeypatch):
    monkeypatch.setattr(M, "DEBUG", True)
    run({2: (good_time(2), lifted(2))}, {2: in_tote(2)})
    out = capsys.readouterr().out
    assert "[pick-debug]" in out and "bin3=taken" in out and "tote=True" in out
