"""Offline checks (no Isaac Sim): the line's motion and what it hides, the scoring, and that the demo is reachable and
takes parts 4 and 8 during their stops.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import TABLE_TOP, ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
KS = range(1, M.N_PARTS + 1)
END_S = M.FIRST_S + M.N_INDEX * M.CYCLE_S  # the line has stopped for good
CUBE3 = (M.CUBE, M.CUBE, M.CUBE)


def state(pos, size=CUBE3, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def carrier_y(k, t):
    return M.CARRIERS[k - 1].path(t)[0][1]


def scene(t=END_S, moved=None):
    """Every carrier where the line has it at ``t``, every part riding on its carrier unless ``moved`` says else."""
    out = {}
    for k in KS:
        c = M.CARRIERS[k - 1].path(t)[0]
        out[f"carrier_{k}"] = state(c, M.CARRIER)
        out[f"part_{k}"] = state((c[0], c[1], M.CUBE_Z))
    for k, s in (moved or {}).items():
        out[f"part_{k}"] = s
    return out


def tray_spot(dx=0.0, dy=0.0, vel=(0.0, 0.0, 0.0)):
    return state((M.TRAY_XY[0] + dx, M.TRAY_XY[1] + dy, M.TRAY_FLOOR_TOP + M.CUBE / 2), vel=vel)


def in_tray(*ks):
    """Parts ``ks`` lying side by side in the tray."""
    spots = np.linspace(-0.05, 0.05, max(len(ks), 1))
    return {k: tray_spot(dy=float(dy) if len(ks) > 1 else 0.0) for k, dy in zip(ks, spots)}


# -- loading ---------------------------------------------------------------------------------------------------


def test_it_loads_with_eight_identical_free_parts():
    assert TASK.free_objects == tuple(f"part_{k}" for k in KS)
    assert len({(p.size, p.mass, p.color) for p in M.PARTS}) == 1  # the count is the only thing telling them apart
    assert TASK.score_over == "end" and M.SAMPLES == (4, 8)


def test_positions_are_plain_floats():
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name
        assert all(type(v) is float for v in o.size), o.name


# -- the line --------------------------------------------------------------------------------------------------


def test_the_line_indexes_one_pitch_every_cycle_and_then_stops():
    ts = np.arange(0.0, END_S + 10.0, 1 / 240)
    for k in KS:
        ys = np.array([carrier_y(k, t) for t in ts])
        vy = np.array([M.CARRIERS[k - 1].path(t)[1][1] for t in ts])
        assert np.all(np.diff(ys) >= -1e-12)  # only ever forward
        assert np.allclose(np.gradient(ys, ts)[5:-5], vy[5:-5], atol=2e-3)  # the reported velocity is the motion's
        assert np.max(np.abs(np.gradient(vy, ts)[5:-5])) < 0.4  # m/s^2: the cube's friction can carry 7 or more
        assert ys[-1] == pytest.approx(M.WINDOW_Y + (M.N_INDEX - k) * M.PITCH)
    # carriers keep their spacing: they never touch
    for t in ts[::24]:
        ys = [carrier_y(k, t) for k in KS]
        assert np.allclose(-np.diff(ys), M.PITCH)


def test_each_part_stops_in_the_window_exactly_once_in_order():
    for k in KS:
        a, b = M.at_window(k)
        for t in np.linspace(a, b, 20):
            assert carrier_y(k, t) == pytest.approx(M.WINDOW_Y, abs=1e-9), (k, t)
        assert carrier_y(k, a - M.TRAVEL_S) == pytest.approx(M.WINDOW_Y - M.PITCH)
        assert carrier_y(k, b + M.TRAVEL_S) == pytest.approx(M.WINDOW_Y + M.PITCH)
    assert M.at_window(8)[1] + M.TRAVEL_S < 60  # the last part has passed well within the 120 s run


def _cube_hidden(y, margin=0.015):
    """The cube centred at ``y`` along the line is at least ``margin`` under one of the cover's roofs."""
    lo, hi = y - M.CUBE / 2, y + M.CUBE / 2
    return hi <= M.WINDOW_Y - M.WINDOW_HALF - margin or lo >= M.WINDOW_Y + M.WINDOW_HALF + margin


def test_only_the_part_in_its_stop_is_reachable():
    for k in KS:  # at the start and once the line has stopped, the window is empty
        assert _cube_hidden(carrier_y(k, 0.0)) and _cube_hidden(carrier_y(k, END_S))
    for n in KS:  # while part n stands in the window, every other cube is 1.5 cm or more under a roof
        t = sum(M.at_window(n)) / 2
        assert all(_cube_hidden(carrier_y(k, t)) for k in KS if k != n)
    # the cover reaches over every carrier wherever it is, from the start to the end
    for k in KS:
        for t in (0.0, END_S):
            y = carrier_y(k, t)
            assert abs(y - M.WINDOW_Y) + M.CARRIER[1] / 2 < M.HALF_LEN


def test_carriers_ride_inside_the_cover_and_the_stand_is_off_the_table():
    assert M.WALL_IN - M.CARRIER[0] / 2 >= 0.01
    assert M.ROOF_LOW - M.CUBE_TOP >= 0.015
    stand = next(o for o in TASK.objects if o.name == "stand_left")
    table_left = station.TABLE_CENTER[1] + station.TABLE_SIZE[1] / 2
    assert stand.pos[1] - stand.size[1] / 2 > table_left
    assert M.WINDOW_Y - M.HALF_LEN > -(table_left + 0.05)  # the right end overhangs the table by under 5 cm


def test_the_tray_is_clear_of_the_cover():
    tray_far = M.TRAY_XY[0] + M.TRAY_IN / 2 + M.WALL
    assert (M.LINE_X - M.COVER_OUT) - tray_far >= 0.015


# -- scoring -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moved, expected",
    [
        ({}, 0.0),  # every part rode past
        (in_tray(4), 0.5),  # one sample
        (in_tray(8), 0.5),
        (in_tray(4, 8), 1.0),  # both: success
        (in_tray(3, 4, 8), 0.75),  # one wrong part taken
        (in_tray(2, 4, 6, 8), 0.5),  # every second part
        (in_tray(3, 7), 0.0),  # off by one (counting the first as part 0): floor at 0
        (in_tray(4, 5), 0.25),
        (in_tray(*KS), 0.0),  # everything into the tray
    ],
)
def test_score_levels(moved, expected):
    objs = scene(moved=moved)
    assert M.score(objs) == pytest.approx(expected)
    assert M.success(objs) == (expected == 1.0)


def test_a_wrong_part_counts_wherever_it_left_the_line():
    on_table = state((0.2, -0.1, TABLE_TOP["station"] + M.CUBE / 2))
    in_gripper = state((0.3, 0.2, TABLE_TOP["station"] + 0.2))
    assert M.score(scene(moved={**in_tray(4, 8), 2: on_table})) == pytest.approx(0.75)
    assert M.score(scene(moved={**in_tray(4, 8), 6: in_gripper})) == pytest.approx(0.75)
    assert not M.success(scene(moved={**in_tray(4, 8), 2: on_table}))


def test_a_sample_counts_only_lying_still_in_the_tray():
    held_over = state((M.TRAY_XY[0], M.TRAY_XY[1], M.TRAY_FLOOR_TOP + 0.06))
    falling = tray_spot(vel=(0.0, 0.0, -0.2))
    on_table = state((0.2, -0.1, TABLE_TOP["station"] + M.CUBE / 2))
    for s in (held_over, falling, on_table):
        assert M.score(scene(moved={4: s})) == 0.0  # not credited, and not a wrong part either
    assert M.score(scene(moved={**in_tray(8), 4: on_table})) == pytest.approx(0.5)


def test_parts_are_judged_against_the_carriers_where_they_are_now():
    mid = M.FIRST_S + 2 * M.CYCLE_S + 0.6  # mid-index: every part riding along
    assert M.score(scene(t=mid)) == 0.0 and M.tally(scene(t=mid)) == (0, 0)


# -- the demo -----------------------------------------------------------------------------------------------------


def _spans(plan):
    t, out = 0.0, {}
    for w in plan:
        out[w.label] = (t, t + w.duration)
        t += w.duration
    return out


def test_the_demo_is_reachable():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 60  # about 800 control steps of the 2,400


def test_the_demo_takes_parts_4_and_8_in_their_stops():
    spans = _spans(M.demo()["left"])
    for k in M.SAMPLES:
        stop, leave = M.at_window(k)
        assert spans[f"descend to part {k}"][0] >= stop + 0.1
        assert spans[f"lift part {k}"][0] <= leave - 0.8  # lifting well before the line moves on
    # it never goes low over the window during any other part's stop
    low = [spans[f"descend to part {k}"][0] for k in M.SAMPLES]
    for k in KS:
        if k not in M.SAMPLES:
            a, b = M.at_window(k)
            assert all(not (a - 0.5 < t < b + 0.5) for t in low)


def test_the_demo_fingers_clear_the_cover_and_the_passing_cubes():
    assert abs(M.YAW) == pytest.approx(np.pi / 2)  # fingers close across the line, not along it
    finger_half = 0.012  # a finger pad is about 2 cm wide; allow 2.4
    assert M.WINDOW_HALF - finger_half >= 0.015  # at the window's centre the fingers are 3.8 cm from each mouth
    assert M.WAIT_Z - PAD_TIP_PAST_TCP - M.CUBE_TOP >= 0.03  # waiting, the pad tips are well over passing cubes
    assert M.GRASP_Z - PAD_TIP_PAST_TCP - M.CARRIER_TOP >= 0.01  # grasping, they stay off the carrier
    open_half = -FINGER_OPEN + 0.008  # each open finger's outer face from the TCP
    assert M.TRAY_IN / 2 - open_half >= 0.015  # opening in the tray, the fingers clear its walls
    held_bottom = M.TRAVEL_Z - (M.GRASP_Z - M.CUBE_Z) - M.CUBE / 2
    assert held_bottom - M.ROOF_TOP >= 0.03  # a carried cube passes well over the cover
