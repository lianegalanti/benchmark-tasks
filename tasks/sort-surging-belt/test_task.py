"""Offline checks (no Isaac Sim): the belt's speed schedule and the pusher, what the tunnel hides, every scoring level,
and that the demo is reachable and takes each cube in time, aiming where the pushed cube now is.

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
CUBE3 = (M.CUBE, M.CUBE, M.CUBE)
TS = np.arange(0.0, 40.0, 1 / 240)


def state(pos, size=CUBE3, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def scene(t=40.0, moved=None):
    """Every carrier where the belt has it at ``t``, every cube riding on its carrier unless ``moved`` says else."""
    out = {}
    for n in M.NAMES:
        c = M.carrier_path(n)(t)[0]
        out[f"carrier_{n}"] = state(c, M.CARRIER)
        out[n] = state((c[0], c[1], M.CUBE_Z))
    out.update(moved or {})
    return out


def in_bin(colour, dx=0.0, dy=0.0, vel=(0.0, 0.0, 0.0)):
    return state((M.BIN_X + dx, M.BIN_Y[colour] + dy, M.BIN_FLOOR_TOP + M.CUBE / 2), vel=vel)


SPOTS = ((-0.035, -0.035), (-0.035, 0.035), (0.035, -0.035), (0.035, 0.035))  # four cubes fit on a bin's floor


def sorted_as(**where):
    """Cubes lying on the floors of the named bins, side by side: sorted_as(red_1="red", blue_1="red")."""
    count = {"red": 0, "blue": 0}
    out = {}
    for n, colour in where.items():
        dx, dy = SPOTS[count[colour]]
        out[n] = in_bin(colour, dx, dy)
        count[colour] += 1
    return out


ALL_RIGHT = dict(red_1="red", red_2="red", blue_1="blue", blue_2="blue")

# -- loading -----------------------------------------------------------------------------------------------------


def test_it_loads_with_four_free_cubes_two_of_each_colour():
    assert TASK.free_objects == M.NAMES
    assert sorted(M.COLOUR.values()) == ["blue", "blue", "red", "red"]
    assert TASK.score_over == "end"


def test_positions_and_sizes_are_plain_floats():
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name
        assert all(type(v) is float for v in o.size), o.name


# -- the belt ----------------------------------------------------------------------------------------------------


def _belt():
    run = np.array([M.RUN(t) for t in TS])
    return run[:, 0], run[:, 1]


def test_the_belt_cruises_stops_surges_and_slows_on_schedule():
    for t, v in ((1.0, M.CRUISE), (8.9, M.CRUISE), (9.5, 0.0), (14.4, 0.0), (14.9, M.SURGE), (19.4, M.SURGE),
                 (19.9, M.CRUISE), (28.4, M.CRUISE), (28.9, 0.0), (60.0, 0.0)):
        assert M.RUN(t)[1] == pytest.approx(v, abs=1e-9), t
    assert M.SURGE == pytest.approx(2 * M.CRUISE)


def test_the_belt_moves_smoothly_forward_and_reports_its_speed():
    s, v = _belt()
    assert np.all(np.diff(s) >= -1e-12)
    assert np.allclose(np.gradient(s, TS)[2:-2], v[2:-2], atol=1e-3)
    accel = np.max(np.abs(np.gradient(v, TS)))
    assert accel < 0.5  # m/s^2: abrupt to the eye, far below what a cube's friction (about 8 m/s^2) can carry
    assert accel > 0.3  # but each change of speed takes only 0.3 s


def test_carriers_keep_their_spacing_and_never_touch():
    ys = [M.START_Y[n] for n in M.NAMES]
    assert all(a - b >= M.CARRIER[1] + 0.015 for a, b in zip(ys, ys[1:]))  # 2 cm or more between carriers


def test_the_carriers_start_on_the_line_and_end_inside_the_tunnel():
    for n in M.NAMES:
        y0, y1 = M.part_y(n, 0.0), M.part_y(n, 60.0)
        assert y0 - M.CARRIER[1] / 2 > M.RIGHT_END_Y
        assert y0 + M.CUBE / 2 < M.END_Y  # every cube is out in the open at the start
        assert y1 - M.CUBE / 2 >= M.END_Y + 0.05  # and 5 cm or more under the roof at the end
        assert y1 + M.CARRIER[1] / 2 < M.TUNNEL_END_Y  # without hitting the tunnel's end wall
    assert M.part_y("red_2", M.T_END) - M.CUBE / 2 > M.END_Y  # the last cube is inside before the belt stops


def test_the_tunnel_is_too_low_to_reach_into_and_clears_the_carriers():
    assert M.ROOF_LOW - M.CUBE_TOP == pytest.approx(0.02)  # a 2 cm slot: fingers can't get over a cube in there
    assert M.WALL_IN - M.CARRIER[0] / 2 >= 0.01


def test_only_the_tunnel_overhangs_the_table():
    table_left = station.TABLE_CENTER[1] + station.TABLE_SIZE[1] / 2
    post_l = next(o for o in TASK.objects if o.name == "post_left")
    assert post_l.pos[1] - post_l.size[1] / 2 > table_left
    assert M.RIGHT_END_Y > -table_left - 0.02  # the right end overhangs by 2 cm at most, with no post
    assert M.TUNNEL_END_Y - table_left <= 0.25  # the tunnel overhangs the left edge by 25 cm at most
    assert M.END_Y < table_left - 0.2  # its mouth is well on the table


# -- the pusher ----------------------------------------------------------------------------------------------------


def test_the_pusher_moves_only_while_the_belt_stands_still():
    a, b = M.STOPPED
    assert a < M.PUSH_T0 and M.PUSH_DONE < b - 1.0
    for t in TS:
        if not a <= t <= b:
            assert M.BAR_PATH(t)[0][0] == pytest.approx(M.BAR_REST_FACE + M.BAR[0] / 2), t


def test_the_pushed_cube_stands_square_in_front_of_the_bar():
    y = M.part_y(M.PUSHED, M.PUSH_T0)
    assert y == pytest.approx(M.PUSH_Y, abs=0.005)
    assert M.BAR[1] - M.CUBE >= 0.015  # the bar's face is wider than the cube by 1 cm either side, give or take 5 mm


def test_the_pusher_shoves_the_cube_4_cm_and_leaves_it_on_its_carrier():
    out = M.BAR_PATH(M.PUSH_T0 + M.PUSH_OUT_S)[0][0] - M.BAR[0] / 2
    assert out == pytest.approx(M.BAR_OUT_FACE)
    shoved_far_face = M.LINE_X + M.CUBE / 2 - M.NUDGE
    assert out == pytest.approx(shoved_far_face)
    near_face = M.LINE_X - M.CUBE / 2 - M.NUDGE
    assert near_face - (M.LINE_X - M.CARRIER[0] / 2) >= 0.01  # still 1 cm on the carrier
    # it pushes low: under the height at which a push would tip a 4 cm cube (half its width over friction ~1)
    assert M.BAR_Z + M.BAR[2] / 2 - M.CARRIER_TOP <= 0.015
    assert M.BAR_Z - M.BAR[2] / 2 - M.CARRIER_TOP >= 0.002  # and rides clear of the carrier
    # the shove misses an unchanged grasp: the open fingers' inner faces (4.7 cm from the TCP) land on the cube
    assert M.NUDGE > -FINGER_OPEN - M.CUBE / 2 + 0.01


def test_the_pusher_touches_no_other_cube_and_rests_clear_of_the_belt():
    lo, hi = M.PUSH_T0, M.PUSH_DONE
    for n in M.NAMES:
        if n != M.PUSHED:
            for t in np.linspace(lo, hi, 20):
                assert abs(M.part_y(n, t) - M.PUSH_Y) - M.BAR[1] / 2 - M.CUBE / 2 >= 0.03
    assert M.BAR_REST_FACE - (M.LINE_X + M.CARRIER[0] / 2) >= 0.015
    guide_near = M.LINE_X + 0.10
    assert guide_near - (M.LINE_X + M.CARRIER[0] / 2) >= 0.02
    assert M.GUIDE_TOP < M.BAR_Z - M.BAR[2] / 2


# -- scoring -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "where, expected",
    [
        ({}, 0.0),  # everything went into the tunnel
        (dict(red_1="red"), 0.25),
        (dict(red_1="red", blue_2="blue"), 0.5),
        (dict(red_1="red", red_2="red", blue_2="blue"), 0.75),
        (ALL_RIGHT, 1.0),
        (dict(red_1="red", red_2="red", blue_1="blue", blue_2="red"), 0.5),  # one in the wrong bin
        (dict(red_1="red", red_2="red", blue_1="red"), 0.25),
        (dict(red_1="red", blue_1="red"), 0.0),
        (dict(red_1="blue", red_2="blue", blue_1="red", blue_2="red"), 0.0),  # every one swapped: floor at 0
    ],
)
def test_score_levels(where, expected):
    objs = scene(moved=sorted_as(**where))
    assert M.score(objs) == pytest.approx(expected)
    assert M.success(objs) == (expected == 1.0)


def test_dumping_every_cube_in_one_bin_scores_nothing():
    assert M.score(scene(moved=sorted_as(red_1="red", red_2="red", blue_1="red", blue_2="red"))) == 0.0
    assert M.score(scene(moved=sorted_as(red_1="blue", red_2="blue", blue_1="blue", blue_2="blue"))) == 0.0


def test_a_cube_counts_only_still_inside_the_bin_below_its_rim():
    top = TABLE_TOP["station"]
    held_over = state((M.BIN_X, M.BIN_Y["red"], M.BIN_RIM + 0.03))
    falling = in_bin("red", vel=(0.0, 0.0, -0.2))
    beside = state((M.BIN_X, M.BIN_Y["red"] + M.BIN_IN[1] / 2 + 0.03, top + M.CUBE / 2))
    on_rim = state((M.BIN_X, M.BIN_Y["red"] + M.BIN_IN[1] / 2 + 0.005, M.BIN_RIM + M.CUBE / 2))
    for s in (held_over, falling, beside, on_rim):
        assert M.score(scene(moved={"red_1": s})) == 0.0
    # a wrong cube held over the other bin is not (yet) a wrong placement either
    assert M.score(scene(moved={**sorted_as(red_1="red"), "blue_1": held_over})) == pytest.approx(0.25)
    # a cube leaning in a corner or on the other cube still counts if its centre is below the rim
    leaning = state((M.BIN_X + 0.05, M.BIN_Y["red"] + 0.04, M.BIN_FLOOR_TOP + 0.03))
    assert M.score(scene(moved={"red_1": leaning})) == pytest.approx(0.25)


def test_cubes_riding_on_the_belt_score_nothing_at_any_time():
    for t in (0.0, 10.0, 15.0, 25.0, 40.0):
        assert M.score(scene(t=t)) == 0.0 and M.tally(scene(t=t)) == (0, 0)


def test_the_bins_are_clear_of_the_belt_and_far_from_each_other():
    bin_far = M.BIN_X + M.BIN_IN[0] / 2 + M.BIN_WALL
    assert (M.LINE_X - M.CARRIER[0] / 2) - bin_far >= 0.05 - 1e-9
    bin_near = M.BIN_X - M.BIN_IN[0] / 2 - M.BIN_WALL
    assert bin_near - 0.035 >= 0.02  # 2 cm or more in front of the arm bases' 3.5 cm front faces
    assert M.BIN_Y["blue"] - M.BIN_Y["red"] - (M.BIN_IN[1] + 2 * M.BIN_WALL) > 0.4


# -- the demo -------------------------------------------------------------------------------------------------------


def _spans(plan):
    """Each label's first start and last end."""
    t, out = 0.0, {}
    for w in plan:
        a, _ = out.get(w.label, (t, t))
        out[w.label] = (a, t + w.duration)
        t += w.duration
    return out


def _tcp_track(side):
    plan = M.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    n = round(arm.duration / DT) + 5
    acts = np.array([arm.action7((k + 1) * DT)[0] for k in range(n)])
    return arm, acts


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable(side):
    arm, acts = _tcp_track(side)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(acts[:, :6] >= ik.LOWER) and np.all(acts[:, :6] <= ik.UPPER)
    assert arm.duration < 30  # about 500 control steps


def test_the_demo_takes_every_cube_in_reach_and_in_its_own_half():
    plans = M.demo()
    for n in M.NAMES:
        side = M.ARM[n]
        spans = _spans(plans[side])
        a, _ = spans[f"follow {n} down"]
        _, b = spans[f"lift {n} off"]
        for t in (a, b):
            y = M.part_y(n, t)
            assert y + M.CUBE / 2 < M.END_Y - 0.04, (n, t)  # 4 cm or more before the tunnel's mouth
            assert (y <= -0.18) if side == "right" else (y >= 0.05), (n, t)
        assert M.ARM[n] == ("right" if M.COLOUR[n] == "red" else "left")


def test_the_demo_grasps_the_pushed_cube_after_the_shove_where_it_now_is():
    spans = _spans(M.demo()["left"])
    down, _ = spans[f"follow {M.PUSHED} down"]
    assert down >= M.PUSH_DONE + 1.0  # the bar has been back at rest for 1 s
    _, lifted = spans[f"lift {M.PUSHED} off"]
    assert lifted <= M.T_GO - 0.5  # lifted clear 0.5 s before the belt restarts
    assert M.grasp_x(M.PUSHED) == pytest.approx(M.LINE_X - M.NUDGE)
    over = next(w for w in M.demo()["left"] if w.label == f"follow {M.PUSHED} down")
    assert over.pos[0] == pytest.approx(M.LINE_X - M.NUDGE)


def test_the_demo_takes_red_2_while_the_belt_stands_still():
    spans = _spans(M.demo()["right"])
    a, _ = spans["follow red_2 down"]
    _, b = spans["lift red_2 off"]
    assert M.STOPPED[0] <= a and b <= M.STOPPED[1] - 1.0


def test_the_demo_fingers_clear_the_carrier_the_bins_and_the_pusher():
    assert abs(M.YAW) == pytest.approx(np.pi / 2)  # fingers close across the belt
    assert M.GRASP_Z - PAD_TIP_PAST_TCP - M.CARRIER_TOP >= 0.01
    assert M.PRE_Z - PAD_TIP_PAST_TCP - M.CUBE_TOP >= 0.025
    open_half = -FINGER_OPEN + 0.008  # each open finger's outer face from the TCP
    assert M.BIN_IN[0] / 2 - open_half >= 0.015 - 1e-9
    # the far finger, grasping a cube that rode in the middle, stays 2 cm or more from the bar at rest
    assert M.BAR_REST_FACE - (M.LINE_X + open_half) >= 0.02
    held_bottom = M.TRAVEL_Z - (M.GRASP_Z - M.CUBE_Z) - M.CUBE / 2
    assert held_bottom - M.BIN_RIM >= 0.03


def test_the_two_arms_stay_apart():
    _, la = _tcp_track("left")
    _, ra = _tcp_track("right")
    left_tcps = [ik.tcp_pose(q, station.BASES["left"])[0] for q in la[::4, :6]]
    right_tcps = [ik.tcp_pose(q, station.BASES["right"])[0] for q in ra[::4, :6]]
    n = min(len(left_tcps), len(right_tcps))
    gaps = [np.linalg.norm(left_tcps[k] - right_tcps[k]) for k in range(n)]
    assert min(gaps) > 0.2
    assert min(p[1] for p in left_tcps) > max(p[1] for p in right_tcps)


def test_the_demo_ends_well_before_the_run_limit():
    ends = [sum(w.duration for w in plan) for plan in M.demo().values()]
    assert max(ends) < 30.0  # success holds 1 s later: about 500 of the 1,200 steps
