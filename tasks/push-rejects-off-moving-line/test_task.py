"""Offline checks (no Isaac Sim): the belt, the line, the reject table and the tunnel, every scoring level (including
the strategies that must score nothing), and that the demo is reachable, clears the cartons and carriers, and pushes
each red carton onto the table in time.

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
TS = np.arange(0.0, 40.0, 1 / 240)
TOP = TABLE_TOP["station"]


def state(pos, size=None, vel=(0.0, 0.0, 0.0)):
    size = M.CARTON if size is None else size
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def scene(t=40.0, moved=None):
    """Every carrier where the belt has it at ``t``, every carton riding on its carrier unless ``moved`` says else."""
    out = {}
    for n in M.NAMES:
        c = M.carrier_path(n)(t)[0]
        out[f"carrier_{n}"] = state(c, M.CARRIER)
        out[n] = state((c[0], c[1], M.CARTON_Z))
    out.update(moved or {})
    return out


def shelved(y, dx=0.0, vel=(0.0, 0.0, 0.0)):
    """A carton lying on the reject table at ``y``, its centre at the demo's spot (plus ``dx``)."""
    return state((M.SHELF_GOAL_X + dx, y, M.SHELF_TOP + M.CARTON[2] / 2), vel=vel)


ON_TABLE_TOP = lambda y: state((0.12, y, TOP + M.CARTON[2] / 2))  # noqa: E731  pushed off toward the robot
ON_FLOOR = lambda y: state((0.6, y, station.FLOOR_Z + M.CARTON[2] / 2))  # noqa: E731
TIPPED = lambda n, t=40.0: state(  # noqa: E731  lying on its 10 x 5 cm side, still on its carrier
    (M.LINE_X, M.carrier_path(n)(t)[0][1], M.CARRIER_TOP + M.CARTON[0] / 2), size=(0.05, 0.10, 0.12))

# -- loading -----------------------------------------------------------------------------------------------------


def test_it_loads_with_four_free_cartons_two_of_them_rejects():
    assert TASK.free_objects == M.NAMES
    assert M.REJECTS == ("reject_1", "reject_2") and M.GOODS == ("good_1", "good_2")
    assert TASK.score_over == "end"


def test_positions_and_sizes_are_plain_floats():
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name
        assert all(type(v) is float for v in o.size), o.name


def test_the_cartons_are_too_wide_to_grip_in_one_hand():
    assert min(M.CARTON[:2]) > -2 * FINGER_OPEN + 0.005  # the open fingers are 9.4 cm apart


# -- the belt ----------------------------------------------------------------------------------------------------


def test_the_belt_runs_at_5_cm_s_without_stopping_until_every_carrier_is_in_the_tunnel():
    for t in (0.5, 5.0, 10.0, 15.0, M.T_END - 0.01):
        assert M.RUN(t)[1] == pytest.approx(M.SPEED, abs=1e-9), t
    for t in (M.T_END + M.RAMP_S, 30.0, 60.0):
        assert M.RUN(t)[1] == 0.0
    run = np.array([M.RUN(t) for t in TS])
    assert np.all(np.diff(run[:, 0]) >= -1e-12)
    assert np.allclose(np.gradient(run[:, 0], TS)[2:-2], run[2:-2, 1], atol=1e-3)
    assert np.max(np.abs(np.gradient(run[:, 1], TS))) < 0.3  # gentle starts and stops: no carton slides


def test_carriers_and_cartons_keep_their_spacing():
    ys = [M.START_Y[n] for n in M.NAMES]
    for a, b in zip(ys, ys[1:]):
        assert a - b - M.CARRIER[1] >= 0.01 - 1e-9  # 1 cm between carriers
        assert a - b - M.CARTON[1] >= 0.03 - 1e-9  # 3 cm between cartons
    assert M.CARRIER[0] - M.CARTON[0] >= 0.02 - 1e-9 and M.CARRIER[1] - M.CARTON[1] >= 0.02 - 1e-9


def test_the_carriers_start_on_the_line_and_end_inside_the_tunnel():
    for n in M.NAMES:
        y0, y1 = M.part_y(n, 0.0), M.part_y(n, 60.0)
        assert y0 - M.CARRIER[1] / 2 >= M.RIGHT_END_Y
        assert y0 + M.CARTON[1] / 2 < M.END_Y  # every carton is out in the open at the start
        assert y1 - M.CARRIER[1] / 2 >= M.END_Y + 0.03  # and wholly under the roof at the end
        assert y1 + M.CARRIER[1] / 2 < M.TUNNEL_END_Y  # without hitting the tunnel's end wall


def test_the_tunnel_is_too_low_to_reach_into_and_clears_the_carriers():
    assert M.ROOF_LOW - M.CARTON_TOP == pytest.approx(0.02)
    assert M.WALL_IN - M.CARRIER[0] / 2 >= 0.01


def test_everything_but_the_tunnel_is_on_the_table():
    x0 = station.TABLE_CENTER[0] - station.TABLE_SIZE[0] / 2
    x1 = station.TABLE_CENTER[0] + station.TABLE_SIZE[0] / 2
    y_edge = station.TABLE_SIZE[1] / 2
    assert M.RIGHT_END_Y > -y_edge and M.SHELF_Y[0] > -y_edge
    assert x0 < M.SHELF_X[0] and M.SHELF_X[1] + M.LIP_T < x1
    assert M.END_Y < y_edge - 0.25
    assert M.TUNNEL_END_Y - y_edge <= 0.30
    post = next(o for o in TASK.objects if o.name == "post_left")
    assert post.pos[1] - post.size[1] / 2 > y_edge


# -- the reject table -------------------------------------------------------------------------------------------


def test_the_reject_table_sits_just_below_and_beside_the_carriers():
    assert M.SHELF_X[0] - (M.LINE_X + M.CARRIER[0] / 2) == pytest.approx(0.01)  # 1 cm gap: kinematic carriers never touch
    assert M.CARRIER_TOP - M.SHELF_TOP == pytest.approx(0.005)  # a carton pushed across drops 5 mm, it can't catch
    assert M.SHELF_Y[0] < M.RIGHT_END_Y and M.SHELF_Y[1] == M.END_Y  # it runs the whole open line
    assert M.SHELF_X[1] - M.SHELF_X[0] - M.ON_SHELF_IN >= M.CARTON[0]  # a carton fits well inside its fence
    # the tunnel's far wall starts where the table's left lip ends, without overlapping it
    assert M.LINE_X + M.WALL_IN >= M.SHELF_X[0]


def test_a_push_low_on_the_carton_slides_it_rather_than_tipping_it():
    mu = (1.2 + 0.8) / 2  # the cartons' and the carriers' static friction, averaged as PhysX does
    contact_h = M.CARTON[2]  # worst case: the fingers touch the carton at its top edge
    assert mu * contact_h < 0.9 * M.CARTON[0] / 2


# -- scoring -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "moved, expected",
    [
        ({}, 0.0),  # nothing done: the rejects went into the tunnel
        ({"reject_1": shelved(0.0)}, 0.5),
        ({"reject_2": shelved(0.2)}, 0.5),
        ({"reject_1": shelved(-0.1), "reject_2": shelved(0.15)}, 1.0),
        ({"reject_1": ON_TABLE_TOP(0.0)}, 0.125),  # pushed off toward the robot
        ({"reject_1": ON_FLOOR(0.0), "reject_2": ON_TABLE_TOP(0.2)}, 0.25),
        ({"reject_1": shelved(-0.1), "reject_2": ON_TABLE_TOP(0.2)}, 0.625),
        ({"reject_1": shelved(-0.1), "reject_2": shelved(0.15), "good_1": shelved(0.0)}, 0.5),  # one good one too
        ({"reject_1": shelved(-0.1), "good_2": ON_TABLE_TOP(0.0)}, 0.0),
        ({"reject_1": shelved(-0.1), "reject_2": shelved(0.15), "good_2": TIPPED("good_2")}, 0.5),
        ({"reject_1": TIPPED("reject_1")}, 0.0),  # knocked over on its carrier: still on the line
    ],
)
def test_score_levels(moved, expected):
    objs = scene(moved=moved)
    assert M.score(objs) == pytest.approx(expected)
    assert M.success(objs) == (expected == 1.0)


def test_pushing_or_sweeping_every_carton_off_scores_nothing():
    everything_shelved = {n: shelved(-0.3 + 0.13 * i) for i, n in enumerate(M.NAMES)}
    assert M.score(scene(moved=everything_shelved)) == 0.0
    everything_swept = {n: ON_TABLE_TOP(-0.3 + 0.13 * i) for i, n in enumerate(M.NAMES)}
    assert M.score(scene(moved=everything_swept)) == 0.0
    everything_on_floor = {n: ON_FLOOR(-0.3 + 0.13 * i) for i, n in enumerate(M.NAMES)}
    assert M.score(scene(moved=everything_on_floor)) == 0.0


def test_a_carton_counts_on_the_table_only_lying_still_well_onto_it():
    for s in (
        shelved(0.0, vel=(0.0, 0.05, 0.0)),  # still sliding
        state((M.SHELF_X[0] + M.ON_SHELF_IN - 0.005, 0.0, M.SHELF_TOP + M.CARTON[2] / 2)),  # only just across the gap
        state((M.SHELF_GOAL_X, 0.0, M.SHELF_TOP + 0.12)),  # held up over it
        state((M.SHELF_GOAL_X, M.END_Y + 0.05, M.SHELF_TOP + M.CARTON[2] / 2)),  # past its end
    ):
        objs = scene(moved={"reject_1": s})
        assert not M.on_shelf(s)
        assert M.score(objs) == pytest.approx(0.125)  # off the line, but not on the table
    tipped_on_table = state((M.SHELF_GOAL_X, 0.0, M.SHELF_TOP + M.CARTON[0] / 2), size=(0.05, 0.10, 0.12))
    assert M.score(scene(moved={"reject_1": tipped_on_table})) == pytest.approx(0.5)  # on its side still counts


def test_cartons_riding_on_the_belt_score_nothing_at_any_time():
    for t in (0.0, 5.0, 10.0, 20.0, 40.0):
        objs = scene(t=t)
        assert M.score(objs) == 0.0 and M.tally(objs) == (0, 0, 0)


def test_a_good_carton_lifted_off_its_carrier_is_spoiled():
    c = M.carrier_path("good_1")(40.0)[0]
    lifted = state((c[0], c[1], M.CARRIER_TOP + 0.10))
    objs = scene(moved={"reject_1": shelved(0.0), "reject_2": shelved(0.2), "good_1": lifted})
    assert M.score(objs) == pytest.approx(0.5)


def test_distance_falls_to_zero_when_both_rejects_are_on_the_table():
    assert M.distance(scene(t=0.0)) > 0.1
    assert M.distance(scene(moved={"reject_1": shelved(-0.1), "reject_2": shelved(0.15)})) == 0.0


# -- the demo -------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, out = 0.0, {}
    for w in plan:
        a, _ = out.get(w.label, (t, t))
        out[w.label] = (a, t + w.duration)
        t += w.duration
    return out


def _tcp_track(side):
    arm = ArmPlan(M.demo()[side], station.BASES[side])
    n = round(arm.duration / DT) + 5
    acts = np.array([arm.action7((k + 1) * DT)[0] for k in range(n)])
    return arm, acts


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable(side):
    arm, acts = _tcp_track(side)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(acts[:, :6] >= ik.LOWER) and np.all(acts[:, :6] <= ik.UPPER)


def test_the_demo_phases_line_up_with_its_waypoints():
    for n in M.REJECTS:
        spans = _spans(M.demo()[M.ARM[n]])
        for label, a, b in M.phases(n):
            assert spans[label][0] == pytest.approx(a, abs=1e-6) and spans[label][1] == pytest.approx(b, abs=1e-6)


def test_each_arm_pushes_in_its_own_half_and_before_the_tunnel():
    for n in M.REJECTS:
        for label, a, b in M.phases(n):
            for t in (a, b):
                y = M.part_y(n, t)
                assert (y <= -0.01) if M.ARM[n] == "right" else (y >= 0.05), (label, t)
                if not label.startswith(("hold", "rise")):  # by then the carton is on the table, off the line
                    assert y + M.CARTON[1] / 2 <= M.END_Y - 0.08, (label, t)  # 8 cm or more before the tunnel


def test_the_fingers_come_down_clear_of_the_carton_and_the_carrier_and_push_low():
    assert M.TIP_Z - PAD_TIP_PAST_TCP - M.CARRIER_TOP == pytest.approx(0.006)  # pad tips 6 mm above the carrier
    near_face = M.LINE_X - M.CARTON[0] / 2
    assert near_face - (M.START_X + M.FINGER_REACH) >= 0.015 - 1e-9
    assert M.START_X + M.FINGER_REACH < M.LINE_X - M.CARRIER[0] / 2  # in front of the carrier, not over it
    assert M.CLEAR_Z - PAD_TIP_PAST_TCP - M.CARTON_TOP >= 0.03 - 1e-9
    # only the fingers touch: the housing stays over the cartons' tops while pushing
    assert M.TIP_Z - PAD_TIP_PAST_TCP + M.HOUSING_ABOVE_TIP - M.CARTON_TOP >= 0.02


def test_the_push_leaves_the_carton_on_the_table_whatever_the_contact_point():
    for contact in (M.CONTACT, M.CONTACT_MAX, 0.004):
        centre = M.PUSH_TO_X + contact + M.CARTON[0] / 2
        assert centre >= M.SHELF_X[0] + M.ON_SHELF_IN + 0.005, contact
        assert centre + M.CARTON[0] / 2 <= M.SHELF_X[1] - 0.02, contact  # 2 cm short of the fence or more
        assert centre - M.CARTON[0] / 2 > M.LINE_X + M.CARRIER[0] / 2  # wholly off the carrier


def test_the_push_is_gentle():
    dist = M.PUSH_TO_X - M.TOUCH_X
    assert dist == pytest.approx(0.16)
    assert 1.5 * dist / M.PUSH_S <= 0.13  # smoothstep peak speed, m/s


def test_the_fingers_stay_well_away_from_the_neighbouring_cartons():
    # the fingers follow the pushed carton along the line, so the neighbours stay 13 cm away centre to centre
    pitch = min(a - b for a, b in zip([M.START_Y[n] for n in M.NAMES], [M.START_Y[n] for n in M.NAMES][1:]))
    finger_half_width = 0.034  # the fingers' widest extent along the line (gripper meshes)
    assert pitch - M.CARTON[1] / 2 - finger_half_width >= 0.04


def test_the_two_arms_stay_apart():
    _, la = _tcp_track("left")
    _, ra = _tcp_track("right")
    left_tcps = [ik.tcp_pose(q, station.BASES["left"])[0] for q in la[::4, :6]]
    right_tcps = [ik.tcp_pose(q, station.BASES["right"])[0] for q in ra[::4, :6]]
    n = min(len(left_tcps), len(right_tcps))
    assert min(np.linalg.norm(left_tcps[k] - right_tcps[k]) for k in range(n)) > 0.2


def test_the_demo_ends_well_before_the_run_limit():
    ends = [sum(w.duration for w in plan) for plan in M.demo().values()]
    assert max(ends) < 20.0  # success holds 1 s later: about 400 of the 1,200 steps
