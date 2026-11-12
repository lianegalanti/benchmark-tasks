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
    assert M.SHELF_X[1] - M.SHELF_X[0] >= M.CARTON[0] + 0.05  # a carton fits well inside its fence
    # the tunnel's far wall starts where the table's left lip ends, without overlapping it
    assert M.LINE_X + M.WALL_IN >= M.SHELF_X[0]


def test_a_push_low_on_the_carton_slides_it_rather_than_tipping_it():
    mu = (1.2 + 0.8) / 2  # the cartons' and the carriers' static friction, averaged as PhysX does
    contact_h = M.CARTON[2]  # worst case: the fingers touch the carton at its top edge
    assert mu * contact_h < 0.9 * M.CARTON[0] / 2


# -- scoring -------------------------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_monitor():
    M.reset_state()
    yield
    M.reset_state()


AWAY = {"left": ((0.20, 0.30, 0.95), (0.20, 0.30, 1.07)), "right": ((0.20, -0.30, 0.95), (0.20, -0.30, 1.07))}


def gripper_over(c, gap):
    """A gripper pointing down whose pad tips are ``gap`` above the top of carton state ``c``."""
    tip = np.array([c.pos[0], c.pos[1], c.high[2] + gap])
    return (tip, tip + np.array([0.0, 0.0, M.GRIPPER_LEN]))


def run(objects_over_time, grippers=None):
    """Feed the monitor one step per scene, then score the last one."""
    for objs in objects_over_time:
        M.monitor(objs, grippers or AWAY)
    return M.score(objects_over_time[-1])


@pytest.mark.parametrize(
    "moved, expected",
    [
        ({}, 0.0),  # nothing done: the rejects went into the tunnel
        ({"reject_1": shelved(0.0)}, 0.5),
        ({"reject_2": shelved(0.2)}, 0.5),
        ({"reject_1": shelved(-0.1), "reject_2": shelved(0.15)}, 1.0),
        ({"reject_1": ON_TABLE_TOP(0.0)}, 0.0),  # pushed off toward the robot: no credit
        ({"reject_1": ON_FLOOR(0.0), "reject_2": ON_TABLE_TOP(0.2)}, 0.0),  # floor and table top: no credit
        ({"reject_1": shelved(-0.1), "reject_2": ON_TABLE_TOP(0.2)}, 0.5),
        ({"reject_1": shelved(-0.1), "reject_2": ON_FLOOR(0.2)}, 0.5),
        ({"reject_1": shelved(-0.1), "reject_2": shelved(0.15), "good_1": shelved(0.0)}, 0.5),  # one good one too
        ({"reject_1": shelved(-0.1), "good_2": ON_TABLE_TOP(0.0)}, 0.0),
        ({"reject_1": shelved(-0.1), "reject_2": shelved(0.15), "good_2": TIPPED("good_2")}, 0.5),
        ({"reject_1": TIPPED("reject_1")}, 0.0),  # knocked over on its carrier: still on the line
    ],
)
def test_score_levels(moved, expected):
    objs = scene(moved=moved)
    assert run([objs]) == pytest.approx(expected)
    assert M.success(objs) == (expected == 1.0)


def test_pushing_or_sweeping_every_carton_off_scores_nothing():
    everything_shelved = {n: shelved(-0.3 + 0.13 * i) for i, n in enumerate(M.NAMES)}
    assert run([scene(moved=everything_shelved)]) == 0.0
    everything_swept = {n: ON_TABLE_TOP(-0.3 + 0.13 * i) for i, n in enumerate(M.NAMES)}
    assert run([scene(moved=everything_swept)]) == 0.0
    everything_on_floor = {n: ON_FLOOR(-0.3 + 0.13 * i) for i, n in enumerate(M.NAMES)}
    assert run([scene(moved=everything_on_floor)]) == 0.0


def test_a_carton_on_the_floor_or_the_station_table_scores_nothing():
    assert run([scene(moved={"reject_1": ON_FLOOR(0.0), "reject_2": ON_FLOOR(0.2)})]) == 0.0
    M.reset_state()
    assert run([scene(moved={"reject_1": ON_TABLE_TOP(0.0), "reject_2": ON_TABLE_TOP(0.2)})]) == 0.0


def test_a_carton_held_over_the_reject_table_scores_nothing():
    # held still by two hands 3 cm, 1.5 cm and 1.1 cm above the table top, released or not
    for lift in (0.03, 0.015, 0.011):
        held = state((M.SHELF_GOAL_X, 0.0, M.SHELF_TOP + lift + M.CARTON[2] / 2))
        objs = scene(moved={"reject_1": held})
        assert not M.on_shelf(held), lift
        assert run([objs], {"left": gripper_over(held, -0.02), "right": AWAY["right"]}) == 0.0
        M.reset_state()
        assert run([objs]) == 0.0  # even with no gripper near: not lying on the table
        M.reset_state()


def test_a_carton_lying_on_the_table_counts_only_once_let_go():
    c = shelved(0.0)
    objs = scene(moved={"reject_1": c})
    for gap in (-0.03, 0.0, 0.02, 0.039):  # pressed down on, or a gripper within 4 cm of its top
        assert run([objs], {"left": AWAY["left"], "right": gripper_over(c, gap)}) == 0.0, gap
    beside = (np.array([c.low[0] - 0.035, 0.0, c.low[2] + 0.01]), np.array([c.low[0] - 0.035, 0.0, c.low[2] + 0.13]))
    assert run([objs], {"left": AWAY["left"], "right": beside}) == 0.0  # fingers 3.5 cm from its near face
    assert run([objs], {"left": AWAY["left"], "right": gripper_over(c, 0.041)}) == pytest.approx(0.5)
    assert run([objs]) == pytest.approx(0.5)


def test_the_bottom_to_surface_tolerance_is_1_cm():
    for lift, counts in ((0.0, True), (0.009, True), (0.011, False), (-0.011, False)):
        c = state((M.SHELF_GOAL_X, 0.0, M.SHELF_TOP + lift + M.CARTON[2] / 2))
        assert M.on_shelf(c) == counts, lift


def test_a_carton_counts_on_the_table_only_wholly_on_it_and_still():
    for s in (
        shelved(0.0, vel=(0.0, 0.05, 0.0)),  # still sliding
        state((M.SHELF_X[0] + M.CARTON[0] / 2 - 0.005, 0.0, M.SHELF_TOP + M.CARTON[2] / 2)),  # over the gap's edge
        state((M.SHELF_GOAL_X, M.END_Y - 0.03, M.SHELF_TOP + M.CARTON[2] / 2)),  # over its left fence
    ):
        assert not M.on_shelf(s)
        assert run([scene(moved={"reject_1": s})]) == 0.0
        M.reset_state()
    tipped_on_table = state((M.SHELF_GOAL_X, 0.0, M.SHELF_TOP + M.CARTON[0] / 2), size=(0.05, 0.10, 0.12))
    assert run([scene(moved={"reject_1": tipped_on_table})]) == pytest.approx(0.5)  # on its side, it still lies there


def test_a_brown_carton_knocked_off_and_put_back_still_costs():
    knocked = scene(t=10.0, moved={"good_1": ON_TABLE_TOP(0.2)})
    end = scene(moved={"reject_1": shelved(-0.1), "reject_2": shelved(0.15)})  # good_1 back on its carrier
    assert run([scene(t=5.0), knocked, end]) == pytest.approx(0.5)
    assert not M.success(end)
    M.reset_state()
    tipped_then_righted = scene(t=10.0, moved={"good_2": TIPPED("good_2", 10.0)})
    assert run([tipped_then_righted, end]) == pytest.approx(0.5)


def test_a_red_carton_parked_on_the_station_table_then_pushed_onto_the_reject_table_never_counts():
    # knocked toward the robot onto the station table, left there, and pushed across once the belt has stopped
    parked = scene(t=8.0, moved={"reject_1": ON_TABLE_TOP(0.0)})
    end = scene(moved={"reject_1": shelved(0.0), "reject_2": shelved(0.15)})
    assert run([scene(t=5.0), parked, scene(t=25.0, moved={"reject_1": ON_TABLE_TOP(0.0)}), end]) == pytest.approx(0.5)
    assert not M.success(end)


def test_a_red_carton_pushed_off_the_end_of_the_reject_table_side_then_back_never_counts():
    beyond = scene(t=12.0, moved={"reject_2": state((0.42, M.END_Y + 0.08, TOP + M.CARTON[2] / 2))})
    end = scene(moved={"reject_1": shelved(-0.1), "reject_2": shelved(0.15)})
    assert run([beyond, end]) == pytest.approx(0.5)


def test_a_red_carton_lifted_more_than_1_cm_never_counts():
    # the instruction says push: lifted over the line (two hands) and set down on the reject table scores nothing
    c = M.carrier_path("reject_1")(4.0)[0]
    for lift in (0.011, 0.05):
        M.reset_state()
        lifted = scene(t=4.0, moved={"reject_1": state((c[0], c[1], M.CARRIER_TOP + lift + M.CARTON[2] / 2))})
        end = scene(moved={"reject_1": shelved(-0.1), "reject_2": shelved(0.15)})
        assert run([lifted, end]) == pytest.approx(0.5), lift
    M.reset_state()
    slid = scene(t=4.0, moved={"reject_1": state((c[0], c[1], M.CARRIER_TOP + 0.009 + M.CARTON[2] / 2))})
    assert run([slid, scene(moved={"reject_1": shelved(-0.1), "reject_2": shelved(0.15)})]) == 1.0


def test_a_red_carton_pushed_across_the_gap_onto_the_table_stays_on_route():
    # the demo's push, every 5 mm: centre from the line to the table, its bottom on the carrier, then on the table
    for x in np.arange(M.LINE_X, M.SHELF_GOAL_X + 1e-9, 0.005):
        z = M.CARRIER_TOP if x - M.CARTON[0] / 2 < M.LINE_X + M.CARRIER[0] / 2 else M.SHELF_TOP
        y = M.part_y("reject_1", 3.0)
        M.monitor(scene(t=3.0, moved={"reject_1": state((x, y, z + M.CARTON[2] / 2))}), AWAY)
    assert M._STATE["lost"] == set()


def test_an_open_gripper_resting_on_a_reject_is_not_let_go():
    c = shelved(0.0)
    objs = scene(moved={"reject_1": c, "reject_2": shelved(0.15)})
    # fingers wide open (4.7 cm each), straddling the carton, pad tips 1 cm below its top: the segment is 6 cm (half the
    # carton's 12 cm) plus a little from its sides, more than the closed-finger 4 cm margin
    tip = np.array([c.pos[0], c.pos[1], c.high[2] - 0.01])
    open_grip = (tip, tip + np.array([0.0, 0.0, M.GRIPPER_LEN]), -FINGER_OPEN)
    assert M.gripper_distance(c, open_grip) == 0.0
    over = (tip + np.array([0.0, 0.0, 0.05]), tip + np.array([0.0, 0.0, 0.05 + M.GRIPPER_LEN]), -FINGER_OPEN)
    assert 0.04 <= M.gripper_distance(c, over) < 0.04 - FINGER_OPEN  # 4 cm over its top, fingers open: still too close
    assert run([objs], {"left": AWAY["left"], "right": over}) == pytest.approx(0.5)
    M.reset_state()
    closed = (over[0], over[1], 0.0)
    assert run([objs], {"left": AWAY["left"], "right": closed}) == pytest.approx(1.0)
    M.reset_state()
    far = (over[0] + np.array([0, 0, 0.05]), over[1] + np.array([0, 0, 0.05]), -FINGER_OPEN)
    assert run([objs], {"left": AWAY["left"], "right": far}) == pytest.approx(1.0)  # 9 cm over: let go


def test_a_red_carton_that_fell_to_the_floor_never_counts():
    fell = scene(t=10.0, moved={"reject_1": ON_FLOOR(0.0)})
    end = scene(moved={"reject_1": shelved(-0.1), "reject_2": shelved(0.15)})
    assert run([fell, end]) == pytest.approx(0.5)


def test_the_monitor_flags_nothing_while_the_line_runs_untouched():
    run([scene(t=t) for t in np.arange(0.0, 40.0, 0.5)])
    assert M._STATE["spoiled"] == set() and M._STATE["lost"] == set()
    for t in (0.0, 5.0, 10.0, 20.0, 40.0):
        objs = scene(t=t)
        assert M.score(objs) == 0.0 and M.tally(objs) == (0, 0)


def test_a_good_carton_lifted_off_its_carrier_is_spoiled():
    c = M.carrier_path("good_1")(40.0)[0]
    lifted = state((c[0], c[1], M.CARRIER_TOP + 0.10))
    objs = scene(moved={"reject_1": shelved(0.0), "reject_2": shelved(0.2), "good_1": lifted})
    assert run([objs]) == pytest.approx(0.5)


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
        assert centre - M.CARTON[0] / 2 >= M.SHELF_X[0] + 0.01, contact  # wholly on the table, 1 cm to spare
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


def _demo_end_box(n, contact):
    """Where the demo leaves carton ``n``: pushed across at the end of its push, with the fingers touching it
    ``contact`` ahead of the TCP."""
    push_end = next(b for label, _, b in M.phases(n) if label.startswith("push"))
    x = M.PUSH_TO_X + contact + M.CARTON[0] / 2
    return state((x, M.part_y(n, push_end), M.SHELF_TOP + M.CARTON[2] / 2))


def test_the_gripper_segment_points_down_from_the_pad_tips_while_pushing():
    for side, n in (("right", "reject_1"), ("left", "reject_2")):
        arm, acts = _tcp_track(side)
        _, a, b = next(p for p in M.phases(n) if p[0].startswith("push"))
        q = acts[round((a + b) / 2 / DT) - 1, :6]
        tip, top, _ = M.gripper_segment(q, station.BASES[side])
        assert tip[2] - M.CARRIER_TOP == pytest.approx(0.006, abs=1e-3)  # the pad tips, 6 mm over the carriers
        assert top - tip == pytest.approx([0.0, 0.0, M.GRIPPER_LEN], abs=1e-3)


def test_the_demo_lets_go_in_time_for_the_success_hold():
    # yam_scripted stops 3 s after the plan's last waypoint: both rejects must count (both grippers 4 cm or more away)
    # at least 1.5 s before then, so that success can hold its 1 s
    ends = {side: sum(w.duration for w in M.demo()[side]) for side in ("left", "right")}
    stop = max(ends.values()) + 3.0
    tracks = {side: _tcp_track(side)[1] for side in ("left", "right")}
    for contact in (0.004, M.CONTACT, M.CONTACT_MAX):
        boxes = {n: _demo_end_box(n, contact) for n in M.REJECTS}
        clear_from = 0.0
        for k in range(round(stop / DT)):
            segs = {side: M.gripper_segment(acts[min(k, len(acts) - 1), :6], station.BASES[side])
                    for side, acts in tracks.items()}
            near = min((M.gripper_distance(boxes[n], seg) for n in M.REJECTS for seg in segs.values()
                        if (k + 1) * DT > next(b for label, _, b in M.phases(n) if label.startswith("push"))),
                       default=1.0)
            if near < M.RELEASE_CLEAR:
                clear_from = (k + 1) * DT
        print(f"contact {contact}: both rejects let go from {clear_from:.2f} s, rollout stops at {stop:.2f} s")
        assert clear_from + 1.0 <= stop - 1.5, (contact, clear_from, stop)
        # parked, each gripper ends well clear of both cartons
        for side, acts in tracks.items():
            seg = M.gripper_segment(acts[-1, :6], station.BASES[side])
            assert min(M.gripper_distance(boxes[n], seg) for n in M.REJECTS) >= 0.1


def test_the_demo_scores_full_with_the_monitor():
    end = scene(moved={n: _demo_end_box(n, M.CONTACT) for n in M.REJECTS})
    rest = {side: M.gripper_segment(_tcp_track(side)[1][-1, :6], station.BASES[side]) for side in ("left", "right")}
    assert run([scene(t=t) for t in (0.0, 1.0, 2.0)] + [end], rest) == 1.0 and M.success(end)
