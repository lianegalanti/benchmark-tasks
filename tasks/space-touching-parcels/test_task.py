"""Offline checks (no Isaac Sim): the belt, the layout, the scanner's record and every scoring level, and that the demo
is reachable, keeps the arms apart and clear of the scanner, and spaces the parcels in time (replayed with the parcels
riding the belt or held in the fingers).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
RUN_LIMIT_S = 800 * DT  # the run limit: 800 control steps
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")


def state(pos, size=M.PARCEL, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def riding(shifts=None, x=None, z=None):
    """A function of t giving each parcel's state if it rides the belt, moved ``shifts[name]`` along it from the start
    (at t = 0). ``x`` / ``z`` override a parcel's centre across the belt / height."""
    shifts, x, z = shifts or {}, x or {}, z or {}

    def at(t):
        return {n: state((x.get(n, M.LINE_X), M.riding_y(n, t, shifts.get(n, 0.0)), z.get(n, M.PARCEL_Z)))
                for n in M.NAMES}

    return at


def play(at, until=RUN_LIMIT_S):
    """Run the scanner's record over the control steps of a trial, the parcels as ``at(t)`` gives them; return the
    final (score, why)."""
    M.reset_state()
    for k in range(round(until / DT) + 1):
        M.update(k * DT, at(k * DT))
    return M.result()


# -- loading and layout ----------------------------------------------------------------------------------------------


def test_it_loads_with_three_free_parcels_and_a_belt():
    assert TASK.free_objects == M.NAMES
    assert "belt" in TASK.names and TASK.score_over == "end"


def test_positions_and_sizes_are_plain_floats():
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name
        assert all(type(v) is float for v in o.size), o.name


def test_the_parcels_start_nose_to_tail_touching_on_the_belt():
    ys = [M.START[n] for n in M.NAMES]
    assert ys[0] + M.PARCEL[1] / 2 == pytest.approx(M.FRONT_Y0)
    assert np.diff(ys) == pytest.approx([-(M.PARCEL[1] + M.TOUCH)] * 2)
    for p in M.PARCELS:
        assert p.pos[2] - M.PARCEL[2] / 2 == pytest.approx(M.BELT_TOP)
        assert M.on_belt(state(p.pos)) is None
    assert M.START["parcel_3"] - M.PARCEL[1] / 2 - 0.13 > M.START_Y + 0.10  # room behind the train to set one back


def test_the_belt_floats_over_its_bed_and_the_arch_clears_the_parcels():
    assert M.BELT_Z - M.BELT_T / 2 - M.BED_TOP == pytest.approx(0.002)
    assert M.HEAD_LOW - M.MARK - M.PARCEL_TOP > 0.07  # a parcel passes 7.6 cm under the red line
    assert M.POST_IN - M.BELT_W / 2 >= 0.015  # the posts stand 1.5 cm outside the belt's edges
    # parcels gather against the end stop well past the scan line: all three, closed up, clear it by 20 cm
    assert M.END_Y - 3 * M.PARCEL[1] - M.SCAN_Y > 0.20
    assert M.START_Y > -M.TABLE_EDGE_Y and M.SCAN_Y + M.HEAD[1] / 2 < M.TABLE_EDGE_Y


def test_the_belt_ramps_up_smoothly_then_runs_at_5_cm_per_s():
    ts = np.arange(0.0, 30.0, 1 / 240)
    run = np.array([M.RUN(t) for t in ts])
    assert run[0, 0] == 0.0 and run[0, 1] == 0.0
    assert np.all(np.diff(run[:, 0]) >= 0)
    assert np.allclose(np.gradient(run[:, 0], ts)[3:-3], run[3:-3, 1], atol=1e-3)
    assert np.max(np.abs(np.diff(run[:, 1]))) * 240 < 0.2  # peak acceleration 0.16 m/s², far below sliding
    late = ts >= M.RAMP_S
    assert np.allclose(run[late, 1], M.SPEED)
    assert M.RUN(10.0)[0] == pytest.approx(M.SPEED * (10.0 - M.RAMP_S / 2))


def test_the_belt_gives_its_surface_velocity_and_stays_put():
    assert M.BELT.surface_velocity(5.0, 1 / 240) == pytest.approx([0.0, M.SPEED, 0.0])
    assert M.BELT.surface_velocity(0.0, 1 / 240)[1] < 1e-4  # standing at the start


def test_untouched_the_parcels_pass_well_within_the_run_limit():
    # even a parcel set down at the far upstream end of the belt as late as 14 s still passes in time
    assert (M.SCAN_Y - M.START_Y) / M.SPEED + 14.0 < RUN_LIMIT_S - 1.0


# -- the scanner's record and the score -------------------------------------------------------------------------------


def test_doing_nothing_scores_0_the_gaps_are_the_2_mm_the_parcels_started_with():
    score, why = play(riding())
    assert score == 0.0
    assert M.gaps() == pytest.approx([M.TOUCH, M.TOUCH], abs=1e-3), why


@pytest.mark.parametrize(
    ("shifts", "expected"),
    [
        ({"parcel_1": 0.10, "parcel_3": -0.10}, 1.0),  # the demo's plan
        ({"parcel_2": -0.06, "parcel_3": -0.12}, 1.0),  # both back, one after the other
        ({"parcel_3": -0.10}, 0.5),  # only one gap opened
        ({"parcel_1": 0.04, "parcel_3": -0.10}, 0.75),  # one gap 4.2 cm: half credit
        ({"parcel_1": 0.04, "parcel_3": -0.03}, 0.5),  # 4.2 and 3.2 cm
        ({"parcel_1": 0.02, "parcel_3": -0.06}, 0.5),  # 2.2 cm (nothing) and 6.2 cm
        ({"parcel_1": 0.047, "parcel_3": -0.047}, 0.5),  # 4.9 cm each: just short
        ({"parcel_1": 0.049, "parcel_3": -0.049}, 1.0),  # 5.1 cm each
        ({"parcel_3": 0.40}, 0.5),  # parcel_3 set 12.6 cm ahead of the train: the order changes, gaps still count
    ],
)
def test_gap_credit_for_each_of_the_two_gaps(shifts, expected):
    score, why = play(riding(shifts))
    assert score == expected, why


def test_the_gap_is_measured_to_within_a_millimetre_between_control_steps():
    play(riding({"parcel_1": 0.0731, "parcel_3": -0.0517}))
    assert M.gaps() == pytest.approx([0.0751, 0.0537], abs=5e-4)


def test_the_order_of_reaching_the_line_decides_which_gaps_count():
    play(riding({"parcel_3": 0.40}))  # parcel_3 set 12.6 cm ahead of parcel_1
    order = sorted(M.NAMES, key=lambda n: M._STATE[n]["front_t"])
    assert order == ["parcel_3", "parcel_1", "parcel_2"]


def test_side_by_side_parcels_overlap_at_the_line_and_score_nothing_for_that_gap():
    # parcel_2 moved beside parcel_1 across the belt (each still on it), parcel_3 well back
    at = riding({"parcel_2": M.PARCEL[1] + M.TOUCH, "parcel_3": -0.10},
                x={"parcel_1": M.BELT_NEAR + 0.035, "parcel_2": M.BELT_FAR - 0.035})
    score, why = play(at)
    assert M.gaps()[0] < 0 and score == 0.5, why


def _with(base, t0, t1, name, pos):
    """``base``, except that parcel ``name`` is at ``pos(t)`` from ``t0`` to ``t1``."""

    def at(t):
        out = base(t)
        if t0 <= t <= t1:
            out[name] = state(pos(t))
        return out

    return at


def _front_reaches_line(name, shift=0.0):
    return (M.SCAN_Y - M.START[name] - shift - M.PARCEL[1] / 2) / M.SPEED + M.RAMP_S / 2


GOOD = {"parcel_1": 0.10, "parcel_3": -0.10}


def test_a_parcel_knocked_off_the_belt_fails_the_trial():
    t = _front_reaches_line("parcel_2") - 3.0
    score, why = play(_with(riding(GOOD), t, 99.0, "parcel_2", lambda _: (0.2, 0.0, M.TOP + M.PARCEL[2] / 2)))
    assert score == 0.0 and "parcel_2" in why


def test_a_parcel_carried_over_the_line_in_the_air_fails():
    # parcel_1 lifted 4 cm and carried past the line, then set down on the belt beyond it
    t0 = 3.0
    t1 = 7.0

    def pos(t):
        y = M.riding_y("parcel_1", t, 0.10) + (t - t0) / (t1 - t0) * 0.30
        return (M.LINE_X, y, M.PARCEL_Z + 0.04 if t < t1 else M.PARCEL_Z)

    score, why = play(_with(riding(GOOD), t0, t1, "parcel_1", pos))
    assert score == 0.0 and "not lying on the belt" in why


def test_a_parcel_over_the_belts_edge_at_the_line_fails():
    score, why = play(riding(GOOD, x={"parcel_2": M.BELT_FAR - M.PARCEL[0] / 2 + 0.005}))
    assert score == 0.0 and "edge" in why
    # anywhere on the belt is fine, its footprint just inside the edge
    assert play(riding(GOOD, x={"parcel_2": M.BELT_FAR - M.PARCEL[0] / 2 - 0.001}))[0] == 1.0


def test_a_parcel_stacked_on_another_fails():
    on_top = {"parcel_2": M.PARCEL_Z + M.PARCEL[2]}
    score, why = play(riding({"parcel_2": M.PARCEL[1] + M.TOUCH, "parcel_3": -0.10}, z=on_top))
    assert score == 0.0 and "parcel_2" in why


def test_a_parcel_held_back_off_the_belt_fails():
    t = 4.0
    held = _with(riding(GOOD), t, 99.0, "parcel_3", lambda _: (M.LINE_X, -0.40, M.PARCEL_Z + 0.04))
    score, why = play(held)
    assert score == 0.0 and "parcel_3 held off the belt for more than 5 s" in why


def test_a_parcel_held_back_on_the_belt_until_the_trial_ends_fails():
    held = _with(riding(GOOD), 4.0, 99.0, "parcel_3", lambda _: (M.LINE_X, -0.40, M.PARCEL_Z))
    score, why = play(held)
    assert score == 0.0 and "parcel_3 has not passed" in why


# -- staying on the belt before the scanner --------------------------------------------------------------------------


def _carry(name, t0, t1, path, shift_after):
    """``riding(GOOD)``, except that parcel ``name`` follows ``path(t)`` from ``t0`` to ``t1`` and afterwards rides the
    belt moved ``shift_after`` from its start."""
    base, after = riding(GOOD), riding({**GOOD, name: shift_after})

    def at(t):
        out = base(t)
        if t0 <= t <= t1:
            out[name] = state(path(t))
        elif t > t1:
            out[name] = after(t)[name]
        return out

    return at


def test_staging_a_parcel_on_the_table_then_returning_it_scores_0():
    # parcel_3 lifted, set down on the table beside the belt for 2 s, then put back on the belt 10 cm back: the gaps
    # at the scanner would earn full credit, but the parcel left the belt
    off_x = M.BELT_NEAR - 0.08
    on_table = M.TOP + M.PARCEL[2] / 2

    def path(t):
        y = M.riding_y("parcel_3", 2.0)
        if t < 2.5:
            return (M.LINE_X + (off_x - M.LINE_X) * (t - 2.0) / 0.5, y, M.PARCEL_Z + 0.06)
        if t < 4.5:
            return (off_x, y, on_table)
        return (M.LINE_X, M.riding_y("parcel_3", t, -0.10), M.PARCEL_Z + 0.03)

    at = _carry("parcel_3", 2.0, 5.0, path, -0.10)
    score, why = play(at)
    assert score == 0.0 and "parcel_3 taken off the belt" in why
    # the same gaps without the detour score full credit
    assert play(riding(GOOD))[0] == 1.0


def test_carrying_a_parcel_out_past_the_belts_edge_in_the_air_scores_0():
    # held 10 cm up, its centre 1 cm past the far edge for a moment, then back
    def path(t):
        return (M.BELT_FAR + 0.01, M.riding_y("parcel_3", t, -0.10), M.PARCEL_Z + 0.10)

    score, why = play(_carry("parcel_3", 3.0, 3.2, path, -0.10))
    assert score == 0.0 and "taken off the belt" in why


def test_a_brief_lift_over_the_belt_is_allowed_a_long_one_is_not():
    def path(t):
        return (M.LINE_X, M.riding_y("parcel_3", t, -0.10), M.PARCEL_Z + 0.08)

    for seconds, expected in ((2.0, 1.0), (4.9, 1.0), (5.2, 0.0)):
        score, why = play(_carry("parcel_3", 2.0, 2.0 + seconds, path, -0.10))
        assert score == expected, (seconds, why)
    # a lift of 4 s, a touchdown of 0.3 s, then 4 s more in the air: one lift of 8.3 s, not two of 4 s
    def dipping(t):
        z = M.PARCEL_Z if 6.0 <= t < 6.3 else M.PARCEL_Z + 0.08
        return (M.LINE_X, M.riding_y("parcel_3", t, -0.10), z)

    score, why = play(_carry("parcel_3", 2.0, 10.3, dipping, -0.10))
    assert score == 0.0 and "held off the belt for more than 5 s" in why
    # resting on the belt 0.5 s between the two lifts ends the first one
    def resting(t):
        z = M.PARCEL_Z if 6.0 <= t < 6.55 else M.PARCEL_Z + 0.08
        return (M.LINE_X, M.riding_y("parcel_3", t, -0.10), z)

    assert play(_carry("parcel_3", 2.0, 10.5, resting, -0.10))[0] == 1.0
    # its centre right at the belt's edge, but over the belt: still allowed
    edge = lambda t: (M.BELT_FAR - 0.001, M.riding_y("parcel_3", t, -0.10), M.PARCEL_Z + 0.08)  # noqa: E731
    assert play(_carry("parcel_3", 2.0, 3.0, edge, -0.10))[0] == 1.0


def test_setting_a_parcel_down_on_another_then_returning_it_scores_0():
    # parcel_3 rests on parcel_2 for 1 s (lifted for 2 s in all), then goes back on the belt 10 cm back
    def path(t):
        if 2.5 <= t < 3.5:
            return (M.LINE_X, M.riding_y("parcel_2", t), M.PARCEL_Z + M.PARCEL[2])
        return (M.LINE_X, M.riding_y("parcel_3", t, -0.10), M.PARCEL_Z + 0.07)

    score, why = play(_carry("parcel_3", 2.0, 4.0, path, -0.10))
    assert score == 0.0 and "parcel_3 set down on another parcel" in why
    # carried over a neighbour, 1 cm clear of its top, it's a lift
    over = lambda t: (M.LINE_X, M.riding_y("parcel_2", t), M.PARCEL_Z + M.PARCEL[2] + 0.012)  # noqa: E731
    assert play(_carry("parcel_3", 2.0, 2.4, over, -0.10))[0] == 1.0  # then set down 10 cm back, as in GOOD


def test_a_parcel_knocked_off_and_put_back_before_the_scanner_scores_0():
    def path(t):
        return (M.BELT_NEAR - 0.06, M.riding_y("parcel_2", 3.0), M.TOP + M.PARCEL[2] / 2)

    score, why = play(_carry("parcel_2", 3.0, 3.5, path, 0.0))
    assert score == 0.0 and "parcel_2 taken off the belt" in why


def test_lifting_a_parcel_after_it_has_passed_changes_nothing():
    # parcel_1, once past the line, lifted off the belt and carried away downstream
    gone = _front_reaches_line("parcel_1", 0.10) + M.PARCEL[1] / M.SPEED + 0.5
    score, _ = play(_with(riding(GOOD), gone, 99.0, "parcel_1", lambda _: (0.2, 0.9, M.TOP + 0.2)))
    assert score == 1.0


def test_a_passed_parcel_brought_back_in_front_of_the_next_shortens_its_gap():
    # parcel_1, once past the line, set back on the belt 2 cm in front of parcel_2, which has not reached the line yet
    gone = _front_reaches_line("parcel_1", 0.10) + M.PARCEL[1] / M.SPEED + 0.5
    back = lambda t: (M.LINE_X, M.riding_y("parcel_2", t, M.PARCEL[1] + 0.02), M.PARCEL_Z)  # noqa: E731
    score, _ = play(_with(riding(GOOD), gone, 99.0, "parcel_1", back))
    assert M.gaps()[0] == pytest.approx(0.02, abs=0.002) and score == 0.5


def test_a_parcel_pushed_forward_after_the_one_ahead_has_left_shortens_its_gap():
    # parcel_2 slid 8 cm ahead once parcel_1 has left the line: the gap measured as parcel_2 reaches the line counts
    left = _front_reaches_line("parcel_1", 0.10) + M.PARCEL[1] / M.SPEED + 0.2
    at = _with(riding(GOOD), left, 99.0, "parcel_2",
               lambda t: (M.LINE_X, M.riding_y("parcel_2", t, 0.08), M.PARCEL_Z))
    score, _ = play(at)
    assert M.gaps()[0] == pytest.approx(0.102 - 0.08, abs=0.002) and score == 0.5


def test_gaps_are_measured_between_the_parcels_whatever_the_belt_does():
    # the parcels ride 1.1 cm/s slower than the belt's surface, as the friction belt lets them slip in Isaac Sim:
    # the gaps are still the 10.2 cm between them (belt travel between the photo-eye's events would read 13 cm)
    slip = 0.011

    def at(t):
        return {n: state((M.LINE_X, M.START[n] + GOOD.get(n, 0.0) + M.RUN(t)[0] * (1 - slip / M.SPEED), M.PARCEL_Z))
                for n in M.NAMES}

    score, why = play(at, until=35.0)
    assert score == 1.0, why
    assert M.gaps() == pytest.approx([0.102, 0.102], abs=5e-4)
    # with less than 5 cm between them, slip doesn't lift the gaps into full credit
    small = {"parcel_1": 0.045, "parcel_3": -0.045}

    def at_small(t):
        return {n: state((M.LINE_X, M.START[n] + small.get(n, 0.0) + M.RUN(t)[0] * (1 - slip / M.SPEED), M.PARCEL_Z))
                for n in M.NAMES}

    assert play(at_small, until=35.0)[0] == 0.5


def test_lifting_a_parcel_while_it_straddles_the_line_fails():
    mid = _front_reaches_line("parcel_2") + 0.5 * M.PARCEL[1] / M.SPEED
    score, why = play(_with(riding(GOOD), mid, mid + 1.0, "parcel_2",
                            lambda t: (M.LINE_X, M.riding_y("parcel_2", mid), M.PARCEL_Z + 0.05)))
    assert score == 0.0 and "parcel_2" in why


def test_a_new_trial_starts_with_an_empty_record():
    play(riding(GOOD))
    assert M.success({})
    M.reset_state()
    assert M.score({}) == 0.0 and not M.success({})


# -- the demo ----------------------------------------------------------------------------------------------------------


def _plans():
    return {arm: ArmPlan(plan, station.BASES[arm]) for arm, plan in M.demo().items()}


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        a, _ = spans.get(w.label, (t, t))
        spans[w.label] = (a, t + w.duration)
        t += w.duration
    return spans


def _steps(plans):
    return round(max(p.duration for p in plans.values()) / DT) + 5


def test_the_demo_is_reachable_with_both_arms():
    plans = _plans()
    for arm, plan in plans.items():
        actions = np.array([plan.action7((k + 1) * DT)[0] for k in range(_steps(plans))])
        assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1, arm
        assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER), arm
        # yam_scripted ends the rollout 3 s after the plan: it must outlast success (19.5 s) and fit the run limit
        assert 19.5 + 1.0 < plan.duration < RUN_LIMIT_S - 3.0


def test_the_demo_grips_the_parcels_across_the_belt_clear_of_the_belt_and_lifts_them():
    for plan in M.demo().values():
        assert abs(abs(plan[-1].yaw) - np.pi / 2) < 1e-9  # fingers close along x, across the belt
    assert M.PARCEL[0] + 0.015 * 2 < 2 * abs(FINGER_OPEN)  # 1.7 cm either side between open pads and a parcel
    assert M.GRASP_Z - PAD_TIP_PAST_TCP - M.BELT_TOP >= 0.015  # pad tips stop 1.5 cm or more above the belt
    assert M.PRE_Z - PAD_TIP_PAST_TCP - M.PARCEL_TOP == pytest.approx(0.03)
    assert M.OFF_Z - M.GRASP_Z >= 0.05  # lifted 5 cm: a parcel tilting in the fingers still clears the belt


def test_the_demo_follows_each_parcel_along_the_belt_while_at_it():
    for arm, name, shift, meet, _, _ in M.MOVES:
        plan = M.demo()[arm]
        t = 0.0
        for w in plan:
            t += w.duration
            if w.label in ("follow it down", "close", "lift"):
                assert w.pos[1] == pytest.approx(M.riding_y(name, t)), (arm, w.label, t)
            if w.label in ("set down", "let go", "up"):
                assert w.pos[1] == pytest.approx(M.riding_y(name, t, shift)), (arm, w.label, t)
        assert M.demo()[arm][0].pos[1] == pytest.approx(M.riding_y(name, meet))


def _replay():
    """The demo replayed: each parcel rides the belt, except while a gripper holds it (from the end of closing to the
    middle of letting go), when it hangs in the fingers; let go, it rides the belt from where it was set down."""
    plans = _plans()
    spans = {arm: _spans(M.demo()[arm]) for arm in plans}
    who = {name: arm for arm, name, *_ in M.MOVES}
    offset = M.PARCEL_Z - M.GRASP_Z  # the parcel's centre above the TCP, as gripped
    let_go = {}  # name -> (time, y) where it was set down

    def at(t):
        out = {}
        for n in M.NAMES:
            arm = who.get(n)
            if arm is None:
                out[n] = state((M.LINE_X, M.riding_y(n, t), M.PARCEL_Z))
                continue
            held = (spans[arm]["close"][1], sum(spans[arm]["let go"]) / 2)
            if t < held[0]:
                out[n] = state((M.LINE_X, M.riding_y(n, t), M.PARCEL_Z))
            elif t <= held[1]:
                tcp = plans[arm].target(t)[0]
                out[n] = state(tcp + [0.0, 0.0, offset])
                let_go[n] = (t, tcp[1])
            else:
                t0, y0 = let_go[n]
                out[n] = state((M.LINE_X, y0 + M.RUN(t)[0] - M.RUN(t0)[0], M.PARCEL_Z))
        return out

    return at


def test_the_demo_replayed_spaces_the_parcels_with_wide_margins():
    score, why = play(_replay())
    assert score == 1.0, why
    assert min(M.gaps()) > 2 * M.MIN_GAP - 0.002, why  # about 10 cm each, twice what the scanner needs
    last = max(M._STATE[n]["rear_t"] for n in M.NAMES)
    assert last + 1.5 < M.END_S  # success has held 1 s before the plans end
    assert M.END_S + 3.0 < RUN_LIMIT_S  # yam_scripted stops 3 s after the plans end: still within the run limit


def test_the_demo_sets_both_parcels_down_well_before_the_first_reaches_the_line():
    play(_replay())
    first = min(M._STATE[n]["front_t"] for n in M.NAMES)
    for arm in ("left", "right"):
        assert _spans(M.demo()[arm])["let go"][1] + 2.0 < first, arm


def test_the_held_parcels_stay_clear_of_the_others_and_on_the_belt_when_set_down():
    at = _replay()
    plans = _plans()
    for k in range(_steps(plans)):
        s = at(k * DT)
        for a in M.NAMES:
            for b in M.NAMES:
                if a < b:
                    lo = np.maximum(s[a].low, s[b].low)
                    hi = np.minimum(s[a].high, s[b].high)
                    assert np.any(hi - lo < 0.001), (k * DT, a, b)  # never inside each other
    for n in ("parcel_1", "parcel_3"):
        assert M.on_belt(at(10.0)[n]) is None


def _links(arm, q):
    base = np.asarray(station.BASES[arm])
    fk = forward_kinematics(YAM_URDF, {j: float(v) for j, v in zip(ARM_JOINTS, q, strict=True)})
    tcp, _ = ik.tcp_pose(q, base)
    pts = [fk[n][:3, 3] + base for n in CHAIN] + [tcp]
    return np.array([a + s * (b - a) for a, b in zip(pts[:-1], pts[1:]) for s in np.linspace(0, 1, 8)])


def test_the_two_arms_stay_well_apart():
    plans = _plans()
    closest = np.inf
    for k in range(_steps(plans)):
        for p in plans.values():
            p.action7((k + 1) * DT)
        a, b = _links("left", plans["left"].q), _links("right", plans["right"].q)
        closest = min(closest, float(np.min(np.linalg.norm(a[:, None] - b[None], axis=2))))
    # link centre lines (and the TCPs). Closest at about 2.7 s, wrist to wrist (link4-link5), both arms down at their
    # parcels: 13.8 cm between centre lines, so with wrist housings about 7 cm across, 6 cm or more of clear air.
    assert closest > 0.13


def test_the_left_gripper_stays_clear_of_the_scanner_arch():
    plans = _plans()
    near_face = M.SCAN_Y - M.HEAD[1] / 2
    for k in range(_steps(plans)):
        t = (k + 1) * DT
        plans["left"].action7(t)
        pts = _links("left", plans["left"].q)
        assert np.max(pts[:, 1]) < near_face - 0.08, t  # every link and the TCP 8 cm short of the arch


# -- the robot's collision meshes against each other and the parcels --------------------------------------------------
# The URDF's collision meshes of each arm's lower links, gripper and fingers, posed by forward kinematics on the demo's
# commanded joints (and finger opening) at every control step.

MESH_LINKS = ("link3", "link4", "link5", "gripper", "tip_left", "tip_right")


def _stl(path):
    import re
    import struct

    raw = path.read_bytes()
    if raw[:5] == b"solid" and b"facet" in raw[:300]:
        return np.array(re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw), float)
    n = struct.unpack("<I", raw[80:84])[0]
    rows = np.frombuffer(raw[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return rows["v"].reshape(-1, 3).astype(float)


def _meshes():
    import re

    from yam_arena.urdf import rpy_matrix

    urdf = Path(YAM_URDF)
    text = urdf.read_text()
    out = {}
    for link in MESH_LINKS:
        block = text.split(f'<link name="{link}"')[1].split("</link>")[0]
        parts = re.findall(r'<collision>.*?<origin xyz="([^"]+)" rpy="([^"]+)".*?filename="([^"]+)"', block, re.S)
        pts = np.vstack([(rpy_matrix(np.array(r.split(), float)) @ _stl(urdf.parent / f).T).T + np.array(x.split(), float)
                         for x, r, f in parts])
        out[link] = pts[:: max(1, len(pts) // 400)]  # about 400 vertices per link
    return out


MESHES = _meshes()


def _mesh_points(arm, q, finger):
    fk = forward_kinematics(YAM_URDF, {**{j: float(v) for j, v in zip(ARM_JOINTS, q, strict=True)},
                                       "joint7": float(finger), "joint8": float(finger)})
    base = np.asarray(station.BASES[arm])
    return np.vstack([(fk[n][:3, :3] @ MESHES[n].T).T + fk[n][:3, 3] + base for n in MESH_LINKS])


def test_the_arms_collision_meshes_stay_8_cm_apart_and_touch_only_their_own_parcel():
    plans = _plans()
    at = _replay()
    own = {arm: name for arm, name, *_ in M.MOVES}
    closest_arms = np.inf
    closest = {}
    for k in range(_steps(plans)):
        t = (k + 1) * DT
        pts = {}
        for arm, plan in plans.items():
            plan.action7(t)
            pts[arm] = _mesh_points(arm, plan.q, plan.target(t)[2])
        closest_arms = min(closest_arms, float(np.min(np.linalg.norm(pts["left"][:, None] - pts["right"][None], axis=2))))
        parcels = at(t)
        for arm in plans:
            for n in M.NAMES:
                p = parcels[n]
                d = float(np.min(np.linalg.norm(np.maximum(np.maximum(p.low - pts[arm], pts[arm] - p.high), 0), axis=1)))
                closest[arm, n] = min(closest.get((arm, n), np.inf), d)
    assert closest_arms > 0.08
    for (arm, n), d in closest.items():
        if n != own[arm]:
            assert d > 0.02, (arm, n, d)  # never within 2 cm of another parcel, its 2 mm neighbour included


def test_the_demo_keeps_its_parcels_over_the_belt_and_lifts_them_briefly():
    at = _replay()
    plans = _plans()
    up, longest, edge = {}, dict.fromkeys(M.NAMES, 0.0), np.inf
    for k in range(_steps(plans)):
        t = k * DT
        s = at(t)
        for n in M.NAMES:
            if s[n].low[2] > M.BELT_TOP + M.ON_BELT:
                up.setdefault(n, t)
                longest[n] = max(longest[n], t - up[n])
            else:
                up.pop(n, None)
            edge = min(edge, s[n].pos[0] - M.BELT_NEAR, M.BELT_FAR - s[n].pos[0])
            assert not M._on_another(n, s), (t, n)
    assert max(longest.values()) < M.LIFT_MAX - 2.0, longest  # about 2.6 s: 2.4 s or more to spare
    assert edge > 0.07  # the centres stay on the belt's centre line, 8 cm from either edge


def test_the_demo_outlasts_the_last_parcel_even_with_the_belt_slipping():
    # in Isaac Sim the parcels ride about 1 cm/s slower than the belt's surface (the last parcel passed at about 25.6 s);
    # at 1.2 cm/s of slip from the start the last parcel (parcel_3, set 13 cm back) still passes, and success holds
    # 1 s, before the plans end (28.7 s against 30 s)
    slow = M.SPEED - 0.012
    rear = M.START["parcel_3"] - M.PARCEL[1] / 2 - 0.13
    passed = (M.SCAN_Y - rear) / slow + M.RAMP_S
    assert passed + 1.0 < M.END_S
