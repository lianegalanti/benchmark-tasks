"""Offline checks (no Isaac Sim): the belt, the layout, the scanner's record, the contact watch and every scoring
level, and that the demo is reachable, stays clear of the other parcels, the right arm and the scanner, and merges
the parcel into the gap with wide margins (replayed with the parcels riding the belt, standing on the infeed table or
held in the fingers, and the arms' collision hulls placed by forward kinematics along the plan).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.analysis import fk_link_poses_world
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
RUN_LIMIT_S = 600 * DT  # the run limit: 600 control steps
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")
SQUARE = np.array([0.0, 0.0, 0.0, 1.0])


def yawed(deg):
    return np.array([0.0, 0.0, np.sin(np.radians(deg) / 2), np.cos(np.radians(deg) / 2)])


def state(pos, quat=SQUARE):
    """A parcel at ``pos``, its world-aligned bounding box grown by a turn about the vertical as the simulator's is."""
    pos = np.asarray(pos, float)
    turn, _ = M.turn_and_tilt(quat)
    c, s = np.cos(np.radians(turn)), np.sin(np.radians(turn))
    half = np.array([M.PARCEL[0] * c + M.PARCEL[1] * s, M.PARCEL[0] * s + M.PARCEL[1] * c, M.PARCEL[2]]) / 2
    return ObjectState(pos, np.asarray(quat, float), np.zeros(3), np.zeros(3), pos - half, pos + half)


def merged(offset=0.0, t0=6.0, x=M.LINE_X, quat=SQUARE, shifts=None, xs=None, quats=None, ratio=1.0):
    """A function of t giving each parcel's state: the train rides the belt from the start (each moved
    ``shifts[name]`` along it, ``xs[name]`` across it, turned ``quats[name]``); the merge parcel stands on the infeed
    table until ``t0``, then rides the belt from the gap's middle plus ``offset``."""
    shifts, xs, quats = shifts or {}, xs or {}, quats or {}

    def at(t):
        run = ratio * M.RUN(t)[0]
        out = {n: state((xs.get(n, M.LINE_X), M.START[n] + shifts.get(n, 0.0) + run, M.PARCEL_Z), quats.get(n, SQUARE))
               for n in M.TRAIN}
        out[M.MERGE] = (state((*M.INFEED_XY, M.PARCEL_Z)) if t < t0
                        else state((x, M.GAP0 + run + offset, M.PARCEL_Z), quat))
        return out

    return at


def arm_poses(side, q, finger=FINGER_OPEN):
    """An arm's link poses (4x4, station frame) at joints ``q`` and finger joints ``finger``, as the simulator
    reports them to the contact watch."""
    named = {j: float(v) for j, v in zip(ARM_JOINTS, q, strict=True)}
    named["joint7"] = named["joint8"] = float(finger)
    return fk_link_poses_world(named, station.BASES[side])


RESTING = {side: arm_poses(side, READY) for side in ("left", "right")}  # both arms in their ready pose


def reaching(pos, yaw=M.YAW, finger=FINGER_OPEN):
    """The left arm with its TCP at ``pos``, pointing down, turned ``yaw``."""
    q, pe, _ = ik.solve(np.asarray(pos, float), ik.matrix_to_quat_xyzw(grip_rotation(yaw)), np.array(READY, float),
                        base_pos=station.BASES["left"], iters=200)
    assert pe < 1e-3
    return {"left": arm_poses("left", q, finger), "right": RESTING["right"]}


def play(at, until=RUN_LIMIT_S, arms=None):
    """Run the scanner's record and the contact watch over the control steps of a trial, the parcels as ``at(t)``
    gives them and the arms' link poses as ``arms(t)`` does (default: both at rest); return the final (score, why)."""
    M.reset_state()
    for k in range(round(until / DT) + 1):
        t = k * DT
        s = at(t)
        M.update(t, s)
        M.watch_contacts(t, s, RESTING if arms is None else arms(t))
    return M.result()


def _with(base, t0, t1, name, pos):
    """``base``, except that parcel ``name`` is at ``pos(t)`` from ``t0`` to ``t1``."""

    def at(t):
        out = base(t)
        if t0 <= t <= t1:
            out[name] = state(pos(t))
        return out

    return at


# -- loading and layout ----------------------------------------------------------------------------------------------


def test_it_loads_with_five_free_parcels_and_a_belt():
    assert TASK.free_objects == M.NAMES
    assert all(isinstance(o, M.AwakeBox) for o in TASK.objects if o.name in M.NAMES)  # they never sleep
    assert "belt" in TASK.names and "infeed_table" in TASK.names and TASK.score_over == "end"


def test_positions_and_sizes_are_plain_floats():
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name
        assert all(type(v) is float for v in o.size), o.name


def test_the_line_has_one_24_cm_gap_and_6_cm_between_the_others():
    ys = [M.START[n] for n in M.TRAIN]
    assert -np.diff(ys) - M.PARCEL[1] == pytest.approx([M.SPACING, M.GAP, M.SPACING])
    assert ys[-1] - M.PARCEL[1] / 2 == pytest.approx(M.REAR_Y0) and M.REAR_Y0 - M.START_Y >= 0.04
    assert M.GAP - M.PARCEL[1] - 2 * M.CLEAR == pytest.approx(0.07)  # 3.5 cm of slack either way when centred
    assert M.SPACING > M.CLEAR + 0.015  # the line itself is spaced for the scanner, with room to spare
    for p in M.PARCELS:
        assert p.pos[2] - M.PARCEL[2] / 2 == pytest.approx(M.BELT_TOP)
        assert M.on_belt(state(p.pos)) is None


def test_the_infeed_table_is_level_with_the_belt_beside_it_and_holds_the_parcel():
    assert M.TOP + M.INFEED[2] == pytest.approx(M.BELT_TOP)
    far_edge = M.INFEED_XY[0] + M.INFEED[0] / 2
    assert 0.0 < M.BED_NEAR - far_edge <= 0.02  # a 1.5 cm slot to the conveyor's bed
    p = M.MERGE_PARCEL
    assert p.pos[2] - M.PARCEL[2] / 2 == pytest.approx(M.BELT_TOP)
    for i in (0, 1):  # the parcel stands 3 cm or more inside the table's edges
        assert M.INFEED[i] / 2 - M.PARCEL[i] / 2 >= 0.03
    assert M.on_belt(state(p.pos)) is not None  # not on the belt: off its near edge


def test_the_belt_floats_over_its_bed_and_the_arch_clears_the_parcels():
    assert M.BELT_Z - M.BELT_T / 2 - M.BED_TOP == pytest.approx(0.002)
    assert M.HEAD_LOW - M.MARK - M.PARCEL_TOP > 0.07
    assert M.POST_IN - M.BELT_W / 2 >= 0.015
    assert M.END_Y - 5 * M.PARCEL[1] - M.SCAN_Y > 0.12  # all five queue against the end stop, 12 cm past the line
    assert M.END_Y - 5 * (M.PARCEL[1] * np.cos(np.radians(30)) + M.PARCEL[0] / 2) - M.SCAN_Y > 0.03  # even all turned 30°
    assert M.START_Y > -M.TABLE_EDGE_Y and M.SCAN_Y + M.HEAD[1] / 2 < M.TABLE_EDGE_Y
    assert M.INFEED_XY[1] + M.INFEED[1] / 2 < M.SCAN_Y - 0.2  # the infeed table is well short of the arch


def test_the_belt_ramps_up_smoothly_then_runs_at_6_cm_per_s():
    ts = np.arange(0.0, 30.0, 1 / 240)
    run = np.array([M.RUN(t) for t in ts])
    assert run[0, 0] == 0.0 and run[0, 1] == 0.0
    assert np.all(np.diff(run[:, 0]) >= 0)
    assert np.max(np.abs(np.diff(run[:, 1]))) * 240 < 0.2  # peak acceleration 0.19 m/s², far below sliding
    assert np.allclose(run[ts >= M.RAMP_S, 1], M.SPEED)
    # driven faster than the belt's speed by the measured carry, so the parcels ride at the belt's speed
    assert M.BELT.surface_velocity(5.0, 1 / 240) == pytest.approx([0.0, M.SPEED / M.CARRY, 0.0])
    assert M.BELT.carry == M.CARRY == pytest.approx(0.845)


def test_the_gap_passes_in_front_of_the_infeed_table_and_the_line_passes_within_the_run_limit():
    t_abreast = (M.INFEED_XY[1] - M.GAP0) / M.SPEED + M.RAMP_S / 2
    assert 4.5 < t_abreast < 6.5
    last_gone = (M.SCAN_Y - M.REAR_Y0) / M.SPEED + M.RAMP_S / 2
    assert last_gone == pytest.approx(18.58, abs=0.01)
    assert last_gone + 1.0 < RUN_LIMIT_S - 10.0  # success by 19.6 s of a 30 s limit


def test_turn_and_tilt_read_a_parcels_orientation():
    assert M.turn_and_tilt(SQUARE) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert M.turn_and_tilt(yawed(20))[0] == pytest.approx(20.0)
    assert M.turn_and_tilt(yawed(180))[0] == pytest.approx(0.0, abs=1e-6)  # either way round is square
    assert M.turn_and_tilt(yawed(-170))[0] == pytest.approx(10.0)
    assert M.turn_and_tilt(yawed(90))[0] == pytest.approx(90.0)
    on_side = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])  # rolled 90 degrees about the belt's line
    assert M.turn_and_tilt(on_side) == pytest.approx((0.0, 90.0), abs=1e-6)


# -- the scanner's record and the score -------------------------------------------------------------------------------


def test_doing_nothing_scores_0():
    score, why = play(merged(t0=99.0))
    assert score == 0.0 and "merge_parcel has not passed" in why


def test_centred_in_the_gap_square_scores_1_with_6_5_cm_either_side():
    score, why = play(merged())
    assert score == 1.0, why
    assert M.clear_between("parcel_2", M.MERGE) == pytest.approx(0.075, abs=5e-4)
    assert M.clear_between(M.MERGE, "parcel_3") == pytest.approx(0.075, abs=5e-4)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (dict(offset=0.034), 1.0),  # 4.1 cm ahead: still enough
        (dict(offset=0.04), 2 / 3),  # 3.5 cm ahead: short in front
        (dict(offset=-0.04), 2 / 3),  # 3.5 cm behind: short behind
        (dict(quat=yawed(12)), 1.0),  # turned 12 degrees: square enough
        (dict(quat=yawed(25)), 2 / 3),  # turned 25 degrees
        (dict(quat=yawed(25), offset=0.045), 1 / 3),  # turned, and 3 cm ahead: only clear behind is met
        (dict(quat=yawed(25), offset=-0.045), 1 / 3),  # turned, and 3 cm behind: only clear ahead is met
        (dict(quat=yawed(30), offset=0.055), 1 / 3),  # turned 30 degrees, 1.1 cm short of parcel_2 (not touching)
        (dict(shifts={"parcel_3": -0.01}), 1.0),  # parcel_3 1 cm off where it started (belt slip): within tolerance
    ],
)
def test_a_third_for_each_condition_once_the_parcel_passes_in_the_gap_untouching(kwargs, expected):
    score, why = play(merged(**kwargs))
    assert score == pytest.approx(expected), why
    assert "touch" not in why.split("; no touch")[0]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(shifts={"parcel_3": -0.02}),  # parcel_3 2 cm back: something pushed it
        dict(xs={"parcel_1": M.LINE_X + 0.02}),  # parcel_1 pushed 2 cm across the belt
        dict(quats={"parcel_2": yawed(8)}),  # parcel_2 turned 8 degrees
        dict(shifts={"parcel_2": 0.04, "parcel_3": -0.04}),  # the gap widened 8 cm to make room
    ],
)
def test_a_disturbed_line_scores_0_even_when_no_touch_was_seen(kwargs):
    # what a touch between two control steps would leave behind
    score, why = play(merged(**kwargs))
    assert score == 0.0 and "disturbed" in why, why


def test_the_clearances_are_measured_to_within_a_millimetre_between_control_steps():
    play(merged(offset=0.0173))
    assert M.clear_between("parcel_2", M.MERGE) == pytest.approx(0.075 - 0.0173, abs=5e-4)
    assert M.clear_between(M.MERGE, "parcel_3") == pytest.approx(0.075 + 0.0173, abs=5e-4)


@pytest.mark.parametrize(
    "offset",
    [
        -M.GAP / 2 - 0.10,  # behind parcel_3, between it and parcel_4: they overlap at the line
        -(M.GAP0 - M.START["parcel_4"]) - 0.20,  # behind the whole line, with plenty of room: not in the gap
        M.START["parcel_1"] + 0.20 - M.GAP0,  # ahead of the whole line
    ],
)
def test_a_parcel_merged_anywhere_but_the_gap_scores_0(offset):
    score, why = play(merged(offset=offset, t0=1.0))
    assert score == 0.0 and ("gap" in why or "has not passed" in why or "touched" in why), why


def test_a_merged_parcel_on_top_of_another_or_over_the_edge_or_carried_through_fails():
    on_top = _with(merged(), 0.0, 99.0, M.MERGE,
                   lambda t: (M.LINE_X, M.riding_y("parcel_2", t), M.PARCEL_Z + M.PARCEL[2]))
    assert play(on_top)[0] == 0.0
    score, why = play(merged(x=M.BELT_FAR - M.PARCEL[0] / 2 + 0.005))
    assert score == 0.0 and "edge" in why
    assert play(merged(x=M.BELT_FAR - M.PARCEL[0] / 2 - 0.001))[0] == 1.0  # just inside the edge is fine
    held = _with(merged(), 6.0, 99.0, M.MERGE, lambda t: (M.LINE_X, M.gap_y(t), M.PARCEL_Z + 0.03))
    score, why = play(held)
    assert score == 0.0 and "not lying on the belt" in why


def test_a_train_parcel_knocked_off_the_belt_fails_the_trial():
    score, why = play(_with(merged(), 7.0, 99.0, "parcel_2", lambda _: (0.15, 0.2, M.TOP + M.PARCEL[2] / 2)))
    assert score == 0.0 and "parcel_2" in why


def test_a_merged_parcel_tipped_onto_its_side_is_not_square():
    on_side = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
    score, why = play(merged(quat=on_side))
    assert score == pytest.approx(2 / 3), why


# -- touching the line's parcels: 0, whenever it happens ---------------------------------------------------------------


def test_the_gaps_measure_boxes_and_hulls():
    a = state((M.LINE_X, 0.0, M.PARCEL_Z))
    assert M.parcel_gap(a, state((M.LINE_X, M.PARCEL[1] + 0.02, M.PARCEL_Z))) == pytest.approx(0.02)
    assert M.parcel_gap(a, state((M.LINE_X, M.PARCEL[1] - 0.01, M.PARCEL_Z))) == pytest.approx(-0.01)
    corner_in = state((M.LINE_X + M.PARCEL[0] / 2 + 0.03, 0.0, M.PARCEL_Z), yawed(45))  # a turned box's edge
    assert M.parcel_gap(a, corner_in) < 0.0
    # the closed fingertips: pad tips 1 cm into the top of a parcel, then 2 cm above it
    from yam_arena.analysis import world_hulls
    tips = (M.LINE_X, 0.0, M.PARCEL_TOP - 0.01 + PAD_TIP_PAST_TCP)
    d, link = M.arm_gap(world_hulls(reaching(tips, finger=M.FINGER_CLOSED)["left"]), a)
    assert d < 0.0 and link in ("tip_left", "tip_right")
    d, _ = M.arm_gap(world_hulls(reaching((tips[0], tips[1], tips[2] + 0.03), finger=M.FINGER_CLOSED)["left"]), a)
    assert d == pytest.approx(0.02, abs=0.003)
    # both arms at rest are far from the line
    assert all(M.arm_gap(world_hulls(RESTING[s]), a)[0] > 0.10 for s in RESTING)


def test_pushing_a_neighbour_back_to_widen_the_gap_then_merging_scores_0():
    # the reviewer's cheat: the fingers shove parcel_3 back 1 cm (within the spacing tolerance) at 5 to 6 s, then the
    # parcel is merged into the widened gap perfectly; the arm touched parcel_3, so 0
    def arms(t):
        if 5.0 <= t <= 6.0:
            face = M.riding_y("parcel_3", t, -0.01) + M.PARCEL[1] / 2
            return reaching((M.LINE_X, face + 0.004, M.PARCEL_Z + PAD_TIP_PAST_TCP), yaw=0.0)
        return RESTING

    base = merged(shifts={"parcel_3": -0.01})
    assert play(base)[0] == 1.0  # without the touch it would score 1
    score, why = play(base, arms=arms)
    assert score == 0.0 and "left arm" in why and "parcel_3" in why, why


def test_brushing_a_parcel_with_the_arm_once_scores_0():
    # a single control step with the gripper on parcel_1's top, nothing moved
    def arms(t):
        if abs(t - 3.0) < 1e-9:
            return reaching((M.LINE_X, M.riding_y("parcel_1", t), M.PARCEL_TOP + PAD_TIP_PAST_TCP + 0.002),
                            finger=M.FINGER_CLOSED)
        return RESTING

    score, why = play(merged(), arms=arms)
    assert score == 0.0 and "parcel_1" in why and "3.00 s" in why, why


def test_hovering_close_over_a_parcel_without_touching_is_allowed():
    def arms(t):
        if 3.0 <= t <= 4.0:
            return reaching((M.LINE_X, M.riding_y("parcel_1", t), M.PARCEL_TOP + PAD_TIP_PAST_TCP + 0.015),
                            finger=M.FINGER_CLOSED)
        return RESTING

    score, why = play(merged(), arms=arms)
    assert score == 1.0, why


def test_the_merged_parcel_bumping_a_neighbour_scores_0_even_if_the_line_is_unchanged():
    # set down against parcel_2's trailing face for 0.3 s, then slid back to the gap's middle: spacing all fine at the
    # scanner, but it touched parcel_2
    def bump(t):
        return (M.LINE_X, M.riding_y("parcel_2", t) - M.PARCEL[1] - 0.002, M.PARCEL_Z)

    at = _with(merged(), 6.0, 6.3, M.MERGE, bump)
    score, why = play(at)
    assert score == 0.0 and "merged parcel touched parcel_2" in why, why
    clear = _with(merged(), 6.0, 6.3, M.MERGE, lambda t: (bump(t)[0], bump(t)[1] - 0.01, bump(t)[2]))
    assert play(clear)[0] == 1.0  # 1.2 cm away instead: no touch


def test_a_touch_is_kept_for_the_whole_trial():
    at = _with(merged(), 6.0, 6.0, M.MERGE, lambda t: (M.LINE_X, M.riding_y("parcel_3", t), M.PARCEL_Z + 0.04))
    M.reset_state()
    for k in range(round(RUN_LIMIT_S / DT) + 1):
        t = k * DT
        s = at(t)
        M.update(t, s)
        M.watch_contacts(t, s, RESTING)
        if t > 6.0:
            assert M.result()[0] == 0.0, t
    assert "6.00 s" in M.result()[1]


def _pass_times():
    """When the merged parcel (riding the gap's middle) reaches the scan line and when it has left it."""
    play(merged())
    return M._STATE[M.MERGE]["front_t"], M._STATE[M.MERGE]["rear_t"]


@pytest.mark.parametrize("window", ["whole pass", "last moment"])
def test_holding_the_merged_parcel_through_the_scanner_scores_0(window):
    # the gripper keeps hold of it (pads closed on it, sliding it along the belt with the gap) into its pass; it lies
    # on the belt, square, clear on both sides, but it was steered, not let go
    front, rear = _pass_times()
    t0, t1 = (front - 1.0, rear) if window == "whole pass" else (front - 1.0, front + 0.05)

    def arms(t):
        if t0 <= t <= t1:
            return reaching((M.LINE_X, M.gap_y(t), M.GRASP_Z), finger=M.FINGER_CLOSED)
        return RESTING

    score, why = play(merged(), arms=arms)
    assert score == 0.0 and "merged parcel under the scanner" in why, why


def test_letting_go_just_before_the_scanner_is_allowed():
    # holding it until 0.1 s before it reaches the line, then the gripper opens and rises 2 cm clear: no touch in the
    # pass. Touching it before the pass (setting it down) is how it gets there.
    front, rear = _pass_times()

    def arms(t):
        if front - 1.0 <= t <= front - 0.1:
            return reaching((M.LINE_X, M.gap_y(t), M.GRASP_Z), finger=M.FINGER_CLOSED)
        if front - 0.1 < t <= rear:
            return reaching((M.LINE_X, M.gap_y(t), M.PARCEL_TOP + PAD_TIP_PAST_TCP + 0.02))
        return RESTING

    score, why = play(merged(), arms=arms)
    assert score == 1.0, why
    assert 0.01 < M._STATE["closest"]["held"] < 0.03, why


def test_parcels_queueing_against_the_end_stop_after_the_scanner_may_touch():
    # once both have passed the scan line, the merged parcel may close up on parcel_2 (as they gather at the stop)
    def at(t):
        out = merged()(t)
        m = out[M.MERGE]
        p2 = out["parcel_2"]
        if m.low[1] > M.SCAN_Y + 0.01:  # past the line: closed up on parcel_2
            out[M.MERGE] = state((M.LINE_X, p2.pos[1] - M.PARCEL[1], M.PARCEL_Z))
        return out

    score, why = play(at)
    assert score == 1.0, why


def test_a_new_trial_starts_with_an_empty_record():
    play(merged())
    assert M.success({})
    M.reset_state()
    assert M.score({}) == 0.0 and not M.success({})


# -- the demo ----------------------------------------------------------------------------------------------------------


def _plan():
    return ArmPlan(M.demo()[M.ARM], station.BASES[M.ARM])


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        a, _ = spans.get(w.label, (t, t))
        spans[w.label] = (a, t + w.duration)
        t += w.duration
    return spans


def test_the_demo_uses_the_left_arm_and_is_reachable():
    assert list(M.demo()) == ["left"]
    plan = _plan()
    actions = np.array([plan.action7((k + 1) * DT)[0] for k in range(round(plan.duration / DT) + 5)])
    assert plan.worst_ik_mm < 0.5 and plan.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_demo_runs_until_after_success_and_within_the_run_limit():
    # yam_scripted stops the rollout 3 s after the plan ends: the plan must outlast the last parcel's pass + 1 s
    last_gone = (M.SCAN_Y - M.REAR_Y0) / M.SPEED + M.RAMP_S / 2
    assert last_gone + 1.0 + 1.0 < _plan().duration < RUN_LIMIT_S - 3.0


def test_the_demo_grips_across_the_belt_clear_of_the_table_and_carries_over_the_others():
    assert all(abs(abs(w.yaw) - np.pi / 2) < 1e-9 for w in M.demo()[M.ARM])  # fingers close along x
    assert M.PARCEL[0] + 0.015 * 2 < 2 * abs(FINGER_OPEN)  # 1.7 cm either side between open pads and the parcel
    assert M.GRASP_Z - PAD_TIP_PAST_TCP - M.BELT_TOP >= 0.015  # pad tips stop 1.5 cm or more above table / belt
    assert M.PRE_Z - PAD_TIP_PAST_TCP - M.PARCEL_TOP == pytest.approx(0.03)
    carried_low = M.CARRY_Z - (M.GRASP_Z - M.PARCEL_Z) - M.PARCEL[2] / 2
    assert carried_low - M.PARCEL_TOP >= 0.015  # the carried parcel's underside 1.5 cm over the others' tops
    # the open pads come down on the infeed table clear of the conveyor's bed (2.8 cm short of its edge)
    assert M.INFEED_XY[0] + abs(FINGER_OPEN) < M.BED_NEAR - 0.02
    # held up to 1.7 cm off the TCP across the belt (the finger pair slides as one), still 3.3 cm inside its edges
    assert M.BELT_W / 2 - M.PARCEL[0] / 2 - (abs(FINGER_OPEN) - M.PARCEL[0] / 2) > 0.03


def test_the_demo_follows_the_gap_while_setting_the_parcel_down():
    t = 0.0
    for w in M.demo()[M.ARM]:
        t += w.duration
        if w.label in ("set down", "hold it on the belt", "let go", "let the fingers open", "up"):
            assert w.pos[:2] == pytest.approx(M.place_xy(t)), (w.label, t)
    spans = _spans(M.demo()[M.ARM])
    assert spans["carry over the gap"][1] == pytest.approx(spans["set down"][0])


def test_the_demo_gives_the_slow_fingers_time_and_carries_gently():
    # measured in Isaac Sim: about 1 s for the fingers to close from fully open, and as long to open
    plan = M.demo()[M.ARM]
    spans = _spans(plan)
    assert spans["let the fingers close"][1] - spans["close"][1] >= 1.3
    assert spans["let the fingers open"][1] - spans["let go"][1] >= 1.0
    for label in ("let the fingers close", "lift"):
        assert all(w.finger == M.FINGER_CLOSED for w in plan if w.label == label)
    # peak acceleration of the TCP (smoothstep: 6 d / T^2) while the parcel is held
    assert 6 * (M.CARRY_Z - M.GRASP_Z) / M.LIFT_S**2 < 0.5
    assert 6 * np.hypot(M.LINE_X - M.INFEED_XY[0], 0.03) / M.CARRY_S**2 < 0.5


def test_the_demo_sets_the_parcel_on_the_belt_and_holds_it_there_before_opening():
    plan = M.demo()[M.ARM]
    spans = _spans(plan)
    underside = M.PLACE_Z - (M.GRASP_Z - M.PARCEL_Z) - M.PARCEL[2] / 2 - M.BELT_TOP
    assert 0.0 <= underside <= 0.005
    assert M.PLACE_Z - PAD_TIP_PAST_TCP - M.BELT_TOP >= 0.015  # the pad tips still 1.5 cm above the belt
    assert spans["hold it on the belt"][1] - spans["hold it on the belt"][0] >= 0.3 - 1e-9
    assert spans["set down"][1] - spans["set down"][0] >= 1.2 - 1e-9  # slow: peak 0.25 m/s² down
    held = [w for w in plan if w.label in ("set down", "hold it on the belt")]
    assert all(w.finger == M.FINGER_CLOSED for w in held)


def _replay(ratio=1.0):
    """The demo replayed: the train rides the belt at ``ratio`` times its speed; the merge parcel stands on the
    infeed table until the gripper has closed, hangs in the fingers until the middle of letting go, then rides the
    belt (at the same ratio) from where it was set down."""
    plan = _plan()
    spans = _spans(M.demo()[M.ARM])
    held = (spans["let the fingers close"][1], sum(spans["let go"]) / 2)
    offset = np.array([0.0, 0.0, M.PARCEL_Z - M.GRASP_Z])
    let_go = {}

    def at(t):
        out = {n: state((M.LINE_X, M.START[n] + ratio * M.RUN(t)[0], M.PARCEL_Z)) for n in M.TRAIN}
        if t < held[0]:
            out[M.MERGE] = state((*M.INFEED_XY, M.PARCEL_Z))
        elif t <= held[1]:
            tcp = plan.target(t)[0]
            out[M.MERGE] = state(tcp + offset)
            let_go["y"] = (t, tcp[1])
        else:
            t0, y0 = let_go["y"]
            out[M.MERGE] = state((M.LINE_X, y0 + ratio * (M.RUN(t)[0] - M.RUN(t0)[0]), M.PARCEL_Z))
        return out

    return at


def _demo_arms():
    """The arms' link poses along the demo plan (forward kinematics of its IK, fingers as commanded)."""
    plan, cache = _plan(), {}

    def arms(t):
        k = round(t / DT)
        if k not in cache:
            if k == 0:
                q, finger = np.array(READY, float), FINGER_OPEN
            else:
                grip = plan.action7(min(t, plan.duration))[0][6]
                q, finger = plan.q.copy(), grip * FINGER_OPEN
            cache[k] = {"left": arm_poses("left", q, finger), "right": RESTING["right"]}
        return cache[k]

    return arms


@pytest.mark.parametrize("ratio", [1.0, 0.93, 1.07])
def test_the_demo_never_comes_near_touching_the_line(ratio):
    # replayed with the arms' collision hulls along the plan: neither arm comes within 4 cm of the line's parcels and
    # the merged parcel stays 3 cm or more from them until past the scanner, even with the line riding 7 % off speed
    score, why = play(_replay(ratio), arms=_demo_arms())
    assert score == 1.0, why
    assert M._STATE["touch"] is None
    assert M._STATE["closest"]["arms"] > 0.04, why
    assert M._STATE["closest"]["merged"] > (0.05 if ratio == 1.0 else 0.03), why
    assert M._STATE["closest"]["held"] > 0.10, why  # parked well away while the merged parcel passes


def test_the_demo_replayed_merges_the_parcel_with_wide_margins():
    score, why = play(_replay())
    assert score == 1.0, why
    ahead, behind = M.clear_between("parcel_2", M.MERGE), M.clear_between(M.MERGE, "parcel_3")
    assert ahead == pytest.approx(0.075, abs=1e-3) and behind == pytest.approx(0.075, abs=1e-3), why  # 3.5 cm slack
    assert M.order() == ["parcel_1", "parcel_2", M.MERGE, "parcel_3", "parcel_4"]


@pytest.mark.parametrize("ratio", [0.93, 1.07])
def test_the_demo_still_scores_1_if_the_parcels_ride_7_percent_off_the_belt_speed(ratio):
    # the belt's carry is a measured correction; a residual 7 % slip or overshoot moves the gap 3.2 cm by the drop
    score, why = play(_replay(ratio))
    assert score == 1.0, why


@pytest.mark.parametrize("ratio", [0.81, 0.9])
def test_scoring_measures_lengths_so_slow_riding_parcels_are_judged_by_their_real_spacing(ratio):
    # the whole line riding slow (the uncorrected belt model rode at 0.81): untouched, the spacing is unchanged
    score, why = play(merged(ratio=ratio))
    assert score == 1.0, why
    assert M.clear_between("parcel_1", "parcel_2") == pytest.approx(M.SPACING, abs=5e-4)
    assert M.clear_between("parcel_2", "parcel_3") == pytest.approx(M.GAP, abs=5e-4)


def test_the_merged_parcel_never_overlaps_another_and_stays_inside_the_gap_over_the_belt():
    at, plan = _replay(), _plan()
    for k in range(round(plan.duration / DT)):
        t = k * DT
        s = at(t)
        m = s[M.MERGE]
        for n in M.TRAIN:
            lo, hi = np.maximum(m.low, s[n].low), np.minimum(m.high, s[n].high)
            assert np.any(hi - lo < 0.001), (t, n)
        if m.high[0] > M.BELT_NEAR and t < 12.0:  # over the belt, before it reaches the line
            assert m.low[1] >= s["parcel_3"].high[1] + 0.02, t
            assert m.high[1] <= s["parcel_2"].low[1] - 0.02, t


def test_the_merged_parcel_is_let_go_well_before_the_line_reaches_the_arch():
    spans = _spans(M.demo()[M.ARM])
    play(_replay())
    assert spans["let go"][1] + 2.0 < M._STATE[M.MERGE]["front_t"]


def _links(arm, q):
    base = np.asarray(station.BASES[arm])
    fk = forward_kinematics(YAM_URDF, {j: float(v) for j, v in zip(ARM_JOINTS, q, strict=True)})
    tcp, _ = ik.tcp_pose(q, base)
    pts = [fk[n][:3, 3] + base for n in CHAIN] + [tcp]
    return np.array([a + s * (b - a) for a, b in zip(pts[:-1], pts[1:]) for s in np.linspace(0, 1, 8)])


def test_the_left_arm_stays_clear_of_the_right_arm_and_the_scanner_arch():
    plan = _plan()
    right = _links("right", np.array(READY, float))
    near_face = M.SCAN_Y - M.HEAD[1] / 2
    closest = np.inf
    for k in range(round(plan.duration / DT)):
        plan.action7((k + 1) * DT)
        pts = _links("left", plan.q)
        closest = min(closest, float(np.min(np.linalg.norm(pts[:, None] - right[None], axis=2))))
        assert np.max(pts[:, 1]) < near_face - 0.10, k  # every link and the TCP 10 cm short of the arch
    assert closest > 0.25


def test_the_left_arm_stays_above_the_line_parcels_while_they_pass_under_it():
    # whenever the TCP is over the belt, it is over the gap (inside it by 2 cm or more) or 3 cm above the parcels' tops
    plan, at = _plan(), _replay()
    for k in range(round(plan.duration / DT)):
        t = (k + 1) * DT
        tcp = plan.target(t)[0]
        if abs(tcp[0] - M.LINE_X) > M.BELT_W / 2 + 0.05:
            continue
        s = at(t)
        in_gap = s["parcel_3"].high[1] + 0.02 < tcp[1] < s["parcel_2"].low[1] - 0.02
        assert in_gap or tcp[2] - PAD_TIP_PAST_TCP > M.PARCEL_TOP + 0.03, t


def test_the_parcels_are_spawned_with_sleeping_disabled():
    # a stand-in for yam_arena.scene: its task_object builds the rigid body config from the module's global
    import types

    import motionkit

    fake = types.ModuleType("fake_scene")
    exec("def PhysxRigidBodyPropertiesCfg(**k):\n    return dict(k)\n"
         "def task_object(spec):\n    return PhysxRigidBodyPropertiesCfg()\n"
         "def object_states(env, names):\n    return {}\n", fake.__dict__)
    motionkit._patch_scene(fake)
    assert fake.task_object(M.MERGE_PARCEL) == {"sleep_threshold": 0.0}
    assert fake.task_object(M.INFEED_TABLE) == {}  # everything else as yam_arena makes it
    assert fake.PhysxRigidBodyPropertiesCfg() == {}  # put back afterwards
