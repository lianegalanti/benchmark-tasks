"""Offline checks (no Isaac Sim): the cup's path, the handover monitor, the scoring, and that the demo is reachable and
hands the part over gently.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
CUP_PARTS = ("cup_floor", "cup_wall_near", "cup_wall_far", "cup_wall_right", "cup_wall_left")
PART_SIZE = (2 * M.RADIUS, 2 * M.RADIUS, M.HEIGHT)
OPEN_GAP = 0.094
HELD_GAP = 0.040
FAR = np.array([0.0, 0.6, 1.2])  # the idle arm's TCP, well away


def state(pos, size=PART_SIZE, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.asarray(spin, float),
                       pos - half, pos + half)


def cup_at(t):
    pos, vel = M.PATH(t)
    return state(pos, size=(M.CUP_IN, M.CUP_IN, M.FLOOR), vel=vel)


def objects(part, t):
    return {"part": part, "cup_floor": cup_at(t)}


def part_under(tcp, vel=(0.0, 0.0, 0.0)):
    """The part held from the top, as the demo grips it."""
    return state(np.asarray(tcp) - (0, 0, M.HEIGHT / 2 - M.GRIP_BELOW_TOP), vel=vel)


def run(t0, t1, tcp_of, gap_of, part_of):
    """Feed the monitor control steps from episode time t0 to t1: tcp_of(t), gap_of(t), part_of(t, tcp, gap) give
    the right arm's TCP, pad gap and the part's state."""
    for t in np.arange(t0, t1 + 1e-9, DT):
        tcp, gap = np.asarray(tcp_of(t), float), gap_of(t)
        M.update(float(t), part_of(t, tcp, gap), cup_at(t), {"left": (FAR, OPEN_GAP), "right": (tcp, gap)})


def handover(release_height, lag_speed=0.0, open_at=9.2, t_end=10.2, over=(0.0, 0.0)):
    """Ride over the cup with the part held, its base ``release_height`` above the floor, open at ``open_at``. The
    gripper drifts against the cup at ``lag_speed`` (m/s, along x). After release the part falls onto the floor and
    rides along. Episode times: the cup is moving at release (9.2 s) and at the end (10.2 s)."""
    M.reset_state()
    M._STATE["lifted"] = True  # carried over from the table

    def tcp_of(t):
        x, y, _ = M.PATH(t)[0]
        z = M.FLOOR_TOP + release_height + M.HEIGHT - M.GRIP_BELOW_TOP
        return (x + over[0] + lag_speed * (t - open_at), y + over[1], z)

    def part_of(t, tcp, gap):
        if gap < OPEN_GAP - 1e-9:
            return part_under(tcp, vel=M.PATH(t)[1])
        x, y, _ = tcp
        return state((x, y, M.FLOOR_TOP + M.HEIGHT / 2), vel=M.PATH(t)[1])

    run(open_at - 1.0, t_end, tcp_of, lambda t: HELD_GAP if t < open_at else OPEN_GAP, part_of)
    return t_end


# -- the cup -------------------------------------------------------------------------------------------------------


def test_it_loads_with_the_cup_driven_and_only_the_part_free():
    assert TASK.free_objects == ("part",)
    assert set(CUP_PARTS) <= set(TASK.names) and TASK.score_over == "end"


def test_the_cup_wanders_slowly_with_brief_stops():
    ts = np.arange(0.0, 2 * M.CYCLE_S, 1 / 240)
    ps = np.array([M.PATH(t)[0] for t in ts])
    vs = np.array([M.PATH(t)[1] for t in ts])
    assert np.allclose(ps[:, 2], M.FLOOR_Z)
    speed = np.linalg.norm(vs, axis=1)
    assert 0.05 < speed.max() < 0.065  # about 5 to 6 cm/s at most
    for i in range(2):
        assert np.allclose(np.gradient(ps[:, i], ts)[5:-5], vs[5:-5, i], atol=2e-3)  # reported velocity is the motion
    accel = np.linalg.norm(np.gradient(vs, ts, axis=0)[5:-5], axis=1)
    assert accel.max() < 0.1  # m/s^2: a standing part neither slides nor tips in the cup
    # stopped (under 5 mm/s) about 2 s per 16.8 s loop, and never for longer than 1.2 s at a time
    still = speed < 0.005
    assert 0.08 < still.mean() < 0.2
    runs, n = [], 0
    for s in still:
        n = n + 1 if s else 0
        runs.append(n)
    assert max(runs) / 240 < 1.2
    for (xy, _, _), t in zip(M.LEGS, np.cumsum([tr + pa for _, tr, pa in M.LEGS])):
        assert M.PATH(t - 1e-6)[0][:2] == pytest.approx(xy, abs=1e-6)
    assert M.PATH(M.CYCLE_S)[0][:2] == pytest.approx(M.START_XY, abs=1e-9)


def test_the_cup_parts_move_as_one_without_touching():
    specs = {o.name: o for o in TASK.objects}
    for t in (0.0, 1.3, 4.3, 8.6, 15.0):
        floor = M.PATH(t)[0]
        for name in CUP_PARTS:
            pos, vel = specs[name].path(t)
            assert np.allclose(vel, M.PATH(t)[1])
            assert np.allclose(pos - floor, np.subtract(specs[name].pos, specs["cup_floor"].pos))
    boxes = [(np.asarray(specs[n].pos) - np.asarray(specs[n].size) / 2,
              np.asarray(specs[n].pos) + np.asarray(specs[n].size) / 2) for n in CUP_PARTS]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (lo1, hi1), (lo2, hi2) = boxes[i], boxes[j]
            assert np.any((lo1 >= hi2 - 1e-9) | (lo2 >= hi1 - 1e-9)), (CUP_PARTS[i], CUP_PARTS[j])
    assert specs["cup_floor"].pos[2] - M.FLOOR / 2 == pytest.approx(M.TOP + 0.002)
    assert all(isinstance(v, float) for o in TASK.objects for v in o.pos)  # plain floats for Isaac Lab


def test_the_cup_stays_in_front_of_the_right_arm_and_clear_of_the_part():
    base = np.asarray(station.BASES["right"])
    out = M.CUP_IN / 2 + M.GAP + M.WALL
    for t in np.arange(0.0, M.CYCLE_S, 0.05):
        x, y, _ = M.PATH(t)[0]
        assert np.hypot(x - base[0], y - base[1]) < 0.38  # within the arm's reach
        gap_x = max(x - out - (M.PART_XY[0] + M.RADIUS), 0.0)
        gap_y = max(M.PART_XY[1] - M.RADIUS - (y + out), 0.0)
        assert np.hypot(gap_x, gap_y) > 0.03  # the cup never comes near the standing part
        assert x - out > 0.2


# -- the monitor and scoring ---------------------------------------------------------------------------------------


def test_holding_needs_the_pads_closed_on_the_part():
    tcp = np.array([0.3, -0.3, 0.85])
    part = part_under(tcp).pos
    assert M.holding(tcp, HELD_GAP, part)
    assert not M.holding(tcp, OPEN_GAP, part)  # open
    assert not M.holding(tcp, 0.0, part)  # closed on nothing
    assert not M.holding(tcp + (0.05, 0, 0), HELD_GAP, part)  # closed beside it


def test_a_gentle_matched_handover_scores_full():
    t = handover(release_height=0.01)
    rel = M._STATE["releases"]
    assert len(rel) == 1 and rel[0]["gentle"] and rel[0]["speed"] < 1e-6
    x, y, _ = M.PATH(t)[0]
    resting = state((x, y, M.FLOOR_TOP + M.HEIGHT / 2), vel=M.PATH(t)[1])
    assert M.success(objects(resting, t))
    assert M.score(objects(resting, t)) == 0.5  # the partial credit (success itself scores 1.0)


@pytest.mark.parametrize("kw", [
    dict(release_height=0.03),  # dropped from 3 cm
    dict(release_height=0.01, lag_speed=0.04),  # let go while moving 4 cm/s against the cup
    dict(release_height=0.01, over=(0.06, 0.0)),  # let go over the wall
])
def test_a_drop_toss_or_mismatched_release_scores_half(kw):
    t = handover(**kw)
    assert len(M._STATE["releases"]) == 1 and not M.gentle_handover()
    x, y, _ = M.PATH(t)[0]
    resting = state((x, y, M.FLOOR_TOP + M.HEIGHT / 2), vel=M.PATH(t)[1])
    assert not M.success(objects(resting, t))
    assert M.score(objects(resting, t)) == 0.5


def test_thresholds_are_at_2_cm_and_3_cm_per_s():
    handover(release_height=0.019)
    assert M.gentle_handover()
    handover(release_height=0.01, lag_speed=0.025)
    assert M.gentle_handover()
    handover(release_height=0.01, lag_speed=0.035)
    assert not M.gentle_handover()


def test_the_end_state_needs_the_part_resting_in_the_cup_and_let_go():
    t = handover(release_height=0.01)
    x, y, _ = M.PATH(t)[0]
    v = M.PATH(t)[1]
    z = M.FLOOR_TOP + M.HEIGHT / 2
    assert M.in_cup(objects(state((x + 0.02, y - 0.02, z), vel=v), t))
    assert not M.in_cup(objects(state((x, y, z)), t))  # still while the cup moves: sliding against it
    assert not M.in_cup(objects(state((x, y, z), vel=v, spin=(0, 0, 2.0)), t))  # spinning
    assert not M.in_cup(objects(state((x, y, z + 0.03), vel=v), t))  # above the floor
    assert not M.in_cup(objects(state((x, y + 0.08, z), vel=v), t))  # beside the cup
    lying = state((x, y, M.FLOOR_TOP + M.RADIUS), size=(M.HEIGHT, 2 * M.RADIUS, 2 * M.RADIUS), vel=v)
    assert M.in_cup(objects(lying, t))  # tipped over but inside still counts as in the cup


def test_holding_it_in_the_cup_is_not_a_handover():
    M.reset_state()
    M._STATE["lifted"] = True  # carried over from the table
    t0 = 8.0

    def tcp_of(t):
        x, y, _ = M.PATH(t0 + t)[0]
        return (x, y, M.FLOOR_TOP + M.HEIGHT - M.GRIP_BELOW_TOP)

    run(0.0, 2.0, tcp_of, lambda t: HELD_GAP, lambda t, tcp, g: part_under(tcp, vel=M.PATH(t0 + t)[1]))
    x, y, _ = M.PATH(t0 + 2.0)[0]
    held = state((x, y, M.FLOOR_TOP + M.HEIGHT / 2), vel=M.PATH(t0 + 2.0)[1])
    assert not M.in_cup(objects(held, t0 + 2.0)) and M.score(objects(held, t0 + 2.0)) == 0.25


def test_partial_credit_for_lifting_and_nothing_otherwise():
    M.reset_state()
    standing = state(M.PART.pos)
    M.update(0.05, standing, cup_at(0.05), {"left": (FAR, OPEN_GAP), "right": (FAR, OPEN_GAP)})
    assert M.score(objects(standing, 0.05)) == 0.0
    lifted = state((*M.PART_XY, M.TOP + M.LIFT_M + M.HEIGHT / 2 + 0.001))
    M.update(0.10, lifted, cup_at(0.10), {"left": (FAR, OPEN_GAP), "right": (FAR, OPEN_GAP)})
    M.update(0.15, standing, cup_at(0.15), {"left": (FAR, OPEN_GAP), "right": (FAR, OPEN_GAP)})
    assert M.score(objects(standing, 0.15)) == 0.25  # dropped back on the table keeps 0.25
    M.reset_state()
    assert M.score(objects(state((0.18, 0.0, M.TOP + M.HEIGHT / 2)), 0.2)) == 0.0  # pushed along the table


# -- the demo ------------------------------------------------------------------------------------------------------


def _plan():
    plan = M.demo()["right"]
    return plan, ArmPlan(plan, station.BASES["right"])


def test_the_demo_is_reachable():
    plan, arm = _plan()
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 30


def test_the_demo_follows_the_cup_at_every_control_step():
    plan, arm = _plan()
    # the follow waypoints end on the control-step grid, so the commanded TCP moves with the cup, not stop-and-go
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label in ("follow", "lower", "ride", "release", "ride open", "rise"):
            assert round(t / DT) * DT == pytest.approx(t, abs=1e-9)
    # every follow target is over the cup one lead ahead (the arm's PD trails it by about 60 ms), and level with the
    # part's base on the cup floor from the end of the lowering until the rise
    for t in np.arange(M.MEET_S + DT, M.LEAVE_S + 1e-9, DT):
        assert np.allclose(arm.target(t)[0][:2], M.PATH(t + M.FOLLOW_LEAD)[0][:2], atol=1e-6), t
    for t in np.arange(M.LOWER_S[1], M.RISE_S[0] + 1e-9, DT):
        assert arm.target(t)[0][2] == pytest.approx(M.RELEASE_Z, abs=1e-6)
    assert 0.04 <= M.FOLLOW_LEAD <= 0.08
    # commanded speed relative to the (leading) cup is nil from the end of the lowering until the rise
    for t in np.arange(M.LOWER_S[1] + DT, M.RISE_S[0] + 1e-9, DT):
        lead = M.FOLLOW_LEAD
        rel = (arm.target(t)[0] - M.PATH(t + lead)[0]) - (arm.target(t - DT)[0] - M.PATH(t - DT + lead)[0])
        assert np.linalg.norm(rel) / DT < 1e-4, t


def test_the_demo_lets_go_slowly_while_the_cup_moves_steadily():
    plan, arm = _plan()
    fingers = {round(t, 2): arm.target(t)[2] for t in np.arange(M.MEET_S, M.LEAVE_S + 1e-9, DT)}
    assert all(f == pytest.approx(FINGER_CLOSED) for t, f in fingers.items() if t <= M.OPEN_S + 1e-9)
    assert fingers[round(M.RISE_S[0], 2)] == pytest.approx(FINGER_OPEN)
    opening = [f for t, f in sorted(fingers.items()) if t > M.OPEN_S]
    assert all(b <= a + 1e-12 for a, b in zip(opening, opening[1:]))  # only ever opens
    # per finger, the pads let go of the 4 cm part at 2 cm open; the grip force (50 N cap, 5000 N/m) fades from 15 N
    # to 0 over 3 control steps or more, and the pads are 4.6 cm apart (the monitor's release) after that
    on_part = M.RADIUS
    t_15n = min(t for t, f in fingers.items() if -f >= on_part - 15 / 5000)
    t_free = min(t for t, f in fingers.items() if -f >= on_part)
    t_seen = min(t for t, f in fingers.items() if -2 * f >= M.HOLD_GAP[1])
    assert t_free - t_15n >= 3 * DT - 1e-9 and t_seen > t_free
    assert t_seen == pytest.approx(M.RELEASED_S, abs=DT + 1e-9)
    # the cup doesn't stop from the grip loosening until the arm has risen (it stops at 10.6 s) ...
    for t in np.arange(M.OPEN_S, M.RISE_S[1] + 1e-9, 1 / 240):
        assert np.linalg.norm(M.PATH(t)[1]) > 0.01, t
    # ... and moves at 5 cm/s or more, barely changing speed, through the opening and 0.3 s after
    for t in np.arange(M.OPEN_S, M.RELEASED_S + 0.3, 1 / 240):
        assert np.linalg.norm(M.PATH(t)[1]) > 0.05
        accel = (M.PATH(t + 1e-3)[1] - M.PATH(t - 1e-3)[1]) / 2e-3
        assert np.linalg.norm(accel) < 0.03  # the arm's 60 ms lag changes by under 2 mm/s
    # rides along open for 0.4 s or more after the pads let go before rising
    assert M.RISE_S[0] - M.RELEASED_S >= 0.4


def test_the_demo_sets_the_part_on_the_floor_and_clears_the_rim():
    held_base = M.RELEASE_Z + M.GRIP_BELOW_TOP - M.HEIGHT
    assert held_base - M.FLOOR_TOP == pytest.approx(M.RELEASE_GAP)
    # set down on the floor (commanded without the arm's sag under the part, so the floor carries it): a grip up to
    # 1.5 cm higher on the part than planned still lets go well under the 2 cm limit
    assert M.RELEASE_GAP == 0.0 and M.RELEASE_GAP + 0.015 < M.GENTLE_HEIGHT
    assert M.RELEASE_Z - PAD_TIP_PAST_TCP > M.RIM + 0.015  # the pad tips stay 1.5 cm or more above the rim
    assert M.TRAVEL_Z + M.GRIP_BELOW_TOP - M.HEIGHT > M.RIM + 0.05  # carried well above the rim
    assert M.GRASP_Z - PAD_TIP_PAST_TCP > M.TOP + 0.03  # the pads grip near the top, far off the table
    assert M.GRIP_BELOW_TOP >= 0.012 + 0.004  # the pads' flat faces reach 1.2 cm above the TCP: 4 mm or more of the
    # part's top stays clear of their chamfered upper edge
    assert M.RADIUS + 0.015 < M.CUP_IN / 2  # 3 cm between the part and each wall


def test_the_demo_through_the_monitor_scores_full():
    """The commanded TCP and fingers fed to the monitor (the arm tracking exactly, one lead behind the target) give
    one gentle release."""
    plan, arm = _plan()
    M.reset_state()
    for k in range(1, round(arm.duration / DT) + 1):
        t = k * DT
        tcp, _, finger, _ = arm.target(t)
        if M.MEET_S < t <= M.LEAVE_S:
            tcp = arm.target(t)[0] - (M.PATH(t + M.FOLLOW_LEAD)[0] - M.PATH(t)[0])  # the arm trails by the lead
        held = 3.1 <= t and -2 * finger < M.HOLD_GAP[1]  # closed on the part from the end of "close"
        gap = max(-2 * finger, HELD_GAP) if held else OPEN_GAP if t < 3.1 else -2 * finger
        if t < 3.1:
            part = state(M.PART.pos)
        elif held:
            part = part_under(tcp)
        else:
            x, y, _ = M.PATH(t)[0]
            part = state((x, y, M.FLOOR_TOP + M.HEIGHT / 2), vel=M.PATH(t)[1])
        M.update(t, part, cup_at(t), {"left": (FAR, OPEN_GAP), "right": (tcp, gap)})
    rel = M._STATE["releases"]
    assert len(rel) == 1 and rel[0]["gentle"], rel
    assert rel[0]["height"] == pytest.approx(0.0, abs=1e-6) and rel[0]["speed"] < 1e-3
    assert rel[0]["t"] == pytest.approx(M.RELEASED_S, abs=DT + 1e-9)
