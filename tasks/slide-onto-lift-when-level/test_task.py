"""Offline checks (no Isaac Sim): the lift's cycle, the layout, the scoring, and that the demo is reachable, pushes only
while the deck waits level, and is off the deck before it moves.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20


def carton(pos, tilt_deg=0.0, yaw_deg=0.0, vel=(0.0, 0.0, 0.0)):
    """The carton at ``pos`` (centre), tilted about the y axis (its far side down for positive angles), then yawed."""
    a, b = math.radians(tilt_deg) / 2, math.radians(yaw_deg) / 2
    qy = np.array([0.0, math.sin(a), 0.0, math.cos(a)])
    qz = np.array([0.0, 0.0, math.sin(b), math.cos(b)])
    # qz * qy (xyzw)
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qy
    q = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                  w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    pos, half = np.asarray(pos, float), np.asarray(M.CARTON, float) / 2
    return ObjectState(pos, q, np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def deck(t):
    pos, vel = M.PATH(t)
    half = np.asarray(M.DECK) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), vel, np.zeros(3), pos - half, pos + half)


def on_bench(x, **k):
    return carton((x, M.LINE_Y, M.BENCH_TOP + M.CARTON[2] / 2), **k)


def on_deck(x, t, **k):
    return carton((x, M.LINE_Y, M.PATH(t)[0][2] + M.DECK[2] / 2 + M.CARTON[2] / 2), **k)


def run(pairs):
    M.MONITOR.reset()
    return [M.MONITOR.update(c, d) for c, d in pairs]


LEVEL_T = 5.0  # inside the first level wait
LOW_T, HIGH_T = 1.0, 10.0


# -- the lift --------------------------------------------------------------------------------------------------------


def test_it_loads_with_only_the_carton_free_and_scores_at_the_end():
    assert TASK.free_objects == ("carton",)
    assert TASK.score_over == "end" and "lift_deck" in TASK.names


def test_the_lift_waits_at_low_level_high_level_on_an_18_s_cycle():
    top = lambda t: M.PATH(t)[0][2] + M.DECK[2] / 2  # noqa: E731
    for t, z in ((0.0, M.LOW_TOP), (2.9, M.LOW_TOP), (4.6, M.LEVEL_TOP), (7.4, M.LEVEL_TOP), (9.1, M.HIGH_TOP),
                 (11.9, M.HIGH_TOP), (13.6, M.LEVEL_TOP), (16.4, M.LEVEL_TOP), (18.1, M.LOW_TOP)):
        assert top(t) == pytest.approx(z, abs=1e-9), t
    assert M.level_windows(36.0) == [(4.5, 7.5), (13.5, 16.5), (22.5, 25.5), (31.5, 34.5)]
    assert M.LEVEL_TOP == pytest.approx(M.BENCH_TOP - 0.004)
    assert M.HIGH_TOP - M.BENCH_TOP > 0.04 and M.BENCH_TOP - M.LOW_TOP > 0.05
    ts = np.arange(0.0, 36.0, 1 / 240)
    zs = np.array([M.PATH(t)[0][2] for t in ts])
    vz = np.array([M.PATH(t)[1][2] for t in ts])
    assert np.max(np.abs(np.diff(zs))) < 0.06 / 1.5 * 1.6 / 240  # continuous
    assert np.allclose(np.gradient(zs, ts)[5:-5], vz[5:-5], atol=2e-3)  # the reported velocity is the motion's
    assert np.max(np.abs(np.gradient(vz, ts)[5:-5])) < 0.4  # m/s^2: far below g, the carton rides it
    xy = np.array([M.PATH(t)[0][:2] for t in ts])
    assert np.allclose(xy, xy[0])  # only vertical


def test_the_deck_clears_the_table_bench_and_mast():
    assert M.LOW_TOP - M.DECK[2] - M.TOP > 0.005  # its underside above the table at the low stop
    assert M.DECK_X[0] - M.EDGE_X == pytest.approx(0.015)
    assert M.MAST.pos[0] - M.MAST.size[0] / 2 - M.DECK_X[1] == pytest.approx(0.01)
    assert M.MAST.pos[2] + M.MAST.size[2] / 2 > M.HIGH_TOP + 0.05


def test_the_carton_is_too_wide_to_grip_and_starts_near_the_edge():
    assert min(M.CARTON[:2]) > 0.094 + 0.02
    assert M.EDGE_X - (M.START[0] + M.CARTON[0] / 2) == pytest.approx(0.03)
    assert M.START[0] - M.CARTON[0] / 2 - M.BENCH_X[0] > 0.06  # room behind it for the fingers
    assert M.DECK[0] - M.CARTON[0] >= 0.05  # 3 cm or so of deck each side of a centred carton


def test_pushing_low_on_the_carton_cannot_tip_it():
    """With the bench's friction (0.4 static), a push below this height slides it rather than tipping it."""
    tip_h = M.CARTON[0] / (2 * M.STEEL[0])
    assert tip_h > M.CARTON[2]  # taller than the carton: no push on its side tips it on the bench


def test_poses_are_plain_floats():
    """Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects at reset."""
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in spec.size), spec.name
    for stop in M.STOPS:
        assert all(type(v) is float for v in stop)


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_success_needs_it_wholly_on_the_deck_upright_and_riding_it():
    x = M.DECK_CX
    assert M.success({"carton": on_deck(x, LEVEL_T), "lift_deck": deck(LEVEL_T)})
    t = 8.25  # the deck rising to its high stop: riding with it is at rest relative to it
    riding = on_deck(x, t, vel=M.PATH(t)[1])
    assert M.success({"carton": riding, "lift_deck": deck(t)})
    assert M.success({"carton": on_deck(x, LEVEL_T, yaw_deg=10), "lift_deck": deck(LEVEL_T)})
    assert not M.success({"carton": on_deck(x, LEVEL_T, vel=(0.1, 0, 0)), "lift_deck": deck(LEVEL_T)})  # sliding
    assert not M.success({"carton": on_deck(x - 0.04, LEVEL_T), "lift_deck": deck(LEVEL_T)})  # hangs over the gap
    assert not M.success({"carton": on_deck(x, LEVEL_T, tilt_deg=20), "lift_deck": deck(LEVEL_T)})
    assert not M.success({"carton": on_bench(M.START[0]), "lift_deck": deck(LEVEL_T)})


def test_partial_credit_levels():
    d = deck(LEVEL_T)
    assert run([(on_bench(M.START[0]), d)]) == [0.0]
    assert run([(on_bench(M.START[0] + 0.049), d)]) == [0.0]
    assert run([(on_bench(M.START[0] + 0.051), deck(LOW_T))]) == [0.25]  # moved 5 cm, deck low: still on the bench
    over = M.DECK_X[0] + 0.025 - M.CARTON[0] / 2  # 2.5 cm of its underside over the level deck
    assert run([(on_bench(over), d)]) == [0.5]
    assert run([(on_bench(over), deck(LOW_T))]) == [0.25]  # the same overhang over a low deck isn't on it
    assert run([(on_deck(M.DECK_CX, LEVEL_T, vel=(0.05, 0, 0)), d)]) == [0.5]  # wholly on but still sliding


def test_partial_credit_is_kept_but_a_tip_or_fall_zeroes_it_for_good():
    d = deck(LEVEL_T)
    over = M.DECK_X[0] + 0.025 - M.CARTON[0] / 2
    tipped = on_bench(over, tilt_deg=25)
    assert run([(on_bench(over), d), (on_bench(M.START[0]), d)]) == [0.5, 0.5]  # kept
    assert run([(on_bench(over), d), (tipped, d), (on_bench(over), d)]) == [0.5, 0.0, 0.0]
    fallen = carton((M.EDGE_X, M.LINE_Y, M.TOP + M.CARTON[2] / 2))  # flat on the table
    assert run([(on_bench(M.START[0] + 0.06), d), (fallen, d)]) == [0.25, 0.0]
    M.MONITOR.reset()
    M.MONITOR.update(tipped, d)
    assert not M.success({"carton": on_deck(M.DECK_CX, LEVEL_T), "lift_deck": d})  # righted later: no success
    M.MONITOR.reset()


def test_a_carton_resting_level_on_the_low_deck_counts_as_on_it():
    """Its underside 5 cm below the bench, on the low deck: not fallen (only below the low stop is)."""
    assert run([(on_deck(M.DECK_CX, LOW_T), deck(LOW_T))]) == [0.5]
    assert run([(on_deck(M.DECK_CX, HIGH_T), deck(HIGH_T))]) == [0.5]


# -- the demo --------------------------------------------------------------------------------------------------------


def spans():
    t, out = 0.0, {}
    for w in M.demo()["right"]:
        out[w.label] = (t, t + w.duration)
        t += w.duration
    return out


def test_the_demo_is_reachable():
    plan = M.demo()["right"]
    arm = ArmPlan(plan, station.BASES["right"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < M.RUN_S / 4


def test_the_demo_pushes_only_while_the_deck_waits_level_and_is_off_it_before_it_moves():
    s = spans()
    start, end = M.WINDOW
    assert s["push onto the deck"][0] >= start + 0.05
    # the carton's leading edge reaches the gap only after 3 cm + the 1.5 cm finger gap of travel
    assert s["back off the deck"][1] <= end - 0.15
    assert s["down behind it"][1] <= start - 1.0  # in place well before the deck stops


def test_the_push_leaves_the_carton_wholly_on_the_deck_with_margin():
    end_c = M.PUSH_TO_X + M.FINGER_HALF + M.CARTON[0] / 2
    assert end_c - M.CARTON[0] / 2 - M.DECK_X[0] >= 0.03
    assert M.DECK_X[1] - (end_c + M.CARTON[0] / 2) >= 0.025
    push = M.PUSH_TO_X - M.PUSH_FROM_X
    peak_acc = 6 * push / M.PUSH_S**2  # smoothstep's peak acceleration
    assert peak_acc < 0.2 * M.STEEL[1] * 9.81  # the carton decelerates with the fingers, it doesn't run on


def test_the_fingers_clear_the_carton_and_the_deck():
    assert M.PUSH_FROM_X + M.FINGER_HALF + 0.015 == pytest.approx(M.START[0] - M.CARTON[0] / 2)
    assert M.CLEAR_Z - PAD_TIP_PAST_TCP - (M.BENCH_TOP + M.CARTON[2]) >= 0.03
    tips = M.PUSH_Z - PAD_TIP_PAST_TCP
    assert tips - M.LEVEL_TOP >= 0.009 and tips - M.BENCH_TOP >= 0.005
    assert M.BACK_X + M.FINGER_HALF < M.EDGE_X - 0.01  # backed off behind the bench edge
    assert M.PUSH_FROM_X - M.FINGER_HALF - M.BENCH_X[0] > 0.02  # fingers come down over the bench


def test_the_arm_stays_well_above_the_bench_along_the_demo():
    base = np.asarray(station.BASES["right"])
    arm = ArmPlan(M.demo()["right"], base)
    for k in range(round(arm.duration / DT)):
        q = arm.action7((k + 1) * DT)[0][:6]
        links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})
        for n in ("link3", "link4", "link5", "gripper"):
            assert links[n][2, 3] + base[2] - M.BENCH_TOP > 0.12, (n, k)
