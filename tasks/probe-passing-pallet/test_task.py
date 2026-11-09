"""Offline checks (no Isaac Sim): the line's motion, the probe and its stand, the test monitor and every scoring level,
and that the demo is reachable, keeps the probe's tip on each moving pad long enough before the end line, and keeps
its fingers clear.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
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
FIRST, SECOND = M.NAMES
FAR = [np.array([0.2, 0.3, 1.2]), np.array([0.3, -0.3, 1.2])]


def body(pos, tilt_deg=0.0):
    pos = np.asarray(pos, float)
    a = math.radians(tilt_deg) / 2
    return ObjectState(pos, np.array([math.sin(a), 0.0, 0.0, math.cos(a)]), np.zeros(3), np.zeros(3), pos, pos)


def pad(name, t):
    pos = M.PAD_PATHS[name](t)[0]
    half = np.asarray(M.PAD) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3), np.zeros(3), pos - half, pos + half)


def probe_at_tip(tip, tilt_deg=0.0):
    """The probe placed so that its tip's end is at ``tip`` (upright when untilted)."""
    p = body((0, 0, 0), tilt_deg)
    offset = M._rotation(p.quat_xyzw) @ np.array([0.0, 0.0, M.TIP_END])
    return body(np.asarray(tip, float) - offset, tilt_deg)


def grip_of(probe):
    return np.asarray(probe.pos) + M._rotation(probe.quat_xyzw) @ np.array([0.0, 0.0, M.GRIP_UP])


def run(frames):
    """Frames are (t, probe, tcps); returns (score, success, tested)."""
    M.reset_state()
    for t, probe, tcps in frames:
        objects = {"probe": probe, **{f"pad_{n}": pad(n, t) for n in M.NAMES}}
        M.update(t, objects, tcps)
    return M.score(None), M.success(None), M.tested()


def follow(name, t0, seconds, dz=0.0, dy=0.0, tilt=0.0, held=True):
    """The probe tracking pallet ``name``'s pad from ``t0`` for ``seconds``, tip ``dz`` above the pad's top."""
    out = []
    for t in np.arange(t0, t0 + seconds + 1e-9, DT):
        c = M.PAD_PATHS[name](t)[0]
        probe = probe_at_tip((c[0], c[1] + dy, c[2] + M.PAD[2] / 2 + dz), tilt)
        out.append((float(t), probe, [grip_of(probe), FAR[1]] if held else FAR))
    return out


def in_stand():
    probe = body((*M.STAND_XY, M.PROBE_Z0))
    return probe


# -- the scene -----------------------------------------------------------------------------------------------------


def test_it_loads_with_the_pallets_and_pads_driven_and_only_the_probe_free():
    assert TASK.free_objects == ("probe",)
    for n in M.NAMES:
        assert f"pallet_{n}" in TASK.names and f"pad_{n}" in TASK.names
    assert (Path(M.__file__).parent / "assets" / "probe.usda").is_file()


def test_the_probe_file_matches_the_task_sizes():
    import make_probe

    text = (Path(M.__file__).parent / "assets" / "probe.usda").read_text()
    assert text == make_probe.probe_usda(M.HANDLE, M.TIP, M.PROBE_MASS)


def test_the_line_runs_at_4_cm_per_s_and_pads_ride_on_their_pallets():
    for n in M.NAMES:
        assert np.linalg.norm(M.PATHS[n](0.0)[1]) == 0.0
        assert M.PATHS[n](3.0)[1][1] == pytest.approx(M.SPEED)
        assert M.pallet_y(n, M.arrival(n, 0.3)) == pytest.approx(0.3)
        for t in (0.0, 5.0, 30.0, 60.0):
            d = M.PAD_PATHS[n](t)[0] - M.PATHS[n](t)[0]
            assert d[:2] == pytest.approx([0, 0]) and d[2] == pytest.approx(M.PAD_Z - M.PALLET_Z)
    assert M.PAD_Z - M.PAD[2] / 2 - M.PALLET_TOP == pytest.approx(M.PAD_GAP)  # pad and pallet never touch
    assert M.arrival(SECOND, 0.1) - M.arrival(FIRST, 0.1) == pytest.approx(7.5)
    assert M.STARTS[SECOND] - M.PALLET[1] / 2 > -M.BED[1] / 2  # both start on the bed
    assert M.TURN_Y - M.PALLET[1] / 2 > M.TABLE_LEFT_Y
    assert M.arrival(SECOND, M.END_Y) < 30.0  # all decided well within the run limit (40 s)


def test_a_still_probe_cannot_stay_on_a_pad_for_a_full_test():
    # the pad's outline passes a fixed point in (pad + 0) / speed: under a second, far short of 2 s
    assert (M.PAD[1]) / M.SPEED < M.LEVELS[-1][0]


def test_the_probe_sits_in_its_stand_clear_of_the_table():
    tip = M.PROBE_Z0 + M.TIP_END
    assert tip - M.TOP > 0.003  # the tip hangs in the well, off the table
    assert (M.WELL - M.TIP[0]) / 2 >= 0.003  # clearance round the tip
    assert M.HANDLE[0] / 2 - M.WELL / 2 >= 0.004  # the handle overlaps the wall tops it rests on
    assert M.HANDLE[0] / 2 < M.WELL / 2 + M.WALL
    # the stand is clear of the line
    assert M.STAND_XY[0] + M.WELL / 2 + M.WALL < M.LINE_X - M.BED[0] / 2 - 0.05


# -- scoring -------------------------------------------------------------------------------------------------------


def test_nothing_scores_at_the_start():
    assert run([(0.0, in_stand(), FAR)]) == (0.0, False, {FIRST: None, SECOND: None})


def test_credit_levels_per_pallet_average_over_both():
    t1, t2 = M.arrival(FIRST, 0.1), M.arrival(SECOND, 0.1)
    assert run(follow(FIRST, t1, 2.0) + follow(SECOND, t2, 2.0))[:2] == (1.0, True)
    assert run(follow(FIRST, t1, 2.0))[:2] == (0.5, False)
    assert run(follow(FIRST, t1, 2.0) + follow(SECOND, t2, 1.5))[:2] == (0.875, False)
    assert run(follow(FIRST, t1, 1.0) + follow(SECOND, t2, 1.2))[:2] == (0.5, False)
    assert run(follow(FIRST, t1, 0.0))[:2] == (0.125, False)  # one frame of contact: 0.25 for that pallet
    assert run(follow(FIRST, t1, 1.9) + follow(SECOND, t2, 1.9))[:2] == (0.75, False)


def test_a_break_restarts_the_count_and_the_best_run_is_kept():
    t1 = M.arrival(FIRST, 0.1)
    frames = follow(FIRST, t1, 1.2) + follow(FIRST, t1 + 1.25, 1.2, dz=0.01) + follow(FIRST, t1 + 2.5, 1.2)
    score, ok, best = run(frames)
    assert best[FIRST] == pytest.approx(1.2) and score == 0.25 and not ok


def test_on_the_pad_means_inside_its_outline_and_within_4_mm_above_or_5_mm_into_it():
    t1 = M.arrival(FIRST, 0.1)
    assert run(follow(FIRST, t1, 2.0, dz=0.0035))[2][FIRST] == pytest.approx(2.0)
    assert run(follow(FIRST, t1, 2.0, dz=-0.0045))[2][FIRST] == pytest.approx(2.0)
    assert run(follow(FIRST, t1, 2.0, dz=0.006))[2][FIRST] is None  # hovering
    assert run(follow(FIRST, t1, 2.0, dy=0.014))[2][FIRST] == pytest.approx(2.0)
    assert run(follow(FIRST, t1, 2.0, dy=0.017))[2][FIRST] is None  # off the pad's edge


def test_the_probe_must_be_held_by_its_handle_and_upright():
    t1 = M.arrival(FIRST, 0.1)
    assert run(follow(FIRST, t1, 2.0, held=False))[2][FIRST] is None  # set on the pad and let go
    assert run(follow(FIRST, t1, 2.0, tilt=15))[2][FIRST] == pytest.approx(2.0)
    assert run(follow(FIRST, t1, 2.0, tilt=30))[2][FIRST] is None  # laid over
    # held by the tip end instead of the handle
    frames = [(t, p, [M.tip_end(p), FAR[1]]) for t, p, _ in follow(FIRST, t1, 2.0)]
    assert run(frames)[2][FIRST] is None


def test_contact_past_the_end_line_does_not_count():
    t = M.arrival(FIRST, M.END_Y) + 0.05
    assert run(follow(FIRST, t, 2.0))[2][FIRST] is None
    # a test running into the end line stops counting there
    t0 = M.arrival(FIRST, M.END_Y) - 1.0
    assert run(follow(FIRST, t0, 3.0))[2][FIRST] == pytest.approx(0.95, abs=0.06)


def test_the_wrong_pad_counts_for_its_own_pallet():
    t2 = M.arrival(SECOND, 0.1)
    assert run(follow(SECOND, t2, 2.0))[2] == {FIRST: None, SECOND: pytest.approx(2.0)}


# -- the demo ------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, out = 0.0, []
    for w in plan:
        out.append((t, t + w.duration, w))
        t += w.duration
    return out


def test_the_demo_is_reachable_and_within_the_joint_limits():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 30


def test_the_demo_follows_each_pad_and_finishes_well_before_the_end_line():
    plan = M.demo()["left"]
    for name in M.NAMES:
        steps = [(t1, w) for _, t1, w in _spans(plan) if w.label.endswith(name) and "over the line" not in w.label]
        for t1, w in steps:
            assert w.pos[0] == M.LINE_X and w.pos[1] == pytest.approx(M.pallet_y(name, t1), abs=1e-9)
        dwell = [t1 for t1, w in steps if w.label.startswith("test")]
        assert dwell[-1] - dwell[0] >= M.TEST_S + 0.6  # 0.6 s spare
        assert M.pallet_y(name, max(t1 for t1, _ in steps)) < M.END_Y - 0.10


def test_the_planned_demo_scores_full_with_the_probe_in_hand():
    """Replay the plan's targets with the probe held rigidly where it was grasped: the monitor gives 1.0."""
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    closed = next(t1 for _, t1, w in _spans(plan) if w.label == "close on the handle")
    frames = []
    for k in range(round(arm.duration / DT)):
        t = (k + 1) * DT
        p = arm.target(t)[0]
        probe = in_stand() if t < closed else body(p - np.array([0, 0, M.GRIP_UP]))
        frames.append((t, probe, [p, FAR[1]]))
    score, ok, best = run(frames)
    assert (score, ok) == (1.0, True)
    assert all(b >= M.TEST_S + 0.5 for b in best.values())


def test_a_lag_along_the_line_still_scores_full():
    """The arm trailing its targets by 0.2 s (8 mm along the line) still keeps the tip on the pads."""
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    frames = []
    for k in range(round(5.0 / DT), round(arm.duration / DT)):
        t = (k + 1) * DT
        p = arm.target(t - 0.2)[0]
        probe = body(p - np.array([0, 0, M.GRIP_UP]))
        frames.append((t, probe, [p, FAR[1]]))
    assert run(frames)[:2] == (1.0, True)


def test_the_demo_fingers_clear_the_stand_pallets_and_pads():
    tips = M.GRASP_Z - PAD_TIP_PAST_TCP
    assert tips - (M.TOP + M.STAND_H) >= 0.03  # grasping the handle, fingertips well above the stand
    assert M.TOUCH_Z - PAD_TIP_PAST_TCP - M.PAD_TOP >= 0.05  # on the pad, fingertips 5 cm over it
    # the carried tip clears the stand when lifted and the pallets when carried over the line
    assert M.HOVER_Z - M.TCP_ABOVE_TIP - (M.TOP + M.STAND_H) >= 0.03
    assert M.HOVER_Z - M.TCP_ABOVE_TIP - M.PAD_TOP >= 0.025


def test_the_right_arm_stays_out_of_the_way():
    from yam_arena.constants import READY

    right = ik.tcp_pose(np.array(READY), station.BASES["right"])[0]  # yam_scripted holds the right arm at READY
    assert right[2] > M.HOVER_Z - 0.05 or right[0] < M.LINE_X - M.BED[0] / 2 - 0.05
