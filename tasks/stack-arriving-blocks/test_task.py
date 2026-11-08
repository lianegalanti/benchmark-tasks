"""Offline checks (no Isaac Sim): the line's motion, the lost-block rule, the tower scoring, and that the demo is
reachable, on time and clear of the blocks.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
FAR = [np.array([0.2, 0.3, 1.2]), np.array([0.3, -0.3, 1.0])]  # grasp points well away from the tower
RED, YELLOW, BLUE = M.NAMES


def cube(pos, vel=(0.0, 0.0, 0.0), tilt_deg=0.0):
    pos, half = np.asarray(pos, float), M.CUBE / 2
    a = math.radians(tilt_deg) / 2
    quat = np.array([math.sin(a), 0.0, 0.0, math.cos(a)])
    return ObjectState(pos, quat, np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def level(k, dx=0.0, dy=0.0):
    """A cube standing at tower level ``k`` (0 on the pad), offset by dx, dy from the pad's centre."""
    return cube((M.PAD_XY[0] + dx, M.PAD_XY[1] + dy, M.PAD_TOP + (k + 0.5) * M.CUBE))


def on_line(name, t=0.0):
    return cube((M.LINE_X, M.carrier_y(name, t), M.CARRIER_TOP + M.CUBE / 2))


def play(frames, tcps=FAR):
    """Run the monitor over states from a reset; return (score, success) after the last."""
    M.reset_state()
    for objects in frames:
        M.update(objects, tcps)
    return M.score(frames[-1]), M.success(frames[-1])


def test_it_loads_with_the_carriers_driven_and_only_the_blocks_free():
    assert TASK.free_objects == M.NAMES
    assert TASK.score_over == "end"
    for c in ("red", "yellow", "blue"):
        assert f"carrier_{c}" in TASK.names


def test_the_line_runs_at_4_cm_per_s_and_brings_a_block_every_7_s():
    for name in M.NAMES:
        path = M.PATHS[name]
        assert path(0.0)[0] == pytest.approx(np.array(next(c.pos for c in M.CARRIERS if c.path is path)))
        assert np.linalg.norm(path(0.0)[1]) == 0.0 and np.linalg.norm(path(1 / 240)[1]) < 1e-3  # starts from rest
        assert path(3.0)[1][1] == pytest.approx(M.SPEED)
        assert M.carrier_y(name, M.arrival(name, 0.3)) == pytest.approx(0.3)
    gaps = np.diff([M.arrival(n, M.MEET_Y) for n in M.NAMES])
    assert gaps == pytest.approx([7.0, 7.0])
    ts = np.arange(0.0, 30.0, 1 / 240)
    for name in M.NAMES:
        ps = np.array([M.PATHS[name](t)[0] for t in ts])
        vs = np.array([M.PATHS[name](t)[1] for t in ts])
        assert np.max(np.linalg.norm(np.diff(ps, axis=0), axis=1)) < 0.16 / 240  # continuous
        on = ts < M.arrival(name, M.TURN_Y) - 0.05  # along the line, before it turns down at the end
        assert np.allclose(np.gradient(ps[on, 1], ts[on])[5:-5], vs[on, 1][5:-5], atol=2e-3)
        accel = np.max(np.abs(np.gradient(vs[on, 1], ts[on])[5:-5]))
        assert accel < 0.15  # m/s^2: far below what the blocks' friction (about 10 m/s^2) can carry


def test_carriers_stay_on_the_bed_and_turn_down_clear_of_the_table():
    bed_hi = M.BED[1] / 2
    assert M.TURN_Y - M.CARRIER[1] / 2 > M.TABLE_LEFT_Y > bed_hi  # turning down, a carrier misses table and bed
    assert M.STARTS[BLUE] - M.CARRIER[1] / 2 > -bed_hi  # the last carrier starts on the bed
    end = M.PATHS[RED](60.0)[0]
    assert end[1] == pytest.approx(M.TURN_Y) and end[2] == pytest.approx(M.CARRIER_Z - M.DROP)
    for a, b in zip(M.NAMES, M.NAMES[1:]):  # carriers never overlap
        assert M.STARTS[a] - M.STARTS[b] - M.CARRIER[1] > 0.15


def test_nothing_scores_at_the_start():
    objects = {n: on_line(n) for n in M.NAMES}
    assert play([objects]) == (0.0, False)
    assert M.lost() == {}


def test_tower_levels_score_a_quarter_each_and_full_score_needs_alignment_stillness_and_let_go():
    base = {RED: level(0), YELLOW: level(1, 0.005), BLUE: level(2, 0.01)}
    assert play([base]) == (0.75, True)
    assert play([{RED: level(0), YELLOW: on_line(YELLOW, 5.0), BLUE: on_line(BLUE, 5.0)}]) == (0.25, False)
    assert play([{RED: level(0), YELLOW: level(1, dy=0.01), BLUE: on_line(BLUE, 5.0)}]) == (0.5, False)
    # stacked in any order
    assert play([{BLUE: level(0), RED: level(1), YELLOW: level(2)}]) == (0.75, True)
    # three stacked but the top one 2 cm off the one below: stacked, not aligned
    assert play([{RED: level(0), YELLOW: level(1), BLUE: level(2, 0.02)}]) == (0.75, False)
    # still moving
    moving = dict(base, **{BLUE: cube(level(2).pos, vel=(0, 0, -0.05))})
    assert play([moving]) == (0.75, False)
    # a gripper still at the top block
    assert play([base], tcps=[level(2).pos.copy(), FAR[1]]) == (0.75, False)
    assert play([base], tcps=None)[1] is False


def test_blocks_that_dont_stand_squarely_dont_count():
    floating = {RED: level(0), YELLOW: cube(level(1).pos + [0, 0, 0.03]), BLUE: on_line(BLUE, 5.0)}
    assert play([floating])[0] == 0.25
    off_top = {RED: level(0), YELLOW: level(1, 0.03), BLUE: on_line(BLUE, 5.0)}  # centre past the lower top face
    assert play([off_top])[0] == 0.25
    tipped = {RED: level(0), YELLOW: cube(level(1).pos, tilt_deg=20), BLUE: on_line(BLUE, 5.0)}
    assert play([tipped])[0] == 0.25
    beside = {RED: cube((M.PAD_XY[0] + 0.06, M.PAD_XY[1], M.TOP + M.CUBE / 2)), YELLOW: on_line(YELLOW),
              BLUE: on_line(BLUE)}
    assert play([beside])[0] == 0.0


def test_lost_blocks_never_count_even_on_the_tower():
    tower = {RED: level(0), YELLOW: level(1), BLUE: level(2)}
    # blue went past the end line, then was put on top: two count
    past = dict(tower, **{BLUE: cube((M.LINE_X, M.END_Y + 0.01, M.CARRIER_TOP + M.CUBE / 2))})
    assert play([past, tower]) == (0.5, False)
    assert M.lost() == {BLUE: "passed the end line"}
    # red was set down on the table first, then built in at the bottom: the tower above it doesn't stand on the pad
    on_table = dict(tower, **{RED: cube((0.25, 0.1, M.TOP + M.CUBE / 2))})
    assert play([on_table, tower]) == (0.0, False)
    assert M.lost() == {RED: "touched the table"}
    # a block riding the line just before the end line is fine
    assert play([{n: cube((M.LINE_X, M.END_Y - 0.005, M.CARRIER_TOP + M.CUBE / 2)) for n in M.NAMES}])[0] == 0.0
    assert M.lost() == {}


def test_untouched_blocks_are_lost_at_the_end_line():
    for name in M.NAMES:
        t = M.arrival(name, M.END_Y)
        assert M.carrier_y(name, t + 0.05) > M.END_Y
    assert M.arrival(BLUE, M.END_Y) < 30.0  # all decided well within the run limit (60 s)


def test_the_layout_fits_the_arms_and_keeps_clear():
    assert M.PAD_XY[0] + M.PAD[0] / 2 < M.LINE_X - M.BED[0] / 2 - 0.08  # pad well clear of the line
    assert M.END_Y - M.MEET_Y > 0.2
    # the end line is in reach of the left arm (so the deadline, not reach, decides)
    assert np.hypot(M.LINE_X, M.END_Y - station.BASES["left"][1]) < 0.45


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


def test_the_demo_follows_each_block_and_takes_it_before_the_end_line():
    plan = M.demo()["left"]
    for name in M.NAMES:
        steps = [(t1, w) for _, t1, w in _spans(plan) if name in w.label and not w.label.startswith("rise")]
        for t1, w in steps:  # every tracking waypoint is over the block's centre at its time
            assert w.pos[0] == M.LINE_X and w.pos[1] == pytest.approx(M.carrier_y(name, t1), abs=1e-9)
        lifted = max(t1 for t1, w in steps if w.label.startswith("lift"))
        assert M.carrier_y(name, lifted) < M.END_Y - 0.15
        down = [t1 for t1, w in steps if w.label.startswith("follow")][0]
        assert down >= M.arrival(name, M.MEET_Y)


def test_the_demo_fingers_clear_the_blocks_on_the_line_and_the_tower():
    from yam_arena.constants import FINGER_CLOSED  # noqa: F401

    block_top = M.CARRIER_TOP + M.CUBE
    assert M.PRE_Z - PAD_TIP_PAST_TCP - block_top > 0.02  # waiting over the line
    assert M.GRASP_Z - PAD_TIP_PAST_TCP - M.CARRIER_TOP >= 0.015  # grasping: tips over the carrier
    open_half = -FINGER_OPEN + 0.003  # a finger's outer face, from the TCP, open (3 mm pad)
    # whenever the open hand moves between the tower and the line, its tips stay 1.5 cm over the tower's top
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    built = 0
    for t0, t1, w in _spans(plan):
        if w.label == "release":
            built += 1
        if w.label.startswith("over the line") and built:
            tower_top = M.PAD_TOP + built * M.CUBE
            for t in np.arange(t0, t1, DT / 2):
                p = arm.target(t)[0]
                near_x = p[0] - open_half  # the finger nearer the arms (fingers close along x)
                if near_x < M.PAD_XY[0] + M.CUBE / 2 + 0.005:
                    assert p[2] - PAD_TIP_PAST_TCP - tower_top > 0.015, (w.label, t)


def test_the_held_block_clears_the_tower_on_the_way_across():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    built = 0
    for t0, t1, w in _spans(plan):
        if w.label == "across to the tower":
            tower_top = M.PAD_TOP + built * M.CUBE
            for t in np.arange(t0, t1, DT / 2):
                p = arm.target(t)[0]
                underside = p[2] - 0.004 - M.CUBE / 2
                assert underside - tower_top > 0.02
        if w.label == "release":
            built += 1


def test_the_demo_lets_go_on_the_seats_and_ends_clear():
    plan = M.demo()["left"]
    releases = [w for w in plan if w.label == "release"]
    assert len(releases) == 3
    for k, w in enumerate(releases):
        assert w.pos[:2] == M.PAD_XY
        assert w.pos[2] - 0.004 - M.CUBE / 2 - (M.PAD_TOP + k * M.CUBE) == pytest.approx(M.DROP_M)
    end = np.array(plan[-1].pos)
    top_centre = np.array([*M.PAD_XY, M.PAD_TOP + 2.5 * M.CUBE])
    assert np.linalg.norm(end - top_centre) > M.CLEAR_M + 0.02
    right = ik.tcp_pose(np.array(READY), station.BASES["right"])[0]  # yam_scripted holds the right arm at READY
    assert np.linalg.norm(right - top_centre) > M.CLEAR_M
