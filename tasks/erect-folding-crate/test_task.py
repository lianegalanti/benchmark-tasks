"""Offline checks (no Isaac Sim): the crate's catalogue size and folded layout, the catch geometry, the scoring of
every outcome, and that the scripted demo is reachable and ends with both grippers clear.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402

UP = g.UP
START = np.array([g.XC, g.YC, g.TOP])
LEVEL = np.array([0.0, 0.0, 0.0, 1.0])
AWAY = [(np.array([0.18, 0.33, g.TOP + 0.29]), np.array([0.18, 0.33, g.TOP + 0.177])),
        (np.array([0.18, -0.33, g.TOP + 0.29]), np.array([0.18, -0.33, g.TOP + 0.177]))]
ON_RAIL = [(g.rail_world("wall_left", UP) + [0, 0, 0.10], g.rail_world("wall_left", UP) - [0, 0, 0.013])] + AWAY[1:]


def joints(walls=(), latched=(), pressed=()):
    """Walls in ``walls`` upright, others folded; catches in ``pressed`` pushed in 4 mm; ``latched`` sets both the
    walls and leaves the catch out."""
    j = {g.hinge_joint(w): 0.0 for w in g.WALLS} | {g.pawl_joint(p): 0.0 for p in g.PAWLS}
    for p in latched:
        walls = (*walls, g.PAWLS[p].wall, g.PAWLS[p].holds)
    for w in walls:
        j[g.hinge_joint(w)] = UP
    for p in pressed:
        j[g.pawl_joint(p)] = -0.004
    return j


def play(steps, js, pos=START, quat=LEVEL, segs=AWAY):
    """Reset, run the monitor ``steps`` control steps at 20 Hz with the given state; return (score, success)."""
    M.reset_state()
    for _ in range(steps):
        M.update(pos, quat, js, segs, 0.05)
    return M.score({}), M.success({})


def test_catalogue_crate():
    assert (g.LENGTH, g.WIDTH, g.HEIGHT) == (0.400, 0.300, 0.174)
    assert (g.INNER_LENGTH, g.INNER_WIDTH) == (0.374, 0.274)
    assert g.MASS == pytest.approx(g.M_BASE + 2 * g.M_LONG + 2 * g.M_SHORT + 4 * g.M_PAWL)
    assert g.M_BASE > 0.3
    # Upright, every wall's top reaches the crate's full height.
    for name, w in g.WALLS.items():
        assert w.point((0, 0, w.height), UP)[2] == pytest.approx(g.HEIGHT), name


def test_folded_walls_stack_without_touching():
    """Short walls fold lowest; the long walls lie on top, clear of them even where the catches stand out."""
    short_top = g.SHORT_HINGE_Z + g.T_WALL
    long_bottom = g.LONG_HINGE_Z
    assert long_bottom - g.PAWL_OUT - short_top >= 0.0015
    assert long_bottom - (g.SHORT_HINGE_Z + g.T_WALL / 2 + g.RAIL_R) >= 0.003      # short rails under long walls
    # Free edges of each folded pair leave a gap (fingers go down there).
    gap_long = 2 * (g.IN_X - g.LONG_H)
    gap_short = 2 * (g.IN_Y - g.SHORT_H)
    assert gap_long > 0.05 and gap_short > 0.12
    folded_height = g.LONG_HINGE_Z + g.T_WALL
    assert folded_height == pytest.approx(0.078)


def test_wall_frames_raise_upward():
    for name, w in g.WALLS.items():
        top_flat, top_up = w.point((0, 0, w.height), 0.0), w.point((0, 0, w.height), UP)
        assert top_flat[2] == pytest.approx(w.hinge[2])
        # folded inward, toward the crate's middle
        assert np.dot(top_flat[:2] - np.asarray(w.hinge[:2]), w.inward[:2]) == pytest.approx(w.height)
        # the outer face is outboard of the hinge when upright
        outer = w.point((0, -g.T_WALL, 0.05), UP)
        assert np.dot(outer[:2] - np.asarray(w.hinge[:2]), w.inward[:2]) == pytest.approx(-g.T_WALL), name


def test_long_walls_stand_short_walls_need_their_catches():
    """A long wall pressed onto its stop stands (its centre of mass is outboard of the hinge); short of about 5
    degrees from upright it falls back in."""
    lean = math.degrees(math.atan2(g.T_WALL / 2, g.LONG_H / 2))
    assert 4.0 < lean < 6.5


def test_catch_overlaps_short_wall_and_clears_when_pushed():
    edge = g.HOOK_REACH
    assert g.SHORT_W / 2 < g.IN_X - g.PAWL_OUT - 0.0005        # the plate itself stays clear of the pawls
    for name, p in g.PAWLS.items():
        w = g.WALLS[p.wall]
        tip = w.point(p.rest_local + [0, g.PAWL_OUT, 0], UP)
        assert abs(tip[0]) == pytest.approx(g.IN_X - g.PAWL_OUT)
        assert edge - abs(tip[0]) >= 0.004                       # the short wall's hooks pass over 4.5 mm of pawl
        assert g.PAWL_TRAVEL - g.PAWL_OUT >= 0.0005              # pushed fully in, it is below the inner face
        sq = w.point(p.rest_local + [p.out * g.PAWL_W / 2, 0, 0], UP)
        assert g.IN_Y - abs(sq[1]) == pytest.approx(g.PAWL_GAP)  # square face 1 mm inside the upright short wall
        assert g.SHORT_HINGE_Z < tip[2] < g.HEIGHT


def test_hooks_sit_at_the_pawls():
    for name in g.SHORT:
        w = g.WALLS[name]
        _, caps = g.wall_shapes(w)
        hooks = [c for c in caps if c[4] == "Z"]
        assert len(hooks) == 2
        for c, r, length, _, _ in hooks:
            p = w.point(c, UP)
            assert abs(p[0]) + r == pytest.approx(g.HOOK_REACH)
            assert p[2] - length / 2 - r < g.PAWL_Z - g.PAWL_H / 2 and p[2] + length / 2 + r > g.PAWL_Z + g.PAWL_H / 2
            inner = abs(w.point((c[0], c[1] + r, c[2]), UP)[1])
            assert inner == pytest.approx(g.IN_Y)                 # flush with the plate's inner face


def test_latched_needs_wall_past_the_pawl():
    """Within 1 degree of upright the short wall's inner face, at the pawl's top, is within 1.8 mm of upright: the
    pawl (1 mm inside) could not be fully out unless the wall had passed it."""
    v = g.PAWL_Z + g.PAWL_H / 2 - g.SHORT_HINGE_Z
    assert v * math.sin(g.UPRIGHT_TOL) < g.PAWL_GAP + g.PAWL_OUT


def test_start_scores_zero():
    assert play(30, joints()) == (0.0, False)


def test_long_walls_standing_score_point_one_each():
    assert play(30, joints(walls=g.LONG)) == (pytest.approx(0.2), False)
    assert play(30, joints(walls=("wall_far",))) == (pytest.approx(0.1), False)


def test_long_wall_held_up_or_short_of_upright_scores_nothing():
    far = g.rail_world("wall_far", UP)
    held = [(far + [0, 0, 0.10], far - [0, 0, 0.013]), AWAY[1]]
    assert play(30, joints(walls=("wall_far",)), segs=held) == (0.0, False)
    j = joints(walls=("wall_far",))
    j[g.hinge_joint("wall_far")] = UP - math.radians(4.0)
    assert play(30, j) == (0.0, False)


def test_short_walls_balanced_up_without_catches_score_nothing():
    assert play(30, joints(walls=tuple(g.WALLS), pressed=tuple(g.PAWLS))) == (pytest.approx(0.2), False)


def test_short_wall_short_of_upright_is_not_latched():
    j = joints(latched=tuple(g.PAWLS))
    j[g.hinge_joint("wall_left")] = UP - math.radians(2.0)
    assert play(30, j)[0] == pytest.approx(0.2 + 0.3)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_partial_credit_per_latched_corner(n):
    names = list(g.PAWLS)[:n]
    j = joints(walls=tuple(g.WALLS), latched=names, pressed=[p for p in g.PAWLS if p not in names])
    assert play(30, j) == (pytest.approx(0.2 + 0.15 * n), False)


def test_all_latched_but_held_is_not_success():
    assert play(30, joints(latched=tuple(g.PAWLS)), segs=ON_RAIL) == (pytest.approx(0.8), False)
    assert M.score({}) < 1.0


def test_all_latched_released_still_is_success():
    assert play(21, joints(latched=tuple(g.PAWLS)))[1] is True


def test_success_needs_a_second_of_stillness():
    assert play(20, joints(latched=tuple(g.PAWLS)))[1] is False
    M.reset_state()
    j = joints(latched=tuple(g.PAWLS))
    for k in range(40):
        M.update(START + [0.0, 0.003 * (k % 2), 0.0], LEVEL, j, AWAY, 0.05)
    assert not M.success({}) and M.score({}) == pytest.approx(0.8)


def test_base_not_flat_is_not_success():
    a = math.radians(4.0)
    quat = np.array([math.sin(a / 2), 0, 0, math.cos(a / 2)])
    assert play(30, joints(latched=tuple(g.PAWLS)), quat=quat) == (pytest.approx(0.8), False)


def test_falling_off_the_table_once_zeroes_the_trial():
    """The violation latches: putting the crate back and finishing it still scores 0."""
    M.reset_state()
    j = joints(latched=tuple(g.PAWLS))
    M.update(START - [0, 0, 0.5], LEVEL, joints(), AWAY, 0.05)
    for _ in range(40):
        M.update(START, LEVEL, j, AWAY, 0.05)
    assert M.score({}) == 0.0 and not M.success({})


@pytest.mark.parametrize("case", ["lifted", "tipped"])
def test_lifting_or_tilting_while_setting_up_is_allowed(case):
    """Lifting the crate 5 cm or tilting it 30 degrees on the way is no violation; finishing it still succeeds."""
    M.reset_state()
    if case == "lifted":
        M.update(START + [0, 0, 0.05], LEVEL, joints(), AWAY, 0.05)
    else:
        a = math.radians(30)
        M.update(START, np.array([0, math.sin(a / 2), 0, math.cos(a / 2)]), joints(), AWAY, 0.05)
    assert not M._STATE["violation"]
    j = joints(latched=tuple(g.PAWLS))
    for _ in range(40):
        M.update(START, LEVEL, j, AWAY, 0.05)
    assert M.success({})


def test_small_slide_and_tilt_are_not_violations():
    M.reset_state()
    a = math.radians(10)
    M.update(START + [0.03, 0.02, 0.0], np.array([math.sin(a / 2), 0, 0, math.cos(a / 2)]), joints(), AWAY, 0.05)
    assert not M._STATE["violation"]


def test_released_rule_uses_gripper_capsules():
    boxes = M.crate_boxes(START, np.eye(3), joints(latched=tuple(g.PAWLS)))
    for a, b in AWAY:
        d = M._dist_to_boxes(a + np.outer(np.linspace(0, 1, 12), b - a), boxes)
        assert d - M.GRIP_R >= M.CLEAR
    a, b = ON_RAIL[0]
    assert M._dist_to_boxes(a + np.outer(np.linspace(0, 1, 12), b - a), boxes) < M.GRIP_R


@pytest.fixture(scope="module")
def plan():
    return demo.offline_plan()


def test_demo_reachable(plan):
    left, right, steps = plan
    for arm in (left, right):
        assert arm.worst_mm < 0.5 and arm.worst_deg < 0.1
        qs = np.array([q for q, _ in arm.trace])
        assert ((qs >= ik.LOWER - 1e-6) & (qs <= ik.UPPER + 1e-6)).all()
        assert np.abs(np.diff(qs, axis=0)).max() < 0.15        # no joint jumps more than 0.15 rad in a step


def test_demo_is_open_loop():
    """The script runs without any simulator state (offline_plan gives it no way to read one)."""
    src = (HERE / "demo.py").read_text()
    body = src[src.index("def script("):src.index("def offline_plan(")]
    assert "crate_joints" not in body and "states(" not in body
    assert demo.PULL_S >= 2 * 1.5 and demo.PULL_RATE * demo.PULL_S <= math.radians(30)


def test_demo_fits_run_limit(plan):
    *_, steps = plan
    # plus 1 s of stillness and the 1 s success hold, with room to spare
    assert steps < 2400 - 3 * 20 - 200


def test_demo_ends_clear_of_the_erected_crate(plan):
    left, right, _ = plan
    from yam_arena import station

    boxes = M.crate_boxes(START, np.eye(3), joints(latched=tuple(g.PAWLS)))
    for arm, side in ((left, "left"), (right, "right")):
        tcp, rot = ik.tcp_pose(arm.trace[-1][0], station.BASES[side])
        a, b = tcp - 0.10 * rot[:, 2], tcp + M.PAD_TIP * rot[:, 2]
        assert M._dist_to_boxes(a + np.outer(np.linspace(0, 1, 12), b - a), boxes) - M.GRIP_R >= M.CLEAR + 0.01
        assert arm.trace[-1][1] == pytest.approx(1.0)        # gripper open


def test_demo_rail_pinch_and_brace(plan):
    """The pinch closes 2 mm per finger into the 18 mm rail; while the left wall clicks in, the right gripper is
    on the folded right wall's rail (the brace)."""
    gap = demo.G_PINCH * 0.0939
    assert 2 * g.RAIL_R - gap == pytest.approx(0.004, abs=0.0005)
    pre = demo.G_PRE * 0.0939
    # part-open fingers fit the slot on one side and the gap past the free edge on the other
    assert pre / 2 - g.RAIL_R >= 0.008 and pre / 2 + 0.010 <= g.SLOT_H + g.RAIL_R - 0.005


def test_instruction_is_goal_only():
    assert M.INSTRUCTION == "Set up the folded crate so it is ready to fill, then let go of it."
    assert TASK.success_hold_s == 1.0
