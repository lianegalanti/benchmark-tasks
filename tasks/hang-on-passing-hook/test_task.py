"""Offline checks (no Isaac Sim): the conveyor's motion and the scene's clearances, the scoring at every level, and
the scripted demo, played against a rigid-grip model of the parts: reachable, within joint limits, on time, the
eye threaded over the hook without touching it, and every arm link (as the URDF's collision hulls) clear of the
hooks, the rail and the parts it isn't holding.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
"""

import math
import re
from pathlib import Path

import numpy as np
import pytest

import geometry as g
from yam_arena import analysis, ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import DOWN_GRIP, ArmPlan
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
I3 = np.eye(3)
FAR = []  # no arm link anywhere near the parts
GRIPPER_LINKS = ["gripper", "tip_left", "tip_right"]


def gripper_at(tcp, finger=FINGER_CLOSED, up=False):
    """One gripper's collision hulls (its body and both finger links), pointing straight down (or, ``up``, straight
    up) with its fingers at ``finger`` and its TCP (between the finger pads) at ``tcp``: solved by IK in front of the
    left arm, then the gripper moved rigidly to ``tcp``."""
    base = station.BASES["left"]
    here = np.array([0.30, 0.30, g.TOP + 0.20])
    q, err, _ = ik.solve(here, DOWN_GRIP, base_pos=base)
    assert err < 1e-4
    joints = dict(zip(ARM_JOINTS, map(float, q), strict=True)) | {"joint7": finger, "joint8": finger}
    shift = np.eye(4)
    if up:
        shift[:3, :3] = np.diag([1.0, -1.0, -1.0])  # half a turn about x
    shift[:3, 3] = np.asarray(tcp, float) - shift[:3, :3] @ ik.tcp_pose(q, base)[0]
    poses = {k: shift @ v for k, v in analysis.fk_link_poses_world(joints, base).items()}
    return analysis.world_hulls(poses, GRIPPER_LINKS)


def box_hull(centre, size):
    """A box as a collision hull (for a stand-in obstacle)."""
    c, h = np.asarray(centre, float), np.asarray(size, float) / 2
    planes = [np.r_[n, n @ c + abs(n) @ h] for n in np.r_[np.eye(3), -np.eye(3)]]
    return [("box", analysis.Hull(vertices=g.corners(c, size), planes=np.array(planes)))]


def hung_on(i, t, x=g.HUNG_X):
    """A part hanging upright on hook ``i`` at time ``t``."""
    return np.array([x, g.hook_y(i, t), g.HUNG_Z]), I3


def standing(name):
    return g.part_start(name), I3


def run(frames, delay=lambda i, t: 0.0):
    """Play (t, parts, arm hulls) frames through the monitor from a fresh start; return (score, success). Hook ``i``
    runs ``delay(i, t)`` seconds behind its schedule (held back by the robot)."""
    M.reset_state()
    for t, parts, hulls in frames:
        full = {p: standing(p) for p in g.PARTS}
        full.update(parts)
        M.update(full, [g.hook_y(i, t - delay(i, t)) for i in range(g.N_HOOKS)], hulls, t)
    return M.score({}), M.success({})


def times(t0, t1, dt=0.05):
    return np.arange(t0, t1 + 1e-9, dt)


# -- layout and motion -------------------------------------------------------------------------------------------------

def test_it_loads_with_only_the_two_parts_free():
    assert TASK.free_objects == g.PARTS
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0
    assert sum(1 for o in M.OBJECTS if o.name.startswith("hook")) == 3 * g.N_HOOKS


def test_poses_are_plain_floats():
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in spec.size), spec.name
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", ())), spec.name


def test_the_usd_part_matches_the_geometry():
    text = (HERE / "assets" / "hanger_part.usda").read_text()
    for name, (centre, size) in g.PART_BOXES.items():
        block = text.split(f'def Cube "{name}"')[1].split("}")[0]
        scale = [float(v) for v in re.search(r"scale = \(([^)]*)\)", block).group(1).split(",")]
        assert scale == pytest.approx(size, abs=1e-6), name
        m = re.search(r"translate = \(([^)]*)\)", block)
        shift = [float(v) for v in m.group(1).split(",")] if m else [0.0, 0.0, 0.0]
        assert shift == pytest.approx(centre, abs=1e-6), name
    assert f"physics:mass = {g.MASS}" in text


def test_the_hooks_move_steadily_and_each_hooks_boxes_ride_together():
    for spec in M.HOOKS:
        p0, v0 = spec.path(0.0)
        p1, v1 = spec.path(10.0)
        assert v0 == pytest.approx([0.0, -g.SPEED, 0.0]) and v1 == pytest.approx(v0)
        assert p1 - p0 == pytest.approx([0.0, -10 * g.SPEED, 0.0])
    for i in range(g.N_HOOKS):
        boxes = g.hook_boxes(i)
        bar_lo = boxes["bar"][0] - boxes["bar"][1] / 2
        bar_hi = boxes["bar"][0] + boxes["bar"][1] / 2
        tip_hi = boxes["tip"][0] + boxes["tip"][1] / 2
        stem_lo = boxes["stem"][0] - boxes["stem"][1] / 2
        stem_hi = boxes["stem"][0] + boxes["stem"][1] / 2
        assert bar_lo[0] - tip_hi[0] == pytest.approx(g.GAP) and stem_lo[0] - bar_hi[0] == pytest.approx(g.GAP)
        assert g.RAIL_BOTTOM - stem_hi[2] == pytest.approx(0.005)  # stems hang just under the rail, not touching
    rail_y = (g.RAIL_CENTRE[1] - g.RAIL[1] / 2, g.RAIL_CENTRE[1] + g.RAIL[1] / 2)
    for i in range(g.N_HOOKS):  # the rail is over every hook from the start to the run limit (45 s)
        assert rail_y[0] < g.hook_y(i, 45.0) and g.hook_y(i, 0.0) < rail_y[1]


def test_hooks_start_off_the_table_and_leave_in_order():
    assert all(g.hook_y(i, 0.0) > g.TABLE_EDGE_Y for i in range(g.N_HOOKS))
    exits = [g.exit_time(i) for i in range(g.N_HOOKS)]
    assert exits == pytest.approx([26.0, 32.0, 38.0])
    assert exits[-1] * 20 < 900  # the last hook leaves well inside the 900-step run limit
    assert g.EXIT_Y > -g.TABLE_EDGE_Y  # the mark is on the table


def test_clearances_are_generous():
    assert (g.W_IN - g.BAR) / 2 >= 0.02  # the bar in the eye: 2.4 cm either side
    assert g.H_IN - (g.TIP_Z1 - g.TIP_Z0) >= 0.035  # threading: 1.9 cm above the tip and below the bar
    opening_top = g.THREAD_Z + g.H_IN / 2
    opening_bottom = g.THREAD_Z - g.H_IN / 2
    assert opening_top - g.TIP_Z1 >= 0.015 and g.TIP_Z0 - opening_bottom >= 0.015
    assert g.lowest_z(*hung_on(0, 0.0)) - g.TOP >= 0.02  # hung, the part's block is 2.5 cm above the table
    assert -FINGER_OPEN * 2 / 2 - g.TAB[1] / 2 >= 0.015  # open fingers clear the tab by 4 cm a side
    assert -M.RELEASE_OPEN - g.TAB[1] / 2 >= 0.015  # let-go opening: 1.9 cm a side
    for name in g.PARTS:  # standing parts are clear of the hooks' path
        pts = g.part_points(*standing(name))
        for c, s in g.hook_boxes(0, 0.0).values():
            c = c.copy()
            c[1] = g.PART_XY[name][1]
            assert analysis.box_signed_distance(pts, c, I3, s).min() > 0.05
    pad_tips = g.part_start("part_a")[2] + M.TCP_ABOVE_ORIGIN - PAD_TIP_PAST_TCP
    assert pad_tips - (g.H_IN / 2 + g.EYE_T + g.part_start("part_a")[2]) == pytest.approx(0.002, abs=1e-9)


def test_hung_part_rests_on_the_bar_with_the_bar_through_its_eye():
    pos, R = hung_on(1, 3.0)
    assert g.bar_through_eye(pos, R, g.hook_y(1, 3.0))
    assert not g.bar_through_eye(pos, R, g.hook_y(1, 3.0) + 0.05)  # beside the eye
    bar_bottom = pos + R @ (g.PART_BOXES["top_bar"][0] - [0, 0, g.EYE_T / 2])
    assert bar_bottom[2] == pytest.approx(g.BAR_TOP)
    assert g.COM[2] < 0  # the centre of mass hangs below the eye: it hangs upright


def test_a_part_turned_away_or_lying_on_the_bar_is_not_threaded():
    y = g.hook_y(0, 0.0)
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # eye facing along the rail
    assert not g.bar_through_eye(np.array([g.HUNG_X, y, g.HUNG_Z]), turned, y)
    lying_on = np.array([g.HUNG_X, y, g.BAR_TOP + 0.05])
    assert not g.bar_through_eye(lying_on, I3, y)


# -- scoring -----------------------------------------------------------------------------------------------------------

def test_untouched_scores_zero():
    assert run([(t, {}, FAR) for t in times(0.0, 40.0, 0.5)]) == (0.0, False)


def test_lifted_only():
    lifted = {"part_a": (g.part_start("part_a") + [0, 0, 0.04], I3)}
    frames = [(1.0, lifted, FAR)] + [(t, {}, FAR) for t in times(2.0, 40.0, 0.5)]
    assert run(frames) == (pytest.approx(0.1), False)


def test_threaded_but_still_held_by_the_tab_when_the_hook_leaves():
    def frame(t):
        pos, R = hung_on(0, t)
        return t, {"part_a": (pos, R)}, gripper_at(g.tab_centre(pos, R) + [0, 0, -0.005], -g.TAB[1] / 2)

    assert run([frame(t) for t in times(20.0, 27.0)]) == (pytest.approx(0.25), False)


def test_hung_and_let_go_scores_half_per_part():
    frames = [(t, {"part_a": hung_on(0, t)}, FAR) for t in times(20.0, 27.0)]
    assert run(frames) == (pytest.approx(0.5), False)
    assert M._STATE["credited"]["part_a"] == 0


def test_both_hung_on_their_own_hooks_succeeds():
    frames = [(t, {"part_a": hung_on(0, t), "part_b": hung_on(1, t)}, FAR) for t in times(20.0, 33.0)]
    assert run(frames) == (pytest.approx(1.0), True)
    frames = [(t, {"part_a": hung_on(2, t), "part_b": hung_on(1, t)}, FAR) for t in times(20.0, 39.0)]
    assert run(frames) == (pytest.approx(1.0), True)  # any two hooks


def test_both_on_one_hook_counts_once():
    frames = [(t, {"part_a": hung_on(0, t, 0.27), "part_b": hung_on(0, t, 0.35)}, FAR) for t in times(20.0, 39.0)]
    assert run(frames) == (pytest.approx(0.75), False)


def test_a_gripper_holding_the_lower_block_far_from_the_tab_gets_no_hanging_credit():
    """The reviewer's case: the bar through the eye and the eye on the bar, but a gripper holds the block under the
    eye, its grasp point 10 cm from the tab (the old 5 cm tab test would have credited it)."""
    def frame(t, finger=-(g.OUT_W / 2)):
        pos, R = hung_on(0, t)
        block = pos + R @ g.PART_BOXES["block"][0]
        assert np.linalg.norm(block - g.tab_centre(pos, R)) > 0.09
        return t, {"part_a": (pos, R)}, gripper_at(block, finger)

    assert run([frame(t) for t in times(20.0, 27.0)]) == (pytest.approx(0.25), False)
    # the gripper open around the block without touching it still counts: within 2 cm
    assert run([frame(t, FINGER_OPEN) for t in times(20.0, 27.0)]) == (pytest.approx(0.25), False)


def test_a_gripper_propping_the_block_from_below_gets_no_hanging_credit():
    def frame(t, drop):
        pos, R = hung_on(0, t)
        under = pos + R @ (g.PART_BOXES["block"][0] - [0, 0, g.BLOCK[2] / 2]) - [0, 0, drop]
        return t, {"part_a": (pos, R)}, gripper_at(under - [0, 0, PAD_TIP_PAST_TCP], up=True)

    assert run([frame(t, 0.0) for t in times(20.0, 27.0)])[0] == pytest.approx(0.25)  # pad tips pushing up on the block
    assert run([frame(t, 0.03) for t in times(20.0, 27.0)])[0] == pytest.approx(0.5)  # 3 cm below: let go


@pytest.mark.parametrize("part_box", sorted(g.PART_BOXES))
def test_any_arm_link_within_2cm_of_any_part_of_it_blocks_credit(part_box):
    """A 1 cm stand-in link beside each of the part's five boxes: 1.5 cm away blocks credit, 2.5 cm away doesn't."""
    def frames(d):
        out = []
        for t in times(24.0, 27.0):
            pos, R = hung_on(0, t)
            c, sz = g.PART_BOXES[part_box]
            side = pos + R @ (c + [-(sz[0] / 2 + d + 0.005), 0, 0])  # on the robot's side of that box
            out.append((t, {"part_a": (pos, R)}, box_hull(side, (0.01, 0.01, 0.01))))
        return out

    assert run(frames(0.015))[0] == pytest.approx(0.25)
    assert run(frames(0.025))[0] == pytest.approx(0.5)


def test_the_hook_must_support_it():
    # held with its eye 1 cm above the bar (nothing in the scene touching it): not resting on the hook
    frames = [(t, {"part_a": (hung_on(0, t)[0] + [0, 0, 0.01], I3)}, FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)
    frames = [(t, {"part_a": (hung_on(0, t)[0] + [0, 0, 0.004], I3)}, FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.5)  # within the 5 mm tolerance
    pos, R = hung_on(0, 0.0)
    assert g.rest_gap(pos, R, g.hook_y(0, 0.0)) == pytest.approx(0.0)
    assert g.rest_gap(pos, R, g.hook_y(0, 0.0) + 0.05) is None


def test_a_hook_the_robot_held_back_credits_nothing():
    """A real slip-clutch drive stalls when an arm holds a hook: a hook more than 1 cm (0.2 s) behind its schedule
    credits no part, even if it later hangs freely on it."""
    def frames(stop_s):
        delay = lambda i, t: stop_s if (i == 0 and t > 15.0) else 0.0  # noqa: E731
        fr = [(t, {"part_a": hung_on(0, t - delay(0, t))}, FAR) for t in times(10.0, 28.0)]
        return fr, delay

    assert run(*frames(0.5))[0] == pytest.approx(0.25)  # stopped 0.5 s: 2.5 cm behind
    assert run(*frames(0.1))[0] == pytest.approx(0.5)  # 5 mm behind: within tolerance
    assert run(*frames(0.0))[0] == pytest.approx(0.5)


def test_the_other_part_must_not_touch_it():
    """Nothing but its hook may touch a hung part: the other part (held by the other arm, say) propping it against
    the side of its eye blocks the credit; 3 cm away it doesn't."""
    def frames(gap):
        out = []
        for t in times(20.0, 27.0):
            pos, R = hung_on(0, t)
            out.append((t, {"part_a": (pos, R), "part_b": (pos + [0, -(g.OUT_W + gap), 0], R)}, FAR))
        return out

    assert run(frames(0.0))[0] == pytest.approx(0.25)  # touching: part A 0.25, part B nothing
    assert run(frames(0.015))[0] == pytest.approx(0.25)
    assert run(frames(0.03))[0] == pytest.approx(0.5)
    a, b = hung_on(0, 0.0), hung_on(0, 0.0)
    assert M.part_clearance(a, (b[0] + [0, -(g.OUT_W + 0.03), 0], I3)) == pytest.approx(0.03, abs=1e-6)


def test_letting_go_only_just_before_the_mark_is_not_enough():
    held = lambda t: gripper_at(g.tab_centre(*hung_on(0, t)), -g.TAB[1] / 2)  # noqa: E731
    exit_t = g.exit_time(0)
    for free_s, expected in ((0.5, 0.25), (0.9, 0.25), (1.0, 0.5), (3.0, 0.5)):
        frames = [(t, {"part_a": hung_on(0, t)}, held(t) if t < exit_t - free_s - 1e-6 else FAR)
                  for t in times(20.0, 27.0)]
        assert run(frames)[0] == pytest.approx(expected), free_s
    # let go long before, then touched again for one step just before the mark: the hang starts over
    frames = [(t, {"part_a": hung_on(0, t)}, held(t) if abs(t - 25.5) < 1e-6 else FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)


def test_the_part_must_be_on_the_hook_as_it_passes_the_mark():
    # hung, then knocked off before the exit: only "was on a hook"
    frames = [(t, {"part_a": hung_on(0, t)}, FAR) for t in times(15.0, 20.0)]
    frames += [(t, {"part_a": (np.array([0.3, -0.2, g.TOP - g.PART_BOTTOM]), I3)}, FAR) for t in times(20.05, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)
    # hung on a hook only after it passed the mark: the hook was judged already
    frames = [(t, {}, FAR) for t in times(0.0, 26.45)]
    frames += [(t, {"part_a": hung_on(0, t)}, FAR) for t in times(26.5, 30.0)]
    assert run(frames)[0] == pytest.approx(0.25)
    # credited parts stay credited after their hook has gone
    frames = [(t, {"part_a": hung_on(0, t)}, FAR) for t in times(25.0, 27.0)]
    frames += [(t, {"part_a": (np.array([0.0, -1.5, 0.3]), I3)}, FAR) for t in times(27.05, 30.0)]
    assert run(frames)[0] == pytest.approx(0.5)


def test_reset_clears_the_record():
    run([(t, {"part_a": hung_on(0, t)}, FAR) for t in times(25.0, 27.0)])
    M.reset_state()
    assert M.score({}) == 0.0 and not M.success({}) and not M._STATE["judged"]


# -- the demo, against rigid-grip parts --------------------------------------------------------------------------------

DT = 1 / 20
PLANS = M.demo()


def _spans(plan):
    t, out = 0.0, {}
    for w in plan:
        a, b = out.get(w.label, (t, t))
        out[w.label] = (min(a, t), t + w.duration)
        t += w.duration
    return out


@pytest.fixture(scope="module")
def played():
    """Each arm's joints at every control step of the demo, until 34 s."""
    out = {}
    for side, plan in PLANS.items():
        arm = ArmPlan(plan, station.BASES[side])
        qs = [arm.action7((k + 1) * DT)[0][:6] for k in range(round(34.0 / DT))]
        out[side] = (arm, np.array(qs))
    return out


def test_the_demo_is_reachable_and_within_joint_limits(played):
    for side, (arm, qs) in played.items():
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(qs >= ik.LOWER) and np.all(qs <= ik.UPPER), side


def test_the_demo_lets_go_well_before_each_hook_leaves_and_inside_reach():
    for side, plan in PLANS.items():
        part, hook, t0 = M.JOBS[side]
        spans = _spans(plan)
        assert spans["carry"][1] == pytest.approx(t0)
        assert spans["release"][1] < g.exit_time(hook) - 5
        y_release = g.hook_y(hook, spans["release"][1])
        assert abs(y_release - station.BASES[side][1]) < 0.12  # let go in front of its own arm


def _part_pose(side, t, q):
    """The part a demo arm carries, held rigidly from when its fingers close until they start opening; then hung."""
    part, hook, t0 = M.JOBS[side]
    spans = _spans(PLANS[side])
    if t < spans["close"][1]:
        return standing(part)
    if t < spans["release"][0]:
        tcp, _ = ik.tcp_pose(q, station.BASES[side])
        return tcp - [0.0, 0.0, M.TCP_ABOVE_ORIGIN], I3
    return np.array([g.HUNG_X, g.hook_y(hook, t), g.HUNG_Z]), I3


def test_the_tcp_follows_the_hook_and_threads_the_eye_without_touching_it(played):
    for side, (arm, qs) in played.items():
        part, hook, t0 = M.JOBS[side]
        spans = _spans(PLANS[side])
        for k in range(round(spans["carry"][1] / DT), round(spans["release"][0] / DT) - 1):
            t = (k + 1) * DT
            pos, R = _part_pose(side, t, qs[k])
            if t >= t0:
                assert pos[1] == pytest.approx(g.hook_y(hook, t), abs=1e-3)  # matching the hook's motion
            pts = g.part_points(pos, R)
            for i in range(g.N_HOOKS):
                for c, s in g.hook_boxes(i, t).values():
                    assert analysis.box_signed_distance(pts, c, I3, s).min() > 0.002, (side, t, i)
        tl = spans["lower"][1]
        pos, R = _part_pose(side, tl, qs[round(tl / DT) - 1])
        assert g.bar_through_eye(pos, R, g.hook_y(hook, tl))
        bar_gap = pos[2] + g.H_IN / 2 - g.BAR_TOP
        assert 0.0 < bar_gap < 0.006  # let go 3 mm above the bar


def _arm_hulls(side, q):
    poses = analysis.fk_link_poses_world(dict(zip(ARM_JOINTS, map(float, q), strict=True)), station.BASES[side])
    return analysis.world_hulls(poses)


def _min_hull_box(hulls, c, s):
    a = analysis.hulls_to_box(hulls, c, I3, s)[0]
    pts = g.box_surface(c, s, 0.01)
    b = min(analysis._signed(pts, h).min() for _, h in hulls)
    return min(a, b)


def test_every_arm_link_stays_clear_of_the_hooks_rail_and_other_parts(played):
    """Every 0.2 s of the demo: each arm's collision hulls at least 1.5 cm from every hook box and the rail, and from
    any part it isn't holding (the hung parts ride their hooks)."""
    rail = (np.array(g.RAIL_CENTRE), np.array(g.RAIL))
    worst = math.inf
    for k in range(0, round(34.0 / DT), 4):
        t = (k + 1) * DT
        poses = {side: _part_pose(side, t, played[side][1][k]) for side in PLANS}
        for side, (arm, qs) in played.items():
            hulls = _arm_hulls(side, qs[k])
            spans = _spans(PLANS[side])
            boxes = [rail] + [cs for i in range(g.N_HOOKS) for cs in g.hook_boxes(i, t).values()]
            for c, s in boxes:
                d = _min_hull_box(hulls, c, s)
                worst = min(worst, d)
                assert d > 0.015, (side, t)
            for other in PLANS:
                own_busy = other == side and spans["descend"][0] <= t <= spans["rise"][1]
                if own_busy:
                    continue  # its own part, between grasping and rising off it
                pos, R = poses[other]
                pts = g.part_points(pos, R, 0.01)
                d = min(analysis._signed(pts, h).min() for _, h in hulls)
                assert d > 0.015, (side, other, t)
    assert worst > 0.015


def _all_arm_hulls(played, k):
    """Both arms' collision hulls (every link, fingers at their commanded opening) at control step ``k``."""
    out = []
    for side, (arm, qs) in played.items():
        finger = float(arm.target((k + 1) * DT)[2])
        joints = dict(zip(ARM_JOINTS, map(float, qs[k]), strict=True)) | {"joint7": finger, "joint8": finger}
        out += analysis.world_hulls(analysis.fk_link_poses_world(joints, station.BASES[side]))
    return out


def test_the_demo_scores_full_under_the_monitor(played):
    """The demo played through the scorer's monitor, with rigid-grip parts and both arms' every link: each part hangs
    freely on its hook for the last second before the mark, with the arms well over 2 cm away."""
    M.reset_state()
    worst = math.inf
    for k in range(round(34.0 / DT)):
        t = (k + 1) * DT
        parts = {M.JOBS[side][0]: _part_pose(side, t, played[side][1][k]) for side in PLANS}
        hulls = _all_arm_hulls(played, k)
        M.update(parts, [g.hook_y(i, t) for i in range(g.N_HOOKS)], hulls, t)
        for side in PLANS:
            part, hook, _ = M.JOBS[side]
            if g.exit_time(hook) - M.HANG_S - 0.5 <= t <= g.exit_time(hook):
                worst = min(worst, M.arm_clearance(*parts[part], hulls))
    assert M.score({}) == pytest.approx(1.0) and M.success({})
    assert M._STATE["credited"] == {"part_a": 0, "part_b": 1}
    assert worst > 0.05
