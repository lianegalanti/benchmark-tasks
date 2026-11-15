"""Offline checks (no Isaac Sim): the plate's real dimensions, the conveyor's motion and the scene's clearances, the
scoring at every level, and the scripted demo, played against a rigid-grip model of the plates: reachable, within
joint limits, on time, the hole threaded over the hook's leg without touching it, and every arm link (as the URDF's
collision hulls) clear of the hooks, the rail and the plates it isn't holding.

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
    """A plate hanging upright by hole 0 on hook ``i`` at time ``t``."""
    return np.array([x, g.hook_y(i, t), g.HUNG_Z]), I3


def standing(name):
    return g.part_start(name), I3


STILL = (np.array([0.0, -g.SPEED, 0.0]), np.zeros(3))  # riding along with a hook, not turning


def run(frames, delay=lambda i, t: 0.0):
    """Play (t, parts, arm hulls) frames through the monitor from a fresh start; return (score, success). A plate
    given as (origin, rotation) rides at its hook's velocity without turning; give (origin, rotation, velocity,
    spin) to set them. Hook ``i`` runs ``delay(i, t)`` seconds behind its schedule (held back by the robot)."""
    M.reset_state()
    for t, parts, hulls in frames:
        full = {p: (*standing(p), np.zeros(3), np.zeros(3)) for p in g.PARTS}
        full.update({k: (v if len(v) == 4 else (*v, *STILL)) for k, v in parts.items()})
        M.update(full, [g.hook_y(i, t - delay(i, t)) for i in range(g.N_HOOKS)], hulls, t)
    return M.score({}), M.success({})


def times(t0, t1, dt=0.05):
    return np.arange(t0, t1 + 1e-9, dt)


def pitch(deg):
    """A turn of ``deg`` about the rail (y): the plate tipped toward or away from the robot."""
    a = math.radians(deg)
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


def roll(deg):
    """A turn of ``deg`` about the hook's leg (x): the plate swung sideways, in its own plane."""
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def hung_turned(i, t, R):
    """A plate turned by ``R`` about the point of hole 0's upper edge that rests on hook ``i``'s leg."""
    pos, _ = hung_on(i, t)
    contact = pos + [0, 0, g.R_HOLE]
    return contact - R @ [0, 0, g.R_HOLE], R


# -- layout and motion -------------------------------------------------------------------------------------------------

def test_it_loads_with_only_the_two_plates_free():
    assert TASK.free_objects == g.PARTS
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0
    assert sum(1 for o in M.OBJECTS if o.name.startswith("hook")) == 3 * g.N_HOOKS
    assert sum(1 for o in M.OBJECTS if o.name.startswith("block")) == 8


def test_poses_are_plain_floats():
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in spec.size), spec.name
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", ())), spec.name
        assert spec.mass is None or type(spec.mass) is float, spec.name


def test_the_plate_is_the_real_fitting():
    """A 3-hole flat plate strut fitting: 1/4 x 1-5/8 x 5-3/8 in, 9/16 in holes on 1-7/8 in centres, 13/16 in from the
    ends; steel, 0.257 kg with its round holes."""
    assert (g.THICK, g.WIDTH, g.LENGTH) == pytest.approx((0.00635, 0.041275, 0.136525))
    assert (g.HOLE, g.PITCH, g.END) == pytest.approx((0.0142875, 0.047625, 0.0206375))
    assert 2 * g.END + 2 * g.PITCH == pytest.approx(g.LENGTH) and g.HEIGHT == pytest.approx(g.LENGTH)
    assert g.MASS == pytest.approx(0.257)
    vol = sum(float(np.prod(s)) for _, s in g.PART_BOXES.values())  # the model's boxes: square holes
    assert vol == pytest.approx(g.LENGTH * g.WIDTH * g.THICK - 3 * g.HOLE ** 2 * g.THICK)
    com = sum(float(np.prod(s)) * c for c, s in g.PART_BOXES.values()) / vol
    assert com == pytest.approx(g.COM, abs=1e-9)
    assert g.hole_above_com(*standing("plate_a"), 0) == pytest.approx(g.PITCH)  # hole 0: 4.8 cm above it
    assert g.hole_above_com(np.zeros(3), g.flipped(), 2) == pytest.approx(g.PITCH)
    assert g.hole_above_com(np.zeros(3), I3, 1) == pytest.approx(0.0)


def test_the_usd_part_matches_the_geometry():
    text = (HERE / "assets" / "strut_plate.usda").read_text()
    assert text.count('def Cube "') == len(g.PART_BOXES) == 10
    for name, (centre, size) in g.PART_BOXES.items():
        block = text.split(f'def Cube "{name}"')[1].split("}")[0]
        scale = [float(v) for v in re.search(r"scale = \(([^)]*)\)", block).group(1).split(",")]
        assert scale == pytest.approx(size, abs=1e-6), name
        m = re.search(r"translate = \(([^)]*)\)", block)
        shift = [float(v) for v in m.group(1).split(",")] if m else [0.0, 0.0, 0.0]
        assert shift == pytest.approx(centre, abs=1e-6), name
    assert f"physics:mass = {g.MASS}" in text
    com = [float(v) for v in re.search(r"centerOfMass = \(([^)]*)\)", text).group(1).split(",")]
    assert com == pytest.approx(g.COM, abs=1e-6)


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
        assert tip_hi[2] - bar_hi[2] == pytest.approx(g.TIP_RISE) and bar_hi[2] == pytest.approx(g.BAR_TOP)
        assert boxes["bar"][1][1:] == pytest.approx([0.004, 0.004])  # 4 mm wire
    rail_y = (g.RAIL_CENTRE[1] - g.RAIL[1] / 2, g.RAIL_CENTRE[1] + g.RAIL[1] / 2)
    for i in range(g.N_HOOKS):  # the rail is over every hook from the start to the run limit (45 s)
        assert rail_y[0] < g.hook_y(i, 45.0) and g.hook_y(i, 0.0) < rail_y[1]


def test_hooks_start_off_the_table_and_leave_in_order():
    assert all(g.hook_y(i, 0.0) > g.TABLE_EDGE_Y for i in range(g.N_HOOKS))
    exits = [g.exit_time(i) for i in range(g.N_HOOKS)]
    assert exits == pytest.approx([26.0, 32.0, 38.0])
    assert exits[-1] * 20 < 900  # the last hook leaves well inside the 900-step run limit
    assert g.EXIT_Y > -g.TABLE_EDGE_Y  # the mark is on the table


def test_clearances():
    # threading: the leg and its upturn (8 mm in all) through the 14.3 mm hole, 3.1 mm above and below
    assert g.THREAD_Z + g.R_HOLE - g.TIP_Z1 == pytest.approx(g.TIP_Z0 - (g.THREAD_Z - g.R_HOLE))
    assert g.THREAD_Z + g.R_HOLE - g.TIP_Z1 > 0.003
    assert (g.HOLE - g.WIRE) / 2 > 0.005  # the wire in the hole: 5 mm either side
    assert g.lowest_z(*hung_on(0, 0.0)) - g.TOP > 0.025  # hung, the plate's lower end is 2.7 cm above the table
    assert -FINGER_OPEN * 2 - g.WIDTH > 0.05  # open fingers clear the plate's edges by 2.6 cm a side
    assert -M.RELEASE_OPEN * 2 - g.WIDTH > 0.029  # let-go opening: 1.5 cm a side
    assert SLOT_PLAY_X > 0.002 and SLOT_PLAY_Y > 0.004
    for name in g.PARTS:  # standing plates are clear of the hooks' path, and inside their blocks without touching
        pts = g.part_points(*standing(name))
        for c, s in g.hook_boxes(0, 0.0).values():
            c = c.copy()
            c[1] = g.PART_XY[name][1]
            assert analysis.box_signed_distance(pts, c, I3, s).min() > 0.05
        for c, s in g.holder_boxes(name).values():
            assert analysis.box_signed_distance(pts, c, I3, s).min() > 0.001
    # the pads pinch the plate's long edges beside hole 0, 1.3 cm out from the wire on either side
    pad_tips = g.part_start("plate_a")[2] + M.TCP_ABOVE_ORIGIN - PAD_TIP_PAST_TCP
    assert pad_tips - g.part_start("plate_a")[2] == pytest.approx(g.PART_TOP - M.GRIP_DEPTH - PAD_TIP_PAST_TCP)
    assert g.WIDTH / 2 - g.WIRE / 2 > 0.018
    assert pad_tips > g.TOP + g.SLOT_DEPTH + 0.05  # far above the block


SLOT_PLAY_X = g.SLOT[0] - g.THICK
SLOT_PLAY_Y = g.SLOT[1] - g.WIDTH


def test_hung_plate_rests_on_the_leg_by_hole_0_above_its_centre_of_mass():
    pos, R = hung_on(1, 3.0)
    y = g.hook_y(1, 3.0)
    assert g.hole_on_bar(pos, R, y) == 0 and g.rest_gap(pos, R, y) == pytest.approx(0.0)
    assert not g.bar_through_hole(pos, R, y + 0.02)  # beside the hole
    hole_top = pos[2] + g.R_HOLE
    assert hole_top == pytest.approx(g.BAR_TOP)
    assert g.hole_above_com(pos, R, 0) > 0.04  # it hangs below the hook


def test_the_hole_must_be_an_end_hole_above_the_centre_of_mass():
    y = g.hook_y(0, 0.0)
    # hung by hole 1 (at the centre of mass): through a hole, but not hanging by an end hole
    pos = np.array([g.HUNG_X, y, g.HUNG_Z + g.PITCH])
    assert g.hole_on_bar(pos, I3, y) == 1 and g.rest_gap(pos, I3, y) is None
    # turned end for end and hung by hole 2: fine
    pos = np.array([g.HUNG_X, y, g.HUNG_Z - 2 * g.PITCH])
    assert g.hole_on_bar(pos, g.flipped(), y) == 2 and g.rest_gap(pos, g.flipped(), y) == pytest.approx(0.0)
    # upright, with the leg through hole 2 at the bottom (propped up from below): the hole is below the centre of mass
    pos = np.array([g.HUNG_X, y, g.HUNG_Z + 2 * g.PITCH])
    assert g.hole_on_bar(pos, I3, y) == 2 and g.rest_gap(pos, I3, y) is None
    # turned to face along the rail, or lying flat on the leg: not through a hole
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert not g.bar_through_hole(np.array([g.HUNG_X, y, g.HUNG_Z]), turned, y)
    assert not g.bar_through_hole(np.array([g.HUNG_X, y, g.BAR_TOP + 0.01]), pitch(90), y)


# -- scoring -----------------------------------------------------------------------------------------------------------

def test_untouched_scores_zero():
    assert run([(t, {}, FAR) for t in times(0.0, 40.0, 0.5)]) == (0.0, False)


def test_lifted_only():
    lifted = {"plate_a": (g.part_start("plate_a") + [0, 0, 0.04], I3)}
    frames = [(1.0, lifted, FAR)] + [(t, {}, FAR) for t in times(2.0, 40.0, 0.5)]
    assert run(frames) == (pytest.approx(0.1), False)
    in_slot = {"plate_a": (g.part_start("plate_a") + [0, 0, 0.02], I3)}  # raised 2 cm: still in its slot
    assert run([(1.0, in_slot, FAR)]) == (0.0, False)


def _pinch_top(pos, R, finger=-g.WIDTH / 2):
    """A gripper pinching the plate's long edges at its top end, as the demo holds it."""
    return gripper_at(pos + R @ [0, 0, M.TCP_ABOVE_ORIGIN], finger)


def test_threaded_but_still_held_when_the_hook_leaves():
    def frame(t):
        pos, R = hung_on(0, t)
        return t, {"plate_a": (pos, R)}, _pinch_top(pos, R)

    assert run([frame(t) for t in times(20.0, 27.0)]) == (pytest.approx(0.25), False)


def test_hung_and_let_go_scores_half_per_plate():
    frames = [(t, {"plate_a": hung_on(0, t)}, FAR) for t in times(20.0, 27.0)]
    assert run(frames) == (pytest.approx(0.5), False)
    assert M._STATE["credited"]["plate_a"] == 0


def test_both_hung_on_their_own_hooks_succeeds():
    frames = [(t, {"plate_a": hung_on(0, t), "plate_b": hung_on(1, t)}, FAR) for t in times(20.0, 33.0)]
    assert run(frames) == (pytest.approx(1.0), True)
    frames = [(t, {"plate_a": hung_on(2, t), "plate_b": hung_on(1, t)}, FAR) for t in times(20.0, 39.0)]
    assert run(frames) == (pytest.approx(1.0), True)  # any two hooks


def test_hung_upside_down_by_hole_2_also_counts():
    frames = [(t, {"plate_a": (hung_on(0, t)[0] - [0, 0, 2 * g.PITCH], g.flipped())}, FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.5)


def test_hung_by_the_middle_hole_gets_no_hanging_credit():
    """Hole 1 is at the centre of mass: hung there, the plate isn't hanging below the hook."""
    frames = [(t, {"plate_a": (hung_on(0, t)[0] + [0, 0, g.PITCH], I3)}, FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)


def test_both_on_one_hook_counts_once():
    frames = [(t, {"plate_a": hung_on(0, t, g.HUNG_X), "plate_b": hung_on(0, t, g.HUNG_X + 0.04)}, FAR)
              for t in times(20.0, 39.0)]
    assert run(frames) == (pytest.approx(0.75), False)


def test_a_gripper_holding_the_lower_end_gets_no_hanging_credit():
    """The earlier review's case: the leg through the hole and resting on it, but a gripper holds the plate's lower
    end, 11 cm from the hole; open around it without touching, still within 2 cm: no credit either."""
    def frame(t, finger=-g.WIDTH / 2):
        pos, R = hung_on(0, t)
        low = pos + R @ [0, 0, g.PART_BOTTOM + 0.012]
        assert np.linalg.norm(low - pos) > 0.1
        return t, {"plate_a": (pos, R)}, gripper_at(low, finger)

    assert run([frame(t) for t in times(20.0, 27.0)]) == (pytest.approx(0.25), False)
    assert run([frame(t, -g.WIDTH / 2 - 0.012) for t in times(20.0, 27.0)]) == (pytest.approx(0.25), False)


def test_a_gripper_propping_the_plate_from_below_gets_no_hanging_credit():
    def frame(t, drop):
        pos, R = hung_on(0, t)
        under = pos + R @ [0, 0, g.PART_BOTTOM] - [0, 0, drop]
        return t, {"plate_a": (pos, R)}, gripper_at(under - [0, 0, PAD_TIP_PAST_TCP], up=True)

    assert run([frame(t, 0.0) for t in times(20.0, 27.0)])[0] == pytest.approx(0.25)  # pad tips pushing up on it
    assert run([frame(t, 0.03) for t in times(20.0, 27.0)])[0] == pytest.approx(0.5)  # 3 cm below: let go


@pytest.mark.parametrize("part_box", sorted(g.PART_BOXES))
def test_any_arm_link_within_2cm_of_any_part_of_it_blocks_credit(part_box):
    """A 1 cm stand-in link beside each of the plate's ten boxes: 1.5 cm away blocks credit, 2.5 cm away doesn't."""
    def frames(d):
        out = []
        for t in times(24.0, 27.0):
            pos, R = hung_on(0, t)
            c, sz = g.PART_BOXES[part_box]
            side = pos + R @ (c + [-(sz[0] / 2 + d + 0.005), 0, 0])  # on the robot's side of that box
            out.append((t, {"plate_a": (pos, R)}, box_hull(side, (0.01, 0.01, 0.01))))
        return out

    assert run(frames(0.015))[0] == pytest.approx(0.25)
    assert run(frames(0.025))[0] == pytest.approx(0.5)


def test_the_hook_must_support_it():
    # held with its hole's edge 1 cm above the leg (nothing in the scene touching it): not resting on the hook
    frames = [(t, {"plate_a": (hung_on(0, t)[0] + [0, 0, 0.01], I3)}, FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)
    frames = [(t, {"plate_a": (hung_on(0, t)[0] + [0, 0, 0.004], I3)}, FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.5)  # within the 5 mm tolerance
    pos, R = hung_on(0, 0.0)
    assert g.rest_gap(pos, R, g.hook_y(0, 0.0)) == pytest.approx(0.0)
    assert g.rest_gap(pos, R, g.hook_y(0, 0.0) + 0.02) is None


def test_it_must_hang_plumb():
    for deg, expected in ((5, 0.5), (9, 0.5), (12, 0.25), (30, 0.25)):
        for turn in (pitch, roll):
            frames = [(t, {"plate_a": hung_turned(0, t, turn(deg))}, FAR) for t in times(20.0, 27.0)]
            assert run(frames)[0] == pytest.approx(expected), (deg, turn.__name__)


def test_it_must_have_settled():
    """Still swinging or sliding in the last second: no hanging credit; settled for the whole second: credit."""
    vel = np.array([0.0, -g.SPEED, 0.0])
    for spin, speed, expected in ((0.0, 0.0, 0.5), (0.4, 0.0, 0.5), (0.8, 0.0, 0.25), (0.0, 0.02, 0.5),
                                  (0.0, 0.05, 0.25)):
        frames = [(t, {"plate_a": (*hung_on(0, t), vel + [0, 0, speed], np.array([spin, 0, 0]))}, FAR)
                  for t in times(20.0, 27.0)]
        assert run(frames)[0] == pytest.approx(expected), (spin, speed)
    # swinging until 0.5 s before the mark, then still: not settled for the whole second
    exit_t = g.exit_time(0)
    frames = [(t, {"plate_a": (*hung_on(0, t), vel, np.array([1.0 if t < exit_t - 0.5 else 0.0, 0, 0]))}, FAR)
              for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)


def test_a_hook_the_robot_held_back_credits_nothing():
    """A real slip-clutch drive stalls when an arm holds a hook: a hook more than 1 cm (0.2 s) behind its schedule
    credits no plate, even if one later hangs freely on it."""
    def frames(stop_s):
        delay = lambda i, t: stop_s if (i == 0 and t > 15.0) else 0.0  # noqa: E731
        fr = [(t, {"plate_a": hung_on(0, t - delay(0, t))}, FAR) for t in times(10.0, 28.0)]
        return fr, delay

    assert run(*frames(0.5))[0] == pytest.approx(0.25)  # stopped 0.5 s: 2.5 cm behind
    assert run(*frames(0.1))[0] == pytest.approx(0.5)  # 5 mm behind: within tolerance
    assert run(*frames(0.0))[0] == pytest.approx(0.5)


def test_the_other_plate_must_not_touch_it():
    """Nothing but its hook may touch a hung plate: the other plate (held by the other arm, say) against its edge
    blocks the credit; 3 cm away it doesn't."""
    def frames(gap):
        out = []
        for t in times(20.0, 27.0):
            pos, R = hung_on(0, t)
            out.append((t, {"plate_a": (pos, R), "plate_b": (pos + [0, -(g.WIDTH + gap), 0], R)}, FAR))
        return out

    assert run(frames(0.0))[0] == pytest.approx(0.25)  # touching: plate A 0.25, plate B nothing
    assert run(frames(0.015))[0] == pytest.approx(0.25)
    assert run(frames(0.03))[0] == pytest.approx(0.5)
    a = hung_on(0, 0.0)
    assert M.part_clearance(a, (a[0] + [0, -(g.WIDTH + 0.03), 0], I3)) == pytest.approx(0.03, abs=1e-6)


def test_letting_go_only_just_before_the_mark_is_not_enough():
    held = lambda t: _pinch_top(*hung_on(0, t))  # noqa: E731
    exit_t = g.exit_time(0)
    for free_s, expected in ((0.5, 0.25), (0.9, 0.25), (1.0, 0.5), (3.0, 0.5)):
        frames = [(t, {"plate_a": hung_on(0, t)}, held(t) if t < exit_t - free_s - 1e-6 else FAR)
                  for t in times(20.0, 27.0)]
        assert run(frames)[0] == pytest.approx(expected), free_s
    # let go long before, then touched again for one step just before the mark: the hang starts over
    frames = [(t, {"plate_a": hung_on(0, t)}, held(t) if abs(t - 25.5) < 1e-6 else FAR) for t in times(20.0, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)


def test_the_plate_must_be_on_the_hook_as_it_passes_the_mark():
    # hung, then knocked off before the exit: only "was on a hook"
    frames = [(t, {"plate_a": hung_on(0, t)}, FAR) for t in times(15.0, 20.0)]
    frames += [(t, {"plate_a": (np.array([0.3, -0.2, g.TOP - g.PART_BOTTOM]), I3)}, FAR) for t in times(20.05, 27.0)]
    assert run(frames)[0] == pytest.approx(0.25)
    # hung on a hook only after it passed the mark: the hook was judged already
    frames = [(t, {}, FAR) for t in times(0.0, 26.45)]
    frames += [(t, {"plate_a": hung_on(0, t)}, FAR) for t in times(26.5, 30.0)]
    assert run(frames)[0] == pytest.approx(0.25)
    # credited plates stay credited after their hook has gone
    frames = [(t, {"plate_a": hung_on(0, t)}, FAR) for t in times(25.0, 27.0)]
    frames += [(t, {"plate_a": (np.array([0.0, -1.5, 0.3]), I3)}, FAR) for t in times(27.05, 30.0)]
    assert run(frames)[0] == pytest.approx(0.5)


def test_reset_clears_the_record():
    run([(t, {"plate_a": hung_on(0, t)}, FAR) for t in times(25.0, 27.0)])
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
        if t >= spans["carry"][0]:
            tcp = tcp + M.TRACKING  # the arm's measured lag and sag, which the plan aims ahead of
        return tcp - [0.0, 0.0, M.TCP_ABOVE_ORIGIN], I3
    return np.array([g.HUNG_X, g.hook_y(hook, t), g.HUNG_Z]), I3


def test_the_tcp_follows_the_hook_and_threads_the_hole_without_touching_it(played):
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
        assert g.hole_on_bar(pos, R, g.hook_y(hook, tl)) == 0
        assert g.rest_gap(pos, R, g.hook_y(hook, tl)) == pytest.approx(0.002, abs=1e-4)  # let go 2 mm above the leg


def _finger(arm, t):
    """The fingers' opening at ``t``: as commanded, but held apart by the plate's 41.3 mm width when closed on it."""
    return min(float(arm.target(t)[2]), -g.WIDTH / 2)


def _arm_hulls(side, q, finger):
    joints = dict(zip(ARM_JOINTS, map(float, q), strict=True)) | {"joint7": finger, "joint8": finger}
    return analysis.world_hulls(analysis.fk_link_poses_world(joints, station.BASES[side]))


def _min_hull_box(hulls, c, s):
    a = analysis.hulls_to_box(hulls, c, I3, s)[0]
    pts = g.box_surface(c, s, 0.01)
    b = min(analysis._signed(pts, h).min() for _, h in hulls)
    return min(a, b)


def test_every_arm_link_stays_clear_of_the_hooks_rail_and_other_parts(played):
    """Every 0.2 s of the demo: each arm's collision hulls (fingers as wide as the plate while closed on it) at least
    1.5 cm from every hook box and the rail, and from any plate it isn't holding (the hung plates ride their hooks)."""
    rail = (np.array(g.RAIL_CENTRE), np.array(g.RAIL))
    worst = math.inf
    for k in range(0, round(34.0 / DT), 4):
        t = (k + 1) * DT
        poses = {side: _part_pose(side, t, played[side][1][k]) for side in PLANS}
        for side, (arm, qs) in played.items():
            hulls = _arm_hulls(side, qs[k], _finger(arm, t))
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
        finger = _finger(arm, (k + 1) * DT)
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
        parts = {M.JOBS[side][0]: (*_part_pose(side, t, played[side][1][k]), *STILL) for side in PLANS}
        hulls = _all_arm_hulls(played, k)
        M.update(parts, [g.hook_y(i, t) for i in range(g.N_HOOKS)], hulls, t)
        for side in PLANS:
            part, hook, _ = M.JOBS[side]
            if g.exit_time(hook) - M.HANG_S - 0.5 <= t <= g.exit_time(hook):
                worst = min(worst, M.arm_clearance(*parts[part][:2], hulls))
    assert M.score({}) == pytest.approx(1.0) and M.success({})
    assert M._STATE["credited"] == {"plate_a": 0, "plate_b": 1}
    assert worst > 0.05
