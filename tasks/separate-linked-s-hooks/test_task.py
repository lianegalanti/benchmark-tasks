"""Offline checks (no Isaac Sim): the layout and its clearances, the scoring at every level, and the scripted demo,
played against a rigid-grip model of hook B: reachable, within joint limits, and with every arm link (as the URDF's
collision hulls) clear of the table, the display and hook A, and of hook B except where the fingers grip it.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

import demo
import geometry as g
from yam_arena import analysis, ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, READY
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
LEFT = np.asarray(station.BASES["left"], float)
RA, RB = g.rot(g.A_START_XYZW), g.rot(g.B_START_XYZW)
A_REST = g.A_START - [0.0, 0.0, g.DROP]
B_REST = g.B_START - [0.0, 0.0, 2 * g.DROP]
ZERO = np.zeros(3)


def on_peg(x=g.PEG_TIP_X + demo.ON_PEG):
    """Hook B hanging upright on the peg by its top curl, its plane at ``x``."""
    curl = np.array([x, g.PEG_Y, g.PEG_Z - (g.CHORD - g.ROD_R - g.SUPPORT_R)])
    return curl - [0.0, 0.0, g.H], demo.UPRIGHT_B


FAR = [(np.array([0.0, 0.6, 1.2]), 0.094, np.array([0.0, 0.6, 1.3]), np.array([0.0, 0.6, 1.2]))] * 2


def step(a, b, grippers=FAR, moving=False):
    v = np.array([0.0, 0.0, 0.1]) if moving else ZERO
    M.update([(a[0], a[1], v, ZERO), (b[0], b[1], ZERO, ZERO)], grippers)
    return M.score({}), M.success({})


def gripper_at(tcp, R_tcp, opening):
    tcp = np.asarray(tcp, float)
    return (tcp, opening, tcp - R_tcp[:, 2] * 0.134, tcp + R_tcp[:, 2] * 0.013)


def _support_points(name, n=81):
    a0, a1 = next((a0, a1) for nm, a0, a1 in g.SUPPORTS if nm == name)
    return g.seg_points(a0, a1, n)


def _box_dist(points, center, size):
    return float(analysis.box_signed_distance(np.asarray(points), np.asarray(center), np.eye(3), np.asarray(size)).min())


def _fixed_boxes():
    boxes = [(np.array(M.BOARD.pos), np.array(M.BOARD.size))]
    boxes += [(np.array(b.pos), np.array(b.size)) for b in M.BRACKETS]
    return boxes


def hook_clear_of_display(pos, R):
    """Distance (m) from the hook's surface to the board, brackets, rail and peg."""
    pts = g.world(pos, R, g.DENSE)
    d = min(_box_dist(pts, c, s) for c, s in _fixed_boxes())
    for name in ("rail", "peg"):
        sp = _support_points(name)
        d = min(d, float(np.linalg.norm(pts[:, None] - sp[None], axis=2).min()) - g.SUPPORT_R)
    return d - g.ROD_R


# -- layout -----------------------------------------------------------------------------------------------------------

def test_it_loads_with_the_two_hooks_free():
    assert TASK.free_objects == ("hook_a", "hook_b")
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0
    assert (HERE / "assets" / "s_hook.usda").is_file()


def test_poses_are_plain_floats():
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", ())), spec.name


def test_the_usd_hook_matches_the_geometry():
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(HERE / "assets" / "s_hook.usda"))
    cache = UsdGeom.XformCache()
    caps = [p for p in stage.Traverse() if p.IsA(UsdGeom.Capsule)]
    assert len(caps) == len(g.SEGMENTS)
    for p, (a, b) in zip(caps, g.SEGMENTS):
        cap = UsdGeom.Capsule(p)
        h = cap.GetHeightAttr().Get()
        m = cache.GetLocalToWorldTransform(p)
        ends = [np.array(m.Transform(Gf.Vec3d(s * h / 2, 0, 0))) for s in (-1, 1)]
        assert np.allclose(ends, [a, b], atol=1e-6)
        assert cap.GetRadiusAttr().Get() == pytest.approx(g.ROD_R)


def test_the_hook_is_an_s_with_generous_curls():
    assert g.HEIGHT == pytest.approx(0.15) and g.WIDTH == pytest.approx(0.066)
    assert np.allclose(g.dense().mean(axis=0), 0.0, atol=1e-9)        # centre of mass at the origin (symmetric)
    assert g.GAP >= 0.025                                              # each curl's opening: 4.0 cm clear
    assert g.INNER - g.ROD_R >= 0.015 and g.INNER - g.SUPPORT_R >= 0.015  # room round a rod or a support inside


def test_the_start_is_two_hooks_hooked_together_hanging_from_the_rail():
    assert g.on_support(g.A_START, RA) == "rail" and g.on_support(g.B_START, RB) is None
    assert g.linked(g.A_START, RA, g.B_START, RB)
    assert g.gap(g.A_START, RA, g.B_START, RB) == pytest.approx(g.DROP, abs=2e-4)   # 1 mm apart before settling
    assert g.gap(A_REST, RA, B_REST, RB) == pytest.approx(0.0, abs=2e-4)            # touching once settled
    pts = g.world(g.A_START, RA, g.DENSE)
    rail = np.hypot(pts[:, 0] - g.RAIL_X, pts[:, 2] - g.RAIL_Z).min() - g.ROD_R - g.SUPPORT_R
    assert rail == pytest.approx(g.DROP, abs=2e-4)
    assert g.lowest_z(g.B_START, RB) - g.TOP > 0.05                   # hook B hangs 6 cm above the table
    assert hook_clear_of_display(g.A_START, RA) > -1e-3                # A touches only the rail (1 mm off it)
    assert hook_clear_of_display(g.B_START, RB) > 0.03


def test_a_hook_hung_on_the_peg_is_clear_of_the_board_and_the_other_hook():
    pos, R = on_peg()
    assert g.on_support(pos, R) == "peg"
    assert _box_dist(g.world(pos, R, g.DENSE), *_fixed_boxes()[0]) - g.ROD_R > 0.06
    assert g.gap(A_REST, RA, pos, R) > 0.05
    assert g.lowest_z(pos, R) - g.TOP > 0.05


def test_the_display_is_in_front_of_the_left_arm_and_off_the_camera_stand():
    stand = station.camera_stand(station.top_camera("calibrated"))["stand_foot"][2]
    for x, y in ((g.RAIL_X, g.HOOK_Y), (g.PEG_TIP_X, g.PEG_Y), (g.BOARD_POS[0], g.BOARD_POS[1])):
        assert 0.10 <= x <= 0.50 and abs(y - stand[1]) + abs(x - stand[0]) > 0.3
    assert g.BOARD_POS[1] - g.BOARD[1] / 2 > -0.1 and g.PEG_Y > -0.03  # left of the midline but for the board's end and the peg


# -- scoring ----------------------------------------------------------------------------------------------------------

A0, B0 = (A_REST, RA), (B_REST, RB)


def test_untouched_scores_zero():
    M._reset_state()
    assert step(A0, B0) == (0.0, False)


def test_hook_b_lifted_off_scores_a_quarter_and_keeps_it():
    M._reset_state()
    assert step(A0, (B_REST + [0, 0, 0.008], RB), moving=True)[0] == 0.0   # 8 mm: not yet
    assert step(A0, (B_REST + [0, 0, 0.012], RB), moving=True)[0] == 0.25
    assert g.linked(A_REST, RA, B_REST + [0, 0, 0.012], RB)
    assert step(A0, B0) == (0.25, False)                                     # hung back on: kept


def test_unhooked_scores_half():
    M._reset_state()
    b = (B_REST + [0.0, -demo.SIDE, demo.LIFT], RB)
    assert not g.linked(A_REST, RA, *b)
    held = [gripper_at(demo.grip_tcp(*b), demo.grip_rotation(RB), 0.01), FAR[1]]
    assert step(A0, b, held, moving=True) == (0.5, False)


def test_lifted_but_still_through_the_curl_is_not_unhooked():
    M._reset_state()
    assert step(A0, (B_REST + [0, 0, 0.015], RB))[0] == 0.25
    assert step(A0, (B_REST + [0, 0.02, 0.015], RB))[0] == 0.25   # moved toward its tip: still round A's curl


def test_each_hook_alone_on_its_own_support_succeeds_either_way_round():
    M._reset_state()
    assert step(A0, on_peg()) == (0.5, True)
    M._reset_state()
    peg_a = on_peg()[0] + [-0.01, 0.0, 0.0], g.rz(-math.pi / 2)            # turned the other way: still a hang
    rail_b = (g.A_START + [0.0, 0.06, -g.DROP], RA)
    assert step(rail_b, peg_a) == (0.5, True)


def test_both_on_the_rail_or_hanging_from_each_other_is_not_success():
    M._reset_state()
    other = (A_REST + [0.0, -0.07, 0.0], RA)
    assert g.on_support(*other) == "rail" and not g.linked(A_REST, RA, *other)
    assert step(A0, other) == (0.5, False)
    M._reset_state()
    a_on_peg = (on_peg()[0], RB)
    hanging = (a_on_peg[0] + (B_REST - A_REST), RB @ RA.T @ RB)
    assert step(a_on_peg, hanging)[1] is False


def test_no_success_while_held_swinging_touching_or_with_a_gripper_near():
    pos, R = on_peg()
    M._reset_state()
    assert step(A0, (pos, R), moving=True) == (0.5, False)
    near = [gripper_at(demo.grip_tcp(pos, R), demo.grip_rotation(R), 0.094), FAR[1]]
    M._reset_state()
    assert step(A0, (pos, R), near) == (0.5, False)                        # let go but fingers still round it
    held = [gripper_at(demo.grip_tcp(pos, R), demo.grip_rotation(R), 0.01), FAR[1]]
    M._reset_state()
    assert step(A0, (pos, R), held) == (0.5, False)


def test_putting_a_hook_down_zeroes_everything_for_good():
    M._reset_state()
    step(A0, on_peg(), moving=True)
    flat = g.rot((math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)))   # lying on the table
    on_table = np.array([0.2, 0.2, g.TOP + g.ROD_R + 0.001])
    assert step(A0, (on_table, flat))[0] == 0.0
    assert step(A0, on_peg()) == (0.0, False)
    M._reset_state()
    assert step(A0, (np.array([0.3, 0.7, g.TOP - 0.4]), RB))[0] == 0.0     # dropped off the table


def test_reset_clears_the_memory():
    M._reset_state()
    step(A0, on_peg(), moving=True)
    M._reset_state()
    assert M.score({}) == 0.0 and M._STATE["best"] == 0.0 and not M._STATE["down"]


# -- the demo, against a rigid-grip hook B ----------------------------------------------------------------------------

class FakeCtx:
    """Hook A stays on the rail. Hook B hangs still until the left fingers close on it, then follows the grip
    rigidly; once they open it drops onto the peg and hangs upright there."""

    def __init__(self):
        self.left, self.right = demo.Arm("left"), demo.Arm("right")
        self.poses = {"hook_a": (A_REST.copy(), RA.copy()), "hook_b": (B_REST.copy(), RB.copy())}
        self.attached, self.released, self.offset = False, False, None

    def hook(self, name):
        p, R = self.poses[name]
        return p.copy(), R.copy()

    def update(self):
        L = self.left
        p, R = self.poses["hook_b"]
        if not self.attached and not self.released and L.grip < 0.05:
            self.attached = True
            self.offset = (L.rot.T @ (p - L.pos), L.rot.T @ R)
        elif self.attached and L.grip > 0.3:
            self.attached, self.released = False, True
            self.poses["hook_b"] = on_peg(p[0])
        if self.attached:
            self.poses["hook_b"] = (L.pos + L.rot @ self.offset[0], L.rot @ self.offset[1])


def _hulls(q, grip, holding=False):
    qd = dict(zip(ARM_JOINTS, q))
    finger = FINGER_OPEN * grip
    if holding:  # the fingers stop on the rod's sides
        finger = min(finger, -g.ROD_R)
    qd["joint7"] = qd["joint8"] = finger
    return [h for h in analysis.world_hulls(analysis.fk_link_poses_world(qd, LEFT)) if h[0] != "base"]


def _dist(points, hulls):
    return min(analysis._signed(points, h).min() for _, h in hulls)


@pytest.fixture(scope="module")
def played():
    ctx = FakeCtx()
    log = []
    for _ in demo.separate(ctx):
        ctx.update()
        L = ctx.left
        pb, Rb = ctx.poses["hook_b"]
        log.append(dict(q=L.q.copy(), grip=L.grip, tcp=L.pos.copy(), rot=L.rot.copy(), pb=pb.copy(), Rb=Rb.copy(),
                        attached=ctx.attached, released=ctx.released))
    return ctx, log


def test_the_demo_is_reachable_within_limits_and_time(played):
    ctx, log = played
    assert ctx.left.worst_mm < 0.5 and ctx.left.worst_deg < 0.1
    qs = np.array([e["q"] for e in log])
    assert np.all(qs >= ik.LOWER - 1e-9) and np.all(qs <= ik.UPPER + 1e-9)
    assert len(log) / demo.CONTROL_HZ < 40  # s; the run limit is 90 s
    assert ctx.attached is False and ctx.released is True


def test_the_demo_grips_the_stroke_with_the_fingers_closing_across_it(played):
    _, log = played
    first = next(e for e in log if e["attached"])
    assert np.linalg.norm(first["tcp"] - demo.grip_tcp(B_REST, RB)) < 1e-3
    c = first["rot"][:, 0]
    assert abs(c @ (RB @ g.STROKE_DIR)) < 1e-6 and abs(c @ RB[:, 1]) < 0.01   # square to the stroke, in the plane


def test_hook_b_leaves_hook_a_through_its_opening_without_rubbing(played):
    _, log = played
    worst, unhooked = 1.0, None
    for i, e in enumerate(log):
        rise = e["pb"][2] - B_REST[2]
        if e["attached"] and rise > 0.001:  # once lifting (B rests on A until then): never closer than it has risen
            worst = min(worst, g.gap(A_REST, RA, e["pb"], e["Rb"]) / min(rise, 0.005))
            if unhooked is None and not g.linked(A_REST, RA, e["pb"], e["Rb"]):
                unhooked = i
    assert unhooked is not None
    print("hook gap while carried", round(worst, 4))
    assert worst > 0.95  # the hooks part as B rises and stay at least 5 mm apart while it leaves through the opening


def test_hook_b_is_carried_clear_of_the_display_and_slides_onto_the_peg(played):
    _, log = played
    worst = 1.0
    for e in log:
        if e["attached"] and e["tcp"][2] > demo.grip_tcp(B_REST, RB)[2] + 0.01:
            lowered = g.world(e["pb"], e["Rb"], g.CURLS[0])[2] < g.PEG_Z - 0.001
            if not (lowered and g.on_support(e["pb"], e["Rb"]) == "peg"):  # until it is let down onto the peg
                worst = min(worst, hook_clear_of_display(e["pb"], e["Rb"]))
            assert g.lowest_z(e["pb"], e["Rb"]) - g.TOP > 0.05
        curl = g.world(e["pb"], e["Rb"], g.CURLS[0])
        if e["attached"] and g.on_support(e["pb"], e["Rb"]) == "peg" and curl[2] > g.PEG_Z - 0.001:  # sliding on
            r = math.hypot(curl[1] - g.PEG_Y, curl[2] - g.PEG_Z)
            assert r < g.INNER - g.SUPPORT_R - 0.012     # the peg runs through the curl with 1.2 cm to spare
    print("hook B clear of the display while carried", round(worst, 4))
    assert worst > 0.012  # the closest is the peg passing the middle stroke inside the curl, 1.2 cm off
    k = next(i for i, e in enumerate(log) if e["released"])
    before = log[k - 1]
    top = g.world(before["pb"], before["Rb"], g.CURLS[0])[2] + g.CHORD - g.ROD_R
    assert top - (g.PEG_Z + g.SUPPORT_R) == pytest.approx(demo.ABOVE, abs=0.001)   # lets go 2 mm above the peg


def test_the_arm_stays_clear_of_the_table_display_and_hooks(played):
    _, log = played
    worst = {"table": 1.0, "display": 1.0, "hook_a": 1.0, "hook_b": 1.0, "fingers_b": 1.0}
    rail, peg = _support_points("rail"), _support_points("peg")
    pa = g.world(A_REST, RA, g.DENSE)
    for i, e in enumerate(log):
        if i % 2:
            continue
        gripping = e["attached"]
        hulls = _hulls(e["q"], e["grip"], gripping)
        worst["table"] = min(worst["table"], min(h.vertices[:, 2].min() for _, h in hulls) - g.TOP)
        verts = np.vstack([h.vertices for _, h in hulls])
        d = min(_box_dist(verts, c, s) for c, s in _fixed_boxes())
        d = min(d, _dist(rail, hulls) - g.SUPPORT_R, _dist(peg, hulls) - g.SUPPORT_R)
        worst["display"] = min(worst["display"], d)
        worst["hook_a"] = min(worst["hook_a"], _dist(pa, hulls) - g.ROD_R)
        pb = g.world(e["pb"], e["Rb"], g.DENSE)
        body = [h for h in hulls if h[0] not in ("tip_left", "tip_right")]
        worst["hook_b"] = min(worst["hook_b"], _dist(pb, body) - g.ROD_R)
        if not gripping and e["grip"] >= demo.PART_OPEN - 0.01:  # open (not closing on, or just opened off, the rod)
            tips = [h for h in hulls if h[0] in ("tip_left", "tip_right")]
            worst["fingers_b"] = min(worst["fingers_b"], _dist(pb, tips) - g.ROD_R)
    print("margins", {k: round(float(v), 4) for k, v in worst.items()})
    assert worst["table"] > 0.05, worst
    assert worst["display"] > 0.015, worst
    assert worst["hook_a"] > 0.015, worst
    assert worst["hook_b"] > 0.015, worst
    assert worst["fingers_b"] > 0.008, worst


def test_the_demo_ends_with_each_hook_alone_and_the_gripper_clear(played):
    _, log = played
    M._reset_state()
    right = gripper_at(*ik.tcp_pose(np.array(READY, float), station.BASES["right"]), -FINGER_OPEN * 2)
    for x in log[::5] + [log[-1]]:
        left = gripper_at(x["tcp"], x["rot"], -FINGER_OPEN * 2 * x["grip"])
        M.update([(A_REST, RA, ZERO, ZERO), (x["pb"], x["Rb"], ZERO, ZERO)], [left, right])
    assert M.success({}) and M._STATE["best"] == 0.5
