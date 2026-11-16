"""Offline checks (no Isaac Sim): the layout and assets, the clipper's mechanism (lever, heel, jaws), the cut model
(closures, nicks, pinches, re-arming), every scoring level and violation, and the demo (reach, joint limits,
clearances, and the scorer's verdict on the planned motion).

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

PLAN = demo.plan()
STEPS = PLAN[:-1]
EYE = np.eye(3)
FAR = [M.gripper_boxes(np.array([0.0, s * 0.30, g.TABLE + 0.40]), np.diag([1.0, -1.0, -1.0]), 0.094)
       for s in (1, -1)]
REST_FINGERS = {n: g.finger_pose_at(n, 0.0) for n in g.NAILS}
FLAT = {n: 0.0 for n in g.NAILS}


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()


def clipper_at(name, u, w=-g.NAIL_T / 2, v=0.0, yaw=0.0):
    """Clipper body pose with its lower edge's centre at (u, v, w) on ``name``'s nail (as the demo holds it), turned
    ``yaw`` rad about the nail's normal."""
    rc = g.rot_z(0.0) @ demo.cut_rot(name)
    n0, nr = g.nail_pose(name, *REST_FINGERS[name])
    rc = nr @ g.rot_z(yaw) @ nr.T @ rc
    origin = n0 + nr @ np.array([u, v, w]) - rc @ np.array([g.E_X, 0.0, g.Z_LE])
    return origin, rc


def step(pose, gap, fingers=None, flex=None, vel=(0.0, 0.0, 0.0), grips=None):
    th, b = g.jaw_state_from_gap(gap)
    return M.update(fingers or REST_FINGERS, flex or FLAT, pose[0], pose[1], th, b, np.asarray(vel, float),
                    FAR if grips is None else grips)


def holding(name):
    """The left gripper pressing ``name`` down on the towel (as the demo does), the right one far away."""
    return [M.gripper_boxes(*demo.press_pose(name), 0.0), FAR[1]]


def snip(name, u, held=True, **kw):
    """Open, close on the nail (held shut for the closure's two steps), open again; the other hand holding the finger
    unless ``held`` is False."""
    pose = clipper_at(name, u, **{k: v for k, v in kw.items() if k in ("w", "v", "yaw")})
    grips = holding(name) if held else None
    step(pose, g.HOLD_GAP, grips=grips)
    step(pose, g.NAIL_T, grips=grips)
    step(pose, g.NAIL_T, grips=grips)
    step(pose, g.HOLD_GAP, grips=grips)


DOWN_POSE = (g.CLIPPER_START, g.R_STAND)


def finish(**kw):
    """The clipper back in its slot, let go, still."""
    step(DOWN_POSE, g.GAP0, **kw)


# -- layout, assets and size ------------------------------------------------------------------------------------------


def test_it_loads_with_a_fixed_hand_and_a_loose_clipper():
    assert TASK.names[:2] == ("hand", "clipper")
    assert TASK.score_over == "end" and M.SUCCESS_HOLD_S == 1.0
    assert M.HAND.disable_gravity and not M.HAND.self_collisions
    assert M.CLIPPER.self_collisions
    assert dict(M.CLIPPER.joint_pos) == {"jaw": 0.0, "lever": 0.0}
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos)
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", (0.0,)))
    for d in M.HAND.drives + M.CLIPPER.drives:
        assert all(type(v) in (str, float) for v in d)


def test_full_size_hand_and_clipper():
    # adult hand: index 8.8 cm and middle 9.7 cm from knuckle to the quick; nails 10.5 and 11.5 mm wide
    assert g.hyp_x("index") == pytest.approx(0.088) and g.hyp_x("middle") == pytest.approx(0.097)
    assert 0.010 <= g.FINGERS["index"]["W"] <= 0.013 and 0.010 <= g.FINGERS["middle"]["W"] <= 0.013
    # standard lever fingernail clipper: 58 mm long, 14 mm wide, 25 g, jaws 4 mm open, 13 mm edges
    assert g.E_X + g.SAG == pytest.approx(0.0586) and g.ARM_W == 0.014 and g.CLIPPER_MASS == pytest.approx(0.025)
    assert g.GAP0 == pytest.approx(0.004) and 2 * g.JAW_HW == pytest.approx(0.013)
    for n in g.NAILS:
        e = g.free_edge0(n)
        assert 2.9e-3 <= e.min() and e.max() <= 4.1e-3     # 3-4 mm of white edge to start


def test_the_assets_match_the_geometry():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    st = Usd.Stage.Open(str(HERE / "assets" / "hand.usda"))
    assert st.GetPrimAtPath("/hand/Physics/root_joint")
    for n in g.NAILS:
        j = UsdPhysics.RevoluteJoint(st.GetPrimAtPath(f"/hand/Physics/{n}"))
        assert j.GetAxisAttr().Get() == "Y"
        assert np.array(j.GetLocalPos0Attr().Get()) == pytest.approx(g.finger_origin(n) - g.HAND0, abs=1e-6)
        for b in range(g.N_BINS):
            pj = UsdPhysics.PrismaticJoint(st.GetPrimAtPath(f"/hand/Physics/{n}_b{b}"))
            assert pj.GetAxisAttr().Get() == "X"
            assert pj.GetLowerLimitAttr().Get() < -g.free_edge0(n)[b]
    st = Usd.Stage.Open(str(HERE / "assets" / "clipper.usda"))
    assert not st.GetPrimAtPath("/clipper/Physics/root_joint")       # free-floating
    jaw = UsdPhysics.RevoluteJoint(st.GetPrimAtPath("/clipper/Physics/jaw"))
    lev = UsdPhysics.RevoluteJoint(st.GetPrimAtPath("/clipper/Physics/lever"))
    assert jaw.GetUpperLimitAttr().Get() == pytest.approx(math.degrees(g.THETA_SHUT), abs=1e-4)
    assert lev.GetLowerLimitAttr().Get() == pytest.approx(math.degrees(g.LEVER_MIN), abs=1e-4)
    assert not st.GetPrimAtPath("/clipper/Geometry/body/pin").HasAPI(UsdPhysics.CollisionAPI)


# -- the clipper's mechanism ------------------------------------------------------------------------------------------


def test_heel_rests_on_the_upper_arm_at_rest():
    assert g.theta_from_beta(g.PHI0) == pytest.approx(0.0, abs=1e-6)
    assert g.theta_from_beta(g.PHI0 + 0.05) == 0.0                # lever lifted: the heel leaves the arm


def test_pressing_the_lever_closes_the_jaws():
    betas = np.linspace(g.PHI0, g.BETA_SHUT, 12)
    thetas = [g.theta_from_beta(b) for b in betas]
    assert np.all(np.diff(thetas) > 0)
    assert g.jaw_gap(g.theta_from_beta(g.BETA_SHUT)) == pytest.approx(0.0, abs=2e-6)
    # the lever's stop leaves the jaws within 0.1 mm of meeting: the jaw's own stop takes the rest
    assert 0.0 < g.jaw_gap(g.theta_from_beta(g.PHI0 + g.LEVER_MIN)) < 1e-4 or \
        g.theta_from_beta(g.PHI0 + g.LEVER_MIN) >= g.THETA_SHUT - 2e-3


def test_lever_ratio_and_forces_are_a_real_clippers():
    # about 9 mm of travel at the lever's end shuts 4 mm of jaw: a ratio near 4 at the heel
    travel = g.pad_gap(g.GAP0) - g.pad_gap(0.0)
    assert 0.008 < travel < 0.012
    assert 0.5 < g.lever_force(0.0) < 1.5 and 4.0 < g.lever_force(g.THETA_SHUT) < 8.0
    # holding the jaws 3 mm open takes a light squeeze; the squeeze command presses past shut
    assert g.grip_for_gap(g.HOLD_GAP) * g.FULL_OPEN < g.pad_gap(g.HOLD_GAP)
    assert g.squeeze_grip() * g.FULL_OPEN < g.pad_gap(0.0) - 0.002


def test_the_geared_cam_follows_the_heel():
    # the mimic joint's fixed ratio stays within 0.1 degree of the heel's exact cam over the whole stroke
    for b in np.linspace(g.BETA_SHUT, g.PHI0, 30):
        assert abs(g.heel_theta(b) - g.theta_from_beta(b)) < math.radians(0.1)


def test_a_cut_needs_the_squeeze():
    pose = clipper_at("index", demo.U_CUT)
    th, b = g.jaw_state_from_gap(g.HOLD_GAP)
    hold = holding("index")
    M.update(REST_FINGERS, FLAT, *pose, th, b, np.zeros(3), hold, 0.0)
    th, b = g.jaw_state_from_gap(g.NAIL_T)
    for _ in range(4):                                              # jaws on the nail, but pressed lightly
        M.update(REST_FINGERS, FLAT, *pose, th, b, np.zeros(3), hold, g.F_CUT - 1.0)
    assert not M._STATE["cut"]["index"].any()
    for _ in range(2):
        M.update(REST_FINGERS, FLAT, *pose, th, b, np.zeros(3), hold, g.F_CUT + 1.0)
    assert M._STATE["cut"]["index"].all()


def test_the_upper_edge_meets_the_lower_one():
    lo, up = g.lower_edge_c(), g.upper_edge_c(g.THETA_SHUT)
    assert np.abs(up - lo).max() < 2e-4     # the arc about the rivet leaves it 0.15 mm behind


# -- the cut model ---------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", g.NAILS)
def test_one_square_snip_leaves_the_band_straight_across(name):
    snip(name, demo.U_CUT)
    L = M._STATE["L"][name]
    assert M._STATE["cut"][name].all() and not M.violation()
    assert np.all((L >= g.BAND[0]) & (L <= g.BAND[1]))
    assert L.max() - L.min() <= g.STRAIGHT
    assert M.nail_score(name) == pytest.approx(0.5)
    assert M.bin_targets(name) == pytest.approx(list(L - g.free_edge0(name)))


def test_a_cut_needs_the_jaws_shut():
    pose = clipper_at("index", demo.U_CUT)
    for gap in (g.HOLD_GAP, 2.5e-3, 1.5e-3, 0.9e-3, g.HOLD_GAP):     # never at or under nail + 0.3 mm
        step(pose, gap)
    assert not M._STATE["cut"]["index"].any()
    step(pose, g.HOLD_GAP)
    step(pose, g.NAIL_T)                                            # one step shut is not enough
    step(pose, g.HOLD_GAP)
    assert not M._STATE["cut"]["index"].any()


def test_hovering_or_closing_beside_the_nail_cuts_nothing():
    for w in (3.0e-3, -2.5e-3):                                     # the nail outside the shut jaws
        M.reset_state()
        snip("index", demo.U_CUT, w=w)
        assert not M._STATE["cut"]["index"].any() and not M.violation()
    M.reset_state()
    snip("index", demo.U_CUT, v=0.02)                               # 2 cm to the side
    assert not M._STATE["cut"]["index"].any()


def test_closing_without_reopening_cuts_once():
    pose = clipper_at("index", 2.6e-3)
    step(pose, g.HOLD_GAP)
    for _ in range(4):
        step(pose, g.NAIL_T)
    deeper = clipper_at("index", 1.5e-3)
    for gap in (1.2e-3, g.NAIL_T, g.NAIL_T):                        # not reopened 1.5 mm: no new closure
        step(deeper, gap)
    assert M._STATE["L"]["index"].max() > 2.5e-3
    assert len(M._STATE["closures"]) == 1


def test_overlapping_snips_only_cut_where_the_jaws_cross():
    snip("index", 2.4e-3, v=-0.008)          # only the bins under the jaws' left part
    cut = M._STATE["cut"]["index"]
    assert cut.any() and not cut.all()
    vb = g.bin_v("index")
    assert np.all(vb[cut] < 0.0) or np.all(vb[cut] > 0.0)


def test_deep_cut_nicks_the_skin_and_zeroes_the_trial():
    snip("middle", 0.3e-3)
    assert M.violation() and "nicked" in M._STATE["why"][0]
    finish()
    assert M.score(None) == 0.0 and not M.success(None)


def test_closing_on_the_attached_nail_is_a_nick():
    snip("index", -2.0e-3)
    assert M.violation()


def test_jaws_closing_on_the_fingertip_pinch_the_skin():
    # the clipper 4 mm lower and 1 mm in: the jaws close on the fingertip under the nail, not on the nail
    pose = clipper_at("index", -1.0e-3, w=-4.0e-3)
    step(pose, g.HOLD_GAP)
    step(pose, 1.5e-3)
    assert M.violation() and "skin" in M._STATE["why"][0]


def test_cutting_in_two_snips_is_allowed():
    snip("index", 2.8e-3)
    snip("index", demo.U_CUT)
    L = M._STATE["L"]["index"]
    assert M._STATE["cut"]["index"].all() and np.all((L >= g.BAND[0]) & (L <= g.BAND[1]))


def test_a_skewed_cut_is_not_straight():
    snip("middle", 1.75e-3, yaw=math.radians(6.0))
    L = M._STATE["L"]["middle"]
    assert M._STATE["cut"]["middle"].all()
    assert L.max() - L.min() > g.STRAIGHT
    assert M.nail_score("middle") < 0.5


# -- scoring levels ---------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_zero():
    finish()
    assert M.score(None) == 0.0 and not M.success(None)


def test_both_nails_trimmed_and_let_go_is_success():
    snip("index", demo.U_CUT)
    snip("middle", demo.U_CUT)
    finish()
    assert M._STATE["total"] == pytest.approx(1.0)
    assert M.success(None) and M.score(None) == pytest.approx(0.99)


def test_still_holding_the_clipper_caps_at_0_9():
    snip("index", demo.U_CUT)
    snip("middle", demo.U_CUT)
    held = clipper_at("middle", 0.02)
    step(held, g.HOLD_GAP)                                           # not down (in the air, by the hand)
    assert M.score(None) == pytest.approx(0.9) and not M.success(None)
    near = [FAR[0], M.gripper_boxes(*demo.pick_pose(g.pad_gap(g.GAP0)), g.pad_gap(g.GAP0))]
    step(DOWN_POSE, g.GAP0, grips=near)                              # down but a gripper still on it
    assert not M.success(None) and M.score(None) == pytest.approx(0.9)


def test_one_nail_done_scores_half():
    snip("index", demo.U_CUT)
    finish()
    assert M.score(None) == pytest.approx(0.5)


def test_too_long_scores_only_straightness_and_steadiness():
    snip("middle", 2.9e-3)                                         # 2.9 mm left in the middle: too long
    finish()
    L = M._STATE["L"]["middle"]
    assert np.all(L > g.BAND[1] - 0.6e-3)
    assert M.nail_score("middle") == pytest.approx(0.3 * np.mean((L >= 1e-3) & (L <= 2e-3)) + 0.2)


def test_too_short_but_no_nick_loses_band_credit():
    snip("index", 0.95e-3)                                         # 0.95 mm: under the band, not into the quick
    finish()
    assert not M.violation()
    assert M.nail_score("index") == pytest.approx(0.1 + 0.1 * (np.ptp(M._STATE["L"]["index"]) <= g.STRAIGHT))
    assert M.nail_score("index") < 0.25


def test_an_unsteady_finger_loses_its_steadiness_credit():
    flex = {"index": math.radians(12.0), "middle": 0.0}
    fingers = {"index": g.finger_pose_at("index", math.radians(12.0)), "middle": REST_FINGERS["middle"]}
    step(clipper_at("middle", 0.03), g.HOLD_GAP, fingers=fingers, flex=flex)   # pushed 12 degrees at some point
    snip("index", demo.U_CUT)
    snip("middle", demo.U_CUT)
    finish()
    assert M._STATE["total"] == pytest.approx(0.9)
    assert not M.success(None)


def test_steadiness_needs_the_other_hand_holding_the_finger():
    snip("index", demo.U_CUT, held=False)
    snip("middle", demo.U_CUT)
    finish()
    assert M.nail_score("index") == pytest.approx(0.4) and M.nail_score("middle") == pytest.approx(0.5)
    assert M._STATE["total"] == pytest.approx(0.9) and not M.success(None)


def test_steadiness_needs_a_cut():
    step(clipper_at("index", 0.03), g.HOLD_GAP)
    finish()
    assert M.nail_score("index") == 0.0 and M.score(None) == 0.0


def test_clipper_off_the_table_zeroes():
    snip("index", demo.U_CUT)
    step((np.array([0.3, -0.2, g.TABLE - 0.2]), EYE), g.GAP0)
    assert M.violation() and M.score(None) == 0.0


# -- the demo --------------------------------------------------------------------------------------------------------


def test_demo_is_reachable():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert worst[side][0] < 0.5 and worst[side][1] < 0.1
        q = np.array([s[side][0] for s in STEPS])
        assert np.all(q >= ik.LOWER - 1e-6) and np.all(q <= ik.UPPER + 1e-6)


def test_demo_fits_the_run_limit():
    assert len(STEPS) < 3600 - 3 * demo.CONTROL_HZ
    assert [s[0] for s in PLAN[-1]["snips"]] == ["index", "middle"]


def test_left_hand_holds_each_finger_while_its_nail_is_cut():
    for s in STEPS:
        if s["edge"] is not None:
            assert s["flags"].get(f"held_{s['edge'][0]}") and not s["flags"].get(f"done_{s['edge'][0]}")


def _hand_capsules(skip=None):
    caps = list(g.static_capsules_world())
    for n in g.NAILS:
        if n != skip:
            caps += g.finger_capsules_world(n, *REST_FINGERS[n])
    return caps


def test_the_towel_holds_the_fingers_middle_segments():
    tc, tr, ts = g.towel_pose()
    for n in g.NAILS:
        local = tr.T @ (g.pinch_point(n) - tc)
        assert abs(local[0]) < ts[0] / 2 - 0.004 and abs(local[1]) < ts[1] / 2 - 0.008
        assert 0.0 < math.degrees(g.held_flex(n)) < 1.0                # pressed onto it: under a degree
    for a, b, rad in g.static_capsules_world():                        # nothing of the hand rests in it
        for p in (a, b):
            local = tr.T @ (p - tc)
            if abs(local[0]) < ts[0] / 2 and abs(local[1]) < ts[1] / 2:
                assert p[2] - rad > g.TOWEL_TOP


def test_demo_clearances():
    """Grippers clear of the hand, the hand rest and each other; the clipper clear of every finger but the one being
    trimmed (whose tip it is at), and of that finger's tissue by 0.5 mm."""
    pc, pr, ph = g.palm_box()
    rc, rr = g.rest_pose()
    rest = (rc, rr, np.array(g.REST_SIZE) / 2)
    tc, tr, ts = g.towel_pose()
    towel = (tc, tr, ts / 2)
    worst = {"right_hand": 1.0, "left_hand": 1.0, "arms": 1.0, "clip_other": 1.0, "clip_tissue": 1.0}
    for s in STEPS[::2]:
        grips = {k: M.gripper_boxes(s[k][2], s[k][3], max(s[k][1] * g.FULL_OPEN, 0.0)) for k in ("left", "right")}
        # the finger the left hand is over (closing on it, holding it or letting go of it)
        pinched = [n for n in g.NAILS if np.linalg.norm((s["left"][2] - g.pinch_point(n))[:2]) < 0.005]
        for side in ("left", "right"):
            pts = np.vstack([g.box_points(*b, 5) for b in grips[side]])
            caps = _hand_capsules(skip=pinched[0] if side == "left" and pinched else None)
            d = min(float(g.point_capsule_dist(pts, *c).min()) for c in caps)
            d = min(d, min(g.boxes_gap(b, (pc, pr, ph), 5) for b in grips[side]),
                    min(g.boxes_gap(b, rest, 5) for b in grips[side]), min(g.boxes_gap(b, towel, 5) for b in grips[side]))
            worst[f"{side}_hand"] = min(worst[f"{side}_hand"], d)
        worst["arms"] = min(worst["arms"], min(g.boxes_gap(a, b, 5) for a in grips["left"] for b in grips["right"]))
        if s["holding"]:
            cpos, crot, gap = s["clipper"]
            th, b = g.jaw_state_from_gap(gap)
            pts = np.vstack([g.box_points(*bx, 5) for bx in g.clipper_boxes(cpos, crot, th, b)])
            near = s["edge"][0] if s["edge"] else None
            d = min(float(g.point_capsule_dist(pts, *c).min()) for c in _hand_capsules(skip=near))
            worst["clip_other"] = min(worst["clip_other"], d)
            if near:
                lo, up = g.edge_segments_world(cpos, crot, th)
                f = REST_FINGERS[near]
                depth = max(float(g.tissue_dist(near, (np.vstack([lo, up]) - f[0]) @ f[1]).max()), -1.0)
                worst["clip_tissue"] = min(worst["clip_tissue"], -depth)
    assert worst["right_hand"] > 0.008, worst
    assert worst["left_hand"] > 0.003, worst
    assert worst["arms"] > 0.02, worst
    assert worst["clip_other"] > 0.005, worst
    assert worst["clip_tissue"] > 0.0005, worst


def test_the_planned_demo_scores_full_marks():
    """The plan replayed through the monitor, with each finger where the left hand holds it, and the jaws stopping on
    the nail when they shut on it (they clamp it mid-mouth: the plan aims the nail at the middle of the mouth), 0.45 mm
    nearer the quick than aimed (as measured in Isaac Sim)."""
    M.reset_state()
    for i, s in enumerate(STEPS):
        cpos, crot, gap = s["clipper"]
        fingers, flex = dict(REST_FINGERS), dict(FLAT)
        for n in g.NAILS:
            if s["flags"].get(f"held_{n}") and not s["flags"].get(f"done_{n}"):
                flex[n] = g.held_flex(n)
                fingers[n] = g.finger_pose_at(n, flex[n])
        if s["edge"] is not None and gap <= g.CUT_GAP:
            n = s["edge"][0]
            nr = g.nail_pose(n, *fingers[n])[1]
            lower_w = -g.NAIL_T / 2 * math.cos(demo.PITCH)
            cpos = cpos + nr[:, 2] * (lower_w - s["edge"][2]) + nr[:, 0] * (demo.U_CUT - s["edge"][1])
            gap = g.NAIL_T
        grips = [M.gripper_boxes(s[k][2], s[k][3], s[k][1] * g.FULL_OPEN) for k in ("left", "right")]
        step((cpos, crot), gap, grips=grips, fingers=fingers, flex=flex)
    assert not M.violation()
    assert M._STATE["total"] == pytest.approx(1.0) and M.success(None)
    assert {c["nail"] for c in M._STATE["closures"]} == set(g.NAILS)
