"""Offline checks (no Isaac Sim): the layout and the mechanism (slot, star wheel, the lever stroke's geometry, the
adjuster's clicks, pawl and drag), every scoring level and violation, and the demo (reach, joint limits, clearances,
end state).

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


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()


def star_angle(clicks: float) -> float:
    return g.TIGHTEN * clicks * g.PITCH


def shoe_joints(clicks: float, extra_mm: float = 0.0):
    """The shoes' joint positions a rigid, round drum allows after ``clicks`` (each stopped at the drum once it gets
    there; ``extra_mm`` more travel, e.g. a sim's penetration)."""
    u = min(g.target_mm(math.floor(clicks + 0.5)), g.C0_MM) / 1000 + extra_mm / 1000
    return [sign * u for _, _, sign, _ in g.SHOES]


def step(clicks=0.0, spoon=(g.SPOON_START, g.YAW90), vel=(0.0, 0.0, 0.0), grips=None, drum=0.0, shoes=None):
    return M.update(g.B0, EYE, drum, star_angle(clicks), shoe_joints(clicks) if shoes is None else shoes, spoon[0],
                    spoon[1], np.asarray(vel, float), FAR if grips is None else grips)


def click_to(n: int, **kw):
    """Turn the wheel one notch at a time to ``n`` clicks (as the spoon would), then a still step."""
    for k in range(1, n + 1):
        step(k - 0.45, **kw)
        step(k - 0.55 + 0.1, **kw)   # past half a tooth: the lever drops into the next notch
        step(k, **kw)
    step(n, **kw)


# -- layout, assets and size ----------------------------------------------------------------------------------------


def test_it_loads_with_a_fixed_corner_and_a_loose_spoon():
    assert TASK.names == ("corner", "spoon")
    assert TASK.score_over == "end" and M.SUCCESS_HOLD_S == 1.0
    assert dict(M.CORNER.joint_pos) == {"drum": 0.0, "star": 0.0, "shoe_primary": 0.0, "shoe_secondary": 0.0}
    assert M.CORNER.self_collisions          # the linings must be able to touch the drum
    assert M.SPOON.mass == pytest.approx(g.SPOON_MASS)
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos) and all(type(v) is float for v in spec.rot_xyzw)
    for d in M.CORNER.drives:
        assert all(type(v) in (str, float) for v in d)


def test_the_assets_match_the_geometry():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    st = Usd.Stage.Open(str(HERE / "assets" / "corner.usda"))
    assert st.GetPrimAtPath("/corner/Physics/root_joint")             # fixed to the world
    drum = UsdPhysics.RevoluteJoint(st.GetPrimAtPath("/corner/Physics/drum"))
    star = UsdPhysics.RevoluteJoint(st.GetPrimAtPath("/corner/Physics/star"))
    assert drum.GetAxisAttr().Get() == "Y" and star.GetAxisAttr().Get() == "X"
    assert np.array(star.GetLocalPos0Attr().Get()) == pytest.approx(g.SW, abs=1e-6)
    assert star.GetLowerLimitAttr().Get() == pytest.approx(-math.degrees((g.N_MAX + 0.5) * g.PITCH), abs=1e-3)
    for side, _, sign, _ in g.SHOES:
        j = UsdPhysics.PrismaticJoint(st.GetPrimAtPath(f"/corner/Physics/{side}"))
        assert j.GetAxisAttr().Get() == "X"
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        assert sign * lo <= 0 or sign * hi <= 0                    # can retract a little
        assert max(sign * lo, sign * hi) >= g.target_mm(g.N_MAX) / 1000   # and reach past the drum (it stops them)
        pads = [p for p in st.GetPrimAtPath(f"/corner/Geometry/base/{side}").GetChildren()
                if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert sorted(p.GetName() for p in pads) == ["pad_0", "pad_1", "pad_2"]   # only the pads collide
    mat = st.GetPrimAtPath("/corner/Looks/lining_physics")
    assert UsdPhysics.MaterialAPI(mat).GetDynamicFrictionAttr().Get() == pytest.approx(g.MU_LINING)
    assert mat.GetAttribute("physxMaterial:frictionCombineMode").Get() == "min"
    band = [p for p in st.GetPrimAtPath("/corner/Geometry/base/drum").GetChildren() if p.GetName().startswith("band")]
    assert len(band) == g.N_BAND
    teeth = [p for p in st.GetPrimAtPath("/corner/Geometry/base/star").GetChildren() if p.GetName().startswith("tooth")]
    assert len(teeth) == g.N_TEETH
    r = UsdGeom.Imageable(st.GetPrimAtPath("/corner/Geometry/base/drum")).ComputeWorldBound(0, "default")
    rng = r.ComputeAlignedRange()
    assert rng.GetMax()[0] == pytest.approx(g.DRUM_R_OUT, abs=0.002)
    sp = Usd.Stage.Open(str(HERE / "assets" / "spoon.usda"))
    root = sp.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(g.SPOON_MASS)
    r = UsdGeom.Imageable(root).ComputeWorldBound(0, "default").ComputeAlignedRange()
    assert np.array(r.GetMax()) - np.array(r.GetMin()) == pytest.approx([g.SPOON_LEN, g.HANDLE[1], g.HANDLE[2]],
                                                                         abs=1e-4)


def test_full_size_parts():
    assert 2 * g.DRUM_R_IN == pytest.approx(0.2286)                 # 9 in drum
    assert 0.20 <= g.SPOON_LEN <= 0.25 and g.BLADE[2] >= 0.0015     # 22.5 cm spoon, 1.6 mm blade
    assert 0.025 <= 2 * g.TOOTH_R[1] <= 0.035 and g.N_TEETH == 10   # 30 mm star wheel
    assert 2 * g.STUD_PCD_R == pytest.approx(0.1143)                # 5 x 114.3 mm studs
    assert g.SLOT_Z[1] - g.SLOT_Z[0] == pytest.approx(0.016) and 2 * g.SLOT_HX == pytest.approx(0.022)


def test_the_corner_stands_clear_of_the_table_and_the_spoon_lies_flat():
    lowest = min(float(g.box_points(*b, 2)[:, 2].min()) for b in g.corner_boxes_world())
    assert lowest >= g.TABLE - 1e-6
    assert g.B0[2] - g.PLATE_R - g.TABLE >= 0.04
    sb = g.spoon_boxes(g.SPOON_START, g.YAW90)
    assert min(float(g.box_points(*b, 2)[:, 2].min()) for b in sb) == pytest.approx(g.TABLE + 0.0005, abs=1e-6)
    assert min(g.boxes_gap(a, b, 5) for a in sb for b in g.corner_boxes_world()) > 0.02


def test_the_blade_fits_the_slot_with_play():
    assert 2 * g.SLOT_HX - g.BLADE[1] >= 0.009                        # 5 mm each side across
    assert g.SLOT_Z[1] - g.SLOT_Z[0] - g.BLADE[2] >= 0.014
    # the wheel's teeth are narrower than the blade, so a blade off-centre by up to 3 mm still bears on them
    assert (g.BLADE[1] - g.SW_W) / 2 >= 0.003


# -- the stroke (the blade on the teeth, in the wheel's plane) --------------------------------------------------------


def _hub_gap(alpha, d):
    c, h, ang = g.blade_rect_2d(alpha, d)
    local = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]]) @ -c
    return math.hypot(*np.maximum(np.abs(local) - h, 0.0)) - 0.010


def test_the_blade_goes_in_under_the_lower_tooth_with_play():
    for d in np.linspace(-0.03, g.D_ENGAGE, 40):
        assert g.blade_tooth_gap(g.ALPHA_LOW, d, 0.0) >= 0.0008
        assert g.blade_slot_gap(g.ALPHA_LOW, d) >= 0.0
    assert _hub_gap(g.ALPHA_LOW, g.D_ENGAGE) >= 0.001
    # its end sits below the wheel's axis, under the lower tooth of the gap facing the slot (at 198 degrees)
    end = g.FULCRUM + g.D_ENGAGE * g.blade_dir(g.ALPHA_LOW) - g.SW
    assert end[2] < -0.005 and 0.011 < math.hypot(end[1], end[2]) < g.TOOTH_R[1]


def _push(short=0.0):
    """How far (rad) the lever stroke turns the wheel, the blade's end ``short`` of D_ENGAGE."""
    phi = 0.0
    for a in np.linspace(g.ALPHA_LOW, g.ALPHA_TOP, 161):
        while g.blade_tooth_gap(a, g.D_ENGAGE - short, phi) < 0.0 and phi < 2 * g.PITCH:
            phi += math.radians(0.05)
    return phi


def test_a_stroke_turns_the_wheel_one_click_whatever_the_depth():
    """Handle down: the blade's end lifts the lower tooth past half a tooth (a click) and well short of a second;
    nearly the same with the blade's end 4 mm short (the arm giving way) or 1 mm too deep."""
    for short in (-0.001, 0.0, 0.002, 0.004):
        phi = _push(short)
        assert g.PITCH / 2 + math.radians(10) <= phi <= 3 * g.PITCH / 2 - math.radians(15), short
    # the blade pivots on the slot's bottom edge, never into the slot's top; mid-stroke its end reaches the root of
    # the tooth and slides up the hub, bottomed there: the command is at most 2 mm past the hub, which the arm takes up
    # (in Isaac Sim the blade's end rides 10-10.4 mm from the axis)
    for a in np.linspace(g.ALPHA_LOW, g.ALPHA_TOP, 300):
        assert g.blade_slot_gap(a, g.D_ENGAGE) >= -0.0001
        assert _hub_gap(a, g.D_ENGAGE) >= -0.002


def test_the_return_stroke_clears_the_teeth():
    """One notch on: level the blade, slide it back, raise the handle, and slide it in under the next tooth."""
    for a in np.linspace(g.ALPHA_TOP, 0.0, 10):
        assert g.blade_tooth_gap(a, g.D_ENGAGE, g.PITCH) >= 0.0002
    for d in np.linspace(g.D_ENGAGE, g.D_BACK, 20):
        assert g.blade_tooth_gap(0.0, d, g.PITCH) >= 0.0010
    for a in np.linspace(0.0, g.ALPHA_LOW, 20):
        assert g.blade_tooth_gap(a, g.D_BACK, g.PITCH) >= 0.0030
        assert g.blade_slot_gap(a, g.D_BACK) >= -0.0001
    # one notch on, the teeth are where they were: the next stroke starts the same
    assert g.blade_tooth_gap(g.ALPHA_LOW, g.D_ENGAGE, g.PITCH) == pytest.approx(
        g.blade_tooth_gap(g.ALPHA_LOW, g.D_ENGAGE, 0.0))


def test_lever_ratio_keeps_the_hand_force_small():
    """Click torque 0.10 N m at the blade's end (about 12 mm from the axis) is about 8.5 N; on the slot's edge as the
    fulcrum, the hand at the handle's centre needs a fifth of that."""
    tip_force = g.CLICK_TORQUE / 0.012
    tip_arm = g.D_ENGAGE
    hand_arm = g.TIP_X - g.D_ENGAGE
    assert tip_force * tip_arm / hand_arm < 2.5


# -- the adjuster's model ----------------------------------------------------------------------------------------------


def test_the_shoes_reach_the_drum_through_their_pads():
    """Each lining's three contact spheres sit C0_MM inside the 228.6 mm bore, on the lining's face; the band's
    segments are round to 0.01 mm, so contact starts within a fifth of a click of the nominal count."""
    for side, mid, sign, half in g.SHOES:
        for _, c, rad in g.shoe_pads(side):
            c = np.asarray(c)
            assert math.hypot(c[0], c[2]) + rad == pytest.approx(g.DRUM_R_IN - g.C0_MM / 1000, abs=1e-9)
            assert g.SHOE_Y[0] < c[1] < g.SHOE_Y[1]
            ang = math.atan2(c[2], c[0])
            assert abs((ang - mid + math.pi) % (2 * math.pi) - math.pi) < half      # within the lining's arc
            assert sign * c[0] > 0
    band = g.drum_boxes()
    da = 2 * math.pi / g.N_BAND
    assert g.DRUM_R_IN / math.cos(da / 2) - g.DRUM_R_IN < 0.00001
    assert all(math.hypot(c[0], c[2]) - s[0] / 2 == pytest.approx(g.DRUM_R_IN) for _, c, s, _ in band)
    assert g.N_CONTACT == pytest.approx(g.C0_MM / g.CLICK_MM)


def test_calibration_numbers():
    """The light-drag window comes from the cited data: 69 N at a wheel stud (on the 114.3 mm circle) is the most
    drag a manual accepts; the lining's friction is the middle of edge code FF."""
    assert g.T_HI == pytest.approx(69.0 * g.STUD_ARM, abs=0.05)
    assert 0.35 <= g.MU_LINING <= 0.45
    assert g.T_LO >= 6 * g.T_BEAR
    assert 2 * g.DRUM_R_IN == pytest.approx(9.0 * 0.0254)


def test_click_count_to_torque_table():
    """Clicks -> clearance -> torque with a rigid, round drum (the README's table): free up to 19 clicks, touching at
    20, light drag 22-28, dragging at 29-30, driven on from 31."""
    assert g.ideal_torque(19) == pytest.approx(g.T_BEAR)
    assert g.T_BEAR < g.ideal_torque(20) < g.ideal_torque(21) < g.T_LO
    for n in range(22, 29):
        assert g.T_LO + 0.1 <= g.ideal_torque(n) <= g.T_HI - 0.1, n
    for n in (29, 30):
        assert g.T_HI + 0.1 <= g.ideal_torque(n) <= g.T_LOCK - 0.1, n
    assert g.ideal_torque(31) > g.T_LOCK
    # each click past contact adds the same: 2 x K x 0.05 mm x mu x r
    step_nm = 2 * g.K_SHOE * g.CLICK_MM / 1000 * g.MU_LINING * g.DRUM_R_IN
    assert g.ideal_torque(25) - g.ideal_torque(24) == pytest.approx(step_nm)


def test_torque_comes_from_the_measured_shoe_positions():
    """The monitor reads each shoe's measured travel; a shoe the drum stops short of its command presses on it."""
    click_to(25)
    assert M._STATE["torque"] == pytest.approx(g.ideal_torque(25))
    assert M._STATE["clear_mm"] == pytest.approx((0.0, 0.0), abs=1e-9)
    # the same 25 clicks with the shoes free to reach their command (no drum): no drag at all
    step(25, shoes=[s * g.target_mm(25) / 1000 for _, _, s, _ in g.SHOES])
    assert M._STATE["torque"] == pytest.approx(g.T_BEAR) and M.score(None) == 0.5
    # one shoe on the drum, the other short of it: only one presses
    step(25, shoes=[g.C0_MM / 1000, -g.target_mm(25) / 1000])
    f = g.shoe_force(g.target_mm(25) / 1000, g.C0_MM / 1000)
    assert M._STATE["forces"] == pytest.approx((f, 0.0), abs=0.01)


def test_the_wheel_clicks_past_half_a_tooth_and_the_pawl_stops_it_turning_back():
    n, target, stop = M.adjuster_step(star_angle(0.45), 0)
    assert n == 0 and target == pytest.approx(0.0) and stop is None
    n, target, stop = M.adjuster_step(star_angle(0.55), 0)
    assert n == 1 and target == pytest.approx(star_angle(1)) and stop is None
    n, target, stop = M.adjuster_step(star_angle(0.7), 1)      # pushed back a little: the spring holds it
    assert n == 1 and stop is None
    n, target, stop = M.adjuster_step(star_angle(0.3), 1)      # back past the half-way point: the pawl stops it
    assert n == 1 and stop == pytest.approx(star_angle(1 - g.PAWL))
    n, _, _ = M.adjuster_step(star_angle(2.6), 1)              # two teeth in one push count two clicks
    assert n == 3


# -- scoring -------------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_0():
    step()
    assert M.score(None) == 0.0 and not M.success(None)


def test_partial_credit_rises_with_the_shoes_travel():
    scores = []
    for n in (5, 10, 15, 19, 20, 21):
        M.reset_state()
        click_to(n)
        scores.append(M.score(None))
    assert scores == sorted(scores) and len(set(scores)) == len(scores)
    assert scores[0] == pytest.approx(0.5 * 0.25 / M.TRAVEL_LO_MM, abs=1e-3)
    assert 0.45 < scores[-1] < 0.5


@pytest.mark.parametrize("n", [22, 25, 28])
def test_light_drag_with_the_spoon_down_and_let_go_is_success(n):
    click_to(n)
    assert M.success(None) and M.score(None) == 0.75


def test_light_drag_with_the_spoon_still_in_the_slot_scores_075():
    o, r = g.spoon_pose_b(0.0, g.D_ENGAGE)
    click_to(25, spoon=(g.B0 + o, r))
    assert not M.success(None) and M.score(None) == 0.75


def test_light_drag_while_holding_the_spoon_or_the_drum_scores_075():
    click_to(25)
    tcp, rot = demo.spoon_tcp(g.SPOON_START, g.YAW90)
    step(25, grips=[FAR[0], M.gripper_boxes(tcp, rot, 0.024)])
    assert not M.success(None) and M.score(None) == 0.75
    M.reset_state()
    click_to(25)
    p, r = demo.stud_pose(0.0)
    step(25, grips=[M.gripper_boxes(p, r, 0.012), FAR[1]])
    assert not M.success(None) and M.score(None) == 0.75


def test_moving_spoon_or_turning_drum_is_not_success():
    click_to(25)
    step(25, vel=(0.05, 0.0, 0.0))
    assert not M.success(None)
    step(25, drum=math.radians(1.0))
    assert not M.success(None)
    step(25, drum=math.radians(1.0))
    assert M.success(None)


@pytest.mark.parametrize("n", [29, 30])
def test_past_the_window_is_a_dragging_brake(n):
    click_to(n)
    for _ in range(30):
        step(n)
    assert M.score(None) == 0.25 and not M.success(None) and not M.violation()


def test_driving_the_shoes_onto_the_drum_scores_0_for_good():
    click_to(31)
    for _ in range(M.LOCK_STEPS):
        step(31)
    assert M.score(None) == 0.0 and M.violation()
    step(25)              # even if the wheel were somehow back in the window
    assert M.score(None) == 0.0 and not M.success(None)


def test_driven_on_at_the_end_scores_0_before_the_latch():
    """Torque over 4.8 N m when the trial ends scores 0 even before the 0.5 s latch."""
    click_to(31)
    assert M._STATE["torque"] > g.T_LOCK and not M.violation()
    assert M.score(None) == 0.0 and not M.success(None)


def test_a_momentary_spike_does_not_latch():
    click_to(25)
    hard = [s * (g.C0_MM - 1.0) / 1000 for _, _, s, _ in g.SHOES]     # shoes pushed 1 mm back for a moment
    for _ in range(M.LOCK_STEPS - 1):
        step(25, shoes=hard)
    assert M._STATE["torque"] > g.T_LOCK
    step(25)
    assert not M.violation() and M.success(None)


def test_the_drag_window_is_read_from_torque_not_clicks():
    """Same 25 clicks, but a drum 0.1 mm larger in radius (the shoes stop 0.1 mm further out): a lighter 1.6 N m, still
    light drag; 0.2 mm smaller: a dragging brake (4.2 N m). The score follows the torque, not the count."""
    click_to(25, shoes=shoe_joints(25, extra_mm=0.1))
    t = M._STATE["torque"]
    assert g.T_LO < t < g.ideal_torque(25) and M.success(None)
    M.reset_state()
    click_to(25, shoes=shoe_joints(25, extra_mm=-0.2))
    for _ in range(5):
        step(25, shoes=shoe_joints(25, extra_mm=-0.2))
    assert M._STATE["torque"] > g.T_HI and M.score(None) == 0.25


def test_spoon_off_the_table_scores_0():
    click_to(25)
    step(25, spoon=(np.array([0.2, -0.6, g.TABLE - 0.3]), g.YAW90))
    step(25)
    assert M.score(None) == 0.0 and not M.success(None)


def test_spoon_set_down_anywhere_flat_counts_and_a_dropped_one_on_the_table_is_fine():
    click_to(25, spoon=(np.array([0.15, -0.20, g.SPOON_START[2]]), g.rot_z(0.4) @ g.YAW90))
    assert M.success(None)
    M.reset_state()
    on_side = g.YAW90 @ g.rot_x(math.pi / 2)        # on its edge: the handle stands 24 mm tall: still down
    click_to(25, spoon=(np.array([0.15, -0.20, g.TABLE + g.HANDLE[1] / 2]), on_side))
    assert M.success(None)


def test_spoon_propped_up_is_not_down():
    tilted = g.rot_x(math.radians(30)) @ g.YAW90
    click_to(25, spoon=(np.array([0.15, -0.20, g.TABLE + 0.05]), tilted))
    assert not M.success(None) and M.score(None) == 0.75


def test_measured_in_the_corner_frame():
    """The scorer reads clicks from the star joint and the drum from the corner's measured pose: a corner
    reported 1 cm off moves the drum region with it."""
    p, r = demo.stud_pose(0.0, dy=demo.STUD_DY + 0.06)     # the fingertips 32 mm beyond the studs' ends
    grip = M.gripper_boxes(p, r, 0.012)
    near = M.gap_to_drum(grip, g.B0, EYE)
    moved = M.gap_to_drum(grip, g.B0 + [0.0, -0.01, 0.0], EYE)
    assert near == pytest.approx(0.032, abs=0.002) and moved == pytest.approx(near + 0.01, abs=0.001)


# -- the demo --------------------------------------------------------------------------------------------------------


def test_demo_reach_and_limits():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert worst[side][0] < 0.5 and worst[side][1] < 0.1
        for st in STEPS:
            q = st[side][0]
            assert np.all(q >= ik.LOWER - 1e-9) and np.all(q <= ik.UPPER + 1e-9)


def test_demo_fits_the_run_limit_and_makes_clicks_in_the_window():
    run_limit = 4800
    assert len(STEPS) + 3 * demo.CONTROL_HZ + 60 < run_limit
    assert STEPS[-1]["clicks"] == g.DEMO_CLICKS == 25
    assert g.T_LO + 1.0 < g.ideal_torque(g.DEMO_CLICKS) < g.T_HI - 1.0


def test_demo_does_not_read_simulator_objects():
    src = (HERE / "demo.py").read_text()
    for bad in ("object_states", "root_link_pose", "body_pos_w", "_STATE[\"clicks\"]", "joints.get"):
        assert bad not in src.split("def run(")[0]
    # in run(), the task's state is only printed and reported, never fed back into the script
    run_src = src.split("def run(")[1]
    assert "script.clicks" in run_src and "S['clicks']" in run_src
    assert "script." not in run_src.split("S = task._STATE")[0].replace("script.steps()", "").replace(
        "script.left", "").replace("script.right", "").replace("script.turns", "").replace("script.clicks", "")


def _near(boxes, centre, reach=0.25):
    return [b for b in boxes if np.linalg.norm(b[0] - centre) < reach]


def test_grippers_clear_the_corner_and_the_table():
    corner = g.corner_boxes_world(band=False)       # the band: as a cylinder below
    band = g.drum_world_cylinders()[0]
    for k, st in enumerate(STEPS[::3]):
        for side in ("left", "right"):
            q, grip, tcp, rot = st[side]
            boxes = M.gripper_boxes(tcp, rot, 0.094 * grip if grip > 0 else 0.0)
            pts = np.vstack([g.box_points(*b, 5) for b in boxes])
            assert pts[:, 2].min() >= g.TABLE + 0.002, (side, k)
            near = _near(corner, tcp)
            gap = min((g.boxes_gap(a, b, 5) for a in boxes for b in near), default=1.0)
            assert gap >= 0.015, (side, k, gap)
            # the drum's band and face (the left hand holds only a stud standing 45 mm proud of the face)
            assert float(g.point_cyl_dist(pts, *band).min()) >= 0.015, (side, k)


def test_spoon_clear_of_the_corner_except_its_blade_in_the_slot():
    corner = g.corner_boxes_world()
    for k, st in enumerate(STEPS[::2]):
        pos, rot = st["spoon"]
        handle, blade = g.spoon_boxes(pos, rot)
        near = _near(corner, handle[0], 0.2)
        assert min((g.boxes_gap(handle, b, 5) for b in near), default=1.0) >= 0.015, k
        if st["stroke"] is None:
            near = _near(corner, blade[0], 0.2)
            assert min((g.boxes_gap(blade, b, 5) for b in near), default=1.0) >= 0.003, k


def test_the_hands_stay_apart():
    for st in STEPS[::5]:
        a = M.gripper_boxes(st["left"][2], st["left"][3], 0.05)
        b = M.gripper_boxes(st["right"][2], st["right"][3], 0.05)
        assert min(g.boxes_gap(x, y, 3) for x in a for y in b) > 0.08


def test_the_spoon_is_held_on_the_handle_and_the_stud_between_the_fingers():
    for st in STEPS:
        if st["stroke"] is not None:
            assert st["holding"] and st["right"][1] == pytest.approx(0.0)
    p, r = demo.stud_pose(math.radians(20))
    stud = g.stud_world(0, math.radians(20))
    assert np.linalg.norm((p - stud)[[0, 2]]) < 1e-9
    assert r[:, 0] @ np.array([math.cos(math.radians(20)), 0, -math.sin(math.radians(20))]) == pytest.approx(-1.0)
    # the pads (13 mm past the TCP toward the drum) end 17 mm from its face; the stud reaches 45 mm out
    assert p[1] - 0.013 - (g.B0[1] + g.FACE_Y[1]) == pytest.approx(0.017)


def test_demo_end_state_scores_success():
    end = STEPS[-1]
    assert end["stroke"] is None and not end["holding"] and end["drum"] == 0.0
    grips = [M.gripper_boxes(end[s][2], end[s][3], 0.094 * end[s][1]) for s in ("left", "right")]
    click_to(end["clicks"], spoon=end["spoon"], grips=grips)
    assert M.success(None), M._STATE


def test_arm_links_clear_the_corner():
    """The arms' links (as 3.5 cm capsules from link2 to link5) stay at least 3 cm clear of the corner and stand."""
    from yam_arena import YAM_URDF, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.urdf import forward_kinematics

    corner, cyls = g.corner_boxes_world(), g.corner_cylinders_world()
    chain = ("link1", "link2", "link3", "link4", "link5", "gripper")
    for st in STEPS[::6]:
        for side in ("left", "right"):
            links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, st[side][0])})
            pts = [links[n][:3, 3] + np.asarray(station.BASES[side]) for n in chain]
            seg = np.vstack([a + np.outer(np.linspace(0, 1, 10), b - a) for a, b in zip(pts[:-1], pts[1:])])
            d = min(min(g.point_box_dist(seg, *b).min() for b in corner),
                    min(g.point_cyl_dist(seg, *c).min() for c in cyls))
            assert d - 0.035 >= 0.03, side
