"""Offline checks (no Isaac Sim): the layout and assets, the recess and the bit's seat, the blow detector and what each
kind of blow does, every scoring level and violation, and the demo (reach, joint limits, clearances, the blows' aim,
end state).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
import re
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
DT = 1 / 240
FAR = [M.gripper_boxes(np.array([0.0, s * 0.30, g.TABLE + 0.40]), np.diag([1.0, -1.0, -1.0]), 0.094)
       for s in (1, -1)]
START = dict(driver_pos=g.DRIVER_START, driver_rot=EYE, hammer_pos=g.HAMMER_START, hammer_rot=g.HAMMER_STRIKE)


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()


def monitor(driver_pos=None, driver_rot=None, hammer_pos=None, hammer_rot=None, grips=None, vel=0.0):
    a = dict(START)
    a.update({k: v for k, v in dict(driver_pos=driver_pos, driver_rot=driver_rot, hammer_pos=hammer_pos,
                                     hammer_rot=hammer_rot).items() if v is not None})
    v = np.array([vel, 0.0, 0.0])
    return M.update(g.F0, EYE, a["driver_pos"], a["driver_rot"], v, a["hammer_pos"], a["hammer_rot"], v,
                    FAR if grips is None else grips)


def hammer_at(cap, above, rot=None):
    """Hammer origin with its face's centre ``above`` over ``cap`` (face down)."""
    rot = g.HAMMER_STRIKE if rot is None else rot
    return np.asarray(cap, float) + [0.0, 0.0, above] - rot @ np.array([0.0, 0.0, g.FACE_TIP]), rot


def swing(screw="a", speed=2.5, tip=None, driver_rot=EYE, offset=(0.0, 0.0), rot=None, start=0.10, impulse=0.6,
          stroke=g.STROKE, held=True, stroke0=0.0):
    """Bring the hammer down onto the driver's cap at ``speed`` (one substep at a time) from ``start`` above it, the
    bit's tip at ``tip`` (default: fully home in ``screw``); the contact passes ``impulse`` in its first substep and the
    stroke closes to ``stroke``; then the hammer rests on the cap and is lifted off. Returns the blows judged."""
    tip = demo.seated_tip(screw) if tip is None else np.asarray(tip, float)
    axis = driver_rot[:, 2]
    cap = tip + driver_rot @ np.array([0.0, 0.0, g.DRIVER_LEN])
    off = np.array([offset[0], offset[1], 0.0])
    n0 = len(M._STATE["blows"])
    M._STATE["held"] = held                       # (the control-step monitor sets it from the grippers' gap)

    def sub(h, j=0.0, q=stroke0):
        M.blow_step(tip, driver_rot, tip, axis, q, *hammer_at(cap + off, h, rot), j, DT)

    h = start
    while h > 0.0:
        sub(h)
        h -= speed * DT
    sub(0.0, impulse, (stroke0 + stroke) / 2)
    for k in range(g.WINDOW + 5):                 # the stroke closes, the spring returns it; the hammer rests on the cap
        sub(0.0, 0.05, stroke if k < 4 else stroke0)
    for k in range(10):                           # lifted off again
        sub(0.002 * k)
    return M._STATE["blows"][n0:]


# -- layout, assets and size ----------------------------------------------------------------------------------------


def test_it_loads_with_a_fixed_hub_and_loose_tools():
    assert TASK.names == ("hub", "block", "stand", "driver", "hammer")
    assert TASK.score_over == "end" and M.SUCCESS_HOLD_S == 1.0
    assert dict(M.HUB.joint_pos) == {"screw_a": 0.0}            # the other screw is already out
    assert not M.HUB.self_collisions and M.BLOCK.mass is None and M.STAND.mass is None
    assert M.DRIVER.articulation and dict(M.DRIVER.joint_pos) == {"stroke": 0.0, "cam": 0.0}
    assert g.BODY_MASS + g.ANVIL_MASS + g.SLIDER_MASS == pytest.approx(g.DRIVER_MASS)
    assert M.HAMMER.mass == pytest.approx(0.62)
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos) and all(type(v) is float for v in spec.rot_xyzw)
    for d in M.HUB.drives + M.DRIVER.drives:
        assert all(type(v) in (str, float) for v in d)


def test_the_return_spring_and_the_cam():
    from make_assets import SPRING_REST

    # the spring pushes the anvil out with 20 N at the open end and 35 N bottomed: a hand's push barely moves it
    assert g.SPRING_K * (0.0 - SPRING_REST) == pytest.approx(20.0)
    assert g.SPRING_K * (g.STROKE - SPRING_REST) == pytest.approx(35.0)
    assert dict(M.DRIVER.drive_targets) == {"stroke": SPRING_REST}
    # the cam turns the bit its 12 degrees over the full stroke; "bottomed" is 87 % of it
    assert math.degrees(g.CAM * g.STROKE) == pytest.approx(math.degrees(g.TURN_PER_BLOW))
    assert 0.8 * g.STROKE <= g.BOTTOM < g.STROKE


def test_the_driver_body_lies_flat_on_the_pads():
    # the body's facets face every 30 degrees: one faces along the right hand's closing axis
    a = math.degrees(math.atan2(demo.DRIVER_GRIP[1, 0], demo.DRIVER_GRIP[0, 0])) % 30.0
    assert min(a, 30.0 - a) < 1e-6
    for _, c, s, r in g.driver_body_boxes():
        assert np.linalg.norm(np.asarray(c)[:2]) + s[0] / 2 == pytest.approx(0.018)


def test_the_assets_match_the_geometry():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    d = Usd.Stage.Open(str(HERE / "assets" / "driver.usda"))
    stroke = UsdPhysics.PrismaticJoint(d.GetPrimAtPath("/driver/Physics/stroke"))
    assert stroke.GetAxisAttr().Get() == "Z" and stroke.GetUpperLimitAttr().Get() == pytest.approx(g.STROKE)
    assert UsdPhysics.RevoluteJoint(d.GetPrimAtPath("/driver/Physics/cam")).GetAxisAttr().Get() == "Z"
    assert not d.GetPrimAtPath("/driver/Physics/root_joint")              # a free tool
    def schemas(prim):        # as authored (usd-core doesn't know PhysX's schemas, so GetAppliedSchemas drops them)
        return list(prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems())

    assert "PhysxContactReportAPI" in schemas(d.GetPrimAtPath("/driver/Geometry/body"))
    h = Usd.Stage.Open(str(HERE / "assets" / "hammer.usda"))
    assert "PhysxContactReportAPI" in schemas(h.GetPrimAtPath("/hammer"))
    st = Usd.Stage.Open(str(HERE / "assets" / "hub.usda"))
    assert st.GetPrimAtPath("/hub/Physics/root_joint")
    for k, pos in g.SCREWS.items():
        j = UsdPhysics.RevoluteJoint(st.GetPrimAtPath(f"/hub/Physics/screw_{k}"))
        assert j.GetAxisAttr().Get() == "Z"
        assert np.array(j.GetLocalPos0Attr().Get()) == pytest.approx(pos, abs=1e-6)
        kids = [p.GetName() for p in st.GetPrimAtPath(f"/hub/Geometry/base/screw_{k}").GetChildren()]
        assert sum(n.startswith("petal") for n in kids) == g.N_PETALS
        cross = st.GetPrimAtPath(f"/hub/Geometry/base/screw_{k}/cross_0")
        assert not cross.HasAPI(UsdPhysics.CollisionAPI)          # the drawn cross doesn't collide
    for name in ("driver", "hammer", "block", "stand"):
        assert (HERE / "assets" / f"{name}.usda").is_file()


def test_full_size_parts():
    assert 2 * g.RING_R[1] == pytest.approx(0.262)                 # 262 mm front rotor
    assert 2 * g.STUD_PCD_R == pytest.approx(0.100)                # 4 x 100 studs, M12
    assert 2 * g.RECESS_R == pytest.approx(0.0066)                 # ISO 7046 M6: PH3, m = 6.6 mm
    assert 2 * g.HEAD_R == pytest.approx(0.0116, abs=0.0006)
    assert g.DRIVER_LEN == pytest.approx(0.17) and 2 * g.CAP_R == pytest.approx(0.039)
    assert g.HEAD_MASS + g.HANDLE_MASS == pytest.approx(g.HAMMER_MASS)
    assert g.HANDLE_Y0 + g.HANDLE[1] == pytest.approx(0.335)


def test_layout_is_clear_and_in_reach():
    # nothing overlaps at the start: the tools are clear of the hub and the block holds the driver
    tools = M.tools_boxes(g.DRIVER_START, EYE, g.HAMMER_START, g.HAMMER_STRIKE)
    assert g.gap_to_fixture(tools) >= 0.02
    # the hammer stands on its face on the stand's lower step; its handle lies on the upper step; its balance point
    # is over the stand, between the face and the step's far end; the stand clears the rotor by 2 cm
    face = g.HAMMER_START + g.HAMMER_STRIKE @ np.array([0.0, 0.0, g.FACE_TIP])
    assert face[2] == pytest.approx(g.TABLE + g.STAND[1], abs=0.001)
    handle_bottom = g.HAMMER_START[2] - g.HANDLE[2] / 2
    assert handle_bottom == pytest.approx(g.TABLE + g.STAND[1] + g.STAND_STEP[2], abs=0.001)
    assert g.STAND_STEP[0] - g.FACE_R >= 0.01 and g.STAND_STEP[0] < g.HAMMER_COM[1] < g.STAND_STEP[1]
    stand_near = g.STAND_C[1] - 0.03
    rotor_edge = g.F0[1] + math.sqrt(g.RING_R[1] ** 2 - (g.STAND_C[0] + g.STAND[0] / 2 - g.F0[0]) ** 2)
    assert stand_near - rotor_edge >= 0.02
    # the driver's body sits on the block; the nose and bit hang in the pocket with play all round
    assert g.DRIVER_START[2] + 0.050 == pytest.approx(g.TABLE + g.BLOCK[1])
    assert g.POCKET[0] / 2 - 0.011 >= 0.0035
    assert g.DRIVER_START[2] - (g.TABLE + g.BLOCK[1] - g.POCKET[1]) >= 0.004
    assert 0.018 - g.POCKET[0] / 2 >= 0.0025


def test_the_screws_sit_between_the_studs_and_the_pilot():
    for p in g.SCREWS.values():
        r = np.linalg.norm(p[:2])
        assert r - g.HOLE_HALF * math.sqrt(2) - g.PILOT[0] > 0.0
        for a in g.STUD_ANGLES:
            stud = g.STUD_PCD_R * np.array([math.cos(a), math.sin(a)])
            # the driver's nose (22 mm) clears every stud by at least 15 mm
            assert np.linalg.norm(p[:2] - stud) - 0.011 - g.STUD_R >= 0.015
    # the face plate leaves an opening round each head
    for name, c, s, _ in g.fixture_boxes():
        if name.startswith("face"):
            for p in g.SCREWS.values():
                gap = np.abs(np.asarray(c[:2]) - p[:2]) - np.asarray(s[:2]) / 2
                assert max(gap) >= -1e-9          # no strip reaches into the square opening round the head
                assert max(np.abs(np.asarray(c[:2]) - p[:2]) - np.asarray(s[:2]) / 2 + g.HOLE_HALF) >= g.HOLE_HALF


def test_recess_walls_form_the_cone_and_the_bit_seats_deep():
    # each petal's inner face passes through the cone at its centre
    for name, c, s, r in g.screw_boxes():
        if not name.startswith("petal"):
            continue
        inner = np.asarray(c) - r[:, 2] * s[2] / 2
        radius = math.hypot(inner[0], inner[1])
        assert radius == pytest.approx((g.RECESS_R + g.RECESS_BOTTOM_R) / 2, abs=2e-5)
        assert inner[2] == pytest.approx(-g.RECESS_D / 2, abs=2e-5)
    # the bit's third step meets the walls 3.1 mm down: 1.1 mm deeper than a seat needs, 0.5 mm off the floor
    assert g.SEAT_DEPTH == pytest.approx(0.00312, abs=1e-5)
    assert g.SEAT_DEPTH - g.SEAT_MIN >= 0.001 and g.RECESS_D - g.SEAT_DEPTH >= 0.0004
    # the bit's shank is wider than the recess: it can't go in past the head's top
    assert g.DRIVER_PARTS[3][1] > g.RECESS_R


def test_seated_measures_offset_depth_and_tilt():
    s = g.screw_world("a")
    ok, radial, depth, tilt = g.seated(s - [0, 0, g.SEAT_DEPTH], [0, 0, 1], s)
    assert ok and radial == pytest.approx(0) and depth == pytest.approx(g.SEAT_DEPTH) and tilt == pytest.approx(0)
    assert not g.seated(s - [0, 0, 0.0015], [0, 0, 1], s)[0]                     # only 1.5 mm in
    assert not g.seated(s + [0.0012, 0, -g.SEAT_DEPTH], [0, 0, 1], s)[0]        # 1.2 mm off the axis
    tilted = g.rot_x(math.radians(6)) @ np.array([0, 0, 1.0])
    assert not g.seated(s - [0, 0, g.SEAT_DEPTH], tilted, s)[0]                  # 6 degrees off square
    assert g.seated(s - [0, 0, g.SEAT_DEPTH], g.rot_x(math.radians(4)) @ np.array([0, 0, 1.0]), s)[0]


# -- blows -----------------------------------------------------------------------------------------------------------


def test_a_good_blow_turns_the_screw_12_degrees_once():
    blows = swing("a", speed=2.5)
    assert [b["kind"] for b in blows] == ["good"] and blows[0]["screw"] == "a"
    assert blows[0]["speed"] == pytest.approx(2.5, rel=0.02) and blows[0]["stroke_mm"] == pytest.approx(3.0)
    assert M._STATE["turned"]["a"] == pytest.approx(math.radians(12))


def test_resting_or_pressing_the_hammer_on_the_cap_is_no_blow():
    tip = demo.seated_tip("a")
    cap = tip + [0, 0, g.DRIVER_LEN]
    for k in range(200):                     # set down on the cap slowly, then pressed on it
        M.blow_step(tip, EYE, tip, EYE[:, 2], 0.0, *hammer_at(cap, max(0.0, 0.10 - 0.001 * k)), 0.1, DT)
    assert M._STATE["blows"] == [] and M._STATE["pending"] is None


def test_a_contact_below_the_impulse_threshold_is_no_blow():
    assert swing("a", impulse=g.J_MIN * 0.9) == []


def test_without_a_5_cm_wind_up_the_next_contact_is_no_blow():
    assert [b["kind"] for b in swing("a")] == ["good"]
    assert swing("a", start=0.04) == []                   # lifted only 4 cm: not armed
    assert [b["kind"] for b in swing("a", start=0.06)] == ["good"]


def test_a_blow_on_a_driver_nobody_holds_does_nothing():
    blows = swing("a", held=False)
    assert [b["kind"] for b in blows] == ["unheld"] and M._STATE["turned"]["a"] == 0.0


def test_the_hold_is_read_from_the_grippers():
    hand = M.gripper_boxes(demo.seated_tip("a") + [0, 0, g.GRIP_Z], demo.DRIVER_GRIP, 0.036)
    monitor(driver_pos=demo.seated_tip("a"), grips=[hand, FAR[1]])
    assert M._STATE["held"]
    monitor(driver_pos=demo.seated_tip("a"))
    assert not M._STATE["held"]


def test_pressing_the_stroke_down_then_tapping_it_does_not_count():
    # a 35 N push on the body bottoms the stroke (3 mm) before the hammer touches it: the blow drives it no deeper
    from make_assets import SPRING_REST

    assert g.SPRING_K * (g.STROKE - SPRING_REST) == pytest.approx(35.0)
    blows = swing("a", stroke0=g.STROKE, stroke=g.STROKE)
    assert [b["kind"] for b in blows] == ["short"] and M._STATE["turned"]["a"] == 0.0
    # half pressed (1.5 mm) and struck to the bottom: only 1.5 mm deeper, short too
    assert [b["kind"] for b in swing("a", stroke0=0.0015, stroke=g.STROKE)] == ["short"]


def test_a_blow_that_does_not_bottom_the_stroke_turns_nothing():
    blows = swing("a", stroke=0.0015)
    assert [b["kind"] for b in blows] == ["short"] and M._STATE["turned"]["a"] == 0.0
    assert M._STATE["camouts"]["a"] == 0


def test_a_slow_tap_does_nothing():
    blows = swing("a", speed=1.6)
    assert [b["kind"] for b in blows] == ["tap"] and M._STATE["turned"]["a"] == 0.0
    assert M._STATE["camouts"]["a"] == 0


def test_a_glancing_or_off_centre_blow_does_nothing():
    assert [b["kind"] for b in swing("a", offset=(0.0, 0.022))] == ["glancing"]
    rot = g.rot_x(math.radians(25)) @ g.HAMMER_STRIKE
    assert [b["kind"] for b in swing("a", rot=rot)] == ["glancing"]
    assert M._STATE["turned"]["a"] == 0.0 and M._STATE["camouts"]["a"] == 0


def test_a_miss_is_no_blow():
    assert swing("a", offset=(0.0, 0.040), impulse=0.0) == []      # no contact with the driver: nothing


def test_a_blow_with_the_bit_off_every_screw_does_nothing():
    blows = swing(tip=g.screw_world("a") + [0.0, 0.0, 0.02])
    assert [b["kind"] for b in blows] == ["air"]


def test_unseated_or_tilted_blows_cam_out_and_two_strip_the_screw():
    s = g.screw_world("a")
    assert [b["kind"] for b in swing(tip=s - [0, 0, 0.001])] == ["camout"]           # only 1 mm in
    assert M._STATE["camouts"]["a"] == 1 and not M._STATE["ruined"]["a"]
    tilt = g.rot_x(math.radians(7))
    blows = swing(tip=s - [0, 0, g.SEAT_DEPTH], driver_rot=tilt)
    assert [b["kind"] for b in blows] == ["camout"] and blows[0]["tilt_deg"] == pytest.approx(7, abs=0.1)
    assert M._STATE["ruined"]["a"]
    # stripped: good blows no longer turn it
    assert [b["kind"] for b in swing("a")] == ["stripped"] and M._STATE["turned"]["a"] == 0.0


def test_blow_attributes_the_driver_as_it_was_just_before_contact():
    blows = swing("a")
    assert blows[0]["depth_mm"] == pytest.approx(g.SEAT_DEPTH * 1000, abs=0.01)
    assert blows[0]["radial_mm"] == pytest.approx(0.0)


def test_the_removed_screws_hole_is_open():
    assert len(g.HOLES) == 2 and len(g.SCREWS) == 1
    for name, c, sz, _ in g.fixture_boxes():
        if name.startswith("face"):
            gap = np.abs(np.asarray(c[:2]) - g.OUT_HOLE[:2]) - np.asarray(sz[:2]) / 2
            assert max(gap) >= g.HOLE_HALF - 1e-9 or max(gap) >= -1e-9


def test_blows_in_the_empty_hole_do_nothing():
    out = g.F0 + g.OUT_HOLE                     # the screw already taken out: its hole is open
    assert [b["kind"] for b in swing(tip=out - [0, 0, 0.003])] == ["air"]
    assert M._STATE["turned"]["a"] == 0.0 and M._STATE["camouts"]["a"] == 0


# -- scoring -----------------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_zero():
    monitor()
    assert M.score(None) == 0.0 and not M.success(None)


@pytest.mark.parametrize("good,camouts,want", [(1, 0, 0.32), (2, 0, 0.64), (3, 0, 0.8), (5, 0, 0.8), (3, 1, 0.7),
                                               (1, 1, 0.22)])
def test_partial_credit_per_good_blow(good, camouts, want):
    for _ in range(camouts):
        swing(tip=g.screw_world("a") - [0, 0, 0.001])
    for _ in range(good):
        swing("a")
    monitor(driver_pos=demo.seated_tip("a"))         # the driver still on the screw: not done
    assert M.score(None) == pytest.approx(want, abs=1e-3)


def test_loose_and_tools_down_and_let_go_is_success():
    for _ in range(3):
        swing("a")
    monitor()
    assert M._STATE["off_hub"] and M._STATE["clear"] and M._STATE["still"]
    assert M.success(None) and M.score(None) == 0.9


def test_a_cam_out_costs_0_1_and_rules_out_success():
    swing(tip=g.screw_world("a") - [0, 0, 0.001])
    for _ in range(3):
        swing("a")
    monitor()
    assert M._STATE["level"] == pytest.approx(0.9) and not M.success(None) and M.score(None) == pytest.approx(0.9)


def test_a_stripped_screw_scores_nothing():
    for _ in range(2):
        swing(tip=g.screw_world("a") - [0, 0, 0.001])
    for _ in range(3):
        swing("a")
    monitor()
    assert M.score(None) == 0.0 and not M.success(None)


def test_not_done_while_holding_moving_or_leaving_a_tool_on_the_hub():
    for _ in range(3):
        swing("a")
    near = [M.gripper_boxes(g.DRIVER_START + [0, 0, g.GRIP_Z], demo.DRIVER_GRIP, 0.036), FAR[0]]
    monitor(grips=near)
    assert not M._STATE["clear"] and M.score(None) == pytest.approx(0.8)
    monitor(vel=0.05)
    assert not M._STATE["still"] and not M.success(None)
    monitor(driver_pos=demo.seated_tip("a"))
    assert not M._STATE["off_hub"] and M.score(None) == pytest.approx(0.8)
    on_hat = hammer_at(g.F0 + [0.0, 0.06, g.FACE + 0.001], 0.0)
    monitor(hammer_pos=on_hat[0], hammer_rot=on_hat[1])
    assert not M._STATE["off_hub"]


def test_a_tool_off_the_table_scores_zero():
    for _ in range(3):
        swing("a")
    monitor(hammer_pos=np.array([0.2, 0.7, 0.3]))
    assert M.violation() and M.score(None) == 0.0
    M.reset_state()
    monitor(driver_pos=np.array([0.2, -0.7, 0.2]))
    assert M.violation() and M.score(None) == 0.0


def test_setting_tools_down_on_the_table_or_back_in_the_block_both_count():
    for _ in range(3):
        swing("a")
    lying = g.rot_x(math.pi / 2)                     # the driver on its side on the table
    monitor(driver_pos=np.array([0.25, -0.20, g.TABLE + 0.0195]), driver_rot=lying)
    assert M.success(None)


# -- the demo --------------------------------------------------------------------------------------------------------


def test_demo_reach_and_limits():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert worst[side][0] < 0.5 and worst[side][1] < 0.1
        for st in STEPS:
            q = st[side][0]
            assert np.all(q >= ik.LOWER - 1e-9) and np.all(q <= ik.UPPER + 1e-9)


def test_demo_fits_the_run_limit_and_strikes_each_screw():
    run_limit = 3600
    assert len(STEPS) + 3 * demo.CONTROL_HZ + 60 < run_limit
    strikes = PLAN[-1]["strikes"]
    assert [s for _, s in strikes] == ["a"] * demo.BLOWS
    assert demo.BLOWS * g.TURN_PER_BLOW >= g.LOOSE + g.TURN_PER_BLOW - 1e-9   # one spare blow per screw
    for k, screw in strikes:
        assert STEPS[k]["seated"] == screw and STEPS[k]["holding"] == {"driver": True, "hammer": True}


def test_demo_does_not_read_simulator_objects():
    src = (HERE / "demo.py").read_text()
    for bad in ("object_states", "root_link_pose", "body_pos_w", "pose_np", "joints.get"):
        assert bad not in src.split("def run(")[0]
    run_src = src.split("def run(")[1]
    # in run(), the task's state is only printed and reported, never fed back into the script
    assert "script.flags" not in run_src and "script.n_blows" not in run_src
    before = run_src.split("next(gen)")[0]
    assert "_STATE" not in before and not re.search(r"\bS\[", before)
    # the only sensing fed back: the arms' own joint and finger encoders
    assert "object_states" not in run_src and "pose_np" not in run_src


def test_the_blow_aims_the_face_square_on_the_cap():
    for screw in ("a",):
        cap = demo.cap_at(screw)
        tcp, rot = demo.strike_tcp(cap)
        o, rh = demo.hammer_from_tcp(tcp, rot)
        h, lateral, tilt = M.face_vs_cap(demo.seated_tip(screw), EYE, o, rh)
        assert h == pytest.approx(0, abs=1e-9) and lateral == pytest.approx(0, abs=1e-9)
        assert tilt == pytest.approx(0, abs=1e-6)
        # the swing, in joint space about the elbow: wound up, the face is well over the 5 cm the next blow needs; at
        # the target it is past the cap; through the cap its arc runs within 15 degrees of the driver's axis
        from yam_arena import station

        base = np.asarray(station.BASES["left"], float)
        q, _, _ = ik.solve(tcp, ik.matrix_to_quat_xyzw(rot), q_init=np.array([-0.6, 1.4, 1.0, 0.4, -0.6, 1.57]),
                           base_pos=base, iters=300)

        def face_h(dq):
            qq = q.copy()
            qq[demo.SWING_JOINT] += dq
            return M.face_vs_cap(demo.seated_tip(screw), EYE, *demo.hammer_from_tcp(*ik.tcp_pose(qq, base)))

        assert face_h(0.0)[0] == pytest.approx(0, abs=1e-4)
        assert face_h(demo.WIND)[0] >= g.H_ARM + 0.10
        assert face_h(-demo.THROUGH)[0] < -0.05
        d1, d2 = face_h(1e-3), face_h(-1e-3)
        dh, dl = d1[0] - d2[0], np.hypot(d1[1], d2[1])
        assert math.degrees(math.atan2(dl, abs(dh))) < 15.0
    # the aim allowance for the wrist's flex is small next to the cap (39 mm) and the 15 mm tolerance
    assert np.linalg.norm(demo.AIM) < 0.025 and demo.AIM[2] == 0.0
    # the hand holds the handle 12 cm from the head, the pads (4 cm) clear of the head
    assert g.HAMMER_GRIP_Y - 0.02 - g.FACE_R >= 0.08


def test_the_hammer_grip_stays_the_same_from_pick_to_strike():
    tcp, srot = demo.hammer_tcp(g.HAMMER_START, g.HAMMER_STRIKE)
    # nothing under the handle where the lower finger goes (the stand's step ends 3 cm short of the pads)
    assert g.HAMMER_GRIP_Y - 0.02 - g.STAND_STEP[1] >= 0.025
    assert srot[:, 2] == pytest.approx([1, 0, 0])                    # from behind when it strikes
    assert abs(srot[:, 0] @ [0, 0, 1]) == pytest.approx(1)           # fingers top and bottom of the handle


def test_grippers_clear_the_hub_the_table_and_the_studs():
    fix = g.fixture_world_boxes()
    cyls = g.fixture_world_cylinders()
    for k, st in enumerate(STEPS[::3]):
        for side in ("left", "right"):
            q, grip, tcp, rot = st[side]
            boxes = M.gripper_boxes(tcp, rot, 0.094 * grip if grip > 0 else 0.0)
            pts = np.vstack([g.box_points(*b, 5) for b in boxes])
            assert pts[:, 2].min() >= g.TABLE + 0.004, (side, k)
            d = min(min(g.point_cyl_dist(pts, *c).min() for c in cyls),
                    min(g.point_box_dist(pts, *b).min() for b in fix))
            assert d >= 0.015, (side, k, d)


def test_the_carried_tools_clear_the_hub_except_the_seated_bit():
    for k, st in enumerate(STEPS[::2]):
        hammer = g.hammer_parts(*st["hammer"])
        assert g.gap_to_fixture(hammer) >= 0.01, k
        body = g.driver_boxes(st["driver"], EYE)[1:]          # nose, body and cap
        assert g.gap_to_fixture(body) >= 0.01, k
        far = min(np.linalg.norm(st["driver"] - g.screw_world(k)) for k in g.SCREWS) > 0.015
        if st["seated"] is None and far:
            assert g.gap_to_fixture(g.driver_boxes(st["driver"], EYE)) >= 0.008, k


def test_the_carried_tools_clear_each_other_and_the_other_hand():
    for k, st in enumerate(STEPS[::2]):
        hammer = g.hammer_parts(*st["hammer"])
        driver = g.driver_boxes(st["driver"], EYE)
        # the hammer only meets the driver at its cap (face on cap), never its body, nose or the hand on it
        gap = min(g.boxes_gap(a, b, 5) for a in hammer for b in driver[:-1])
        assert gap >= 0.008, k
        right = M.gripper_boxes(st["right"][2], st["right"][3], 0.036 if st["holding"]["driver"] else 0.094)
        assert min(g.boxes_gap(a, b, 5) for a in hammer for b in right) >= 0.015, k
        left = M.gripper_boxes(st["left"][2], st["left"][3], 0.03 if st["holding"]["hammer"] else 0.094)
        assert min(g.boxes_gap(a, b, 5) for a in driver for b in left) >= 0.015, k


def test_the_hands_stay_apart():
    for st in STEPS[::5]:
        a = M.gripper_boxes(st["left"][2], st["left"][3], 0.05)
        b = M.gripper_boxes(st["right"][2], st["right"][3], 0.05)
        assert min(g.boxes_gap(x, y, 3) for x in a for y in b) > 0.025


def test_the_bit_goes_home_square_over_each_screw():
    for screw in ("a",):
        seated = [st for st in STEPS if st["seated"] == screw]
        assert seated
        for st in seated:
            assert g.seated(st["driver"], [0, 0, 1], g.screw_world(screw))[0]
        # the right hand presses 2 mm past home; its grip is across the body's middle
        tcp = seated[0]["right"][2]
        assert tcp[2] - demo.seated_tip(screw)[2] - g.GRIP_Z == pytest.approx(-demo.PRESS, abs=1e-4)


def test_demo_end_state_scores_success():
    end = STEPS[-1]
    assert not any(end["holding"].values()) and end["seated"] is None
    for _ in range(3):
        swing("a")
    grips = [M.gripper_boxes(end[s][2], end[s][3], 0.094 * end[s][1]) for s in ("left", "right")]
    monitor(driver_pos=end["driver"], hammer_pos=end["hammer"][0], hammer_rot=end["hammer"][1], grips=grips)
    assert M.success(None), M._STATE


def test_arm_links_clear_the_hub():
    """The arms' links (as 3.5 cm capsules from link1 to the gripper) stay at least 2 cm clear of the hub and rotor."""
    from yam_arena import YAM_URDF, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.urdf import forward_kinematics

    cyls = g.fixture_world_cylinders()
    chain = ("link1", "link2", "link3", "link4", "link5", "gripper")
    for st in STEPS[::6]:
        for side in ("left", "right"):
            links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, st[side][0])})
            pts = [links[n][:3, 3] + np.asarray(station.BASES[side]) for n in chain]
            seg = np.vstack([a + np.outer(np.linspace(0, 1, 10), b - a) for a, b in zip(pts[:-1], pts[1:])])
            d = min(g.point_cyl_dist(seg, *c).min() for c in cyls)
            assert d - 0.035 >= 0.02, side
