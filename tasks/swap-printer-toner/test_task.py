"""Offline checks (no Isaac Sim): the layout and the mechanism (rails, keyed bosses, seat notch, door interlock), every
scoring level and violation, and the demo (reach, joint limits, clearances, end state).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.task_file import load
from yam_arena.urdf import forward_kinematics

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402

EYE = np.eye(3)
SEAT = g.printer_to_world(g.SEAT_C)
OLD_ON_TABLE = np.array([*g.TABLE_C["old"], g.TABLE + g.C_SIZE[2] / 2])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(a):
    return g.rot_y(a)


def gap(a, b):
    return min(g.boxes_gap(x[1:], y[1:]) for x in a for y in b)


def arm_caps(q, side):
    base = np.asarray(station.BASES[side])
    links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})
    tcp, rot = ik.tcp_pose(q, base)
    return M.arm_capsules([links[n][:3, 3] + base for n in M.CHAIN], tcp, rot[:, 2]), tcp


def far_arms():
    """Both arms at their ready pose, fingers open."""
    from yam_arena.constants import READY

    return {s: {"caps": arm_caps(np.array(READY), s)[0], "grasp": arm_caps(np.array(READY), s)[1],
                "opening": 0.094} for s in ("left", "right")}


def holding(pos, rot=EYE):
    """Arms with the right gripper closed on the handle of a cartridge at ``pos``."""
    arms = far_arms()
    arms["right"] = dict(arms["right"], grasp=M.handle_centre(pos, rot), opening=0.016)
    return arms


def step(old=(g.OLD_START, EYE), new=(g.NEW_START, EYE), door=g.OPEN, arms=None, printer=(None, EYE), speed=0.0,
         new_vel=None):
    pp = g.printer_to_world((0, 0, 0)) if printer[0] is None else printer[0]
    v = np.array([speed, 0.0, 0.0])
    M.update(pp, printer[1], door, {"old": (np.asarray(old[0], float), old[1], v),
                                    "new": (np.asarray(new[0], float), new[1], v if new_vel is None else new_vel)},
             arms or far_arms())


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()


# -- layout and mechanism --------------------------------------------------------------------------------------------


def test_it_loads_with_a_free_standing_printer_and_two_loose_cartridges():
    assert TASK.names == ("printer", "old_cartridge", "new_cartridge")
    assert TASK.score_over == "end" and M.SUCCESS_HOLD_S == 1.0
    assert dict(M.PRINTER.joint_pos) == {"hinge": pytest.approx(math.radians(80))}
    assert not M.PRINTER.self_collisions
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos)  # plain floats for Isaac Lab


def test_the_assets_match_the_geometry():
    pytest.importorskip("pxr")
    import pxr.Usd
    import pxr.UsdGeom
    import pxr.UsdPhysics

    stage = pxr.Usd.Stage.Open(str(HERE / "assets" / "printer.usda"))
    hinge = pxr.UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/printer/Physics/hinge"))
    assert hinge.GetAxisAttr().Get() == "Y"
    assert hinge.GetUpperLimitAttr().Get() == pytest.approx(80.0)
    assert not stage.GetPrimAtPath("/printer/Physics/root_joint")   # not fixed to the table
    body = stage.GetPrimAtPath("/printer/Geometry/body")
    rng = pxr.UsdGeom.Imageable(body).ComputeWorldBound(0, "default").ComputeAlignedRange()
    assert np.array(rng.GetMin()) == pytest.approx([0.0, -0.2, 0.0], abs=2e-3)
    assert np.array(rng.GetMax()) == pytest.approx([0.36, 0.2, 0.23], abs=2e-3)
    roller = stage.GetPrimAtPath("/printer/Geometry/body/transfer_roller")
    r = pxr.UsdGeom.Imageable(roller).ComputeWorldBound(0, "default").ComputeAlignedRange()
    assert np.array(r.GetMax()) - np.array(r.GetMin()) == pytest.approx([0.012, 0.30, 0.012], abs=1e-4)  # along y
    for which in ("old", "new"):
        st = pxr.Usd.Stage.Open(str(HERE / "assets" / f"cartridge_{which}.usda"))
        root = st.GetDefaultPrim()
        assert root.HasAPI(pxr.UsdPhysics.RigidBodyAPI)
        assert pxr.UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(g.MASS[which])
        r = pxr.UsdGeom.Imageable(root).ComputeWorldBound(0, "default").ComputeAlignedRange()
        assert np.array(r.GetMax()) - np.array(r.GetMin()) == pytest.approx([0.101, 0.367, 0.138], abs=1.5e-3)
        # the body collides as its box; what is seen of it is the moulded shell inside that box
        body = st.GetPrimAtPath("/cartridge/body")
        assert body.HasAPI(pxr.UsdPhysics.CollisionAPI)
        assert pxr.UsdGeom.Imageable(body).ComputeVisibility(0) == pxr.UsdGeom.Tokens.invisible
        for name in ("handle", "post_left", "post_right", "boss_left", "boss_right"):
            assert st.GetPrimAtPath(f"/cartridge/{name}").HasAPI(pxr.UsdPhysics.CollisionAPI)
        for name in ("shell", "end_cap_l", "end_cap_r", "drum", "front_label", "top_label"):
            assert not st.GetPrimAtPath(f"/cartridge/{name}").HasAPI(pxr.UsdPhysics.CollisionAPI)
    # the printer's details (output tray, control panel, vents, bay lining, door recess) are only seen
    details = stage.GetPrimAtPath("/printer/Geometry/body/details")
    assert len(details.GetChildren()) > 20
    for prim in [*details.GetChildren(), stage.GetPrimAtPath("/printer/Geometry/body/door/finger_recess")]:
        assert not prim.HasAPI(pxr.UsdPhysics.CollisionAPI), prim.GetPath()


def test_the_shell_is_inside_the_collision_box():
    import make_assets

    prof = np.array(make_assets.shell_profile())
    assert np.all(np.abs(prof) <= np.array([g.C_SIZE[0], g.C_SIZE[2]]) / 2 + 1e-9)


def test_the_handle_is_a_carry_loop_on_the_front_edge():
    """A grip bar along the cartridge's length on two posts, leaning 30 degrees toward the front: 13 cm clear between
    the posts for a hand (or the 6.8 cm wide gripper fingers), a finger gap of about 2 cm under the bar, standing
    about 5 cm above the top, in the front half of the top."""
    boxes = {n: (np.array(c), np.array(sz), p) for n, c, sz, p in g.handle_boxes()}
    bar_c, bar_s, pitch = boxes["handle"]
    assert pitch == pytest.approx(-math.radians(30))
    clear = 2 * (abs(boxes["post_left"][0][1]) - g.POST_W / 2)
    assert clear == pytest.approx(0.13)
    cart = {n: (c, rot, h) for n, c, rot, h in g.cartridge_world((0, 0, 0))}
    c, rot, h = cart["handle"]
    corners = c + (np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h) @ rot.T
    top = g.C_SIZE[2] / 2
    assert 0.014 < corners[:, 2].min() - top < 0.022 and 0.045 < corners[:, 2].max() - top < 0.05
    assert -g.C_SIZE[0] / 2 < corners[:, 0].min() and corners[:, 0].max() < 0.0


def test_full_size_objects():
    assert 0.35 <= 2 * g.P_HALF_W <= 0.42 and 0.33 <= g.P_DEPTH <= 0.40 and 0.20 <= g.P_HEIGHT <= 0.26
    assert g.C_SIZE == (0.10, 0.34, 0.09) and 0.5 <= g.MASS["old"] < g.MASS["new"] <= 0.9


def test_the_old_cartridge_starts_seated_and_the_new_one_clear():
    old = g.cartridge_world(g.OLD_START)
    assert gap(old, g.printer_world()) == pytest.approx(0.0005, abs=2e-4)   # 0.5 mm above the notch floors
    assert gap(old, g.door_world(g.OPEN)) > 0.01
    assert gap(g.cartridge_world(g.NEW_START), g.printer_world()) > 0.015   # beside the printer
    assert M.seated(g.OLD_START, EYE)


def test_a_seated_cartridge_lets_the_door_shut():
    assert gap(g.cartridge_world(SEAT), g.door_world(0.0)) >= 0.004
    assert g.FRONT_GAP == pytest.approx(0.005)


@pytest.mark.parametrize("boss_x", [g.LIP_X - 0.006, g.LIP_X - 0.03, 0.04, 0.0])
def test_an_unseated_cartridge_blocks_the_door_short_of_shut(boss_x):
    """On the rails' end lip, or anywhere up the rails, the cartridge is hit by the door before it is within 2
    degrees of shut (the shut threshold), so a shut door proves a seated cartridge."""
    cart = g.cartridge_world(g.printer_to_world(g.line_centre(boss_x)))
    for deg in (0.0, 1.0, 2.0, 3.0):
        assert gap(cart, g.door_world(math.radians(deg))) < -0.003


def test_the_rails_guide_the_bosses_with_3_mm_play():
    for bx in np.linspace(g.RAIL_FROM, g.LIP_X - 0.008, 6):
        cart = g.cartridge_world(g.printer_to_world(g.line_centre(bx, demo.LINE_LIFT)))
        assert 0.0010 < gap(cart, g.printer_world()) < 0.004
        low = g.cartridge_world(g.printer_to_world(g.line_centre(bx, demo.LINE_LIFT - 0.006)))
        assert gap(low, g.printer_world()) < 0   # 6 mm lower: into the lower rail


def test_the_bosses_are_keyed():
    """Turned end for end, the 18 mm boss meets the 16 mm slot and can't enter the rails."""
    centre = g.printer_to_world(g.line_centre(g.RAIL_FROM + 0.01, demo.LINE_LIFT))
    rot = rz(math.pi)
    turned = g.cartridge_world(centre + rot @ [-g.BOSS_X, 0, 0] + [g.BOSS_X, 0, 0], rot)
    assert gap(turned, g.printer_world()) < 0


@pytest.mark.parametrize("dx", [0.0, -0.004, -0.008, -0.015])
def test_a_cartridge_cannot_be_dropped_into_the_notches_from_above(dx):
    """The slots' ceilings run over the notches to the back stops: lowered straight down at or just in front of its
    seat, a cartridge's bosses hit them, so the only way in is along the rails."""
    printer = [b for b in g.printer_world() if b[0].startswith(("ceiling", "rail"))]
    gaps = [gap(g.cartridge_world(SEAT + [dx, 0, dz]), printer) for dz in np.linspace(0.0, 0.08, 33)]
    assert min(gaps) < -0.003


def test_the_bosses_rise_out_of_the_notches_under_the_ceiling():
    lifted = g.cartridge_world(SEAT + [0, 0, demo.UNSEAT])
    assert gap(lifted, g.printer_world()) > 0.001
    assert demo.UNSEAT > g.LIP_Z - g.SEAT_Z + 0.002


def test_the_seat_notch_holds_the_bosses():
    """Seated, the bosses sit 12 mm below the lip with 2 mm play front and back: the cartridge has to be lifted to
    come out."""
    assert g.LIP_Z - g.SEAT_Z == pytest.approx(0.012)
    fwd = g.cartridge_world(SEAT + [-0.004, 0, 0])
    assert gap(fwd, g.printer_world()) < 0


def test_the_door_stays_shut_and_stays_open_under_its_own_weight():
    com = g.door_com()
    tip = math.degrees(math.atan2(com[0], com[2]))
    assert 8.0 < tip < 20.0                      # shut, its weight is behind the hinge
    assert math.degrees(g.OPEN) > tip + 30       # open, well in front


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_0():
    step()
    assert M.score(None) == 0.0 and not M.success(None)


def test_partial_credit_levels():
    step(old=(OLD_ON_TABLE, EYE))
    assert M.score(None) == 0.2
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE))
    assert M.score(None) == 0.5
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=math.radians(20))
    assert M.score(None) == 0.5
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=math.radians(1.5))
    assert M.score(None) == 0.75


def test_full_score_needs_released_seated_shut_clear_and_still():
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0)
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0)
    assert M.success(None)
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0, speed=0.05)
    assert not M.success(None)                                     # still moving
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0, arms=holding(SEAT))
    assert M.score(None) == 0.2 and not M.success(None)             # still held: not seated


def test_seating_tolerances():
    for off, ok in (([0.003, 0, 0], True), ([0, 0, 0.005], False), ([0.0, 0.0045, 0], False)):
        step(old=(OLD_ON_TABLE, EYE), new=(SEAT + off, EYE), door=0.0)
        assert (M.score(None) == 0.75) is ok
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, ry(math.radians(3))), door=0.0)
    assert M.score(None) == 0.2
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, rz(math.radians(3))), door=0.0)
    assert M.score(None) == 0.2


def test_only_the_new_cartridge_counts_as_seated():
    """Re-inserting the old cartridge, or leaving it in, scores nothing."""
    step(old=(SEAT, EYE), new=(g.NEW_START, EYE), door=0.0)
    assert M.score(None) == 0.0
    M.reset_state()
    step(old=(g.NEW_START, EYE), new=(SEAT, EYE), door=0.0)        # old put where the new one was: 0.75 (it is out)
    assert M.score(None) == 0.75


def test_the_old_one_must_be_out_on_the_table_and_let_go():
    out = np.array([g.X_FRONT - 0.06, 0.0, g.TABLE + 0.2])
    step(old=(out, EYE), arms=holding(out))
    assert M.score(None) == 0.0                                    # still held
    on_top = np.array([g.X_FRONT + 0.2, 0.0, g.TABLE + g.P_HEIGHT + 0.045])
    step(old=(on_top, EYE))
    assert M.score(None) == 0.0                                    # on the printer, not the table
    half = np.array([g.X_FRONT - 0.03, 0.0, g.TABLE + 0.045])
    assert not M.out_of_printer(half, EYE)


@pytest.mark.parametrize("where", ["bar_end", "boss", "body_end"])
def test_a_gripper_closed_anywhere_on_the_cartridge_is_not_let_go(where):
    """Holding the seated new cartridge by the end of its 16 cm grip bar, by a boss or by the body, rather than at the
    bar's middle, still counts as holding it: no credit for "seated and let go"."""
    bar_end = M.handle_centre(SEAT, EYE) + [0, 0.07, 0]
    point = {"bar_end": bar_end, "boss": SEAT + [g.BOSS_X, -(g.C_SIZE[1] / 2 + g.BOSS_OUT), 0.0],
             "body_end": SEAT + [0.0, g.C_SIZE[1] / 2 + 0.02, 0.0]}[where]
    arms = far_arms()
    arms["right"] = dict(arms["right"], grasp=point, opening=0.016)
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0, arms=arms)
    assert M.score(None) == 0.2
    arms["right"] = dict(arms["right"], opening=0.06)          # fingers open there: let go
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0, arms=arms)
    assert M.score(None) == 0.75


@pytest.mark.parametrize("why, kwargs", [
    ("dropped", dict(new=(g.NEW_START + [0, 0, 0.1], EYE))),
    ("tipped", dict(new=(g.NEW_START, ry(math.radians(35))))),
    ("moved", dict(printer=(g.printer_to_world((0.0, 0.015, 0.0)), EYE))),
    ("tipped", dict(printer=(g.printer_to_world((0, 0, 0)), ry(math.radians(3))))),
    ("left the table", dict(new=(np.array([0.2, -0.7, g.TABLE - 0.1]), EYE))),
])
def test_violations_latch_for_the_whole_trial(why, kwargs):
    if why == "dropped":   # held high, then let go: it falls
        lifted = g.NEW_START + [0, 0, 0.1]
        step(new=(lifted, EYE), arms=holding(lifted))
        step(new=(lifted - [0, 0, 0.05], EYE), new_vel=[0, 0, -1.0])
    else:
        step(**kwargs)
    assert M.violation() and any(why in w for w in M._STATE["why"])
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0)
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0)
    assert M.score(None) == 0.0 and not M.success(None)


def test_setting_a_cartridge_down_is_not_a_drop():
    lifted = g.NEW_START + [0, 0, 0.005]
    step(new=(lifted, EYE), arms=holding(lifted))
    step(new=(g.NEW_START, EYE), new_vel=[0, 0, -0.3])       # let go 5 mm up: lands at 0.31 m/s
    assert not M.violation()


def test_pushing_a_let_go_cartridge_down_the_rails_is_not_a_drop():
    """Let go at the rails' entry and pushed home, it descends 3 cm along the rails: not a drop."""
    entry = g.printer_to_world(g.line_centre(g.RAIL_FROM))
    step(new=(entry, EYE), arms=holding(entry))
    for k in np.linspace(0, 1, 8):
        c = entry + k * (SEAT - entry)
        step(new=(c, EYE), new_vel=[0.05, 0, -0.015])
    assert not M.violation()


def test_seating_is_judged_in_the_printer_frame():
    """A printer nudged 6 mm (within its 1 cm limit) with the cartridge seated in it still counts as seated and shut."""
    shift = np.array([0.006, 0.002, 0.0])
    moved = (g.printer_to_world((0, 0, 0)) + shift, EYE)
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT + shift, EYE), door=0.0, printer=moved)
    assert not M.violation() and M.score(None) == 0.75
    M.reset_state()
    step(old=(OLD_ON_TABLE, EYE), new=(SEAT, EYE), door=0.0, printer=(g.printer_to_world((0, 0, 0)) + shift, EYE))
    assert M.score(None) == 0.2                              # the printer moved, the cartridge did not: unseated


def test_touching_the_transfer_roller_is_a_violation():
    arms = far_arms()
    r = g.printer_to_world(g.ROLLER_C)
    arms["left"] = dict(arms["left"], caps=[(r + [0, 0, 0.03], r + [0, 0, 0.30], 0.025)])
    step(arms=arms)
    assert M.violation() and "roller" in M._STATE["why"][0]
    M.reset_state()
    arms["left"] = dict(arms["left"], caps=[(r + [0, 0, 0.04], r + [0, 0, 0.30], 0.025)])
    step(arms=arms)
    assert not M.violation()


# -- the demo --------------------------------------------------------------------------------------------------------

PLAN = demo.plan()
WORST = PLAN.pop()["worst"]


def test_the_demo_is_reachable_within_joint_limits():
    for side, (mm, deg) in WORST.items():
        assert mm < 0.5 and deg < 0.1, side
    for s in PLAN:
        for side in ("left", "right"):
            q = s[side][0]
            assert np.all(q >= ik.LOWER - 1e-6) and np.all(q <= ik.UPPER + 1e-6)


def test_the_demo_fits_the_run_limit():
    assert len(PLAN) + 3 * 20 < 2400 and len(PLAN) < 1200


def _replay(every: int = 2):
    """Walk the plan, carrying each cartridge with the gripper that holds it (position and turn); yield (index, step,
    cartridge poses (centre, rotation))."""
    # settled: the old one resting on its notch floors (the printer stands 0.5 mm up until it settles too)
    pose = {"old": (SEAT + [0.0, 0.0, 0.0005], EYE), "new": (g.NEW_START.copy(), EYE)}
    off, prev = {}, {"left": None, "right": None}
    for i, s in enumerate(PLAN):
        for side in ("left", "right"):
            held = s["holding"][side]
            _, _, pos, rot = s[side]
            if held and prev[side] != held:
                c, r = pose[held]
                off[side] = (rot.T @ (c - pos), rot.T @ r)
            if held:
                pose[held] = (pos + rot @ off[side][0], rot @ off[side][1])
            prev[side] = held
        if i % every == 0 or i == len(PLAN) - 1:
            yield i, s, pose


def test_along_the_demo_nothing_runs_into_the_printer():
    printer = [x[1:] for x in g.printer_world()]
    pushing = False
    for i, s, pose in _replay():
        door = [x[1:] for x in g.door_world(s["door"])]
        pushing = pushing or s["door"] < g.OPEN - 1e-6
        for side in ("left", "right"):
            _, grip, pos, rot = s[side]
            for b in demo.gripper_boxes(pos, rot, opening=min(1.0, grip)):
                assert min(g.boxes_gap(b, p) for p in printer) > 0.004, (i, side)
                d = min(g.boxes_gap(b, p) for p in door)
                # the left fingers press up to 6 mm into where the door would be while they push it
                assert d > (-0.007 if side == "left" and (pushing or i > len(PLAN) * 0.6) else 0.02), (i, side, d)
            caps, _ = arm_caps(s[side][0], side)
            assert M.capsules_gap(caps[:-1], printer) > 0.01, (i, side)
        for name, (c, r) in pose.items():
            cart = g.cartridge_world(c, r)
            assert gap(cart, g.printer_world()) > 0.0004, (i, name)
            assert gap(cart, g.door_world(s["door"])) > (0.006 if s["door"] == g.OPEN else 0.002), (i, name)
            assert M.tilt_of(r) < math.radians(0.5), (i, name)               # carried level
        assert gap(g.cartridge_world(*pose["old"]), g.cartridge_world(*pose["new"])) > 0.02


def test_the_demo_grips_only_the_handle():
    for i, s, pose in _replay(1):
        for side in ("left", "right"):
            held = s["holding"][side]
            if held:
                _, grip, pos, rot = s[side]
                cart = g.cartridge_world(*pose[held])
                fingers = demo.gripper_boxes(pos, rot, grip)[0]
                assert g.boxes_gap(fingers, cart[0][1:]) > 0.004, i                      # clear of the body ...
                for post in (b for b in cart if b[0].startswith("post")):
                    assert g.boxes_gap(fingers, post[1:]) > 0.01, i                      # ... and the handle's posts
                bar = next(b for b in cart if b[0] == "handle")
                assert g.boxes_gap(fingers, bar[1:]) < 0, i                              # the fingers are on the bar
                assert np.linalg.norm(pos - M.handle_centre(*pose[held])) < M.HOLD_R / 3, i


def test_the_demo_ends_in_a_full_score_state():
    *_, (i, s, pose) = _replay()
    assert np.linalg.norm(pose["new"][0] - SEAT) < 0.0025 and M.tilt_of(pose["new"][1]) < 1e-6
    assert abs(M.yaw_of(pose["new"][1])) < 1e-6 and abs(M.yaw_of(pose["old"][1])) < 1e-6
    arms = {}
    for side in ("left", "right"):
        caps, tcp = arm_caps(s[side][0], side)
        arms[side] = {"caps": caps, "grasp": tcp, "opening": 0.094}
    old_rest = pose["old"][0] - [0, 0, 0.003]       # it is let go 3 mm above the table
    step(old=(old_rest, EYE), new=(SEAT, EYE), door=0.0, arms=arms)
    step(old=(old_rest, EYE), new=(SEAT, EYE), door=0.0, arms=arms)
    assert M.score(None) == 0.75 and M.success(None)


def test_a_seated_cartridge_rests_level_on_its_bosses_and_the_floor_ribs():
    """The bosses sit behind the cartridge's centre of mass; two ribs on the bay floor carry its front, so it can't
    tip forward by more than about half a degree."""
    pivot = SEAT + [g.BOSS_X, 0, -0.005]          # the bosses' bottom edge
    rot = ry(-math.radians(1.0))                  # front down by 1 degree
    tipped = g.cartridge_world(pivot + rot @ (SEAT - pivot), rot)
    assert gap(tipped, [b for b in g.printer_world() if b[0].startswith("support")]) < 0


def test_the_bosses_stay_on_the_notch_floors_across_the_side_play():
    """The bosses' tips are 1.5 mm from the side walls, so the cartridge can't slide sideways off a notch floor."""
    tip = g.C_SIZE[1] / 2 + g.BOSS_OUT
    assert g.WALL_IN - tip == pytest.approx(0.0015)
    assert tip - 0.0015 - g.RAIL_Y[0] > 0.005     # at least 5 mm of each boss still over its notch floor and rail
