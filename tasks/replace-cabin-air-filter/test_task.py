"""Offline checks (no Isaac Sim): the layout and the mechanism (stops, flexing panels, keyed slot), every scoring level
and violation, and the demo (reach, joint limits, clearances, end state).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, READY
from yam_arena.task_file import load
from yam_arena.urdf import forward_kinematics

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402

EYE = np.eye(3)
SEAT = M.SEAT_WORLD
OLD_DOWN = np.array([*g.OLD_DOWN, g.TABLE + g.F_SIZE[2] / 2])
STOP = g.OPEN_STOP
LOW = g.LOWERED
CHAIN = ("link2", "link3", "link4", "link5", "gripper")


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def far_arms():
    """Both arms at their ready pose, fingers open (3 cm from the open glovebox)."""
    out = {}
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(np.array(READY), station.BASES[side])
        out[side] = {"tcp": tcp, "rot": rot, "opening": 0.094}
    return out


def holding(pos, rot=EYE):
    """Arms with the right gripper closed on the tab of a filter at ``pos`` (fingers closed on the 16 mm tab)."""
    arms = far_arms()
    arms["right"] = dict(arms["right"], tcp=M.tab_point(pos, rot), opening=0.016)
    return arms


def step(theta=STOP, pins=(0.0, 0.0), old=(g.OLD_START, EYE), new=(g.NEW_START, EYE), arms=None, speed=0.0):
    v = np.array([speed, 0.0, 0.0])
    M.update(theta, pins, {"old": (np.asarray(old[0], float), old[1], v), "new": (np.asarray(new[0], float), new[1], v)},
             arms or far_arms())


def lowered():
    step(theta=LOW)


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()


# -- layout and mechanism --------------------------------------------------------------------------------------------


def test_it_loads_with_a_fixed_dash_and_two_loose_filters():
    assert TASK.names == ("dash", "old_filter", "new_filter", "parts_bin")
    assert TASK.score_over == "end" and M.SUCCESS_HOLD_S == 1.0
    assert dict(M.DASH.joint_pos)["hinge"] == pytest.approx(STOP - math.radians(0.5))
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos)  # plain floats for Isaac Lab


def test_the_assets_match_the_geometry():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(HERE / "assets" / "dash.usda"))
    hinge = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/dash/Physics/hinge"))
    assert hinge.GetUpperLimitAttr().Get() == pytest.approx(math.degrees(LOW))
    for side in ("left", "right"):
        pin = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(f"/dash/Physics/pin_{side}"))
        assert pin.GetUpperLimitAttr().Get() == pytest.approx(g.PIN_TRAVEL)
    assert stage.GetPrimAtPath("/dash/Physics/root_joint")                 # fixed to the table
    cache = UsdGeom.XformCache()

    def centre(path):
        m = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        return np.array(m.ExtractTranslation())

    origin = np.array([0.0, 0.0, 0.0])
    for name, c, _, _ in g.dash_boxes():
        assert centre(f"/dash/Geometry/body/{name}") == pytest.approx(origin + c, abs=1e-6), name
    hinge_o = np.array([0.0, 0.0, g.HINGE_Z])
    for name, c, _, _ in g.bin_boxes():
        assert centre(f"/dash/Geometry/body/bin/{name}") == pytest.approx(hinge_o + c, abs=1e-6), name
    for sign, side in ((1, "left"), (-1, "right")):
        for name, c, _, _ in g.pin_boxes(sign):
            assert centre(f"/dash/Geometry/body/bin/pin_{side}/{name}") == pytest.approx(hinge_o + c, abs=1e-6)
    for which in ("old", "new"):
        st = Usd.Stage.Open(str(HERE / "assets" / f"filter_{which}.usda"))
        root = st.GetDefaultPrim()
        assert UsdPhysics.MassAPI(root).GetMassAttr().Get() == pytest.approx(g.F_MASS)


def test_full_size_parts():
    assert g.F_SIZE == (0.200, 0.215, 0.030) and g.F_MASS == pytest.approx(0.15)
    assert 2 * g.HALF_W == pytest.approx(0.36)                              # a 36 cm glovebox ...
    assert g.PANEL[3] - g.PANEL[2] > 0.2 and g.BACK_TOP[0] - g.PANEL[0] > 0.15   # ... 21 cm tall, 16 cm deep
    assert 0.6 <= g.BIN_MASS <= 1.2


def test_the_bin_rests_on_its_stops():
    """Open, the bin's weight is in front of the hinge, and both bumps' square faces meet the fascia's rear at the
    stop angle, about 48 degrees open."""
    assert 40 < math.degrees(STOP) < 55
    com = g.bin_rot(STOP) @ np.array(g.BIN_COM)
    assert com[0] < -0.03                                                   # it tips open onto the stops
    fascia = [b for b in g.dash_world() if b[0] == "fascia_left"]
    assert g.gap(g.bump_parts(STOP + math.radians(0.5)), fascia) < -0.0005


def test_squeezing_both_panels_lets_the_bin_past_the_stops():
    """Each panel must go in by 8 mm (about 10 N against its spring, 15 N at its 14 mm travel) for its bump to pass the
    fascia; squeezed, the bin swings through to its lowered stop with every part clear of the dash."""
    need = max(M.NEED)
    assert 0.006 <= need <= 0.009
    assert 9.0 <= g.PIN_RATE * (need - g.PIN_TARGET) <= 15.0
    assert g.PIN_RATE * (g.PIN_TRAVEL - g.PIN_TARGET) <= 16.0
    dash = g.dash_world()
    for deg in np.arange(math.degrees(STOP) - 2, math.degrees(LOW) + 0.1, 1.0):
        assert g.gap(g.bin_world(math.radians(deg), (0.0105, 0.0105)), dash) > 0.002, deg


def test_one_panel_alone_does_not_let_it_past():
    fascia = [b for b in g.dash_world() if b[0] == "fascia_right"]
    for deg in (52, 60):
        right = [b for b in g.bin_world(math.radians(deg), (0.014, 0.0)) if b[0].endswith("right")]
        assert g.gap(right, fascia) < -0.003


def test_the_bin_swings_clear_of_the_dash():
    dash = g.dash_world()
    for deg in list(range(0, 47, 4)) + list(range(74, 111, 4)):
        assert g.gap(g.bin_world(math.radians(deg)), dash) > 0.0025, deg


def test_pushed_back_up_the_ramps_squeeze_the_panels():
    """From below, the fascia's edge meets each bump's 20-degree chamfer, not its square face, so pushing the bin up
    pushes the panels in until the bumps snap back behind the fascia."""
    assert g.RAMP_L == pytest.approx(g.RAMP_RISE / math.tan(math.radians(20)))
    first = next(t for t in reversed(M.NEED_GRID) if M.needed(t) > 0)
    assert 0 < M.needed(first) < 0.002                                       # the travel needed grows gradually
    grid = [M.needed(t) for t in M.NEED_GRID if t > STOP + math.radians(16)]
    assert all(a >= b - 1e-4 for a, b in zip(grid, grid[1:]))


def test_the_slot_is_keyed_and_has_play():
    """The filter fits with its rib up in the key channel and its tab toward the cabin, with 3.75 mm play each side
    and 6 mm above; upside down, or end for end, the rib meets the slot's face."""
    dash = g.dash_world()
    seat = SEAT
    assert g.gap(g.filter_world(seat + [0, 0, 0.0005]), [b for b in dash if not b[0].startswith("face_")]) > 0.0004
    assert g.SLOT_HALF_W - g.F_SIZE[1] / 2 >= 0.0025
    inside = seat + [0.05, 0, 0]
    for rot in (rx(math.pi), rz(math.pi), rz(math.pi) @ rx(math.pi)):
        turned = seat + [0.05, 0, 0] if rot is not None else inside
        assert g.gap(g.filter_world(turned, rot), dash) < -0.002


def test_a_seated_filter_rests_on_the_slot_floor_with_its_tab_on_the_face():
    tab = [b for b in g.filter_world(SEAT) if b[0] == "tab"]
    face = [b for b in g.dash_world() if b[0].startswith("face_high")]
    assert g.gap(tab, face) == pytest.approx(0.0, abs=1e-3)
    assert g.gap(g.filter_world(SEAT), [b for b in g.dash_world() if b[0] == "tunnel_floor"]) == pytest.approx(0, abs=1e-6)


def test_the_new_filter_lies_on_its_stand_and_the_old_is_seated():
    assert g.NEW_START[2] - g.F_SIZE[2] / 2 == pytest.approx(g.STAND_TOP)
    assert M.seated(g.OLD_START - [0, 0, 0.0005], EYE)
    lo, hi = g.STAND[0] - g.STAND[1][:2] / 2, g.STAND[0] + g.STAND[1][:2] / 2
    frame = [b for b in g.filter_world(g.NEW_START) if b[0] == "frame"]
    pts = np.vstack([g.box_points(c, r, h, 2) for _, c, r, h in frame])
    assert np.all(pts[:, :2] > lo) and np.all(pts[:, :2] < hi)
    tab = [b for b in g.filter_world(g.NEW_START) if b[0] == "tab"][0]
    assert tab[1][0] - tab[3][0] < lo[0]                              # the tab overhangs the stand's front edge


def test_the_slot_mouth_has_lead_ins_clear_of_a_seated_tab():
    """A 45-degree sill below and hood above the slot's mouth (the hood only over the outer 4 cm of each side, clear of the tab and the fingers) guide a filter that comes
    in up to 1.6 cm low or 1 cm high; neither touches the seated filter."""
    parts = [b for b in g.dash_world() if b[0] in ("sill", "hood_l", "hood_r")]
    assert len(parts) == 3
    assert g.gap(g.filter_world(SEAT), parts) >= 0.0


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_0():
    step()
    assert M.score(None) == 0.0 and not M.success(None)


def test_partial_credit_levels():
    step(theta=math.radians(65), pins=(0.012, 0.012))
    assert M.score(None) == 0.0                                       # not yet past the stops
    lowered()
    assert M.score(None) == 0.25
    step(theta=LOW, old=(OLD_DOWN, EYE))
    assert M.score(None) == 0.5
    step(theta=LOW, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    assert M.score(None) == 0.75
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    assert M.success(None)


def test_lowering_counts_even_if_it_is_raised_again_but_the_filters_are_judged_at_the_end():
    lowered()
    step(theta=STOP)
    assert M.score(None) == 0.25
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT + [0.01, 0, 0], EYE))   # 1 cm proud: not fully in
    assert M.score(None) == 0.5


def test_full_score_needs_the_bin_up_released_and_still():
    lowered()
    step(theta=math.radians(80), old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    assert not M.success(None)                                       # not back up past its stops
    step(theta=STOP, pins=(0.006, 0.0), old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    assert not M.success(None)                                       # a panel still pushed in
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE), speed=0.05)
    assert not M.success(None)                                       # moving
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE), arms=holding(SEAT))
    assert M.score(None) == 0.5 and not M.success(None)              # the new one still held
    arms = far_arms()
    near = M.SEAT_WORLD + [-0.2, 0, 0.0]
    arms["left"] = dict(arms["left"], tcp=near, rot=demo.grip_rot(-80.0), opening=0.094)
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE), arms=arms)
    assert not M.success(None)                                       # a gripper within 2 cm of the filter
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    assert M.success(None)


@pytest.mark.parametrize("off, ok", [([0.0025, 0, 0], True), ([0.004, 0, 0], False), ([0, 0.005, 0], False),
                                     ([0, 0, 0.005], False)])
def test_seating_depth_and_alignment(off, ok):
    lowered()
    step(theta=LOW, old=(OLD_DOWN, EYE), new=(SEAT + off, EYE))
    assert (M.score(None) == 0.75) is ok


def test_seating_tilt():
    lowered()
    step(theta=LOW, old=(OLD_DOWN, EYE), new=(SEAT, rz(math.radians(4))))
    assert M.score(None) == 0.5


def test_only_the_new_filter_counts_and_the_old_must_be_on_the_table():
    lowered()
    step(theta=LOW, old=(SEAT, EYE), new=(OLD_DOWN, EYE))            # swapped: the old one put back
    assert M.score(None) == 0.25
    held = OLD_DOWN + [0, 0, 0.1]
    step(theta=LOW, old=(held, EYE), arms={"left": far_arms()["left"],
                                          "right": dict(far_arms()["right"], tcp=M.tab_point(held, EYE), opening=0.016)})
    assert M.score(None) == 0.25                                     # held, not set down
    on_bin = g.bin_point((0.0, 0.0, 0.0), LOW) + [-0.12, 0.0, 0.08]
    step(theta=LOW, old=(on_bin, EYE))
    assert M.score(None) == 0.25                                     # not on the table
    tipped = OLD_DOWN + [0, 0, 0.05]
    step(theta=LOW, old=(tipped, rx(math.radians(80))))
    assert M.score(None) == 0.5                                      # anywhere on the table, any way up


def test_let_go_means_every_gripper_part_2_cm_from_the_filter():
    """A gripper clamping the old filter's frame, or resting its open fingers 1 cm from it, still holds it: only every
    part of both grippers 2 cm or more away counts as let go."""
    lowered()
    arms = far_arms()
    frame_grip = OLD_DOWN + np.array([0.05, 0.0, 0.0])                # TCP on the frame, away from the tab
    arms["left"] = dict(arms["left"], tcp=frame_grip, rot=demo.grip_rot(-90.0, 0.0, 90.0), opening=0.03)
    step(theta=LOW, old=(OLD_DOWN, EYE), arms=arms)
    assert M._STATE["held"]["old"] and M.score(None) == 0.25
    tcp, rot = arms["left"]["tcp"], arms["left"]["rot"]
    for lift in (0.01, 0.15):
        a = dict(arms["left"], tcp=tcp + [0, 0, lift + 0.03], opening=0.094)
        step(theta=LOW, old=(OLD_DOWN, EYE), arms={"left": a, "right": far_arms()["right"]})
        gap = M.gap_to(M.gripper_boxes(a["tcp"], a["rot"], a["opening"]), [b[1:] for b in g.filter_world(OLD_DOWN)])
        assert M._STATE["held"]["old"] == (gap < M.LET_GO), (lift, gap)
    assert M.score(None) == 0.5


def test_forcing_the_bin_past_a_stop_is_a_violation():
    step(theta=math.radians(56), pins=(0.0, 0.012))                  # left panel not pushed in: bump through the fascia
    assert M.violation() and "forced" in M._STATE["why"][0]
    step(theta=LOW, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    assert M.score(None) == 0.0 and not M.success(None)


def test_squeezed_or_cammed_through_is_not_a_violation():
    for deg in np.arange(math.degrees(STOP), math.degrees(LOW), 1.0):
        th = math.radians(deg)
        step(theta=th, pins=(M.needed(th), M.needed(th)))
    assert not M.violation()


@pytest.mark.parametrize("kwargs, why", [
    (dict(theta=LOW + math.radians(4)), "hinge"),
    (dict(new=(np.array([0.3, -0.7, g.TABLE - 0.3]), EYE)), "fell off the table"),
])
def test_violations_latch_for_the_whole_trial(kwargs, why):
    lowered()
    step(**kwargs)
    assert M.violation() and any(why in w for w in M._STATE["why"])
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    step(theta=STOP, old=(OLD_DOWN, EYE), new=(SEAT, EYE))
    assert M.score(None) == 0.0 and not M.success(None)


def test_dropping_a_filter_onto_the_table_is_not_a_violation():
    lowered()
    step(theta=LOW, old=(OLD_DOWN + [0, 0, 0.1], EYE), speed=1.5)
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
    assert len(PLAN) + 3 * 20 < 2400 and len(PLAN) < 1400


def replay(every: int = 1):
    """Walk the plan, carrying each filter with the gripper that holds it; yield (index, step, filter poses)."""
    pose = {"old": (g.OLD_START.copy(), EYE), "new": (g.NEW_START.copy(), EYE)}
    rel, prev = {}, {"left": None, "right": None}
    for i, s in enumerate(PLAN):
        for side in ("left", "right"):
            held = s["holding"][side]
            _, _, pos, rot = s[side]
            if held and prev[side] != held:
                p, r = pose[held]
                rel[side] = (rot.T @ (p - pos), rot.T @ r)
            if held:
                rp, rr = rel[side]
                pose[held] = (pos + rot @ rp, rot @ rr)
            prev[side] = held
        if i % every == 0 or i == len(PLAN) - 1:
            yield i, s, pose


def link_points(q, side):
    base = np.asarray(station.BASES[side])
    links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})
    return [links[n][:3, 3] + base for n in CHAIN]


def capsule_gap(points, boxes, r=0.035):
    best = math.inf
    for a, b in zip(points[:-2], points[1:-1]):          # links 2-5 (the gripper is checked as boxes)
        seg = a + np.outer(np.linspace(0, 1, 10), b - a)
        for _, c, rr, h in boxes:
            best = min(best, float(g.point_box_dist(seg, c, rr, h).min()) - r)
    return best


def test_along_the_demo_nothing_runs_into_anything_it_should_not():
    """Arms clear of the dash and the bin, filters clear of everything, except where the demo means to touch: the
    fingers on the thumb pads, on the filters' tabs, on the bin's back wall when raising it, and a filter's tab on the
    housing face when it is pushed home."""
    for i, s, pose in replay(4):
        theta = s["theta"]
        bw = g.bin_world(theta, s["pins"])
        dash = g.dash_world()
        for side in ("left", "right"):
            q, grip, pos, rot = s[side]
            boxes = demo.gripper_boxes(pos, rot, 0.094 * grip)
            fingers = demo.finger_cloud(pos, rot, 0.094 * grip)       # the fingers' real collision meshes
            assert capsule_gap(link_points(q, side), dash + bw) > 0.004, (i, side)
            for name, *d in dash + bw:
                if name.startswith("pad") or (name == "back" and s.get("raising")):
                    continue
                assert float(g.point_box_dist(fingers, *d).min()) > 0.002, (i, side, name)
                assert g.boxes_gap(boxes[2], d) > 0.002, (i, side, name)     # the housing
            for k, (p, r) in pose.items():
                if s["holding"][side] == k or np.linalg.norm(pos - M.tab_point(p, r)) < 0.08:
                    continue                                          # holding it, or taking or letting go of its tab
                assert M.gap_to(boxes, [b[1:] for b in g.filter_world(p, r)]) > 0.004, (i, side, k)
        for k, (p, r) in pose.items():
            parts = [b for b in dash if not (k == "new" and b[0].startswith("face_") and i > len(PLAN) // 2)
                     and not (k == "old" and i < 400)]
            assert g.gap(g.filter_world(p, r), parts + bw) > 0.0004, (i, k)


def test_the_demo_presses_the_pads_and_raises_the_bin_by_its_back_wall():
    pressed = [s for s in PLAN if min(s["pins"]) > 0.0105]
    assert len(pressed) > 60
    raising = [s for s in PLAN if s.get("raising")]
    assert raising and min(s["theta"] for s in raising) < STOP - math.radians(2)


def test_the_demo_ends_in_a_full_score_state():
    *_, (i, s, pose) = replay()
    assert np.linalg.norm(pose["new"][0] - SEAT) < 0.003
    arms = {}
    for side in ("left", "right"):
        q = s[side][0]
        tcp, rot = ik.tcp_pose(q, station.BASES[side])
        arms[side] = {"tcp": tcp, "rot": rot, "opening": 0.094 * s[side][1]}
    old = (np.array([*pose["old"][0][:2], g.TABLE + g.F_SIZE[2] / 2]), EYE)   # it settles flat where it was let go
    lowered()
    step(theta=STOP, old=old, new=(SEAT, EYE), arms=arms)
    step(theta=STOP, old=old, new=(SEAT, EYE), arms=arms)
    assert M.score(None) == 0.75 and M.success(None)
    assert M.out_of_dash(*pose["old"])
