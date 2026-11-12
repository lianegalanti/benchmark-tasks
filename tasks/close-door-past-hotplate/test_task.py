"""Offline checks (no Isaac Sim): the layout, the hot-zone rule, the scoring, that the obvious reach crosses the hot
zone, and that the demo is reachable, pushes the door shut and keeps every part of both arms out of the zone.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan, grip_rotation
from yam_arena.task_file import load
from yam_arena.urdf import forward_kinematics

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import geometry as g  # noqa: E402  (task.py put this folder on the path)
from scenekit import BodyState  # noqa: E402

DT = 1 / 20
RUN_LIMIT_S = 2400 * DT


def capsules(q, side):
    """One arm's capsules from the URDF's forward kinematics, as the monitor builds them from the simulator's links."""
    base = np.asarray(station.BASES[side])
    links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})
    tcp, rot = ik.tcp_pose(q, base)
    return M.arm_capsules([links[n][:3, 3] + base for n in M.CHAIN], tcp, rot[:, 2])


AT_READY = {side: capsules(np.array(READY), side) for side in ("left", "right")}


def door(angle):
    z = np.zeros(3)
    return {"cabinet": BodyState(pos=z, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=z, ang_vel=z, low=z, high=z,
                                 joints={"hinge": float(angle)})}


def play(angles, caps=None):
    """Run the monitor over door angles (deg), from a reset; return the score after each step."""
    M.reset_state()
    out = []
    for i, a in enumerate(angles):
        c = AT_READY["left"] + AT_READY["right"] if caps is None or caps[i] is None else caps[i]
        M.update(math.radians(a), c)
        out.append(M.score(door(math.radians(a))))
    return out


def reach(side, pos, yaw, q=None):
    q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(grip_rotation(yaw)), q if q is not None else READY,
                            base_pos=station.BASES[side], iters=300)
    return q, err, rerr


# -- layout ----------------------------------------------------------------------------------------------------------


def test_it_loads_with_the_cabinet_articulated_and_nothing_loose():
    assert TASK.names == ("cabinet", "hot_plate", "hot_plate_burner", "hot_plate_lamp")
    assert TASK.free_objects == () and TASK.score_over == "end"
    assert dict(M.CABINET.joint_pos) == {"hinge": pytest.approx(math.radians(60))}
    assert not M.CABINET.self_collisions
    assert all(isinstance(v, float) for v in M.CABINET.pos)  # plain floats for Isaac Lab


def test_the_asset_matches_the_geometry():
    pytest.importorskip("pxr")
    import pxr.Usd
    import pxr.UsdGeom
    import pxr.UsdPhysics
    stage = pxr.Usd.Stage.Open(str(HERE / "assets" / "cabinet.usda"))
    hinge = pxr.UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/cabinet/Physics/hinge"))
    assert hinge.GetAxisAttr().Get() == "Z"
    assert hinge.GetLowerLimitAttr().Get() == pytest.approx(0.0)
    assert hinge.GetUpperLimitAttr().Get() == pytest.approx(90.0)
    panel = stage.GetPrimAtPath("/cabinet/Geometry/carcass/door/panel")
    rng = pxr.UsdGeom.Imageable(panel).ComputeWorldBound(0, "default").ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    assert lo == pytest.approx([-g.DOOR_T / 2, 0.0, g.DOOR_Z[0]], abs=1e-6)
    assert hi == pytest.approx([g.DOOR_T / 2, g.DOOR_W, g.DOOR_Z[1]], abs=1e-6)
    root = stage.GetPrimAtPath("/cabinet/Geometry/carcass")
    assert root.HasAPI(pxr.UsdPhysics.ArticulationRootAPI)
    assert root.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is False


def test_the_door_swings_clear_of_the_hot_plate_and_stands_just_behind_it():
    plate = g.plate_corners()
    gaps = [g.polygon_gap(g.door_corners(math.radians(a)), plate) for a in np.arange(0.0, 90.01, 0.5)]
    assert min(gaps) >= 0.025  # even opened to its 90-degree stop, the door misses the plate
    at_start = g.polygon_gap(g.door_corners(g.OPEN), plate)
    assert 0.03 <= at_start <= 0.045  # the plate stands 4 cm in front of the door's hinge half
    # the plate is between the right arm and the door; the free edge sticks out past the plate's left side
    assert g.door_point(g.OPEN, g.DOOR_W)[1] - g.PLATE_HI[1] > 0.12
    assert g.PLATE_LO[0] > 0.1 and g.PLATE_HI[0] < g.FRONT_X
    assert g.PLATE_HI - g.PLATE_LO == pytest.approx([0.15, 0.15])
    # the hot zone has no top: the whole column over the plate's footprint
    assert g.ZONE_HI[2] == math.inf and g.ZONE_LO[2] == g.TOP


TOP_OF_DOOR = g.TOP + g.DOOR_Z[1]


def test_the_tape_measure_thresholds():
    assert g.free_edge_gap(g.OPEN) == pytest.approx(0.187, abs=0.001)
    assert g.free_edge_gap(g.CLOSED) == pytest.approx(0.015, abs=0.001)
    assert g.free_edge_gap(g.HALF) == pytest.approx(0.109, abs=0.001)
    assert g.free_edge_gap(g.MOVED) == pytest.approx(0.166, abs=0.001)


# -- the hot-zone rule -----------------------------------------------------------------------------------------------


def test_both_arms_start_clear_of_the_hot_zone():
    """Neither arm's ready pose is over the plate's footprint at any height: the right arm's capsules stand 3.5 cm to
    the right of it (its real parts a little more), so a 1-2 cm placement error on a real table can't score a perfect
    trial 0 at step 0."""
    assert M.zone_gap(AT_READY["left"]) >= 0.3
    assert M.zone_gap(AT_READY["right"]) >= 0.034


def test_the_pre_trial_check_refuses_a_misplaced_plate():
    """The check run after every reset (and on the logged ready pose before a real trial): it passes with the plate
    where it belongs, and refuses a trial if the plate stands 2 cm or more too far toward the right arm."""
    both = AT_READY["left"] + AT_READY["right"]
    assert M.check_arms_clear(both) == pytest.approx(0.035, abs=0.001)
    lo, hi = g.ZONE_LO.copy(), g.ZONE_HI.copy()
    try:
        for shift, ok in ((0.01, True), (0.02, False), (0.04, False)):
            g.ZONE_LO[1], g.ZONE_HI[1] = lo[1] - shift, hi[1] - shift
            if ok:
                M.check_arms_clear(both)
            else:
                with pytest.raises(RuntimeError):
                    M.check_arms_clear(both)
    finally:
        g.ZONE_LO[:], g.ZONE_HI[:] = lo, hi


def test_any_part_over_the_plate_is_an_intrusion_however_high():
    """The reviewer's case: crossing above the plate more than 15 cm over its top. There is no safe height."""
    c = np.array([*((g.PLATE_LO + g.PLATE_HI) / 2), 0.0])
    for above in (0.0, 0.15, 0.16, 0.25, 0.40, 1.0):                  # m above the plate's top
        z = g.PLATE_TOP + above + M.GRIP_R + 0.005                     # the capsule's lowest point is this high
        assert M.zone_gap([(c + [0, 0, z + 0.13], c + [0, 0, z], M.GRIP_R)]) < 0, above
        assert M.zone_gap([(c + [0, 0.3, z], c + [0, -0.3, z], M.ARM_R)]) < 0, above   # a link passing across it
    # the reach-over: the right arm's tool point carried across the plate 25 cm above its top scores 0
    high = [(np.array([0.285, -0.30, g.PLATE_TOP + 0.25]), np.array([0.285, 0.05, g.PLATE_TOP + 0.25]), M.GRIP_R)]
    assert play([60, 40, 29, 10, 3, 0], [None, None, high, None, None, None]) == [0.0, 0.25, 0.0, 0.0, 0.0, 0.0]
    assert M.violation() and not M.success(door(0.0))
    M.reset_state()


def test_the_zone_rule_is_the_footprint_edge_at_every_height():
    c = np.array([*((g.PLATE_LO + g.PLATE_HI) / 2), 0.0])
    for z in (g.TOP + 0.08, g.PLATE_TOP + 0.30, g.PLATE_TOP + 0.60):   # low, high and very high
        beside = np.array([g.PLATE_HI[0] + M.GRIP_R + 0.005, c[1], z])  # 0.5 cm beyond the far edge
        assert M.zone_gap([(beside + [0, 0, 0.13], beside, M.GRIP_R)]) > 0, z
        beside[0] -= 0.01                                                # 0.5 cm over it
        assert M.zone_gap([(beside + [0, 0, 0.13], beside, M.GRIP_R)]) < 0, z
        left = np.array([c[0], g.PLATE_HI[1] + M.ARM_R + 0.005, z])      # beside the left edge, then over it
        assert M.zone_gap([(left, left + [0, 0, 0.1], M.ARM_R)]) > 0, z
        left[1] -= 0.01
        assert M.zone_gap([(left, left + [0, 0, 0.1], M.ARM_R)]) < 0, z


def test_the_obvious_reach_crosses_the_hot_zone():
    """Pushing the door anywhere on its hinge half or a little beyond (up to 12 cm from the hinge), with either arm,
    puts the arm over the plate; and the right arm's straight-line reach from its ready pose to the door passes over
    it."""
    for side in ("left", "right"):
        for r in (0.06, 0.09, 0.11, 0.12):
            q, err, _ = reach(side, (*g.door_point(g.OPEN, r, M.PUSH_OFF), M.PUSH_Z), M._yaw(g.OPEN))
            assert err < 0.002, (side, r)
            assert M.zone_gap(capsules(q, side)) < 0, (side, r)
    start, _ = ik.tcp_pose(np.array(READY), station.BASES["right"])
    for r in (0.11, 0.18):
        goal = np.array([*g.door_point(g.OPEN, r, M.PUSH_OFF), M.PUSH_Z])
        line = start + np.outer(np.linspace(0, 1, 50), goal - start)
        inside = np.all((line >= g.ZONE_LO) & (line <= g.ZONE_HI), axis=1)
        assert inside.any(), r  # the TCP itself passes through the zone
    # with the right arm, only the outer 5 cm of the door can be pushed without it being over the plate
    q, _, _ = reach("right", (*g.door_point(g.OPEN, 0.15, M.PUSH_OFF), M.PUSH_Z), M._yaw(g.OPEN))
    assert M.zone_gap(capsules(q, "right")) < 0


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_success_needs_the_door_within_4_degrees_and_no_intrusion():
    M.reset_state()
    assert M.success(door(math.radians(3.9))) and M.success(door(0.0))
    assert not M.success(door(math.radians(4.5)))
    assert M.distance(door(0.0)) == pytest.approx(0.0, abs=1e-9)
    M._STATE["intrusion"] = True
    assert not M.success(door(0.0))
    M.reset_state()


def test_success_cannot_hold_in_the_starting_state():
    """The door starts 60 degrees open, so neither success() nor any partial credit holds at the start; and a reset
    that leaves the door anywhere else (as when the hinge's start position wasn't written and it sat closed) stops
    the trial instead of scoring it."""
    M.reset_state()
    assert not M.success(door(M.CABINET.joint_pos[0][1])) and dict(M.CABINET.joint_pos) == {"hinge": g.OPEN}
    assert play([60]) == [0.0]
    M.check_start(g.OPEN)
    M.check_start(g.OPEN - math.radians(1.5))
    for wrong in (0.0, g.CLOSED, g.HALF, -g.OPEN, g.MAX_OPEN):
        with pytest.raises(RuntimeError):
            M.check_start(wrong)
    with pytest.raises(KeyError):  # no silent default when the hinge isn't read
        M.success({"cabinet": door(g.OPEN)["cabinet"].__class__(**{**door(g.OPEN)["cabinet"].__dict__, "joints": {}})})


def test_partial_credit_is_the_best_so_far():
    assert play([60, 60, 61, 75]) == [0.0] * 4                     # untouched, or opened wider: nothing
    assert play([60, 55, 50.5]) == [0.0, 0.0, 0.0]                 # less than 10 degrees
    assert play([60, 49.5, 55]) == [0.0, 0.25, 0.25]               # 10 degrees, then swung back: keeps 0.25
    assert play([60, 40, 29.5, 45]) == [0.0, 0.25, 0.5, 0.5]       # halfway
    assert play([60, 30, 10, 3]) == [0.0, 0.5, 0.5, 0.5]           # closed: success (1.0) comes from success()


def test_an_intrusion_scores_0_for_the_rest_of_the_trial():
    low = [(np.array([0.285, -0.135, 0.95]), np.array([0.285, -0.135, 0.85]), M.GRIP_R)]
    assert play([60, 29, 20, 5, 0], [None, None, low, None, None]) == [0.0, 0.5, 0.0, 0.0, 0.0]
    assert M.violation() and not M.success(door(0.0))
    M.reset_state()  # a new trial starts clean
    assert not M.violation() and M.score(door(0.0)) == 0.0


# -- the demo --------------------------------------------------------------------------------------------------------


def _plan():
    plan = M.demo()["left"]
    return plan, ArmPlan(plan, station.BASES["left"])


def test_the_demo_is_reachable_within_joint_limits():
    _, arm = _plan()
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert np.all(actions[:, 5] >= ik.LOWER[5] + math.radians(4))  # wrist joint6 keeps 4 degrees of its stop
    assert arm.duration < 20 < RUN_LIMIT_S


def test_along_the_demo_both_arms_stay_out_of_the_hot_zone():
    """Every capsule the monitor checks, from the URDF's forward kinematics on the demo's commanded joints, at every
    control step: the moving left arm at least 1.4 cm clear of the zone (the whole column over the plate), the right
    arm, at its ready pose, 3.4 cm."""
    _, arm = _plan()
    worst = math.inf
    for k in range(round(arm.duration / DT) + 5):
        arm.action7((k + 1) * DT)
        worst = min(worst, M.zone_gap(capsules(arm.q, "left")))
    assert worst >= 0.014, worst
    assert M.zone_gap(AT_READY["right"]) >= 0.034


def test_the_demo_pushes_the_door_all_the_way_shut_and_holds_it():
    plan, _ = _plan()
    pushes = [w for w in plan if w.label.startswith("push to") or w.label in ("onto the door", "hold closed")]
    angles = []
    for w in pushes:
        a = 0.0 if w.label == "hold closed" else g.OPEN if w.label == "onto the door" else math.radians(
            float(w.label.split()[2]))
        angles.append(a)
        assert np.dot(np.asarray(w.pos[:2]) - g.HINGE, g.u(a)) == pytest.approx(M.PUSH_R, abs=1e-9)
        # the TCP sits PUSH_OFF out from where the door's face is at that angle: the fingers (34 mm half-width)
        # reach 8 mm past the face, so the door leads them by about 2 degrees and is pushed shut
        assert np.dot(np.asarray(w.pos[:2]) - g.HINGE, g.n(a)) - g.DOOR_T / 2 == pytest.approx(M.PUSH_OFF, abs=2e-3)
        assert 0.004 <= M.GRIP_HALF - M.PUSH_OFF <= 0.012
        assert w.pos[2] - PAD_TIP_PAST_TCP > g.TOP + g.DOOR_Z[0] + 0.03 and w.pos[2] + 0.09 < TOP_OF_DOOR
    assert angles[0] == g.OPEN and angles[-1] == 0.0
    assert all(0 <= a - b <= math.radians(5) + 1e-9 for a, b in zip(angles, angles[1:]))  # 5-degree steps
    hold = next(w for w in plan if w.label == "hold closed")
    assert hold.duration >= M.SUCCESS_HOLD_S + 0.5
    # coming down 3 cm beyond the free edge, and moving along the door to the push point, the fingers stay 1.2 cm
    # off the plane of its face even turned 10 degrees off square
    down = next(w for w in plan if w.label == "along the door")
    off = np.dot(np.asarray(down.pos[:2]) - g.HINGE, g.n(g.OPEN)) - g.DOOR_T / 2
    extent = M.GRIP_HALF * math.cos(g.OPEN - M.MAX_YAW) + 0.051 * math.sin(g.OPEN - M.MAX_YAW)
    assert off - extent >= 0.0115
    first = next(w for w in plan if w.label == "down")
    rel = np.asarray(first.pos[:2]) - g.HINGE
    assert np.dot(rel, g.u(g.OPEN)) >= g.DOOR_W + 0.03
    assert np.dot(rel, g.n(g.OPEN)) - g.DOOR_T / 2 - extent >= 0.0115


def test_the_demo_closes_the_door_well_within_the_run_limit():
    plan, _ = _plan()
    t, closed_at = 0.0, None
    for w in plan:
        t += w.duration
        if w.label == "push to 0 deg":
            closed_at = t
    assert closed_at is not None and closed_at + M.SUCCESS_HOLD_S < 15.0
