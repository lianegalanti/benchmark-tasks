"""Offline checks (no Isaac Sim): the frame asset, the hanger and pin geometry, how the frame hangs from each notch,
the scoring, and that the demo and its correction pass are reachable.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

import geometry as g
import make_assets
from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load
from yam_arena.urdf import forward_kinematics

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
DT = 1 / 20


def quat(rot):
    return ik.matrix_to_quat_xyzw(rot)


def level_hang(notch=g.LEVEL_NOTCH):
    return g.hang_centre(notch), np.eye(3)


def hanging_at(notch, tilt):
    """The frame hanging from ``notch`` at roll ``tilt`` (rails against the board)."""
    rot = g.rot_x(tilt)
    pin = np.array([g.BOARD_X - 0.008, g.PIN_Y, g.PIN_Z])
    local_pin = np.array([g.D / 2 - 0.008, notch * g.PITCH, g.rest_z_local()])
    return pin - rot @ local_pin, rot


def corner_tilt(notch):
    """The roll at which the frame hanging from ``notch`` comes to rest on its lowest corner, or its equilibrium."""
    eq = g.equilibrium_tilt(notch)
    if g.lowest_z_hanging(notch, eq) >= g.TABLE:
        return eq
    ts = np.linspace(0.0, eq, 2000)
    return float(next(t for t in ts if g.lowest_z_hanging(notch, t) < g.TABLE))


FAR = [(np.array([-0.5, 0.0, 1.5]), np.array([-0.5, 0.1, 1.5]), 0.03)]   # arms well away
STILL = (np.zeros(3), np.zeros(3))


def step(pos, rot, vel=STILL, arms=FAR):
    M.MONITOR.update(pos, quat(rot), vel[0], vel[1], arms)
    return M.MONITOR.score(), M.MONITOR.success()


# -- the task file and its asset ------------------------------------------------------------------------------------
def test_it_loads_with_only_the_frame_free_and_scores_at_the_end():
    assert TASK.free_objects == ("frame",)
    assert {"board", "pin", "stand_base", "stand_front", "stand_back"} <= set(TASK.names)
    assert TASK.score_over == "end"


def test_poses_are_plain_floats():
    """Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects at reset."""
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name


def test_the_frame_asset_is_up_to_date_and_its_mass_matches():
    assert (HERE / "assets" / "frame.usda").read_text() == make_assets.frame_usda()
    moments, axes = make_assets.principal()
    inertia, com = g.inertia()
    assert np.allclose(axes @ np.diag(moments) @ axes.T, inertia, atol=1e-12)
    assert np.isclose(np.linalg.det(axes), 1.0)
    assert M.FRAME.mass == pytest.approx(0.5) and com[1] == pytest.approx(2 * g.PITCH)
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(HERE / "assets" / "frame.usda"))
    root = stage.GetDefaultPrim()
    mass = UsdPhysics.MassAPI(root)
    assert mass.GetMassAttr().Get() == pytest.approx(g.MASS)
    assert np.allclose(np.array(mass.GetCenterOfMassAttr().Get()), g.COM, atol=1e-6)
    shapes = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(shapes) == 6 + 12


# -- hanger and pin -------------------------------------------------------------------------------------------------
def test_teeth_form_90_degree_notches_that_seat_the_pin():
    tips = [c[1] for c in g.teeth()]
    assert np.allclose(np.diff(tips), g.PITCH)
    notches = (np.array(tips[:-1]) + np.array(tips[1:])) / 2
    assert np.allclose(notches, g.NOTCHES * g.PITCH)
    assert g.TIP_Z == pytest.approx(g.APEX_Z - g.PITCH / 2)
    # the seated pin's top is inside the notch, its bottom below the teeth's tips: it can't slide sideways
    assert g.APEX_Z - g.PIN_REST + g.PIN_R < g.APEX_Z
    assert g.APEX_Z - g.PIN_REST - g.PIN_R < g.TIP_Z
    # the bar ends where the outer teeth end, under the top rail, inside the recess
    assert g.HANGER_HALF_W == pytest.approx(tips[-1] + g.PITCH / 2)
    assert g.HANGER_X[1] < g.D / 2 and g.HANGER_HALF_W < g.W / 2 - g.RAIL


def test_the_pin_reaches_under_the_hanger_but_not_to_the_backing():
    for gap in (0.0, g.APPROACH_GAP):
        rails_back = g.BOARD_X - gap
        hanger_far = rails_back - (g.D / 2 - g.HANGER_X[1])        # the hanger's face toward the board
        backing = rails_back - (g.D / 2 - g.BACKING_X)
        tip = g.BOARD_X - g.PIN_OUT
        assert hanger_far - tip >= 0.005                           # 5 mm or more under the hanger
        assert tip - backing >= 0.002                              # its tip 2 mm or more off the backing


def test_the_frame_hangs_level_only_from_the_notch_above_its_centre_of_mass():
    assert g.LEVEL_NOTCH == 2 and g.equilibrium_tilt(2) == pytest.approx(0.0, abs=1e-9)
    for k in (1, 3):  # one tooth off: crooked, but within 10 degrees and clear of the table
        t = math.degrees(abs(g.equilibrium_tilt(k)))
        assert 6.0 < t < 8.5
        assert g.lowest_z_hanging(k, g.equilibrium_tilt(k)) > g.TABLE + 0.005
    for k in (0, 4):  # two teeth off: more than 10 degrees, and its low corner reaches the table first
        assert math.degrees(abs(g.equilibrium_tilt(k))) > 12
        assert g.lowest_z_hanging(k, g.equilibrium_tilt(k)) < g.TABLE
        assert math.degrees(abs(corner_tilt(k))) > M.CROOKED_DEG
    assert g.lowest_z_hanging(2, 0.0) - g.TABLE == pytest.approx(0.031, abs=0.001)


def test_friction_cannot_hold_the_frame_level_one_tooth_off():
    """Rough bound on how far from its equilibrium friction can hold the frame (rad): the rails' bottom against the
    board (friction 0.15) and the pin turning in its notch (0.3), against gravity's restoring torque."""
    h = g.rest_z_local() - g.COM[2]
    lever = g.D / 2 - 0.008 - g.COM[0]          # centre of mass in front of the pin's contact
    drop = g.rest_z_local() + g.H / 2           # pin to the bottom rail
    board = 0.1 * (lever / drop) * drop / h
    pin = 0.3 * g.PIN_R / math.cos(math.radians(45)) / h
    hold = math.degrees(board + pin)
    assert hold < 2.0  # so the frame can't stay more than 2 degrees off level either
    assert math.degrees(abs(g.equilibrium_tilt(1))) - hold > M.LEVEL_DEG + 1.0


def test_on_pin_needs_the_pin_in_a_notch():
    pos, rot = level_hang()
    assert g.on_pin(pos, rot) and g.back_gap(pos, rot) == pytest.approx(0.0)
    assert not g.on_pin(pos + [0, 0, 0.01], rot)                   # held 1 cm above it
    assert not g.on_pin(pos - [0.02, 0, 0], rot)                   # in front of it
    assert not g.on_pin(g.hang_centre(9), rot)                     # beyond the end of the bar
    for k in (1, 3):
        assert g.on_pin(*hanging_at(k, g.equilibrium_tilt(k)))


def test_the_back_fingers_pass_beside_the_board():
    for notch in (0, 1, 2):  # the notches the demo and its correction use
        c = g.hang_centre(notch)
        for side in ("left", "right"):
            y = M.grip_point(c, side)[1]
            board_edge = g.PIN_Y + math.copysign(g.BOARD_W / 2, y)
            assert abs(y - board_edge) - g.FINGER_HALF_W >= 0.015, (notch, side)
    # open, each finger clears the 3.5 cm rail by 2.9 cm
    assert (0.094 - g.D) / 2 >= 0.015


def test_the_start_is_in_the_stand_and_scores_nothing():
    M.MONITOR.reset()
    pos, rot = g.START, np.eye(3)
    assert g.START[2] - g.H / 2 == pytest.approx(g.TABLE + g.STAND_BASE_T)
    assert (g.STAND_SLOT - g.D) / 2 == pytest.approx(0.003)
    assert step(pos, rot) == (0.0, False)
    assert g.back_gap(pos, rot) > M.LIFT_NEAR


# -- scoring --------------------------------------------------------------------------------------------------------
def test_hanging_level_by_itself_and_still_succeeds():
    M.MONITOR.reset()
    pos, rot = level_hang()
    assert step(pos, rot) == (0.5, True)
    tilted = g.rot_x(math.radians(1.8))
    assert step(pos, tilted)[1]
    assert not step(pos, rot, vel=(np.array([0, 0.02, 0]), np.zeros(3)))[1]      # still moving
    assert not step(pos, rot, vel=(np.zeros(3), np.array([0.1, 0, 0])))[1]      # still turning


def test_crooked_held_or_resting_on_the_table_gets_partial_credit():
    M.MONITOR.reset()
    assert step(*hanging_at(1, g.equilibrium_tilt(1))) == (0.5, False)
    assert step(*hanging_at(3, g.equilibrium_tilt(3))) == (0.5, False)
    assert step(*hanging_at(0, corner_tilt(0))) == (0.0, False)                   # on its corner: not hanging
    step(g.hang_centre(2) + [-0.02, 0, 0.03], np.eye(3))                          # ... after being lifted to the board
    assert step(*hanging_at(0, corner_tilt(0))) == (0.25, False)
    pos, rot = level_hang()
    grip = M.grip_point(pos, "left")
    held = [(grip + [0, 0, 0.08], grip + [0, 0, 0.013], M.GRIP_R)]
    assert step(pos, rot, arms=held) == (0.25, False)                             # level, but still held
    assert step(pos + [0, 0, 0.01], rot) == (0.25, False)                         # lifted to the board, not on the pin


def test_lifted_is_kept_and_a_drop_scores_zero_for_good():
    M.MONITOR.reset()
    assert step(g.START + [0, 0, 0.05], np.eye(3))[0] == 0.0                      # lifted, but not to the board
    near = g.hang_centre(2) + [-0.02, 0, 0.03]
    assert step(near, np.eye(3))[0] == 0.25
    assert step(g.START, np.eye(3))[0] == 0.25                                    # put back in the stand: kept
    assert step(near, np.eye(3), vel=(np.array([0, 0, -0.8]), np.zeros(3)))[0] == 0.0
    assert step(*level_hang()) == (0.0, False)                                    # dropped: 0 for the rest


def test_score_and_success_read_the_monitor_and_reset_with_the_episode():
    M.MONITOR.reset()
    step(*level_hang())
    assert M.success({}) and M.score({}) == 0.5
    M._reset()
    assert not M.success({}) and M.score({}) == 0.0


# -- the demo -------------------------------------------------------------------------------------------------------
def _play(plans):
    arms = {s: ArmPlan(p, station.BASES[s]) for s, p in plans.items()}
    n = round(max(a.duration for a in arms.values()) / DT) + 5
    acts = {s: np.array([a.action7((k + 1) * DT)[0] for k in range(n)]) for s, a in arms.items()}
    return arms, acts


@pytest.mark.parametrize("notch", [g.LEVEL_NOTCH, 0])
def test_the_demo_is_reachable_and_both_arms_move_together(notch):
    plans = M.demo(notch)
    assert [w.duration for w in plans["left"]] == [w.duration for w in plans["right"]]
    arms, acts = _play(plans)
    for side, arm in arms.items():
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(acts[side][:, :6] >= ik.LOWER) and np.all(acts[side][:, :6] <= ik.UPPER)
    assert arms["left"].duration < 25  # plus settling, well inside the 120 s run limit
    # from the grip to letting go, the two grip points stay 24 cm apart, level
    t0 = sum(w.duration for w in plans["left"][:2])
    t1 = sum(w.duration for w in plans["left"][:8])
    for t in np.arange(t0, t1, 0.1):
        pl, pr = arms["left"].target(t)[0], arms["right"].target(t)[0]
        assert np.allclose(pl - pr, [0, 2 * g.GRIP_Y, 0], atol=1e-9)


def test_the_demo_lifts_clear_of_the_stand_and_lowers_the_level_notch_onto_the_pin():
    steps = {label: c for _, c, _, label in M.frame_path()}
    lifted = steps["lift out of the stand"]
    assert lifted[2] - g.H / 2 - (g.TABLE + g.STAND_BASE_T + g.STAND_RAIL_H) >= 0.02
    above = steps["carry to the board, hanger above the pin"]
    assert g.back_gap(above, np.eye(3)) > g.PIN_OUT + 0.02                         # comes in with the pin clear
    over = steps["back against the board"]
    tips = over[2] + g.TIP_Z
    assert tips > g.PIN_Z + g.PIN_R + 0.005                                        # teeth 5 mm+ over the pin
    on = steps["lower the notch onto the pin"]
    assert g.on_pin(on, np.eye(3)) and g.back_gap(on, np.eye(3)) == pytest.approx(g.APPROACH_GAP)


def _links(q, side):
    fk = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q)})
    base = np.asarray(station.BASES[side])
    return [fk[n][:3, 3] + base for n in M.CHAIN if n in fk]


def test_after_the_demo_both_arms_are_clear_of_the_hanging_frame():
    arms, acts = _play(M.demo())
    pos, rot = level_hang()
    caps = []
    for side in ("left", "right"):
        q = acts[side][-1, :6]
        chain = _links(q, side)
        tcp, r = ik.tcp_pose(q, station.BASES[side])
        caps += [(a, b, M.ARM_R) for a, b in zip(chain[:-1], chain[1:])]
        caps.append((chain[-1], tcp + r[:, 2] * M.PAD_TIP, M.GRIP_R))
    assert min(M.capsule_gap(a, b, r, pos, rot) for a, b, r in caps) >= M.CLEAR + 0.01


def test_the_correction_pass_is_reachable_from_the_crooked_hangs():
    import demo

    _, acts = _play(M.demo(0))           # the arms start where the middle-notch hang left them, backed away
    for notch in (0, 1, 3):
        pos, rot = hanging_at(notch, corner_tilt(notch))
        now, target, roll = demo.correction_plan(M, g, pos, rot)
        assert now == notch and abs(target - g.LEVEL_NOTCH) < abs(notch - g.LEVEL_NOTCH)
        assert roll == pytest.approx(corner_tilt(notch))
        poses = demo.correction_poses(g, pos, roll, target)
        assert g.on_pin(g.hang_centre(target) - [0, 0, 0.001], np.eye(3), tol=0.0025)
        for side in ("left", "right"):
            offset = demo.grasp_roll(side, roll) - roll
            q = acts[side][-1, :6]
            p0, r0 = ik.tcp_pose(q, station.BASES[side])
            p1, r1 = demo.arm_targets(M, g, poses[0][0], poses[0][1], side, offset, poses[0][4], poses[0][5])
            for s in np.linspace(0.0, 1.0, 41):  # the plain move from there to past the frame's end
                past_end = abs(p0[1] + s * (p1[1] - p0[1]) - pos[1]) - g.FINGER_HALF_W - g.W / 2
                assert past_end > 0.015    # the fingers stay 1.5 cm past the frame's end on the way
                q, pe, re = ik.solve(p0 + s * (p1 - p0), ik.matrix_to_quat_xyzw(ik.slerp_matrix(r0, r1, s)), q,
                                     base_pos=station.BASES[side], iters=80)
                assert pe < 5e-4 and re < math.radians(0.1), (notch, side, "approach", s)
            prev = poses[0]
            for pose in poses:  # each step sampled as the demo interpolates it
                for s in np.linspace(0.0, 1.0, 11):
                    c = prev[0] + s * (pose[0] - prev[0])
                    r = prev[1] + s * (pose[1] - prev[1])
                    above = prev[4] + s * (pose[4] - prev[4])
                    out = prev[5] + s * (pose[5] - prev[5])
                    p, rr = demo.arm_targets(M, g, c, r, side, offset, above, out)
                    q, pe, re = ik.solve(p, ik.matrix_to_quat_xyzw(rr), q, base_pos=station.BASES[side], iters=80)
                    assert pe < 5e-4 and re < math.radians(0.1), (notch, side, pose[0], s, p)
                    assert np.all(q >= ik.LOWER) and np.all(q <= ik.UPPER)
                prev = pose
