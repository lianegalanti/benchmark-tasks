"""Offline checks (no Isaac Sim): the real dimensions and clearances, the scoring at every level and every violation,
and the scripted demo played against a rigid-grip model of the regulator: reachable, within joint limits and the run
limit, the yoke clear of the valve until it slides on, and seated and clamped at the end.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

import demo
import geometry as g
from yam_arena import ik, station
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
CYL_P = np.array(g.C_ORIGIN, float)
FAR = [(np.array([0.0, 0.6, 1.2]), np.array([0.0, 0.6, 1.3]), 0.03)]
START = np.array(g.REG_START, float)


def reg_pose_c(p_c, R_c=np.eye(3)):
    """A regulator pose given in the cylinder frame, as station pose (cylinder unmoved)."""
    return CYL_P + np.asarray(p_c, float), np.asarray(R_c, float)


def seated(dy=0.0, dz=0.0):
    p, R = g.seated_pose_c()
    return reg_pose_c(p + [0.0, dy, dz], R)


def clamp_turns(extra=0.3):
    theta = -2 * math.pi * (g.turns_to_clamp() + extra)
    d = -(g.TIP0_Y - (g.POST_HY - g.RECESS_DEPTH))   # stopped by the recess floor
    return d, theta


def step(pos, R, d=0.0, theta=0.0, caps=FAR, v=(0, 0, 0), w=(0, 0, 0), cyl=(CYL_P, np.eye(3)), held=False):
    M.update(cyl[0], cyl[1], pos, R, np.asarray(v, float), np.asarray(w, float), d, theta, caps, held=held)
    return M.score({}), M.success({})


@pytest.fixture(autouse=True)
def fresh():
    M._reset()
    step(START, np.eye(3))          # first step records the start
    yield
    M._reset()


def lifted():
    step(START + [0, 0, 0.05], np.eye(3))


# -- task file and real dimensions ---------------------------------------------------------------------------------------

def test_it_loads_with_the_cylinder_and_regulator_free():
    assert set(TASK.free_objects) == {"cylinder", "regulator"}
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0
    data = json.loads((HERE / "task.json").read_text())
    assert data["instruction"] == M.INSTRUCTION
    assert data["run_limit"].startswith("4,800 steps")


def test_poses_are_plain_floats():
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in spec.rot_xyzw), spec.name


def test_pin_index_dimensions_are_iso_407_oxygen():
    assert g.PIN_D == 0.004 and g.PIN_LEN == 0.0055 and g.HOLE_W == 0.00475 and g.HOLE_DEPTH >= 0.0055
    assert g.PIN_R_CIRCLE == 0.0143 and math.degrees(g.PIN_ANGLE) == pytest.approx(18.0)   # positions 2 and 5
    assert 2 * g.POST_HY == pytest.approx(0.025) and g.WASHER_T == 0.0016 and 2 * g.WASHER_R == 0.016
    assert (g.HOLE_W - g.PIN_D) / 2 == pytest.approx(0.000375)             # 0.375 mm a side: a real keyed fit
    assert g.PORT_U >= 0.015


def test_full_size_cylinder_and_regulator():
    assert 2 * g.CYL_R == pytest.approx(0.111) and g.CYL_LEN == pytest.approx(0.419)   # size D
    assert 0.3 <= g.REG_MASS + g.SCREW_MASS <= 0.45
    assert 2 * g.TBAR_HALF == pytest.approx(0.06) and g.PITCH == 0.00175
    assert 6.0 < g.turns_to_clamp() < 6.6


def test_the_wrong_way_round_cannot_seat():
    """Turned end for end (pins toward the post's free end), the pins land on solid metal, 2.7 cm from the holes;
    upside down or back to front, the inlet faces the recess side."""
    flip = g.rot_y(math.pi)
    for pin in g.PINS_R:
        c = g.PORT_U + (flip @ pin)[0]
        assert min(abs(c - h[0]) for h in g.HOLES_C) > 5 * g.HOLE_W
    p, _ = g.seated_pose_c()
    for R in (flip, g.rot_x(math.pi), g.rot_x(math.pi) @ flip):
        assert not g.fit(p, R, 0.0, 0.0)["seated"]


def test_the_window_passes_the_post_with_pins_and_tip_clear():
    pins_off = -g.POST_HY - (g.PIN_TIP_Y + demo.SLIDE_DY)
    tip_off = (g.TIP0_Y + demo.SLIDE_DY) - g.POST_HY
    assert pins_off >= 0.002 and tip_off >= 0.002
    assert g.WINDOW_HZ - g.POST_HZ == pytest.approx(0.0005)                 # snug: the window sets the height
    assert g.END_CHAMFER > 0.002                       # ... once the chamfered post end has led it on


def test_the_regulator_rests_flat_on_the_table_with_its_t_bar_clear():
    assert g.REG_START[2] - g.FRAME_HZ == pytest.approx(g.TABLE + 0.001)
    assert g.BODY_HZ == g.FRAME_HZ and g.GAUGE_R < g.BODY_HZ and g.TBAR_R < g.FRAME_HZ
    assert g.BODY_W + 0.03 < 0.094                     # the open fingers clear the 40 mm body by 2.7 cm
    assert abs(g.BARB[2]) + g.BARB[0] < g.FRAME_HZ


def test_cylinder_lies_in_its_cradle_on_the_table():
    s45 = math.sin(math.radians(45))
    assert g.AXIS_Z - g.CYL_R / s45 == pytest.approx(g.V_APEX_Z)
    assert g.V_APEX_Z > g.TABLE + 0.03
    cyl_end_x = g.POST_TOP_X + g.CYL_END
    assert all(g.POST_TOP_X + g.CYL_X0 + 0.03 < x < cyl_end_x - 0.03 for x in g.SADDLE_X)
    assert cyl_end_x < 0.85 and g.END_BOARD_X - cyl_end_x == pytest.approx(0.001)


def test_layout_off_the_camera_stand():
    stand = station.camera_stand(station.top_camera("calibrated"))["stand_foot"][2]
    assert stand[0] < 0.0 and g.REG_START[0] > 0.2


def test_the_screw_angle_counts_whole_turns():
    from regkit import unwrap

    theta, last = 0.0, 0.0
    for k in range(1, 200):                       # 6.6 turns in, read wrapped to (-pi, pi]
        true = -k * math.radians(12.0)
        raw = math.atan2(math.sin(true), math.cos(true))
        theta, last = unwrap(last, theta, raw), raw
        assert theta == pytest.approx(true)


def _in_post(point) -> bool:
    """Whether a cylinder-frame point is inside any of the post's box colliders."""
    for spec in M.CYLINDER.boxes:
        centre, size = np.asarray(spec[0]), np.asarray(spec[1])
        rot = np.eye(3)
        if len(spec) > 3:
            w, x, y, z = spec[3]
            rot = g.quat_matrix((x, y, z, w))
        local = rot.T @ (np.asarray(point, float) - centre)
        if np.all(np.abs(local) < size / 2 - 1e-7):
            return True
    return False


def test_the_holes_are_open_and_countersunk():
    for hole in g.HOLES_C:
        for depth in np.linspace(0.0002, g.HOLE_DEPTH - 0.0002, 12):     # open down the hole's axis
            assert not _in_post(hole + [0.0, depth, 0.0])
        edge = g.HOLE_W / 2
        for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d = np.array([sx, 0.0, sz])
            assert not _in_post(hole + d * (edge + 0.0004) + [0, 0.0001, 0])   # the mouth is wider (countersink)
            assert _in_post(hole + d * (edge + 0.0004) + [0, 0.0006, 0])       # ... sloping in at 45 degrees
            assert _in_post(hole + d * (edge + 0.0002) + [0, 0.0010, 0])       # below it, the 4.75 mm hole
    assert _in_post([g.PORT_U, 0.0, g.POST_HZ - 0.0002]) and _in_post([0.01, -0.012, 0.0])


def test_the_post_end_is_chamfered_and_leads_a_low_window_on():
    """A window up to 3 mm off in height meets the 45-degree chamfer, not the post's end face."""
    e = g.END_CHAMFER
    for z in np.linspace(-g.POST_HZ + e + 0.0002, g.POST_HZ - e - 0.0002, 7):
        assert _in_post([0.0002, 0.0, z])
    assert not _in_post([0.0002, 0.0, g.POST_HZ - e + 0.0004])
    assert not _in_post([0.001, 0.0, g.POST_HZ - 0.0006])       # on the slope's open side
    assert _in_post([e + 0.0003, 0.0, g.POST_HZ - 0.0002])


# -- scoring --------------------------------------------------------------------------------------------------------------

def test_untouched_scores_zero():
    assert step(START, np.eye(3)) == (0.0, False)


def test_yoke_over_post_scores_a_quarter():
    lifted()
    p, R = reg_pose_c((g.PORT_U, demo.SLIDE_DY, 0.0))
    assert step(p, R) == (0.25, False)


def test_yoke_in_front_of_the_post_scores_zero():
    lifted()
    p, R = reg_pose_c((g.PORT_U - 0.048, demo.SLIDE_DY, 0.0))
    assert step(p, R) == (0.0, False)


def test_seated_on_both_pins_scores_half():
    lifted()
    assert step(*seated()) == (0.5, False)
    f = M._STATE["fit"]
    assert f["pin_depth"] == pytest.approx(0.0039) and f["seated"]


def test_pins_on_the_face_or_cocked_is_not_seated():
    lifted()
    p, R = seated(dy=-0.0026)          # pins short of the holes: the face 2.6 mm off
    assert step(p, R)[0] == 0.25
    p, R = seated()
    assert step(p, R @ g.rot_x(math.radians(4.0)))[0] == 0.25   # 4 degrees cocked


def test_clamped_but_held_scores_three_quarters():
    lifted()
    d, theta = clamp_turns()
    near = [(np.array([0.29, -0.04, 0.95]), np.array([0.29, -0.04, 0.88]), 0.02)]
    assert step(*seated(), d=d, theta=theta, caps=near) == (0.75, False)


def test_clamped_released_and_still_succeeds():
    lifted()
    d, theta = clamp_turns()
    for _ in range(M.STILL_STEPS - 1):
        assert step(*seated(), d=d, theta=theta) == (0.75, False)    # not yet still for a second
    assert step(*seated(), d=d, theta=theta) == (0.75, True)


def test_turned_about_the_screw_axis_is_still_seated():
    lifted()
    p, R = seated()
    assert step(p, R @ g.rot_y(math.radians(3.0)))[0] == 0.5


def test_clamp_needs_hand_tight_torque():
    """Hand-tight: it takes at least 0.75 N m to turn the T-bar further, which needs 28 N of clamp force, reached
    29 degrees past first contact; 60 N (1.55 N m) at 62 degrees."""
    lifted()
    d, _ = clamp_turns()
    touching = -2 * math.pi * g.turns_to_clamp()
    f_hand = (g.HAND_TIGHT - g.TURN_TORQUE0) / g.TORQUE_PER_N
    past = f_hand / g.TIP_STIFFNESS / g.PITCH * 2 * math.pi
    assert f_hand == pytest.approx(28.0) and math.degrees(past) == pytest.approx(28.8, abs=0.1)
    assert g.turn_torque(g.TIP_FORCE) == pytest.approx(1.55)
    assert step(*seated(), d=d, theta=touching)[0] == 0.5
    assert step(*seated(), d=d, theta=touching - 0.95 * past)[0] == 0.5
    assert step(*seated(), d=d, theta=touching - 1.05 * past)[0] == 0.75
    assert M._STATE["fit"]["torque"] >= g.HAND_TIGHT


def test_screw_tightened_on_a_yoke_not_on_its_pins_is_not_clamped():
    lifted()
    p, R = seated(dy=-0.0026)
    d = -(g.TIP0_Y + (-0.0026) - g.POST_HY)       # tip on the recess face's rim, not in it
    assert step(p, R, d=d, theta=-2 * math.pi * 7)[0] == 0.25


def test_moving_while_seated_is_not_success():
    lifted()
    d, theta = clamp_turns()
    p, R = seated()
    for k in range(M.STILL_STEPS + 5):
        wobble = g.rot_x(math.radians(0.4 * (-1) ** k))      # rocking 0.8 degrees
        assert step(p, R @ wobble, d=d, theta=theta)[1] is False


def installed_on(cyl_pos, cyl_R=np.eye(3)):
    """The regulator seated and hand-tight on a cylinder lying at (cyl_pos, cyl_R), station frame."""
    p, R = g.seated_pose_c()
    return cyl_pos + cyl_R @ p, cyl_R @ R


def finish(pos, R, cyl):
    """Clamped, let go and still for the 1 s success window; returns the last (score, success)."""
    d, theta = clamp_turns()
    out = None
    for _ in range(M.STILL_STEPS):
        out = step(pos, R, d=d, theta=theta, cyl=cyl)
    return out


def test_set_down_then_installed_scores_full():
    lifted()
    assert step(START + [0.0, 0.05, 0.0], np.eye(3), held=True)[0] == 0.0     # set down again on the table
    assert step(START + [0.0, 0.05, 0.0], np.eye(3)) == (0.0, False) and not M.violation()
    lifted()
    assert finish(*installed_on(CYL_P), cyl=(CYL_P, np.eye(3))) == (0.75, True)


def test_cylinder_nudged_3_cm_then_installed_scores_full():
    """Sliding the cylinder 3 cm along its cradle and rolling it 10 degrees is allowed; the fit is judged on it."""
    lifted()
    moved = (CYL_P + [0.03, 0.0, 0.0], g.rot_x(math.radians(10.0)))
    assert step(START + [0, 0, 0.05], np.eye(3), cyl=moved)[0] == 0.0 and not M.violation()
    assert finish(*installed_on(*moved), cyl=moved) == (0.75, True)
    # the old, unmoved seated pose is not on the moved cylinder's valve
    assert step(*seated(), cyl=moved)[0] == 0.0


def test_cylinder_tipped_or_fallen_out_of_its_cradle_zeroes_everything_for_good():
    lifted()
    assert step(*installed_on(CYL_P), cyl=(CYL_P, g.rot_y(math.radians(31.0))))[0] == 0.0
    assert finish(*installed_on(CYL_P), cyl=(CYL_P, np.eye(3))) == (0.0, False)
    M._reset()
    step(START, np.eye(3))
    tilted = (CYL_P, g.rot_y(math.radians(25.0)))                                            # tilted, still cradled
    assert step(*installed_on(*tilted), cyl=tilted)[0] == 0.5
    assert not M.violation()
    assert step(START, np.eye(3), cyl=(CYL_P - [0, 0, 0.025], np.eye(3)))[0] == 0.0          # fell out of the saddles
    assert "cradle" in M._STATE["why"][0]


def test_regulator_dropped_zeroes_everything():
    lifted()
    assert step(START + [0, 0, 0.03], np.eye(3), v=(0, 0, -0.8))[0] == 0.0
    assert "dropped" in M._STATE["why"][0]
    assert finish(*installed_on(CYL_P), cyl=(CYL_P, np.eye(3))) == (0.0, False)


def test_lowered_fast_in_the_hand_is_not_a_drop():
    lifted()
    assert step(START + [0, 0, 0.03], np.eye(3), v=(0, 0, -0.8), held=True)[0] == 0.0
    assert not M.violation()


def test_off_the_table_zeroes_everything():
    assert step(START - [0, 0, 0.08], np.eye(3))[0] == 0.0 and M.violation()


def test_reset_clears_the_memory():
    step(START, np.eye(3), v=(0, 0, -0.8))
    assert M.violation()
    M._reset()
    assert not M.violation() and M._STATE["cyl0"] is None


def test_held_by_needs_a_closed_gripper_at_the_regulator():
    pos, R = np.array(g.SEATED), np.eye(3)
    grip_tcp = pos + [0.0, g.GRIP_Y, 0.006]
    assert M.held_by([(grip_tcp, 0.04)], pos, R)
    assert not M.held_by([(grip_tcp, 0.094)], pos, R)                  # open fingers
    assert not M.held_by([(grip_tcp + [0, 0, 0.1], 0.04)], pos, R)     # closed, 10 cm away


# -- the demo, against a rigid-grip regulator -----------------------------------------------------------------------------

class FakeCtx(demo.Ctx):
    """Perfect joint tracking; the regulator follows the right grip rigidly while it is closed."""

    def __init__(self, variant=""):
        super().__init__()
        self.reg = START - [0.0, 0.0, 0.001]          # settled onto the table (it spawns 1 mm up)
        self.offset = None

    def reg_pose(self):
        return self.reg.copy(), np.eye(3)

    def measured_turns(self):
        return self.turned

    def reg_in_c(self):
        return self.reg - CYL_P, np.eye(3), np.eye(3)

    def measured(self, side):
        return (self.left if side == "left" else self.right).q.copy()

    def update(self):
        R = self.right
        if self.offset is None and R.grip < 0.4:
            self.offset = self.reg - R.pos
        elif self.offset is not None and R.grip > 0.5:
            self.offset = False
        if isinstance(self.offset, np.ndarray):
            self.reg = R.pos + self.offset


@pytest.fixture(scope="module")
def played():
    ctx = FakeCtx()
    log = []
    for _ in demo.script(ctx):
        ctx.update()
        log.append(dict(lq=ctx.left.q.copy(), rq=ctx.right.q.copy(), reg=ctx.reg.copy(), turned=ctx.turned,
                        lgrip=ctx.left.grip, rgrip=ctx.right.grip, seated=ctx.flags["seated"]))
    return ctx, log


def test_the_demo_is_reachable_within_limits_and_time(played):
    ctx, log = played
    for arm in (ctx.left, ctx.right):
        assert arm.worst_mm < 0.5 and arm.worst_deg < 0.1, arm.side
    for e in log:
        for q in (e["lq"], e["rq"]):
            assert np.all(q >= ik.LOWER - 1e-6) and np.all(q <= ik.UPPER + 1e-6)
    assert len(log) + 3 * demo.CONTROL_HZ < 4800 * 0.85


def test_the_right_hand_grips_the_body_over_the_centre_of_mass(played):
    ctx, log = played
    assert isinstance(ctx.offset, (np.ndarray, bool))
    first = next(i for i, e in enumerate(log) if e["rgrip"] < 0.4)
    assert np.allclose(log[first]["reg"], START - [0.0, 0.0, 0.001])


def test_the_yoke_stays_clear_of_the_post_until_it_slides_on(played):
    """Whenever the yoke's far edge is past the spindle end and the yoke is near the post, it is exactly on the slide
    line (window centred on the post, pins clear) or seated."""
    _, log = played
    on_line = 0
    for e in log:
        p = e["reg"] - CYL_P
        near = abs(p[1]) < 0.05 and abs(p[2]) < 0.06
        if near and p[0] + g.PLATE_X[1] > -0.006 and p[0] < g.PORT_U - 0.0005:
            assert abs(p[2]) < 0.0005 and abs(p[1] - demo.SLIDE_DY) < 0.0005
            on_line += 1
    assert on_line > 2 * demo.CONTROL_HZ


def test_the_demo_ends_seated_and_clamped_and_let_go(played):
    ctx, log = played
    end = log[-1]
    p = end["reg"] - CYL_P
    assert p == pytest.approx([g.PORT_U, demo.PRESS, 0.0], abs=1e-4)
    d = max(-(g.TIP0_Y - (g.POST_HY - g.RECESS_DEPTH)), -g.PITCH * end["turned"] / (2 * math.pi))
    f = g.fit(np.array([g.PORT_U, 0.0, 0.0]), np.eye(3), d, -end["turned"])
    assert f["seated"] and f["clamped"]
    assert end["turned"] / (2 * math.pi) == pytest.approx(g.turns_to_clamp() + demo.EXTRA_TURN, abs=0.01)
    assert end["lgrip"] == 1.0 and end["rgrip"] == 1.0


def test_the_left_wrist_strokes_stay_in_range(played):
    _, log = played
    q6 = [e["lq"][5] for e in log]
    assert max(q6) < 2.0944 - 0.02 and min(q6) > -2.0944 + 0.02


def test_the_demo_reads_no_object_state():
    """The demo's control code uses only the layout and the arms' encoders; ``run`` reads the monitor for its log."""
    import inspect

    for fn in (demo.fit_regulator, demo.turn_screw, demo.settle, demo.Ctx):
        src = inspect.getsource(fn)
        assert "_STATE" not in src and "object_states" not in src and "task." not in src, fn


def test_the_right_fingers_close_on_the_body_only():
    """At the grasp, the closed fingers (39 mm apart) touch the 40 mm body and stay 1 cm clear of the yoke, the gauge
    and the table (the YAM's fingers are 5 cm wide)."""
    from yam_arena import analysis
    from yam_arena.constants import ARM_JOINTS, READY

    base = np.asarray(station.BASES["right"], float)
    rest = np.array([g.REG_START[0], g.REG_START[1], g.TABLE + g.FRAME_HZ])

    def box(lo, hi, n=9):
        t = [np.linspace(a, b, n) for a, b in zip(lo, hi)]
        return np.array(np.meshgrid(*t)).reshape(3, -1).T + rest

    parts = {"plate": box((g.PLATE_X[0], g.INLET_Y - g.PLATE_T, -g.FRAME_HZ), (g.PLATE_X[1], g.INLET_Y, g.FRAME_HZ)),
             "gauge": box((-g.GAUGE_R, g.BODY_Y[0] - g.GAUGE_T, -g.GAUGE_R), (g.GAUGE_R, g.BODY_Y[0], g.GAUGE_R)),
             "table": box((-0.1, -0.15, -g.FRAME_HZ - 0.003), (0.1, 0.1, -g.FRAME_HZ - 0.002)),
             "body": box((-g.BODY_W / 2, g.BODY_Y[0], -g.BODY_HZ), (g.BODY_W / 2, g.BODY_Y[1], g.BODY_HZ))}
    q, _, _ = ik.solve(demo.reg_to_tcp(rest), ik.matrix_to_quat_xyzw(demo.down_across_x()),
                       q_init=np.array(READY, float), base_pos=base, iters=200)
    qd = dict(zip(ARM_JOINTS, q))
    qd["joint7"] = qd["joint8"] = -0.0195
    hulls = analysis.world_hulls(analysis.fk_link_poses_world(qd, base))
    dist = {k: min(analysis._signed(p, h).min() for _, h in hulls) for k, p in parts.items()}
    assert dist["body"] < 0.0 and min(dist["plate"], dist["gauge"], dist["table"]) > 0.01
