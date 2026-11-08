"""Offline checks (no Isaac Sim): the layout and its clearances, the printed marks, the scoring at every level, and the
scripted demo, played against a rigid-grip model of the parcel: reachable, within joint limits, turning the arrows'
end up, and with every arm link (as the URDF's collision hulls) clear of the table and of the parcel except where the
fingers grip it.

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
START_R = g.rot(g.PARCEL_START_XYZW)
UP = np.eye(3)                                            # standing the right way up
UPSIDE_DOWN = np.diag([1.0, -1.0, -1.0])


def standing(R=UP, dx=0.0, dy=0.0):
    """The parcel standing (or lying, for a lying R) on the mark, resting on the tape."""
    pos = np.array([g.MARK_XY[0] + dx, g.MARK_XY[1] + dy, 0.0])
    pos[2] = g.MARK_TOP - g.lowest_z(pos, R)
    return pos, R


FAR = [(np.array([0.0, 0.6, 1.2]), 0.094, np.array([0.0, 0.6, 1.3]), np.array([0.0, 0.6, 1.2]))] * 2


def step(pos, R, grippers=FAR, moving=False):
    v = np.array([0.0, 0.0, 0.1]) if moving else np.zeros(3)
    return M.update(pos, R, v, np.zeros(3), grippers), M.success({})


def gripper_at(tcp, R_tcp, opening):
    tcp = np.asarray(tcp, float)
    return (tcp, opening, tcp - R_tcp[:, 2] * 0.134, tcp + R_tcp[:, 2] * 0.013)


# -- layout and marks -------------------------------------------------------------------------------------------------

def test_it_loads_with_only_the_parcel_free():
    assert TASK.free_objects == ("parcel",)
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0
    assert (HERE / "assets" / "parcel.usda").is_file()


def test_poses_are_plain_floats():
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", ())), spec.name


def test_the_parcel_lies_on_its_side_up_end_toward_the_right_arm():
    assert g.up_dir(START_R) == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)
    assert g.lowest_z(g.PARCEL_START, START_R) - g.TOP == pytest.approx(0.001, abs=1e-6)
    assert g.tilt(START_R) == pytest.approx(math.pi / 2)
    assert not g.classify(g.PARCEL_START, START_R)["upright_any"]


def test_the_parcel_starts_off_the_mark_with_room_between():
    c = g.corners(g.PARCEL_START, START_R)
    mark_left = g.MARK_XY[1] + g.MARK[1] / 2
    assert c[:, 1].min() - mark_left >= 0.06     # 7 cm of bare table between the parcel's end and the tape
    assert g.classify(g.PARCEL_START, START_R)["centre_off"] > 0.15


def test_the_layout_is_in_reach_and_off_the_camera_stand():
    stand = station.camera_stand(station.top_camera("calibrated"))["stand_foot"][2]
    for x, y in (g.PARCEL_XY, g.MARK_XY):
        assert 0.10 <= x <= 0.40 and abs(y - stand[1]) + abs(x - stand[0]) > 0.3


def test_the_open_fingers_clear_the_parcel():
    assert -2 * FINGER_OPEN - g.WIDTH >= 0.022     # 1.2 cm each side of the 7 cm parcel


def test_the_marks_point_to_the_up_end_and_stay_on_the_faces():
    text = (HERE / "assets" / "parcel.usda").read_text()
    assert text.count("PhysicsCollisionAPI") == 1  # one box collider; the marks are visual only
    assert text.count('def Mesh "arrow_head_') == 4 and text.count('def Cube "label_') == 4
    assert 'def Cube "top_panel"' in text
    sw, sl, s0 = g.SHAFT
    assert s0 + sl < g.ARROW_TIP_Z < g.LENGTH / 2 - 0.02        # the head points up, 2.5 cm short of the end
    assert g.LABEL_Z - g.LABEL[1] / 2 <= s0 and g.ARROW_TIP_Z <= g.LABEL_Z + g.LABEL[1] / 2   # arrow on its label
    assert g.LABEL[0] < g.WIDTH and g.HEAD[0] < g.LABEL[0] and g.LABEL[1] < g.LENGTH


def test_the_arrows_are_big_enough_for_the_top_camera():
    """The arrow on the parcel's top face, projected into the 320 px wide top camera: at least 20 px long
    (the 9 cm arrow, seen at 45 degrees from about 65 cm), pointing to the image's right."""
    cam = station.top_camera("calibrated")
    from yam_arena.urdf import quat_xyzw_to_matrix

    Rc = quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw, float))
    hfov = math.radians(69.0)
    f = 160 / math.tan(hfov / 2)

    def px(p):  # OpenGL camera: looks along -z, x right, y up
        v = Rc.T @ (np.asarray(p) - np.asarray(cam.pos))
        return np.array([f * v[0] / -v[2], f * v[1] / -v[2]])

    tail = g.point(g.PARCEL_START, START_R, (0.0, g.WIDTH / 2, g.SHAFT[2]))
    tip = g.point(g.PARCEL_START, START_R, (0.0, g.WIDTH / 2, g.ARROW_TIP_Z))
    a, b = px(tail), px(tip)
    assert np.linalg.norm(b - a) >= 20
    assert b[0] > a[0]  # the arrow points to the image's right, the robot's right


# -- scoring ----------------------------------------------------------------------------------------------------------

def test_untouched_scores_zero():
    M._reset_state()
    assert step(g.PARCEL_START, START_R) == (0.0, False)


def test_lifted_scores_a_quarter_and_keeps_it():
    M._reset_state()
    up = np.array(g.PARCEL_START) + [0, 0, g.LIFTED + 0.002]
    assert step(up, START_R, moving=True)[0] == 0.25
    assert step(g.PARCEL_START, START_R) == (0.25, False)
    M._reset_state()
    assert step(np.array(g.PARCEL_START) + [0, 0, g.LIFTED - 0.005], START_R)[0] == 0.0


def test_tipped_upright_off_the_mark_scores_a_quarter_either_way_up():
    for R in (UP, UPSIDE_DOWN):
        M._reset_state()
        pos = np.array([*g.PARCEL_XY, g.TOP + g.LENGTH / 2])
        assert step(pos, R) == (0.25, False)


def test_right_way_up_on_the_mark_let_go_and_still_succeeds():
    M._reset_state()
    assert step(*standing()) == (1.0, True)
    M._reset_state()
    tilted = np.array([[1, 0, 0], [0, math.cos(0.24), -math.sin(0.24)], [0, math.sin(0.24), math.cos(0.24)]])
    assert step(*standing(tilted, dx=0.02, dy=-0.02)) == (1.0, True)   # 14 degrees, 2.8 cm off: still in


def test_on_the_mark_the_wrong_way_or_lying_scores_half():
    lying = START_R
    for R in (UPSIDE_DOWN, lying):
        M._reset_state()
        assert step(*standing(R)) == (0.5, False)
    M._reset_state()
    leaning = np.array([[1, 0, 0], [0, math.cos(0.3), -math.sin(0.3)], [0, math.sin(0.3), math.cos(0.3)]])
    assert step(*standing(leaning)) == (0.5, False)                     # 17 degrees: not upright


def test_off_the_mark_by_more_than_3_cm_is_not_on_it():
    M._reset_state()
    assert step(*standing(dx=0.025, dy=0.02)) == (0.25, False)         # 3.2 cm: upright, so 0.25


def test_held_moving_or_with_a_gripper_touching_is_not_placed():
    pos, R = standing()
    M._reset_state()
    held = [gripper_at(pos, demo.grips()[1], 0.07), FAR[1]]
    assert step(pos, R, held) == (0.25, False)
    M._reset_state()
    open_beside = [gripper_at(pos, demo.grips()[1], 0.094), FAR[1]]    # let go, but fingers still round it
    assert step(pos, R, open_beside) == (0.25, False)
    M._reset_state()
    assert step(pos, R, moving=True) == (0.25, False)
    assert step(pos, R) == (1.0, True)


def test_held_above_the_mark_is_not_on_it():
    M._reset_state()
    pos, R = standing()
    assert step(pos + [0, 0, 0.02], R) == (0.25, False)


def test_off_the_table_zeroes_everything_for_good():
    M._reset_state()
    step(*standing(), moving=True)
    assert step(np.array([0.3, 0.7, g.TOP - 0.3]), START_R)[0] == 0.0
    assert step(*standing()) == (0.0, False)


def test_reset_clears_the_memory():
    M._reset_state()
    step(*standing(UPSIDE_DOWN))
    M._reset_state()
    assert M.score({}) == 0.0 and M._STATE["best"] == 0.0 and not M._STATE["fallen"]


# -- the demo, against a rigid-grip parcel ----------------------------------------------------------------------------

class FakeCtx:
    """The parcel lies still until the left fingers close on it, then follows the grip rigidly; once they open it
    drops onto the tape and stands where it was let go."""

    def __init__(self):
        self.left, self.right = demo.Arm("left"), demo.Arm("right")
        self.pos, self.R = np.array(g.PARCEL_START, float), START_R.copy()
        self.attached, self.released, self.offset, self.release_gap = False, False, None, None

    def parcel(self):
        return self.pos.copy(), self.R.copy()

    def update(self):
        L = self.left
        if not self.attached and not self.released and L.grip < 0.05:
            self.attached = True
            self.offset = (L.rot.T @ (self.pos - L.pos), L.rot.T @ self.R)
        elif self.attached and L.grip > 0.5:
            self.attached, self.released = False, True
            self.release_gap = g.lowest_z(self.pos, self.R) - g.MARK_TOP
            self.pos = self.pos - [0.0, 0.0, self.release_gap]
        if self.attached:
            self.pos, self.R = L.pos + L.rot @ self.offset[0], L.rot @ self.offset[1]


def _hulls(q, grip, holding=False):
    qd = dict(zip(ARM_JOINTS, q))
    finger = FINGER_OPEN * grip
    if holding:  # the fingers stop on the parcel's sides
        finger = min(finger, -g.WIDTH / 2)
    qd["joint7"] = qd["joint8"] = finger
    return analysis.world_hulls(analysis.fk_link_poses_world(qd, LEFT))


def _parcel_points(pos, R, n=7):
    s = np.linspace(-1.0, 1.0, n)
    grid = np.array([[a, b, c] for a in s for b in s for c in np.linspace(-1, 1, 2 * n)])
    surface = grid[np.abs(grid).max(axis=1) > 0.999]
    return (surface * g.HALF) @ R.T + pos


def _dist(points, hulls):
    return min(analysis._signed(points, h).min() for _, h in hulls)


@pytest.fixture(scope="module")
def played():
    ctx = FakeCtx()
    log = []
    for _ in demo.stand_parcel(ctx):
        ctx.update()
        L = ctx.left
        log.append(dict(q=L.q.copy(), grip=L.grip, tcp=L.pos.copy(), rot=L.rot.copy(), pos=ctx.pos.copy(),
                        R=ctx.R.copy(), attached=ctx.attached, released=ctx.released))
    return ctx, log


def test_the_demo_is_reachable_within_limits_and_time(played):
    ctx, log = played
    assert ctx.left.worst_mm < 0.5 and ctx.left.worst_deg < 0.1
    qs = np.array([e["q"] for e in log])
    assert np.all(qs >= ik.LOWER - 1e-9) and np.all(qs <= ik.UPPER + 1e-9)
    assert np.abs(np.diff(qs, axis=0)).max() < 0.15   # rad per control step: no IK branch jumps
    assert len(log) / demo.CONTROL_HZ < 30            # s; the run limit is 60 s
    assert ctx.attached is False and ctx.released is True


def test_the_demo_grips_the_parcel_at_its_middle_across_its_long_sides(played):
    _, log = played
    first = next(e for e in log if e["attached"])
    assert np.linalg.norm(first["tcp"] - np.asarray(g.PARCEL_START)) < 1e-3
    assert abs(first["rot"][:, 0] @ [1.0, 0.0, 0.0]) == pytest.approx(1.0)  # fingers close along x


def test_the_demo_stands_it_the_right_way_up_on_the_mark(played):
    ctx, log = played
    for e in log:
        if e["attached"] and e["tcp"][2] > g.PARCEL_START[2] + 0.06:
            assert g.lowest_z(e["pos"], e["R"]) > g.TOP + 0.02   # carried clear of the table
    assert ctx.release_gap == pytest.approx(demo.DROP, abs=0.002)  # let go 4 mm above the tape
    k = g.classify(ctx.pos, ctx.R)
    assert k["on_mark"] and k["upright_right"] and k["centre_off"] < 0.005 and k["tilt"] < math.radians(1)


def test_the_arm_stays_clear_of_the_table_and_the_parcel(played):
    _, log = played
    worst = {"table": 1.0, "parcel": 1.0, "fingers": 1.0}
    for i, e in enumerate(log):
        if i % 2:
            continue
        gripping = e["attached"] or (e["grip"] < 0.999 and not e["released"])
        hulls = [h for h in _hulls(e["q"], e["grip"], gripping) if h[0] != "base"]
        low = min(h.vertices[:, 2].min() for _, h in hulls)
        worst["table"] = min(worst["table"], low - g.MARK_TOP)
        pts = _parcel_points(e["pos"], e["R"])
        body = [h for h in hulls if h[0] not in ("tip_left", "tip_right")]
        worst["parcel"] = min(worst["parcel"], _dist(pts, body))
        if not gripping and e["grip"] > 0.999:
            tips = [h for h in hulls if h[0] in ("tip_left", "tip_right")]
            worst["fingers"] = min(worst["fingers"], _dist(pts, tips))
    print("margins", {k: round(float(v), 4) for k, v in worst.items()})
    assert worst["table"] > 0.005, worst
    assert worst["parcel"] > 0.01, worst
    assert worst["fingers"] > 0.008, worst


def test_the_demo_ends_with_full_score(played):
    _, log = played
    M._reset_state()
    right = gripper_at(*ik.tcp_pose(np.array(READY, float), station.BASES["right"]), -FINGER_OPEN * 2)
    for x in log[::5] + [log[-1]]:
        grippers = [gripper_at(x["tcp"], x["rot"], -FINGER_OPEN * 2 * x["grip"]), right]
        M.update(x["pos"], x["R"], np.zeros(3), np.zeros(3), grippers)
    assert M.success({}) and M.score({}) == 1.0 and M._STATE["best"] == 0.25
