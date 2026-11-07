"""Offline checks (no Isaac Sim): the layout and its clearances, the scoring at every level, and the scripted demo,
played against a rigid-grip model of the bottle: reachable, within joint limits, and with every arm link (as the
URDF's collision hulls) clear of the table, the towel, the pegs and the bottle except where the fingers grip it.

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

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
LEFT = np.asarray(station.BASES["left"], float)
START_R = g.rot(g.BOTTLE_START_XYZW)
DOWN_R = np.diag([1.0, -1.0, -1.0])  # mouth (+z of the bottle) straight down


def hung(peg=0, tilt=0.0):
    """The bottle hanging on a peg by its inside bottom (optionally leaning ``tilt`` rad about x)."""
    c, s = math.cos(tilt), math.sin(tilt)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ DOWN_R
    tip = np.array([*g.PEGS[peg], g.PEG_TIP])
    return tip - R[:, 2] * g.INNER_BOTTOM_Z, R


FAR = [(np.array([0.0, 0.6, 1.2]), 0.094, np.array([0.0, 0.6, 1.3]), np.array([0.0, 0.6, 1.2]))] * 2


def step(pos, R, grippers=FAR, moving=False):
    v = np.array([0.0, 0.0, 0.1]) if moving else np.zeros(3)
    M.update(pos, R, v, np.zeros(3), grippers)
    return M.score({}), M.success({})


def gripper_at(tcp, R_tcp, opening):
    tcp = np.asarray(tcp, float)
    return (tcp, opening, tcp - R_tcp[:, 2] * 0.134, tcp + R_tcp[:, 2] * 0.013)


# -- layout -----------------------------------------------------------------------------------------------------------

def test_it_loads_with_only_the_bottle_free():
    assert TASK.free_objects == ("bottle",)
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0
    assert (Path(__file__).parent / "assets" / "bottle.usda").is_file()


def test_poses_are_plain_floats():
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", ())), spec.name


def test_the_bottle_lies_on_the_towel_mouth_toward_the_robot():
    assert g.mouth_dir(START_R) == pytest.approx([-1.0, 0.0, 0.0], abs=1e-9)
    assert g.lowest_z(g.BOTTLE_START, START_R) - g.TOWEL_TOP == pytest.approx(0.001, abs=1e-4)
    a0, a1 = g.axis_segment(g.BOTTLE_START, START_R)
    for p in (a0, a1):  # the whole bottle is over the towel
        assert abs(p[0] - g.TOWEL_XY[0]) < g.TOWEL[0] / 2 and abs(p[1] - g.TOWEL_XY[1]) < g.TOWEL[1] / 2 - g.R_OUT


def test_clearances_are_generous():
    assert g.R_IN - g.PEG_R >= 0.02                   # the peg in the bore: 2.1 cm all round
    assert -FINGER_OPEN - g.R_OUT >= 0.015             # open fingers either side of the bottle: 1.7 cm
    assert g.HUNG_RIM_GAP >= 0.015                     # hung, the mouth 2.2 cm above the rack base
    peg_gap = abs(g.PEGS[0][1] - g.PEGS[1][1]) - g.R_OUT - g.PEG_R
    assert peg_gap >= 0.08                              # a hung bottle is 8 cm from the other peg
    towel_far = g.TOWEL_XY[0] + g.TOWEL[0] / 2
    assert g.RACK_XY[0] - g.BASE[0] / 2 - towel_far > 0.0  # towel and rack don't overlap
    assert g.PEG_TIP - g.BASE_TOP > g.MOUTH_Z - g.INNER_BOTTOM_Z  # the bottle can't bottom out on the base


def test_the_layout_is_inside_the_left_arm_reach_and_off_the_camera_stand():
    stand = station.camera_stand(station.top_camera("calibrated"))["stand_foot"][2]
    for x, y in (g.TOWEL_XY, g.RACK_XY, *g.PEGS):
        assert 0.10 <= x <= 0.50 and abs(y - stand[1]) + abs(x - stand[0]) > 0.3


def test_a_hung_bottle_hangs_below_its_support():
    pos, R = hung()
    com = g.point(pos, R, g.COM_Z)
    assert com[2] < g.PEG_TIP - 0.04  # centre of mass 4.9 cm under the tip: a stable pendulum
    assert g.lowest_z(pos, R) - g.BASE_TOP == pytest.approx(g.HUNG_RIM_GAP, abs=1e-6)


# -- scoring ----------------------------------------------------------------------------------------------------------

def test_untouched_scores_zero():
    M._reset_state()
    assert step(g.BOTTLE_START, START_R) == (0.0, False)


def test_lifted_scores_a_quarter_and_keeps_it():
    M._reset_state()
    up = np.array(g.BOTTLE_START) + [0, 0, g.LIFTED + 0.002]
    assert step(up, START_R, moving=True)[0] == 0.25
    assert step(g.BOTTLE_START, START_R) == (0.25, False)  # set down again: kept


def test_lifted_but_under_3_cm_scores_zero():
    M._reset_state()
    rise = g.TOP + g.LIFTED - 0.003 - g.lowest_z(g.BOTTLE_START, START_R)  # lowest point 2.7 cm up
    assert step(np.array(g.BOTTLE_START) + [0, 0, rise], START_R)[0] == 0.0


def test_over_the_peg_but_held_scores_half():
    M._reset_state()
    pos, R = hung()
    held = [gripper_at(g.point(pos, R, g.COM_Z), np.eye(3), 0.06), FAR[1]]
    assert step(pos, R, held) == (0.5, False)


def test_a_peg_only_2_cm_in_is_not_on_the_peg():
    M._reset_state()
    pos, R = hung()
    pos = pos + [0, 0, (g.MOUTH_Z - g.INNER_BOTTOM_Z) - 0.02]  # the mouth only 2 cm below the tip
    assert step(pos, R, moving=True)[0] == 0.25


def test_mouth_up_or_leaning_too_far_is_not_on_the_peg():
    M._reset_state()
    pos, R = hung(tilt=math.radians(35))
    assert g.peg_depth(pos, R, g.PEGS[0]) > g.INSIDE_MIN
    assert step(pos, R)[0] == 0.25
    M._reset_state()
    up = np.diag([1.0, 1.0, 1.0])
    assert max(g.peg_depth(np.array([*g.PEGS[0], g.PEG_TIP + 0.03]), up, xy) for xy in g.PEGS) < 0
    assert M.milestone(np.array([*g.PEGS[0], g.PEG_TIP + 0.03]), up) == 0.25


def test_hanging_let_go_and_still_succeeds_on_either_peg():
    for peg in (0, 1):
        M._reset_state()
        pos, R = hung(peg)
        assert step(pos, R) == (0.5, True)
        M._reset_state()
        assert step(*hung(peg, tilt=math.radians(10))) == (0.5, True)  # leaning on the peg is still hanging


def test_no_success_while_swinging_held_or_with_a_gripper_beside_it():
    pos, R = hung()
    M._reset_state()
    assert step(pos, R, moving=True) == (0.5, False)
    open_beside = [gripper_at(g.point(pos, R, g.COM_Z), np.eye(3), 0.094), FAR[1]]
    M._reset_state()
    assert step(pos, R, open_beside) == (0.5, False)  # let go but fingers still around it


def test_a_bottle_resting_on_the_base_is_not_hanging():
    M._reset_state()
    pos, R = hung(tilt=math.radians(28))
    if g.lowest_z(pos, R) < g.BASE_TOP + g.RIM_CLEAR:
        assert step(pos, R)[1] is False
    # straight down but sitting on the base is impossible: the peg is longer than the bore
    assert g.PEG_TIP - g.BASE_TOP > g.MOUTH_Z - g.INNER_BOTTOM_Z + 0.02


def test_off_the_table_zeroes_everything_for_good():
    M._reset_state()
    pos, R = hung()
    step(pos, R, moving=True)
    assert step(np.array([0.3, 0.7, g.TOP - 0.3]), START_R)[0] == 0.0
    assert step(pos, R) == (0.0, False)


def test_reset_clears_the_memory():
    M._reset_state()
    step(*hung(), moving=True)
    M._reset_state()
    assert M.score({}) == 0.0 and M._STATE["best"] == 0.0 and not M._STATE["fallen"]


# -- the demo, against a rigid-grip bottle ----------------------------------------------------------------------------

class FakeCtx:
    """The bottle lies still until the left fingers close on it, then follows the grip rigidly; once they open it
    drops onto the peg's tip and hangs straight."""

    def __init__(self):
        self.left, self.right = demo.Arm("left"), demo.Arm("right")
        self.pos, self.R = np.array(g.BOTTLE_START, float), START_R.copy()
        self.attached, self.released, self.offset = False, False, None

    def bottle(self):
        return self.pos.copy(), self.R.copy()

    def update(self):
        L = self.left
        if not self.attached and not self.released and L.grip < 0.05:
            self.attached = True
            self.offset = (L.rot.T @ (self.pos - L.pos), L.rot.T @ self.R)
        elif self.attached and L.grip > 0.5:
            self.attached, self.released = False, True
            self.pos, self.R = hung(g.DEMO_PEG)
        if self.attached:
            self.pos, self.R = L.pos + L.rot @ self.offset[0], L.rot @ self.offset[1]


def _hulls(q, grip, holding=False, links=None):
    qd = dict(zip(ARM_JOINTS, q))
    finger = FINGER_OPEN * grip
    if holding:  # the fingers stop on the bottle's sides
        finger = min(finger, -g.R_OUT)
    qd["joint7"] = qd["joint8"] = finger
    return analysis.world_hulls(analysis.fk_link_poses_world(qd, LEFT), links)


def _bottle_points(pos, R, n=24):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    zs = np.linspace(-g.LENGTH / 2, g.MOUTH_Z, 9)
    return np.array([pos + R[:, 2] * z + g.R_OUT * (R[:, 0] * math.cos(t) + R[:, 1] * math.sin(t))
                     for z in zs for t in a])


def _peg_points(xy):
    return np.array([[xy[0], xy[1], z] for z in np.linspace(g.BASE_TOP, g.PEG_TIP - g.PEG_R, 15)]
                    + [[xy[0], xy[1], g.PEG_TIP - g.PEG_R]])


def _dist(points, hulls):
    return min(analysis._signed(points, h).min() for _, h in hulls)


@pytest.fixture(scope="module")
def played():
    ctx = FakeCtx()
    log = []
    for _ in demo.hang_bottle(ctx):
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
    assert len(log) / demo.CONTROL_HZ < 30  # s; the run limit is 60 s
    assert ctx.attached is False and ctx.released is True


def test_the_demo_grips_the_bottle_at_its_middle_across_its_side_staves(played):
    _, log = played
    first = next(e for e in log if e["attached"])
    com = g.point(g.BOTTLE_START, START_R, g.COM_Z)
    assert np.linalg.norm(first["tcp"] - com) < 1e-3
    assert abs(first["rot"][:, 0] @ [0.0, 1.0, 0.0]) == pytest.approx(1.0)  # fingers close along y


def test_the_demo_turns_the_bottle_mouth_down_before_it_meets_the_peg(played):
    _, log = played
    for e in log:
        if not e["attached"]:
            continue
        lowest = g.lowest_z(e["pos"], e["R"])
        if e["tcp"][2] > g.BOTTLE_START[2] + 0.05:
            assert lowest > g.TOWEL_TOP + 0.01  # carried clear of the towel and table
        mouth = g.point(e["pos"], e["R"], g.MOUTH_Z)
        near = math.hypot(*(mouth[:2] - np.array(g.PEGS[g.DEMO_PEG]))) < 0.1
        if near and mouth[2] < g.PEG_TIP + 0.005:  # mouth level with or below the peg tip: the peg well inside
            assert g.tilt_from_down(e["R"]) < math.radians(1.0)
            r = math.hypot(*(mouth[:2] - np.array(g.PEGS[g.DEMO_PEG])))
            assert r < g.R_IN - g.PEG_R - 0.015
        a0, a1 = g.axis_segment(e["pos"], e["R"])
        other = g.PEGS[1 - g.DEMO_PEG]
        assert g.seg_dist(a0, a1, (*other, g.BASE_TOP), (*other, g.PEG_TIP)) > g.R_OUT + g.PEG_R + 0.05
    k = next(i for i, e in enumerate(log) if e["released"])
    before = log[k - 1]
    gap = g.point(before["pos"], before["R"], g.INNER_BOTTOM_Z)[2] - g.PEG_TIP
    assert gap == pytest.approx(demo.DROP, abs=0.002)  # it lets go 4 mm above the tip


def test_the_arm_stays_clear_of_the_table_towel_pegs_and_bottle(played):
    _, log = played
    worst = {"table": 1.0, "pegs": 1.0, "bottle": 1.0, "fingers": 1.0}
    for i, e in enumerate(log):
        if i % 2:
            continue
        hulls = [h for h in _hulls(e["q"], e["grip"], e["attached"] or e["grip"] < 0.999) if h[0] != "base"]
        low = min(h.vertices[:, 2].min() for _, h in hulls)
        worst["table"] = min(worst["table"], low - g.TOWEL_TOP)
        for xy in g.PEGS:
            worst["pegs"] = min(worst["pegs"], _dist(_peg_points(xy), hulls) - g.PEG_R)
        pts = _bottle_points(e["pos"], e["R"])
        body = [h for h in hulls if h[0] not in ("tip_left", "tip_right")]
        worst["bottle"] = min(worst["bottle"], _dist(pts, body))
        gripping = e["attached"] or e["grip"] < 0.999
        if not gripping:
            tips = [h for h in hulls if h[0] in ("tip_left", "tip_right")]
            worst["fingers"] = min(worst["fingers"], _dist(pts, tips))
    print("margins", {k: round(float(v), 4) for k, v in worst.items()})
    assert worst["table"] > 0.005, worst
    assert worst["pegs"] > 0.015, worst
    assert worst["bottle"] > 0.01, worst
    assert worst["fingers"] > 0.01, worst


def test_the_demo_ends_with_the_bottle_hanging_and_the_gripper_clear(played):
    ctx, log = played
    e = log[-1]
    M._reset_state()
    for x in log[::5] + [e]:
        grippers = [gripper_at(x["tcp"], x["rot"], -FINGER_OPEN * 2 * x["grip"]),
                    gripper_at(*ik.tcp_pose(np.array(READY, float), station.BASES["right"]), -FINGER_OPEN * 2)]
        M.update(x["pos"], x["R"], np.zeros(3), np.zeros(3), grippers)
    assert M.success({}) and M._STATE["best"] == 0.5
