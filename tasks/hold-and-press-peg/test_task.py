"""Offline checks (no Isaac Sim): the assets, the scoring, and that the closed-loop demo is reachable, keeps the arms
apart and seats the pin in a kinematic stand-in for the simulation.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from yam_arena import YAM_URDF, ik  # noqa: E402
from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.task_file import load  # noqa: E402
from yam_arena.urdf import forward_kinematics  # noqa: E402

import demo  # noqa: E402
import geometry as g  # noqa: E402
import make_assets  # noqa: E402
from scenekit import BodyState  # noqa: E402

TASK = load(HERE / "task.py")
M = TASK.module
IDENT = (0.0, 0.0, 0.0, 1.0)


def body(pos, quat=IDENT, vel=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return BodyState(pos=pos, quat_xyzw=np.asarray(quat, float), lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3),
                     low=pos, high=pos)


def yaw_q(a):
    return (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))


def peg_at_depth(depth, dx=0.0, dz=-0.0015, bar=g.MOUTH, quat=IDENT):
    """The pin on the hole's axis (dropped onto the floor by default) with its leading end ``depth`` inside."""
    centre = np.asarray(bar, float) + np.array([dx, -(depth - g.PEG_LEN / 2), dz])
    return body(centre, quat)


def states(peg, bar=None):
    return {"bar": bar or body(g.MOUTH), "peg": peg, "sheet": body((*g.SHEET_XY, g.TOP))}


@pytest.fixture(autouse=True)
def fresh():
    M.reset_tracking(states(body(g.PEG_START)))
    yield


# -- the files ----------------------------------------------------------------------------------------------------


def test_it_loads_with_the_bar_and_pin_free_and_the_sheet_fixed():
    assert set(TASK.free_objects) == {"bar", "peg"}
    assert "sheet" in TASK.names and TASK.score_over == "best"


def test_the_usd_files_match_the_geometry():
    assert make_assets.main(check=True) == 0


def test_the_hole_is_a_closed_cavity_with_generous_clearance():
    boxes = g.bar_boxes()
    lo = {k: np.array(c) - np.array(s) / 2 for k, (c, s) in boxes.items()}
    hi = {k: np.array(c) + np.array(s) / 2 for k, (c, s) in boxes.items()}
    assert hi["floor"][2] == pytest.approx(-g.HOLE / 2) and lo["roof"][2] == pytest.approx(g.HOLE / 2)
    assert lo["wall_pos_x"][0] == pytest.approx(g.HOLE / 2) and hi["wall_neg_x"][0] == pytest.approx(-g.HOLE / 2)
    assert hi["solid_end"][1] == pytest.approx(-g.HOLE_DEPTH)  # the hole's bottom
    assert lo["solid_end"][1] == pytest.approx(-g.BAR_LEN) and max(h[1] for h in hi.values()) == pytest.approx(0)
    assert 0.002 <= g.HOLE - g.PEG_D <= 0.0035  # 3 mm diametral clearance
    assert g.PEG_LEN == g.HOLE_DEPTH  # flush when it bottoms


def test_the_bar_lies_on_the_sheet_and_the_pin_on_the_bare_table():
    sx, sy = g.SHEET_XY
    half = np.array(g.SHEET_SIZE[:2]) / 2
    for p in (g.MOUTH, g.FAR_END):
        assert abs(p[0] - sx) < half[0] - 0.05 and abs(p[1] - sy) < half[1] - 0.03
    assert g.AXIS_Z - g.BAR_W / 2 == pytest.approx(g.SHEET_TOP)
    assert g.PEG_START[1] - g.PEG_LEN / 2 - (sy + half[1]) > 0.05  # pin well off the sheet


def test_one_arm_pushing_drags_the_bar_in_the_simulation():
    """Pushed along the hole floor, the pin's friction on the bar beats the bar's friction on the sheet."""
    peg_on_floor = g.PEG_MASS * 9.81 * min(g.PEG_FRICTION[1], g.BAR_FRICTION[1])  # average of 0.9 and 0.9
    bar_on_sheet = (g.BAR_MASS + g.PEG_MASS) * 9.81 * g.SHEET_FRICTION[0]
    assert peg_on_floor > 3 * bar_on_sheet


# -- scoring --------------------------------------------------------------------------------------------------------


def test_success_needs_the_pin_flush_still_and_the_bar_never_moved_1_cm():
    seated = states(peg_at_depth(g.HOLE_DEPTH - 0.001))
    M.track(seated)
    assert M.success(seated)
    assert M.success(states(peg_at_depth(g.HOLE_DEPTH - 0.0019)))
    assert not M.success(states(peg_at_depth(g.HOLE_DEPTH - 0.004)))  # 4 mm proud
    moving = states(body(peg_at_depth(g.HOLE_DEPTH).pos, vel=(0, -0.05, 0)))
    assert not M.success(moving)


def test_a_bar_pushed_back_after_moving_does_not_score_full():
    pushed = np.asarray(g.MOUTH) + [0, -0.03, 0]
    M.track(states(peg_at_depth(0.03, bar=pushed), bar=body(pushed)))
    back = states(peg_at_depth(g.HOLE_DEPTH))
    M.track(back)
    assert M.max_move() == pytest.approx(0.03)
    assert not M.success(back) and M.score(back) == 0.5


def test_a_turned_bar_counts_by_its_ends():
    turned = body(g.MOUTH, yaw_q(math.radians(5)))  # about the mouth: the far end moves 1.5 cm
    M.track(states(peg_at_depth(g.HOLE_DEPTH, bar=g.MOUTH, quat=yaw_q(math.radians(5))), bar=turned))
    assert M.max_move() == pytest.approx(g.BAR_LEN * 2 * math.sin(math.radians(2.5)), abs=1e-4)


def test_partial_credit():
    def scored(peg, bar_pos=g.MOUTH):
        M.reset_tracking(states(body(g.PEG_START)))
        s = states(peg, bar=body(bar_pos))
        M.track(s)
        return 1.0 if M.success(s) else M.score(s)

    shoved = np.asarray(g.MOUTH) + [0.0, -0.03, 0.0]
    far = np.asarray(g.MOUTH) + [0.0, -0.08, 0.0]
    cases = {
        "seated, bar still": (peg_at_depth(g.HOLE_DEPTH), g.MOUTH, 1.0),
        "seated, bar 3 cm": (peg_at_depth(g.HOLE_DEPTH, bar=shoved), shoved, 0.5),
        "seated, bar 8 cm": (peg_at_depth(g.HOLE_DEPTH, bar=far), far, 0.25),
        "half in": (peg_at_depth(0.035), g.MOUTH, 0.25),
        "tip 3 mm in": (peg_at_depth(0.003), g.MOUTH, 0.0),
        "at the start": (body(g.PEG_START), g.MOUTH, 0.0),
        "on top of the bar": (body(np.asarray(g.MOUTH) + [0, -0.04, g.BAR_W / 2 + g.PEG_R]), g.MOUTH, 0.0),
        "beside the bar": (body(np.asarray(g.MOUTH) + [0.03, -0.04, -0.0075]), g.MOUTH, 0.0),
    }
    for name, (peg, bar_pos, want) in cases.items():
        assert scored(peg, bar_pos) == want, name


def test_a_pin_in_the_wrong_way_round_still_counts():
    """A plain dowel is the same both ways: which end leads doesn't matter."""
    flipped = peg_at_depth(g.HOLE_DEPTH, quat=(0.0, 0.0, 1.0, 0.0))
    M.track(states(flipped))
    assert M.success(states(flipped))


# -- the demo, offline --------------------------------------------------------------------------------------------


class FakeCtx:
    """Kinematic stand-in: the bar stays put; the pin lies still, rides in the left gripper once it closes on it
    (with a 1.5 mm sag and 1 mm sideways error), drops 1.5 mm onto the hole's floor when let go, and is pushed along
    by the closed fingertips."""

    SAG = np.array([0.001, 0.0, -0.0015])

    def __init__(self, bias=(0.0, 0.0, 0.0)):
        self.bias = np.asarray(bias, float)  # extra error of the held pin once it is down at the hole's height
        self.entry_err = None  # the leading end's offset from the axis as it crossed the mouth
        self.left, self.right = demo.Arm("left"), demo.Arm("right")
        self.flags = {"held": False, "done": False}
        self.task = M
        self.peg_c = np.asarray(g.PEG_START, float)
        self.mode, self.offset = "table", None

    def bar(self):
        return np.asarray(g.MOUTH, float), np.eye(3)

    def peg(self):
        return self.peg_c.copy(), np.array([0.0, 1.0, 0.0])

    def update(self):
        L = self.left
        if self.mode == "table" and L.grip < 0.3 and np.linalg.norm(L.pos[:2] - self.peg_c[:2]) < 0.04:
            self.mode, self.offset = "held", self.peg_c - L.pos + self.SAG
        elif self.mode == "held" and L.grip > 0.6:
            self.mode = "free"
            self.peg_c = np.array([g.MOUTH[0], self.peg_c[1], g.AXIS_Z - 0.0015])
        if self.mode == "held":
            low = L.pos[2] < g.TOP + 0.05
            self.peg_c = L.pos + self.offset + (self.bias if low else 0.0)
            lead = self.peg_c - np.array([0.0, g.PEG_LEN / 2, 0.0])
            if self.entry_err is None and lead[1] < g.MOUTH[1]:
                self.entry_err = np.abs((lead - np.asarray(g.MOUTH))[[0, 2]]).max()
        elif self.mode == "free" and L.grip < 0.3:
            tip = L.pos + L.rot @ np.array([0.0, 0.0, PAD_TIP_PAST_TCP])
            front = tip[1] - demo.TIP_FRONT  # the leaning fingertips' front pushes the outer end
            outer = self.peg_c[1] + g.PEG_LEN / 2
            if abs(L.pos[2] - g.AXIS_Z) < 0.02 and abs(L.pos[0] - g.MOUTH[0]) < 0.01 and front < outer:
                self.peg_c[1] -= outer - front
        if self.mode == "free":  # the hole's bottom stops it
            self.peg_c[1] = max(self.peg_c[1], g.MOUTH[1] - g.HOLE_DEPTH + g.PEG_LEN / 2)

    def objects(self):
        return states(body(self.peg_c))

    def fingers(self, side):
        arm = getattr(self, side)
        across = arm.rot @ np.array([0.01, 0.0, 0.0])
        return arm.pos + across, arm.pos - across

    def wrist(self, side):
        return getattr(self, side).rot.copy()


def _chain(arm):
    fk = forward_kinematics(YAM_URDF, dict(zip(ARM_JOINTS, arm.q, strict=True)))
    pts = [fk[n][:3, 3] + arm.base for n in ("link2", "link3", "link4", "link5", "gripper")]
    return np.array([*pts, arm.pos])


def _seg_dist(a, b, c, d, n=12):
    s = np.linspace(0.0, 1.0, n)
    p, q = a + np.outer(s, b - a), c + np.outer(s, d - c)
    return float(np.linalg.norm(p[:, None] - q[None], axis=2).min())


@pytest.fixture(scope="module")
def demo_run():
    ctx = FakeCtx()
    M.reset_tracking(ctx.objects())
    gen = demo.script(ctx, g)
    steps, min_arm_gap, low = 0, np.inf, np.inf
    for _ in gen:
        ctx.update()
        steps += 1
        if steps % 4 == 0:
            a, b = _chain(ctx.left), _chain(ctx.right)
            min_arm_gap = min(min_arm_gap, min(_seg_dist(a[i], a[i + 1], b[j], b[j + 1])
                                               for i in range(len(a) - 1) for j in range(len(b) - 1)))
        low = min(low, ctx.left.pos[2] - PAD_TIP_PAST_TCP, ctx.right.pos[2] - PAD_TIP_PAST_TCP)
        if steps > 3000:
            break
    ctx.update()
    return {"ctx": ctx, "steps": steps, "min_arm_gap": min_arm_gap, "lowest_tip": low}


def test_the_demo_seats_the_pin_in_the_stand_in(demo_run):
    objs = demo_run["ctx"].objects()
    M.track(objs)
    assert M.seated(objs) and M.success(objs)


def test_the_demo_corrects_a_held_pin_that_is_off_the_axis():
    """With the pin 2 mm off the axis in x and z at the hole's height, the loop lines it up before it enters."""
    ctx = FakeCtx(bias=(0.002, 0.0, -0.002))
    M.reset_tracking(ctx.objects())
    for _ in demo.script(ctx, g):
        ctx.update()
    ctx.update()
    assert ctx.entry_err is not None and ctx.entry_err < (g.HOLE - g.PEG_D) / 2
    assert M.seated(ctx.objects())


def test_the_demo_is_reachable_and_within_the_run_limit(demo_run):
    ctx = demo_run["ctx"]
    for arm in (ctx.left, ctx.right):
        assert arm.worst_mm < 0.5 and arm.worst_deg < 0.1, arm.side
        assert np.all(arm.q_low >= ik.LOWER) and np.all(arm.q_high <= ik.UPPER), arm.side
    assert demo_run["steps"] / demo.CONTROL_HZ < 45  # well inside 2,400 steps (120 s), with time to settle


def test_the_arms_stay_apart_and_the_pads_off_the_table(demo_run):
    assert demo_run["min_arm_gap"] > 0.09  # link axes; the links are about 6 cm across
    assert demo_run["lowest_tip"] > g.TOP + 0.002


def _closed_fingers_tcp_frame() -> np.ndarray:
    """The vertices of both closed fingers' collision meshes in the tool-point frame (from the URDF's meshes)."""
    import struct

    def stl(path):
        b = path.read_bytes()
        n = struct.unpack("<I", b[80:84])[0]
        rec = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
        return rec["v"].reshape(-1, 3).astype(float)

    def rpy(r, p, y):
        cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
                @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))

    # tip_left's collision origin and joint7's origin (finger closed), from yam_arm.urdf; tip_right is its mirror
    pts = np.vstack([stl(f) for f in sorted((Path(YAM_URDF).parent / "collision" / "tip_left").glob("*.stl"))])
    pts = (pts + [0.00999321, -0.0914614, 0.129783]) @ rpy(-math.pi, 0.0, math.pi / 2).T
    pts = pts + [0.0445119, -0.0239931, 0.0566829] - [0.0, 0.0, 0.134]
    assert abs(pts[:, 2].max() - PAD_TIP_PAST_TCP) < 0.001
    return np.vstack([pts, pts * [-1.0, 1.0, 1.0]])


def test_the_leaning_fingertips_push_the_pin_flush_without_touching_the_bar():
    """At flush (the fingertips' front on the bar's end face), every part of the closed fingers that is at or past
    the face is inside the hole's cross-section or above the bar, and the fingertips clear the sheet."""
    from yam_arena.demo import grip_rotation

    down, into = np.array([0.0, 0.0, -1.0]), np.array([0.0, -1.0, 0.0])
    lean = demo.rotation_between(down, down * math.cos(demo.PUSH_LEAN) + into * math.sin(demo.PUSH_LEAN))
    rot = lean @ grip_rotation(demo.LEFT_YAW)
    mouth = np.asarray(g.MOUTH, float)
    tip = mouth - into * demo.TIP_FRONT
    tip[2] = g.AXIS_Z - g.HOLE / 2 + demo.PUSH_TIP_UP
    tcp = tip - rot @ np.array([0.0, 0.0, PAD_TIP_PAST_TCP])
    pts = tcp + _closed_fingers_tcp_frame() @ rot.T
    depth = (pts - mouth) @ into
    inside = (np.abs(pts[:, 0] - mouth[0]) < g.HOLE / 2 - 0.001) & (np.abs(pts[:, 2] - g.AXIS_Z) < g.HOLE / 2 - 0.001)
    above = pts[:, 2] > g.SHEET_TOP + g.BAR_W + 0.002
    past = depth > -0.001
    assert np.all(inside[past] | above[past])
    assert abs(depth.max()) < 0.0003  # the tips reach the face (demo.TIP_FRONT ahead of their centre)
    assert depth[~(inside | above)].max() < -0.005  # with 5 mm to spare before anything else touches the bar
    # the tips meet the dropped pin's end face, and nothing touches the sheet
    dropped_axis = g.AXIS_Z - (g.HOLE - g.PEG_D) / 2
    front = pts[depth > depth.max() - 0.001]
    assert np.all(np.abs(front[:, 2] - dropped_axis) < g.PEG_R)
    assert pts[:, 2].min() > g.SHEET_TOP + 0.002


def test_the_held_pin_goes_in_past_its_centre_of_mass_with_the_pads_clear_of_the_bar():
    assert demo.INSERT_TO > g.PEG_LEN / 2 + 0.003
    pads_to_face = g.PEG_LEN - demo.INSERT_TO - demo.GRIP_LEN
    assert pads_to_face >= 0.01


def test_the_holding_pads_grip_the_solid_end_and_clear_the_pusher():
    near, far = demo.HOLD_Y - demo.PAD_HALF_W, demo.HOLD_Y + demo.PAD_HALF_W
    assert near > g.HOLE_DEPTH and far < g.BAR_LEN
    # the holder's finger bodies reach 3.4 cm either side of its tool point; the pusher stays outside the mouth
    assert demo.HOLD_Y - 0.034 > 0.05
