"""Offline checks (no Isaac Sim): the case and cover geometry, the scoring and its violations, and that the scripted
demo is reachable and never trips the monitor.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py      # from this folder
"""

from pathlib import Path
from types import SimpleNamespace

import math

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
g = M.g
DT = 1 / 20
CASE0 = np.array([g.CASE_XY[0], g.CASE_XY[1], g.TOP])
FAR = [np.full((14, 3), 5.0), np.full((14, 3), 5.0)]   # grippers far away


def body(pos, vel=(0, 0, 0), quat=(0, 0, 0, 1), spin=(0, 0, 0)):
    return SimpleNamespace(pos=np.asarray(pos, float), quat_xyzw=np.asarray(quat, float),
                           lin_vel=np.asarray(vel, float), ang_vel=np.asarray(spin, float))


def run(cover_seq, latch_q=0.0, grippers=FAR, case_seq=None):
    """Feed a sequence of cover origins (or bodies) through the monitor; return (score, success)."""
    M.reset_state()
    for i, c in enumerate(cover_seq):
        cover = c if isinstance(c, SimpleNamespace) else body(c)
        case = case_seq[i] if case_seq else body(CASE0)
        M.update(case, cover, latch_q, grippers)
    return M.score({}), M.success({})


def over(s, up=0.0):
    return M.cover_origin(s, up)


# -- loading and geometry --------------------------------------------------------------------------------------------
def test_it_loads_with_case_and_cover_loose():
    assert set(TASK.free_objects) == {"pc_case", "pc_cover"}
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0


def test_real_small_form_factor_sizes():
    assert 2 * g.HX == pytest.approx(0.292) and 2 * g.HY == pytest.approx(0.290) and g.RIM == pytest.approx(0.094)
    assert 0.5 <= g.COVER_MASS <= 0.8 and 4.0 <= g.CASE_MASS <= 6.0


def test_hooks_drop_through_slots_only_in_their_window_and_reach_under_the_flange_when_slid():
    for hx, (a, b) in zip(g.HOOK_X, g.slots()):
        for s, fits in ((g.DROP_MIN, True), (g.DROP_MAX, True), (g.DROP_MIN - 0.001, False),
                        (g.DROP_MAX + 0.001, False), (g.START_S, True), (g.START_S - 0.0035, True),
                        (g.START_S + 0.006, True)):
            lo, hi = hx + s - g.SLIDE, hx + s - g.SLIDE + g.TOE_LEN
            assert (lo >= a - 1e-9 and hi <= b + 1e-9) == fits, (s, fits)
        # fitted: stem 1 mm short of the slot's front end, toe 8 mm under the flange
        assert b - (hx + g.STEM_T) == pytest.approx(0.001)
        assert hx + g.TOE_LEN - b == pytest.approx(0.008)
        # 2 mm either side across the slot
        assert (g.SLOT_W - g.HOOK_W) / 2 == pytest.approx(g.SLOT_PLAY)
    assert g.HOOK_DROP == pytest.approx(0.004)


def test_latch_hole_is_over_the_plunger_only_at_the_end_of_the_slide():
    x = g.LATCH_XY[0]
    for s in np.arange(g.DROP_MIN, g.STOP_S + 1e-9, 0.0005):
        hole = (x - g.HOLE / 2 + s - g.SLIDE, x + g.HOLE / 2 + s - g.SLIDE)
        pops = hole[0] <= x - g.LATCH_SIDE / 2 + 1e-9 and hole[1] >= x + g.LATCH_SIDE / 2 - 1e-9
        assert pops == (g.LATCH_S[0] - 1e-9 <= s <= g.LATCH_S[1] + 1e-9), s
    assert g.LATCH_S[0] > g.ENGAGED_S and g.STOP_S > g.LATCH_S[0]


def test_the_plunger_stays_clear_of_the_hooks_and_the_cover_weight_beats_its_spring():
    for hx in g.HOOK_X:
        toe = (hx + g.DROP_MIN - g.SLIDE, hx + g.STOP_S - g.SLIDE + g.TOE_LEN)
        assert toe[1] < g.LATCH_XY[0] - g.LATCH_SIDE / 2 - 0.01 or toe[0] > g.LATCH_XY[0] + g.LATCH_SIDE / 2 + 0.01
    assert g.LATCH_RATE * g.LATCH_UP < 0.4 * g.COVER_MASS * 9.81
    assert g.LATCH_UP < g.LATCH_TRAVEL     # the cover can press it flush


def test_inside_parts_stay_below_the_hooks_and_the_rim():
    lowest_hook = g.RIM - g.HOOK_DROP
    for (cx, cy, cz), (sx, sy, sz), _ in g.case_boxes()[9:]:
        if sz > 0.005 and abs(cx) < g.HX - 0.02 and abs(cy) < g.HY - 0.02:   # parts inside, not the rim
            assert cz + sz / 2 < lowest_hook - 0.005, (cx, cy)


def test_cover_starts_beside_the_case_on_its_lips_and_clears_it_while_carried():
    assert g.COVER_START_Y - g.COVER_HY - (g.CASE_XY[1] + g.HY) > 0.02
    assert g.COVER_START[2] == pytest.approx(g.TOP + g.LIP_FOOT) and g.LIP_FOOT < g.LIP_DEPTH + 0.001
    assert M.CARRY_UP - g.LIP_DEPTH > 0.02
    assert g.LIP_Y - g.HY == pytest.approx(0.0015)


def test_lips_guide_the_hooks_into_their_slots_across():
    # with both lips outside the walls (1.5 mm gap) and the cover square within the lips' play, every hook is within
    # its slot's side play (the lips, not the hooks, take any side load)
    worst_yaw = math.atan(2 * g.LIP_GAP / (2 * g.HX))
    for dy in np.linspace(-g.LIP_GAP, g.LIP_GAP, 5):
        yaw_room = (g.LIP_GAP - abs(dy)) / g.HX
        for hx in g.HOOK_X:
            assert abs(dy) + min(worst_yaw, yaw_room) * abs(hx + g.TOE_LEN) <= (g.SLOT_W - g.HOOK_W) / 2 + 1e-9
    # the flared feet catch a cover up to 3 mm off across and steer it in
    assert g.FLARE >= 0.003


# -- scoring ------------------------------------------------------------------------------------------------------------
def test_full_credit_needs_latched_released_and_still():
    assert run([over(0.015)], latch_q=0.0) == (1.0, True)
    assert run([over(0.015)], latch_q=-0.0035) == (0.5, False)                    # latch still pressed down
    assert run([body(over(0.015), vel=(0, 0, 0.02))], latch_q=0.0) == (0.75, False)  # still moving


def test_holding_the_latched_cover_is_not_full_credit():
    grips = M.gripper_points(over(0.015) + g.grip_tcp_offset(), ik_rot(), 0.030)
    assert run([over(0.015)], latch_q=0.0, grippers=[grips, FAR[1]]) == (0.75, False)


def test_partial_credit_levels():
    assert run([over(0.002)], latch_q=-0.003)[0] == 0.25          # dropped in, not slid
    assert run([over(0.011)], latch_q=-0.003)[0] == 0.5           # slid, hooks under the flange, latch not up
    assert run([over(0.002, up=0.004)])[0] == 0.0                  # resting on its hooks beside the slots
    assert run([over(0.015, up=0.004)], latch_q=0.0)[0] == 0.0     # set down already slid: hooks on the flange
    assert run([g.COVER_START])[0] == 0.0                          # doing nothing
    assert run([over(0.002 - 0.090)], latch_q=-0.003)[0] == 0.0    # hooks dropped into the next slots back


def test_a_tilted_or_skewed_cover_is_not_seated():
    tilt = (np.sin(np.radians(1.5) / 2), 0, 0, np.cos(np.radians(1.5) / 2))  # 1.5 deg about x: corners 3.8 mm up
    assert run([body(over(0.002), quat=tilt)])[0] == 0.0
    yaw = g.yaw_quat(np.radians(3))
    assert run([body(over(0.002), quat=yaw)])[0] == 0.0
    shifted = over(0.002) + np.array([0, 0.0025, 0])
    assert run([shifted])[0] == 0.0


def test_violations_zero_the_whole_trial_even_after_recovering():
    fitted = over(0.015)
    # dropped (falling at 0.6 m/s), then fitted
    assert run([body(over(0.0, 0.05), vel=(0, 0, -0.6)), fitted]) == (0.0, False)
    # lowered briskly but under the threshold is fine
    assert run([body(over(0.0, 0.01), vel=(0, 0, -0.3)), fitted]) == (1.0, True)
    # case pushed 1.5 cm, then the cover latched
    moved = [body(CASE0 + [0.015, 0, 0]), body(CASE0)]
    assert run([fitted, fitted], case_seq=moved) == (0.0, False)
    # case turned 3 degrees
    turned = [body(CASE0, quat=g.yaw_quat(np.radians(3))), body(CASE0)]
    assert run([fitted, fitted], case_seq=turned)[0] == 0.0
    # a gripper reached into the open case (over the motherboard)
    inside = M.gripper_points(CASE0 + [0.0, 0.0, 0.07], ik_rot(), 0.0)
    assert run([g.COVER_START, fitted], grippers=[inside, FAR[1]])[0] == 0.0
    # cover knocked off the table
    assert run([body((0.3, 0.6, g.TOP - 0.2)), fitted])[0] == 0.0
    # cover tipped into the case
    tipped = body(over(0.0, -0.02), quat=(np.sin(np.radians(30) / 2), 0, 0, np.cos(np.radians(30) / 2)))
    assert run([tipped, fitted])[0] == 0.0
    assert M._STATE["why"]


def ik_rot(yaw=None):
    from yam_arena.demo import grip_rotation
    return grip_rotation(g.YAW["left"] if yaw is None else yaw)


# -- the demo -----------------------------------------------------------------------------------------------------------
def _play(side):
    plan = M.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    rows = []
    for k in range(round(arm.duration / DT) + 5):
        a, label = arm.action7((k + 1) * DT)
        tcp, rot = ik.tcp_pose(a[:6], station.BASES[side])
        rows.append((a, label, tcp, rot))
    return arm, rows


def test_the_demo_is_reachable_within_joint_limits():
    for side in ("left", "right"):
        arm, rows = _play(side)
        acts = np.array([r[0] for r in rows])
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(acts[:, :6] >= ik.LOWER) and np.all(acts[:, :6] <= ik.UPPER), side
        assert arm.duration < 30


def test_the_two_grips_move_as_one_rigid_pair_and_keep_their_cameras_apart():
    (_, left), (_, right) = _play("left"), _play("right")
    for (al, ll, tl, rl), (ar, lr, tr, rr) in zip(left, right):
        assert ll == lr
        if ll in ("pinch the grip", "lift", "carry over the case", "settle", "lower: hooks into slots", "seated",
                  "slide to the latch", "latched"):
            assert np.allclose(tl - tr, [0, 2 * g.GRASP_DY, 0], atol=0.0005)
        cam_l = tl - rl @ [0, 0, 0.134] + rl @ [0, -0.08, 0]
        cam_r = tr - rr @ [0, 0, 0.134] + rr @ [0, -0.08, 0]
        assert np.linalg.norm(cam_l - cam_r) > 0.07


def test_the_demo_never_trips_the_monitor_and_ends_released():
    """Replay the plan: the cover follows the grips rigidly while they pinch it (from 'lift' to 'latched'), resting
    on the rim when pressed and stopped by the hooks' stems, and the latch pops up once the slide reaches it. Check the
    monitor's verdict at every step and the stages on the way."""
    (_, left), (_, right) = _play("left"), _play("right")
    off = g.grip_tcp_offset() + np.array([0, g.GRASP_DY, 0])
    cover = np.asarray(g.COVER_START, float)
    held = {"lift", "carry over the case", "settle", "lower: hooks into slots", "seated", "push to the latch",
            "latched"}
    pinching = held | {"pinch the grip"}
    x0 = g.CASE_XY[0] - g.SLIDE
    M.reset_state()
    prev, stages = cover.copy(), set()
    for (al, label, tl, rl), (ar, _, tr, rr) in zip(left, right):
        if label in held:
            cover = tl - off
            cover[2] = max(cover[2], g.TOP + g.RIM)
            cover[0] = min(cover[0], x0 + g.STOP_S)
        s = cover[0] - x0
        latch_q = 0.0 if g.LATCH_S[0] <= s <= g.LATCH_S[1] else -g.LATCH_UP
        grip_w = g.GRIP["x1"] - g.GRIP["x0"]
        widths = [grip_w if label in pinching else a[6] * 2 * abs(FINGER_OPEN) for a in (al, ar)]
        vel = (cover - prev) / DT
        prev = cover.copy()
        grips = [M.gripper_points(tl, rl, widths[0]), M.gripper_points(tr, rr, widths[1])]
        M.update(body(CASE0), body(cover, vel=vel), latch_q, grips)
        assert not M._STATE["violation"], (label, M._STATE["why"])
        stages.add(M._STATE["stage"])
    assert {"seated", "engaged", "latched"} <= stages
    assert M._STATE["stage"] == "latched" and M._STATE["released"]
    assert M.score({}) == 1.0


def test_pads_stay_above_the_plate_and_rim_while_gripping():
    tip_above_plate = g.PAD_ABOVE_PLATE
    assert tip_above_plate >= 0.005
    assert g.grip_tcp_offset()[2] - PAD_TIP_PAST_TCP == pytest.approx(g.PLATE_T + g.PAD_ABOVE_PLATE)
    # open fingers clear the 16 mm grip by 3.5 cm on each side; closed on it, each finger is 8 mm short of its
    # target, so it presses with 40 N, under the 50 N limit: the grip stays centred between the fingers
    grip_depth = g.GRIP["x1"] - g.GRIP["x0"]
    assert (abs(FINGER_OPEN) * 2 - grip_depth) / 2 > 0.035
    assert 5000.0 * grip_depth / 2 < 50.0


def test_the_plan_leaves_time_for_the_success_hold_and_fits_the_run_limit():
    plan = M.demo()["left"]
    t = 0.0
    for w in plan:
        t += w.duration
        if w.label == "let go":
            released_at = t
    assert plan[-1].label == "back off" and t - released_at >= 2.0
    assert (t + 3.0) / DT < 2400
