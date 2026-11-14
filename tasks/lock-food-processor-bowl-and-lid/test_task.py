"""Offline checks (no Isaac Sim): sizes and layout, the two bayonets, the detents and the interlock in numbers, every
scoring level and violation, and the demo (reach, joint limits, clearances, timing, end state).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402

EYE = np.eye(3)
PLAN = demo.plan()
STEPS = PLAN[:-1]
FAR = {"left": {"tcp": np.array([0.1, 0.35, 1.2]), "rot": EYE, "opening": 0.094},
       "right": {"tcp": np.array([0.1, -0.35, 1.2]), "rot": EYE, "opening": 0.094}}
BASE = g.base_origin()
Z3 = np.zeros(3)
REST = {"rod": 0.0, "lock_detent": 0.0, "lid_detent": 0.0}
PRESSED = {"rod": -0.007, "lock_detent": 0.0, "lid_detent": 0.0}


def bowl_pose(turn=0.0, dz=0.0, dx=0.0, tilt=0.0):
    return g.seated_bowl_origin() + [dx, 0.0, dz], g.rot_z(turn * g.DEG) @ g.rot_y(tilt)


def lid_pose(bowl, turn=0.0, dz=0.0):
    bp, br = bowl
    return bp + br @ np.array([0.0, 0.0, g.BOWL_H + dz]), br @ g.rot_z(turn * g.DEG)


def start_poses():
    b = (np.array([g.BOWL0[0], g.BOWL0[1], g.bowl_start_z()]), g.rot_z(g.BOWL0_YAW * g.DEG))
    lid = (np.array([g.LID0[0], g.LID0[1], g.lid_start_z()]), g.rot_z(g.LID0_YAW * g.DEG))
    return b, lid


def step(bowl, lid, joints=REST, arms=FAR, bv=Z3, lv=Z3):
    M.update(BASE, EYE, bowl[0], bowl[1], np.asarray(bv, float), Z3, joints, lid[0], lid[1], np.asarray(lv, float),
             Z3, arms)


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()
    yield


def boxes_world(boxes, pos, rot):
    """(centre, rotation, half sizes) boxes in the station frame from (centre, size, colour, quat) body boxes."""
    out = []
    for c, s, _, q in boxes:
        out.append((np.asarray(pos) + rot @ np.asarray(c), rot @ g.q_matrix(q), np.asarray(s) / 2))
    return out


def points_gap(points, boxes) -> float:
    best = math.inf
    for c, r, h in boxes:
        local = (points - c) @ r
        best = min(best, float(np.linalg.norm(np.maximum(np.abs(local) - h, 0.0), axis=1).min()))
    return best


# -- sizes and layout ----------------------------------------------------------------------------------------------------
def test_full_size_parts():
    assert 2 * g.BOWL_R_OUT == pytest.approx(0.180)               # 18 cm bowl
    brim_l = math.pi * g.BOWL_R_IN ** 2 * (g.BOWL_H - g.FLOOR_T) * 1000 - math.pi * g.TUBE_R[1] ** 2 * g.TUBE_TOP * 1000
    assert 2.5 < brim_l < 3.0                                    # 9-cup (2.1 l working) bowl, about 2.8 l to the brim
    assert 0.5 <= g.BOWL_MASS <= 0.6 and 0.2 <= g.LID_MASS <= 0.3
    assert (g.FEED_RADIAL, g.FEED_TANGENT) == (0.040, 0.070) and g.FEED_TOP - g.LID_T == pytest.approx(0.100)
    total = g.SEAT_Z + g.BOWL_H + g.FEED_TOP
    assert 0.33 < total < 0.37                                   # assembled height, as real 9-cup processors
    assert g.HOUSING[1][2] == g.DECK and 0.09 <= g.DECK <= 0.11


def test_poses_are_plain_floats():
    for obj in M.OBJECTS:
        assert all(type(v) is float for v in obj.pos)
        assert all(type(v) is float for v in obj.rot_xyzw)


def test_parts_rest_on_the_table_apart():
    (bp, br), (lp, lr) = start_poses()
    bowl = boxes_world(g.bowl_boxes(), bp, br)
    lid = boxes_world(g.lid_boxes(), lp, lr)
    base = boxes_world(g.base_boxes(), BASE, EYE)
    lows = [min((c - np.abs(r) @ h)[2] for c, r, h in bx) for bx in (bowl, lid)]
    assert all(g.TABLE < z < g.TABLE + 0.001 for z in lows)
    pts = np.vstack([g.box_points(c, r, h) for c, r, h in bowl])
    assert points_gap(pts, lid) > 0.02 and points_gap(pts, base) > 0.01
    pts = np.vstack([g.box_points(c, r, h) for c, r, h in lid])
    assert points_gap(pts, base) > 0.01
    stand = station.camera_stand(station.top_camera("calibrated"))["stand_foot"][2]
    assert stand[0] < 0.0


# -- the bayonets ----------------------------------------------------------------------------------------------------------
def test_lugs_drop_between_ledges_at_the_start_turn():
    assert g.lugs_clear_of_ledges(g.TURN, g.BOWL_LUG_ANGLES) >= 3.0
    assert g.lugs_clear_of_ledges(g.TURN, g.LID_LUG_ANGLES) >= 3.0
    # any lug lowered on a ledge instead holds the part far too high to count as seated
    assert g.BASE_LEDGE_Z[1] - g.LUG_Z[0] - g.SEAT_Z > M.SEAT_TOL[1] + 0.01
    assert g.RIM_LEDGE_Z[1] - g.LID_LUG_Z[0] - g.BOWL_H > M.SEAT_TOL[1] + 0.01


def test_locked_lugs_are_under_their_ledges_with_play():
    assert g.lug_under_ledge(g.STOP_TURN) and g.lug_under_ledge(M.LOCK_WIN[1])
    assert not g.lug_under_ledge(M.LOCK_WIN[1] + 0.5)
    seated_lug_top = g.SEAT_Z + g.LUG_Z[1]
    assert g.BASE_LEDGE_Z[0] - seated_lug_top == pytest.approx(0.0035)      # 3.5 mm under the ledge
    assert g.SEAT_Z + g.LUG_Z[0] - g.DECK == pytest.approx(0.0025)            # 2.5 mm over the deck
    assert g.LUG_R[0] - g.BOSS_R == pytest.approx(0.0025)                    # 2.5 mm from the boss
    assert g.SKIRT_R[0] - g.BASE_LEDGE_R[1] == pytest.approx(0.006)
    assert g.SKIRT_R[0] - g.BASE_STOP_R[1] >= 0.005 - 1e-9
    # the skirts clear the ledges and stops by more than the centring play: they cannot perch on them
    assert g.SKIRT_R[0] - g.BASE_STOP_R[1] > g.TUBE_R[0] - g.SHAFT_R + 0.002
    assert g.TUBE_R[0] - g.SHAFT_R == pytest.approx(0.002)                   # the tube centres the bowl first
    assert g.TUBE_R[0] - g.SHAFT_R < g.LUG_R[0] - g.BOSS_R
    assert g.RIM_LEDGE_Z[0] - (g.BOWL_H + g.LID_LUG_Z[1]) == pytest.approx(0.0035)
    assert g.LID_LUG_R[0] - g.BOWL_R_OUT == pytest.approx(0.0025)
    assert g.LID_SKIRT_R[0] - g.RIM_LEDGE_R[1] == pytest.approx(0.005)
    assert g.LID_SKIRT_R[0] - g.RIM_STOP_R[1] >= 0.005 - 1e-9
    assert g.LID_SKIRT_R[0] - g.RIM_STOP_R[1] > g.LID_LUG_R[0] - g.BOWL_R_OUT + 0.002


def test_lug_lengths_and_turn():
    assert 20.0 <= g.TURN <= 30.0
    assert g.STOP_TURN == pytest.approx(-0.5)
    assert g.LUG_HALF * 2 * g.DEG * g.LUG_R[1] > 0.02          # base lugs about 2.4 cm long


def test_ledge_arc_corners_stay_clear_of_the_skirt():
    for lam in g.BOWL_LUG_ANGLES:
        for c, s, _, q in g.arc_boxes(lam - 11.5, lam + 13.5, *g.BASE_LEDGE_R, *g.BASE_LEDGE_Z, g.GREY_PC, step=6.0):
            r = g.q_matrix(q)
            corners = np.array([[a, b, 0] for a in (-1, 1) for b in (-1, 1)]) * np.asarray(s) / 2
            pts = np.asarray(c) + corners @ r.T
            assert np.linalg.norm(pts[:, :2], axis=1).max() < g.SKIRT_R[0] - 0.0025


# -- detents and interlock -----------------------------------------------------------------------------------------------
def test_rim_ledges_have_lead_in_ramps():
    # a lid lug up to 4 mm higher than seated still meets the ramp, not the ledge's end, and the ramp's underside is the
    # ledge's underside where the locked lug sits
    lug_top = g.BOWL_H + g.LID_LUG_Z[1]
    assert g.RIM_LEDGE_Z[0] - lug_top + 0.004 < g.RIM_LEDGE_Z[1] - g.RIM_LEDGE_Z[0]
    assert g.rim_ramp_underside_z(-g.RIM_RAMP_LEN) == pytest.approx(g.RIM_LEDGE_Z[0])
    assert g.rim_ramp_underside_z(-1e-6) == pytest.approx(g.RIM_LEDGE_Z[1], abs=1e-5)
    c, s_, _, q = g.rim_ramp_box(0.0)
    r = g.q_matrix(q)
    corners = np.asarray(c) + (np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
                               * np.asarray(s_) / 2) @ r.T
    assert corners[:, 2].min() >= g.RIM_LEDGE_Z[0] - 0.0012 and corners[:, 2].max() <= g.RIM_LEDGE_Z[1] + 0.003
    assert corners[:, 1].max() <= 0.0012                       # it ends at the ledge's end
    # the lid's detent plunger (at the locked lug's centre) is under the flat part, well short of the ramp
    half = g.LID_DIAMOND / math.sqrt(2.0) / ((g.LID_PLUNGER_R[0] + g.LID_PLUNGER_R[1]) / 2)
    assert g.LEDGE_SPAN[1] - g.RIM_RAMP_DEG > math.degrees(half)


def test_detent_plungers_interfere_and_click_in_at_lock():
    seated_ledge_bottom = g.BASE_LEDGE_Z[0] - g.SEAT_Z
    assert g.LUG_Z[1] + g.DETENT_PROUD - seated_ledge_bottom == pytest.approx(0.0015)
    assert g.LID_LUG_Z[1] - (g.RIM_LEDGE_Z[0] - g.LID_DETENT_PROUD - g.BOWL_H) == pytest.approx(0.0015)
    assert g.DETENT_TRAVEL > 0.0015 + 0.003
    # the plungers sit under / over their ledges radially, and the notches are where they are at lock
    assert g.BASE_LEDGE_R[0] <= g.LOCK_PLUNGER_R[0] and g.LOCK_PLUNGER_R[1] <= g.BASE_LEDGE_R[1]
    assert g.LUG_R[0] <= g.LOCK_PLUNGER_R[0] and g.LOCK_PLUNGER_R[1] <= g.LUG_R[1]
    assert g.RIM_LEDGE_R[0] <= g.LID_PLUNGER_R[0] and g.LID_PLUNGER_R[1] <= g.RIM_LEDGE_R[1]
    assert g.LID_LUG_R[0] <= g.LID_PLUNGER_R[0] and g.LID_PLUNGER_R[1] <= g.LID_LUG_R[1]
    # the notch is wider than the diamond where it meets the ledge's face (1.5 mm in: 3 mm wide)
    assert 2 * g.NOTCH_HALF > 2 * 0.0015


def test_plungers_stay_inside_their_parts_height():
    lo, hi = g.all_boxes_extent(g.lock_plunger_boxes())
    assert lo[2] >= g.LUG_Z[0] and hi[2] <= g.LUG_Z[1] + g.DETENT_PROUD + 1e-9
    lo, hi = g.all_boxes_extent(g.lid_plunger_boxes())
    assert lo[2] >= g.RIM_LEDGE_Z[0] - g.LID_DETENT_PROUD - 1e-9 and hi[2] <= g.RIM_LEDGE_Z[1] + 1e-9


def test_rod_clear_of_a_shifted_lid_skirt():
    # the lid can sit off centre by its lugs' 2.5 mm play: its skirt must still miss the rod's top
    assert g.ROD_R - g.ROD_HALF - g.LID_SKIRT_R[1] > (g.LID_LUG_R[0] - g.BOWL_R_OUT) + 0.002
    assert g.HANDLE_GRIP_R[0] > g.ROD_R + g.ROD_HALF and g.TAB_R[0] < g.ROD_R - g.ROD_HALF < g.ROD_R + g.ROD_HALF < g.TAB_R[1]


def test_spring_numbers():
    assert g.ROD_PRELOAD + g.ROD_RATE * 0.007 == pytest.approx(1.42)
    assert g.DETENT_PRELOAD >= 3.0 and g.LID_DETENT_PRELOAD >= 3.0


def test_tab_presses_rod_only_when_the_lid_is_home():
    rod_top = g.ROD_TOP - g.BOWL_H            # in the lid frame of a seated lid
    # locked: the flat underside is over the rod, 7 mm under its resting top, 2 mm over the handle
    assert rod_top - g.ramp_underside_z(0.0) == pytest.approx(0.007)
    assert rod_top - g.TAB_Z[0] >= g.ROD_ACTUATE
    assert g.TAB_Z[0] - (g.HANDLE_Z[1] - g.BOWL_H) == pytest.approx(0.002)
    assert g.ROD_TRAVEL - 0.007 >= 0.002 - 1e-9
    # the ramp's leading edge starts above the rod's top, so it rides onto it
    assert g.ramp_underside_z(g.TAB_FLAT[0] - g.RAMP_LEN + 1e-6) > rod_top
    # at the drop position the tab is well clear of the rod (and of the handle)
    t_drop = g.ROD_R * math.sin(-g.TURN * g.DEG)        # the rod relative to the tab, along its tangent
    assert math.isinf(g.ramp_underside_z(t_drop))
    assert abs(t_drop) - (g.RAMP_LEN - g.TAB_FLAT[0]) - g.ROD_HALF > 0.015
    # at the end of the lock window (2.5 degrees short of locked) the rod is still pressed past the switch point
    t = g.ROD_R * math.sin(-M.LOCK_WIN[1] * g.DEG)
    assert rod_top - g.ramp_underside_z(t + g.ROD_HALF) >= g.ROD_ACTUATE


def test_rim_grip_clear_of_rim_ledges_and_handle_opposite():
    finger = math.degrees(0.020 / g.BOWL_R_OUT) + 4.0
    a = demo.RIM_GRIP_ANGLE
    for mu in g.LID_LUG_ANGLES:
        assert g.arc_gap_deg((a - finger, a + finger), (mu + g.STOP_SPAN[0], mu + g.LEDGE_SPAN[1])) > 0.0
    assert g.HANDLE_ANGLE - a == pytest.approx(180.0)        # the two grips on a diameter
    assert ((a - g.WALL_PHASE) / (360.0 / g.WALL_N)) % 1.0 == pytest.approx(0.0, abs=1e-9)
    # the handle's bar is thick enough for a full-force grip (fingers stopped 10 mm or more short of closed)
    assert 2 * g.HANDLE_HALF_W >= 0.02 and g.FEED_RADIAL >= 0.02


# -- scoring -------------------------------------------------------------------------------------------------------------
def test_nothing_done_scores_zero():
    b, lid = start_poses()
    step(b, lid)
    assert M.score(None) == 0.0 and not M.success(None)


def test_bowl_seated_scores_quarter():
    b = bowl_pose(turn=g.TURN)
    step(b, start_poses()[1])
    assert M.score(None) == 0.25


def test_bowl_on_its_ledges_is_not_seated():
    b = bowl_pose(turn=0.0, dz=0.0175)
    step(b, start_poses()[1])
    assert M.score(None) == 0.0


def test_bowl_off_centre_or_tilted_is_not_seated():
    step(bowl_pose(turn=g.TURN, dx=0.006), start_poses()[1])
    assert M.score(None) == 0.0
    step(bowl_pose(turn=g.TURN, tilt=math.radians(5)), start_poses()[1])
    assert M.score(None) == 0.0


def test_bowl_locked_scores_half():
    step(bowl_pose(turn=-0.5), start_poses()[1])
    assert M.score(None) == 0.5
    step(bowl_pose(turn=2.4), start_poses()[1])
    assert M.score(None) == 0.5
    step(bowl_pose(turn=4.0), start_poses()[1])
    assert M.score(None) == 0.25


def test_lid_seated_on_locked_bowl_scores_three_quarters():
    b = bowl_pose(turn=-0.5)
    step(b, lid_pose(b, turn=g.TURN))
    assert M.score(None) == 0.75


def test_bowl_dropped_anywhere_in_its_gap_is_not_forced():
    for turn in (24.5, 60.0, 75.0, 92.0):
        M.reset_state()
        step(bowl_pose(turn=turn), start_poses()[1])
        assert not M.violation(), turn
        assert M.score(None) == 0.25


def test_lid_dropped_anywhere_in_its_gap_is_not_forced():
    b = bowl_pose(turn=-0.5)
    for turn in (24.5, 45.0, 55.0, 62.0):
        M.reset_state()
        step(b, lid_pose(b, turn=turn))
        assert not M.violation(), turn
        assert M.score(None) == 0.75


def test_turns_read_from_each_lugs_own_stop():
    assert M.wrap(75.0, 120.0) == pytest.approx(75.0) and M.wrap(-4.0, 120.0) == pytest.approx(-4.0)
    assert M.wrap(115.0, 120.0) == pytest.approx(-5.0) and M.wrap(55.0, 90.0) == pytest.approx(55.0)
    # the widest drop positions are still inside the reading range
    assert 120.0 - 17.0 - g.LUG_HALF < 120.0 + M.WRAP_LOW and 90.0 - 17.0 - g.LUG_HALF < 90.0 + M.WRAP_LOW


def test_lid_seated_on_unlocked_bowl_scores_quarter():
    b = bowl_pose(turn=g.TURN)
    step(b, lid_pose(b, turn=g.TURN))
    assert M.score(None) == 0.25


def test_all_locked_but_held_scores_three_quarters():
    b = bowl_pose(turn=-0.5)
    lid = lid_pose(b, turn=-0.5)
    tcp = lid[0] + g.feed_tube_centre_L(0.055)
    arms = {"left": FAR["left"], "right": {"tcp": tcp, "rot": demo.grip_rot(0.0), "opening": 0.040}}
    step(b, lid, PRESSED, arms)
    assert M._STATE["ready"] and not M._STATE["clear"]
    assert M.score(None) == 0.75 and not M.success(None)


def test_success_locked_pressed_released_still():
    b = bowl_pose(turn=-0.5)
    lid = lid_pose(b, turn=-0.5)
    step(b, lid, PRESSED)
    assert M.success(None)


def test_rod_not_pressed_is_not_success():
    b = bowl_pose(turn=-0.5)
    lid = lid_pose(b, turn=-0.5)
    step(b, lid, {"rod": -0.004, "lock_detent": 0.0, "lid_detent": 0.0})
    assert not M.success(None) and M.score(None) == 0.75


def test_lid_not_turned_home_is_not_success():
    b = bowl_pose(turn=-0.5)
    lid = lid_pose(b, turn=4.0)
    step(b, lid, PRESSED)
    assert not M.success(None)


def test_bowl_locked_the_wrong_way_round_is_half_but_never_success():
    b = bowl_pose(turn=120.0 - 0.5)            # a lug in the next gap: locked, but the handle is off the switch
    lid = lid_pose(b, turn=-0.5)
    step(b, lid, PRESSED)
    assert M.score(None) == 0.75 and not M.success(None)


def test_moving_is_not_success():
    b = bowl_pose(turn=-0.5)
    lid = lid_pose(b, turn=-0.5)
    step(b, lid, PRESSED, lv=(0.0, 0.02, 0.0))
    assert not M.success(None)


def test_set_down_and_lifted_parts_are_fine():
    b, lid = start_poses()
    step((b[0] + [0, 0, 0.2], g.rot_y(0.5)), lid, bv=(0, 0, -1.5))     # carried, tilted, lowered fast
    step(b, (lid[0] + [0.0, 0.05, 0.0], lid[1]), lv=(0, 0, -2.0))       # put down hard on the table
    assert not M.violation()


def test_fell_off_table_zeroes():
    b = bowl_pose(turn=-0.5)
    lid = lid_pose(b, turn=-0.5)
    step(b, (np.array([0.3, -0.7, g.TABLE - 0.3]), EYE))
    step(b, lid, PRESSED)
    assert M.violation() and M.score(None) == 0.0 and not M.success(None)


def test_forced_past_stop_zeroes():
    step(bowl_pose(turn=-4.0), start_poses()[1])
    assert M.violation() and M.score(None) == 0.0


def test_lid_forced_past_stop_zeroes():
    b = bowl_pose(turn=-0.5)
    step(b, lid_pose(b, turn=-4.5))
    assert M.violation()


def test_sprung_part_forced_past_its_end_zeroes():
    b = bowl_pose(turn=-0.5)
    step(b, lid_pose(b, turn=-0.5), {"rod": -0.012, "lock_detent": 0.0, "lid_detent": 0.0})
    assert M.violation()


def test_end_state_counts():
    b = bowl_pose(turn=-0.5)
    step(b, start_poses()[1])
    assert M.score(None) == 0.5
    step(bowl_pose(turn=g.TURN, dz=0.05), start_poses()[1])            # lifted off again
    assert M.score(None) == 0.0


def test_judged_in_the_base_frame():
    # the same relative state with the base's pose rotated: the bowl's turn is measured relative to the base
    r = g.rot_z(0.3)
    bp, br = bowl_pose(turn=-0.5)
    M.update(BASE, r, BASE + r @ (bp - BASE), r @ br, Z3, Z3, REST, np.array([0.3, -0.27, g.lid_start_z()]), EYE,
             Z3, Z3, FAR)
    assert M.score(None) == 0.5


# -- the demo -------------------------------------------------------------------------------------------------------------
def test_demo_reach_and_limits():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        mm, deg = worst[side]
        assert mm < 0.5 and deg < 0.1
        qs = np.array([s[side][0] for s in STEPS])
        assert (qs >= ik.LOWER - 1e-9).all() and (qs <= ik.UPPER + 1e-9).all()


def test_demo_fits_run_limit():
    assert len(STEPS) + 3 * demo.CONTROL_HZ < 3600 / 2


def test_demo_reads_no_object_state():
    src = (HERE / "demo.py").read_text()
    assert "object_states" not in src and "root_link_pose" not in src and ".joints" not in src


def test_demo_turns_reach_lock():
    marks = STEPS[-1]["mark"]
    for m in ("bowl_seated", "bowl_turned", "lid_seated", "lid_turned", "left_away", "right_away"):
        assert marks.get(m)
    end = STEPS[-1]
    assert end["bowl"][1] == pytest.approx(g.STOP_TURN) and end["lid"][1] == pytest.approx(g.STOP_TURN)
    assert np.allclose(end["bowl"][0], g.seated_bowl_origin() + [0, 0, 0.0005])


def test_bowl_carried_over_the_shaft_before_it_comes_down():
    for s in STEPS:
        o, yaw = s["bowl"]
        r = math.hypot(o[0] - g.BASE_XY[0], o[1] - g.BASE_XY[1])
        if 0.002 < r < 0.15:
            assert o[2] + g.SKIRT_Z[0] - g.TABLE > g.SHAFT_TOP + g.SHAFT_CONE + 0.008     # its foot over the tip
        if r < 0.15 and o[2] - g.TABLE < g.BASE_LEDGE_Z[1] - g.LUG_Z[0] + 0.001:     # lugs down at the ledges
            assert r < 0.001
            assert g.lugs_clear_of_ledges(yaw, g.BOWL_LUG_ANGLES) >= 3.0 or g.lug_under_ledge(yaw) \
                or s["mark"].get("bowl_seated")


def test_lid_carried_over_the_rim_before_it_comes_down():
    rim = g.TABLE + g.SEAT_Z + g.BOWL_H
    for s in STEPS:
        o, _ = s["lid"]
        r = math.hypot(o[0] - g.BASE_XY[0], o[1] - g.BASE_XY[1])
        if 0.002 < r < 0.25 and s["holding"]["lid"]:
            assert o[2] + g.LID_SKIRT_Z[0] > rim + 0.008


def test_grippers_clear_of_parts_except_where_they_grip():
    """Every gripper box (fingers, carriage) stays clear of the base, and of the bowl and the lid except the wall panel
    or feed tube walls it pinches; the arms keep 3 cm apart."""
    base = boxes_world(g.base_boxes(), BASE, EYE)
    for i, s in enumerate(STEPS[::3]):
        bo, by = s["bowl"]
        lo, ly = s["lid"]
        bowl = boxes_world(g.bowl_boxes(), bo, g.rot_z(by * g.DEG))
        lid = boxes_world(g.lid_boxes(), lo, g.rot_z(ly * g.DEG))
        grips = {}
        for side in ("left", "right"):
            q, grip, pos, rot = s[side]
            opening = 0.094 * grip + 0.003 * (1 - grip)
            grips[side] = M.gripper_boxes(pos, rot, opening)
            carriage = g.box_points(*grips[side][4])
            roots = np.vstack([g.box_points(*b) for b in grips[side][1:4:2]])
            assert points_gap(carriage, base + bowl + lid) > 0.004, (i, side)
            # the finger roots may close on the pinched wall too (the bowl's wall panels come first in its list, the
            # feed tube's walls are the lid's only boxes above its top)
            lid_low = [b for b, src in zip(lid, g.lid_boxes()) if src[0][2] < g.LID_T]
            held = bowl[g.WALL_N:-3] if side == "left" else bowl[:-3]   # the left pinches the wall, the right the handle
            assert points_gap(roots, base + held + lid_low) > 0.004, (i, side)
            pads = np.vstack([g.box_points(*b) for b in grips[side][0:4:2]])
            assert points_gap(pads, base) > 0.01, (i, side)
        pl = np.vstack([g.box_points(*b) for b in grips["left"]])
        pr = grips["right"]
        assert points_gap(pl, pr) > 0.03


def test_open_fingers_straddle_wall_and_tube_with_room():
    # at the grasp the open pads are at least 2 cm either side of the bowl's wall and of the feed tube's walls
    assert 0.094 / 2 - 0.005 - (g.BOWL_R_OUT - g.BOWL_R_IN) / 2 > 0.04
    assert (0.094 - g.FEED_RADIAL) / 2 > 0.02


def test_demo_ends_released_and_still_parts_score_full():
    end = STEPS[-1]
    bo, by = end["bowl"]
    lo, ly = end["lid"]
    arms = {}
    for side in ("left", "right"):
        q, grip, pos, rot = end[side]
        arms[side] = {"tcp": pos, "rot": rot, "opening": 0.094 * grip}
    b = (bo, g.rot_z(by * g.DEG))
    lid = (bo + [0, 0, g.BOWL_H], g.rot_z(ly * g.DEG))
    step(b, lid, PRESSED, arms)
    assert M.success(None)
