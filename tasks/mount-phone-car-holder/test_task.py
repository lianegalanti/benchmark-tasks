"""Offline checks (no Isaac Sim): the layout, the holder's mechanism (catch, chamfer, trigger travel, clearances), every
scoring level and violation, and the demo (reach, joint limits, clearances, end state).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402

EYE = np.eye(3)
PLAN = demo.plan()
STEPS = PLAN[:-1]
FAR = {"left": {"tcp": np.array([0.2, 0.3, 1.0]), "rot": EYE, "opening": 0.094},
       "right": {"tcp": np.array([0.2, -0.3, 1.0]), "rot": EYE, "opening": 0.094}}
DASH = g.lohi((g.DASH_X, -g.DASH_W / 2, g.TABLE), (g.DASH_X + g.DASH_D, g.DASH_W / 2, g.TABLE + g.DASH_H))


def root_boxes():
    return [g.lohi(lo, hi, g.HOLDER) for lo, hi in [g.PLATE, *g.FEET, g.HOUSING, g.CLIP]]


def jaws_station(dl, dr):
    return [(c + g.HOLDER, r, h) for c, r, h in g.jaw_boxes(dl, dr)]


def cradle_phone(back=-0.001, bottom=0.0, y=0.0, rot=EYE):
    c = np.array([back - g.PHONE_T / 2, y, bottom + g.PHONE_L / 2])
    return g.to_station(c), rot


def step(phone, joints, arms=FAR, vel=(0, 0, 0), w=(0, 0, 0)):
    M.update(g.HOLDER, EYE, joints, phone[0], phone[1], np.asarray(vel, float), np.asarray(w, float), arms)


def joints(dl, dr=None, trig=0.0):
    return {"arm_left": dl, "arm_right": dl if dr is None else dr, "trigger": trig}


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()
    yield


# -- layout ---------------------------------------------------------------------------------------------------------
def test_phone_and_holder_full_size():
    assert (g.PHONE_L, g.PHONE_W, g.PHONE_T, g.PHONE_MASS) == (0.1476, 0.0716, 0.0078, 0.171)
    assert round(2 * g.CLOSED_HALF, 4) == 0.058 and round(2 * (g.CLOSED_HALF + g.LATCHED), 4) == 0.090
    assert 2 * g.CLOSED_HALF < g.PHONE_W < 2 * (g.CLOSED_HALF + g.LATCHED)


def test_phone_starts_leaning_on_dash_edge():
    c, r = g.phone_start(), g.PHONE_R0
    corners = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)]) * g.PHONE_HALF
    pts = c + corners @ r.T
    assert pts[:, 2].min() == pytest.approx(g.TABLE + 0.0003, abs=1e-6)
    # its back face is 0.3 mm off the dash's top front edge
    n = r @ np.array([1.0, 0, 0])
    edge = np.array([g.DASH_X, c[1], g.TABLE + g.DASH_H])
    assert (edge - c) @ n == pytest.approx(g.PHONE_T / 2 + 0.0003, abs=1e-6)
    # stable: the centre of mass lies between the bottom edge and the dash edge (horizontally)
    bottom_x = pts[pts[:, 2] < g.TABLE + 0.001, 0].mean()
    assert bottom_x < c[0] < g.DASH_X - 0.003
    # nothing but air under the phone's top part, and it is clear of the holder
    assert pts[:, 0].min() > g.HOLDER[0] + 0.0 and abs(c[1]) - g.PHONE_W / 2 > 0.07


def test_holder_clear_of_table_and_dash():
    for c, _, h in root_boxes():
        assert c[2] - h[2] > g.TABLE + 0.015
        assert c[0] + h[0] <= g.DASH_X + 1e-9


# -- mechanism ------------------------------------------------------------------------------------------------------
def test_catch_holds_open_arm_and_trigger_releases_it():
    assert g.catch_blocks(g.LATCHED, 0.0)
    assert g.catch_blocks(g.LATCHED, g.RELEASE - 0.0005)
    assert not g.catch_blocks(g.LATCHED, g.RELEASE)
    # the arm stop lies past the latched position, so the tail clears the catch's face with 2 mm to spare
    assert g.ARM_STOP - g.LATCHED >= 0.002 - 1e-9


def test_chamfer_pushes_catch_in_as_tail_passes():
    # tail's outer corner (x = TAIL_X[1]) meets the chamfer partway up it, so the contact is on the slope
    y_hit = g.C_IN + (g.C_X[0] + g.CHAMFER - g.TAIL_X[1])
    assert g.C_IN < y_hit < g.C_IN + g.CHAMFER
    # riding up the chamfer the trigger has to go in by the overlap before the corner clears the catch
    assert g.chamfer_x(g.C_IN + g.CHAMFER, g.OVERLAP) == pytest.approx(g.TAIL_X[1])
    assert g.OVERLAP < g.TRIG_STOP


def test_catch_clear_of_tail_when_arms_grip_phone():
    _, t_out = g.tail_y(g.PHONE_D)
    assert g.C_IN - t_out >= 0.0015


def test_trigger_travel_when_phone_pressed():
    back = -g.TRIG_PROUD - demo.BACK_CLEAR + demo.PRESS      # phone back's commanded x after pressing
    assert 0.0 < back <= 0.003                               # at most 3 mm into the plate: the plate stops it
    # the trigger is then fully in; even 3 mm of arm lag leaves it 3 mm in, at the release point
    assert min(g.TRIG_STOP, g.TRIG_PROUD + back - 0.003) >= g.RELEASE + 0.0005


def test_phone_clears_open_arms_and_trigger():
    # open (latched), each lip is 6 mm clear of the phone's edge; the phone comes in 2.5 mm clear of the trigger pad
    assert g.CLOSED_HALF - 0.003 + g.LATCHED - g.PHONE_W / 2 >= 0.0025
    assert demo.BACK_CLEAR >= 0.0025
    # pressed, the phone's front lies behind the arm lips, so the lips go round its front face
    assert -0.001 - g.PHONE_T > -0.012


def test_spring_numbers():
    clamp = g.ARM_PRELOAD + g.ARM_RATE * g.PHONE_D
    assert 1.2 < clamp < 2.5
    assert 2 * 0.75 * clamp > g.PHONE_MASS * 9.81         # the pads alone can carry the phone
    trig = g.TRIG_PRELOAD + g.TRIG_RATE * g.RELEASE
    assert 2.0 < trig < 3.5


def test_links_do_not_overlap():
    arm = [b for _, lo, hi in g.ARM_BOXES for b in [g.lohi(lo, hi)]]
    mirror = [g.lohi((lo[0], -hi[1], lo[2]), (hi[0], -lo[1], hi[2])) for _, lo, hi in g.ARM_BOXES]
    trig = [g.lohi(lo, hi) for _, lo, hi in g.TRIGGER_BOXES]
    for d in (0.0, g.PHONE_D, g.LATCHED, g.ARM_STOP):
        a = [(c + [0, d, 0], r, h) for c, r, h in arm]
        b = [(c - [0, d, 0], r, h) for c, r, h in mirror]
        assert min(g.boxes_gap(x, y) for x in a for y in b) > 0.01
        for q in (0.0, g.TRIG_STOP):
            t = [(c + [q, 0, 0], r, h) for c, r, h in trig]
            assert min(g.boxes_gap(x, y) for x in a + b for y in t) > 0.0005


# -- scoring --------------------------------------------------------------------------------------------------------
def test_nothing_done_scores_zero():
    for _ in range(5):
        step((g.phone_start(), g.PHONE_R0), joints(0.0))
    assert M.score({}) == 0.0 and not M.success({})


def test_cocked_scores_quarter():
    step((g.phone_start(), g.PHONE_R0), joints(g.LATCHED))
    assert M.score({}) == 0.25


def test_cocked_held_open_by_gripper_does_not_count():
    arms = dict(FAR, left={"tcp": g.to_station((-0.026, 0.034, 0.07)), "rot": demo.grip_rot(), "opening": 0.0})
    step((g.phone_start(), g.PHONE_R0), joints(g.LATCHED), arms)
    assert M.score({}) == 0.0


def test_placed_scores_half():
    step(cradle_phone(back=-0.0085), joints(g.LATCHED))
    assert M.score({}) == 0.5


def test_placed_off_feet_or_tilted_does_not_count():
    step(cradle_phone(back=-0.0085, bottom=0.01), joints(g.LATCHED))
    step(cradle_phone(back=-0.0085, rot=g.rot_y(math.radians(15))), joints(g.LATCHED))
    assert M.score({}) == 0.25


def holding_arms(phone):
    tcp, rot = demo.tcp_for_phone(phone[0], phone[1])
    return dict(FAR, right={"tcp": tcp, "rot": rot, "opening": g.PHONE_W})


def test_clamped_but_held_scores_three_quarters():
    ph = cradle_phone()
    for _ in range(3):
        step(ph, joints(g.PHONE_D), holding_arms(ph))
    assert M.score({}) == 0.75 and not M.success({})


def test_success_released_and_still():
    ph = cradle_phone(back=-0.0042)
    step(ph, joints(g.PHONE_D))
    step(ph, joints(g.PHONE_D))
    assert M.success({}) and M.score({}) == 0.75


def test_phone_gripped_above_feet_still_counts():
    ph = cradle_phone(back=-0.0042, bottom=0.006)        # the pads hold it 6 mm above the feet
    step(ph, joints(g.PHONE_D))
    step(ph, joints(g.PHONE_D))
    assert M.success({})


def test_moving_phone_is_not_success():
    ph = cradle_phone()
    step(ph, joints(g.PHONE_D))
    step(ph, joints(g.PHONE_D), vel=(0, 0, 0.05))
    assert not M.success({})


def test_off_centre_or_unclamped_is_not_success():
    for ph, j in ((cradle_phone(y=0.004), joints(g.PHONE_D + 0.004, g.PHONE_D - 0.004)),
                  (cradle_phone(), joints(g.LATCHED)), (cradle_phone(bottom=0.012), joints(g.PHONE_D))):
        M.reset_state()
        step(ph, j)
        step(ph, j)
        assert not M.success({})


def test_drop_zeroes_everything():
    step(cradle_phone(), joints(g.PHONE_D))
    step(cradle_phone(), joints(g.PHONE_D), vel=(0, 0, -0.8))
    step(cradle_phone(), joints(g.PHONE_D))
    assert M.score({}) == 0.0 and not M.success({}) and M._STATE["why"] == ["the phone was dropped"]


def test_fast_move_while_held_is_not_a_drop():
    ph = cradle_phone(back=-0.04, bottom=0.05)
    step(ph, joints(g.LATCHED), holding_arms(ph), vel=(0, 0, -0.8))
    assert not M.violation()


def test_phone_off_table_zeroes():
    step((np.array([0.3, -0.5, g.TABLE - 0.05]), EYE), joints(0.0))
    assert M.violation()


def test_arm_closed_on_screen_zeroes():
    # the phone held out from the plate when the trigger fires: the lips close onto its glass (arm travel 9.8 mm)
    ph = cradle_phone(back=-0.0085)
    glass = g.PHONE_W / 2 - (g.CLOSED_HALF - 0.003)
    for d in (g.LATCHED, glass, glass):
        step(ph, joints(d))
    assert M.violation()
    M.reset_state()
    # the phone turned edge-on to the robot: the released left arm stops on its face
    d = 0.002
    face = (g.to_station((-0.014, g.CLOSED_HALF - 0.003 + d - g.PHONE_T / 2, g.PHONE_L / 2)), g.rot_z(math.pi / 2))
    for dd in (g.LATCHED, d, d):
        step(face, joints(dd, g.LATCHED))
    assert M.violation()


def test_touching_cocked_or_closed_arm_with_tilted_phone_is_fine():
    # bringing the phone in leaning back 25 degrees, it brushes the cocked arms' lips: not a violation
    ph = (g.to_station((-0.012, 0.0, g.PHONE_L / 2)), g.rot_y(math.radians(25)) @ g.rot_z(math.radians(25)))
    for _ in range(3):
        step(ph, joints(g.LATCHED))
    assert not M.violation()
    # and with the arms closed and empty from the start (never released while the phone was there)
    M.reset_state()
    for _ in range(3):
        step(ph, joints(0.0))
    assert not M.violation()


def test_clean_clamp_is_not_a_face_violation():
    ph = cradle_phone()
    for d in (g.LATCHED, 0.010, g.PHONE_D, g.PHONE_D):
        step(ph, joints(d))
    assert not M.violation() and M.score({}) == 0.75


def test_end_state_counts():
    # clamped, then pulled out and laid back against the dash: no credit kept
    step(cradle_phone(), joints(g.PHONE_D), holding_arms(cradle_phone()))
    assert M.score({}) == 0.75
    step((g.phone_start(), g.PHONE_R0), joints(0.0))
    assert M.score({}) == 0.0
    M.reset_state()
    # cocked, then fired empty: no credit
    step((g.phone_start(), g.PHONE_R0), joints(g.LATCHED))
    assert M.score({}) == 0.25
    step((g.phone_start(), g.PHONE_R0), joints(0.0))
    assert M.score({}) == 0.0


def test_holder_frame_used():
    # a holder measured 2 cm to the left (it can't move, but the scorer works in its own frame)
    shift = np.array([0.0, 0.02, 0.0])
    ph = cradle_phone()
    M.update(g.HOLDER + shift, EYE, joints(g.PHONE_D), ph[0] + shift, ph[1], np.zeros(3), np.zeros(3), FAR)
    assert M._STATE["clamped"]


# -- demo -----------------------------------------------------------------------------------------------------------
def test_demo_reach_and_limits():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert worst[side][0] < 0.5 and worst[side][1] < 0.1
        for st in STEPS:
            q = st[side][0]
            assert np.all(q >= ik.LOWER - 1e-9) and np.all(q <= ik.UPPER + 1e-9)


def test_demo_fits_run_limit():
    assert len(STEPS) + 3 * demo.CONTROL_HZ < 2400


def grips(st, side):
    _, gr, pos, rot = st[side]
    return M.gripper_boxes(pos, rot, 0.094 * gr)


def test_grippers_clear_dash_holder_and_each_other():
    for st in STEPS:
        for side in ("left", "right"):
            b = grips(st, side)
            assert M.gap_to(b, [DASH]) > 0.008, side
            assert M.gap_to(b, root_boxes()) > 0.004, side
        assert M.gap_to(grips(st, "left"), grips(st, "right")) > 0.02


def test_cocking_push_reaches_stop_and_backs_off():
    lip = g.CLOSED_HALF - 0.003
    pushes = [st["left"][2] - g.HOLDER for st in STEPS if st["mark"].get("push_+1") and not st["mark"].get("pushed_+1")]
    far = max(p[1] for p in pushes)
    assert far + demo.FINGER_HALF == pytest.approx(lip + g.ARM_STOP + demo.PUSH_OVER)
    # the fingertips are in the arms' band, clear of the trigger pad and the plate
    p = pushes[-1]
    assert g.JAW_Z[0] < p[2] - 0.013 < g.JAW_Z[1]
    assert p[2] - 0.013 > 0.050 + 0.005
    assert M.gap_to(M.gripper_boxes(p + g.HOLDER, demo.grip_rot(), 0.0), root_boxes()) > 0.004


def test_phone_path_clears_open_arms():
    # while the phone comes in and down (before it is pressed), it stays 2.5 mm clear of the open arms
    for st in STEPS:
        if st["holding"] and not st["mark"].get("press") and st["mark"].get("pushed_-1"):
            c, r = st["phone"]
            gap = min(g.boxes_gap(j, g.phone_box(c, r)) for j in jaws_station(g.LATCHED, g.LATCHED))
            assert gap >= 0.0024


def test_phone_pressed_and_released_pose_scores():
    pressed = [st for st in STEPS if st["mark"].get("press") and st["holding"]][-1]
    c, r = pressed["phone"]
    p, rr = M.phone_in_holder(c, r, g.HOLDER, EYE)
    p = np.array([min(p[0], -g.PHONE_T / 2), p[1], p[2]])     # the back plate stops it
    assert M.clamped(p, rr, g.PHONE_D, g.PHONE_D)
    assert M.tilt(rr) < 1e-6


def test_demo_ends_released():
    end = STEPS[-1]
    c, r = [st for st in STEPS if st["mark"].get("press") and st["holding"]][-1]["phone"]
    for side in ("left", "right"):
        assert M.gap_to(grips(end, side), [g.phone_box(c, r)] + jaws_station(g.PHONE_D, g.PHONE_D)) > M.CLEAR
    # and right after releasing, as the fingers slide up, they never press down on the phone's top
    rel = [st for st in STEPS if st["mark"].get("release")]
    top = c[2] + g.PHONE_L / 2
    for st in rel:
        _, _, pos, rot = st["right"]
        tip = pos + 0.013 * rot[:, 2]
        assert tip[2] > top - 0.06


def test_demo_does_not_read_sim_objects():
    src = (HERE / "demo.py").read_text()
    script = src[src.index("class Script"):src.index("def plan")]
    assert "object_states" not in script and "_STATE" not in script and "emb" not in script


def test_carried_phone_clears_dash_holder_and_table():
    held = [st for st in STEPS if st["holding"]]
    lifted = held[len(held) // 20:]          # once it has left the table and the dash edge
    for st in lifted:
        c, r = st["phone"]
        box = g.phone_box(c, r)
        if st["mark"].get("pushed_-1") and not st["mark"].get("press"):
            assert min(g.boxes_gap(box, b) for b in root_boxes()) >= 0.0024 or M.lowest_z(c - g.HOLDER, r) < 0.004
        assert g.boxes_gap(box, DASH) > 0.003
    # the phone's start is only touched by the gripper's pads: the open fingers clear its faces
    grasp, rot = demo.grasp_pose()
    p_rel, r_rel = demo.phone_in_grip()
    assert abs(p_rel[0]) < 1e-9      # centred between the fingers (TCP x is the closing axis)


def test_phone_falling_from_far_fingers_is_a_drop():
    ph = cradle_phone(back=-0.04, bottom=0.05)
    tcp, rot = demo.tcp_for_phone(ph[0], ph[1])
    arms = dict(FAR, right={"tcp": tcp + [0, 0, 0.06], "rot": rot, "opening": 0.0})   # the fingers 6 cm above it
    step(ph, joints(0.0), arms, vel=(0, 0, -0.8))
    assert M.violation()


def test_pinching_top_edge_and_lowering_fast_is_not_a_drop():
    ph = cradle_phone(back=-0.04, bottom=0.05)
    top = ph[0] + [0, 0, g.PHONE_L / 2 - 0.01]
    from yam_arena.demo import grip_rotation
    rot = g.rot_z(math.pi / 2) @ grip_rotation(0.0)                      # fingers closing across the thickness
    arms = dict(FAR, right={"tcp": top, "rot": rot, "opening": g.PHONE_T})
    step(ph, joints(0.0), arms, vel=(0, 0, -0.8))
    assert not M.violation()
