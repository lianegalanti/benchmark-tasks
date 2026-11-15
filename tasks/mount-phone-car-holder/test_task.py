"""Offline checks (no Isaac Sim): the layout, the holder's mechanism (spring-closed arms with no catch, clearances), every
scoring level and violation, and the demo (reach, joint limits, clearances, holding the arms open, end state).

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


def joints(dl, dr=None):
    return {"arm_left": dl, "arm_right": dl if dr is None else dr}


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()
    yield


# -- layout ---------------------------------------------------------------------------------------------------------
def test_phone_and_holder_full_size():
    assert (g.PHONE_L, g.PHONE_W, g.PHONE_T, g.PHONE_MASS) == (0.1476, 0.0716, 0.0078, 0.171)
    assert round(2 * g.CLOSED_HALF, 4) == 0.058 and round(2 * (g.CLOSED_HALF + g.ARM_STOP), 4) == 0.094
    # closed, the arms are narrower than the phone; at the stop the lips clear it by 8 mm a side
    assert 2 * g.CLOSED_HALF < g.PHONE_W and g.ARM_STOP - g.ADMIT_D >= 0.008


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
USD = (HERE / "assets" / "holder.usda").read_text()


def test_arms_spring_shut_with_no_catch():
    # the springs pull each arm shut at every travel, from closed (preload) to the stop: nothing balances them open
    for d in np.linspace(0.0, g.ARM_STOP, 19):
        assert g.spring_force(d) >= g.ARM_PRELOAD - 1e-9
    assert g.ARM_TARGET < 0.0
    # the holder has only the two arm joints: no trigger, catch, latch or detent link that could hold them open
    assert USD.count("PhysicsPrismaticJoint") == 2
    for word in ("trigger", "catch", "latch", "detent", "tail"):
        assert word not in USD.lower()
    # the arms' own links are a pad, lip, jaw, pull tab and rack: none reaches anything that could hold it
    assert [n for n, _, _ in g.ARM_BOXES] == ["pad", "lip", "jaw", "tab", "bar"]


def test_task_starts_closed_and_springs_target_closed():
    jp = dict(M.HOLDER.joint_pos)
    assert jp == {"arm_left": 0.0, "arm_right": 0.0}
    for name, target in M.HOLDER.drive_targets:
        assert target == g.ARM_TARGET < 0.0


def test_spring_numbers():
    clamp = g.spring_force(g.PHONE_D)
    assert 1.8 < clamp < 2.5
    assert 2 * 0.75 * clamp > g.PHONE_MASS * 9.81          # the pads alone can carry the phone
    assert 5.5 < 2 * g.spring_force(demo.PULL) < 7.0        # pulling one tab out works both springs (geared)


def test_pull_tabs_outside_phone_path():
    # the tabs stand in front of the jaws, outboard of the phone's edges at every travel
    for d in (0.0, g.PHONE_D, g.ARM_STOP):
        for c, _, h in g.tab_boxes(d, d):
            assert abs(c[1]) - h[1] >= g.PHONE_W / 2 - 0.001
            assert c[0] + h[0] <= -0.016 + 1e-9


def test_links_do_not_overlap():
    arm = [g.lohi(lo, hi) for _, lo, hi in g.ARM_BOXES]
    mirror = [g.lohi((lo[0], -hi[1], lo[2]), (hi[0], -lo[1], hi[2])) for _, lo, hi in g.ARM_BOXES]
    for d in (0.0, g.PHONE_D, g.ARM_STOP):
        a = [(c + [0, d, 0], r, h) for c, r, h in arm]
        b = [(c - [0, d, 0], r, h) for c, r, h in mirror]
        assert min(g.boxes_gap(x, y) for x in a for y in b) > 0.01


# -- scoring --------------------------------------------------------------------------------------------------------
def test_nothing_done_scores_zero():
    for _ in range(5):
        step((g.phone_start(), g.PHONE_R0), joints(0.0))
    assert M.score({}) == 0.0 and not M.success({})


def holding_arms(phone):
    tcp, rot = demo.tcp_for_phone(phone[0], phone[1])
    return dict(FAR, right={"tcp": tcp, "rot": rot, "opening": g.PHONE_W})


def tab_arms(d, phone=None):
    """The left gripper pinching the left pull tab at arm travel d (and the right holding the phone, if given)."""
    arms = dict(FAR) if phone is None else holding_arms(phone)
    arms["left"] = {"tcp": demo.tab_tcp(d), "rot": demo.tab_rot(), "opening": 0.004}
    return arms


def test_arms_held_open_scores_zero():
    # the arms held open by a gripper on the pull tab, the phone elsewhere: no credit
    step((g.phone_start(), g.PHONE_R0), joints(demo.PULL), tab_arms(demo.PULL))
    assert M.score({}) == 0.0


def test_phone_on_feet_between_open_arms_scores_zero():
    # the phone standing on the feet, square, while the robot holds the arms open (they never close on it): no credit
    ph = cradle_phone(back=-0.0025)
    for _ in range(3):
        step(ph, joints(demo.PULL), tab_arms(demo.PULL))
    assert M.score({}) == 0.0 and not M.success({})
    # ... and even with both grippers away, arms that are open (not closed on the phone) score nothing
    M.reset_state()
    for _ in range(3):
        step(ph, joints(demo.PULL))
    assert M.score({}) == 0.0 and not M.success({})


def test_pads_off_the_phone_do_not_grip():
    # the pads must touch the phone: 0.5 mm off still counts, 1.5 mm off (arms held just short of it) does not
    ph = cradle_phone(back=-0.0025)
    step(ph, joints(g.PHONE_D + 0.0005))
    assert M.score({}) == 0.75
    M.reset_state()
    step(ph, joints(g.PHONE_D + 0.0015))
    assert M.score({}) == 0.0
    M.reset_state()
    far = (ph[0] + [-0.03, 0, 0], ph[1])      # the phone 3 cm in front of the cradle, arms at the same travel
    step(far, joints(g.PHONE_D))
    assert M.score({}) == 0.0


def test_arms_closed_but_not_on_phone_scores_zero():
    # the arms closed (empty, or on the phone's face rather than its edges: the lips on the glass)
    for ph, d in ((cradle_phone(), 0.0), (cradle_phone(back=-0.0085), g.ADMIT_D)):
        M.reset_state()
        step(ph, joints(d))
        step(ph, joints(d))
        assert M.score({}) == 0.0 and not M.success({})


def test_gripped_but_not_seated_scores_half():
    # the arms on the phone's edges, but it is held 3 cm above the feet, or leaning back 12 degrees on the plate
    lean = g.rot_y(math.radians(-12))
    leaning = (g.to_station(lean @ np.array([-g.PHONE_T / 2, 0.0, g.PHONE_L / 2]) + [-0.001, 0, 0]), lean)
    for ph in (cradle_phone(back=-0.0025, bottom=0.03), leaning):
        M.reset_state()
        step(ph, joints(g.PHONE_D))
        assert M.score({}) == 0.5 and not M.success({})


def test_clamped_but_held_scores_three_quarters():
    ph = cradle_phone(back=-0.0025)
    for _ in range(3):
        step(ph, joints(g.PHONE_D), holding_arms(ph))
    assert M.score({}) == 0.75 and not M.success({})


def test_clamped_with_gripper_on_tab_is_not_success():
    # the robot keeps pinching the tab (the hand, not the springs, holds the arm on the phone): 0.75, not success
    ph = cradle_phone(back=-0.0025)
    for _ in range(3):
        step(ph, joints(g.PHONE_D), tab_arms(g.PHONE_D))
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


def test_moving_phone_or_arms_is_not_success():
    ph = cradle_phone()
    step(ph, joints(g.PHONE_D))
    step(ph, joints(g.PHONE_D), vel=(0, 0, 0.05))
    assert not M.success({})
    M.reset_state()
    step(ph, joints(g.PHONE_D + 0.0015))
    step(ph, joints(g.PHONE_D))                         # the arms still closing
    assert not M.success({})


def test_off_centre_or_unclamped_is_not_success():
    for ph, j in ((cradle_phone(y=0.004), joints(g.PHONE_D + 0.004, g.PHONE_D - 0.004)),
                  (cradle_phone(), joints(demo.PULL)), (cradle_phone(bottom=0.012), joints(g.PHONE_D))):
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
    step(ph, joints(0.0), holding_arms(ph), vel=(0, 0, -0.8))
    assert not M.violation()


def test_phone_off_table_zeroes():
    step((np.array([0.3, -0.5, g.TABLE - 0.05]), EYE), joints(0.0))
    assert M.violation()


def test_end_state_counts():
    # clamped, then pulled out and laid back against the dash: no credit kept
    step(cradle_phone(), joints(g.PHONE_D), holding_arms(cradle_phone()))
    assert M.score({}) == 0.75
    step((g.phone_start(), g.PHONE_R0), joints(0.0))
    assert M.score({}) == 0.0
    M.reset_state()
    # clamped, then the arms pulled open again and held there: no credit
    step(cradle_phone(back=-0.0025), joints(g.PHONE_D))
    assert M.score({}) == 0.75
    step(cradle_phone(back=-0.0025), joints(demo.PULL), tab_arms(demo.PULL))
    assert M.score({}) == 0.0


def test_holder_frame_used():
    # a holder measured 2 cm to the left (it can't move, but the scorer works in its own frame)
    shift = np.array([0.0, 0.02, 0.0])
    ph = cradle_phone()
    M.update(g.HOLDER + shift, EYE, joints(g.PHONE_D), ph[0] + shift, ph[1], np.zeros(3), np.zeros(3), FAR)
    assert M._STATE["clamped"]
    M.reset_state()
    M.update(g.HOLDER, EYE, joints(g.PHONE_D), ph[0] + shift, ph[1], np.zeros(3), np.zeros(3), FAR)
    assert not M._STATE["gripped"]


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
        assert M.gap_to(grips(st, "left"), grips(st, "right")) > 0.015


def test_left_gripper_pinches_tab_and_holds_arms_open():
    pull = [st for st in STEPS if st["pinching"]]
    assert pull, "the demo never pinches the tab"
    for st in pull:
        d = st["arms"][0]
        _, gr, pos, rot = st["left"]
        assert gr == 0.0
        tab = [(c + g.HOLDER, r, h) for c, r, h in g.tab_boxes(d, d)][0]
        fingers = M.gripper_boxes(pos, rot, 0.004)
        # both fingers on the tab (pinching a 4 mm fin), each on its own side: the pull is a push on the fin's face
        assert all(g.boxes_gap(f, tab) < 0.0005 for f in (fingers[0], fingers[2]))
        assert rot[:, 0] @ np.array([0.0, 1.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)
    # held open: the arms reach 16 mm (pads 90 mm apart) before the phone comes over, and stay there till it is seated
    window = [st for st in STEPS if st["mark"].get("over") and not st["mark"].get("ease")]
    assert window and window[-1]["mark"].get("seated")
    for st in window:
        assert st["pinching"] and st["arms"][0] == pytest.approx(demo.PULL)


def test_fingers_clear_jaws_except_tab():
    # the left fingers touch only the left tab: they keep 4 mm off both jaws and the right tab throughout
    for st in STEPS:
        d = st["arms"][0]
        jaws = [(c + g.HOLDER, r, h) for c, r, h in g.jaw_boxes(d, d)]
        rtab = [(c + g.HOLDER, r, h) for c, r, h in g.tab_boxes(d, d)][1]
        assert M.gap_to(grips(st, "left"), jaws + [rtab]) > 0.004
        assert M.gap_to(grips(st, "right"), jaws + [(c + g.HOLDER, r, h) for c, r, h in g.tab_boxes(d, d)]) > 0.01


def test_phone_path_clears_open_arms():
    # from over the cradle until it is seated, the phone stays 3.5 mm clear of the held-open arms, even 2 mm short
    # (and clear of the holder's body until it is pressed back onto the plate)
    for st in STEPS:
        if st["mark"].get("over") and not st["mark"].get("press"):
            c, r = st["phone"]
            d = demo.PULL - 0.002
            boxes = [(cc + g.HOLDER, rr, h) for cc, rr, h in g.jaw_boxes(d, d) + g.tab_boxes(d, d)]
            assert min(g.boxes_gap(b, g.phone_box(c, r)) for b in boxes) >= 0.0035
            assert min(g.boxes_gap(b, g.phone_box(c, r)) for b in root_boxes()) >= 0.0009


def test_phone_seated_pose_scores():
    seated = [st for st in STEPS if st["mark"].get("seated") and st["holding"]][-1]
    c, r = seated["phone"]
    p, rr = M.phone_in_holder(c, r, g.HOLDER, EYE)
    pad_gaps = [g.boxes_gap(q, g.phone_box(p, rr)) for q in g.pad_boxes(g.PHONE_D, g.PHONE_D)]
    assert max(pad_gaps) < 1e-9
    assert M.clamped(p, rr, g.PHONE_D, g.PHONE_D, pad_gaps)
    assert M.tilt(rr) < 1e-6
    # the left gripper lets go of the tab only once the phone is between the pads, 1.5 mm short of its edges
    eased = [st for st in STEPS if st["mark"].get("ease") and st["pinching"]][-1]
    assert eased["arms"][0] == pytest.approx(g.PHONE_D + demo.EASE)
    assert eased["holding"]


def test_demo_ends_released():
    end = STEPS[-1]
    c, r = [st for st in STEPS if st["mark"].get("seated") and st["holding"]][-1]["phone"]
    arms = [(cc + g.HOLDER, rr, h) for cc, rr, h in g.jaw_boxes(g.PHONE_D, g.PHONE_D) + g.tab_boxes(g.PHONE_D, g.PHONE_D)]
    for side in ("left", "right"):
        assert M.gap_to(grips(end, side), [g.phone_box(c, r)] + arms) > M.CLEAR
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
        # pressed back, the phone's back is on the plate's face (never past it); it stays clear of everything else
        body = root_boxes()[1:] if st["pressing"] else root_boxes()
        assert min(g.boxes_gap(box, b) for b in body) >= 0.0009
        assert not st["pressing"] or c[0] + g.PHONE_T / 2 <= g.HOLDER[0] + 1e-9
        assert g.boxes_gap(box, DASH) > 0.003
    p_rel, _ = demo.phone_in_grip()
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
