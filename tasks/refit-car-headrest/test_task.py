"""Offline checks (no Isaac Sim): the layout, the sleeves, notches and lock, every scoring level and violation, and
the demo (reach, joint limits, clearances, end state).

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
FAR = {"left": {"tcp": np.array([0.2, 0.3, 1.0]), "rot": EYE, "opening": 0.094},
       "right": {"tcp": np.array([0.2, -0.3, 1.0]), "rot": EYE, "opening": 0.094}}


def at(depth, x=0.0, y=0.0):
    """The headrest upright with its post tips ``depth`` below the collars' tops."""
    return np.array([x, y, -depth]), EYE


def step(o, r=EYE, q=0.0, arms=FAR, vel=(0, 0, 0), w=(0, 0, 0)):
    M.update(np.asarray(o, float) + g.SEAT, r, q, np.asarray(vel, float), np.asarray(w, float), arms)


def holding(o, r, sides=("left", "right")):
    return {side: (dict(zip(("tcp", "rot"), demo.tcp_for(side, o, r)), opening=2 * g.EAR_T) if side in sides
                   else FAR[side]) for side in ("left", "right")}


def post_boxes_f(o, r, side="right"):
    return [b for k, b in g.headrest_boxes(o, r).items() if k.startswith(f"post_{side}")]


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()
    yield


# -- layout ---------------------------------------------------------------------------------------------------------
def test_headrest_full_size():
    assert (2 * g.CUSH_W, g.CUSH_H, 2 * g.CUSH_T) == pytest.approx((0.26, 0.17, 0.105))
    assert 2 * g.POST_R == pytest.approx(0.014) and 2 * g.POST_Y == pytest.approx(0.200)
    assert g.POST_LEN == pytest.approx(0.130) and g.MASS == pytest.approx(1.0)
    # the thin ends fit the open gripper (94 mm) with 15 mm to spare on each side
    assert (0.094 - 2 * g.EAR_T) / 2 >= 0.015


def test_seat_on_table_in_reach():
    assert g.SEAT[0] - g.BLOCK_X > 0.25 and g.SEAT[0] + g.BLOCK_X < 0.40
    assert g.C == pytest.approx(0.105)
    # the sleeves' closed ends stand above the table
    assert g.C - g.FLOOR > 0.005


def test_start_pose_rests_on_table_and_collars():
    o, r = g.start_pose()
    hb = g.headrest_boxes(o, r)
    low = min(float(g.box_corners(*b)[:, 2].min()) for b in hb.values())
    assert low == pytest.approx(-g.C + g.REST_GAP, abs=1e-6)
    gaps = {k: min(g.boxes_gap(b, fb) for b in hb.values()) for k, fb in g.SEAT_F.items()}
    assert min(gaps.values()) == pytest.approx(g.REST_GAP, abs=2e-4)
    assert min(v for k, v in gaps.items() if k.startswith("block")) > 0.005
    # stable: the centre of mass lies between the post tips and the cushion's edge on the collars
    com = o + r @ g.COM_H
    edge = o + r @ np.array([g.CUSH_T, 0.0, g.POST_LEN])
    assert o[0] + 0.02 < com[0] < edge[0] - 0.02
    assert 15 < math.degrees(g.LEAN) < 25


# -- mechanism ------------------------------------------------------------------------------------------------------
def pawl_overlap(depth, q, shift=0.0):
    """How far the pawl (lock at q) reaches into the right post's boxes, the post at ``depth`` and pushed ``shift``
    toward +y (positive: toward the pawl)."""
    o, r = at(depth, y=shift)
    pawl = g.lock_boxes_f(q)["pawl"]
    pts = g.box_points(*pawl, 11)
    return max(float(g.point_box_depth(pts, *b).max()) for b in post_boxes_f(o, r))


def test_pawl_fits_each_notch_and_rides_on_the_lands():
    for d in g.DEPTHS:
        # level with a notch the pawl is fully in (its spring holds the post 0.5 mm or more off centre, away from it)
        for dd in (-g.WINDOW + 2e-4, 0.0, g.SEATED - 2e-4):
            assert pawl_overlap(d + dd, 0.0, shift=-0.0005) == 0.0
    # between notches the lands reach into the pawl's rest position: it is pushed back (2.5 mm, or more if the post
    # is not pressed against the far side of its sleeve)
    assert (g.POST_R - (g.BORE - g.POST_R)) - g.PAWL_TIP == pytest.approx(0.0025)
    pawl = g.lock_boxes_f(0.0)["pawl"]
    for d in (0.030, 0.056, 0.086):              # the pawl's centre level with the middles of three lands
        o, r = at(d, y=-0.001)
        lands = [b for k, b in g.headrest_boxes(o, r).items() if k.startswith("post_right_land")]
        assert min(g.boxes_gap(b, pawl, 9) for b in lands) == 0.0


def test_flat_upper_face_stops_the_post_going_down():
    """Locked, pushing the post 1 mm further down drives the notch's flat face into the pawl's top (full width)."""
    for d in g.LOCKED:
        assert pawl_overlap(d, 0.0, shift=-0.0005) == pytest.approx(0.0, abs=1e-6)
        o, r = at(d + 0.001, y=-0.001)
        pawl = g.lock_boxes_f(0.0)["pawl"]
        land = [b for k, b in g.headrest_boxes(o, r).items() if k.startswith("post_right_land")]
        top = np.array([0.0, -g.POST_Y + g.PAWL_TIP + 0.0005, g.PAWL_Z + g.PAWL_H / 2 - 0.0005])
        assert any(g.point_box_depth(top[None], *b)[0] > 0 for b in land)


def test_pressed_button_clears_the_post_everywhere():
    for d in np.arange(0.0, g.FLOOR + 1e-9, 0.0005):
        assert pawl_overlap(d, g.LOCK_TRAVEL, shift=0.001) == 0.0
        o, r = at(d, y=0.001)
        pawl = g.lock_boxes_f(g.LOCK_TRAVEL)["pawl"]
        assert min(g.boxes_gap(b, pawl, 9) for b in post_boxes_f(o, r)) >= 0.0009


def test_ramps_and_chamfer_face_the_right_way():
    """Going up, each lower face slopes away (45 degrees) so the pawl is pushed back; the tip's chamfer meets the
    pawl's top edge on the way in."""
    for k in (1, 2, 3):
        box = g.HEADREST_H[f"post_right_ramp{k}"]
        n = box[1][:, 2]                               # the slab's normal (outer face)
        assert abs(abs(n[1]) - abs(n[2])) < 1e-9 and n[2] * np.sign(n[2]) > 0
    tip = g.HEADREST_H["post_right_tip"]
    assert tip[0][1] + tip[2][1] < -g.POST_Y + g.PAWL_TIP - 0.0009   # the flat tip end stops short of the pawl


def test_sleeves_play_and_lead_in():
    assert g.BORE - g.POST_R == pytest.approx(0.001)
    assert g.FUNNEL_TOP - g.BORE >= 0.005 - 1e-9           # 5 mm capture on every side
    assert g.FLOOR == pytest.approx(g.LOCKED[3])
    assert np.diff(g.DEPTHS) == pytest.approx([0.015] * 3)


def test_lock_spring_forces():
    rest = g.LOCK_PRELOAD
    full = g.LOCK_PRELOAD + g.LOCK_RATE * g.LOCK_TRAVEL
    assert 1.5 <= rest <= 3.5 and 2.5 <= full <= 7


# -- scoring --------------------------------------------------------------------------------------------------------
def test_start_scores_zero():
    o, r = g.start_pose()
    step(o, r)
    assert M.score({}) == 0.0 and not M.violation()


def test_posts_in_scores_quarter():
    o, r = at(0.030)
    step(o, r, q=0.0025, arms=holding(o, r))
    assert M.score({}) == 0.25


def test_shallow_or_beside_does_not_count():
    for o, r in (at(0.012), at(0.030, x=0.015)):
        M.reset_state()
        step(o, r, q=0.0025, arms=holding(o, r))
        assert M.score({}) == 0.0


def test_engaged_in_upper_notches_scores_half():
    for d in g.LOCKED[:3]:
        M.reset_state()
        o, r = at(d)
        step(o, r)
        step(o, r)
        assert M.score({}) == 0.5 and not M.success({})


def test_lowest_held_scores_three_quarters():
    o, r = at(g.LOCKED[3])
    step(o, r, arms=holding(o, r, ("left",)))
    step(o, r, arms=holding(o, r, ("left",)))
    assert M.score({}) == 0.75 and not M.success({})


def test_success_released_and_still():
    o, r = at(g.LOCKED[3])
    step(o, r)
    step(o, r)
    assert M.success({}) and M.score({}) == 0.75


def test_button_still_held_at_the_bottom_is_not_locked():
    o, r = at(g.LOCKED[3])
    step(o, r, q=g.LOCK_TRAVEL)
    step(o, r, q=g.LOCK_TRAVEL)
    assert not M.success({}) and M.score({}) == 0.25


def test_not_fitted_cases():
    cases = [
        (*at(g.LOCKED[3] - 0.005), 0.0),                 # 5 mm above the lowest notch
        (np.array([0.0, 0.0, -g.LOCKED[3]]), g.rot_x(math.radians(4)), 0.0),   # tipped 4 degrees
        (*at(g.LOCKED[3]), 0.002),                        # the pawl not in (button partly pressed)
    ]
    for o, r, q in cases:
        M.reset_state()
        step(o, r, q=q)
        step(o, r, q=q)
        assert not M.success({}) and M.score({}) < 0.75


def test_moving_is_not_success():
    o, r = at(g.LOCKED[3])
    step(o, r)
    step(o, r, vel=(0, 0, 0.03))
    assert not M.success({})


def test_off_table_zeroes():
    step(np.array([-0.6, 0.0, -g.C - 0.3]), g.rot_x(math.pi / 2))
    assert M.violation() and M.score({}) == 0.0


def test_lying_or_leaning_is_fine():
    o = np.array([-0.40, 0.0, -g.C + g.CUSH_T])
    step(o, g.rot_y(-math.pi / 2))
    assert not M.violation()
    o, r = g.start_pose()
    step(o, r, vel=(0, 0, -0.3))
    assert not M.violation()


def test_forced_zeroes():
    o, r = at(g.FLOOR + 0.006)                     # 6 mm through the sleeves' closed ends (button held)
    step(o, r, q=g.LOCK_TRAVEL)
    assert M.violation() and M._STATE["why"] == ["the headrest was forced into the seat or past the pawl"]
    M.reset_state()
    o, r = at(g.LOCKED[0] + 0.006)                 # forced 6 mm past the pawl in the first notch
    step(o, r)
    assert M.violation()
    M.reset_state()
    o, r = at(g.FLOOR + 0.002)                     # 2 mm: contact, not forced
    step(o, r, q=g.LOCK_TRAVEL)
    assert not M.violation()


def test_forced_past_the_pawl_between_samples_zeroes():
    """The post carried past the first notch's flat face between two samples, with the button out at both: forced,
    even though neither sample shows the post inside the pawl."""
    o, r = at(g.LOCKED[0])
    step(o, r)
    o, r = at(g.LOCKED[1])                         # one notch further down, the pawl back in the next notch
    step(o, r)
    assert M.violation() and M._STATE["why"] == ["the headrest was forced into the seat or past the pawl"]
    # the same move with the button held in at both ends is fine
    M.reset_state()
    o, r = at(g.LOCKED[0])
    M.update(o + g.SEAT, r, g.LOCK_TRAVEL, np.zeros(3), np.zeros(3), FAR, qb=g.BUTTON_GAP + g.LOCK_TRAVEL)
    o, r = at(g.LOCKED[1])
    M.update(o + g.SEAT, r, g.LOCK_TRAVEL, np.zeros(3), np.zeros(3), FAR, qb=g.BUTTON_GAP + g.LOCK_TRAVEL)
    assert not M.violation()
    # pulling it up (rising is free) and going down to the first notch from above are fine
    M.reset_state()
    for d, q in ((g.LOCKED[2], 0.0), (g.LOCKED[1], 0.0), (0.030, 0.003), (g.LOCKED[0], 0.0)):
        step(*at(d), q=q)                          # (riding on a land, the pawl is pushed back 3 mm)
    assert not M.violation()


def test_dropped_headrest_zeroes():
    o, r = at(0.060)
    M.update(o + g.SEAT, r, g.LOCK_TRAVEL, np.array([0.0, 0.0, -0.7]), np.zeros(3), FAR,
             qb=g.BUTTON_GAP + g.LOCK_TRAVEL)        # let go with the button in: it falls down the sleeves
    assert M.violation() and M._STATE["why"] == ["the headrest was dropped"]
    M.reset_state()
    o, r = g.start_pose()                            # toppling over onto the table, unheld
    step(o, r, w=(0.0, 9.0, 0.0))                    # (its centre then falls at 0.6 m/s)
    assert M.violation()


def test_carried_or_set_down_gently_is_fine():
    o, r = at(0.030)
    step(o, r, q=0.003, arms=holding(o, r), vel=(0, 0, -0.8))   # lowered fast while held
    o2, r2 = g.start_pose()
    step(o2, r2, vel=(0, 0, -0.1))                     # set down gently, then let go
    step(o2, r2)
    assert not M.violation()


def test_end_state_counts():
    o, r = at(g.LOCKED[3])
    step(o, r, arms=holding(o, r))
    assert M.score({}) == 0.75
    o, r = g.start_pose()
    step(o, r)
    assert M.score({}) == 0.0


# -- demo -----------------------------------------------------------------------------------------------------------
def test_demo_reach_and_limits():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert worst[side][0] < 0.5 and worst[side][1] < 0.1
        for st in STEPS:
            q = st[side][0]
            assert np.all(q >= ik.LOWER - 1e-9) and np.all(q <= ik.UPPER + 1e-9)
        # no jumps between control steps
        qs = np.array([st[side][0] for st in STEPS])
        assert np.abs(np.diff(qs, axis=0)).max() < 0.1


def test_demo_fits_run_limit():
    assert len(STEPS) + 3 * demo.CONTROL_HZ < 2400


def grips(st, side):
    _, gr, pos, rot = st[side]
    return M.gripper_boxes(pos, rot, 0.094 * gr)


def seat_station(qb=0.0):
    out = list(g.seat_boxes_station().values())
    moving = list(g.lock_boxes_f(g.latch_from_button(qb)).values()) + list(g.button_boxes_f(qb).values())
    return out, [g.shifted(b, g.SEAT) for b in moving]


def real_tips(st, side="right"):
    """The closed fingertips as yam_arena's finger meshes give them: 12.8 mm thick across both (x), 27 mm wide, from
    4.7 mm behind to 12.8 mm past the TCP."""
    _, _, pos, rot = st[side]
    return [(pos + rot @ np.array([0.0, 0.0, 0.004]), rot, np.array([0.0064, 0.0137, 0.0088]))]


def pressing(st):
    return bool(st["mark"].get("press") and not st["mark"].get("release"))


def test_grippers_clear_seat_and_each_other():
    fixed, _ = seat_station()
    near = [b for k, b in g.seat_boxes_station().items() if not k.startswith(("housing", "collar_right"))]
    for st in STEPS:
        for side in ("left", "right"):
            # pressing, the box model's pads (wider and thicker than the real fingertips) reach into the button's
            # housing: it is left out then, and the real fingertips are checked against it below
            boxes = near if (pressing(st) and side == "right") else fixed
            assert M.gap_to(grips(st, side), boxes) > 0.003, side
        assert M.gap_to(grips(st, "left"), grips(st, "right"), far=0.12) > 0.1


def test_whole_gripper_stays_above_the_seat_top():
    """Every part of both grippers stays above the seat back's top (the upholstery runs 220 mm each side)."""
    top = g.SEAT[2] + g.TRIM
    for st in STEPS:
        for side in ("left", "right"):
            for c, r, h in grips(st, side):
                assert float(g.box_corners(c, r, h)[:, 2].min()) > top + 0.003, side


def test_real_fingertips_press_the_cap_and_nothing_else():
    fixed, _ = seat_station()
    for st in STEPS:
        if not pressing(st):
            continue
        tips = real_tips(st)
        assert M.gap_to(tips, fixed) > 0.0005
        assert M.gap_to(tips, headrest_station(st)) > 0.003
        _, moving = seat_station(st["qb"])
        latch = moving[:3]
        assert M.gap_to(tips, latch) > 0.003
    st = [st for st in STEPS if st["mark"].get("lower") and not st["mark"].get("unlock")][0]
    _, _, pos, rot = st["right"]
    lead = pos + demo.LEAD
    assert abs(lead[1] - (g.SEAT[1] + demo.CAP_Y)) < 1e-6 and abs(lead[0] - g.SEAT[0]) < 1e-6
    assert lead[2] == pytest.approx(g.SEAT[2] + g.BUTTON_PROUD - demo.PRESS, abs=1e-6)


def test_only_the_fingertips_touch_the_button():
    for st in STEPS:
        _, moving = seat_station(st["qb"])
        b = grips(st, "right")
        assert M.gap_to([b[1], b[3], b[4]], moving) > 0.003     # roots and carriage clear
        assert M.gap_to(grips(st, "left"), moving) > 0.02
        if not st["mark"].get("press"):
            # gripping the cushion's right end at the start, the open fingers pass 2 mm (box model; the real
            # fingertips are thinner) from the button's cap below it
            assert M.gap_to(b, moving) > 0.0015


def test_button_drives_the_latch_through_its_wedge():
    for d in np.linspace(0.0, g.BUTTON_GAP + g.LOCK_TRAVEL, 12):
        q = g.latch_from_button(d)
        stem = g.button_boxes_f(d)["stem"]
        latch = g.lock_boxes_f(q)
        gap = min(g.boxes_gap(stem, latch[k], 13) for k in ("slope", "follower"))
        depth = max(float(g.point_box_depth(g.box_points(*stem, 13), *latch[k]).max()) for k in ("slope", "follower"))
        assert depth < 2e-4 and (gap < 2e-4 if d > g.BUTTON_GAP + 1e-4 else gap < 0.0005)
    # neither moving part runs into the seat at either end of its travel
    for q, d in ((0.0, 0.0), (g.LOCK_TRAVEL, g.BUTTON_TRAVEL)):
        for b in list(g.lock_boxes_f(q).values()) + list(g.button_boxes_f(d).values()):
            assert min(g.boxes_gap(b, fb, 9) for fb in g.SEAT_F.values()) >= 0.0004


def test_button_pressed_to_its_stop_and_let_out():
    pressed = [st["q"] for st in STEPS if st["mark"].get("lower") and not st["mark"].get("unlock")]
    assert min(pressed) == pytest.approx(g.LOCK_TRAVEL)
    assert STEPS[-1]["q"] == 0.0 and STEPS[-1]["qb"] == 0.0


def test_arm_links_clear_seat():
    from yam_arena.ik import YAM_URDF, forward_kinematics
    from yam_arena.constants import ARM_JOINTS

    fixed, _ = seat_station()
    for st in STEPS[::5]:
        for side in ("left", "right"):
            q = st[side][0]
            links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q)})
            base = np.asarray(station.BASES[side])
            for name, tf in links.items():
                if name in ("gripper",) or "finger" in name or "tip" in name:
                    continue
                p = tf[:3, 3] + base
                d = min(float(g.point_box_dist(p[None], *b)[0]) for b in fixed)
                assert d > 0.05, (side, name)


def headrest_station(st):
    o, r = st["pose"]
    return list(g.headrest_boxes(o + g.SEAT, r).values())


def test_gripper_bodies_clear_headrest():
    # roots and carriage never touch the headrest; the pads only once closed on the ends (or let go by 1 cm)
    for st in STEPS:
        hb = headrest_station(st)
        for side in ("left", "right"):
            b = grips(st, side)
            assert M.gap_to([b[1], b[3], b[4]], hb) > 0.003, side
            if side not in st["holding"] and st["mark"].get("grasped") and st[side][1] > 0.999:
                assert M.gap_to(b, hb) > 0.005, side


def test_open_fingers_clear_the_ends_on_the_way_in():
    for st in STEPS:
        if st["mark"].get("grasped"):
            break
        for side in ("left", "right"):
            if st[side][1] > 0.9:
                b = grips(st, side)
                assert M.gap_to([b[0], b[2]], headrest_station(st)) > 0.006, side


def test_held_path_clears_seat():
    for st in STEPS:
        if not st["holding"]:
            continue
        o, r = st["pose"]
        hb = g.headrest_boxes(o, r)
        if st["stage"] != "start":
            low = min(float(g.box_corners(*b)[:, 2].min()) for b in hb.values())
            assert low > -g.C + 0.005
        for name, fb in g.SEAT_F.items():
            gap = min(g.boxes_gap(b, fb, 9) for b in hb.values())
            limit = 0.0009                              # posts in the sleeves: 1 mm play, centred
            if name.startswith("floor"):
                limit = 0.0004                          # lowered to 0.5 mm above the closed ends
            if st["stage"] == "start":
                limit = 0.0                             # it still rests on the collars as the lift begins
            assert gap >= limit, (st["stage"], name, gap)


def test_inserted_straight_and_clicked():
    o, r = [st for st in STEPS if st["stage"] == "clicked"][0]["pose"]
    d = g.depth_of(o, r, "right")
    assert g.DEPTHS[0] - g.WINDOW < d < g.LOCKED[0]      # past where the pawl springs in, short of its stop
    o, r = [st for st in STEPS if st["stage"] == "bottom"][0]["pose"]
    assert abs(g.depth_of(o, r, "right") - g.DEPTHS[3]) < g.WINDOW


def test_demo_end_state_scores_and_is_released():
    end = STEPS[-1]
    o, r = end["pose"]
    arms = {}
    for side in ("left", "right"):
        _, gr, pos, rot = end[side]
        arms[side] = {"tcp": pos, "rot": rot, "opening": 0.094 * gr}
    step(o, r, end["q"], arms)
    step(o, r, end["q"], arms)
    assert M.success({})
    # right after the left fingers open, they are clear of the cushion's end faces
    rel = [st for st in STEPS if st["mark"].get("release")]
    opened = [st for st in rel if st["left"][1] >= demo.OPEN - 1e-9][0]
    b = grips(opened, "left")
    assert M.gap_to([b[0], b[2]], headrest_station(opened)) > 0.006


def test_demo_does_not_read_sim_objects():
    src = (HERE / "demo.py").read_text()
    script = src[src.index("class Script"):src.index("def plan")]
    assert "object_states" not in script and "_STATE" not in script and "emb" not in script
