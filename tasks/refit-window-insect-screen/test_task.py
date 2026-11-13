"""Offline checks (no Isaac Sim): the layout, the window's channels and the plungers, every scoring level and
violation, and the demo (reach, joint limits, clearances, end state).

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
FRAME_W = g.frame_boxes_w()


def frame_station():
    return {k: (c + g.WINDOW, r, h) for k, (c, r, h) in FRAME_W.items()}


def step(o, r=EYE, press=(0.0, 0.0), arms=FAR, vel=(0, 0, 0), w=(0, 0, 0)):
    """One monitor step with the screen at (o, R) in W, plungers pushed in ``press`` (m)."""
    M.update(np.asarray(o, float) + g.WINDOW, r, {"plunger_left": -press[0], "plunger_right": -press[1]},
             np.asarray(vel, float), np.asarray(w, float), arms)


FITTED = (np.zeros(3), EYE, (g.SEAT_PRESS, g.SEAT_PRESS))


def pivot_pose(z, lean=0.0, x=0.0):
    return g.pose_from_pivot(np.array([x, 0.0, z]), lean)


def holding(o, r, opening=g.FRAME_T):
    return {side: dict(zip(("tcp", "rot"), demo.tcp_for_screen(side, o, r)), opening=opening)
            for side in ("left", "right")}


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()
    yield


# -- layout ---------------------------------------------------------------------------------------------------------
def test_screen_full_size():
    assert (g.SCREEN_W, g.SCREEN_H, g.FRAME_T, g.RAIL_B) == (0.45, 0.6, 0.01, 0.02)
    assert 0.4 <= g.SCREEN_MASS <= 0.55
    # the screen covers the opening between the jambs and overlaps each jamb
    assert g.SCREEN_W / 2 - g.RAIL_B > g.JAMB_Y[0] and g.SCREEN_W / 2 > g.JAMB_Y[1]


def test_start_pose_leans_on_jambs_and_clears_sill_and_head():
    o, r = g.start_pose()
    corners = g.box_corners(*g.lohi((-g.FRAME_T / 2, -g.SCREEN_W / 2, 0.0), (g.FRAME_T / 2, g.SCREEN_W / 2,
                                                                            g.SCREEN_H)))
    pts = o + corners @ r.T
    assert pts[:, 2].min() == pytest.approx(-g.SILL_H + g.REST_GAP, abs=1e-6)
    assert pts[:, 0].max() == pytest.approx(g.JAMB_X - g.REST_GAP, abs=1e-6)
    # stable: its centre of mass lies between its foot and the jambs, leaning back about 7 degrees
    com = o + r @ np.array([0.0, 0.0, g.SCREEN_H / 2])
    assert g.START_BOTTOM_X < com[0] < g.JAMB_X and 9.5 < math.degrees(g.START_LEAN) < 11.5
    boxes = g.screen_boxes(o, r)
    for name, fb in FRAME_W.items():
        gap = min(g.boxes_gap(b, fb) for b in boxes.values())
        if name.startswith("jamb"):
            # the top back edge rests on the jamb's face (checked at the jamb's middle)
            y = sum(g.JAMB_Y) / 2 * (1 if name.endswith("left") else -1)
            p = o + r @ np.array([g.FRAME_T / 2, y, g.SCREEN_H])
            assert float(g.point_box_dist(p[None], *fb)[0]) < 0.001
        else:
            assert gap > 0.006, name


def test_window_and_screen_on_the_table_in_reach():
    assert g.WINDOW[0] + g.FRONT_X > 0.30 and g.SILL_H == pytest.approx(0.050)
    assert g.LENGTH < 0.30


# -- mechanism ------------------------------------------------------------------------------------------------------
def test_channel_depths_and_play():
    assert g.TOP_DEPTH == pytest.approx(0.024) and g.BOT_LIP_FRONT == pytest.approx(0.008)
    assert g.BOT_HALF - g.FRAME_T / 2 >= 0.0025 - 1e-9
    assert g.TOP_HALF - g.FRAME_T / 2 >= 0.0025
    # fitted, the top rail is 8 mm up in the top channel and the pins bear 3 mm on its ceiling
    assert g.SCREEN_H - g.MOUTH == pytest.approx(0.008)
    assert g.SEAT_PRESS == pytest.approx(0.003)
    o, r, _ = FITTED
    assert g.pin_press(o, r) == pytest.approx(g.SEAT_PRESS)


def test_pushed_up_the_bottom_clears_the_lip():
    o, r = pivot_pose(g.CEIL - demo.UNDER_CEIL)
    bot = g.edge_corners(o, r, 0.0)
    assert bot[:, 2].min() - g.BOT_LIP_FRONT >= 0.006
    # the pins then are in 17.5 mm, within their 19 mm travel
    assert g.pin_press(o, r) == pytest.approx(g.PIN_PROUD - demo.UNDER_CEIL)
    assert g.pin_press(o, r) < g.PIN_TRAVEL - 0.001
    # without pushing the plungers in, the bottom edge can't get over the lip
    o, r = pivot_pose(g.CEIL - g.PIN_PROUD)
    assert g.edge_corners(o, r, 0.0)[:, 2].min() < g.BOT_LIP_FRONT


def test_plunger_forces_a_few_newtons():
    seated = g.PIN_PRELOAD + g.PIN_RATE * g.SEAT_PRESS
    full = g.PIN_PRELOAD + g.PIN_RATE * (g.PIN_PROUD - demo.UNDER_CEIL)
    assert 0.6 < seated < 1.2 and 2.0 < full < 3.5
    # pushing up takes the screen's weight plus both plungers: about 5.6 N per arm
    assert (g.SCREEN_MASS * 9.81 + 2 * full) / 2 < 7.0


# -- scoring --------------------------------------------------------------------------------------------------------
def test_start_scores_zero():
    o, r = g.start_pose()
    step(o, r)
    assert M.score({}) == 0.0 and not M.success({}) and not M.violation()


def test_top_pushed_in_scores_quarter():
    o, r = pivot_pose(g.CEIL - 0.002, math.radians(6))
    press = (g.pin_press(o, r),) * 2
    step(o, r, press, holding(o, r))
    assert M.score({}) == 0.25


def test_top_in_but_plungers_not_pushed_does_not_count():
    o, r = pivot_pose(g.MOUTH + 0.006, math.radians(6))
    step(o, r, (0.0, 0.0), holding(o, r))
    assert M.score({}) == 0.0


def test_bottom_over_lip_scores_half():
    o, r = pivot_pose(g.CEIL - 0.002)
    press = (g.pin_press(o, r),) * 2
    step(o, r, press, holding(o, r))
    assert M.score({}) == 0.5


def test_bottom_outside_lip_is_not_over():
    o, r = pivot_pose(g.CEIL - 0.002, math.radians(2))      # bottom 21 mm in front of the channel
    step(o, r, (g.pin_press(o, r),) * 2, holding(o, r))
    assert M.score({}) == 0.25


def test_fitted_but_held_scores_three_quarters():
    o, r, press = FITTED
    for _ in range(2):
        step(o, r, press, holding(o, r))
    assert M.score({}) == 0.75 and not M.success({})


def test_success_released_and_still():
    o, r, press = FITTED
    step(o, r, press)
    step(o, r, press)
    assert M.success({}) and M.score({}) == 0.75


def test_fitted_with_play_still_counts():
    o, r, press = FITTED
    step(o + [0.0025, 0.012, 0.0], g.rot_y(math.radians(-0.8)), press)
    step(o + [0.0025, 0.012, 0.0], g.rot_y(math.radians(-0.8)), press)
    assert M.success({})


def test_not_fitted_cases():
    o, r, press = FITTED
    cases = [
        (o + [0, 0, 0.009], r, press),                      # bottom edge resting on the front lip
        (o, g.rot_z(math.radians(5)), press),               # twisted 5 degrees out of the window's plane
        (o + [0, 0.02, 0], r, press),                       # 20 mm off centre
        (o, r, (0.0, g.SEAT_PRESS)),                        # a plunger not bearing on the ceiling
    ]
    for oo, rr, pp in cases:
        M.reset_state()
        step(oo, rr, pp)
        step(oo, rr, pp)
        assert not M.success({}) and M.score({}) < 0.75


def test_moving_screen_is_not_success():
    o, r, press = FITTED
    step(o, r, press)
    step(o, r, press, vel=(0, 0, 0.03))
    assert not M.success({})


def test_off_table_zeroes():
    step(np.array([-0.2, -0.5, -0.30]), g.rot_x(math.pi / 2))
    assert M.violation() and M.score({}) == 0.0


def test_lying_on_table_is_fine():
    # laid down flat on the table in front of the window (at rest): no violation, score 0
    o = np.array([-0.70, 0.0, -g.SILL_H + g.FRAME_T / 2])
    step(o, g.rot_y(-math.pi / 2))
    assert not M.violation() and M.score({}) == 0.0


def test_forced_into_frame_zeroes():
    o, r, press = FITTED
    step(o + [0, 0, -0.006], r, press)                      # bottom rail 6 mm into the sill
    assert M.violation() and M._STATE["why"] == ["the screen was forced into the window frame"]
    M.reset_state()
    step(o + [0, 0, -0.002], r, press)                      # 2 mm: contact, not forced
    assert not M.violation()


def test_dropped_screen_zeroes():
    # let go above the channel: it falls freely
    o, r, press = FITTED
    step(o + [0, 0, 0.012], r, press)
    step(o + [0, 0, 0.010], r, press, vel=(0, 0, -0.7))
    assert M.violation() and M.score({}) == 0.0 and M._STATE["why"] == ["the screen was dropped"]
    M.reset_state()
    # tipping over toward the room, unheld: near the table its centre falls fast (here 1.1 m/s)
    r2 = g.rot_y(-1.2)
    o2 = np.array([-0.08, 0.0, -g.SILL_H + 0.01])
    step(o2, r2, w=(0.0, -4.0, 0.0))
    assert M.violation()


def test_carried_or_gently_set_down_is_fine():
    o, r, press = FITTED
    # carried down fast in both hands: not a drop
    step(o + [0, 0, 0.010], r, press, holding(o + [0, 0, 0.010], r), vel=(0, 0, -0.8))
    assert not M.violation()
    # let down onto the table slowly, then let go; leaning against the jambs
    o2, r2 = g.start_pose()
    step(o2, r2, vel=(0, 0, -0.05))
    step(o2, r2)
    assert not M.violation()
    # just let go 1 mm above the floor: a short drop well under 0.5 m/s
    step(o + [0, 0, 0.001], r, press, vel=(0, 0, -0.14))
    assert not M.violation()


def test_turned_front_to_back_still_fits():
    o, r, press = FITTED
    rr = g.rot_z(math.pi)
    step(o, rr, press)
    step(o, rr, press)
    assert M.success({})


def test_end_state_counts():
    o, r, press = FITTED
    step(o, r, press, holding(o, r))
    assert M.score({}) == 0.75
    o2, r2 = g.start_pose()
    step(o2, r2)
    assert M.score({}) == 0.0


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


def test_grippers_clear_frame_and_each_other():
    fr = list(frame_station().values())
    for st in STEPS:
        for side in ("left", "right"):
            assert M.gap_to(grips(st, side), fr) > 0.005, side
        assert M.gap_to(grips(st, "left"), grips(st, "right")) > 0.1


def test_grippers_clear_end_stops_by_15mm():
    stops = [b for k, b in frame_station().items() if k.startswith("stop")]
    assert len(stops) == 4
    for st in STEPS:
        for side in ("left", "right"):
            assert M.gap_to(grips(st, side), stops) >= 0.015, side


def test_end_stops_bound_the_screen():
    # 3 mm play at each end; the screen can't slide out sideways or come in from the side
    assert g.STOP_Y[0] - g.SCREEN_W / 2 == pytest.approx(0.003)
    o, r, _ = FITTED
    for k, fb in FRAME_W.items():
        if k.startswith("stop"):
            assert min(g.boxes_gap(b, fb, 9) for b in g.screen_boxes(o, r).values()) >= 0.0029
            assert min(g.boxes_gap(b, fb, 9) for b in g.screen_boxes(o + [0, 0.006, 0], r).values()) == 0.0 \
                or "right" in k


def test_arm_links_clear_frame():
    from yam_arena.ik import forward_kinematics, YAM_URDF
    from yam_arena.constants import ARM_JOINTS

    fr = list(frame_station().values())
    for st in STEPS[::5]:
        for side in ("left", "right"):
            q = st[side][0]
            links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q)})
            base = np.asarray(station.BASES[side])
            for name, tf in links.items():
                if name in ("gripper",) or "finger" in name or "tip" in name:
                    continue
                p = tf[:3, 3] + base
                d = min(float(g.point_box_dist(p[None], *b)[0]) for b in fr)
                assert d > 0.05, (side, name)


def screen_station(st):
    o, r = st["screen"]
    return list(g.screen_boxes(o + g.WINDOW, r, pins=False).values())


def test_gripper_bodies_clear_screen():
    # the housings and finger roots never touch the screen; the pads touch it only once closed on the stile
    for st in STEPS:
        sb = screen_station(st)
        for side in ("left", "right"):
            b = grips(st, side)
            assert M.gap_to([b[1], b[3], b[4]], sb) > 0.003, side
            if st[side][1] > 0.3:            # open fingers: the pads clear the frame's faces
                assert M.gap_to([b[0], b[2]], sb) > 0.003 or st["mark"].get("grasped") is None and False, side


def test_held_screen_path_clears_frame():
    for st in STEPS:
        if not st["holding"]:
            continue
        o, r = st["screen"]
        # above the table (3 mm), in front of the sill until it is over it
        if st["stage"] != "start":
            assert g.edge_corners(o, r, 0.0)[:, 2].min() > -g.SILL_H + 0.003
        boxes = g.screen_boxes(o, r, pins=False)
        for name, fb in FRAME_W.items():
            gap = min(g.boxes_gap(b, fb, 9) for b in boxes.values())
            limit = 0.0008 if (st["stage"] in ("swung", "down") and name == "sill") else 0.002   # let down to 1 mm
            if st["stage"] in ("under", "up", "swung") and name == "head":
                limit = 0.0009                # pushed up to 1.5 mm under the ceiling (the pins take the load)
            if st["stage"] == "start" and name.startswith("jamb"):
                limit = 0.0                   # it still rests on the jambs as the lift begins
            assert gap >= limit, (st["stage"], name, gap)
        # the pins touch only the ceiling and stay within their travel
        press = g.pin_press(o, r)
        assert press <= g.PIN_TRAVEL - 0.001
        pins = [g.posed(g.lohi(*g.pin_box(s, -press)), o, r) for s in (1.0, -1.0)]
        for name, fb in FRAME_W.items():
            if name != "head":
                assert min(g.boxes_gap(p, fb) for p in pins) >= 0.002, name


def test_bottom_passes_over_lip_with_room():
    swing = [st for st in STEPS if st["stage"] in ("up", "swung") and st["mark"].get("swing")]
    lowest = []
    for st in swing:
        o, r = st["screen"]
        bot = g.posed(g.lohi(*g.RAILS["bottom_rail"]), o, r)
        if np.abs(g.edge_corners(o, r, 0.0)[:, 0]).max() <= g.BOT_HALF:
            continue                          # over the channel, between the lips (2.5 mm play each side)
        lowest.append(g.boxes_gap(bot, FRAME_W["sill_lip_front"], 9))
    assert min(lowest) >= 0.006


def test_demo_end_state_scores_and_is_released():
    end = STEPS[-1]
    o, r = end["screen"]
    press = (g.pin_press(o, r),) * 2
    arms = {}
    for side in ("left", "right"):
        _, gr, pos, rot = end[side]
        arms[side] = {"tcp": pos, "rot": rot, "opening": 0.094 * gr}
    step(o, r, press, arms)
    step(o, r, press, arms)
    assert M.success({})
    # right after the fingers open, they are clear of the frame's faces before drawing back
    rel = [st for st in STEPS if st["mark"].get("release")]
    opened = [st for st in rel if st["left"][1] >= demo.OPEN - 1e-9][0]
    for side in ("left", "right"):
        b = grips(opened, side)
        assert M.gap_to([b[0], b[2]], screen_station(opened)) > 0.015


def test_demo_does_not_read_sim_objects():
    src = (HERE / "demo.py").read_text()
    script = src[src.index("class Script"):src.index("def plan")]
    assert "object_states" not in script and "_STATE" not in script and "emb" not in script
