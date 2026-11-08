"""Offline checks (no Isaac Sim): the sway, the plants' geometry, the monitor and scoring, and that the demo is
reachable and follows the ripe fruit.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
PARTS = ("stem", "cup_floor", "cup_wall_near", "cup_wall_far", "cup_wall_right", "cup_wall_left")
FRUIT_SIZE = (2 * M.R,) * 3
SPECS = {o.name: o for o in TASK.objects}


def state(pos, size=FRUIT_SIZE, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half,
                       pos + half)


def floor_at(which, t):
    spec = SPECS[f"{which}_cup_floor"]
    pos, vel = spec.path(t)
    return state(pos, size=spec.size, vel=vel)


def seated(which, t, d=(0.0, 0.0, 0.0)):
    return state(M.seat(which, t) + np.asarray(d), vel=M.PATHS[which](t)[1])


def objects(t, ripe=None, green=None):
    return {"ripe_tomato": ripe if ripe is not None else seated("ripe", t),
            "green_tomato": green if green is not None else seated("green", t),
            "ripe_cup_floor": floor_at("ripe", t), "green_cup_floor": floor_at("green", t)}


IN_BASKET = state((M.BASKET_XY[0] + 0.03, M.BASKET_XY[1] - 0.02, M.BASKET_FLOOR_TOP + M.R))


def play(steps):
    """Reset the monitor and feed it (t, objects) control steps; return the last objects."""
    M.reset_state()
    last = None
    for t, objs in steps:
        M.update(objs)
        last = objs
    return last


# -- the sway --------------------------------------------------------------------------------------------------------


def test_it_loads_with_the_plants_driven_and_only_the_fruit_free():
    assert set(TASK.free_objects) == {"ripe_tomato", "green_tomato"}
    for which in ("ripe", "green"):
        assert {f"{which}_{p}" for p in PARTS} <= set(TASK.names)
    assert TASK.score_over == "end"


def test_the_plants_sway_smoothly_and_never_stop():
    ts = np.arange(0.0, 3 * M.SWAY_S, 1 / 240)
    for which in ("ripe", "green"):
        ps = np.array([M.PATHS[which](t)[0] for t in ts])
        vs = np.array([M.PATHS[which](t)[1] for t in ts])
        assert np.allclose(ps[:, 2], M.FRUIT_Z)
        for i in range(2):
            assert np.allclose(np.gradient(ps[:, i], ts)[5:-5], vs[5:-5, i], atol=2e-3)  # velocity is the motion's
        speed = np.linalg.norm(vs, axis=1)
        full = ts >= M.RAMP_S
        assert speed[0] == pytest.approx(0.0, abs=1e-9)  # it starts from rest
        assert 0.06 < speed.max() < 0.075  # peak 6 to 7.5 cm/s
        assert speed[full].min() > 0.015  # never stops once swaying: at least 1.5 cm/s
        accel = np.linalg.norm(np.gradient(vs, ts, axis=0)[5:-5], axis=1)
        assert accel.max() < 0.2  # m/s^2: the seated fruit rides along (it would need about 5.7 m/s^2 to climb out)
        late = np.array([M.PATHS[which](t)[0] for t in np.arange(M.RAMP_S, M.RAMP_S + 2 * M.SWAY_S, 1 / 240)])
        cx, cy = M.CENTRES[which]
        assert late[:, 0].max() - late[:, 0].min() == pytest.approx(2 * M.AMP_X, abs=1e-3)
        assert late[:, 1].max() - late[:, 1].min() == pytest.approx(2 * M.AMP_Y, abs=1e-3)
        assert late[:, 0].mean() == pytest.approx(cx, abs=2e-3) and late[:, 1].mean() == pytest.approx(cy, abs=2e-3)
    # it repeats every sway once full-size, and the green plant is a quarter sway ahead of the ripe one
    for t in (2.5, 3.3, 7.0):
        assert np.allclose(M.PATHS["ripe"](t)[0], M.PATHS["ripe"](t + M.SWAY_S)[0])
        assert np.allclose(M.PATHS["green"](t)[0][:2] - M.CENTRES["green"],
                           M.PATHS["ripe"](t + M.SWAY_S / 4)[0][:2] - M.CENTRES["ripe"])


def test_tracking_is_needed_the_sway_is_wider_than_the_open_grip_forgives():
    # the open fingers forgive about (9.4 - 6) / 2 = 1.7 cm along their closing direction (world y): the ripe fruit
    # swings 3.5 cm either side of its centre, so a grasp at a fixed point catches it only by chance
    assert M.AMP_Y > 2 * (0.094 - 2 * M.R) / 2


def test_each_plant_is_a_rigid_set_of_boxes_that_do_not_touch():
    for which in ("ripe", "green"):
        names = [f"{which}_{p}" for p in PARTS]
        seat0 = M.seat(which, 0.0)
        offsets = {n: np.subtract(SPECS[n].pos, seat0) for n in names}
        for t in (0.0, 1.1, 3.7, 9.9):
            for n in names:
                pos, vel = SPECS[n].path(t)
                assert np.allclose(vel, M.PATHS[which](t)[1])
                assert np.allclose(pos - M.seat(which, t), offsets[n])
        boxes = [(np.asarray(SPECS[n].pos) - np.asarray(SPECS[n].size) / 2,
                  np.asarray(SPECS[n].pos) + np.asarray(SPECS[n].size) / 2) for n in names]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                gap = np.max(np.maximum(boxes[i][0] - boxes[j][1], boxes[j][0] - boxes[i][1]))
                assert gap >= M.GAP - 1e-9, (names[i], names[j])
        assert SPECS[f"{which}_stem"].pos[2] - SPECS[f"{which}_stem"].size[2] / 2 == pytest.approx(M.TOP + 0.005)


def test_the_fruit_sits_on_the_cup_rim_edges_clear_of_the_floor():
    # seated, the 6 cm ball touches the four top inner edges of the 3 cm opening, and hangs 0.4 cm into it
    for which in ("ripe", "green"):
        c = np.asarray(SPECS[f"{which}_tomato"].pos)
        assert np.hypot(M.SEAT_HALF, c[2] - M.RIM) == pytest.approx(M.R, abs=1e-9)
        floor_top = SPECS[f"{which}_cup_floor"].pos[2] + M.FLOOR / 2
        assert c[2] - M.R - floor_top > 0.01
        assert np.allclose(c, M.seat(which, 0.0))


def test_the_two_plants_and_the_basket_stay_well_apart():
    ts = np.arange(0.0, 3 * M.SWAY_S, 1 / 120)
    d = [np.linalg.norm(M.seat("ripe", t)[:2] - M.seat("green", t)[:2]) for t in ts]
    # the open fingers around the ripe fruit reach 4.7 cm plus a finger (about 1 cm) from its centre, along y;
    # the green fruit's near side stays at least 2 cm beyond that
    assert min(d) - M.R - (0.047 + 0.01) > 0.02
    cup_reach = M.SEAT_HALF + M.WALL
    basket_far_x = M.BASKET_XY[0] + M.BASKET_IN / 2 + M.B_WALL
    basket_left_y = M.BASKET_XY[1] + M.BASKET_IN / 2 + M.B_WALL
    for t in ts:
        x, y, _ = M.seat("ripe", t)
        assert x - cup_reach - basket_far_x > 0.0 or y - cup_reach - basket_left_y > 0.1


# -- monitor and scoring ---------------------------------------------------------------------------------------------


def test_success_needs_the_ripe_fruit_in_the_basket_and_the_green_one_never_out():
    t = 20.0
    last = play([(t, objects(t, ripe=IN_BASKET))])
    assert M.success(last) and M.score(last) == 0.5  # score() is only partial credit; success ends with 1.0
    # still falling into the basket
    assert not M.success(objects(t, ripe=state(IN_BASKET.pos, vel=(0, 0, -0.2))))
    # the green fruit was knocked out once and put back: it was disturbed
    steps = [(10.0, objects(10.0, green=seated("green", 10.0, (0.0, 0.0, 0.05)))), (t, objects(t, ripe=IN_BASKET))]
    last = play(steps)
    assert not M.success(last)


def test_scores_for_every_outcome():
    t0, t1 = 8.0, 20.0
    lifted = seated("ripe", t0, (0.0, 0.0, M.LIFT_M + 0.002))
    on_table = state((0.25, -0.15, M.TOP + M.R))
    green_on_table = state((0.25, 0.15, M.TOP + M.R))
    green_in_basket = state((M.BASKET_XY[0] - 0.03, M.BASKET_XY[1] + 0.03, M.BASKET_FLOOR_TOP + M.R))

    def outcome(steps):
        last = play(steps)
        return 1.0 if M.success(last) else M.score(last)

    # 1.0: ripe in the basket, green untouched
    assert outcome([(t0, objects(t0, ripe=lifted)), (t1, objects(t1, ripe=IN_BASKET))]) == 1.0
    # 0.5: ripe in the basket, green knocked to the table, or taken as well
    assert outcome([(t0, objects(t0, ripe=lifted)), (t1, objects(t1, ripe=IN_BASKET, green=green_on_table))]) == 0.5
    assert outcome([(t0, objects(t0, ripe=lifted)), (t1, objects(t1, ripe=IN_BASKET, green=green_in_basket))]) == 0.5
    # 0.25: ripe picked, then dropped on the table, or still held above the basket
    assert outcome([(t0, objects(t0, ripe=lifted)), (t1, objects(t1, ripe=on_table))]) == 0.25
    held = state((*M.BASKET_XY, M.BASKET_FLOOR_TOP + M.R + 0.06))
    assert outcome([(t0, objects(t0, ripe=lifted)), (t1, objects(t1, ripe=held))]) == 0.25
    # 0: nothing done; only the green one taken; the ripe one knocked off its cup onto the table without a pick
    assert outcome([(t0, objects(t0)), (t1, objects(t1))]) == 0.0
    assert outcome([(t0, objects(t0)), (t1, objects(t1, green=green_in_basket))]) == 0.0
    assert outcome([(t0, objects(t0)), (t1, objects(t1, ripe=on_table))]) == 0.0


def test_a_fruit_rocking_in_its_seat_is_still_in_its_cup():
    t = 9.0
    for d in ((0.008, 0.0, 0.002), (0.0, -0.01, 0.004), (0.0, 0.0, -0.005)):
        assert M.in_cup(seated("green", t, d), floor_at("green", t))
    for d in ((0.02, 0.0, 0.0), (0.0, 0.0, 0.02), (0.0, 0.0, -0.03)):
        assert not M.in_cup(seated("green", t, d), floor_at("green", t))


# -- the demo --------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        a, b = spans.get(w.label, (t, t))
        spans[w.label] = (min(a, t), t + w.duration)
        t += w.duration
    return spans


def test_the_demo_is_reachable():
    plan = M.demo()["right"]
    arm = ArmPlan(plan, station.BASES["right"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 15  # far inside the 120 s run limit


def test_the_demo_follows_the_fruit_every_control_step_while_near_it():
    plan = M.demo()["right"]
    arm = ArmPlan(plan, station.BASES["right"])
    spans = _spans(plan)
    for t in np.arange(M.MEET_S + DT, M.LEAVE_S + 1e-9, DT):
        tcp, _, finger, _ = arm.target(t)
        assert np.allclose(tcp[:2], M.seat("ripe", t + M.LEAD_S)[:2], atol=1e-9), t
    # the fingers close only once the TCP is down at the grasp height, and the lift starts once they are closed
    assert spans["close"][0] >= M.DESCEND_S[1] - 1e-9
    for t in np.arange(M.DESCEND_S[1], M.CLOSE_S[0], DT):
        assert arm.target(t)[2] == pytest.approx(FINGER_OPEN)
    assert arm.target(M.LIFT_S[0])[2] == pytest.approx(FINGER_CLOSED)
    assert spans["lift"][0] >= M.CLOSE_S[1]
    # the fruit is moving throughout: no pause to wait for
    for t in np.arange(M.DESCEND_S[0], M.LIFT_S[1], DT):
        assert np.linalg.norm(M.PATHS["ripe"](t)[1][:2]) > 0.015


def test_the_fingers_clear_the_fruit_and_the_cup():
    # hovering over the fruit, the pad tips are 2 cm or more above its top
    assert M.HOVER_Z - PAD_TIP_PAST_TCP - (M.FRUIT_Z + M.R) > 0.02
    # at the grasp, the pad tips are 1.5 cm or more above the cup rim, and below the fruit's centre (pads on its middle)
    tips = M.GRASP_Z - PAD_TIP_PAST_TCP
    assert tips - M.RIM >= 0.015 and M.FRUIT_Z - tips >= 0.009
    # lifted, the fruit's underside is above where its top was, its centre 2 cm past the 3 cm pick mark
    held_low = M.LIFT_Z - M.GRIP_ABOVE_CENTRE - M.R
    assert held_low - (M.FRUIT_Z + M.R) >= 0.0 and held_low + M.R - M.FRUIT_Z >= M.LIFT_M + 0.02
    # carried over the basket wall, and let go 1 cm above its floor
    assert held_low - (M.TOP + M.B_WALL_H) > 0.05
    assert M.PLACE_Z - M.GRIP_ABOVE_CENTRE - M.R - M.BASKET_FLOOR_TOP == pytest.approx(0.01)
    # the open fingers fit inside the basket around the fruit
    assert M.BASKET_IN / 2 - (0.047 + 0.01) > 0.01
