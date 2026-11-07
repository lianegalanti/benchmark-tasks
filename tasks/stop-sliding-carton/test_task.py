"""Offline checks (no Isaac Sim): the deck's layout, the untouched slide, the scoring, and that the demo is reachable,
in place before the carton arrives, and pushes slowly enough that the carton stops with the fingers.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest
from motionkit import LaunchedBox, Marker, SlickBox, slide

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
G = 9.81
REST_Z = M.DECK_TOP + M.CARTON[2] / 2


def state(pos, size=M.CARTON, vel=(0.0, 0.0, 0.0), tilt_deg=0.0):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    a = math.radians(tilt_deg) / 2
    quat = np.array([math.sin(a), 0.0, 0.0, math.cos(a)])  # tilted about x
    return ObjectState(pos, quat, np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def time_to_travel(d):
    a = M.MU * G
    return (M.V0 - math.sqrt(M.V0**2 - 2 * a * d)) / a


def test_it_loads_with_only_the_carton_free():
    assert TASK.free_objects == ("carton",)
    assert isinstance(M.CARTON_SPEC, LaunchedBox) and M.CARTON_SPEC.velocity == (0.0, M.V0, 0.0)
    assert isinstance(M.DECK, SlickBox) and M.DECK.friction == (M.MU, M.MU)
    assert isinstance(M.SQUARE, Marker)  # seen, not touched: no seam for the carton to catch on
    assert TASK.score_over == "end"


def test_one_deck_and_the_square_painted_on_it():
    (x0, x1), (y0, y1) = M.DECK_X, M.DECK_Y
    d = M.DECK
    assert np.allclose(np.r_[d.pos[:2] - np.array(d.size[:2]) / 2], (x0, y0))
    assert np.allclose(np.r_[d.pos[:2] + np.array(d.size[:2]) / 2], (x1, y1))
    assert d.pos[2] + d.size[2] / 2 == pytest.approx(M.DECK_TOP)
    sq = M.SQUARE
    assert np.allclose(sq.pos[:2], M.ZONE_XY) and np.allclose(sq.size[:2], (M.ZONE, M.ZONE))
    assert sq.pos[2] - sq.size[2] / 2 == pytest.approx(M.DECK_TOP) and sq.size[2] <= 0.001
    assert np.all(M.ZONE_LO > (x0, y0)) and np.all(M.ZONE_HI < (x1, y1))


def test_the_carton_slides_beside_the_square_and_cannot_be_grasped():
    assert M.LANE_X + M.CARTON[0] / 2 < M.DECK_X[1]
    assert M.LANE_X - M.CARTON[0] / 2 > M.ZONE_HI[0]  # never over the square while sliding
    assert M.START_Y - M.CARTON[1] / 2 > M.DECK_Y[0]
    assert min(M.CARTON[:2]) > -2 * FINGER_OPEN  # wider than the open pads (94 mm)


def test_untouched_it_slides_off_the_left_end_at_a_real_slide_speed():
    assert 0.2 <= M.V0 <= 0.5
    to_edge = M.DECK_Y[1] - M.START_Y  # its centre past the end: it tips off
    d_stop, _ = slide(M.V0, M.MU, 100.0)
    assert d_stop > to_edge + 0.15  # 15 cm to spare
    assert slide(M.V0, 1.25 * M.MU, 100.0)[0] > to_edge  # still off with 25 % more friction
    t_off = time_to_travel(to_edge)
    _, v_off = slide(M.V0, M.MU, t_off)
    assert v_off > 0.15 and t_off == pytest.approx(2.3, abs=0.05)
    t_beside = time_to_travel(M.ZONE_LO[1] - M.START_Y)
    assert t_beside == pytest.approx(1.5, abs=0.05)
    assert slide(M.V0, M.MU, t_off)[0] == pytest.approx(to_edge)
    # sliding off the end: its centre is past the deck, so it no longer scores
    gone = state((M.LANE_X, M.DECK_Y[1] + 0.01, REST_Z), vel=(0, v_off, 0))
    assert M.score({"carton": gone}) == 0.0


def test_success_needs_it_upright_still_and_wholly_inside_the_square():
    x, y = M.ZONE_XY
    margin = (M.ZONE - M.CARTON[0]) / 2
    assert M.success({"carton": state((x + margin - 0.002, y - margin + 0.002, REST_Z))})
    assert not M.success({"carton": state((x, y, REST_Z), vel=(0.0, 0.05, 0.0))})  # still sliding
    assert not M.success({"carton": state((x + margin + 0.01, y, REST_Z))})  # sticks out
    assert not M.success({"carton": state((x, y, REST_Z), tilt_deg=30)})  # tipped
    assert not M.success({"carton": state((x, y, REST_Z + 0.03))})  # held up


def test_partial_credit_covers_every_ending():
    x, y = M.ZONE_XY
    cases = {
        "sticks out of the square": (state((x + 0.06, y, REST_Z)), 0.5),
        "in the square, still moving": (state((x, y, REST_Z), vel=(-0.03, 0, 0)), 0.5),
        "stopped in its lane": (state((M.LANE_X, y, REST_Z)), 0.25),
        "stopped by the right end": (state((M.LANE_X, -0.5, REST_Z)), 0.25),
        "fell off onto the table": (state((M.LANE_X, M.DECK_Y[1] + 0.08, M.TOP + M.CARTON[2] / 2)), 0.0),
        "off the far edge": (state((M.DECK_X[1] + 0.06, y, M.TOP + M.CARTON[2] / 2)), 0.0),
        "tipped over in the square": (state((x, y, M.DECK_TOP + 0.05), tilt_deg=90), 0.0),
        "held up over the square": (state((x, y, REST_Z + 0.03)), 0.0),
    }
    for name, (s, want) in cases.items():
        assert M.score({"carton": s}) == want, name
    assert M.score({"carton": state((x, y, REST_Z), tilt_deg=10)}) == 0.5  # a small tilt is still upright


def _plan_spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        spans[w.label] = (t, t + w.duration)
        t += w.duration
    return spans


def test_the_demo_is_reachable_within_joint_limits():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration + 1.0 < 1200 * DT  # within the run limit, with success held for 1 s


def test_the_fingers_stand_in_its_path_well_before_it_arrives():
    plan = M.demo()["left"]
    spans = _plan_spans(plan)
    arrive = M.arrival_s()
    assert spans["stop in its path"][1] < arrive - 0.6  # in place 0.6 s early, time for the joints to catch up
    assert spans["hold while it runs in"][1] > arrive + 0.5
    # while the arm comes down, the carton is still far to the right of the fingers
    _, by, _ = M.BLOCK
    lead = M.START_Y + slide(M.V0, M.MU, spans["stop in its path"][1])[0] + M.CARTON[1] / 2
    assert by - M.FINGER_HALF - lead > 0.15
    # the fingers stand across the carton's leading face, tips just above the deck
    assert abs(M.BLOCK[0] - M.LANE_X) < 0.01
    tips = M.TIP_Z - PAD_TIP_PAST_TCP
    assert 0.003 < tips - M.DECK_TOP < 0.01


def test_the_stop_and_the_push_leave_it_in_the_middle_of_the_square():
    stopped_y = M.BLOCK[1] - M.FINGER_HALF - M.CARTON[1] / 2
    pushed_x = M.PUSH_TO_X - M.FINGER_HALF - M.CARTON[0] / 2
    assert np.allclose((pushed_x, stopped_y), M.ZONE_XY)
    assert M.PUSH_FROM_X - M.FINGER_HALF > M.LANE_X + M.CARTON[0] / 2 + 0.01  # comes down clear of it
    assert M.CLEAR_Z - PAD_TIP_PAST_TCP > M.DECK_TOP + M.CARTON[2] + 0.015  # passes over it


def test_the_push_slows_more_gently_than_friction_so_the_carton_stops_with_the_fingers():
    plan = M.demo()["left"]
    push = next(w for w in plan if w.label == "push onto the square")
    d = M.PUSH_FROM_X - M.PUSH_TO_X
    peak_decel = 6 * d / push.duration**2  # smoothstep's largest acceleration
    assert peak_decel < 0.8 * M.MU * G
