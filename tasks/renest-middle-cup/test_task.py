"""Offline checks (no Isaac Sim): the cups' geometry, the scoring at every level, and that the demo is reachable,
keeps its fingers clear of the rims and the other arm, and nests the cups in order.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
c = M.c
DT = 1 / 20
OPEN_HALF = -FINGER_OPEN   # an open finger's pad face from the TCP, along world y
PAD = 0.012                # pad thickness beyond its face


def quat_tilt_x(deg):
    a = math.radians(deg) / 2
    return np.array([math.sin(a), 0.0, 0.0, math.cos(a)])


def cup(key, pos, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0)):
    pos = np.asarray(pos, float)
    return ObjectState(pos, np.asarray(quat, float), np.asarray(vel, float), np.zeros(3), pos, pos)


def scene(**over):
    """The cups at their starting places, with any overridden."""
    objs = {M.NAME[k]: cup(k, c.START[k]) for k in c.CUPS}
    for key, state in over.items():
        objs[M.NAME[key]] = state
    return objs


def nested(dx=0.0, dy=0.0, dz=0.0):
    lx, ly = c.LARGE_XY
    return scene(medium=cup("medium", (lx + dx, ly + dy, c.TOP + c.WALL + dz)),
                 small=cup("small", (lx + dx, ly + dy, c.TOP + 2 * c.WALL + dz)))


@pytest.fixture(autouse=True)
def grippers_parked():
    M._STATE["grip_points"] = [np.array(p) for p in M.PARK.values()]
    yield
    M._reset_state()


# -- geometry ---------------------------------------------------------------------------------------------------------
def test_three_free_usd_cups_with_1p5_cm_clearance():
    assert set(TASK.free_objects) == {"cup_small", "cup_medium", "cup_large"}
    assert TASK.score_over == "best"
    for key in c.CUPS:
        assert (HERE / f"assets/cup_{key}.usda").is_file()
    assert c.MEDIUM.inner - c.SMALL.outer == pytest.approx(2 * c.GAP)
    assert c.LARGE.inner - c.MEDIUM.outer == pytest.approx(2 * c.GAP)
    assert c.GAP >= 0.015


def test_each_nested_cup_stands_well_above_the_rim_around_it():
    for inner, outer in c.NEXT_LARGER.items():
        assert c.rim_when_nested(inner) - c.rim_when_nested(outer) > 0.033


def test_the_assets_match_the_dimensions():
    import make_cups

    for key in c.CUPS:
        assert (HERE / f"assets/cup_{key}.usda").read_text() == make_cups.cup_usda(key)


def test_the_medium_cup_cannot_be_gripped_across_but_the_small_one_can():
    assert c.MEDIUM.outer > 2 * OPEN_HALF - 0.01   # under 3 mm a side: the demo pinches a wall instead
    assert 2 * OPEN_HALF - c.SMALL.outer > 2 * 0.015


def test_start_layout_does_not_overlap_and_is_in_reach():
    (lx, ly), (mx, my) = c.LARGE_XY, c.MEDIUM_XY
    assert abs(ly - my) - (c.LARGE.outer + c.MEDIUM.outer) / 2 > 0.10
    assert lx == pytest.approx(mx, abs=0.05)


# -- scoring ----------------------------------------------------------------------------------------------------------
def test_start_and_doing_nothing_score_zero():
    objs = scene()
    assert M.score(objs) == 0.0 and not M.success(objs)


def test_full_nest_succeeds():
    assert M.success(nested())
    assert M.success(nested(dx=0.014, dy=-0.014))   # pushed against the walls still counts


def test_success_needs_grippers_clear_and_cups_still():
    objs = nested()
    lx, ly = c.LARGE_XY
    M._STATE["grip_points"] = [np.array([lx, ly, c.TOP + 0.10])]
    assert not M.success(objs)                      # still holding the small cup
    M._STATE["grip_points"] = None
    assert not M.success(objs)                      # never measured
    M._STATE["grip_points"] = [np.array(p) for p in M.PARK.values()]
    objs[M.NAME["small"]] = cup("small", objs[M.NAME["small"]].pos, vel=(0, 0, -0.05))
    assert not M.success(objs)


def test_wrong_order_or_unseated_cups_do_not_succeed():
    lx, ly = c.LARGE_XY
    # the medium cup resting on the small cup's rim, inside the large cup
    on_rim = scene(medium=cup("medium", (lx, ly, c.TOP + c.WALL + c.SMALL.height)))
    assert not M.success(on_rim) and M.score(on_rim) == 0.0
    # the medium cup turned upside down over the small one
    flipped = scene(medium=cup("medium", (lx, ly, c.TOP + c.WALL + c.SMALL.height + c.MEDIUM.height),
                               quat=(1.0, 0.0, 0.0, 0.0)))
    assert not M.success(flipped) and M.score(flipped) == 0.0
    # tilted 15 degrees
    tilted = nested()
    tilted[M.NAME["small"]] = cup("small", tilted[M.NAME["small"]].pos, quat=quat_tilt_x(15))
    assert not M.success(tilted)
    # the whole nest lifted off the table
    assert not M.success(nested(dz=0.05))


def test_half_credit_for_one_correct_pair_involving_the_medium_cup():
    lx, ly = c.LARGE_XY
    mx, my = c.MEDIUM_XY
    medium_in_large = scene(medium=cup("medium", (lx, ly, c.TOP + c.WALL)),
                            small=cup("small", (0.20, 0.22, c.TOP)))
    small_in_medium = scene(small=cup("small", (mx, my, c.TOP + c.WALL)))
    for objs in (medium_in_large, small_in_medium):
        assert M.score(objs) == 0.5 and not M.success(objs)


def test_quarter_credit_once_the_small_cup_is_lifted_out():
    lx, ly = c.LARGE_XY
    lifted = scene(small=cup("small", (lx, ly, c.TOP + c.LARGE.height + M.LIFT_M + 0.001)))
    just_short = scene(small=cup("small", (lx, ly, c.TOP + c.LARGE.height + M.LIFT_M - 0.003)))
    assert M.score(lifted) == 0.25 and M.score(just_short) == 0.0
    set_aside = scene(small=cup("small", (0.20, 0.22, c.TOP)))   # set down beside: counts as best-so-far only
    assert M.score(set_aside) == 0.0


# -- the demo ---------------------------------------------------------------------------------------------------------
def _run(side):
    plan = M.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    n = round(arm.duration / DT) + 5
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(n)])
    return plan, arm, actions


def _spans(plan):
    t, out = 0.0, {}
    for w in plan:
        out[w.label] = (t, t + w.duration, w)
        t += w.duration
    return out


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable(side):
    _, arm, actions = _run(side)
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert arm.duration < 25


def test_fingertips_stay_above_the_rims_they_pass():
    left, right = _spans(M.demo()["left"]), _spans(M.demo()["right"])
    # small cup gripped in the large cup, and released into the medium cup
    assert left["grip the small cup"][2].pos[2] - PAD_TIP_PAST_TCP - (c.TOP + c.LARGE.height) > 0.04
    tips = left["release the small cup"][2].pos[2] - PAD_TIP_PAST_TCP
    assert tips - (c.TOP + c.rim_when_nested("medium")) >= 0.015
    # medium cup released into the large cup
    tips = right["release the medium cup"][2].pos[2] - PAD_TIP_PAST_TCP
    assert tips - (c.TOP + c.rim_when_nested("large")) >= 0.015
    # pad tips stay above the table when gripping the loose medium cup
    assert right["pinch the wall"][2].pos[2] - PAD_TIP_PAST_TCP - c.TOP > 0.04


def test_the_inner_finger_fits_inside_the_medium_cup_when_open():
    # pinching the -y wall: the open inner finger's pad runs from TCP + OPEN_HALF to TCP + OPEN_HALF + PAD
    far_wall_inner = c.MEDIUM.inner / 2
    finger_far = M.PINCH_DY + OPEN_HALF + PAD
    assert far_wall_inner - finger_far > 0.015


def test_carried_cups_clear_the_rims_they_cross():
    s_bottom = M.S_TRAVEL - (c.SMALL.height - M.GRIP_BELOW_RIM)
    assert s_bottom - (c.TOP + c.rim_when_nested("medium")) >= 0.015
    m_bottom = M.M_TRAVEL - (c.MEDIUM.height - M.GRIP_BELOW_RIM)
    assert m_bottom - (c.TOP + c.LARGE.height) >= 0.03


def test_the_arms_take_turns_over_the_large_cup():
    left, right = _spans(M.demo()["left"]), _spans(M.demo()["right"])
    assert right["carry over the large cup"][0] >= left["hold it aside"][1] + 0.1
    assert left["over the nested medium cup"][0] >= right["park"][1] - 1e-9
    # while the right arm works over the large cup, the left gripper (holding the small cup) is 15 cm or more away
    assert np.hypot(M.ASIDE[0] - c.LARGE_XY[0], M.ASIDE[1] - (c.LARGE_XY[1] + M.PINCH_DY)) > 0.15


def test_the_demo_ends_in_success_geometry():
    """Where the demo releases each cup, the cup is seated once it drops the last few millimetres, and both grippers
    end parked clear of the stack."""
    left, right = _spans(M.demo()["left"]), _spans(M.demo()["right"])
    lx, ly = c.LARGE_XY
    m_tcp = right["release the medium cup"][2].pos
    s_tcp = left["release the small cup"][2].pos
    assert (m_tcp[0], m_tcp[1] - M.PINCH_DY) == pytest.approx((lx, ly))
    assert s_tcp[:2] == pytest.approx((lx, ly))
    m_bottom = m_tcp[2] - (c.MEDIUM.height - M.GRIP_BELOW_RIM)
    s_bottom = s_tcp[2] - (c.SMALL.height - M.GRIP_BELOW_RIM)
    assert 0 < m_bottom - (c.TOP + c.WALL) <= 0.005
    assert 0 < s_bottom - (c.TOP + 2 * c.WALL) <= 0.005
    M._STATE["grip_points"] = [np.array(left["park"][2].pos), np.array(right["park"][2].pos)]
    assert M.success(nested())
    assert left["park"][1] < 2400 * DT and right["park"][1] < 2400 * DT
