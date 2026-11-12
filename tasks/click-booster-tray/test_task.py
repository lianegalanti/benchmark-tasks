"""Offline checks (no Isaac Sim): the ratchet's geometry, how the tray fits the booster, every scoring level and
violation, and that the demo is reachable, clears the seat and fits the run limit.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py      # from this folder
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import geometry as g  # noqa: E402  (task.py put this folder on the path)

DT = 1 / 20
RUN_LIMIT_STEPS = 1200
BOOSTER_POS = np.array([*g.BOOSTER_XY, g.TABLE])
FAR = [np.array([0.1, 0.3, 1.1]), np.array([0.1, -0.3, 1.1])]   # both grippers parked well away


def booster(dx=0.0, tilt=0.0):
    h = math.radians(tilt) / 2
    return SimpleNamespace(pos=BOOSTER_POS + [dx, 0, 0], quat_xyzw=np.array([math.sin(h), 0, 0, math.cos(h)]),
                           lin_vel=np.zeros(3))


def tray(offset, dz=0.0, left=0.0, right=0.0, vz=0.0, dy=0.0):
    """The tray with its origin at booster x ``offset`` (resting on the armrests unless ``dz``), catches pushed out
    by ``left``/``right`` (m)."""
    pos = BOOSTER_POS + [offset, dy, g.SEATED_Z + dz]
    return SimpleNamespace(pos=pos, quat_xyzw=np.array([0.0, 0, 0, 1]), lin_vel=np.array([0, 0, vz]),
                           sliders={"catch_left": left, "catch_right": -right})


START = SimpleNamespace(pos=BOOSTER_POS + [g.START_OFFSET, 0, g.RAIL_DEPTH], quat_xyzw=np.array([0.0, 0, 0, 1]),
                        lin_vel=np.zeros(3), sliders={"catch_left": 0.0, "catch_right": 0.0})


def at_notch(n, **kw):
    return tray(g.offset_for_catch(g.notch_seat(n)), **kw)


def holding(t):
    """Finger points of a gripper closed on the front rim of tray state ``t``."""
    rim = np.asarray(t.pos) + [g.RIM_X, g.GRIP_Y, g.TCP_IN_TRAY_Z]
    return [rim, rim - [0, 0, PAD_TIP_PAST_TCP]] + FAR[1:]


def play(*steps, fingers=FAR, b=None):
    """Run tray states (or (booster, tray) pairs) through the monitor, with the arms still; (score, success)."""
    M.reset_state()
    for st in (*steps, steps[-1]):     # the last state twice: stillness is judged between steps
        bs, ts = st if isinstance(st, tuple) else (b or booster(), st)
        M.update(bs, ts, fingers, FAR)
    return M.score({}), M.success({})


# -- layout and geometry ---------------------------------------------------------------------------------------------
def test_it_loads_with_both_parts_loose_and_plain_floats():
    assert set(TASK.free_objects) == {"booster", "tray"}
    assert TASK.score_over == "end" and TASK.success_hold_s == 1.0
    assert M.INSTRUCTION == "Put the tray on the booster seat at its middle setting, locked at the second notch."
    for spec in TASK.objects:
        assert all(type(v) is float for v in spec.pos), spec.name
    assert {s.name for s in M.TRAY_PART.sliders} == {"catch_left", "catch_right"}


def test_full_size():
    assert (g.BOOSTER_W, g.BOOSTER_D, g.BOOSTER_H) == (0.40, 0.33, 0.23)
    assert (g.TRAY_W, g.TRAY_D) == (0.42, 0.25) and g.TRAY_MASS == 0.45 and g.BOOSTER_MASS == 2.5
    assert g.ARM_TOP - g.SEAT_TOP == pytest.approx(0.12)         # tray 12 cm above the seat: lap room
    assert [round(g.backrest_gap(n), 3) for n in (1, 2, 3)] == [0.145, 0.12, 0.095]   # tummy room per notch


def test_the_ratchet_holds_one_way():
    depth_in = g.CATCH_IN - g.ARM_OUT                       # catch tip 2 mm off the notch floor at rest
    assert depth_in == pytest.approx(0.002)
    assert g.ARM_OUT + g.RACK_DEPTH - g.CATCH_IN >= 0.005   # it overlaps the square faces by 6 mm: can't pull back
    assert g.CATCH_TRAVEL > g.RACK_DEPTH - depth_in + 0.002  # it can ride over a crest ...
    assert g.CATCH_TRAVEL < g.STOP_OUT - depth_in - 0.002   # ... but not over the stop behind notch 3
    # the notch opening at the catch's depth (45-degree ramps) leaves 3 mm of play
    opening = g.PITCH - g.TOOTH_ROOT + depth_in
    assert opening - g.CATCH_W == pytest.approx(0.003)
    # under the lip with 7 mm to spare (a tray pushed slightly nose-down lifts its catches), inside the teeth's band
    assert g.LIP[0] - (g.ARM_TOP + g.CATCH_Z[1]) == pytest.approx(0.007)
    assert g.ARM_TOP + g.CATCH_Z[0] > g.BAND[0]
    # clicking on: 3.6 N per spring at the crest; over a 45-degree ramp with friction 0.2 that takes 1.5 times as
    # much push, so about 11 N for both catches, plus about 2 N to slide the tray: within real trays' 10-30 N
    crest = g.CATCH_RATE * (g.RACK_DEPTH - depth_in)
    assert crest == pytest.approx(3.6)
    push = 2 * crest * (1 + 0.2) / (1 - 0.2) + 0.4 * g.TRAY_MASS * 9.81
    assert 10.0 <= push <= 30.0


def test_notch_windows_are_separate_and_hold_the_seats():
    wins = [g.notch_window(n) for n in (1, 2, 3)]
    for (a0, a1), (b0, b1) in zip(wins, wins[1:]):
        assert a1 < b0
    for n, (lo, hi) in zip((1, 2, 3), wins):
        assert lo < g.notch_seat(n) < hi
        # with the catch down (2 mm out at most) its front face is behind the next ramp: the window can't be reached
        # on a crest or a ramp
        assert hi + g.CATCH_W / 2 <= g.tooth_start(n) + g.ENGAGED_DEPTH + 0.001
    assert g.notch_window(3)[1] + g.CATCH_W / 2 > g.STOP_X - 0.002
    assert [round(g.front_overhang(n), 3) for n in (1, 2, 3)] == [0.105, 0.08, 0.055]


def _aabbs(boxes, origin):
    out = []
    for b in boxes:
        if len(b) > 3:      # rotated boxes (ramps, lead-ins) lie inside the armrest's envelope
            continue
        c, s = np.asarray(b[0]) + origin, np.asarray(b[1]) / 2
        out.append((c - s, c + s))
    return out


def _overlap(a, b, tol=0.0005):
    return all(a[0][i] < b[1][i] - tol and b[0][i] < a[1][i] - tol for i in range(3))


@pytest.mark.parametrize("offset", [g.START_OFFSET, -0.10, g.offset_for_catch(g.notch_seat(1)),
                                    g.offset_for_catch(g.notch_seat(2)), g.offset_for_catch(g.notch_seat(3))])
def test_tray_body_clears_the_booster(offset):
    seated = offset > g.START_OFFSET
    tz = g.SEATED_Z if seated else g.RAIL_DEPTH
    booster_boxes = _aabbs(g.booster_boxes((0, 0, 0), (0, 0, 0)), np.zeros(3))
    tray_boxes = _aabbs(g.TRAY_BOXES_PLAIN, np.array([offset, 0, tz]))
    for t in tray_boxes:
        for b in booster_boxes:
            assert not _overlap(t, b), (offset, t, b)
    # the catches, at rest, sit in a notch without touching (seated poses) and clear everything else
    if seated:
        for s in (1, -1):
            for c in _aabbs(g.catch_boxes(s, (0, 0, 0)), np.array([offset, 0, tz])):
                for b in booster_boxes:
                    assert not _overlap(c, b), (offset, c, b)


def test_rails_and_skirts_leave_room():
    assert g.ARM_IN - g.RAIL_IN[1] == pytest.approx(0.0015)
    assert g.SKIRT[0] - (g.ARM_OUT + g.STOP_OUT) >= 0.009
    assert g.SEATED_Z - g.RAIL_DEPTH - g.SEAT_TOP >= 0.04       # rails pass well above the seat
    assert g.START_OFFSET + g.TRAY_D / 2 <= -0.005              # the tray starts clear of the booster


# -- scoring ---------------------------------------------------------------------------------------------------------
def test_the_start_scores_0():
    assert play(START) == (0.0, False)
    assert not M._STATE["violation"]


def test_locked_at_notch_2_let_go_and_still_is_full_score():
    assert play(START, at_notch(2)) == (1.0, True)
    assert play(at_notch(2, dy=0.0015)) == (1.0, True)               # the rails' full side play
    assert play(at_notch(2, left=0.001, right=0.0015)) == (1.0, True)


def test_locked_at_notch_2_but_held_or_moving_is_075():
    t = at_notch(2)
    assert play(t, fingers=holding(t)) == (0.75, False)
    M.reset_state()
    M.update(booster(), t, FAR, FAR)
    M.update(booster(), t, FAR, [FAR[0] + [0.01, 0, 0], FAR[1]])   # an arm still moving
    assert (M.score({}), M.success({})) == (0.75, False)


@pytest.mark.parametrize("n", [1, 3])
def test_locked_in_another_notch_is_half(n):
    assert play(at_notch(n)) == (0.5, False)
    t = at_notch(n)
    assert play(t, fingers=holding(t)) == (0.5, False)


def test_on_the_armrests_but_not_locked_is_a_quarter():
    assert play(at_notch(2, left=0.007)) == (0.25, False)               # one catch on a crest
    assert play(at_notch(2, left=0.007, right=0.007)) == (0.25, False)  # both riding over teeth
    assert play(tray(-0.05)) == (0.25, False)                             # on, catches short of the first tooth
    t = at_notch(2, left=0.007)
    assert play(t, fingers=holding(t)) == (0.0, False)                    # still held: nothing


def test_hovering_or_tilted_is_not_seated():
    assert play(at_notch(2, dz=0.01)) == (0.0, False)
    assert play(at_notch(2, dy=0.012)) == (0.0, False)
    assert play(tray(-0.10)) == (0.0, False)                              # rails not yet over the armrests


def test_knocking_the_booster_is_a_violation_that_latches():
    assert play((booster(0.04), at_notch(2)), (booster(), at_notch(2))) == (0.0, False)
    assert M._STATE["why"] == ["the booster seat was knocked out of place"]
    assert play((booster(tilt=12), START), (booster(), at_notch(2))) == (0.0, False)
    shoved = tray(g.offset_for_catch(g.notch_seat(2)) + 0.02)
    assert play((booster(0.02), shoved)) == (1.0, True)                    # a 2 cm shove is tolerated


def test_dropping_the_tray_is_a_violation_that_latches():
    assert play(tray(-0.10, dz=0.05, vz=-0.8), at_notch(2)) == (0.0, False)
    assert M._STATE["why"] == ["the tray was dropped"]
    assert play(tray(-0.10, dz=0.05, vz=-0.3), at_notch(2)) == (1.0, True)   # lowered briskly is fine


def test_lowering_the_tray_fast_while_holding_it_is_not_a_drop():
    held = tray(-0.10, dz=0.05, vz=-0.8)
    M.reset_state()
    M.update(booster(), held, holding(held), FAR)
    assert not M._STATE["violation"]
    M.update(booster(), at_notch(2), FAR, FAR)
    M.update(booster(), at_notch(2), FAR, FAR)
    assert (M.score({}), M.success({})) == (1.0, True)


def test_off_the_table_is_a_violation():
    floor = SimpleNamespace(pos=np.array([0.3, 0.7, 0.1]), quat_xyzw=np.array([0.0, 0, 0, 1]), lin_vel=np.zeros(3),
                            sliders={"catch_left": 0.0, "catch_right": 0.0})
    assert play(floor, at_notch(2)) == (0.0, False)


def test_release_distance():
    t = at_notch(2)
    rim = np.asarray(t.pos) + [g.RIM_X, g.GRIP_Y, g.TCP_IN_TRAY_Z]
    assert g.tray_clearance([rim], t.pos, t.quat_xyzw) < 0.005
    assert g.tray_clearance([rim + [0, 0, 0.05]], t.pos, t.quat_xyzw) >= 0.02


# -- demo ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable_within_joint_limits(side):
    arm = ArmPlan(M.demo()[side], station.BASES[side])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_demo_is_symmetric_and_fits_the_run_limit():
    plans = M.demo()
    for a, b in zip(plans["left"], plans["right"]):
        assert a.duration == b.duration and a.pos[0] == b.pos[0] and a.pos[1] == -b.pos[1] and a.pos[2] == b.pos[2]
    total = sum(w.duration for w in plans["left"])
    assert total + 3.0 < 0.5 * RUN_LIMIT_STEPS * DT
    hold_after_close = sum(w.duration for w in plans["left"][2:4])
    assert hold_after_close >= 1.3                                       # gripper closes before lifting


def test_the_demo_pushes_into_notch_2_and_not_past_it():
    left = {w.label: w for w in M.demo()["left"]}
    to_offset = lambda w: w.pos[0] - g.BOOSTER_XY[0] - g.RIM_X   # noqa: E731
    push = to_offset(left["slide on past notch 1 into notch 2"]) + g.CATCH_X
    back = to_offset(left["ease back onto the catches"]) + g.CATCH_X
    assert g.notch_window(2)[0] < back < push < g.notch_window(2)[1]
    # the catch's front face stops at least 4 mm short of tooth 3's crest: let go there, the spring pushes it back
    # down the ramp into notch 2
    crest = g.tooth_start(2) + g.RACK_DEPTH
    assert crest - (push + g.CATCH_W / 2) >= 0.003
    assert push - g.CATCH_W / 2 - (g.tooth_start(1) + g.TOOTH_ROOT) >= 0.005   # 6 mm past tooth 2's crest
    # set down with the back edge on the armrests' flat tops, at least 2.5 cm past the lead-in chamfer (the tray
    # lags the grips by about 1.3 cm in the carry), and the catches 1 cm short of the lip
    entry = to_offset(left["set down, back edge first"])
    assert entry + g.TRAY_D / 2 - (g.ARM_X0 + g.LEAD) >= 0.025
    assert g.LIP_X0 - (entry + g.CATCH_X + g.CATCH_W / 2) >= 0.01
    assert g.LIP_X0 < g.tooth_start(0)
    # carried high enough that a tray pivoted 7.5 degrees in the grips (7.3 measured in Isaac Sim, back edge down)
    # clears the armrest tops by 8 mm
    carry = left["over the armrests' front ends"].pos[2] - (g.TABLE + g.SEATED_Z + g.TCP_IN_TRAY_Z)
    assert carry - (g.TRAY_D - 0.002) * math.sin(math.radians(7.5)) >= 0.0085


def test_the_fingers_stay_off_the_booster_and_end_clear_of_the_tray():
    plans = M.demo()
    for side in ("left", "right"):
        for w in plans[side]:
            assert w.yaw == g.GRIP_YAW                       # fingers close along x: they reach 6 cm ahead at most
            reach = 0.057 if w.finger != 0.0 else 0.02       # open: pad at 47 mm plus the finger; closed on the rim
            assert w.pos[0] + reach < g.BOOSTER_XY[0] - 0.02
        last = plans[side][-1]
        t = at_notch(2)
        tcp = np.array(last.pos)
        assert g.tray_clearance([tcp, tcp - [0, 0, PAD_TIP_PAST_TCP]], t.pos, t.quat_xyzw) >= 0.03
