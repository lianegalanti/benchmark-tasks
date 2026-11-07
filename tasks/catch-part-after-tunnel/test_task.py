"""Offline checks (no Isaac Sim): the carrier's path, the tunnel hiding the block, the scoring, and that the demo is
reachable, on time and keeps its fingers out of the tunnel.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP, TCP_OFFSET
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
FINGER_REACH_Y = 0.047 + 0.012  # open pads 94 mm apart along world y, plus the pads' own thickness, from the TCP
FINGER_REACH_X = 0.015  # half the pads' width across


def state(pos, size=M.PART, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def block_y(t):
    return M.PATH(t)[0][1]


def under_roof(y):
    """The whole block is under the roof."""
    return M.TUNNEL_Y[0] <= y - M.PART[1] / 2 and y + M.PART[1] / 2 <= M.TUNNEL_Y[1]


def spans(plan):
    t, out = 0.0, {}
    for w in plan:
        out[w.label] = (t, t + w.duration)
        t += w.duration
    return out


def test_it_loads_with_the_carrier_and_tunnel_fixed_and_only_the_block_free():
    assert TASK.free_objects == ("block",)
    assert {"carrier", "tunnel_roof", "tunnel_wall_near", "tunnel_wall_far", "pad"} <= set(TASK.names)
    assert TASK.score_over == "best"


def test_the_carrier_waits_and_cruises_on_its_cycle():
    left, right = M.LEFT_Y, M.RIGHT_Y
    assert M.TRAVEL_S == pytest.approx(4.9) and M.CYCLE_S == pytest.approx(13.8)
    for t, y in ((0.0, right), (1.9, right), (M.AT_LEFT[0] + 0.05, left), (M.AT_LEFT[1] - 0.05, left),
                 (M.CYCLE_S + 1.0, right), (2.0 + M.TRAVEL_S / 2, (left + right) / 2)):
        assert block_y(t) == pytest.approx(y, abs=1e-9), t
    ts = np.arange(0.0, 2 * M.CYCLE_S, 1 / 240)
    pos = np.array([M.PATH(t)[0] for t in ts])
    vel = np.array([M.PATH(t)[1] for t in ts])
    assert np.allclose(pos[:, 0], M.LINE_X) and np.allclose(pos[:, 2], M.PLATE_Z)  # straight along the rails
    assert np.max(np.abs(np.diff(pos[:, 1]))) < M.SPEED * 1.01 / 240  # continuous, never faster than cruising
    assert np.allclose(np.gradient(pos[:, 1], ts)[5:-5], vel[5:-5, 1], atol=2e-3)  # the reported velocity is real
    assert 0.05 <= np.max(np.abs(vel[:, 1])) <= 0.15
    cruising = np.abs(np.abs(vel[:, 1]) - M.SPEED) < 1e-9
    assert cruising.mean() > 0.5  # constant speed for most of the cycle
    accel = np.max(np.abs(np.gradient(vel[:, 1], ts)[5:-5]))
    assert accel < 0.4  # m/s^2, far below what the block's friction (about 7 m/s^2) can carry


def test_the_block_is_hidden_under_the_roof_for_a_stretch_of_each_pass():
    ts = np.arange(0.0, M.CYCLE_S, 1 / 240)
    hidden = np.array([under_roof(block_y(t)) for t in ts])
    out_pass = ts[(ts < M.AT_LEFT[0]) & hidden]
    back_pass = ts[(ts > M.AT_LEFT[1]) & hidden]
    for stretch in (out_pass, back_pass):
        assert stretch.size and stretch[-1] - stretch[0] > 1.5
        assert np.all(np.diff(stretch) < 2 / 240)  # one unbroken stretch
    assert not under_roof(M.RIGHT_Y) and not under_roof(M.LEFT_Y)  # both stops are outside the tunnel


def test_the_carrier_and_block_fit_through_the_tunnel_but_fingers_do_not():
    assert M.TUNNEL_IN / 2 - M.PLATE[0] / 2 >= 0.01  # side clearance to the plate
    assert M.ROOF_LOW - M.BLOCK_TOP >= 0.02  # the block passes under the roof
    # a top-down grasp puts the TCP near the block's centre, and the fingers and gripper rise 13 cm above the TCP:
    # with the roof under 5 cm above that, the gripper cannot reach a block under it
    assert M.ROOF_LOW - M.GRASP_Z < 0.05 < TCP_OFFSET[2]
    # the stops and rails clear the tunnel's ends and the pad
    assert M.TUNNEL_Y[0] - (M.RIGHT_Y + M.PART[1] / 2) > 0.02
    assert (M.LEFT_Y - M.PART[1] / 2 - FINGER_REACH_Y) - M.TUNNEL_Y[1] > 0.03  # open fingers at the stop are clear
    pad_far_x = M.PAD_XY[0] + M.PAD_SIDE / 2
    assert (M.LINE_X - M.PLATE[0] / 2) - pad_far_x > 0.03


def test_success_needs_the_block_still_on_the_pad():
    x, y = M.PAD_XY
    z = M.PAD_TOP + M.PART[2] / 2
    assert M.success({"block": state((x + 0.04, y - 0.04, z))})
    assert not M.success({"block": state((x, y, z), vel=(0, 0, -0.1))})  # still falling
    assert not M.success({"block": state((x, y, z + 0.03))})  # held above the pad
    assert not M.success({"block": state((x + 0.06, y, M.TOP + M.PART[2] / 2))})  # beside the pad
    assert M.distance({"block": state((x, y, z))}) == pytest.approx(0.0)


def test_partial_credit():
    x, y = M.PAD_XY
    on_carrier = state(M.BLOCK.pos)
    lifted = state((M.LINE_X, M.LEFT_Y, M.PLATE_TOP + M.LIFT_M + M.PART[2] / 2 + 0.001))
    barely_lifted = state((M.LINE_X, M.LEFT_Y, M.PLATE_TOP + M.LIFT_M + M.PART[2] / 2 - 0.002))
    over_pad = state((x, y, M.PAD_TOP + 0.04 + M.PART[2] / 2))
    high_over_pad = state((x, y, M.PAD_TOP + 0.08 + M.PART[2] / 2))
    on_table = state((0.25, 0.0, M.TOP + M.PART[2] / 2))
    scores = [M.score({"block": s}) for s in (on_carrier, lifted, barely_lifted, over_pad, high_over_pad, on_table)]
    assert scores == [0.0, 0.25, 0.0, 0.5, 0.25, 0.0]


def test_the_demo_is_reachable_and_grabs_while_the_block_is_out_of_the_tunnel():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    s = spans(plan)
    stop, leave = M.AT_LEFT
    assert s["descend"][0] >= stop and s["lift"][0] < leave - 0.4  # 0.4 s of lifting before it moves off
    for t in np.arange(s["descend"][0], s["lift"][1], 0.01):
        y = block_y(t)
        assert not under_roof(y) and y - M.PART[1] / 2 > M.TUNNEL_Y[1]  # wholly out of the tunnel
    assert s["descend"][0] < M.CYCLE_S  # the first pass, no waiting a whole cycle
    assert arm.duration < 30


def test_the_demo_fingers_never_enter_the_tunnel_below_its_roof():
    arm = ArmPlan(M.demo()["left"], station.BASES["left"])
    x0, x1 = M.LINE_X - M.TUNNEL_OUT / 2, M.LINE_X + M.TUNNEL_OUT / 2
    y0, y1 = M.TUNNEL_Y
    for t in np.arange(0.0, arm.duration + 0.5, 0.02):
        (x, y, z), *_ = arm.target(t)
        tips = z - PAD_TIP_PAST_TCP
        overlaps = x + FINGER_REACH_X > x0 and x - FINGER_REACH_X < x1 and y + FINGER_REACH_Y > y0 \
            and y - FINGER_REACH_Y < y1
        assert not (overlaps and tips < M.ROOF_TOP + 0.02), t


def test_the_waiting_fingers_clear_the_arriving_block():
    assert M.PRE_Z - PAD_TIP_PAST_TCP - M.BLOCK_TOP > 0.02
    assert M.GRASP_Z - PAD_TIP_PAST_TCP > M.PLATE_TOP  # the pad tips stay above the carrier
