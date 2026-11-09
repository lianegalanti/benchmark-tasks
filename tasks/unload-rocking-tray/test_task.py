"""Offline checks (no Isaac Sim): the rocker's motion and clearances, the scoring, and that the demo is reachable,
grips each bottle at a turn of the rock, and keeps clear of the mat, the bottles and the bins.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
HALF_OPEN = 0.047  # the open fingers' inner faces, either side of the tool point


def quat_x(a):
    return np.array([math.sin(a / 2), 0.0, 0.0, math.cos(a / 2)])


def state(pos, size=M.BOTTLE, quat=(0.0, 0.0, 0.0, 1.0), vel=(0.0, 0.0, 0.0)):
    pos, q = np.asarray(pos, float), np.asarray(quat, float)
    half = np.abs(M.quat_matrix(q)) @ (np.asarray(size, float) / 2)
    return ObjectState(pos, q, np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def rocker_at(t):
    pos, _, q = M.PATH(t)
    return state(pos, M.PLATE, q)


def bottle_on_platform(name, t):
    """A bottle riding on the mat at time t."""
    return state(M.on_platform((0.0, M.BOTTLE_Y[name], M.BOTTLE[2] / 2), t), quat=quat_x(M.tilt(t)))


def bin_spot(side, dy=0.0, quat=(0.0, 0.0, 0.0, 1.0), up=0.0):
    x, y = M.BIN_XY[side]
    return state((x, y + dy, M.BIN_FLOOR_TOP + M.BOTTLE[2] / 2 + up), quat=quat)


# -- the rocker ------------------------------------------------------------------------------------------------------


def test_it_loads_with_only_the_bottles_free():
    assert TASK.free_objects == M.NAMES
    assert "rocker" in TASK.names and TASK.score_over == "end"


def test_the_platform_rocks_10_degrees_each_way_every_4_s_and_starts_level():
    assert M.tilt(0.0) == 0.0
    assert M.tilt(1.0) == pytest.approx(math.radians(10)) and M.tilt(3.0) == pytest.approx(-math.radians(10))
    for t in (0.0, 0.7, 1.0, 2.3, 3.0, 5.1):
        pos, vel, q = M.PATH(t)
        assert np.allclose(pos, (M.ROCKER_X, 0.0, M.PIVOT_Z)) and not vel.any()
        r = M.quat_matrix(q)
        assert math.atan2(r[2, 1], r[1, 1]) == pytest.approx(M.tilt(t), abs=1e-9)  # +y end up for a positive tilt
        assert np.allclose(r[:, 0], (1, 0, 0))  # about x only
    # the end bottles rise and fall 3 cm each way, the middle one hardly moves
    ends = [M.on_platform((0.0, M.SPACING, 0.0), t)[2] for t in np.arange(0, 4, 0.01)]
    assert max(ends) - M.PLATE_TOP == pytest.approx(0.0295, abs=0.001)
    mid = [M.on_platform((0.0, 0.0, 0.0), t) for t in np.arange(0, 4, 0.01)]
    assert np.ptp([p[2] for p in mid]) < 0.001


def test_nothing_slides_on_the_mat():
    w = 2 * math.pi / M.PERIOD_S
    accel = M.AMPLITUDE * w**2 * (M.PLATE[1] / 2)  # the fastest a point on the platform is accelerated
    assert accel < 0.1  # m/s^2: negligible next to gravity
    need = math.tan(M.AMPLITUDE) + accel / 9.81
    assert min(M.MAT_FRICTION) > 3 * need


def test_the_tilted_platform_clears_its_base_pivot_table_and_bins():
    half = np.array(M.PLATE) / 2
    corners = np.array([[sx * half[0], sy * y, -half[2]] for sx in (-1, 1) for sy in (-1, 1)
                        for y in np.linspace(0.0, half[1], 45)])
    for t in np.arange(0.0, 4.0, 0.02):
        r = M.quat_matrix(M.PATH(t)[2])
        under = corners @ r.T + [M.ROCKER_X, 0.0, M.PIVOT_Z]
        assert under[:, 2].min() > M.TOP + 0.01  # the ends stay above the table
        for box, gap in ((M.BASE, 0.012), (M.PIVOT, 0.008)):
            over = np.abs(under[:, 1] - box.pos[1]) <= box.size[1] / 2
            assert (under[over, 2] - (box.pos[2] + box.size[2] / 2)).min() > gap, (box.name, t)
    for x, y in M.BIN_XY.values():
        assert abs(y) - M.BIN_IN / 2 - M.WALL - M.PLATE[1] / 2 > 0.08  # a bottle can't reach a bin by tipping off


def test_the_bottles_start_standing_on_the_level_mat():
    for b in M.BOTTLES:
        assert b.pos[2] - b.size[2] / 2 == pytest.approx(M.PLATE_TOP)
        assert abs(b.pos[1]) + b.size[1] / 2 < M.PLATE[1] / 2 - 0.02
        assert np.allclose(b.pos, M.on_platform((0.0, b.pos[1], b.size[2] / 2), 0.0))


# -- scoring ---------------------------------------------------------------------------------------------------------


def scene(t=0.0, **bottles):
    out = {"rocker": rocker_at(t)}
    for n in M.NAMES:
        out[n] = bottles.get(n, bottle_on_platform(n, t))
    return out


def test_taking_a_bottle_off_is_measured_from_the_tilted_mat_beneath_it():
    M.reset_state()
    t = 1.0  # the left end at its highest
    riding = bottle_on_platform("orange_bottle", t)
    assert not M.taken_off(riding, rocker_at(t))
    raised = state(riding.pos + [0, 0, 0.025], quat=riding.quat_xyzw)
    assert not M.taken_off(raised, rocker_at(t))  # 2.5 cm up: not yet
    raised = state(riding.pos + [0, 0, 0.032], quat=riding.quat_xyzw)
    assert M.taken_off(raised, rocker_at(t))
    # the same height over the level mat would count, but over the raised end it does not
    level = bottle_on_platform("orange_bottle", 0.0)
    assert M.taken_off(state(level.pos + [0, 0, 0.032]), rocker_at(0.0))
    assert not M.taken_off(state(level.pos + [0, 0, 0.032]), rocker_at(t))


def test_per_bottle_credit_covers_every_outcome():
    M.reset_state()
    standing = bin_spot("left")
    on_side = quat_x(math.pi / 2)
    lying = state((M.BIN_XY["left"][0], M.BIN_XY["left"][1], M.BIN_FLOOR_TOP + 0.02), quat=on_side)
    held = state((0.25, 0.25, M.TOP + 0.15))
    on_table = state((0.25, 0.27, M.TOP + 0.02), quat=on_side)
    leaning = bin_spot("right", quat=quat_x(math.radians(25)), up=0.004)
    # never taken off: 0 wherever it is, unless it stands in a bin (it can't get there without being carried)
    for b, want in ((bottle_on_platform("orange_bottle", 0.0), 0.0), (on_table, 0.0), (lying, 0.0), (held, 0.0),
                    (standing, 1.0)):
        assert M.credit("orange_bottle", b) == want
    M.update(scene(orange_bottle=state((0.25, 0.17, M.PLATE_TOP + 0.08))))
    assert M._STATE["lifted"] == {"orange_bottle": True, "green_bottle": False, "purple_bottle": False}
    expected = ((standing, 1.0), (lying, 0.5), (leaning, 0.5), (held, 0.25), (on_table, 0.0),
                (bottle_on_platform("orange_bottle", 2.0), 0.25))
    for b, want in expected:
        assert M.credit("orange_bottle", b) == want
    held_over_bin = bin_spot("left", up=0.06)
    assert M.credit("orange_bottle", held_over_bin) == 0.5
    assert M.credit("orange_bottle", bin_spot("left", up=0.12)) == 0.25  # far above the rim: still carrying it


def test_score_is_the_mean_over_the_bottles_and_resets():
    M.reset_state()
    objs = scene(1.0)
    assert M.score(objs) == 0.0
    objs["orange_bottle"] = bin_spot("left", 0.035)
    assert M.score(objs) == pytest.approx(1 / 3)
    M.update(scene(green_bottle=state((0.25, 0.0, M.PLATE_TOP + 0.08))))
    objs["green_bottle"] = state((0.25, 0.1, M.TOP + 0.15))
    assert M.score(objs) == pytest.approx((1 + 0.25) / 3)
    objs["purple_bottle"] = bin_spot("right")
    objs["green_bottle"] = bin_spot("left", -0.035)
    assert M.score(objs) == pytest.approx(1.0) and M.success(objs)
    objs["green_bottle"] = state(objs["green_bottle"].pos, vel=(0.0, 0.0, -0.05))
    assert not M.success(objs)  # still moving
    M.reset_state()
    assert not any(M._STATE["lifted"].values())


def test_success_needs_every_bottle_standing_in_a_bin():
    M.reset_state()
    objs = scene(orange_bottle=bin_spot("left", 0.035), green_bottle=bin_spot("left", -0.035),
                 purple_bottle=bin_spot("right"))
    assert M.success(objs)
    objs["purple_bottle"] = bin_spot("right", quat=quat_x(math.radians(25)), up=0.004)
    assert not M.success(objs)  # leaning on a wall
    objs["purple_bottle"] = bin_spot("right", up=0.03)
    assert not M.success(objs)  # still held up off the floor
    objs["purple_bottle"] = bottle_on_platform("purple_bottle", 0.0)
    assert not M.success(objs)


# -- the scripted demo -----------------------------------------------------------------------------------------------


def plans():
    out = {}
    for side, wps in M.demo().items():
        arm = ArmPlan(wps, station.BASES[side])
        ts = [(k + 1) * DT for k in range(round(arm.duration / DT) + 5)]
        acts = np.array([arm.action7(t)[0] for t in ts])
        out[side] = (wps, arm, acts, ts)
    return out


PLANS = plans()


def spans(wps):
    t, out = 0.0, {}
    for w in wps:
        out.setdefault(w.label, []).append((t, t + w.duration))
        t += w.duration
    return out


def test_the_demo_is_reachable():
    for side, (wps, arm, acts, _) in PLANS.items():
        assert all(w.duration > 0 for w in wps)
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(acts[:, :6] >= ik.LOWER) and np.all(acts[:, :6] <= ik.UPPER), side
        assert arm.duration < 30


def test_each_bottle_is_gripped_over_a_turn_with_its_end_high():
    for name, side in (("orange_bottle", "left"), ("green_bottle", "left"), ("purple_bottle", "right")):
        turn = M.TURNS[name]
        assert abs(M.tilt(turn)) == pytest.approx(M.AMPLITUDE)
        assert M.BOTTLE_Y[name] * M.tilt(turn) >= 0  # its end of the platform is the high one (or it is central)
        sp = spans(PLANS[side][0])
        (c0, c1), = [s for s in sp["close"] if s[0] < turn < s[1]]
        (_, lift0), = [s for s in sp["hold"] if turn - 1 < s[0] < turn + 1]
        assert c0 == pytest.approx(turn - 0.5) and lift0 == pytest.approx(turn + 0.25)
        # the closing pads reach the bottle's faces when the opening (9.4 cm, closing along a smoothstep) is down
        # to its 4 cm depth; from then until the lift the grip point moves under 3 mm
        u = next(u for u in np.linspace(0, 1, 1001) if 1 - (3 * u * u - 2 * u**3) <= M.BOTTLE[0] / 0.094)
        contact = c0 + u * (c1 - c0)
        assert turn - 0.25 < contact < turn
        g = M.grip_point(name)
        drift = [np.linalg.norm(M.on_platform((0.0, M.BOTTLE_Y[name], M.GRIP_UP), t) - g)
                 for t in np.arange(contact, lift0 + 0.001, 0.01)]
        assert max(drift) < 0.003, name


def test_the_fingers_clear_the_mat_and_the_bottle_while_waiting_and_gripping():
    for name in M.NAMES:
        turn, g = M.TURNS[name], M.grip_point(name)
        for t in np.arange(turn - 1.2, turn + 0.25, 0.01):
            tcp_z = g[2] + (M.PRE_UP if t < turn - 1.0 else 0.0)
            tips = tcp_z - PAD_TIP_PAST_TCP
            mat = max(M.on_platform((0.0, y, 0.0), t)[2] for y in M.BOTTLE_Y[name] + np.array([-0.03, 0.03]))
            assert tips - mat > 0.02, (name, t)
            if t < turn - 1.0:  # at the wait height the pad tips stay over the bottle's top
                top = max(M.on_platform((0.0, M.BOTTLE_Y[name] + dy, M.BOTTLE[2]), t)[2] for dy in (-0.02, 0.02))
                assert tips - top > 0.02, (name, t)
        # the bottle is 4 cm deep between fingers opening 9.4 cm: 2.7 cm clear each side
        assert HALF_OPEN - M.BOTTLE[0] / 2 > 0.015


def held_bottle_low(name, tcp_z):
    """The lowest point of a bottle held as gripped, with the tool point at tcp_z."""
    a = abs(M.tilt(M.TURNS[name]))
    return tcp_z - (M.GRIP_UP * math.cos(a) + M.BOTTLE[1] / 2 * math.sin(a))


def platform_top_max():
    half = M.PLATE[1] / 2
    return M.PIVOT_Z + half * math.sin(M.AMPLITUDE) + M.PLATE[2] / 2 * math.cos(M.AMPLITUDE)


def test_a_carried_bottle_clears_the_platform_the_other_bottles_and_the_bin_walls():
    for name in M.NAMES:
        low = held_bottle_low(name, M.TRAVEL_Z)
        assert low - platform_top_max() > 0.025
        assert low - (M.TOP + M.WALL_H) > 0.05
        # a carry runs at travel height from over the bottle to over its bin spot; the bottles still standing then
        # (picked later) lie well outside that sweep across y, so it never passes over one
        g, p = M.grip_point(name), M.place_point(name)
        for other in M.NAMES:
            if M.TURNS[other] > M.TURNS[name]:
                oy = M.BOTTLE_Y[other]
                assert not (min(g[1], p[1]) - 0.08 < oy < max(g[1], p[1]) + 0.08), (name, other)


def test_bottles_are_let_go_just_above_the_bin_floor_inside_the_walls():
    for name in M.NAMES:
        p = M.place_point(name)
        low = held_bottle_low(name, p[2])
        assert low - M.BIN_FLOOR_TOP == pytest.approx(M.PLACE_GAP)
        assert p[2] - PAD_TIP_PAST_TCP - (M.TOP + M.WALL_H) > 0.01  # pad tips above the walls
        side = M.SPOTS[name][0]
        bx, by = M.BIN_XY[side]
        assert abs(p[0] - bx) + HALF_OPEN + 0.012 < M.BIN_IN / 2  # the open fingers inside the walls
        assert abs(p[1] - by) + M.BOTTLE[1] / 2 + 0.012 < M.BIN_IN / 2  # the bottle inside, leaning as held
    # two bottles in the left bin stand 3 cm apart
    assert abs(M.SPOTS["orange_bottle"][1] - M.SPOTS["green_bottle"][1]) - M.BOTTLE[1] >= 0.03


def test_the_arms_keep_apart():
    from yam_arena import ik as _ik

    (lw, la, lacts, ts), (rw, ra, racts, _) = PLANS["left"], PLANS["right"]
    n = min(len(lacts), len(racts))
    gap = min(np.linalg.norm(_ik.tcp_pose(lacts[k, :6], station.BASES["left"])[0]
                             - _ik.tcp_pose(racts[k, :6], station.BASES["right"])[0]) for k in range(n))
    assert gap > 0.2
