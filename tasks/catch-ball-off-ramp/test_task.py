"""Offline checks (no Isaac Sim): the layout, the ball's untouched run and flight, the catch window, the scoring, and
that the demo is reachable, has the basket in place well before the ball comes and keeps the hand out of its way.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

import motionkit
from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
DT = 1 / 20
UP = np.array([0.0, 0.0, 0.0, 1.0])


def state(pos, vel=(0.0, 0.0, 0.0), quat=UP, half=M.R):
    pos = np.asarray(pos, float)
    return ObjectState(pos, np.asarray(quat, float), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def ball(pos, vel=(0.0, 0.0, 0.0)):
    return state(pos, vel)


def basket(xy=(M.LINE_X, 0.33), z=M.TOP, quat=UP):
    pos = np.array([*xy, z])
    return ObjectState(pos, np.asarray(quat, float), np.zeros(3), np.zeros(3), pos, pos)


def in_cover_point(s=0.1):
    return M.on_chute(s) + M.R * M.NORMAL


def run(steps):
    """Feed (ball, basket) pairs one control step apart from a fresh episode; return the scores."""
    M.MONITOR.reset()
    return [M.MONITOR.update(b, k, k_i * DT) for k_i, (b, k) in enumerate(steps)]


BASKET_AT_START = basket(M.BASKET_START)


def chute_run():
    """The ball waiting, then inside the cover, then just out of the chute's end."""
    return ([(ball(M.START), BASKET_AT_START)] * 3 + [(ball(in_cover_point(s)), BASKET_AT_START) for s in (0.2, 0.1, 0.02)]
            + [(ball(M.flight(0.05)), BASKET_AT_START)])


# -- layout ----------------------------------------------------------------------------------------------------------


def test_it_loads_with_the_ball_and_basket_free_and_scores_at_the_end():
    assert set(TASK.free_objects) == {"ball", "basket"}
    assert {"chute_floor", "chute_cover", "gate", "chute_leg_0"} <= set(TASK.names) and TASK.score_over == "end"


def test_poses_are_plain_floats():
    """Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects at reset."""
    for spec in M.OBJECTS:
        assert all(type(v) is float for v in spec.pos), spec.name


def test_the_ball_rests_on_the_chute_against_the_gate_in_the_open_top():
    assert np.dot(M.START - M.on_chute(M.START_S), M.NORMAL) == pytest.approx(M.R + 0.001)
    assert M.GATE_POS[1] - M.GATE[1] / 2 - (M.START[1] + M.R) == pytest.approx(0.001)
    assert M.START_S + M.R < M.CHUTE_LEN  # the ball sits on the board
    assert not M.in_cover(ball(M.START))  # the waiting ball is uncovered and in view
    assert M.in_cover(ball(in_cover_point(0.1)))
    # the cover's top end stays downhill of the gate's downhill face, with a gap, at every height of its end face
    roof_end = [M.on_chute(M.ROOF_TOP_S) + n * M.NORMAL for n in (M.RAIL_H, M.RAIL_H + M.ROOF_T)]
    assert min(p[1] for p in roof_end) - (M.GATE_FACE_Y + M.GATE[1]) > 0.005
    assert M.RAIL_H - 2 * M.R >= 0.01  # 1 cm over the rolling ball's top
    assert M.GATE[0] / 2 < M.INNER_HALF  # the gate fits between the rails


def test_the_gate_holds_then_lifts_clear_of_the_ball():
    assert np.allclose(M.GATE_BOX.path(0.0)[0], M.GATE_POS) and np.allclose(M.GATE_BOX.path(M.GATE_AT_S)[0], M.GATE_POS)
    up = M.GATE_BOX.path(M.GATE_AT_S + M.GATE_OVER_S)[0]
    floor_under_gate = M.TOP + M.MOUTH_H + (M.MOUTH_Y - M.GATE_FACE_Y) * math.tan(M.THETA)
    assert up[2] - M.GATE[2] / 2 > floor_under_gate + 2 * M.R + 0.02  # 2 cm over a ball passing under it


def test_the_chute_stands_on_its_legs():
    for leg, s in zip(M.LEGS, M.LEG_S):
        assert leg.pos[2] - leg.size[2] / 2 == pytest.approx(M.TOP)
        assert leg.pos[2] + leg.size[2] / 2 == pytest.approx(M._underside(s))
    assert M.on_chute(0.0)[2] - M.TOP == pytest.approx(M.MOUTH_H)


def test_the_basket_asset_matches_the_task_numbers():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(HERE / "assets" / "basket.usda"))
    root = stage.GetDefaultPrim()
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert root.GetAttribute("physics:mass").Get() == pytest.approx(M.BASKET_MASS)
    for name, (centre, size) in M.basket_boxes().items():
        prim = stage.GetPrimAtPath(f"/Basket/{name}")
        assert prim.HasAPI(UsdPhysics.CollisionAPI), name
        assert np.allclose(prim.GetAttribute("xformOp:translate").Get(), centre, atol=1e-6), name
        assert np.allclose(prim.GetAttribute("xformOp:scale").Get(), size, atol=1e-6), name


def test_the_basket_starts_clear_of_the_ball_line_and_the_chute():
    far_side = M.BASKET_START[0] + M.IN_X / 2 + M.WALL
    assert M.LINE_X - M.R - far_side > 0.04  # an untouched ball rolls past it
    assert M.BASKET_START[1] + M.IN_Y / 2 + M.WALL < M.EDGE_Y - 0.03


def test_the_idle_arms_are_clear_of_the_chute_and_the_flight():
    top_end = M.on_chute(M.CHUTE_LEN)
    for side in ("left", "right"):
        tcp, _ = ik.tcp_pose(np.array(READY, float), np.asarray(station.BASES[side]))
        # the ready pad tips are above the chute's highest point (with the gate lifted) or well to the side of it
        gate_top = M.GATE_POS[2] + M.GATE[2] / 2 + M.GATE_RISE
        horizontal = abs(tcp[1] - np.clip(tcp[1], top_end[1], M.MOUTH_Y))
        assert tcp[2] - PAD_TIP_PAST_TCP > gate_top + 0.01 or horizontal > 0.10, side


# -- the ball's run --------------------------------------------------------------------------------------------------


def test_the_ball_flies_off_the_end_and_untouched_lands_on_the_table_then_rolls_off():
    assert M.EXIT_SPEED**2 / M.R > 2 * M.G  # it can't follow the floor's edge: a clean launch
    assert 0.7 < M.EXIT_SPEED < 1.0
    assert 0.10 < M.LAND_Y - M.MOUTH_Y < 0.18  # lands 10 to 18 cm past the chute's end
    # rolling on at nearly its horizontal speed, it reaches the left edge within half a second
    assert (M.EDGE_Y - M.LAND_Y) / (M.EXIT_SPEED * math.cos(M.THETA)) < 0.5
    # PhysX's default angular damping (0.05/s) slows the rolling ball by about 1 % on the chute
    assert 2 / 7 * 0.05 * M.RAMP_S < 0.011
    # rolling needs only a little friction on a 10 degree slope (the station's is 1.0)
    assert 2 / 7 * math.tan(M.THETA) < 0.1


def test_the_flight_clears_the_mouth_leg_and_comes_from_the_chute():
    leg = M.LEGS[0]
    path = np.array([M.flight(t) for t in np.linspace(0, M.fall_time(M.TOP + M.R), 200)])
    lo = np.array(leg.pos) - np.array(leg.size) / 2
    hi = np.array(leg.pos) + np.array(leg.size) / 2
    assert not M._hits_box(path, lo, hi)


def test_the_catch_window_is_wide_and_holds_for_speed_errors():
    lo, hi = M.catch_window()
    assert hi - lo > 0.07  # about 9 cm of basket positions catch the ball
    assert lo + 0.03 <= M.CATCH_Y <= hi - 0.03  # the demo has 3 cm to spare each way
    for f in (0.85, 1.15):  # a ball 15 % slower or faster still lands in the demo's basket
        assert M.caught_by_basket_at(M.CATCH_Y, M.EXIT_SPEED * f), f
    # ... and in one 2 cm off the demo's spot either way at the nominal speed
    assert M.caught_by_basket_at(M.CATCH_Y - 0.02) and M.caught_by_basket_at(M.CATCH_Y + 0.02)
    # a basket set right against the leg under the chute's end is too near: the ball hits its wall or overshoots
    assert not M.caught_by_basket_at(lo - 0.02)


def test_the_ball_lands_well_inside_and_the_walls_hold_it():
    t = M.fall_time(M.TOP + M.FLOOR + M.R)
    land = M.flight(t)
    far_inner = M.CATCH_Y + M.IN_Y / 2
    assert far_inner - (land[1] + M.R) > 0.02  # it lands 2 cm and more short of the far wall
    assert M.RIM - M.FLOOR > 2 * M.R  # the walls are taller than the ball
    # it rolls into the far wall at about 0.84 m/s; with no restitution the friction of its spin against the wall
    # lifts it by at most v^2 (2/7)^2 / 2g, well under the wall's height above it
    v = M.EXIT_SPEED * math.cos(M.THETA)
    assert (2 / 7 * v) ** 2 / (2 * M.G) < 0.01


# -- scoring ---------------------------------------------------------------------------------------------------------


def in_basket_point(k):
    return ball(np.asarray(k.pos) + np.array([0.0, 0.03, M.FLOOR + M.R]))


def test_in_basket_uses_the_baskets_own_frame_and_needs_it_upright():
    k = basket()
    assert M.in_basket(in_basket_point(k), k)
    assert not M.in_basket(ball(np.asarray(k.pos) + [0.0, 0.0, M.RIM + 0.01]), k)  # above the rim
    assert not M.in_basket(ball(np.asarray(k.pos) + [0.0, M.IN_Y / 2 + 0.03, M.R]), k)  # beside it
    tilt = math.radians(20)
    q20 = (math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))
    assert M.in_basket(in_basket_point(k), basket(quat=q20))
    tilt = math.radians(40)
    q40 = (math.sin(tilt / 2), 0.0, 0.0, math.cos(tilt / 2))
    assert not M.in_basket(in_basket_point(k), basket(quat=q40))
    upside_down = (1.0, 0.0, 0.0, 0.0)
    assert not M.in_basket(ball(np.asarray(k.pos) + [0.0, 0.0, -0.03]), basket(quat=upside_down))


def test_a_clean_catch_scores_success():
    k = basket((M.LINE_X, M.CATCH_Y))
    steps = chute_run() + [(in_basket_point(k), k)] * 25
    scores = run(steps)
    assert M.MONITOR.caught and scores[-1] == 0.75
    assert M.success({"ball": in_basket_point(k), "basket": k})
    assert not M.success({"ball": ball(in_basket_point(k).pos, (0, 0.2, 0)), "basket": k})  # still rolling


def test_untouched_the_ball_scores_0():
    rolled = chute_run() + [(ball((M.LINE_X, y, M.TOP + M.R), (0, 0.84, 0)), BASKET_AT_START)
                            for y in np.arange(M.LAND_Y, M.EDGE_Y + 0.05, 0.04)]
    fell = rolled + [(ball((M.LINE_X, M.EDGE_Y + 0.05, M.TOP - 0.3)), BASKET_AT_START)] * 5
    assert run(fell)[-1] == 0.0 and M.MONITOR.touched and M.MONITOR.fallen and not M.MONITOR.caught


def test_caught_then_spilled_keeps_0_75():
    k = basket((M.LINE_X, M.CATCH_Y))
    steps = chute_run() + [(in_basket_point(k), k)] * 2 + [(ball((M.LINE_X, 0.5, M.TOP + M.R)), k)] * 10
    assert run(steps)[-1] == 0.75
    assert not M.success({"ball": ball((M.LINE_X, 0.5, M.TOP + M.R)), "basket": k})


def test_into_the_basket_after_touching_the_table_scores_0_5():
    k = basket((M.LINE_X, 0.45))
    steps = chute_run() + [(ball((M.LINE_X, M.LAND_Y, M.TOP + M.R)), k)] * 3 + [(in_basket_point(k), k)] * 25
    assert run(steps)[-1] == 0.5 and not M.MONITOR.caught
    assert not M.success({"ball": in_basket_point(k), "basket": k})


def test_into_the_basket_too_late_is_not_a_catch():
    """Within the time limit only a ball straight from the chute can be in the basket: one held up, say, takes longer."""
    k = basket((M.LINE_X, M.CATCH_Y))
    held = ball((M.LINE_X, M.MOUTH_Y + 0.05, M.TOP + 0.12))
    steps = chute_run() + [(held, k)] * 25 + [(in_basket_point(k), k)] * 5
    assert run(steps)[-1] == 0.5 and not M.MONITOR.caught


def test_stopped_on_the_table_scores_0_25():
    steps = chute_run() + [(ball((M.LINE_X, M.LAND_Y, M.TOP + M.R)), BASKET_AT_START)] * 10
    assert run(steps)[-1] == 0.25


def test_a_ball_that_never_comes_out_or_is_taken_from_the_top_scores_0():
    stuck = [(ball(M.START), BASKET_AT_START)] * 3 + [(ball(in_cover_point(0.01)), BASKET_AT_START)] * 30
    assert run(stuck)[-1] == 0.0
    # picked off the open top before the gate lifts and put in the basket: it never went through the chute
    k = BASKET_AT_START
    taken = [(ball(M.START), k)] * 3 + [(ball(M.START + [0.0, 0.0, 0.1]), k)] * 3 + [(in_basket_point(k), k)] * 25
    assert run(taken)[-1] == 0.0 and not M.success({"ball": in_basket_point(k), "basket": k})


def test_score_reads_the_episode_clock_and_resets_with_the_episode():
    motionkit._CLOCK.update(t=0.0, running=False)
    for fn in motionkit._RESET:
        fn()
    k = basket((M.LINE_X, M.CATCH_Y))
    for i, (b, kk) in enumerate(chute_run() + [(in_basket_point(k), k)] * 3):
        motionkit._CLOCK["t"] = 6.0 + i * DT
        s = M.score({"ball": b, "basket": kk})
    assert s == 0.75
    for fn in motionkit._RESET:
        fn()
    assert not M.MONITOR.caught and M.MONITOR.last_covered is None


# -- the demo --------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        spans[w.label] = (t, t + w.duration)
        t += w.duration
    return spans


def test_the_demo_is_reachable():
    arm = ArmPlan(M.demo()["left"], station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_demo_grips_the_tab_and_sets_the_basket_on_the_catch_spot():
    plan = M.demo()["left"]
    pick = next(w for w in plan if w.label == "grip the tab").pos
    place = next(w for w in plan if w.label == "let go").pos
    assert pick[0] - M.BASKET_START[0] == pytest.approx(M.TAB_X) and pick[1] == M.BASKET_START[1]
    tips = pick[2] - PAD_TIP_PAST_TCP
    assert M.TOP + M.RIM + 0.008 < tips  # pad tips 1 cm over the rim ...
    assert M.TOP + M.TAB_Z[1] - tips > 0.045  # ... gripping 4.7 cm of the handle's height
    assert M.TAB[0] / 2 > 0.012  # the closed fingers stop 1.5 cm short of their target: the full 50 N squeeze
    from yam_arena.constants import GRIPPER
    assert GRIPPER["stiffness"] * M.TAB[0] / 2 >= GRIPPER["effort"]
    # the handle stays out of the ball's way inside the basket
    assert (M.LINE_X - M.R) - (M.LINE_X + M.TAB_X + M.TAB[0] / 2) > 0.015
    assert place[0] - M.TAB_X == pytest.approx(M.LINE_X) and place[1] == pytest.approx(M.CATCH_Y)
    assert 0.0 < place[2] - pick[2] < 0.006  # let go 4 mm above the table
    assert -FINGER_OPEN - M.TAB[0] / 2 > 0.03  # open, the pads clear the handle by 3 cm and more
    clear = next(w for w in plan if w.label == "up off the tab").pos
    assert clear[2] - PAD_TIP_PAST_TCP - (M.TOP + M.TAB_Z[1]) > 0.015  # lifted off over the handle's top
    # the handle's top stays below the fingers' roots (9 cm above the pad tips)
    assert M.TOP + M.TAB_Z[1] - (pick[2] - PAD_TIP_PAST_TCP) < 0.07


def test_the_basket_is_down_and_the_hand_away_well_before_the_ball_comes():
    plan = M.demo()["left"]
    spans = _spans(plan)
    assert spans["let go"][1] <= M.EXIT_EARLIEST - 1.5  # set down and let go 1.5 s before the ball can come out
    assert spans["let go"][1] < M.GATE_AT_S  # ... before the gate even lifts
    assert spans["back out of the way"][1] <= M.EXIT_EARLIEST - 0.5
    arm = ArmPlan(plan, station.BASES["left"])
    for t in np.arange(spans["back out of the way"][1], spans["wait"][1], DT):
        pos, *_ = arm.target(t)
        assert M.LINE_X - pos[0] > 0.18, t  # the gripper (about 12 cm across its fingers) is 12 cm off the line
    assert arm.duration < 15


def test_the_basket_is_carried_clear_of_the_chute():
    plan = M.demo()["left"]
    spans = _spans(plan)
    arm = ArmPlan(plan, station.BASES["left"])
    leg_top = M.LEGS[0].pos[2] + M.LEGS[0].size[2] / 2
    for t in np.arange(spans["lift"][0], spans["set down"][1], DT):
        tcp, *_ = arm.target(t)
        centre_y = tcp[1]
        near_side = centre_y - M.IN_Y / 2 - M.WALL
        bottom = tcp[2] - M.GRASP_DZ
        # the basket never comes over the chute's end or its leg below the chute's height
        assert near_side > M.MOUTH_Y + 0.02 or bottom > leg_top + 0.08, t


def test_debug_prints_run_without_the_simulator(capsys, monkeypatch):
    monkeypatch.setattr(M, "DEBUG", True)
    motionkit._CLOCK.update(t=0.0, running=False)
    for fn in motionkit._RESET:
        fn()
    k = basket((M.LINE_X, M.CATCH_Y))
    for i in range(4):
        motionkit._CLOCK["t"] = M.GATE_AT_S - 0.6 + i * 0.5
        M.score({"ball": ball(M.START), "basket": k})
    lines = [x for x in capsys.readouterr().out.splitlines() if x.startswith("[ball-debug]")]
    assert len(lines) == 4 and "off_spot=(+0.000,+0.000)" in lines[0] and "ball=" in lines[-1]


def spinning_ball(pos, vel=(0.0, 0.0, 0.0)):
    """A ball as yam_arena reports it while it spins: its bounds are its turned box's, up to R*sqrt(3) out."""
    pos = np.asarray(pos, float)
    half = M.R * math.sqrt(3)
    return ObjectState(pos, np.array([0.3, 0.2, 0.1, 0.927]), np.asarray(vel, float), np.array([0.0, 30.0, 0.0]),
                       pos - half, pos + half)


def test_the_first_sim_runs_landing_is_a_catch():
    """Measured: the basket 3 mm short and 4 mm left of the catch spot; the ball at y 0.263 z 0.142, then lying on the
    basket floor at y 0.348 z 0.033 (heights above the table), spinning, so its reported bounds reach below the
    table top."""
    k = basket((M.LINE_X - 0.003, M.CATCH_Y + 0.004))
    flying = spinning_ball((M.LINE_X, 0.263, M.TOP + 0.142), (0.0, 0.84, -0.3))
    landed = spinning_ball((M.LINE_X, 0.348, M.TOP + 0.033), (0.0, 0.3, 0.0))
    assert landed.low[2] < M.TOP  # what tripped the old check
    still = ObjectState(landed.pos, landed.quat_xyzw, np.zeros(3), np.zeros(3), landed.low, landed.high)
    steps = [(ball(in_cover_point(s)), k) for s in (0.2, 0.1, 0.02)] + [(flying, k), (landed, k)] + [(still, k)] * 25
    assert run(steps)[-1] == 0.75 and M.MONITOR.caught and not M.MONITOR.touched
    assert M.success({"ball": still, "basket": k})


def test_touching_the_table_uses_the_balls_centre_and_ignores_the_baskets_footprint():
    k = basket((M.LINE_X, M.CATCH_Y))
    assert M.touches_table(spinning_ball((M.LINE_X, M.LAND_Y + 0.2, M.TOP + M.R)), k)  # on the table beside it
    assert not M.touches_table(spinning_ball((M.LINE_X, M.LAND_Y + 0.2, M.TOP + M.R + 0.01)), k)  # 1 cm up
    assert not M.touches_table(spinning_ball((M.LINE_X, M.CATCH_Y, M.TOP + M.FLOOR + M.R)), k)  # on its floor
    assert not M.touches_table(spinning_ball((M.LINE_X, M.CATCH_Y, M.TOP + M.R)), k)  # over the basket: never
