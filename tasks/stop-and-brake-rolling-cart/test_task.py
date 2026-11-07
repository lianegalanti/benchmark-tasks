"""Offline checks (no Isaac Sim): the track and cart geometry, the brake lever's toggle and its hold on the slope, the
untouched roll, the scoring, and that the demo is reachable, in place before the cart arrives, and clear at the end.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scenekit import BodyState, Shape, UsdBody

import geometry as g
from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
DT = 1 / 20
DEG = math.radians


def cart(q=None, side=0.0, up=0.0, vel=(0.0, 0.0, 0.0), brake=g.STOWED, roll_deg=0.0, pos=None):
    """A cart state centred ``q`` up the slope (default: where the demo stops it), offset ``side``/``up``."""
    q = M.Q_STOP if q is None else q
    p = g.to_station(q, (side, 0.0, up)) if pos is None else np.asarray(pos, float)
    a = DEG(roll_deg) / 2  # an extra turn about the track's y (a roll)
    rot = g.R_TRACK @ np.array([[math.cos(2 * a), 0, math.sin(2 * a)], [0, 1, 0], [-math.sin(2 * a), 0, math.cos(2 * a)]])
    quat = _quat(rot)
    return BodyState(pos=p, quat_xyzw=quat, lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3), low=p, high=p,
                     joints={"brake": brake})


def _quat(r):
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    return np.array([(r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w), w])


def judge(state, clear=True, fallen=False):
    M._reset_state()
    M._STATE["fallen"] = fallen
    M.observe(state, clear)
    return M.score({"cart": state}), M.success({"cart": state})


# -- scene ---------------------------------------------------------------------------------------------------------
def test_it_loads_with_the_cart_as_a_free_articulation_and_the_assets_built():
    assert TASK.score_over == "end" and TASK.success_hold_s == 2.0
    assert isinstance(M.CART, UsdBody) and M.CART.articulation and M.CART.free_start
    assert not M.CART.self_collisions and dict(M.CART.joint_pos) == {"brake": g.STOWED}
    text = (HERE / "assets" / "cart.usda").read_text()
    assert '"brake"' in text and "physics:lowerLimit = -40" in text and "physics:upperLimit = 12" in text
    assert "PhysicsFixedJoint" not in text  # free: nothing ties the cart to the world
    for s in [*M.RAILS, M.GUIDE]:
        assert isinstance(s, Shape) and s.friction == (g.RAIL_MU, g.RAIL_MU) and s.friction_min
    assert M.DECK.friction == g.DECK_MU and not M.DECK.friction_min
    for o in M.OBJECTS:  # plain floats only (np.float64 breaks Isaac Lab's pose tensors)
        assert all(type(v) is float for v in (*o.pos, *getattr(o, "rot_xyzw", ())))


def test_the_track_slopes_down_to_the_tables_left_edge_and_stays_in_the_table():
    lo_end, hi_end = g.deck_point(0.0), g.deck_point(g.DECK_LEN)
    assert lo_end[1] == pytest.approx(g.END_Y) and g.TABLE_LEFT_Y - lo_end[1] == pytest.approx(0.01)
    assert hi_end[1] == pytest.approx(g.START_END_Y)
    assert hi_end[2] - lo_end[2] == pytest.approx(g.DECK_LEN * math.sin(g.THETA))
    # the deck's underside is below the table top everywhere: it reads as a wedge resting on the table
    for q in (0.0, g.DECK_LEN):
        assert (g.deck_point(q) - g.DECK_T * g.NORMAL)[2] < g.TOP
    assert g.LANE_X + g.DECK_X[0] > 0.10 and g.LANE_X + g.DECK_X[1] < 0.80


def test_the_cart_sits_on_the_rails_guided_and_clear_of_the_deck():
    parts = g.body_parts()
    for side in ("near", "far"):
        lo, hi = parts[f"runner_{side}"]
        assert lo[2] == 0.0  # runners' undersides on the rail tops
        s = -1 if side == "near" else 1
        assert lo[0] == pytest.approx(s * g.RAIL_X - g.RAIL_W / 2)
    # guide shoes straddle the guide bar with 4 mm play each side, 4 mm above the deck, under the body
    lo, hi = parts["shoe_far"]
    assert lo[0] - g.GUIDE_W / 2 == pytest.approx(0.004)
    assert lo[2] - (-g.RAIL_H) == pytest.approx(0.004)
    assert g.BODY_Z[0] - (g.GUIDE_H - g.RAIL_H) == pytest.approx(0.004)  # the bar's top under the body
    assert parts["runner_far"][0][0] > hi[0]  # shoes inboard of the rails


# -- the brake lever -------------------------------------------------------------------------------------------------
def test_stowed_the_pad_clears_the_deck_and_engaged_it_presses_on_it():
    deck_z = -g.RAIL_H
    assert g.lever_corners(g.STOWED)[:, 2].min() - deck_z > 0.006
    assert g.pad_contact(g.ENGAGED)[2] - deck_z == pytest.approx(-g.INTERFERENCE)
    # the pad never sinks deeper than 1 mm beyond its engaged depth on the way over (the robot lifts the cart that
    # little more as it pushes the lever past straight up)
    worst = max(g.pad_depth(DEG(d)) for d in np.arange(-40, 12.01, 0.1))
    assert worst - g.pad_depth(g.ENGAGED) < 0.001


def test_the_lever_is_a_toggle_that_the_pad_and_its_weight_push_into_either_stop():
    com = g.lever_com()
    assert com[2] > 0.015  # its weight above the hinge: it falls toward whichever side it leans
    for phi, sign in ((g.STOWED, -1), (g.ENGAGED, 1)):
        x_com = g.lever_to_track(phi, com)[0] - g.PIVOT_X
        assert np.sign(x_com) == sign  # gravity's torque about +y is x * m * g: toward the stop it rests on
    # engaged: the deck pushes up on the pad's lowest corner, 6 mm outboard of the hinge, turning the lever into its stop;
    # friction on the pad acts along the hinge axis and has no turning effect
    contact = g.pad_contact(g.ENGAGED)
    torque_y = -(contact[0] - g.PIVOT_X) * 1.0
    assert torque_y > 0 and contact[0] - g.PIVOT_X < -0.006


def test_the_lever_never_reaches_the_cart_body_runners_or_bracket():
    for d in np.arange(-40, 12.01, 1.0):
        c = g.lever_corners(DEG(d))
        below_body = c[:, 2] < g.BODY_Z[0]
        assert np.all(c[~below_body, 0] < -g.BODY_X - 0.005), d  # beside the body: outside its near side
        assert np.all(c[:, 0] < -g.RAIL_X - g.RAIL_W / 2 - 0.01), d  # well outboard of the near runner
    lo, hi = g.body_parts()["bracket"]
    assert lo[1] > g.LEVER_Y + g.BAR_HY  # the bracket cheek is beside the bar, not in its swing
    assert hi[2] < g.PIVOT_Z + g.PADDLE_Z[0]  # and below the paddle


def test_engaged_the_brake_holds_the_cart_on_the_slope_and_unbraked_it_rolls():
    m_cart, m_lever = g.BODY_MASS, g.LEVER_MASS
    w = (m_cart + m_lever) * g.G
    # the pad lifts the near side: the cart rests on the far runner's outer edge and the pad
    com_x = (m_lever * g.PIVOT_X) / (m_cart + m_lever)
    far = g.RAIL_X + g.RAIL_W / 2
    pad_x = g.pad_contact(g.ENGAGED)[0]
    n_pad = w * (far - com_x) / (far - pad_x)
    lift_runner = g.INTERFERENCE * (far - (-g.RAIL_X - g.RAIL_W / 2)) / (far - pad_x)
    assert lift_runner > 0.001  # the near runner lifts off its rail
    mu_pad = (g.CART_MU[0] + g.DECK_MU[0]) / 2  # averaged
    hold = mu_pad * n_pad + g.RAIL_MU * (w - n_pad)
    pull = w * math.sin(g.THETA)
    assert hold > 5 * pull
    # unbraked, static friction on the rails is well below the slope: a stopped cart let go rolls away
    assert g.RAIL_MU < 0.4 * math.tan(g.THETA)


# -- the roll --------------------------------------------------------------------------------------------------------
def test_untouched_it_rolls_off_the_end_of_the_track_and_the_table():
    assert g.Q_START + g.CART_HALF_LEN < g.DECK_LEN  # starts wholly on the track
    t_off = M.off_end_s()
    assert t_off == pytest.approx(3.6, abs=0.1)
    assert M.off_end_s(mu=0.012) < 6.0  # still well inside the trial with twice the rails' friction
    v_off = g.accel() * t_off
    assert v_off > 0.4
    state = cart(q=-0.02, vel=(0, v_off, 0))
    assert judge(state) == (0.0, False)


def test_the_cart_reaches_the_stop_slowly_enough_to_be_caught():
    t = M.arrival_s()
    assert 1.9 < t < 2.3
    v = g.accel() * t
    assert v < 0.32  # about as fast as the carton in the stop-sliding-carton task, at twice its mass
    assert M.arrival_s(mu=0.0) > M.TO_BLOCK_S + 0.5  # in place 0.5 s early even on frictionless rails


# -- scoring ---------------------------------------------------------------------------------------------------------
def test_success_needs_it_on_its_rails_braked_still_and_let_go():
    braked = cart(brake=g.ENGAGED, up=0.0008)
    assert judge(braked) == (0.75, True)  # success (held 2 s by the embodiment) scores 1.0
    assert judge(cart(brake=DEG(9.0))) == (0.75, True)
    assert judge(cart(brake=DEG(6.0)))[1] is False  # not yet past the toggle
    assert judge(braked, clear=False) == (0.75, False)  # a gripper still on it
    assert judge(cart(brake=g.ENGAGED, vel=(0, 0.02, 0)))[1] is False  # still sliding
    assert judge(braked, fallen=True) == (0.0, False)


def test_partial_credit_covers_every_ending():
    cases = {
        "braked, still, held": (cart(brake=g.ENGAGED), False, 0.75),
        "stopped by a gripper, brake off": (cart(), False, 0.5),
        "stopped, brake off, arms away (cannot last on the slope)": (cart(), True, 0.5),
        "on the rails, still rolling": (cart(vel=(0, 0.1, 0)), True, 0.25),
        "braked but pushed off its rails onto the deck": (cart(side=0.03, up=-0.01, brake=g.ENGAGED), True, 0.25),
        "held up 1 cm off the rails": (cart(up=0.01), False, 0.25),
        "lifted 5 cm off the track": (cart(up=0.05), False, 0.0),
        "tipped on its side": (cart(roll_deg=80, up=0.02), True, 0.0),
        "fell off the end": (cart(pos=(g.LANE_X, 0.75, g.TOP - 0.3)), True, 0.0),
        "pushed off the near side": (cart(side=-0.2, up=-0.05), True, 0.0),
    }
    for name, (s, clear, want) in cases.items():
        assert judge(s, clear=clear)[0] == want, name
    # a cart that ran off and was somehow put back still scores 0 (the fall is latched)
    assert judge(cart(brake=g.ENGAGED), fallen=True)[0] == 0.0
    M._reset_state()
    M.observe(cart(q=-0.05), True)
    M.observe(cart(brake=g.ENGAGED), True)
    assert M.score({"cart": cart(brake=g.ENGAGED)}) == 0.0


def test_arms_clear_means_two_centimetres_from_the_body_or_the_lever_where_it_is():
    s = cart(brake=g.ENGAGED)
    top = g.to_station(M.Q_STOP, (0.0, 0.0, g.BODY_Z[1] + 0.03))
    assert M.cart_distance(s, top)[0] == pytest.approx(0.03, abs=1e-6)
    handle_top = g.to_station(M.Q_STOP, (g.PIVOT_X, g.LEVER_Y, g.lever_box(g.ENGAGED)[1][2] + 0.01))
    assert M.cart_distance(s, handle_top)[0] == pytest.approx(0.01, abs=1e-3)


# -- the demo --------------------------------------------------------------------------------------------------------
def _spans(plan):
    t, out = 0.0, {}
    for w in plan:
        out[w.label] = (t, t + w.duration)
        t += w.duration
    return out


def _run(side):
    plan = M.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    acts, tcps = [], []
    for k in range(round(arm.duration / DT) + 5):
        a, _ = arm.action7((k + 1) * DT)
        acts.append(a)
        tcps.append(ik.tcp_pose(arm.q, station.BASES[side]))
    return arm, np.array(acts), tcps


def test_the_demo_is_reachable_within_joint_limits():
    for side in ("left", "right"):
        arm, acts, _ = _run(side)
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, side
        assert np.all(acts[:, :6] >= ik.LOWER) and np.all(acts[:, :6] <= ik.UPPER), side
        assert arm.duration + 2.0 + 1.0 < 1200 * DT


def test_the_left_fingers_stand_in_the_carts_path_before_it_arrives_and_hold_until_it_is_braked():
    left, right = _spans(M.demo()["left"]), _spans(M.demo()["right"])
    assert left["stop in its path"][1] < M.arrival_s(mu=0.0) - 0.5
    assert left["hold while it is braked"][1] >= right["hold it at its stop"][1]
    assert right["above the lever"][1] > M.arrival_s(mu=0.009) + 0.8  # comes in after even a slow cart has stopped
    # the fingers face the cart's leading face across its whole width of contact, tips above the guide bar
    tips = g.to_track(np.asarray(M.BLOCK) - [0, 0, PAD_TIP_PAST_TCP])
    assert tips[1][0] == pytest.approx(M.BLOCK_X, abs=1e-9) and abs(M.BLOCK_X) < g.BODY_X - 0.02
    assert g.BODY_Z[0] + 0.01 < tips[1][2] < g.BODY_Z[1] - 0.02
    assert tips[1][2] - (g.GUIDE_H - g.RAIL_H) > 0.015
    # where the cart stops: its leading face against the fingers' flat side
    face_q = M.Q_STOP - g.CART_HALF_LEN
    assert g.to_station(face_q, (0, 0, 0))[1] == pytest.approx(M.STOP_Y - M.FINGER_HALF, abs=0.001)


def test_the_right_fingers_come_down_clear_of_the_handle_and_push_it_past_its_toggle_to_the_stop():
    deck_z = -g.RAIL_H
    tip_z = M.PUSH_Z
    assert tip_z > g.BODY_Z[1] + 0.015  # pushing above the cart's top, never over it (x stays outboard)
    assert tip_z - deck_z > 0.08
    edge_from = M.PUSH_FROM_X + M.FINGER_EDGE
    assert g.lever_corners(g.STOWED)[:, 0].min() - edge_from == pytest.approx(M.PUSH_CLEAR)
    edge_to = M.PUSH_TO_X + M.FINGER_EDGE
    assert edge_to < -g.BODY_X - 0.005  # the fingers stop short of the cart's side
    # the edge, at tip height, passes the handle's face at every angle up to the stop
    for d in np.arange(-40, 12.01, 1.0):
        r = (M.PUSH_H - g.LEVER_HX * math.sin(DEG(d))) / math.cos(DEG(d))
        assert g.PADDLE_Z[0] <= r <= g.PADDLE_Z[1], d  # it meets the paddle, not the thin bar
    assert edge_to > g.handle_face_x(g.ENGAGED, M.PUSH_H)
    assert edge_to > g.handle_face_x(math.radians(5.0), M.PUSH_H) + 0.003  # pushed well past the toggle point
    # the paddle is wider along the cart than the fingers' play in where the cart stops
    assert g.PADDLE_HY >= 0.015


def test_both_arms_end_clear_of_the_braked_cart():
    s = cart(brake=g.ENGAGED)
    for side in ("left", "right"):
        _, _, tcps = _run(side)
        tcp, rot = tcps[-1]
        seg = M._seg(tcp, tcp + rot[:, 2] * PAD_TIP_PAST_TCP)
        assert M.cart_distance(s, seg).min() - M.PAD_R > M.CLEAR, side


def test_the_right_arm_stays_above_the_rolling_cart_until_it_has_stopped():
    s0 = cart(q=g.Q_START)
    _, _, tcps = _run("right")
    t_stop = M.arrival_s(mu=0.0)
    for k, (tcp, rot) in enumerate(tcps):
        t = (k + 1) * DT
        if t > M.arrival_s(mu=0.009) + 0.5:
            break
        for mu in (0.0, 0.006, 0.009):
            q = max(M.Q_STOP, g.Q_START - 0.5 * g.accel(mu) * t * t)
            st = BodyState(pos=g.origin_at(q), quat_xyzw=s0.quat_xyzw, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=s0.low, high=s0.high, joints={"brake": g.STOWED})
            seg = M._seg(tcp, tcp + rot[:, 2] * PAD_TIP_PAST_TCP)
            assert M.cart_distance(st, seg).min() - M.PAD_R > 0.0, (t, mu)
    assert t_stop > 0
