"""Offline checks (no Isaac Sim): the cart's motion, the tray and nest shapes, the scoring rules, and that the demo is
reachable, on time and clear of the cart, the trays and the other arm.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

import geometry as g
import make_tray
from yam_arena import ik, station
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
DT = 1 / 20
FAR = [np.array([0.2, 0.3, 1.2]), np.array([0.3, -0.3, 1.0])]  # grasp points well away from every tray
V = np.array([0.0, M.SPEED, 0.0])
HALF_TRAY = np.array(M.TRAY_SIZE) / 2
HALF_PART = np.array(g.PART) / 2


def state(pos, vel=(0.0, 0.0, 0.0), tilt_deg=0.0, half=HALF_TRAY):
    pos = np.asarray(pos, float)
    a = math.radians(tilt_deg) / 2
    quat = np.array([math.sin(a), 0.0, 0.0, math.cos(a)])
    return ObjectState(pos, quat, np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def cart(y):
    return state((M.LINE_X, y, M.CART_Z), V, half=np.array(g.CART) / 2)


def in_nest(y, dx=0.0, dy=0.0, dz=0.0, tilt_deg=0.0, vel=V):
    """A tray seated in the nest of the cart at ``y``."""
    return state((M.LINE_X + dx, y + dy, M.CART_TOP - g.UNDER + dz), vel, tilt_deg)


def on_outfeed(dx=0.0, dy=0.0, dz=0.0, tilt_deg=0.0, vel=(0.0, 0.0, 0.0)):
    return state((M.OUT_XY[0] + dx, M.OUT_XY[1] + dy, M.PAD_TOP - g.UNDER + dz), vel, tilt_deg)


def on_stand():
    return state((*M.STAND_XY, M.PAD_TOP - g.UNDER))


def parts_in(tray, spill=None):
    """The two parts lying in ``tray``'s compartments (level tray); ``spill`` moves one onto the table."""
    out = {}
    for n, s in zip(M.PARTS, (-1, 1), strict=True):
        p = tray.pos + np.array([0.0, s * g.POCKET_C, g.FLOOR_TOP + g.PART[2] / 2])
        out[n] = state(p, tray.lin_vel, half=HALF_PART)
    if spill:
        out[spill] = state((0.25, -0.2, M.TOP + g.PART[2] / 2), half=HALF_PART)
    return out


def scene(y, full, empty, spill=None):
    objects = {"cart": cart(y), "tray_full": full, "tray_empty": empty}
    objects.update(parts_in(full, spill))
    return objects


def play(frames, tcps=FAR):
    """Run the monitor over states from a reset; return (score, success) after the last."""
    M.reset_state()
    for objects in frames:
        M.update(objects, tcps)
    return M.score(frames[-1]), M.success(frames[-1])


AT = M.END_Y + 0.001  # the cart just at the end line
BEFORE = M.END_Y - 0.01
HELD_FULL = state((0.25, -0.3, M.TOP + 0.15))  # the full tray up in the air, taken off the cart


def start():
    return scene(M.START_Y, in_nest(M.START_Y), on_stand())


def swapped(y=AT, **kw):
    return scene(y, kw.pop("full", on_outfeed()), kw.pop("empty", in_nest(y)), **kw)


# -- the cart ------------------------------------------------------------------------------------------------------


def test_it_loads_with_the_cart_driven_and_the_trays_and_parts_free():
    assert set(TASK.free_objects) == {"tray_full", "tray_empty", *M.PARTS}
    assert TASK.score_over == "end"
    assert {"cart", "nest_near", "nest_far", "nest_left", "nest_right", "stand", "outfeed"} <= set(TASK.names)


def test_the_cart_runs_at_6_cm_per_s_without_stopping():
    assert np.linalg.norm(M.PATH(0.0)[1]) == 0.0 and np.linalg.norm(M.PATH(1 / 240)[1]) < 1e-3  # starts from rest
    ts = np.arange(0.0, 30.0, 1 / 240)
    ps = np.array([M.PATH(t)[0] for t in ts])
    vs = np.array([M.PATH(t)[1] for t in ts])
    on = (ts > M.RAMP_S) & (ts < M.arrival(M.TURN_Y) - 0.05)
    assert np.allclose(vs[on, 1], M.SPEED)  # constant speed from the ramp to the table's edge
    assert np.max(np.linalg.norm(np.diff(ps, axis=0), axis=1)) < 0.16 / 240  # continuous
    assert np.max(np.abs(np.gradient(vs[:, 1], ts))[ts < M.arrival(M.TURN_Y) - 0.05]) < 0.25  # m/s^2: trays don't slide
    assert M.cart_y(M.arrival(0.1)) == pytest.approx(0.1)
    assert M.arrival(M.END_Y) == pytest.approx(16.75, abs=0.01)


def _quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _corners(centre, size, quat):
    rot = _quat_matrix(quat)
    half = np.array(size) / 2
    return np.array([np.array(centre) + rot @ (half * s) for s in np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3)])


def test_the_guide_plates_move_with_the_cart():
    plates = {w.name: w for w in M.NEST}
    for when in (0.0, 0.3, 7.0, 20.0, 30.0):
        c, v = M.PATH(when)
        for side, (dx, dy, dz), _, quat in g.nest_guides():
            p, wv = plates[f"nest_{side}"].path(when)
            assert p == pytest.approx(c + np.array([dx, dy, M.CART_TOP + dz - M.CART_Z]))
            assert wv == pytest.approx(v)
            assert plates[f"nest_{side}"].rot_xyzw == pytest.approx(quat)


def test_the_guide_plates_form_a_funnel_and_never_touch_the_cart_or_each_other():
    """Each plate lies outside the pocket, 1 mm above the cart, within the cart's top, and short of the corners; its
    face toward the pocket rises outward at 55 degrees from the pocket's edge."""
    for side, centre, size, quat in g.nest_guides():
        k = _corners(centre, size, quat)
        rot = _quat_matrix(quat)
        u = rot[:2, 0] / np.linalg.norm(rot[:2, 0])  # outward
        assert abs(np.linalg.norm(quat) - 1) < 1e-9
        across, along = k[:, :2] @ u, k[:, :2] @ np.array([-u[1], u[0]])
        assert across.min() == pytest.approx(g.NEST_IN / 2)  # the pocket's edge
        assert across.max() == pytest.approx(g.GUIDE_OUT) and g.GUIDE_OUT <= g.CART[0] / 2
        assert np.abs(along).max() <= g.NEST_IN / 2 - g.GAP + 1e-9  # 1 mm short of the neighbouring plate
        assert k[:, 2].min() == pytest.approx(g.GAP) and k[:, 2].max() == pytest.approx(g.GUIDE_TOP, abs=1e-3)
        normal = rot[:, 2]  # the face toward the pocket
        assert normal @ np.array([*u, 0.0]) < 0 and math.degrees(math.acos(normal[2])) == pytest.approx(g.GUIDE_DEG)
    # neighbouring plates: one inside |along| <= NEST_IN/2 - GAP, the other outside across >= NEST_IN/2, so disjoint
    assert g.GUIDE_DEG > math.degrees(math.atan(0.8)) + 10  # steeper than the tray's friction angle (0.8)


def test_the_cart_stays_on_the_bed_and_turns_down_past_the_table():
    assert M.START_Y - g.CART[1] / 2 >= -M.BED[1] / 2
    assert M.TURN_Y - g.CART[1] / 2 > M.TABLE_LEFT_Y
    assert M.PATH(40.0)[0][2] == pytest.approx(M.CART_Z - M.DROP)


# -- the trays -----------------------------------------------------------------------------------------------------


def test_the_tray_asset_matches_the_geometry():
    assert (HERE / "assets" / "tray.usda").read_text() == make_tray.usda()


def test_the_tray_shapes():
    parts = {n: (np.array(c), np.array(s)) for n, c, s in g.tray_parts()}
    lo = min(c[2] - s[2] / 2 for c, s in parts.values())
    hi = max(c[2] + s[2] / 2 for c, s in parts.values())
    assert lo == pytest.approx(g.UNDER) and hi == pytest.approx(-g.UNDER)  # origin at the bounding box's centre
    assert max(np.max(np.abs(c[:2]) + s[:2] / 2) for c, s in parts.values()) == pytest.approx(g.TRAY / 2)
    assert g.RIB_TOP - g.TRAY_H == pytest.approx(0.04)  # 4 cm of rib above the rim to pinch
    assert abs(FINGER_OPEN) * 2 - g.RIB_T >= 0.08  # 4 cm either side of the rib in the open gripper
    # the parts fit their compartments with room to spare and lie below the rim
    assert g.PART[1] <= g.POCKET_Y - 0.01 and g.PART[0] <= g.INNER - 0.03
    assert g.FLOOR_TOP + g.PART[2] <= g.RIM - 0.01


def test_a_tray_fits_the_nest_with_1_25_cm_to_spare_and_stays_centred_enough():
    assert (g.NEST_IN - g.TRAY) / 2 == pytest.approx(0.0125)
    assert (g.NEST_IN - g.TRAY) / 2 < M.CENTRE_M  # wherever it lies in the pocket, it counts as centred
    assert M.NEST_TOP - M.CART_TOP > 0.02  # the funnel holds a tray in the pocket
    assert g.NEST_IN / 2 + g.FLARE - g.TRAY / 2 >= 0.025  # 2.5 cm or more of play each side at the funnel's top


def test_the_layout_is_clear():
    cart_near = M.LINE_X - g.CART[0] / 2
    for xy in (M.STAND_XY, M.OUT_XY):
        assert xy[0] + M.PAD[0] / 2 < cart_near - 0.02  # the stand and outfeed are 2 cm or more off the cart's path
        assert M.PAD[0] - g.TRAY >= 0.04 - 1e-9  # a tray fits on them with 2 cm to spare on every side
    assert M.LINE_X - M.BED[0] / 2 > M.STAND_XY[0] + M.PAD[0] / 2


# -- scoring -------------------------------------------------------------------------------------------------------


def test_nothing_scores_at_the_start_or_untouched():
    assert play([start()]) == (0.0, False)
    frames = [scene(M.cart_y(t), in_nest(M.cart_y(t)), on_stand()) for t in np.arange(0.0, 30.0, DT)]
    assert play(frames) == (0.0, False)
    assert M._STATE["load"] == 0.0 and not M._STATE["lifted"]


def test_the_full_swap_scores_1():
    assert play([swapped()]) == (1.0, True)
    assert play([swapped(), swapped(AT + 0.05)]) == (1.0, True)


def test_each_half_scores_half():
    assert play([scene(AT, on_outfeed(), on_stand())])[0] == 0.5  # unloaded only
    assert play([scene(AT, HELD_FULL, in_nest(AT), spill="part_a")])[0] == 0.5  # loaded only: the full tray spilt
    assert play([scene(AT, in_nest(AT + 0.5), in_nest(AT))])[0] == 0.5  # seated, the full tray on no cart, never lifted


def test_partial_credit_for_a_tray_taken_off_but_not_on_the_outfeed():
    lift = scene(-0.2, HELD_FULL, on_stand())
    assert play([lift, scene(AT, HELD_FULL, in_nest(AT))]) == (0.75, False)  # still held
    elsewhere = state((0.25, -0.2, M.TOP - g.UNDER))  # set down on the table
    assert play([lift, scene(AT, elsewhere, on_stand())])[0] == 0.25
    # not lifted (pushed or slid off): nothing
    assert play([scene(AT, elsewhere, on_stand())])[0] == 0.0


def test_partial_credit_for_an_empty_tray_on_the_cart_but_not_seated():
    perched = in_nest(AT, dy=0.07, dz=0.016, tilt_deg=8)  # resting on a nest wall
    assert play([swapped(empty=perched)]) == (0.75, False)
    assert play([swapped(empty=in_nest(AT, dz=0.02))])[0] == 0.75  # level but 2 cm up


@pytest.mark.parametrize(
    "empty, why",
    [
        (on_stand(), "never moved"),
        (in_nest(AT, vel=(0.0, 0.0, 0.0)), "not moving with the cart"),
        (in_nest(AT, dz=0.05), "above the cart"),
        (state((M.LINE_X, AT - 0.2, M.TOP - g.UNDER)), "dropped on the table"),
    ],
)
def test_an_empty_tray_not_on_the_cart_scores_nothing_for_the_nest(empty, why):
    assert play([swapped(empty=empty)])[0] == 0.5, why


def test_the_nest_is_judged_once_at_the_end_line():
    assert play([swapped(BEFORE)])[0] == 0.5  # not judged yet
    late = [swapped(empty=on_stand()), swapped(AT + 0.02)]
    assert play(late)[0] == 0.5  # seated only after the cart passed the line
    gone = [swapped(), swapped(AT + 0.1, empty=state((0.3, 0.6, M.TOP - g.UNDER)))]
    assert play(gone)[0] == 1.0  # what happens after the line doesn't matter


def test_a_tray_still_held_doesnt_count():
    objects = swapped()
    grip = objects["tray_empty"].pos + np.array([0.0, 0.0, g.GRIP_Z])
    assert play([objects], tcps=[grip, FAR[1]])[0] == 0.5
    grip = objects["tray_full"].pos + np.array([0.0, 0.0, g.GRIP_Z])
    assert play([objects], tcps=[FAR[0], grip + [0.0, 0.0, 0.01]])[0] == 0.5
    assert play([objects], tcps=[FAR[0], grip + [0.0, 0.0, 0.06]])[0] == 1.0
    assert play([objects], tcps=None)[0] == 0.0


@pytest.mark.parametrize(
    "full, spill, why",
    [
        (on_outfeed(tilt_deg=15), None, "tilted 15 degrees"),
        (on_outfeed(tilt_deg=180), None, "upside down"),
        (on_outfeed(), "part_a", "a part spilled"),
        (on_outfeed(dx=0.08), None, "centre off the outfeed's top"),
        (on_outfeed(dz=0.02), None, "2 cm above the outfeed"),
        (on_outfeed(vel=(0.0, 0.0, -0.3)), None, "still falling"),
    ],
)
def test_a_full_tray_not_set_down_whole_on_the_outfeed_doesnt_score_its_half(full, spill, why):
    assert play([swapped(full=full, spill=spill)])[0] == 0.5, why


def test_a_dropped_full_tray_scores_nothing_for_its_half():
    lift = scene(-0.2, HELD_FULL, on_stand())
    dropped = state((0.25, -0.2, M.TOP + 0.04), tilt_deg=60)
    assert play([lift, scene(AT, dropped, in_nest(AT))])[0] == 0.5
    assert play([lift, scene(AT, HELD_FULL, in_nest(AT), spill="part_b")])[0] == 0.5


def test_within_the_limits_counts():
    for empty in (in_nest(AT, dx=0.012), in_nest(AT, dy=-0.012), in_nest(AT, dz=0.004), in_nest(AT, tilt_deg=4)):
        assert play([swapped(empty=empty)])[0] == 1.0
    for full in (on_outfeed(dx=0.06), on_outfeed(dz=0.008), on_outfeed(tilt_deg=8)):
        assert play([swapped(full=full)])[0] == 1.0


# -- the demo ------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, out = 0.0, []
    for w in plan:
        out.append((t, t + w.duration, w))
        t += w.duration
    return out


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable_and_within_the_joint_limits(side):
    plan = M.demo()[side]
    arm = ArmPlan(plan, station.BASES[side])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert np.max(np.abs(np.diff(actions[:, :6], axis=0))) / DT < 2.0  # rad/s
    assert arm.duration < 30  # the run limit is 30 s


def test_the_arms_ease_into_and_out_of_the_cart_s_speed():
    for side in ("left", "right"):
        arm = ArmPlan(M.demo()[side], station.BASES[side])
        for t0, t1, w in _spans(M.demo()[side]):
            if w.label == "match the cart's speed":
                end = t1
        p0, p1 = arm.target(end - DT)[0], arm.target(end)[0]
        assert p1[1] == pytest.approx(M.cart_y(end)) and (p1[1] - p0[1]) / DT == pytest.approx(M.SPEED, rel=0.05)


def test_the_demo_tracks_the_cart_and_has_time_to_spare():
    for side in ("left", "right"):
        tracked = [(t1, w) for _, t1, w in _spans(M.demo()[side])
                   if w.duration == DT and not w.label.startswith(("match", "slow down"))]
        assert len(tracked) > 30
        for t1, w in tracked:  # every tracking waypoint is over the cart's centre at its time
            assert w.pos[0] == M.LINE_X and w.pos[1] == pytest.approx(M.cart_y(t1), abs=1e-9)
        wait = [w.duration for w in M.demo()[side] if w.label.startswith("wait for")]
        assert len(wait) == 1 and wait[0] >= 1.0
    right = _spans(M.demo()["right"])
    lifted = max(t1 for _, t1, w in right if w.label == "lift the full tray")
    left = _spans(M.demo()["left"])
    down = min(t0 for t0, _, w in left if w.label == "lower into the nest")
    assert down - lifted > 3.0  # the nest is empty long before the empty tray goes in
    let_go = max(t1 for _, t1, w in left if w.label == "let go in the nest")  # fingers open, about to rise
    assert M.arrival(M.END_Y) - let_go > 2.7  # 2.7 s of track to spare
    assert M.cart_y(let_go) < M.END_Y - 0.16


def test_the_demo_grips_the_ribs_above_the_rims():
    for side, label, origin_z in (("right", "close on the full tray", M.FULL_Z - M.SPAWN_GAP),
                                  ("left", "close on the empty tray", M.EMPTY_Z - M.SPAWN_GAP)):
        close = [w for w in M.demo()[side] if w.label == label]
        assert close[-1].finger == pytest.approx(FINGER_CLOSED)
        tips = close[-1].pos[2] - PAD_TIP_PAST_TCP
        assert tips - (origin_z + g.RIM) == pytest.approx(0.015)  # the pad tips 1.5 cm over the rim
        assert origin_z - g.UNDER + g.RIB_TOP - tips >= 0.02  # 2.5 cm of rib between the pads
    release = [w for w in M.demo()["left"] if w.label == "release in the nest"][-1]
    under = release.pos[2] - M.TCP_OVER_UNDER
    assert under - M.CART_TOP == pytest.approx(M.DROP_M)
    out = [w for w in M.demo()["right"] if w.label == "release on the outfeed table"][-1]
    assert out.pos[2] - M.TCP_OVER_UNDER - M.PAD_TOP == pytest.approx(M.DROP_M)


def _holding(side, label):
    """Whether arm ``side`` holds its tray during the waypoint ``label``."""
    order = [w.label for w in M.demo()[side]]
    close = order.index("close on the full tray" if side == "right" else "close on the empty tray")
    release = order.index("release on the outfeed table" if side == "right" else "release in the nest")
    i = order.index(label)
    return close < i <= release


def test_the_hands_and_held_trays_clear_the_cart_and_the_trays_on_it():
    """At every half step: open finger tips 1.5 cm over whatever is on the cart (the full tray's rib until it is taken,
    the nest walls after) when over it, and a held tray's underside 2.5 cm over the nest walls when over the cart,
    except while going down to or into the nest."""
    plans = M.demo()
    reach = abs(FINGER_OPEN) + 0.01
    taken = max(t1 for _, t1, w in _spans(plans["right"]) if w.label == "close on the full tray")
    for side in ("left", "right"):
        arm = ArmPlan(plans[side], station.BASES[side])
        for t0, t1, w in _spans(plans[side]):
            if w.duration == DT:
                continue  # following the cart, going down to it, closing, lifting, releasing
            for t in np.arange(t0, t1, DT / 2):
                p = arm.target(t)[0]
                y = M.cart_y(t)
                near = abs(p[0] - M.LINE_X) < g.CART[0] / 2 + 0.06
                if not (near and abs(p[1] - y) < g.CART[1] / 2 + 0.06):
                    continue
                top = M.NEST_TOP if t > taken else M.CART_TOP + g.RIB_TOP
                if _holding(side, w.label):
                    assert p[2] - M.TCP_OVER_UNDER - M.NEST_TOP > 0.025, (side, w.label, t)
                else:
                    assert p[2] - PAD_TIP_PAST_TCP - top > 0.015, (side, w.label, t)


def test_the_arms_stay_apart():
    plans = {s: ArmPlan(M.demo()[s], station.BASES[s]) for s in ("left", "right")}
    for t in np.arange(0.0, M.END_S, DT):
        gap = np.linalg.norm(plans["left"].target(t)[0] - plans["right"].target(t)[0])
        assert gap > 0.2


def _replay():
    """The demo replayed kinematically: each tray rides the cart, sits where it was, or hangs in its gripper between
    the end of closing and the end of releasing (then drops the 3 mm onto what is below). Frames per control step."""
    plans = M.demo()
    arms = {s: ArmPlan(plans[s], station.BASES[s]) for s in plans}
    spans = {s: _spans(plans[s]) for s in plans}

    def window(side, close, release):
        c = max(t1 for _, t1, w in spans[side] if w.label == close)
        r = max(t1 for _, t1, w in spans[side] if w.label == release)
        return c, r

    full_w = window("right", "close on the full tray", "release on the outfeed table")
    empty_w = window("left", "close on the empty tray", "release in the nest")
    frames, tcps = [], []
    for k in range(1, round(M.END_S / DT) + 1):
        t = k * DT
        y = M.cart_y(t)
        tcp = {s: arms[s].target(t)[0] for s in arms}
        if t < full_w[0]:
            full = in_nest(y)
        elif t < full_w[1]:
            full = state(tcp["right"] - [0.0, 0.0, M.TCP_OVER_ORIGIN], (0.0, 0.0, 0.0))
        else:
            full = on_outfeed()
        if t < empty_w[0]:
            empty = on_stand()
        elif t < empty_w[1]:
            empty = state(tcp["left"] - [0.0, 0.0, M.TCP_OVER_ORIGIN], V if t > empty_w[1] - 1.0 else (0, 0, 0))
        else:
            empty = in_nest(y)
        frames.append(scene(y, full, empty))
        tcps.append([tcp["left"], tcp["right"]])
    return frames, tcps


def test_the_demo_replayed_kinematically_scores_1():
    frames, tcps = _replay()
    M.reset_state()
    best_before = 0.0
    for objects, tcp in zip(frames, tcps, strict=True):
        M.update(objects, tcp)
        if M._STATE["load"] is None:
            best_before = max(best_before, M.score(objects))
    assert M._STATE["lifted"] and M._STATE["load"] == 0.5
    assert M.score(frames[-1]) == 1.0 and M.success(frames[-1])
    assert best_before == 0.5  # before the end line only the full tray's half can count


def test_the_demo_waits_for_the_gripper_to_close_and_open():
    """The gripper takes about 1 s to close or open (measured in Isaac Sim): 1.3 s or more from each close command's
    end to the next move off the grasp, and 1.0 s or more from each open command's end to the rise, still following
    the cart in the nest."""
    for side in ("left", "right"):
        spans = _spans(M.demo()[side])
        labels = [w.label for _, _, w in spans]
        for close, nxt in (("close on", "lift"), ("release", "rise off")):
            for name in {lb for lb in labels if lb.startswith(close)}:
                end = max(t1 for _, t1, w in spans if w.label == name)
                start = min(t0 for t0, _, w in spans if w.label.startswith(nxt) and t0 >= end - 1e-9)
                held = [w for t0, _, w in spans if end - 1e-9 <= t0 < start - 1e-9]
                assert start - end >= (1.3 if close == "close on" else 1.0) - 1e-9, (side, name)
                for w in held:  # holding still, or following the cart at a constant height
                    assert w.pos[2] == pytest.approx(held[0].pos[2]) and w.finger == pytest.approx(held[0].finger)
                    if w.duration == DT:
                        assert w.pos[0] == M.LINE_X


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_moves_gently(side):
    arm = ArmPlan(M.demo()[side], station.BASES[side])
    ts = np.arange(0.0, arm.duration, DT)
    ps = np.array([arm.target(t)[0] for t in ts])
    v = np.diff(ps, axis=0) / DT
    a = np.diff(v, axis=0) / DT
    assert np.max(np.linalg.norm(v, axis=1)) < 0.4  # m/s
    assert np.max(np.linalg.norm(a, axis=1)) < 1.0  # m/s^2: no jumps in speed (smoothstep moves, eased tracking)
