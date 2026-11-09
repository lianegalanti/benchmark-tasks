"""Offline checks (no Isaac Sim): the line's motion, the tub and lid shapes, the lidding rule and score, and that the
demo is reachable, on time and clear of the tubs and lids.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

import geometry as g
import make_lid
from yam_arena import ik, station
from yam_arena.constants import FINGER_OPEN, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
DT = 1 / 20
FAR = [np.array([0.2, 0.3, 1.2]), np.array([0.3, -0.3, 1.0])]  # grasp points well away from every lid
T1, T2, T3 = M.TUBS
A, B, C = M.LIDS
V = np.array([0.0, M.SPEED, 0.0])


def state(pos, vel=(0.0, 0.0, 0.0), tilt_deg=0.0, half=(0.05, 0.05, 0.02)):
    pos = np.asarray(pos, float)
    a = math.radians(tilt_deg) / 2
    quat = np.array([math.sin(a), 0.0, 0.0, math.cos(a)])
    return ObjectState(pos, quat, np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def carrier(y):
    return state((M.LINE_X, y, M.CARRIER_Z), V)


def on_table(lid):
    return state((M.LID_X, M.LID_YS[lid], M.LID_REST_Z))


def seated_on(y, dx=0.0, dy=0.0, dz=0.0, tilt_deg=0.0, vel=V):
    """A lid seated on the tub whose carrier is at ``y``: its plate's underside on the rim."""
    return state((M.LINE_X + dx, y + dy, M.RIM_Z - g.PLATE_UNDER + dz), vel, tilt_deg)


def scene(ys, lids=None):
    """Carriers at ``ys`` (one per tub), lids on the table unless given."""
    objects = {f"carrier_{t}": carrier(y) for t, y in zip(M.TUBS, ys, strict=True)}
    objects.update({n: on_table(n) for n in M.LIDS})
    objects.update(lids or {})
    return objects


def play(frames, tcps=FAR):
    """Run the monitor over states from a reset; return (score, success) after the last."""
    M.reset_state()
    for objects in frames:
        M.update(objects, tcps)
    return M.score(frames[-1]), M.success(frames[-1])


BEFORE, AT = M.END_Y - 0.01, M.END_Y + 0.001  # a carrier just before and just at the end line
UP = (0.3, 0.1, -0.2)  # carriers upstream, nowhere near the line


# -- the line ------------------------------------------------------------------------------------------------------


def test_it_loads_with_the_tubs_driven_and_only_the_lids_free():
    assert TASK.free_objects == M.LIDS
    assert TASK.score_over == "end"
    for t in M.TUBS:
        assert f"carrier_{t}" in TASK.names
        assert all(f"{t}_{side}" in TASK.names for side in ("front", "back", "left", "right"))


def test_the_line_runs_at_3_cm_per_s_and_brings_a_tub_every_10_s():
    for name in M.TUBS:
        path = M.PATHS[name]
        assert np.linalg.norm(path(0.0)[1]) == 0.0 and np.linalg.norm(path(1 / 240)[1]) < 1e-3  # starts from rest
        assert path(3.0)[1][1] == pytest.approx(M.SPEED)
        assert M.carrier_y(name, M.arrival(name, 0.3)) == pytest.approx(0.3)
    gaps = np.diff([M.arrival(n, M.MEET_Y) for n in M.TUBS])
    assert gaps == pytest.approx([M.SPACING / M.SPEED] * 2) and gaps[0] > 10.0
    ts = np.arange(0.0, 40.0, 1 / 240)
    for name in M.TUBS:
        ps = np.array([M.PATHS[name](t)[0] for t in ts])
        vs = np.array([M.PATHS[name](t)[1] for t in ts])
        assert np.max(np.linalg.norm(np.diff(ps, axis=0), axis=1)) < 0.16 / 240  # continuous
        on = ts < M.arrival(name, M.TURN_Y) - 0.05
        assert np.allclose(np.gradient(ps[on, 1], ts[on])[5:-5], vs[on, 1][5:-5], atol=2e-3)
        assert np.max(np.abs(np.gradient(vs[on, 1], ts[on])[5:-5])) < 0.1  # m/s^2: a seated lid never slides


def test_tub_walls_move_exactly_with_their_carrier_and_never_touch_it_or_each_other():
    walls = {w.name: w for w in M.WALLS}
    for t in M.TUBS:
        for when in (0.0, 0.3, 7.0, 30.0, 45.0):
            c = M.PATHS[t](when)[0]
            for side, (dx, dy), _ in g.tub_walls():
                p, v = walls[f"{t}_{side}"].path(when)
                assert p == pytest.approx(c + np.array([dx, dy, M.WALL_Z - M.CARRIER_Z]))
                assert v == pytest.approx(M.PATHS[t](when)[1])
        boxes = [(np.array([dx, dy, M.WALL_Z]), np.array(size) / 2) for _, (dx, dy), size in g.tub_walls()]
        boxes.append((np.array([0.0, 0.0, M.CARRIER_Z]), np.array(M.CARRIER) / 2))
        for i, (p, h) in enumerate(boxes):
            for q, k in boxes[i + 1:]:
                gap = np.max(np.abs(p - q) - h - k)
                assert gap >= g.WALL_GAP - 1e-9  # separated along some axis by at least 1 mm
        lo = min(dx - s[0] / 2 for _, (dx, _), s in g.tub_walls())
        assert lo == pytest.approx(-g.TUB / 2) and M.RIM_Z - M.TOP == pytest.approx(0.073)


def test_carriers_stay_on_the_bed_apart_and_turn_down_clear_of_the_table():
    assert M.TURN_Y - M.CARRIER[1] / 2 > M.TABLE_LEFT_Y > M.BED[1] / 2
    assert M.STARTS[T3] - M.CARRIER[1] / 2 > -M.BED[1] / 2
    for a, b in zip(M.TUBS, M.TUBS[1:]):
        assert M.STARTS[a] - M.STARTS[b] - g.LID > 0.12  # seated lids on neighbouring tubs are 14 cm apart


# -- the tub and the lid -------------------------------------------------------------------------------------------


def test_the_lid_asset_matches_the_geometry():
    assert (HERE / "assets" / "lid.usda").read_text() == make_lid.usda()


def test_the_skirt_fits_over_the_tub_with_1_cm_to_spare_and_keeps_a_seated_lid_centred():
    assert g.SKIRT_IN - g.TUB / 2 == pytest.approx(0.010)
    assert g.CLEAR < M.CENTRE_M  # however a seated lid sits, its centre is within the 1.5 cm allowed
    assert g.SKIRT_D > 0.01  # the skirt overlaps the walls by 1.5 cm when seated
    parts = {n: (np.array(c), np.array(s)) for n, c, s in g.lid_parts()}
    for n in ("skirt_front", "skirt_back", "skirt_left", "skirt_right"):
        c, s = parts[n]
        assert c[2] + s[2] / 2 == pytest.approx(g.PLATE_UNDER)  # hangs from the plate
        assert np.max(np.abs(c[:2]) + s[:2] / 2) == pytest.approx(g.LID / 2)
        assert np.max(np.abs(c[:2]) - s[:2] / 2) == pytest.approx(g.SKIRT_IN) or n in ("skirt_left", "skirt_right")
    assert g.LID < 0.12 and g.TAB[1] < 0.094 - 0.03  # the grip tab fits the open gripper with 3 cm to spare


def test_the_lids_lie_apart_on_the_table_clear_of_the_line():
    ys = sorted(M.LID_YS.values())
    assert min(np.diff(ys)) - g.LID >= 0.02
    assert M.LID_X + g.LID / 2 < M.LINE_X - M.BED[0] / 2 - 0.05
    assert M.LID_REST_Z + g.SKIRT_BOTTOM == pytest.approx(M.TOP)


# -- scoring -------------------------------------------------------------------------------------------------------


def test_nothing_scores_at_the_start_or_before_a_tub_reaches_the_end_line():
    assert play([scene(UP)]) == (0.0, False)
    assert play([scene((BEFORE, 0.1, -0.2), {A: seated_on(BEFORE)})]) == (0.0, False)
    assert M.judged() == {}


def test_each_lidded_tub_scores_a_third_and_all_three_succeed():
    one = scene((AT, 0.2, -0.05), {A: seated_on(AT)})
    assert play([one]) == (pytest.approx(1 / 3), False)
    assert M.judged() == {T1: A}
    frames = [
        scene((AT, AT - 0.25, AT - 0.5), {B: seated_on(AT)}),
        scene((AT + 0.25, AT, AT - 0.25), {B: seated_on(AT + 0.25), A: seated_on(AT)}),
        scene((AT + 0.5, AT + 0.25, AT), {B: seated_on(AT + 0.5), A: seated_on(AT + 0.25), C: seated_on(AT)}),
    ]
    assert play(frames) == (pytest.approx(1.0), True)
    assert play(frames[:2]) == (pytest.approx(2 / 3), False)
    # the middle tub went by without a lid: two of three
    frames[1] = scene((AT + 0.25, AT, AT - 0.25), {B: seated_on(AT + 0.25)})
    assert play(frames) == (pytest.approx(2 / 3), False)
    assert M.judged()[T2] is None


def test_a_tub_is_judged_once_at_the_end_line():
    lidded = scene((AT, 0.2, -0.05), {A: seated_on(AT)})
    bare = scene((AT + 0.05, 0.25, 0.0))
    assert play([lidded, bare])[0] == pytest.approx(1 / 3)  # the lid falls off past the line: still counts
    late = scene((AT + 0.05, 0.25, 0.0), {A: seated_on(AT + 0.05)})
    assert play([bare_at(AT), late])[0] == 0.0  # lidded only after passing the line: never counts


def bare_at(y):
    return scene((y, 0.2, -0.05))


@pytest.mark.parametrize(
    "lid, why",
    [
        (seated_on(AT, dy=0.02), "centre 2 cm off the axis"),
        (seated_on(AT, dx=-0.016), "centre 1.6 cm off the axis"),
        (seated_on(AT, dz=0.015), "resting on the skirt on the rim"),
        (seated_on(AT, dz=-0.03), "inside the tub"),
        (seated_on(AT, tilt_deg=8), "tilted 8 degrees"),
        (seated_on(AT, tilt_deg=180), "upside down"),
        (seated_on(AT, vel=(0.0, 0.0, 0.0)), "not moving with the tub"),
        (on_table(A), "still on the table"),
    ],
)
def test_a_lid_that_isnt_seated_doesnt_count(lid, why):
    assert play([scene((AT, 0.2, -0.05), {A: lid})])[0] == 0.0, why


def test_a_lid_within_the_limits_counts():
    for lid in (seated_on(AT, dy=0.01), seated_on(AT, dx=0.014), seated_on(AT, dz=0.004), seated_on(AT, tilt_deg=4)):
        assert play([scene((AT, 0.2, -0.05), {A: lid})])[0] == pytest.approx(1 / 3)


def test_a_lid_still_held_doesnt_count():
    objects = scene((AT, 0.2, -0.05), {A: seated_on(AT)})
    tab = objects[A].pos + np.array([0.0, 0.0, g.TAB_CENTRE_Z])
    assert play([objects], tcps=[tab + [0, 0, 0.005], FAR[1]])[0] == 0.0
    assert play([objects], tcps=[FAR[0], tab + [0.0, 0.04, 0.0]])[0] == 0.0
    assert play([objects], tcps=[tab + [0.0, 0.0, 0.06], FAR[1]])[0] == pytest.approx(1 / 3)
    assert play([objects], tcps=None)[0] == 0.0


def test_one_lid_covers_one_tub():
    # a lid between two tubs is seated on neither
    objects = scene((AT, AT - 0.25, -0.2), {A: state((M.LINE_X, AT - 0.125, M.RIM_Z - g.PLATE_UNDER), V)})
    assert play([objects])[0] == 0.0


def test_untouched_tubs_are_all_judged_well_inside_the_run_limit():
    assert M.arrival(T3, M.END_Y) < 35.0  # the run limit is 60 s
    ts = np.arange(0.0, 60.0, DT)
    M.reset_state()
    for t in ts:
        M.update(scene(tuple(M.carrier_y(n, t) for n in M.TUBS)), FAR)
    assert M.judged() == {T1: None, T2: None, T3: None}
    assert M.score({}) == 0.0


# -- the demo ------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, out = 0.0, []
    for w in plan:
        out.append((t, t + w.duration, w))
        t += w.duration
    return out


def test_the_demo_is_reachable_and_within_the_joint_limits():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)
    assert np.max(np.abs(np.diff(actions[:, :6], axis=0))) / DT < 2.0  # rad/s
    assert arm.duration < 40


def test_the_demo_follows_each_tub_and_lets_go_well_before_the_end_line():
    plan = M.demo()["left"]
    for tub in M.TUBS:
        tracked = [(t1, w) for _, t1, w in _spans(plan) if w.label.endswith(f" {tub}")
                   and not w.label.startswith("wait")]
        for t1, w in tracked:  # every tracking waypoint is over the tub's axis at its time
            assert w.pos[0] == M.LINE_X and w.pos[1] == pytest.approx(M.carrier_y(tub, t1), abs=1e-9)
        down = min(t1 for t1, w in tracked if w.label.startswith("lower"))
        assert down - DT >= M.arrival(tub, M.MEET_Y) - 1e-9
        seated = max(t1 for t1, w in tracked if w.label.startswith("lower"))
        assert M.arrival(tub, M.END_Y) - seated >= 4.5  # the lid is on the rim 4.5 s or more before the end line
        opened = max(t1 for t1, w in tracked if w.label.startswith("open on"))
        assert M.arrival(tub, M.END_Y) - opened >= 3.0  # and let go (fingers open) 3 s or more before it
        release = [w for _, _, w in _spans(plan) if w.label == f"release on {tub}"][-1]
        assert release.finger == FINGER_OPEN
        under = release.pos[2] - M.TCP_OVER_ORIGIN + g.PLATE_UNDER
        assert under - M.RIM_Z == pytest.approx(M.DROP_M)


def test_the_demo_has_time_to_spare_before_each_tub():
    plan = M.demo()["left"]
    waits = [w.duration for w in plan if w.label.startswith("wait for")]
    assert len(waits) == 3 and min(waits) >= 0.5


def test_the_demo_holds_still_while_the_slow_gripper_closes_and_opens():
    """The gripper takes about 1 s to close on the fin (measured in Isaac Sim): the arm stays at the grasp for 1.3 s
    or more after the close command, and at the rim for 1.3 s or more after the open command, before moving off."""
    spans = _spans(M.demo()["left"])
    for lid in M.LIDS:
        close = next(t0 for t0, _, w in spans if w.label == f"close on {lid}")
        lift = next(t0 for t0, _, w in spans if w.label == f"lift {lid}")
        assert lift - close >= 1.3
        held = [w for _, _, w in spans if w.label in (f"close on {lid}", f"grip {lid}")]
        assert all(w.pos == held[0].pos for w in held)
    for tub in M.TUBS:
        release = next(t0 for t0, _, w in spans if w.label == f"release on {tub}")
        rise = next(t0 for t0, _, w in spans if w.label == f"rise off {tub}")
        assert rise - release >= 1.3
        heights = {round(w.pos[2], 9) for _, _, w in spans if w.label in (f"release on {tub}", f"open on {tub}")}
        assert len(heights) == 1


def test_the_held_lid_is_carried_gently():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    for t0, t1, w in _spans(plan):
        if w.label.startswith(("lift", "across")):
            ts = np.arange(t0, t1 + 1e-9, 0.01)
            p = np.array([arm.target(t)[0] for t in ts])
            acc = np.linalg.norm(np.gradient(np.gradient(p, ts, axis=0), ts, axis=0), axis=1)
            assert np.max(acc[2:-2]) < 0.35, w.label


def test_the_demo_grasps_each_lid_by_its_tab():
    plan = M.demo()["left"]
    for tub, lid in M.ORDER.items():
        close = next(w for w in plan if w.label == f"close on {lid}")
        assert close.pos[:2] == (M.LID_X, M.LID_YS[lid])
        tips = close.pos[2] - PAD_TIP_PAST_TCP
        plate_top = M.LID_REST_Z + g.PLATE_T / 2
        assert 0.015 <= tips - plate_top <= 0.02  # the pad tips 1.5 cm or more over the plate
        assert M.LID_REST_Z + g.TAB_TOP - tips >= 0.02  # 2 cm of tab between the pads


def _tub_top(tub, t, lidded):
    return M.RIM_Z - g.PLATE_UNDER + g.TAB_TOP if lidded else M.RIM_Z


def test_the_hand_and_the_held_lid_clear_the_tubs_and_the_other_lids():
    """At every half step: the finger tips are 1.5 cm over every tub's top (its rim, or its lid's tab once lidded) and
    every lid lying on the table that they are over, and a held lid's skirt is 2 cm over every rim it is over, except
    over the tub being lidded while lowering onto it."""
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    reach = abs(FINGER_OPEN) + 0.01  # the open fingers' outer faces, either side along y
    spans = _spans(plan)
    taken, lidded, holding = set(), set(), None
    for t0, t1, w in spans:
        if w.label.startswith("close on"):
            holding = w.label.split()[-1]
        for t in np.arange(t0, t1, DT / 2):
            p = arm.target(t)[0]
            tips = p[2] - PAD_TIP_PAST_TCP
            for tub in M.TUBS:
                y = M.carrier_y(tub, t)
                if w.label.endswith(f" {tub}") and not w.label.startswith("wait"):
                    continue  # lowering onto, releasing on, rising off this tub
                if abs(p[0] - M.LINE_X) < g.LID / 2 + 0.02 and abs(p[1] - y) < g.LID / 2 + reach:
                    assert tips - _tub_top(tub, t, tub in lidded) > 0.015, (w.label, t, tub)
                if holding and abs(p[0] - M.LINE_X) < g.LID + 0.01 and abs(p[1] - y) < g.LID + 0.01:
                    skirt = p[2] - M.TCP_OVER_ORIGIN + g.SKIRT_BOTTOM
                    top = _tub_top(tub, t, tub in lidded) if tub in lidded else M.RIM_Z
                    assert skirt - top > 0.02, (w.label, t, tub)
            for lid in M.LIDS:
                if lid in taken or lid == holding:
                    continue
                ly = M.LID_YS[lid]
                if abs(p[0] - M.LID_X) < g.LID / 2 + 0.02 and abs(p[1] - ly) < g.LID / 2 + reach:
                    if w.label.endswith(lid):
                        continue  # going down to this lid to take it
                    assert tips - (M.LID_REST_Z + g.TAB_TOP) > 0.015, (w.label, t, lid)
        if w.label.startswith("release on"):
            lidded.add(w.label.split()[-1])
            if holding:
                taken.add(holding)
            holding = None


def test_the_demo_ends_clear_of_every_lid():
    plan = M.demo()["left"]
    end = np.array(plan[-1].pos)
    t_end = sum(w.duration for w in plan)  # after this, yam_scripted holds the last waypoint
    for tub in M.TUBS:
        for t in np.arange(t_end - 4.0, 40, 0.5):
            tab = np.array([M.LINE_X, M.carrier_y(tub, t), M.RIM_Z - g.PLATE_UNDER + g.TAB_CENTRE_Z])
            assert np.linalg.norm(end - tab) > M.LET_GO_M + 0.03
    right = ik.tcp_pose(np.array(READY), station.BASES["right"])[0]  # yam_scripted holds the right arm at READY
    for tub in M.TUBS:
        for t in np.arange(0, 40, 0.5):
            tab = np.array([M.LINE_X, M.carrier_y(tub, t), M.RIM_Z - g.PLATE_UNDER + g.TAB_CENTRE_Z])
            assert np.linalg.norm(right - tab) > M.LET_GO_M + 0.03
