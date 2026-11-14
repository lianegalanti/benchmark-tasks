"""Offline checks (no Isaac Sim): the coupling's geometry (entry, turn, lock, stop, clearances), the seal spring, the air
lines' kinematics, every scoring level and violation, and the demo (reach, joint limits, clearances, end state).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402

EYE = np.eye(3)
PLAN = demo.plan()
STEPS = PLAN[:-1]
FAR = {"left": {"tcp": np.array([0.2, 0.3, 1.1]), "rot": EYE, "opening": 0.094},
       "right": {"tcp": np.array([0.2, -0.3, 1.1]), "rot": EYE, "opening": 0.094}}
TOL = 0.0002     # m: boxes shrunk by this before an overlap test (touching faces are not overlaps)


def shrink(boxes, d=TOL):
    return [(c, r, np.asarray(h) - d) for c, r, h in boxes]


def trailer_boxes(c_ring=0.0):
    return [b for _, b in g.casting_boxes()] + [b for _, b in g.ring_boxes(c_ring)]


def tractor_in_t(phi, s, names=None):
    p, r = g.rel(phi, s)
    return {n: g.place([b], p, r)[0] for n, b in g.tractor_boxes() if names is None or n in names}


def min_gap(a, b):
    return min(g.boxes_gap(x, y, 5) for x in a for y in b)


def any_overlap(a, b):
    return any(g.boxes_overlap(x, y) for x in shrink(a) for y in shrink(b))


def locked_glad():
    return {c: g.tractor_pose(c, 0.0, g.S_LOCK) for c in g.COLOURS}


def step(glad, rings=None, slides=None, arms=FAR):
    rings = {c: g.C_LOCK for c in g.COLOURS} if rings is None else rings
    M.update(g.NOSE, EYE, {f"seal_{c}": v for c, v in rings.items()}, glad, slides or {}, arms, 0.05)


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()
    yield


# -- the casting and the coupling -------------------------------------------------------------------------------------------
def test_glad_hand_full_size():
    pts = np.vstack([g.box_points(*b, 3) for _, b in g.tractor_boxes()])
    size = pts.max(0) - pts.min(0)
    assert 0.105 <= size[0] <= 0.12 and 0.04 <= size[1] <= 0.075 and 0.04 <= size[2] <= 0.05
    assert g.GH_MASS == 0.35
    assert (g.S_LOCK, g.C_LOCK) == pytest.approx((0.004, 0.003))


def test_entry_descent_touches_only_the_seals():
    trailer = trailer_boxes()
    for s in np.linspace(0.04, g.S_REST + 0.0005, 12):
        assert min_gap(list(tractor_in_t(g.ENTRY, s).values()), trailer) > 0.0
    # pressed fully at the entry angle: only the seals touch; the bosses' faces stay 1.5 mm apart, everything else 2.5 mm
    body = {n: b for n, b in tractor_in_t(g.ENTRY, g.S_MIN).items() if not n.startswith("seal")}
    assert min_gap(list(body.values()), trailer_boxes(g.RING_TRAVEL)) >= g.S_MIN - 1e-9
    rest = [b for n, b in body.items() if not n.startswith("boss")]
    assert min_gap(rest, [b for n, b in g.casting_boxes() if not n.startswith("boss")]) >= 0.0025 - 1e-9


def test_turn_while_pressed_is_clear_until_the_stop():
    trailer = trailer_boxes(g.RING_TRAVEL)
    for phi in np.radians(np.linspace(-90, -3, 30)):
        body = [b for n, b in tractor_in_t(phi, g.S_MIN).items() if not n.startswith("seal")]
        assert not any_overlap(body, trailer), math.degrees(phi)
    # the flange runs under the overhang with the full 2.5 mm of play, and its tip tooth rides in the groove
    assert g.S_LOCK - g.S_MIN >= 0.0025 - 1e-9
    lip = [b for n, b in g.lip_boxes()]
    for phi in np.radians(np.linspace(-30, -3, 10)):
        flange = tractor_in_t(phi, g.S_MIN, ["flange", "tip", "tooth"])
        assert min_gap(list(flange.values()), lip) >= 0.0015, math.degrees(phi)
    # the groove's mouth is a funnel: a tooth arriving up to 4 mm off-centre either way is steered in, not stopped
    a = g.RISER_ARC[0] - g.FUNNEL
    mouth = [b for n, b in g.lip_boxes() if n.startswith("funnel")]
    for dr in (-0.004, 0.004):
        r = 0.0508 + dr
        tooth = (np.array([r * math.cos(a), r * math.sin(a), -0.007 + 0.0015]), g.rot_z(a), np.array([0.00125, 0.006, 0.002]))
        assert not any(g.boxes_overlap(tooth, m) for m in mouth)


def test_stop_and_lock():
    trailer = trailer_boxes(g.C_LOCK)
    at_lock = list(tractor_in_t(0.0, g.S_LOCK).values())
    assert not any_overlap(at_lock, trailer)
    # flange on the stop and on the overhang, both pairs
    over = [b for n, b in g.lip_boxes() if n.startswith("overhang")]
    stop = [g.lohi(*g.STOP)]
    fl = [tractor_in_t(0.0, g.S_LOCK, ["flange"])["flange"]]
    assert min_gap(fl, stop) < 0.0003 and min_gap(fl, over) < 0.0003
    p, r = g.rel(0.0, g.S_LOCK)
    own = [g.lohi(*g.FLANGE)]
    assert min_gap(own, g.place(stop, p, r)) < 0.0003 and min_gap(own, g.place(over, p, r)) < 0.0003


def test_lock_is_geometric_along_the_long_axis():
    """Locked, the tip teeth in the grooves stop the tractor glad hand sliding along its long axis either way (as a
    pull on its hose would): about 2 mm of play each way, no more."""
    trailer = trailer_boxes(g.C_LOCK)
    p, r = g.rel(0.0, g.S_LOCK)
    tractor = [b for _, b in g.tractor_boxes()]
    for dx, blocked in ((0.0015, False), (-0.0015, False), (0.003, True), (-0.003, True), (0.016, True)):
        moved = g.place(tractor, p + np.array([dx, 0.0, 0.0]), r)
        assert any_overlap(moved, trailer) == blocked, dx


    # past the stop is blocked
    assert any_overlap(list(tractor_in_t(math.radians(3), g.S_MIN).values()), trailer_boxes(g.RING_TRAVEL))
    # at the locked turn, the ring can be pressed 2 mm or more only with the flanges under the lips:
    # with the flange above the overhang (separation for 2 mm or less) it would be inside the lip
    for s in (g.S_LOCK + 0.0011, g.S_REST - M.LOCK_C):
        assert any_overlap(list(tractor_in_t(0.0, s).values()), trailer_boxes(g.S_REST - s))


def test_seal_spring_numbers():
    weight = g.GH_MASS * 9.81
    assert g.seal_force(g.C_LOCK) > 2 * weight            # the lips, not gravity, set the locked compression
    assert g.seal_force(M.LOCK_C) > weight + g.HOSE_MASS * 9.81   # gravity alone can't press the ring 2 mm
    assert g.seal_force(g.RING_TRAVEL) < 16.0
    # the demo's press: arm z compliance 2.4 mm/N, so the overshoot past full travel gives more than the ring needs
    assert demo.PRESS_OVER / 0.0025 > g.seal_force(g.RING_TRAVEL)


# -- layout ---------------------------------------------------------------------------------------------------------------------
def test_parked_glad_hand_rests_on_its_stand():
    for c in g.COLOURS:
        p, r = g.park_pose(c)
        boxes = [g.place([b], p, r)[0] for _, b in g.tractor_boxes()]
        pads = g.stand_pads(c)
        assert not any_overlap(boxes, pads)
        for pad in pads:
            assert min_gap(boxes, [pad]) < 0.0003           # both pads touch
        com = p + r @ g.GH_COM
        xs = [r.T @ (pad[0] - p) for pad in pads]
        assert min(x[0] for x in xs) < (r.T @ (com - p))[0] < max(x[0] for x in xs)   # between the pads
        assert min(b[0][2] - abs(b[2][2]) for b in boxes) > g.TABLE   # nothing else reaches the table


def test_fixed_parts_do_not_overlap():
    trailer = [g.lohi(*g.BULKHEAD, g.NOSE), g.lohi(*g.BRACKET, g.NOSE)]
    for c in g.COLOURS:
        t = g.TRAILER_GH[c]
        trailer.append(g.lohi((t[0] - 0.018, t[1] - 0.018, g.TABLE + g.BRACKET[1][2]), (t[0] + 0.018, t[1] + 0.018,
                                                                                         t[2] + g.BACK[0][2])))
        trailer += [g.place([b], t, g.R_TRAILER[c])[0] for _, b in g.casting_boxes()]
    stands = [p for c in g.COLOURS for p in g.stand_pads(c)]
    post = [g.lohi(*g.POST, g.TENDER)]
    assert min_gap(trailer, stands + post) > 0.005
    assert min_gap(stands, post) > 0.005
    for b in trailer[:2] + post:      # standing just clear of the table (the table holds nothing up: they are fixed)
        assert b[0][2] - b[2][2] > g.TABLE


def test_trailer_mount_clear_of_the_coupling():
    for c in g.COLOURS:
        t, rt = g.TRAILER_GH[c], g.R_TRAILER[c]
        assert t[2] + g.BACK[0][2] > g.TABLE + g.BRACKET[1][2]
        tractor = [g.place([b], *g.tractor_pose(c, phi, g.S_MIN))[0] for _, b in g.tractor_boxes()
                   for phi in (g.ENTRY, 0.0)]
        fixed = [g.lohi(*g.BULKHEAD, g.NOSE), g.lohi(*g.BRACKET, g.NOSE)]
        assert min_gap(tractor, fixed) > 0.015


def test_lines_start_at_their_parked_glad_hands():
    for c in g.COLOURS:
        q = g.start_chain(c)
        p, r = g.chain_pose(c, q)
        p0, r0 = g.park_pose(c)
        assert np.allclose(p, p0, atol=1e-9) and np.allclose(r, r0, atol=1e-9)
        assert g.L_MIN + 0.03 < q[2] < g.L_MAX - 0.1
        assert abs(q[4]) < math.radians(70) and g.PITCH_LIM[0] < q[1] < g.PITCH_LIM[1]


def test_usd_assets_match_geometry():
    tender = (HERE / "assets/tender.usda").read_text()
    trailer = (HERE / "assets/trailer.usda").read_text()
    for c in g.COLOURS:
        for j in g.CHAIN_JOINTS:
            assert f'"{c}_{j}"' in tender
        assert f'"{c}_gladhand"' in tender and f'"seal_{c}"' in trailer


# -- scoring --------------------------------------------------------------------------------------------------------------------
def test_both_locked_released_still_is_success():
    glad = locked_glad()
    step(glad)
    assert not M.success(None)                 # stillness needs a previous step
    step(glad)
    assert M.success(None) and M.score(None) == pytest.approx(0.9)


def test_held_scores_little_not_success():
    glad = locked_glad()
    p, r = glad["red"]
    tcp, rot = demo.tcp_for(p, r)
    arms = dict(FAR, right={"tcp": tcp, "rot": rot, "opening": 0.044})
    step(glad, arms=arms)
    step(glad, arms=arms)
    assert not M.success(None) and M.score(None) == pytest.approx(0.65)      # red held: 0.2, blue let go: 0.45
    tb, rb = demo.tcp_for(*glad["blue"])
    arms = {"left": {"tcp": tb, "rot": rb, "opening": 0.044}, "right": arms["right"]}
    step(glad, arms=arms)
    assert M.score(None) == pytest.approx(0.4)                               # holding both locked pairs


def test_partial_levels():
    glad = locked_glad()
    glad["blue"] = g.tractor_pose("blue", g.ENTRY, g.S_REST - 0.001)        # faces together, not turned
    step(glad, rings={"blue": 0.001, "red": g.C_LOCK})
    assert M.score(None) == pytest.approx(0.55) and not M.success(None)
    glad["blue"] = g.park_pose("blue")
    step(glad, rings={"blue": 0.0, "red": g.C_LOCK})
    assert M.score(None) == pytest.approx(0.45)
    glad["red"] = g.park_pose("red")
    step(glad, rings={"blue": 0.0, "red": 0.0})
    assert M.score(None) == 0.0
    assert not M.violation()


def test_lock_thresholds():
    def locked(phi=0.0, s=g.S_LOCK, c=g.C_LOCK, tilt=0.0, dx=0.0):
        p, r = g.tractor_pose("red", phi, s)
        return M.pair_state("red", p + g.R_TRAILER["red"] @ [dx, 0, 0], r @ g.rot_x(tilt), c)["locked"]

    assert locked()
    assert locked(phi=math.radians(-4.5)) and not locked(phi=math.radians(-6))
    assert not locked(c=0.0019) and not locked(tilt=math.radians(6)) and not locked(dx=0.0065) and locked(dx=0.005)


def test_drop_is_a_violation():
    glad = locked_glad()
    step(glad)
    p, r = glad["blue"]
    glad["blue"] = (p - [0, 0, 0.04], r)        # 4 cm in a step: 0.8 m/s, nobody holding it
    step(glad)
    assert M.violation() and M.score(None) == 0.0


def test_carried_down_fast_is_not_a_drop():
    glad = locked_glad()
    p, r = glad["blue"]
    tcp, rot = demo.tcp_for(p, r)
    arms = dict(FAR, left={"tcp": tcp, "rot": rot, "opening": 0.044})
    step(glad, arms=arms)
    glad["blue"] = (p - [0, 0, 0.04], r)
    arms = dict(FAR, left={"tcp": tcp - [0, 0, 0.04], "rot": rot, "opening": 0.044})
    step(glad, arms=arms)
    assert not M.violation()


def test_line_pulled_to_end_is_a_violation():
    _, _, l0, *_ = g.start_chain("red")
    step(locked_glad(), slides={"red": g.L_MAX - l0 - 0.002})
    assert M.violation() and M.score(None) == 0.0


def test_crossed_lines_is_a_violation():
    glad = locked_glad()
    glad["blue"] = g.tractor_pose("red", 0.0, g.S_LOCK)      # the blue line's glad hand locked on the red trailer
    glad["red"] = g.park_pose("red")
    step(glad)
    assert M.violation()


def test_off_table_is_a_violation():
    glad = locked_glad()
    glad["red"] = (np.array([0.3, -0.5, g.TABLE - 0.05]), EYE)
    step(glad)
    assert M.violation()


def test_moving_is_not_still():
    glad = locked_glad()
    step(glad)
    p, r = glad["red"]
    glad["red"] = (p + [0.001, 0, 0], r)        # 2 cm/s
    step(glad)
    assert not M.success(None)


def test_doing_nothing_scores_zero():
    glad = {c: g.park_pose(c) for c in g.COLOURS}
    step(glad, rings={c: 0.0 for c in g.COLOURS})
    step(glad, rings={c: 0.0 for c in g.COLOURS})
    assert M.score(None) == 0.0 and not M.success(None)


# -- the demo -------------------------------------------------------------------------------------------------------------------
def test_demo_reaches_and_respects_limits():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert worst[side][0] < 0.5 and worst[side][1] < 0.1
        q = np.array([s[side][0] for s in STEPS])
        assert (q >= ik.LOWER - 1e-6).all() and (q <= ik.UPPER + 1e-6).all()


def test_demo_fits_run_limit_and_parks():
    assert len(STEPS) + 3 * demo.CONTROL_HZ < 2400
    last = STEPS[-1]
    for side in ("left", "right"):
        assert np.allclose(last[side][2], demo.Arm(side).home[0], atol=1e-6)
    for c in g.COLOURS:
        assert last["mark"][f"{c}_released"]


def test_demo_end_state_scores_success():
    glad = STEPS[-1]["glad"]
    step(glad)
    step(glad)
    assert M.success(None)


def test_demo_lines_stay_in_range():
    for s in STEPS:
        for c, (p, r) in s["glad"].items():
            q = g.chain_angles(c, p, r)
            assert g.L_MIN + 0.02 < q[2] < g.L_MAX - 0.06
            assert abs(q[4]) < math.radians(70) and q[1] < math.radians(80)


def test_demo_turns_about_the_seal_axis_while_pressed():
    for c in g.COLOURS:
        side = demo.SIDE_OF[c]
        twist = [s for s in STEPS if s["mark"].get(f"{c}_twist") and not s["mark"].get(f"{c}_ease")]
        assert twist and all(s[side][4] for s in twist)          # integral frozen while pressing
        for s in twist:
            p, r = demo.glad_for(s[side][2], s[side][3], c)
            phi, tilt, off, sep = g.coupling(c, p, r)
            assert off < 1e-6 and tilt < 1e-6 and sep == pytest.approx(g.S_MIN - demo.PRESS_OVER)
        assert g.coupling(c, *demo.glad_for(twist[-1][side][2], twist[-1][side][3], c))[0] == pytest.approx(demo.TWIST_OVER)


def test_grippers_clear_of_fixed_parts_and_the_other_glad_hand():
    fixed = [g.lohi(*g.BULKHEAD, g.NOSE), g.lohi(*g.BRACKET, g.NOSE), g.lohi(*g.POST, g.TENDER)]
    for c in g.COLOURS:
        fixed += [g.place([b], g.TRAILER_GH[c], g.R_TRAILER[c])[0] for b in trailer_boxes()]
        fixed += g.stand_pads(c)
    for k, s in enumerate(STEPS[::4]):
        for side in ("left", "right"):
            q, grip, tcp, rot, frozen = s[side]
            if frozen:      # pressing: the command is past the stop; check where the stop leaves the gripper instead
                c = "blue" if side == "left" else "red"
                p, r = demo.glad_for(tcp, rot, c)
                phi = g.coupling(c, p, r)[0]
                tcp, rot = demo.tcp_for(*g.tractor_pose(c, phi, g.S_MIN), c)
            opening = 0.094 * grip + 0.044 * (1 - grip)
            boxes = M.gripper_boxes(tcp, rot, opening)
            assert min_gap(boxes, fixed) > 0.004, (k * 4, side)
            other = "red" if side == "left" else "blue"
            env = M.envelope(*s["glad"][other])
            assert min(g.boxes_gap(b, env, 5) for b in boxes) > 0.03


def test_gripper_open_clears_parked_glad_hand_on_the_way_in():
    for c in g.COLOURS:
        side = demo.SIDE_OF[c]
        boxes = [g.place([b], *g.park_pose(c))[0] for _, b in g.tractor_boxes()]
        for s in STEPS[:60]:
            q, grip, tcp, rot, _ = s[side]
            if grip > 0.99:
                assert min_gap(M.gripper_boxes(tcp, rot, 0.094), boxes) > 0.015


def test_carried_glad_hand_clear_of_stands_and_trailer_until_the_seals():
    for c in g.COLOURS:
        side = demo.SIDE_OF[c]
        trailer = [g.place([b], g.TRAILER_GH[c], g.R_TRAILER[c])[0] for b in trailer_boxes()]
        lifted = False
        for s in STEPS:
            if not s["holding"][c] or s["mark"].get(f"{c}_press"):
                continue
            p, r = s["glad"][c]
            boxes = [g.place([b], p, r)[0] for _, b in g.tractor_boxes()]
            if p[2] > g.PARK_Z + 0.02:
                lifted = True
            if lifted:
                assert min_gap(boxes, g.stand_pads(c)) > 0.01
                assert min_gap(boxes, trailer) > 0.0015
        assert lifted


def test_arms_stay_on_their_own_side():
    for s in STEPS:
        assert s["left"][2][1] > 0.05 and s["right"][2][1] < -0.05
        assert np.linalg.norm(s["left"][2][:2] - s["right"][2][:2]) > 0.2
    assert station.BASES["left"][1] > 0
