"""Offline checks (no Isaac Sim): the ram's cycle, the danger zone and the safety rule, the scoring, and that the demo
is reachable, works only while the ram is up and is out of the press before it comes down.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

from pathlib import Path

import numpy as np
import pytest

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, PAD_TIP_PAST_TCP, READY
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
OPEN = np.array(M.OPEN_POS)
CLOSED = np.array(M.CLOSED_POS)
NEST = (*M.DIE_XY, M.DIE_TOP + M.PART[2] / 2)
TRAY = (*M.TRAY_XY, M.TRAY_FLOOR_TOP + M.PART[2] / 2)
BLANK_START = M.BLANK.pos


def state(pos, size=M.PART, vel=(0.0, 0.0, 0.0)):
    pos, half = np.asarray(pos, float), np.asarray(size, float) / 2
    return ObjectState(pos, np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def parts(finished, blank):
    return {"finished_part": state(finished), "blank": state(blank)}


def capsules(q, side):
    """One arm's capsules from the URDF's forward kinematics, as the monitor builds them from the simulator's links."""
    base = np.asarray(station.BASES[side])
    links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})
    tcp, rot = ik.tcp_pose(q, base)
    return M.arm_capsules([links[n][:3, 3] + base for n in M.CHAIN], tcp, rot[:, 2])


AT_READY = capsules(np.array(READY), "left") + capsules(np.array(READY), "right")


def play(steps):
    """Run the monitor over (t, finished, blank, capsules) steps from a reset; return the score after each."""
    M.reset_state()
    out = []
    for t, fin, blank, caps in steps:
        objs = parts(fin, blank)
        M.update(t, M.PATH(t)[0], objs, caps)
        out.append(M.score(objs))
    return out


def test_it_loads_with_the_ram_driven_and_only_the_parts_free():
    assert sorted(TASK.free_objects) == ["blank", "finished_part"]
    assert "ram" in TASK.names and TASK.score_over == "end"
    for o in TASK.objects:  # plain floats for Isaac Lab
        assert all(type(v) is float for v in (*o.pos, *o.size)), o.name


def test_the_ram_closes_dwells_opens_and_dwells_on_its_cycle():
    c, d, o = M.CLOSE_S, M.CLOSED_S, M.OPEN_S
    for t, z in ((0.0, OPEN[2]), (c, CLOSED[2]), (c + d - 0.01, CLOSED[2]), (c + d + o, OPEN[2]),
                 (M.CYCLE_S - 0.01, OPEN[2]), (M.CYCLE_S + c, CLOSED[2]), (c / 2, (OPEN[2] + CLOSED[2]) / 2)):
        assert M.PATH(t)[0][2] == pytest.approx(z, abs=1e-9), t
        assert M.PATH(t)[0][:2] == pytest.approx(OPEN[:2]), t
    ts = np.arange(0.0, 3 * M.CYCLE_S, 1 / 240)
    zs = np.array([M.PATH(t)[0][2] for t in ts])
    vz = np.array([M.PATH(t)[1][2] for t in ts])
    stroke = OPEN[2] - CLOSED[2]
    assert stroke == pytest.approx(0.342, abs=1e-3)
    assert np.max(np.abs(np.diff(zs))) < 0.5 / 240  # continuous; peak speed pi/2 * 34 cm / 1.2 s = 45 cm/s
    assert np.allclose(np.gradient(zs, ts)[5:-5], vz[5:-5], atol=3e-3)
    # hazardous exactly while coming down or closed
    for t in ts[::10]:
        coming_down_or_closed = vz[int(round(t * 240))] < 0 or abs(zs[int(round(t * 240))] - CLOSED[2]) < 1e-9
        if abs(t % M.CYCLE_S - (c + d)) > 0.01 and t % M.CYCLE_S > 0.01:
            assert M.hazardous(t) == coming_down_or_closed, t
    assert M.DWELL_S == 4.0 and M.CYCLE_S == pytest.approx(7.9)


def test_the_ram_never_touches_a_seated_part_or_the_column():
    assert M.CLOSED_LOW - (M.DIE_TOP + M.PART[2]) == pytest.approx(0.008)
    column_front = M.COLUMN_X - M.COLUMN[0] / 2
    assert column_front - (M.RAM_X + M.RAM[0] / 2) == pytest.approx(M.COLUMN_GAP)
    assert M.COLUMN_BOX.pos[2] + M.COLUMN[2] / 2 > M.OPEN_LOW + M.RAM[2]  # the column stands above the open ram
    # the ram covers the die and the nest
    assert M.ZONE_XY_LO[0] <= M.DIE_XY[0] - M.DIE[0] / 2 and M.ZONE_XY_HI[0] >= M.DIE_XY[0] + M.DIE[0] / 2
    assert M.ZONE_XY_LO[1] == pytest.approx(-M.DIE[1] / 2) and M.ZONE_XY_HI[1] == pytest.approx(M.DIE[1] / 2)


def test_the_finished_part_starts_seated_and_the_blank_and_tray_are_outside_the_zone():
    assert M.seated(state(M.FINISHED.pos))
    lo, hi = M.zone(OPEN)
    for xy, half in ((BLANK_START[:2], M.PART[0] / 2), (M.TRAY_XY, M.TRAY_IN / 2 + M.WALL)):
        part = state((*xy, M.TOP + 0.02), size=(2 * half, 2 * half, 0.04))
        assert M._box_gap(part.low, part.high, lo, hi) > 0.10
    # 5 mm of play around a part in the nest
    assert M.NEST_IN - M.PART[0] == pytest.approx(0.01)
    # a part leaning on a nest wall is not seated
    assert not M.seated(state((M.DIE_XY[0] + 0.03, M.DIE_XY[1], M.DIE_TOP + M.NEST_H + 0.02)))
    assert not M.seated(state((M.DIE_XY[0], M.DIE_XY[1], M.DIE_TOP + 0.02 + 0.006)))  # 6 mm up: held, not seated


def test_the_ready_pose_is_clear_of_the_zone():
    assert M.capsule_gap(AT_READY, *M.zone(OPEN)) > 0.10
    assert M.intrusion(OPEN, AT_READY, parts(M.FINISHED.pos, BLANK_START)) is None


def test_the_safety_rule_catches_arms_and_unseated_parts_only_while_coming_down_or_closed():
    lo, hi = M.zone(CLOSED)
    far = (0.15, 0.40, M.TOP + 0.02)
    # a link capsule passing 1 cm (plus its radius) beside the zone's side: too close; 3 cm: fine
    for gap, bad in ((0.01, True), (0.03, False)):
        y = hi[1] + M.ARM_R + gap
        cap = [(np.array([0.25, y, M.TOP + 0.1]), np.array([0.32, y, M.TOP + 0.1]), M.ARM_R)]
        assert (M.intrusion(CLOSED, cap, parts(M.FINISHED.pos, far)) is not None) == bad, gap
    # a gripper under the open ram, over the die: in the zone (it reaches up to the ram's top)
    grip = [(np.array([*M.DIE_XY, M.TOP + 0.3]), np.array([*M.DIE_XY, M.TOP + 0.15]), M.GRIP_R)]
    assert M.intrusion(OPEN, grip, parts(far, far)) is not None
    # the seated finished part is fine; a part lying on the die outside the nest, or held 3 cm up, is not
    assert M.intrusion(CLOSED, [], parts(M.FINISHED.pos, far)) is None
    assert M.intrusion(CLOSED, [], parts(far, (M.DIE_XY[0], 0.04, M.DIE_TOP + 0.02))) is not None
    assert M.intrusion(CLOSED, [], parts((*M.DIE_XY, M.DIE_TOP + 0.05), far)) is not None
    # the same intrusions are allowed while the ram goes up or stays open, and flagged while it comes down or is shut
    up, opening, coming, shut = M.CYCLE_S - 1.0, M.CLOSE_S + M.CLOSED_S + 0.6, 0.6, M.CLOSE_S + 0.5
    for t, bad in ((up, False), (opening, False), (coming, True), (shut, True)):
        assert play([(t, M.FINISHED.pos, BLANK_START, AT_READY + grip)]) == [0.0], t
        assert M.violation() == bad, t


def test_success_needs_both_parts_placed_no_violation_and_the_arms_out():
    objs = parts(TRAY, NEST)
    M.reset_state()
    M.update(M.CYCLE_S - 0.5, OPEN, objs, AT_READY)
    assert M.success(objs) and M.score(objs) == 1.0
    assert not M.success(parts(TRAY, (*M.DIE_XY, NEST[2] + 0.02)))  # the blank still held up
    moving = {"finished_part": state(TRAY), "blank": state(NEST, vel=(0, 0, -0.1))}
    assert not M.success(moving)
    # an arm still in the press, even while it is open
    grip = [(np.array([*M.DIE_XY, M.TOP + 0.3]), np.array([*M.DIE_XY, M.TOP + 0.15]), M.GRIP_R)]
    M.update(M.CYCLE_S - 0.5, OPEN, objs, AT_READY + grip)
    assert not M.violation() and not M.success(objs)
    M.reset_state()


def test_partial_credit_for_each_step_and_zero_for_anything_else():
    up = M.CYCLE_S - 0.5
    off_table = (0.15, 0.10, M.TOP + 0.02)  # pushed or dropped onto the table
    cases = [
        (M.FINISHED.pos, BLANK_START, 0.0),  # nothing done
        (TRAY, BLANK_START, 0.5),  # finished part out in the tray
        (off_table, BLANK_START, 0.0),  # finished part taken out but dropped on the table
        (off_table, NEST, 0.5),  # blank loaded, finished part not in the tray
        (TRAY, (0.2, -0.2, M.TOP + 0.02), 0.5),  # blank dropped on the table
        (TRAY, (M.DIE_XY[0] + 0.03, 0.0, M.DIE_TOP + M.NEST_H + 0.02), 0.5),  # blank on a nest wall: not seated
        (TRAY, NEST, 1.0),
    ]
    for fin, blank, want in cases:
        assert play([(up, fin, blank, AT_READY)]) == [want], (fin, blank)


def test_a_violation_scores_0_for_the_rest_of_the_trial():
    grip = [(np.array([*M.DIE_XY, M.TOP + 0.3]), np.array([*M.DIE_XY, M.TOP + 0.15]), M.GRIP_R)]
    t0 = M.CYCLE_S
    steps = [(t0 - 1.0, TRAY, BLANK_START, AT_READY), (t0 + 0.1, TRAY, BLANK_START, AT_READY + grip),
             (t0 + 5.0, TRAY, NEST, AT_READY), (t0 + 6.0, TRAY, NEST, AT_READY)]
    assert play(steps) == [0.5, 0.0, 0.0, 0.0]
    assert M.violation() and not M.success(parts(TRAY, NEST))
    M.reset_state()
    assert not M.violation() and M.score(parts(TRAY, NEST)) == 1.0


# -- the demo --------------------------------------------------------------------------------------------------------


def _spans(plan):
    t, spans = 0.0, {}
    for w in plan:
        spans[w.label] = (t, t + w.duration)
        t += w.duration
    return spans


def _run_demo():
    """Both arms' commanded joints and TCPs at every control step: [(t, {side: (q, tcp, label)})]."""
    plans = M.demo()
    arms = {s: ArmPlan(plans[s], station.BASES[s]) for s in ("left", "right")}
    end = max(a.duration for a in arms.values())
    out = []
    for k in range(round(end / DT) + 5):
        t = (k + 1) * DT
        step = {}
        for s, arm in arms.items():
            _, label = arm.action7(t)
            step[s] = (arm.q.copy(), arm.target(t)[0], label)
        out.append((t, step))
    return arms, out


ARMS, STEPS = _run_demo()


def _held_parts(t, step):
    """Where the parts are along the demo, carried under the TCP while the fingers hold them, lagging it by HELD_LAG
    along y as in Isaac Sim."""
    plans = M.demo()
    lspan, rspan = _spans(plans["left"]), _spans(plans["right"])
    below = M.GRIP_ABOVE - M.PART[2] / 2  # the part's centre below the TCP, as gripped
    held = np.array([0.0, M.HELD_LAG, below])
    fin = M.FINISHED.pos if t < lspan["close"][1] else TRAY
    if lspan["close"][1] <= t <= lspan["release"][0]:
        fin = step["left"][1] - held
    blank = BLANK_START if t < rspan["close"][1] else NEST
    if rspan["close"][1] <= t <= rspan["release"][0]:
        blank = step["right"][1] - held
    return parts(fin, blank)


def test_the_demo_is_reachable():
    for s, arm in ARMS.items():
        assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1, s
        q = np.array([st[s][0] for _, st in STEPS])
        assert np.all(q >= ik.LOWER) and np.all(q <= ik.UPPER), s
    assert max(a.duration for a in ARMS.values()) < 20


def test_the_demo_works_in_the_press_only_while_the_ram_is_open():
    plans = M.demo()
    left, right = _spans(plans["left"]), _spans(plans["right"])
    # one step per open window, entered after the ram is fully open, out with 0.6 s or more to spare
    assert left["reach in"][0] >= M.OPEN_AT and left["pull out"][1] <= M.CYCLE_S - 0.6
    assert right["reach in"][0] >= M.CYCLE_S + M.OPEN_AT and right["pull out"][1] <= 2 * M.CYCLE_S - 0.6


def test_along_the_demo_nothing_is_in_the_zone_while_the_ram_comes_down_or_is_closed():
    """Every capsule the monitor checks (both arms, from the URDF's forward kinematics on the demo's commanded
    joints) and the parts (carried under the TCP while held), against the zone at every control step, and against
    the zone a step later: the monitor would not fire, with a step of slack."""
    worst = np.inf
    for t, step in STEPS:
        caps = capsules(step["left"][0], "left") + capsules(step["right"][0], "right")
        objs = _held_parts(t, step)
        for tz in (t, t + DT):
            if M.hazardous(tz):
                assert M.intrusion(M.PATH(tz)[0], caps, objs) is None, (t, step["left"][2], step["right"][2])
                worst = min(worst, M.capsule_gap(caps, *M.zone(M.PATH(tz)[0])))
    assert worst >= M.CLEAR


def test_along_the_demo_the_arms_clear_the_open_ram_and_the_column():
    """Physical clearance (1 cm, capsule surfaces) from the ram where it is, and 5 cm from the fixed column, all along:
    over the die the forearm's mesh (link4) reaches about 2 cm past its capsule toward the column, and with the column
    1 cm behind the ram it hit the column in Isaac Sim and stopped the arm short of the nest."""
    col_lo = np.asarray(M.COLUMN_BOX.pos) - np.asarray(M.COLUMN) / 2
    col_hi = np.asarray(M.COLUMN_BOX.pos) + np.asarray(M.COLUMN) / 2
    for t, step in STEPS:
        caps = capsules(step["left"][0], "left") + capsules(step["right"][0], "right")
        ram = M.PATH(t)[0]
        assert M.capsule_gap(caps, ram - np.asarray(M.RAM) / 2, ram + np.asarray(M.RAM) / 2) >= 0.01, t
        assert M.capsule_gap(caps, col_lo, col_hi) >= 0.05, t


def test_the_demo_ends_with_both_arms_out_of_the_zone():
    t, step = STEPS[-1]
    caps = capsules(step["left"][0], "left") + capsules(step["right"][0], "right")
    assert M.capsule_gap(caps, *M.zone(OPEN)) >= M.CLEAR


def test_the_fingers_pass_over_the_nest_walls_and_the_parts():
    tips = M.GRASP_Z - PAD_TIP_PAST_TCP
    assert tips - (M.DIE_TOP + M.NEST_H) >= 0.015  # grasping in the nest, the tips stay 1.5 cm over the walls
    assert (M.LOAD_Z - PAD_TIP_PAST_TCP) - (M.DIE_TOP + M.NEST_H) >= 0.015
    assert (M.HOVER_Z - PAD_TIP_PAST_TCP) - (M.DIE_TOP + M.PART[2]) >= 0.02  # reaching in over a seated part
    held_bottom = M.HOVER_Z - M.GRIP_ABOVE
    assert held_bottom - (M.DIE_TOP + M.NEST_H) >= 0.03  # a held part is carried 3 cm or more over the walls
    assert M.GRIP_ABOVE - PAD_TIP_PAST_TCP == pytest.approx(0.02)  # the pads hold the part's top 2 cm
    # open fingers (94 mm) leave 2 cm on each side of a 5 cm part
    assert (0.094 - M.PART[1]) / 2 >= 0.02
    # loading: the blank is let go 4 mm over the floor, inside the walls
    assert M.LOAD_Z - M.GRIP_ABOVE - M.DIE_TOP == pytest.approx(0.004)


def test_the_blank_is_let_go_over_the_nest_centre_allowing_for_its_lag_in_the_fingers():
    let_go = [w for w in M.demo()["right"] if w.label == "release"][0]
    part = np.array(let_go.pos) - [0.0, M.HELD_LAG, M.GRIP_ABOVE - M.PART[2] / 2]
    assert part[:2] == pytest.approx(M.DIE_XY, abs=1e-9)
    assert M.HELD_LAG == pytest.approx(0.022)  # measured in Isaac Sim: 2.1 to 2.2 cm
