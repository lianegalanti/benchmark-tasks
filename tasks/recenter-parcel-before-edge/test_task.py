"""Offline checks (no Isaac Sim): the conveyor and arch layout, the parcel's untouched ride, the scoring, and that the
demo is reachable, comes down clear of the parcel, pushes without tipping it, stays clear of the arch and leaves the
parcel on the lane's centre.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest
from motionkit import ConveyorBed, LaunchedBox, Marker, ramp_distance

from yam_arena import YAM_URDF, ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import ObjectState, load
from yam_arena.urdf import forward_kinematics

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = 1 / 20
G = 9.81
REST_Z = M.BED_TOP + M.PARCEL[2] / 2


def state(pos, vel=(0.0, 0.0, 0.0), tilt_deg=0.0):
    pos, half = np.asarray(pos, float), np.asarray(M.PARCEL, float) / 2
    a = math.radians(tilt_deg) / 2
    quat = np.array([math.sin(a), 0.0, 0.0, math.cos(a)])  # tilted about x
    return ObjectState(pos, quat, np.asarray(vel, float), np.zeros(3), pos - half, pos + half)


def run(states):
    """Feed parcel states one control step apart from a fresh episode; return the score after each."""
    M.MONITOR.reset()
    out = []
    for k, s in enumerate(states):
        M.MONITOR.update(s, k * DT)
        out.append(M.MONITOR.score(s))
    return out


def riding(x, y):
    return state((x, y, REST_Z), vel=(-0.0155, -0.058, 0.0))


def boxes(name_prefix):
    return [o for o in TASK.objects if o.name.startswith(name_prefix)]


def lo_hi(o):
    half = np.asarray(o.size) / 2
    return np.asarray(o.pos) - half, np.asarray(o.pos) + half


# -- layout --------------------------------------------------------------------------------------------------------


def test_it_loads_with_only_the_parcel_free_and_scores_at_the_end():
    assert TASK.free_objects == ("parcel",)
    assert isinstance(M.BED, ConveyorBed) and isinstance(M.PARCEL_SPEC, LaunchedBox)
    assert M.PARCEL_SPEC.velocity == (0.0, 0.0, 0.0)  # at rest on the rollers when the conveyor starts
    assert all(isinstance(o, Marker) for o in M.ROLLERS + M.LINES)  # drawn, never touched
    assert TASK.score_over == "end"


def test_positions_and_sizes_are_plain_floats():
    for o in TASK.objects:
        assert all(type(v) is float for v in o.pos), o.name
        assert all(type(v) is float for v in o.size), o.name
        assert type(o.yaw) is float, o.name


def test_everything_stands_on_the_table():
    half_x, half_y = station.TABLE_SIZE[0] / 2, station.TABLE_SIZE[1] / 2
    cx, cy = station.TABLE_CENTER[:2]
    for o in TASK.objects:
        lo, hi = lo_hi(o)
        assert lo[0] > cx - half_x and hi[0] < cx + half_x and lo[1] > cy - half_y and hi[1] < cy + half_y, o.name
        assert lo[2] >= M.TOP - 1e-9 or isinstance(o, Marker), o.name


def test_the_bed_rides_clear_of_the_table_and_every_fixed_part():
    lo, hi = lo_hi(M.BED)
    assert lo[2] - M.TOP == pytest.approx(0.002) and hi[2] == pytest.approx(M.BED_TOP)
    for o in [M.RAIL, *boxes("arch_")]:
        olo, ohi = lo_hi(o)
        overlap_xy = olo[0] < hi[0] and ohi[0] > lo[0] and olo[1] < hi[1] and ohi[1] > lo[1]
        if overlap_xy:  # over the bed: it hangs above it
            assert olo[2] - M.BED_TOP >= 0.003 - 1e-9, o.name
        else:  # beside it: at least 3 mm away
            gap = max(olo[0] - hi[0], lo[0] - ohi[0], olo[1] - hi[1], lo[1] - ohi[1])
            assert gap >= 0.003 - 1e-9, o.name


def test_the_lane_lines_and_the_arch_opening_are_centred_on_the_lane():
    near, far = M.LINES
    assert near.pos[0] == pytest.approx(M.LANE_X - 0.07) and far.pos[0] == pytest.approx(M.LANE_X + 0.07)
    # a parcel between the lines (inner edges 13.4 cm apart) is within 2 cm of centre, give or take the paint
    assert (far.pos[0] - near.pos[0]) - M.PARCEL[0] == pytest.approx(2 * M.ON_LANE)
    g_near, g_far = M._arch()[3], M._arch()[4]
    opening = lo_hi(g_far)[0][0] - lo_hi(g_near)[1][0]
    assert opening == pytest.approx(0.20) and (lo_hi(g_far)[0][0] + lo_hi(g_near)[1][0]) / 2 == pytest.approx(M.LANE_X)
    assert opening - M.PARCEL[0] == pytest.approx(2 * M.NEAR_LANE)  # it only fits parcels within 5 cm of centre
    assert lo_hi(g_near)[1][1] == pytest.approx(M.GATE_Y)  # the gate line is the arch's upstream face
    assert lo_hi(M._arch()[2])[0][2] - M.BED_TOP > M.PARCEL[2] + 0.02  # a parcel passes under the roof


def test_the_rollers_are_drawn_square_to_the_drive_direction_and_inside_the_bed():
    axis = np.array([math.cos(M.SKEW), -math.sin(M.SKEW), 0.0])
    assert abs(axis @ M.DRIVE) < 1e-12
    for r in M.ROLLERS:
        assert r.yaw == pytest.approx(-M.SKEW)
        x_extent = r.size[0] * math.cos(M.SKEW) + r.size[1] * math.sin(M.SKEW)
        y_extent = r.size[0] * math.sin(M.SKEW) + r.size[1] * math.cos(M.SKEW)
        assert M.BED_X[0] < r.pos[0] - x_extent / 2 and r.pos[0] + x_extent / 2 < M.BED_X[1]
        assert M.BED_Y[0] < r.pos[1] - y_extent / 2 and r.pos[1] + y_extent / 2 < M.BED_Y[1]


def test_the_bed_surface_drives_diagonally_toward_the_open_side():
    v = M.BED.surface_velocity(5.0)
    assert np.linalg.norm(v) == pytest.approx(M.SPEED)
    assert v[1] == pytest.approx(-0.058, abs=0.001) and v[0] == pytest.approx(-0.0155, abs=0.001)
    assert M.BED.speed(0.0) == 0.0 and M.BED.speed(M.RAMP_S / 2) == pytest.approx(M.SPEED / 2)
    ts = np.arange(0.0, 1.0, 1 / 240)
    accel = np.abs(np.diff([M.BED.speed(t) for t in ts])) * 240
    mu_d = (M.FRICTION_BED[1] + M.PARCEL_SPEC.friction[1]) / 2  # PhysX averages the two
    assert accel.max() < 0.1 * mu_d * G  # the parcel keeps up with the start-up without slipping
    assert ramp_distance(M.SPEED, M.RAMP_S, 3.0) == pytest.approx(M.SPEED * (3.0 - M.RAMP_S / 2))


# -- the untouched ride ----------------------------------------------------------------------------------------------


def test_untouched_it_falls_off_the_open_side_well_before_the_arch():
    assert M.START[0] - M.LANE_X == pytest.approx(-0.04)  # starts 4 cm toward the open side
    assert M.START[1] + M.PARCEL[1] / 2 < M.BED_Y[1] and M.START[0] + M.PARCEL[0] / 2 < M.BED_X[1]
    ts = np.arange(0.0, 12.0, DT)
    path = [M.ride(t) for t in ts]
    off = next(k for k, p in enumerate(path) if p[0] < M.BED_X[0])
    assert ts[off] == pytest.approx(6.0, abs=0.15)
    assert path[off][1] - M.PARCEL[1] / 2 - M.GATE_Y > 0.05  # its leading face is still 5 cm short of the arch
    scores = run([state((p[0], p[1], REST_Z if p[0] > M.BED_X[0] else M.TOP + M.PARCEL[2] / 2)) for p in path])
    assert scores[0] == 0.25 and scores[-1] == 0.0
    # with 20 % less skew (rollers slipping) it still goes off before the arch
    slower = M.START[0] + (M.START[1] - M.GATE_Y) * math.tan(0.8 * M.SKEW) * -1
    assert slower < M.BED_X[0]


# -- scoring -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "offset, want",
    [(0.0, 1.0), (0.019, 1.0), (-0.0199, 1.0), (0.021, 0.5), (-0.035, 0.5), (0.05, 0.5), (-0.051, 0.25)],
)
def test_the_crossing_offset_sets_the_score(offset, want):
    x = M.LANE_X + offset
    scores = run([riding(x, 0.02), riding(x, 0.01), riding(x, -0.001), riding(x - 0.03, -0.01)])
    assert scores[:2] == [0.25, 0.25]
    assert scores[2] == scores[3] == want  # decided at the crossing; later drift doesn't change it
    M.MONITOR.reset()
    for k, s in enumerate([riding(x, 0.01), riding(x, -0.001)]):
        M.MONITOR.update(s, k * DT)
    assert M.success({"parcel": riding(x, -0.001)}) == (want == 1.0)


def test_falling_off_scores_zero_for_good_but_not_after_the_crossing():
    fell = state((0.17, 0.10, M.TOP + M.PARCEL[2] / 2))
    assert run([riding(0.22, 0.12), fell, riding(M.LANE_X, 0.05), riding(M.LANE_X, -0.01)])[-1] == 0.0
    after = run([riding(M.LANE_X, 0.003), riding(M.LANE_X, -0.001), state((0.17, -0.10, M.TOP + 0.03))])
    assert after[-1] == 1.0


def test_partial_credit_covers_every_ending():
    cases = {
        "held still on the rollers to the end": ([riding(M.LANE_X, 0.2)] * 3, 0.25),
        "pinned against the far rail": ([riding(M.BED_X[1] - 0.05, 0.3)] * 3, 0.25),
        "jammed against the near guide's face": ([riding(0.27, 0.051)] * 3, 0.25),
        "held up off the rollers": ([riding(M.LANE_X, 0.2), state((M.LANE_X, 0.2, REST_Z + 0.03))], 0.0),
        "tipped over on the rollers": ([riding(M.LANE_X, 0.2), state((M.LANE_X, 0.2, REST_Z), tilt_deg=90)], 0.0),
        "pushed off the near edge": ([riding(0.25, 0.2), state((0.15, 0.2, M.TOP + 0.03))], 0.0),
        "lifted over the arch and set down beyond it": (
            [riding(M.LANE_X, 0.05), state((M.LANE_X, -0.05, REST_Z + 0.12)), riding(M.LANE_X, -0.17)], 0.25),
        "tipped as it crosses": ([riding(M.LANE_X, 0.01), state((M.LANE_X, -0.001, REST_Z), tilt_deg=30)], 0.0),
    }
    for name, (states, want) in cases.items():
        assert run(states)[-1] == want, name


def test_monitor_ignores_a_repeated_step_and_resets():
    M.MONITOR.reset()
    M.MONITOR.update(riding(M.LANE_X, -0.001), 0.0)
    assert M.MONITOR.offset == pytest.approx(0.0)
    M.MONITOR.reset()
    assert M.MONITOR.offset is None and not M.MONITOR.fallen and not M.MONITOR.past


# -- the demo ------------------------------------------------------------------------------------------------------


def _arm():
    plan = M.demo()["left"]
    arm = ArmPlan(plan, station.BASES["left"])
    ts = [(k + 1) * DT for k in range(round(arm.duration / DT) + 5)]
    qs = [arm.action7(t)[0][:6] for t in ts]
    return plan, arm, ts, qs


PLAN, ARM, TS, QS = _arm()


def test_the_demo_is_reachable_within_joint_limits_and_the_run_limit():
    assert ARM.worst_ik_mm < 0.5 and ARM.worst_ik_deg < 0.1
    q = np.array(QS)
    assert np.all(q >= ik.LOWER) and np.all(q <= ik.UPPER)
    assert M.time_to(M.GATE_Y) + 1.5 < 600 * DT


def test_it_is_wider_than_the_open_gripper():
    assert min(M.PARCEL[:2]) > -2 * FINGER_OPEN


def _target(label_prefix):
    t = 0.0
    out = []
    for w in PLAN:
        t += w.duration
        if w.label.startswith(label_prefix):
            out.append((t, np.asarray(w.pos)))
    return out


def test_the_fingers_come_down_beside_its_near_face_not_onto_it():
    for t, p in _target("over its near side") + _target("down beside it"):
        face = M.ride(t)[0] - M.PARCEL[0] / 2
        assert face - (p[0] + M.FINGER_HALF) == pytest.approx(M.GAP, abs=1e-9)
        assert abs(p[1] - M.ride(t)[1]) < 1e-9
    t_down = _target("down beside it")[-1][0]
    assert t_down < 3.0  # down before the parcel's centre is anywhere near the edge (it is at about 6 s)
    assert M.ride(t_down)[0] - M.BED_X[0] > 0.05
    assert M.TIP_Z - PAD_TIP_PAST_TCP - M.BED_TOP == pytest.approx(0.015)  # pad tips 1.5 cm above the rollers
    assert M.CLEAR_Z - PAD_TIP_PAST_TCP > M.BED_TOP + M.PARCEL[2] + 0.02


def test_the_push_cannot_tip_it():
    """Pushed at most at its top edge, a box tips only if height x friction > half its width."""
    mu_s = (M.FRICTION_BED[0] + M.PARCEL_SPEC.friction[0]) / 2
    assert M.PARCEL[2] * mu_s < 0.8 * M.PARCEL[0] / 2


def test_the_escort_leaves_it_on_the_lane_centre_at_the_crossing():
    t_rel, p = _target("ride beside it")[-1]
    held_x = p[0] + M.FINGER_HALF + M.PARCEL[0] / 2
    assert held_x - M.LANE_X == pytest.approx(M.DRIFT_X)
    assert M.ride(t_rel)[1] == pytest.approx(M.RELEASE_Y, abs=0.004)
    # free from there: it rides and drifts to the gate line
    crossing_x = held_x + (M.ride(t_rel)[1] - M.GATE_Y) * (M.DRIVE[0] / -M.DRIVE[1])
    assert abs(crossing_x - M.LANE_X) < 0.004
    # margins: off by 1 cm in the finger contact and 30 % in the drift, it still scores 1.0
    assert abs(crossing_x - M.LANE_X) + 0.01 + 0.3 * M.DRIFT_X < M.ON_LANE
    # the fingers back off faster than it drifts after them
    back = next(w for w in PLAN if w.label == "back off")
    assert 0.012 / back.duration > 1.5 * M.SPEED * math.sin(M.SKEW)


def test_while_riding_beside_it_the_fingers_keep_pace_with_it():
    escort = _target("ride beside it")
    for t, p in escort:
        assert p[1] == pytest.approx(M.ride(t)[1])
    speeds = [(a[1][1] - b[1][1]) / DT for a, b in zip(escort[1:], escort[:-1])]
    assert np.allclose(speeds, -M.SPEED * math.cos(M.SKEW))


CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")
ARM_R = 0.045  # a generous radius around every link segment and the gripper


def _segment_box_distance(a, b, lo, hi, n=25):
    pts = a + np.linspace(0, 1, n)[:, None] * (b - a)
    d = np.maximum(np.maximum(lo - pts, pts - hi), 0.0)
    return float(np.min(np.linalg.norm(d, axis=1)))


def test_along_the_demo_the_arm_stays_clear_of_the_arch_and_the_rail():
    base = np.asarray(station.BASES["left"])
    fixed = [M.RAIL, *boxes("arch_")]
    worst = math.inf
    for q in QS:
        links = forward_kinematics(YAM_URDF, {n: float(v) for n, v in zip(ARM_JOINTS, q, strict=True)})
        pts = [links[n][:3, 3] + base for n in CHAIN]
        tcp, _ = ik.tcp_pose(np.asarray(q), base)
        pts.append(tcp)
        for a, b in zip(pts[1:-1], pts[2:]):  # from link1 on (the base itself stands still, far from them)
            for o in fixed:
                lo, hi = lo_hi(o)
                worst = min(worst, _segment_box_distance(a, b, lo, hi) - ARM_R)
    assert worst > 0.03, worst


def test_the_fingers_stay_above_the_rollers_and_never_reach_the_arch():
    for w in PLAN:
        assert w.pos[2] - PAD_TIP_PAST_TCP >= M.BED_TOP + 0.015 - 1e-9
        assert w.pos[1] - 0.02 > M.GATE_Y + 0.02  # the fingers (about 1.6 cm either side) 2 cm short of the arch
