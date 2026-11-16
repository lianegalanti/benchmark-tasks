"""Offline checks (no Isaac Sim): the layout and sizes, the nostrils' and the swab's geometry, every scoring level and
violation, the flinch, and the demo (reach, joint limits, clearances, and the whole plan replayed through the scorer
with a stiff-arm model of the soft nose).

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402

PLAN = demo.plan()
STEPS = PLAN[:-1]
FAR = [M.gripper_boxes(np.array([0.0, s * 0.30, g.TABLE + 0.40]), np.diag([1.0, -1.0, -1.0]), 0.094)
       for s in (1, -1)]
RUN_LIMIT = 3000


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()


def tip_pose(side, depth, f=0.0, y=0.0):
    """Swab pose with its tip's end ``depth`` up nostril ``side``, offset (f, y) in the lumen (straight manikin)."""
    p, r = demo.tcp_at_nostril(side, depth, (f, y))
    return g.swab_from_tcp(p, r)


def soft_nose(swab_pos, swab_rot):
    """A stiff arm on the soft nose: if the tip's centre is pushed past its free path inside a nostril, the nose
    gives way by the excess. Returns the nose's (f, y) displacement."""
    end = g.tip_end(swab_pos, swab_rot)
    p0 = M.pose_from_joints()
    local = p0.n_rot.T @ (end - p0.n_pos)
    for side, yc in sorted(g.NOSTRIL_Y.items(), key=lambda kv: abs(local[1] - kv[1])):   # the nearer nostril
        o = np.array([local[2], local[1] - yc])
        if abs(o[0]) < 0.02 and abs(o[1]) < 0.012 and -0.003 < local[0] < 0.035:
            a, b = g.SEMI_F - g.TIP[0], g.SEMI_Y - g.TIP[0]
            s = math.hypot(o[0] / a, o[1] / b)
            if s > 1.0:
                d = o - o / s
                return float(d[0]), float(d[1])
    return 0.0, 0.0


def holding(swab):
    """The right gripper closed on the swab's cap (the left far away)."""
    return [FAR[0], M.gripper_boxes(*g.tcp_for_swab(*swab), g.GRIP[1])]


def step(swab=(g.SWAB_START, g.R_SWAB0), vel=(0.0, 0.0, 0.0), grips=None, pitch=0.0, yaw=0.0, nose=None):
    """One monitor step; by default the swab is held unless it is at its start (seated in its tube)."""
    f, y = soft_nose(*swab) if nose is None else nose
    p = M.pose_from_joints(pitch, yaw, f, y)
    if grips is None:
        grips = FAR if np.allclose(swab[0], g.SWAB_START) else holding(swab)
    return M.update(p, swab[0], np.asarray(vel, float), swab[1], grips)


def circle(side, n_circles=4.25, press=0.003, per=48, depth=0.018):
    a_f = g.SEMI_F - g.TIP[0] + press
    a_y = g.SEMI_Y - g.TIP[0] + press
    for k in range(int(n_circles * per) + 1):
        phi = 2 * math.pi * k / per
        step(tip_pose(side, depth, a_f * math.cos(phi), a_y * math.sin(phi)))


LIFTED = (g.SWAB_START + np.array([0.0, 0.0, 0.16]), g.R_SWAB0)


# -- layout, assets and sizes ---------------------------------------------------------------------------------------
def test_it_loads_with_a_fixed_manikin_a_loose_swab_and_a_fixed_tube():
    assert TASK.names == ("manikin", "swab", "tube")
    assert TASK.score_over == "end" and M.SUCCESS_HOLD_S == 1.0
    assert M.SWAB.mass == g.SWAB_MASS and M.TUBE.mass is None
    assert dict(M.MANIKIN.joint_pos) == {"neck_pitch": 0.0, "neck_yaw": 0.0, "nose_f": 0.0, "nose_y": 0.0}
    assert not M.MANIKIN.self_collisions


def test_the_assets_have_the_joints_and_links_the_scorer_reads():
    from pxr import Usd, UsdPhysics

    st = Usd.Stage.Open(str(HERE / "assets/manikin.usda"))
    joints = {p.GetName(): p for p in st.Traverse() if p.IsA(UsdPhysics.Joint)}
    assert set(joints) == {"neck_pitch", "neck_yaw", "nose_f", "nose_y", "root_joint"}
    links = {p.GetName() for p in st.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert links == {"base", "neck_link", "head", "nosef", "nose"}
    nose = UsdPhysics.PrismaticJoint(joints["nose_f"])
    assert nose.GetUpperLimitAttr().Get() == pytest.approx(g.NOSE_LIMIT)
    for f in ("swab.usda", "tube.usda"):
        assert Usd.Stage.Open(str(HERE / "assets" / f)).GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)


def test_full_size_parts():
    assert 2 * g.HEAD_AX == pytest.approx([0.192, 0.152, 0.230])          # adult head: length, breadth, height
    assert 2 * g.SEMI_F == pytest.approx(0.015) and 2 * g.SEMI_Y == pytest.approx(0.010)   # adult nostril
    assert g.SWAB_LEN == pytest.approx(0.163, abs=1e-3)                   # 6 in swab
    assert 2 * g.TIP[0] == pytest.approx(0.0055) and g.TIP[2] - g.TIP[1] == pytest.approx(0.015)
    assert g.TUBE_LEN == pytest.approx(0.150) and 2 * g.TUBE_R_OUT == pytest.approx(0.0155)
    nostril = g.nostril_station("left")
    assert nostril[2] - g.TABLE == pytest.approx(0.213, abs=0.002)
    assert g.BUST[1] == pytest.approx(0.40)


def test_the_swab_hangs_in_its_tube_at_the_start():
    p = M.pose_from_joints()
    assert M.seated(p, *g.swab_from_tcp(*g.tcp_for_swab(g.SWAB_START, g.R_SWAB0)))
    end = g.tip_end(g.SWAB_START, g.R_SWAB0)
    assert M.tip_in_bore(p, end)
    assert end[2] - (g.TUBE_POS[2] - g.TUBE_LEN + g.TUBE_BOTTOM) > 0.02        # off the bottom
    assert g.PLUG[0] + 0.001 <= g.TUBE_R_IN and g.CONE[1] + 0.0035 <= g.TUBE_R_IN   # plug play and lead-in


def test_the_grip_section_takes_the_whole_pad():
    # the pads are 40 mm wide along the swab and close on the 16 mm flats, clear of the 18 mm flange
    assert g.GRIP[0] / 2 >= 0.020 + 0.001
    assert g.GRIP[1] < 2 * g.FLANGE[0]


def test_nothing_reaches_into_a_nostril():
    us = np.linspace(-0.0005, g.LUMEN_DEPTH, 14)
    for side, yc in g.NOSTRIL_Y.items():
        pts = np.array([[u, yc + 0.999 * g.SEMI_Y * math.sin(t), 0.999 * g.SEMI_F * math.cos(t)]
                        for u in us for t in np.linspace(0, 2 * math.pi, 40)])
        for _, c, s, r in g.lumen_staves("left") + g.lumen_staves("right"):
            assert g.point_box_dist(pts, c, r, s / 2).min() > 0.0
        for _, c, s in g.nose_outer_boxes():
            assert g.point_box_dist(pts, c, np.eye(3), s / 2).min() > 0.0


def test_the_linings_close_round_each_nostril():
    for side, yc in g.NOSTRIL_Y.items():
        ring = np.array([[0.01, yc + (g.SEMI_Y + 0.001) * math.sin(t), (g.SEMI_F + 0.001) * math.cos(t)]
                         for t in np.linspace(0, 2 * math.pi, 120)])
        d = np.min([g.point_box_dist(ring, c, r, s / 2) for _, c, s, r in g.lumen_staves(side)], axis=0)
        assert d.max() == 0.0


def test_the_vestibule_runs_clear_of_the_face_to_past_the_depth_limit():
    p = M.pose_from_joints()
    for side in g.NOSTRIL_Y:
        pts = np.array([g.nostril_station(side, d) for d in np.linspace(0.0, g.TOO_DEEP, 12)])
        assert g.ellipsoid_dist(pts, p.h_pos + p.h_rot @ g.HEAD_C, p.h_rot, g.HEAD_AX).min() > 0.0


def test_free_play_round_the_tip_in_a_nostril():
    assert g.SEMI_F - g.TIP[0] == pytest.approx(0.00475) and g.SEMI_Y - g.TIP[0] == pytest.approx(0.00225)
    assert g.DISP_BAND == pytest.approx((0.0012, 0.008))
    assert g.DISP_BAND[1] < g.NOSE_LIMIT


# -- scoring --------------------------------------------------------------------------------------------------------
def test_doing_nothing_scores_0():
    for _ in range(40):
        step()
    assert M.score(None) == 0.0 and not M.success(None) and not M.violation()


def test_lifting_the_swab_out_and_putting_it_back_unused_scores_0():
    step(LIFTED)
    for _ in range(30):
        step()
    assert M._STATE["used"] and M._STATE["seated"] and M.score(None) == 0.0


def test_four_circles_in_one_nostril_score_035():
    step(LIFTED)
    circle("left")
    assert M._STATE["circles"] == {"left": 4, "right": 0}
    assert M.score(None) == pytest.approx(0.35) and not M.violation()


def test_partial_credit_per_circle():
    step(LIFTED)
    circle("left", n_circles=2.5)
    assert M._STATE["circles"]["left"] == 2
    assert M.score(None) == pytest.approx(0.35 * 2 / 4)


def test_back_and_forth_does_not_count():
    step(LIFTED)
    a_f, a_y = g.SEMI_F - g.TIP[0] + 0.003, g.SEMI_Y - g.TIP[0] + 0.003
    for _ in range(6):
        for phi in list(np.linspace(0, 3, 30)) + list(np.linspace(3, 0, 30)):
            step(tip_pose("left", 0.018, a_f * math.cos(phi), a_y * math.sin(phi)))
    assert M._STATE["circles"]["left"] == 0


def test_circles_in_the_air_or_too_light_do_not_count():
    step(LIFTED)
    circle("left", press=-0.001)              # inside the free path: no contact
    circle("left", press=0.0008)              # 0.8 mm: under 0.3 N
    assert M._STATE["circles"]["left"] == 0 and M.score(None) == 0.0


def test_circles_too_hard_do_not_count():
    step(LIFTED)
    circle("left", press=0.0085)              # over 2 N
    assert M._STATE["circles"]["left"] == 0


def test_circles_too_shallow_do_not_count():
    step(LIFTED)
    circle("left", depth=0.008)
    assert M._STATE["circles"]["left"] == 0


def test_too_deep_scores_0():
    step(LIFTED)
    circle("left")
    step(tip_pose("left", g.TOO_DEEP + 0.001))
    assert M.violation() and M.score(None) == 0.0


def test_full_job_with_hands_off_is_success():
    step(LIFTED)
    circle("left")
    circle("right")
    for _ in range(10):
        step()
    assert M._STATE["seated"] and M._STATE["released"]
    assert M.success(None) and M._STATE["level"] == 1.0


def test_capped_but_still_holding_scores_without_the_release():
    step(LIFTED)
    circle("left")
    circle("right")
    tcp, rot = g.tcp_for_swab(g.SWAB_START, g.R_SWAB0)
    holding = [FAR[0], M.gripper_boxes(tcp, rot, 0.016)]
    for _ in range(5):
        step(grips=holding)
    assert not M.success(None) and M.score(None) == pytest.approx(0.7 + 0.2)


def test_both_nostrils_not_capped_scores_07():
    step(LIFTED)
    circle("left")
    circle("right")
    step(LIFTED)
    assert M.score(None) == pytest.approx(0.7)


@pytest.mark.parametrize("where", ["table", "lip", "cheek", "nose_outside", "rack"])
def test_contaminated_tip_scores_0(where):
    step(LIFTED)
    circle("left")
    p0 = M.pose_from_joints()
    if where == "table":
        pos = np.array([0.20, -0.05, g.TABLE + g.TIP_END - 0.0005])
        rot = np.column_stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    else:
        r = demo.nose_rot()
        rs = r @ g.GRIP_IN_S
        if where == "lip":
            target = p0.h_pos + g.LIPS[0][1] + np.array([-g.LIPS[0][2] - g.TIP[0] + 0.0005, 0.0, 0.0])
        elif where == "cheek":
            target = p0.h_pos + g.HEAD_C + np.array([-0.07, -0.045, -0.03])
        elif where == "nose_outside":
            target = g.nostril_station("left", 0.005) + np.array([0.0, -0.010 - g.TIP[0] + 0.0005, 0.0])
        else:
            target = g.TUBE_POS + np.array([0.0, g.RACK[1] / 2 + g.TIP[0], -g.TUBE_LEN + g.RACK[2] - 0.01])
        pos = target - rs @ np.array([g.TIP_END - g.TIP[0], 0.0, 0.0])
        rot = rs
    step((pos, rot), nose=(0.0, 0.0))
    assert M.violation() and M.score(None) == 0.0, M._STATE["gaps"]


def test_the_swab_dropped_on_the_table_scores_0():
    step(LIFTED)
    a = math.asin((g.GRIP[2] / 2 - g.TIP[0]) / g.TIP_END)       # lying on its cap, the tip on the table
    x = np.array([0.0, math.cos(a), -math.sin(a)])
    y = np.array([-1.0, 0.0, 0.0])
    rot = np.column_stack([x, y, np.cross(x, y)])
    pos = np.array([0.15, -0.05, g.TABLE + g.GRIP[2] / 2])
    step((pos, rot))
    assert M.violation()


def test_pushing_the_head_past_15_degrees_scores_0():
    step(LIFTED)
    circle("left")
    step(LIFTED, pitch=math.radians(16))
    assert M.violation() and M.score(None) == 0.0


def test_pressing_the_nose_from_outside_while_the_tip_hangs_in_the_nostril_does_not_count():
    step(LIFTED)
    a = 0.003
    for k in range(4 * 48 + 12):
        phi = 2 * math.pi * k / 48
        step(tip_pose("left", 0.018), nose=(a * math.cos(phi), a * math.sin(phi)))
    assert M._STATE["circles"]["left"] == 0


def test_rubbing_deep_in_the_nostril_is_not_touching_the_face():
    # the lining encloses the tip: the face behind it does not count as a touch
    step(LIFTED)
    circle("left", depth=0.0215, press=0.005)
    circle("right", depth=0.0215, press=0.005)
    assert not M.violation() and M._STATE["circles"] == {"left": 4, "right": 4}


def test_circles_with_only_part_of_the_tip_in_do_not_count():
    step(LIFTED)
    circle("left", depth=0.013)
    assert M._STATE["circles"]["left"] == 0


def test_letting_go_of_the_swab_outside_its_tube_scores_0():
    step(LIFTED)
    circle("left")
    circle("right")
    step(LIFTED, grips=FAR)
    assert M.violation() and M.score(None) == 0.0


def test_the_tip_rubbing_the_nostrils_rim_scores_0():
    step(LIFTED)
    circle("left")
    # the tip's end just inside the opening, pressed out past the oval onto the rim's outer skin (the ala)
    p0 = M.pose_from_joints()
    r = demo.nose_rot()
    rs = r @ g.GRIP_IN_S
    target = g.nostril_station("left", 0.001) + p0.n_rot @ np.array([0.0, -(g.SEMI_Y + g.WALL + 0.0015), 0.0])
    pos = target - rs @ np.array([g.TIP_END - g.TIP[0], 0.0, 0.0])
    step((pos, rs), nose=(0.0, 0.0))
    assert M.violation(), M._STATE["gaps"]


def test_moving_swab_or_head_is_not_success():
    step(LIFTED)
    circle("left")
    circle("right")
    step(vel=(0.05, 0.0, 0.0))
    assert not M.success(None)
    M._STATE["pitch"] = math.radians(1.0)
    step()
    assert not M.success(None)


def test_measured_in_the_patients_frame():
    # a head turned 5 degrees: the same circles relative to the nose still count
    step(LIFTED)
    a_f, a_y = g.SEMI_F - g.TIP[0] + 0.003, g.SEMI_Y - g.TIP[0] + 0.003
    yaw = math.radians(5)
    for k in range(4 * 48 + 12):
        phi = 2 * math.pi * k / 48
        hp, hr = g.head_pose(0.0, yaw)
        npos, nrot = g.nose_pose(hp, hr)
        end = npos + nrot @ np.array([0.018, g.NOSTRIL_Y["left"] + a_y * math.sin(phi), a_f * math.cos(phi)])
        rs = hr @ demo.nose_rot() @ g.GRIP_IN_S
        pos = end - rs @ np.array([g.TIP_END, 0.0, 0.0])
        o = np.array([a_f * math.cos(phi), a_y * math.sin(phi)])
        s = math.hypot(o[0] / (g.SEMI_F - g.TIP[0]), o[1] / (g.SEMI_Y - g.TIP[0]))
        d = o - o / s
        p = M.pose_from_joints(0.0, yaw, d[0], d[1])
        M.update(p, pos, np.zeros(3), rs, holding((pos, rs)))
    assert M._STATE["circles"]["left"] == 4 and not M.violation()


# -- the demo -------------------------------------------------------------------------------------------------------
def test_demo_reach_and_limits():
    w = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert w[side][0] < 0.5 and w[side][1] < 0.1, w
    q = np.array([s[side][0] for s in STEPS for side in ("left", "right")])
    assert (q >= ik.LOWER - 1e-6).all() and (q <= ik.UPPER + 1e-6).all()
    for side in ("left", "right"):
        qs = np.array([s[side][0] for s in STEPS])
        assert np.abs(np.diff(qs, axis=0)).max() < 0.06      # no jumps between control steps (rad)


def test_demo_fits_the_run_limit():
    assert len(STEPS) + 3 * 20 < RUN_LIMIT


def test_demo_does_not_read_simulator_objects():
    src = (HERE / "demo.py").read_text()
    for word in ("object_states", "root_link_pose", "body_link_pose", "_STATE[\"circles\"] if", "links["):
        assert word not in src.split("def run(")[0]


def test_demo_replayed_through_the_scorer_succeeds():
    """The whole plan, with the swab where the script holds it and a stiff arm pressing the soft nose."""
    tcp0, rot0 = g.tcp_for_swab(g.SWAB_START, g.R_SWAB0)
    for s in STEPS:
        grips = [M.gripper_boxes(s[k][2], s[k][3], 0.094 * s[k][1]) for k in ("left", "right")]
        if s["holding"]:
            grips[1] = M.gripper_boxes(s["right"][2], s["right"][3], g.GRIP[1])
        M.update(M.pose_from_joints(0.0, 0.0, *soft_nose(*s["swab"])), s["swab"][0], np.zeros(3), s["swab"][1],
                 grips)
        assert not M.violation(), (s["phase"], M._STATE["why"], M._STATE["gaps"])
    assert M._STATE["circles"] == {"left": 4, "right": 4}
    assert M._STATE["level"] == 1.0
    assert M._STATE["max_depth"] < g.DEPTH_BAND[1]


def test_grippers_clear_the_manikin_table_and_tube():
    p = M.pose_from_joints()
    for s in STEPS:
        for side in ("left", "right"):
            q, grip, tcp, rot = s[side]
            boxes = M.gripper_boxes(tcp, rot, 0.094 * grip if not (side == "right" and s["holding"]) else g.GRIP[1])
            pts = np.vstack([g.box_points(*b, 4) for b in boxes])
            assert pts[:, 2].min() > g.TABLE + 0.01
            body = min(g.point_box_dist(pts, g.M0 + c, np.eye(3), sz / 2).min() for _, c, sz in g.bust_boxes())
            assert body > 0.01, (side, s["phase"])
            if side == "right":
                face = g.ellipsoid_dist(pts, p.h_pos + g.HEAD_C, np.eye(3), g.HEAD_AX).min()
                assert face > 0.015, s["phase"]


def test_the_hands_stay_apart():
    for s in STEPS:
        assert np.linalg.norm(s["left"][2] - s["right"][2]) > 0.15
