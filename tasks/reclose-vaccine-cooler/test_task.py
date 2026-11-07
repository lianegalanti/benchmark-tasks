"""Offline checks (no Isaac Sim): the lid asset, the fit of lid, cooler and cartons, every scoring level, and that the
demo's moves are reachable within the joint limits and the run limit.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py      # from this folder
"""

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yam_arena import ik, station
from yam_arena.constants import PAD_TIP_PAST_TCP
from yam_arena.demo import ArmPlan
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402
import make_lid  # noqa: E402

DT = 1 / 20
RUN_LIMIT_STEPS = 2400


def body(pos, tilt_deg=0.0, yaw_deg=0.0, vel=0.0):
    """A state: centre/origin ``pos``, turned ``yaw_deg`` about z, then tipped ``tilt_deg`` about its x axis."""
    a, t = math.radians(yaw_deg) / 2, math.radians(tilt_deg) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qx = np.array([math.sin(t), 0.0, 0.0, math.cos(t)])
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    quat = np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])
    return SimpleNamespace(pos=np.asarray(pos, float), quat_xyzw=quat, lin_vel=np.array([vel, 0.0, 0.0]))


LID_ON = body((*g.COOLER_XY, g.RIM))
LID_ASIDE = body((*g.LID_ASIDE_XY, g.TABLE + g.SKIRT_H))
CARTON_Z = g.FLOOR_TOP + g.CARTON[2] / 2
ORANGE_IN = body((*g.CARTON_XY["orange_carton"], CARTON_Z))
WHITE_IN = body((*g.CARTON_XY["white_carton"], CARTON_Z))
ON_TRAY = body((*g.TRAY_XY, g.TRAY_TOP + g.CARTON[2] / 2))
ON_TABLE = body((0.15, -0.15, g.TABLE + g.CARTON[2] / 2))
AWAY = [(np.array([0.2, 0.3, 1.0]), 0.094), (np.array([0.2, -0.3, 1.0]), 0.094)]


def play(*steps, grips=AWAY):
    """Run (lid, orange, white) states through the monitor; the score and success after the last."""
    M.reset_state()
    for lid, orange, white in steps:
        M.update({"lid": lid, "orange_carton": orange, "white_carton": white}, grips)
    return M.score({}), M.success({})


# -- asset and layout ------------------------------------------------------------------------------------------------
def test_it_loads_with_the_lid_and_cartons_loose():
    assert set(TASK.free_objects) == {"lid", "orange_carton", "white_carton"}
    assert TASK.score_over == "end" and TASK.success_hold_s == 2.0
    assert M.INSTRUCTION == "Take the orange vaccine carton out of the cooler and put it on the green tray."
    assert "lid" not in M.INSTRUCTION.lower()
    for spec in TASK.objects:  # Isaac Lab wants plain floats
        assert all(type(v) is float for v in spec.pos), spec.name
        assert all(type(v) is float for v in getattr(spec, "rot_xyzw", (0.0,))), spec.name


def test_the_lid_asset_is_up_to_date_and_parses():
    assert (HERE / "assets" / "lid.usda").read_text() == make_lid.usda()
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(HERE / "assets" / "lid.usda"))
    box = UsdGeom.BBoxCache(0, ["default"]).ComputeWorldBound(stage.GetDefaultPrim()).ComputeAlignedRange()
    assert np.allclose(box.GetMin(), (-g.LID_HALF[0], -g.LID_HALF[1], -g.SKIRT_H), atol=1e-4)
    assert np.allclose(box.GetMax(), (g.LID_HALF[0], g.LID_HALF[1], g.PLATE_T + g.HANDLE[2]), atol=1e-4)
    assert stage.GetDefaultPrim().GetAttribute("physics:mass").Get() == pytest.approx(g.LID_MASS)
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]) == 6


def test_the_lid_fits_loosely_and_clears_the_cartons():
    for i in range(2):
        assert g.SKIRT_IN[i] - g.COOLER_HALF[i] == pytest.approx(0.008)  # 8 mm gap all round
    assert g.SKIRT_GAP < g.SEAT_XY                                       # a lid flat on the rim is always centred
    assert g.RIM - g.SKIRT_H > g.TABLE + 0.05                            # the skirt hangs well clear of the table
    assert g.RIM - (g.FLOOR_TOP + g.CARTON[2]) >= 0.008 - 1e-9            # the seated plate clears the cartons
    # a lid sitting on its skirt edge on the rim, off to one side, is tilted or raised: not seated
    assert not g.lid_seated((g.COOLER_XY[0] + 0.03, 0.0, g.RIM + g.SKIRT_H), LID_ON.quat_xyzw)
    assert not g.lid_seated(LID_ON.pos, body(LID_ON.pos, tilt_deg=8).quat_xyzw)
    assert g.lid_seated(LID_ON.pos + [0.006, -0.006, 0.002], body(LID_ON.pos, yaw_deg=3).quat_xyzw)


def test_the_grippers_have_room():
    # handle: open fingers 3.45 cm either side of it; fingertips 1 cm above the plate when gripping
    assert (g.GRIPPER_OPEN - g.HANDLE[1]) / 2 >= 0.03
    assert g.HANDLE_TCP_DZ - PAD_TIP_PAST_TCP >= 0.008
    # carton: pads 1.75 cm clear of it, 3.3 cm clear of the walls (room for the fingers' thickness)
    assert (g.CARTON_OPEN - g.CARTON[0]) / 2 >= 0.015
    assert g.INNER_HALF[0] - g.CARTON_OPEN / 2 >= 0.03
    # the fingers (about 2 cm wide, along y) stay off the white carton and the side wall
    ox, oy = g.CARTON_XY["orange_carton"]
    wx, wy = g.CARTON_XY["white_carton"]
    assert (wy - g.CARTON[1] / 2) - (oy + 0.01) >= 0.015
    assert (oy - 0.01) - (g.COOLER_XY[1] - g.INNER_HALF[1]) >= 0.015
    # the fingertips reach only 3.3 cm below the rim, so the gripper's body stays above the walls
    tips = g.carton_grasp((ox, oy))[2] - PAD_TIP_PAST_TCP
    assert g.RIM - tips <= 0.035
    assert tips - g.FLOOR_TOP >= 0.04


def test_the_layout_leaves_room_to_set_the_lid_aside():
    assert g.LID_ASIDE_XY[1] - g.LID_HALF[1] - (g.COOLER_XY[1] + g.COOLER_HALF[1]) >= 0.05
    assert g.COOLER_XY[1] - g.COOLER_HALF[1] - (g.TRAY_XY[1] + g.TRAY[1] / 2) >= 0.1
    # carried lid and carton pass over the rim with 2 cm to spare
    assert g.LID_CARRY_Z - g.HANDLE_TCP_DZ - g.PLATE_T - g.SKIRT_H - g.RIM >= 0.02 - 1e-9
    assert g.CARTON_CARRY_Z + g.CARTON_TCP_DOWN - g.CARTON[2] - g.RIM >= 0.02 - 1e-9


def test_the_start_scores_0():
    assert play((LID_ON, ORANGE_IN, WHITE_IN)) == (0.0, False)
    assert not M._STATE["violation"] and M._STATE["seated"]


# -- scoring ---------------------------------------------------------------------------------------------------------
def test_full_score_needs_the_carton_on_the_tray_and_the_lid_back_on():
    assert play((LID_ASIDE, ON_TABLE, WHITE_IN), (LID_ON, ON_TRAY, WHITE_IN)) == (1.0, True)
    lying = body((*g.TRAY_XY, g.TRAY_TOP + g.CARTON[0] / 2), tilt_deg=90)
    assert play((LID_ON, lying, WHITE_IN)) == (1.0, True)                         # lying on the tray counts
    nudged = body(LID_ON.pos + [0.007, 0.004, 0.001], yaw_deg=2)
    assert play((nudged, ON_TRAY, WHITE_IN)) == (1.0, True)


def test_lid_left_off_scores_half():
    assert play((LID_ASIDE, ON_TRAY, WHITE_IN)) == (0.5, False)
    tilted = body((g.COOLER_XY[0] + 0.02, 0.0, g.RIM + 0.01), tilt_deg=6)   # skirt riding on the rim
    assert play((tilted, ON_TRAY, WHITE_IN)) == (0.5, False)
    handle = g.handle_centre(LID_ON.pos, LID_ON.quat_xyzw)
    held = [(handle, 0.025), AWAY[1]]
    assert play((LID_ON, ON_TRAY, WHITE_IN), grips=held) == (0.5, False)          # seated but still held


def test_lid_on_with_the_carton_out_but_not_on_the_tray_scores_a_quarter():
    assert play((LID_ON, ON_TABLE, WHITE_IN)) == (0.25, False)
    off_edge = body((g.TRAY_XY[0] + 0.07, g.TRAY_XY[1], g.TABLE + g.CARTON[2] / 2))
    assert play((LID_ON, off_edge, WHITE_IN)) == (0.25, False)
    held = [AWAY[0], (ON_TRAY.pos, 0.035)]
    assert play((LID_ON, ON_TRAY, WHITE_IN), grips=held) == (0.25, False)         # still in the gripper
    assert play((LID_ON, body(ON_TRAY.pos, vel=0.05), WHITE_IN)) == (0.25, False)  # still moving


def test_other_outcomes_score_0():
    assert play((LID_ASIDE, ON_TABLE, WHITE_IN)) == (0.0, False)
    assert play((LID_ASIDE, ORANGE_IN, WHITE_IN)) == (0.0, False)                  # opened, nothing taken
    knocked = body((g.COOLER_XY[0], -0.03, g.FLOOR_TOP + g.CARTON[0] / 2), tilt_deg=90)
    assert play((LID_ON, knocked, WHITE_IN)) == (0.0, False)                       # toppled inside, lid back on


def test_taking_the_white_carton_out_is_a_violation_that_latches():
    white_out = body((0.15, 0.15, g.TABLE + g.CARTON[2] / 2))
    assert play((LID_ASIDE, ON_TRAY, white_out), (LID_ON, ON_TRAY, WHITE_IN)) == (0.0, False)
    assert M._STATE["why"] == ["the white carton was taken out of the cooler"]
    lifted = body((*g.CARTON_XY["white_carton"], g.RIM + 0.01))
    assert play((LID_ASIDE, ORANGE_IN, lifted), (LID_ON, ON_TRAY, WHITE_IN)) == (0.0, False)


def test_anything_off_the_table_is_a_violation():
    floor = body((0.3, 0.6, 0.1))
    assert play((floor, ON_TRAY, WHITE_IN), (LID_ON, ON_TRAY, WHITE_IN)) == (0.0, False)
    assert play((LID_ON, floor, WHITE_IN)) == (0.0, False)


# -- demo ------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("side", ["left", "right"])
def test_the_demo_is_reachable_within_joint_limits(side):
    plan = demo.nominal_waypoints()[side]
    arm = ArmPlan(plan, station.BASES[side])
    actions = np.array([arm.action7((k + 1) * DT)[0] for k in range(round(arm.duration / DT) + 5)])
    assert arm.worst_ik_mm < 0.5 and arm.worst_ik_deg < 0.1
    assert np.all(actions[:, :6] >= ik.LOWER) and np.all(actions[:, :6] <= ik.UPPER)


def test_the_demo_fits_the_run_limit():
    plans = demo.nominal_waypoints()
    # the right arm starts once the lid is set aside; the left arm takes the lid again once the carton is out
    lid_aside = sum(w.duration for w in plans["left"][:7])
    carton_out = lid_aside + sum(w.duration for w in plans["right"][:4])
    end = max(carton_out + sum(w.duration for w in plans["left"][8:]) + 0.3 + 0.5 + 0.5 + 2.5,
              lid_aside + sum(w.duration for w in plans["right"]) + 2.5)
    assert end + 2.0 < 0.5 * RUN_LIMIT_STEPS * DT  # done in under half the run limit, success held 2 s


def test_the_arms_keep_apart():
    plans = demo.nominal_waypoints()
    aside_hover = np.array(plans["left"][7].pos)
    for w in plans["right"]:
        assert np.linalg.norm(np.array(w.pos[:2]) - aside_hover[:2]) > 0.25
    # while the right arm carries the carton to the tray, the left arm's lid moves only after the carton is out
    assert plans["right"][3].label == "lift out"
