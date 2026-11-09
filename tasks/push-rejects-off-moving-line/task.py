"""On the station table, a conveyor runs straight across the table from the robot's right to its left, 26 cm in front
of the arm bases, at a steady 5 cm/s. Four grey steel carrier plates (14 cm across the line, 12 cm along it) ride on
it nose to tail, each with one carton (12 cm across the line, 10 cm along it, 5 cm tall) standing in its middle. From
the front of the line: brown, red, brown, red. The two red cartons are rejects. At the left end of the line, 34 cm
left of the midpoint between the arms, the conveyor goes into a grey sheet-steel tunnel too low to reach into.

Beside the line, on its far side, a grey steel reject table (18 cm deep) runs the length of the line on the table,
5 mm lower than the carriers' tops and 1 cm from their far edge, with a low dark fence along its far edge and ends.

The robot must push each red carton off the line onto the reject table before it goes into the tunnel, and leave the
brown cartons standing on their carriers. The cartons are too wide to grip in one hand: they are meant to be pushed
across, while they move.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "The two red cartons on the moving line are rejects. ..." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, belt_path, speed_schedule  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.station import FLOOR_Z  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # the score is where the cartons are when the trial ends


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the belt's speed ------------------------------------------------------------------------------------------------
# +y (to the robot's left) is the direction of flow. The line never stops while there is anything to do: it runs at
# 5 cm/s from the start and stops only at T_END, when every carrier is inside the tunnel.
SPEED = 0.05  # m/s
T_END = 19.4
RAMP_S = 0.3
RUN = speed_schedule(((0.0, SPEED), (T_END, 0.0)), RAMP_S)

# -- the line ---------------------------------------------------------------------------------------------------------
LINE_X = 0.26  # the belt's centre line, metres in front of the arm bases
END_Y = 0.34  # the tunnel's mouth: the end of the line the robot works on
RIGHT_END_Y = -0.61  # the conveyor's right end, 2 cm behind the last carrier
TUNNEL_END_Y = 0.92  # the tunnel's closed far end, 27 cm past the table's left edge
RAIL_H, RAIL_W, RAIL_DX = 0.015, 0.015, 0.04  # two steel rails, centred 4 cm either side of the line
CARRIER = (0.14, 0.12, 0.015)  # across the belt (x), along it (y), thick
CARRIER_Z = TOP + RAIL_H + 0.002 + CARRIER[2] / 2  # 2 mm above the rails: the carriers ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
CARTON = (0.12, 0.10, 0.05)  # across the belt (x), along it (y), tall: wider than the open gripper (9.4 cm) both ways
CARTON_MASS = 0.2
CARTON_Z = CARRIER_TOP + CARTON[2] / 2
CARTON_TOP = CARRIER_TOP + CARTON[2]
GREY_STEEL = (0.62, 0.64, 0.66)
DARK = (0.18, 0.18, 0.2)
BROWN = (0.62, 0.45, 0.27)
RED = (0.82, 0.12, 0.1)

# The cartons from the front of the line back: name, kind, where its carrier starts along the line. Nose to tail
# 13 cm apart: 1 cm between carriers, 3 cm between cartons.
PARTS = (("good_1", "good", -0.14), ("reject_1", "reject", -0.27), ("good_2", "good", -0.40),
         ("reject_2", "reject", -0.53))
KIND = {name: kind for name, kind, _ in PARTS}
START_Y = {name: y for name, _, y in PARTS}
NAMES = tuple(KIND)
REJECTS = tuple(n for n in NAMES if KIND[n] == "reject")
GOODS = tuple(n for n in NAMES if KIND[n] == "good")


def carrier_path(name: str):
    return belt_path((LINE_X, START_Y[name], CARRIER_Z), (0.0, 1.0, 0.0), RUN)


def part_y(name: str, t: float) -> float:
    """Where a carton riding on its carrier is along the line at ``t``."""
    return float(START_Y[name] + RUN(t)[0])


CARRIERS = [
    DrivenBox(f"carrier_{n}", pos=(LINE_X, START_Y[n], CARRIER_Z), size=CARRIER, color=GREY_STEEL, path=carrier_path(n))
    for n in NAMES
]
CARTONS = [
    Box(n, pos=(LINE_X, START_Y[n], CARTON_Z), size=CARTON, mass=CARTON_MASS, color=RED if k == "reject" else BROWN)
    for n, k, _ in PARTS
]
RAILS = [
    Box(f"rail_{side}", pos=_floats((LINE_X + dx, (RIGHT_END_Y + TUNNEL_END_Y) / 2, TOP + RAIL_H / 2)),
        size=_floats((RAIL_W, TUNNEL_END_Y - RIGHT_END_Y, RAIL_H)), color=DARK)
    for side, dx in (("near", -RAIL_DX), ("far", RAIL_DX))
]

# -- the reject table --------------------------------------------------------------------------------------------------
# A fixed steel table top along the far side of the line, 1 cm from the carriers and 5 mm lower than their tops, so a
# carton pushed across drops a little onto it and doesn't catch on its edge. A 3 cm fence runs along its far edge and
# both ends (the left end lip stands just short of the tunnel's mouth).
SHELF_X = (LINE_X + CARRIER[0] / 2 + 0.01, LINE_X + CARRIER[0] / 2 + 0.01 + 0.18)  # near and far edges: 34 to 52 cm
SHELF_Y = (-0.62, END_Y)  # right and left ends
SHELF_TOP = CARRIER_TOP - 0.005
LIP_T, LIP_H = 0.01, 0.03
SHELF_GREY = (0.55, 0.57, 0.6)


def _shelf() -> list[Box]:
    (x0, x1), (y0, y1) = SHELF_X, SHELF_Y
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    lip_z = SHELF_TOP + LIP_H / 2
    return [
        Box("reject_table", pos=_floats((xc, yc, (TOP + SHELF_TOP) / 2)),
            size=_floats((x1 - x0, y1 - y0, SHELF_TOP - TOP)), color=SHELF_GREY),
        Box("fence_far", pos=_floats((x1 + LIP_T / 2, yc, (TOP + SHELF_TOP + LIP_H) / 2)),
            size=_floats((LIP_T, y1 - y0, SHELF_TOP + LIP_H - TOP)), color=DARK),
        Box("fence_right", pos=_floats((xc, y0 + LIP_T / 2, lip_z)), size=_floats((x1 - x0, LIP_T, LIP_H)), color=DARK),
        Box("fence_left", pos=_floats((xc, y1 - LIP_T / 2, lip_z)), size=_floats((x1 - x0, LIP_T, LIP_H)), color=DARK),
    ]


# -- the tunnel ------------------------------------------------------------------------------------------------------
# Sheet steel over the line from END_Y on: a roof 2 cm above the cartons' tops on two side walls that stand on the
# table (and, past its edge, hang down to the rails' level), closed at the far end and held up by a post.
WALL_IN = CARRIER[0] / 2 + 0.0125  # the side walls' inner faces, 1.25 cm outside the carriers
SHEET = 0.01
ROOF_T = 0.008
ROOF_LOW = CARTON_TOP + 0.02
TUNNEL_OUT = WALL_IN + SHEET
TUNNEL_LEN = TUNNEL_END_Y - END_Y


def _tunnel() -> list[Box]:
    yc = (END_Y + TUNNEL_END_Y) / 2
    wall_h = ROOF_LOW - TOP
    return [
        Box("tunnel_roof", pos=_floats((LINE_X, yc, ROOF_LOW + ROOF_T / 2)),
            size=_floats((2 * TUNNEL_OUT, TUNNEL_LEN, ROOF_T)), color=GREY_STEEL),
        Box("tunnel_near", pos=_floats((LINE_X - WALL_IN - SHEET / 2, yc, TOP + wall_h / 2)),
            size=_floats((SHEET, TUNNEL_LEN, wall_h)), color=GREY_STEEL),
        Box("tunnel_far", pos=_floats((LINE_X + WALL_IN + SHEET / 2, yc, TOP + wall_h / 2)),
            size=_floats((SHEET, TUNNEL_LEN, wall_h)), color=GREY_STEEL),
        Box("tunnel_end", pos=_floats((LINE_X, TUNNEL_END_Y + SHEET / 2, TOP + wall_h / 2)),
            size=_floats((2 * WALL_IN, SHEET, wall_h)), color=GREY_STEEL),
        # past the table's left edge the tunnel's far end stands on a post to the floor
        Box("post_left", pos=_floats((LINE_X, TUNNEL_END_Y - 0.03, (TOP + FLOOR_Z) / 2)),
            size=_floats((2 * TUNNEL_OUT, 0.04, TOP - FLOOR_Z)), color=DARK),
    ]


OBJECTS = [*RAILS, *CARRIERS, *CARTONS, *_shelf(), *_tunnel()]

# -- scoring ---------------------------------------------------------------------------------------------------------
ON_SHELF_IN = 0.03  # a carton counts as on the reject table with its centre at least this far past the table's edge
HELD_ABOVE = 0.08  # a carton whose centre is this far above a surface it would lie on is held up, not lying there


def on_shelf(c) -> bool:
    """Lying on the reject table: centre 3 cm or more past its near edge and inside its fence, low (not held up above
    it) and still."""
    x, y, z = c.pos
    inside = SHELF_X[0] + ON_SHELF_IN <= x <= SHELF_X[1] and SHELF_Y[0] + LIP_T < y < SHELF_Y[1] - LIP_T
    return bool(inside and z < SHELF_TOP + HELD_ABOVE and c.resting())


def on_line(c, carrier) -> bool:
    """Still on its carrier in any pose: centre over the carrier's footprint and not held up above it."""
    d = c.pos - carrier.pos
    return bool(abs(d[0]) < CARRIER[0] / 2 and abs(d[1]) < CARRIER[1] / 2 and c.pos[2] < CARRIER_TOP + HELD_ABOVE)


def standing(c, carrier) -> bool:
    """On its carrier and standing upright on it: centre within 1 cm of the height it rides at."""
    return on_line(c, carrier) and abs(c.pos[2] - (carrier.pos[2] + CARRIER[2] / 2 + CARTON[2] / 2)) < 0.01


def tally(objects) -> tuple[int, int, int]:
    """(rejects on the reject table, rejects off the line elsewhere, brown cartons no longer standing on their
    carriers)."""
    shelved = sum(on_shelf(objects[n]) for n in REJECTS)
    elsewhere = sum(not on_shelf(objects[n]) and not on_line(objects[n], objects[f"carrier_{n}"]) for n in REJECTS)
    spoiled = sum(not standing(objects[n], objects[f"carrier_{n}"]) for n in GOODS)
    return shelved, elsewhere, spoiled


def score(objects):
    """0.5 for each red carton lying on the reject table; 0.125 for each red carton off the line but not on the table
    (on the table top, the floor, in a gripper); minus 0.5 for each brown carton not standing on its carrier (pushed
    off, knocked over, lifted). Never below 0."""
    shelved, elsewhere, spoiled = tally(objects)
    return max(0.0, 0.5 * shelved + 0.125 * elsewhere - 0.5 * spoiled)


def success(objects):
    """Both red cartons on the reject table and both brown cartons standing on their carriers."""
    return tally(objects) == (len(REJECTS), 0, 0)


SHELF_GOAL_X = 0.42  # where the demo leaves a pushed carton's centre: its near face 2 cm onto the table


def distance(objects):
    """Metres from the first red carton not yet on the reject table to the spot across the line from it on the
    table."""
    for n in REJECTS:
        c = objects[n]
        if not on_shelf(c):
            goal = np.array([SHELF_GOAL_X, c.pos[1], SHELF_TOP + CARTON[2] / 2])
            return float(np.linalg.norm(c.pos - goal))
    return 0.0


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Open loop, timed against the belt. The right arm pushes reject_1 early, while it passes in front of the right arm;
# the left arm pushes reject_2 later, once it has come past the midpoint. Each arm comes down with its fingers closed
# in front of the carton's near face, clear of the carrier, follows the carton along the line, closes the gap and
# pushes it straight across the line onto the reject table while still following the line, then rises and goes back.
# The fingers close across the line (yaw +90 degrees for the right arm, -90 for the left: each the way round that keeps
# its wrist in range), so the carton meets the flat outer side of a finger.
DT = 0.05  # one control step at 20 Hz
YAW = {"right": float(np.pi / 2), "left": float(-np.pi / 2)}
TIP_ABOVE = 0.006  # pad tips this far above the carrier tops while pushing
TIP_Z = CARRIER_TOP + PAD_TIP_PAST_TCP + TIP_ABOVE  # TCP height while pushing
CLEAR_Z = CARTON_TOP + PAD_TIP_PAST_TCP + 0.03  # pad tips 3 cm above the cartons' tops
# The closed fingers seen from the side (from the gripper's meshes): within the 5 cm of a carton's height above the pad
# tips they reach 4 to 10 mm ahead of the TCP, at most 2.4 cm counting their wider part higher up; the gripper housing
# starts 8.4 cm above the pad tips. In Isaac Sim a 6 cm carton was touched 1.8 cm ahead of the TCP (stop-sliding-carton).
CONTACT = 0.010  # TCP to where the fingers touch the carton (planning value)
CONTACT_MAX = 0.018  # the most it can be
FINGER_REACH = 0.024  # the furthest any part of the fingers reaches ahead of the TCP
HOUSING_ABOVE_TIP = 0.084
START_X = LINE_X - CARTON[0] / 2 - FINGER_REACH - 0.015  # coming down: every part of the fingers 1.5 cm from the carton
TOUCH_X = LINE_X - CARTON[0] / 2 - CONTACT  # the fingers against the carton's near face
PUSH_TO_X = SHELF_GOAL_X - CARTON[0] / 2 - CONTACT  # leaves the carton's centre at SHELF_GOAL_X
DESCEND_S, CLOSE_IN_S, PUSH_S, HOLD_S, RISE_S, BACK_S = 0.6, 0.4, 2.0, 0.2, 0.6, 1.2
ARM = {"reject_1": "right", "reject_2": "left"}
MEET = {"reject_1": 1.4, "reject_2": 11.8}  # when each arm starts down in front of its carton
REST = {"right": (0.20, -0.30, CLEAR_Z + 0.08), "left": (0.20, 0.30, CLEAR_Z + 0.08)}


def _track(name: str, t0: float, seconds: float, xz0, xz1, label: str, yaw: float) -> list[Waypoint]:
    """Waypoints one control step apart that follow carton ``name`` along the line from ``t0``, while the TCP's x and
    z go smoothly from ``xz0`` to ``xz1``. The fingers stay closed."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = k / n
        s = s * s * (3 - 2 * s)
        x, z = (xz0[i] + s * (xz1[i] - xz0[i]) for i in range(2))
        out.append(Waypoint(DT, (x, part_y(name, t0 + k * DT), z), FINGER_CLOSED, label, yaw))
    return out


def phases(name: str) -> list[tuple[str, float, float]]:
    """(label, start, end) of each tracked phase of the push on ``name`` (episode seconds)."""
    out, t = [], MEET[name]
    for label, dur in (("down", DESCEND_S), ("close in", CLOSE_IN_S), ("push", PUSH_S), ("hold", HOLD_S),
                       ("rise", RISE_S)):
        out.append((f"{label} {name}", t, t + dur))
        t += dur
    return out


def _push(name: str) -> list[Waypoint]:
    """From the arm's start: over the spot in front of carton ``name`` where it will be at its meeting time, then
    follow it down, close in, push it across onto the reject table, hold, rise, and go back to rest."""
    side, meet = ARM[name], MEET[name]
    y = part_y(name, meet)
    go = min(2.0, meet - 0.4)
    yaw = YAW[side]
    plan = [Waypoint(go, (START_X, y, CLEAR_Z), FINGER_CLOSED, f"over {name}", yaw)]
    if meet - go > 1e-9:
        plan.append(Waypoint(meet - go, (START_X, y, CLEAR_Z), FINGER_CLOSED, f"wait for {name}", yaw))
    path = {"down": ((START_X, CLEAR_Z), (START_X, TIP_Z)), "close in": ((START_X, TIP_Z), (TOUCH_X, TIP_Z)),
            "push": ((TOUCH_X, TIP_Z), (PUSH_TO_X, TIP_Z)), "hold": ((PUSH_TO_X, TIP_Z), (PUSH_TO_X, TIP_Z)),
            "rise": ((PUSH_TO_X, TIP_Z), (PUSH_TO_X, CLEAR_Z))}
    for label, t0, t1 in phases(name):
        plan += _track(name, t0, t1 - t0, *path[label.split(f" {name}")[0]], label, yaw)
    plan.append(Waypoint(BACK_S, REST[side], FINGER_CLOSED, f"back from {name}", yaw))
    return plan


def demo():
    """Right arm: reject_1 from 1.4 s, while it passes in front of the right arm, pushed onto the table from 2.4 to
    4.4 s. Left arm: reject_2 from 11.8 s, once it is past the midpoint, pushed from 12.8 to 14.8 s. Each carton ends
    with its centre 16 cm across from where it rode (its near face 2 cm onto the reject table)."""
    return {ARM[n]: _push(n) for n in REJECTS}
