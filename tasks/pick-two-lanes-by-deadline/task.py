"""Two straight conveyor lanes run side by side across the station table in front of the left arm, carrying parts from
the robot's right to its left. The near lane runs at 10 cm/s, the far lane at 2 cm/s, five times slower. A grey
sheet-steel cover hides both lanes except one open window, 34 cm long, in front of the left arm: parts come out from
under the feed cover on the right and ride under the run-out cover on the left, past which they are out of reach.
Each part is a bare aluminium cube (4 x 4 x 4 cm) on its own grey steel carrier; each lane brings two, four in all.
A grey open bin (14 x 14 cm, 6 cm walls) stands on the table in front of the left arm, nearer the arms.

The robot must put every part in the bin before it rides under the run-out cover. The order that works is not the
order the parts come out: each time, a far-lane (slow) part comes out first, and a near-lane (fast) part comes out
about 2 s later, behind it, but reaches the run-out cover 12 s sooner. Taking the first part to come out loses the
fast one; taking the fast one first leaves plenty of time for the slow one.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py \\
           --instruction "Four parts will come out on the two lanes ..." ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, cruise_time, episode_time, on_reset, on_step, release_cruise  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import TABLE_TOP, Box  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]
SCORE_OVER = "end"  # the score is what lies in the bin when the trial ends

# -- the lanes -----------------------------------------------------------------------------------------------------
# +y (the robot's left) is the flow direction. Both lanes share one cover with one open window, from FEED_Y to END_Y.
LANE_X = {"near": 0.29, "far": 0.41}  # centre lines, metres in front of the arm bases
SPEED = {"near": 0.10, "far": 0.02}  # m/s: the near lane is five times as fast
FEED_Y, END_Y = 0.13, 0.47  # the window: parts come out from under the feed cover at FEED_Y; END_Y is the end line
FEED_BACK_Y, RUNOUT_BACK_Y = -0.09, 0.635  # the closed back ends of the two covers (their inner faces)
RAMP_S = 0.5  # a carrier speeds up to its lane's speed, and slows to a stop, over 0.5 s
RAIL_H, RAIL_W, RAIL_DX = 0.015, 0.015, 0.02  # two steel rails per lane, centred 2 cm either side of its centre line
CARRIER = (0.065, 0.065, 0.015)
CARRIER_Z = TOP + RAIL_H + 0.002 + CARRIER[2] / 2  # 2 mm above the rails: the carriers ride, they don't rub
CARRIER_TOP = CARRIER_Z + CARRIER[2] / 2
CUBE = 0.04
CUBE_Z = CARRIER_TOP + CUBE / 2
CUBE_TOP = CARRIER_TOP + CUBE
GREY_STEEL = (0.62, 0.64, 0.66)
DARK = (0.18, 0.18, 0.2)
ALUMINIUM = (0.80, 0.81, 0.83)

# Each lane has two carriers, staged nose to tail under the feed cover and released onto the running lane at fixed
# times; each runs on under the run-out cover and stops there. Part 1 of a lane is the lead carrier's.
START_Y = {1: 0.08, 2: -0.01}  # the lead carrier's front edge is 1.75 cm inside the feed cover
STOP_Y = {1: 0.595, 2: 0.505}  # both stop wholly under the run-out cover
RELEASE_S = {("far", 1): 0.0, ("near", 1): 3.3, ("far", 2): 6.25, ("near", 2): 13.15}
PARTS = tuple(RELEASE_S)  # (lane, k) in the order they come out of the feed cover


def name(lane: str, k: int) -> str:
    return f"{lane}_{k}"


def carrier_path(lane: str, k: int):
    x = LANE_X[lane]
    return release_cruise((x, START_Y[k], CARRIER_Z), (x, STOP_Y[k], CARRIER_Z), SPEED[lane], RELEASE_S[lane, k],
                          RAMP_S)


def passes(lane: str, k: int, y: float) -> float:
    """When part ``k`` of ``lane`` (riding its carrier) has its centre at ``y``."""
    x = LANE_X[lane]
    return cruise_time((x, START_Y[k], 0.0), (x, y, 0.0), SPEED[lane], RELEASE_S[lane, k], RAMP_S)


CARRIERS = [
    DrivenBox(f"carrier_{name(lane, k)}", pos=(LANE_X[lane], START_Y[k], CARRIER_Z), size=CARRIER, color=GREY_STEEL,
              path=carrier_path(lane, k))
    for lane, k in PARTS
]
CUBES = [
    Box(f"part_{name(lane, k)}", pos=(LANE_X[lane], START_Y[k], CUBE_Z), size=(CUBE, CUBE, CUBE), mass=0.17,
        color=ALUMINIUM)
    for lane, k in PARTS
]
RAIL_LEN = RUNOUT_BACK_Y - FEED_BACK_Y
RAILS = [
    Box(f"rail_{lane}_{side}", pos=(LANE_X[lane] + dx, (FEED_BACK_Y + RUNOUT_BACK_Y) / 2, TOP + RAIL_H / 2),
        size=(RAIL_W, RAIL_LEN, RAIL_H), color=DARK)
    for lane in LANE_X
    for side, dx in (("near", -RAIL_DX), ("far", RAIL_DX))
]

# -- the covers ----------------------------------------------------------------------------------------------------
# Two sheet-steel tunnels over both lanes, one each side of the window: two side walls standing on the table and a
# roof 2 cm above the cubes' tops, closed at the outer end. Between them the window is open from the table up.
WALL_IN = 0.0475  # the side walls' inner faces, this far outside the outer lanes' centre lines (1.5 cm past a carrier)
SHEET = 0.01
ROOF_T = 0.008
ROOF_LOW = CUBE_TOP + 0.02
ROOF_TOP = ROOF_LOW + ROOF_T
COVER_X_IN = (LANE_X["near"] - WALL_IN, LANE_X["far"] + WALL_IN)
COVER_X_OUT = (COVER_X_IN[0] - SHEET, COVER_X_IN[1] + SHEET)


def _cover(tag: str, mouth_y: float, back_y: float) -> list[Box]:
    """A tunnel over both lanes, open at ``mouth_y`` and closed by an end wall whose inner face is at ``back_y``."""
    out = back_y + (SHEET if back_y > mouth_y else -SHEET)  # the end wall's outer face
    y0, y1 = min(mouth_y, out), max(mouth_y, out)
    cx, cy, length = (COVER_X_OUT[0] + COVER_X_OUT[1]) / 2, (y0 + y1) / 2, y1 - y0
    span_x = COVER_X_OUT[1] - COVER_X_OUT[0]
    wall_h, wall_z = ROOF_LOW - TOP, (TOP + ROOF_LOW) / 2
    return [
        Box(f"cover_{tag}_near", pos=(COVER_X_OUT[0] + SHEET / 2, cy, wall_z), size=(SHEET, length, wall_h),
            color=GREY_STEEL),
        Box(f"cover_{tag}_far", pos=(COVER_X_OUT[1] - SHEET / 2, cy, wall_z), size=(SHEET, length, wall_h),
            color=GREY_STEEL),
        Box(f"cover_{tag}_end", pos=(cx, (back_y + out) / 2, wall_z), size=(span_x - 2 * SHEET, SHEET, wall_h),
            color=GREY_STEEL),
        Box(f"cover_{tag}_roof", pos=(cx, cy, ROOF_LOW + ROOF_T / 2), size=(span_x, length, ROOF_T),
            color=GREY_STEEL),
    ]


COVERS = [*_cover("feed", FEED_Y, FEED_BACK_Y), *_cover("runout", END_Y, RUNOUT_BACK_Y)]

# -- the bin -------------------------------------------------------------------------------------------------------
BIN_XY = (0.14, 0.30)  # in front of the left arm (base at y 0.305), nearer the arms than the lanes
BIN_IN = 0.12  # inner width and depth
WALL, FLOOR, WALL_H = 0.01, 0.01, 0.06
BIN_FLOOR_TOP = TOP + FLOOR
BIN_RIM = TOP + WALL_H
BIN_GREY = (0.42, 0.44, 0.47)


def _bin() -> list[Box]:
    x, y = BIN_XY
    out = BIN_IN / 2 + WALL / 2
    span = BIN_IN + 2 * WALL
    z = TOP + WALL_H / 2
    return [
        Box("bin_floor", pos=(x, y, TOP + FLOOR / 2), size=(span, span, FLOOR), color=BIN_GREY),
        Box("bin_wall_near", pos=(x - out, y, z), size=(WALL, span, WALL_H), color=BIN_GREY),
        Box("bin_wall_far", pos=(x + out, y, z), size=(WALL, span, WALL_H), color=BIN_GREY),
        Box("bin_wall_right", pos=(x, y - out, z), size=(BIN_IN, WALL, WALL_H), color=BIN_GREY),
        Box("bin_wall_left", pos=(x, y + out, z), size=(BIN_IN, WALL, WALL_H), color=BIN_GREY),
    ]


OBJECTS = [*RAILS, *CARRIERS, *CUBES, *COVERS, *_bin()]

# -- the rule: straight from the line into the bin -------------------------------------------------------------------
# A part counts only if it goes straight from its carrier into the bin. One that spends AWAY_S or more in a row away
# from its own carrier, from both grippers and from over the bin (put down or dropped on the table, a cover, another
# carrier or part) is lost for the rest of the trial, so parking parts, or knocking them off the line to pick up
# later, scores nothing for them.
AWAY_S = 0.5
HELD_M = 0.05  # a part's centre this near the point between a gripper's finger pads counts as held by it

_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.update(away_since={}, lost=set())


reset_state()


def on_own_carrier(part, carrier) -> bool:
    """Riding its carrier: centre over the carrier's top, underside within 1 cm of it."""
    over = abs(part.pos[0] - carrier.pos[0]) < CARRIER[0] / 2 and abs(part.pos[1] - carrier.pos[1]) < CARRIER[1] / 2
    return bool(over and abs(part.low[2] - carrier.high[2]) < 0.01)


def over_bin(part) -> bool:
    """The part's centre is over the bin's inside (between the walls), at any height."""
    dx, dy = part.pos[0] - BIN_XY[0], part.pos[1] - BIN_XY[1]
    return bool(abs(dx) < BIN_IN / 2 and abs(dy) < BIN_IN / 2)


def update(parts: dict, carriers: dict, tcps, t: float) -> None:
    """One control step of the rule. ``parts`` and ``carriers`` map (lane, k) to object states; ``tcps`` are the
    grippers' TCP positions; ``t`` is the episode time (s)."""
    for key, part in parts.items():
        if key in _STATE["lost"]:
            continue
        held = any(np.linalg.norm(part.pos - np.asarray(p)) < HELD_M for p in tcps)
        if held or over_bin(part) or on_own_carrier(part, carriers[key]):
            _STATE["away_since"].pop(key, None)
            continue
        since = _STATE["away_since"].setdefault(key, t)
        if t - since >= AWAY_S - 1e-9:
            _STATE["lost"].add(key)


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.scene import object_states

    names = [f"part_{name(*p)}" for p in PARTS] + [f"carrier_{name(*p)}" for p in PARTS]
    st = object_states(environment, names)
    tcps = [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0] for side in ("left", "right")]
    parts = {p: st[f"part_{name(*p)}"] for p in PARTS}
    carriers = {p: st[f"carrier_{name(*p)}"] for p in PARTS}
    update(parts, carriers, tcps, episode_time())


def lost(lane: str, k: int) -> bool:
    return (lane, k) in _STATE["lost"]


# -- scoring -------------------------------------------------------------------------------------------------------


def in_bin(part) -> bool:
    """In the bin: centre between the walls, underside below the rim (on the floor or on another part), and still."""
    return bool(over_bin(part) and part.low[2] < BIN_RIM and part.resting())


def binned(objects) -> int:
    """How many parts lie in the bin and were not lost on the way."""
    return sum(in_bin(objects[f"part_{name(lane, k)}"]) and not lost(lane, k) for lane, k in PARTS)


def score(objects):
    """0.25 for each part lying in the bin (and not lost under the rule above)."""
    return 0.25 * binned(objects)


def success(objects):
    """All four parts lying in the bin, none lost."""
    return binned(objects) == len(PARTS)


def distance(objects):
    """Metres from the first part (in the order they come out) not yet in the bin to the middle of the bin floor."""
    goal = np.array([*BIN_XY, BIN_FLOOR_TOP + CUBE / 2])
    for lane, k in PARTS:
        part = objects[f"part_{name(lane, k)}"]
        if not in_bin(part):
            return float(np.linalg.norm(part.pos - goal))
    return 0.0


# -- the scripted demo ---------------------------------------------------------------------------------------------
# Open loop, timed against the lanes. The left arm takes the parts in the order they would leave reach: each fast
# near-lane part first, though it comes out after the slow far-lane part, then the slow one. The fingers close across
# the lane (yaw -90 degrees), so a cube rides in between the open pads; the arm waits at grasp height in front of a
# coming part and closes as its centre passes, moving along with it while the fingers close, then lifts straight up
# (the carrier slides on from under it) and drops it in the bin.
ORDER = (("near", 1), ("far", 1), ("near", 2), ("far", 2))
YAW = float(-np.pi / 2)
TRAVEL_Z = TOP + 0.17  # TCP height for moving about: a held cube's underside is 8 cm above the parts on the lanes
GRASP_Z = CUBE_Z + 0.004  # TCP a little above the cube's centre: the pad tips stop 1.5 cm above the carrier
DROP_Z = BIN_RIM + 0.035  # a held cube's underside is 1.9 cm above the bin's rim when let go
CATCH_Y = (0.22, 0.40)  # where the demo closes on a part: 9 cm or more from either cover's mouth
CLOSE_S, LIFT_S, CARRY_S, OPEN_S = 0.6, 0.6, 1.3, 0.4
TO_LANE_S, DESCEND_S, SETTLE_S = 1.3, 0.7, 0.3
FIRST_TO_LANE_S = 1.5  # from the arm's ready pose
DROP_XY = {
    ("near", 1): (BIN_XY[0] - 0.025, BIN_XY[1] - 0.025),
    ("far", 1): (BIN_XY[0] + 0.025, BIN_XY[1] - 0.025),
    ("near", 2): (BIN_XY[0] - 0.025, BIN_XY[1] + 0.025),
    ("far", 2): (BIN_XY[0] + 0.025, BIN_XY[1] + 0.025),
}


def part_y(lane: str, k: int, t: float) -> float:
    return float(carrier_path(lane, k)(t)[0][1])


def catch_plan(lane: str, k: int, t_now: float, first: bool = False) -> tuple[list[Waypoint], float, float]:
    """Waypoints that take part ``k`` of ``lane`` as early as possible from ``t_now`` (but not before its centre
    reaches CATCH_Y[0]) and drop it in the bin. Returns the waypoints, when the fingers start closing, and when the
    plan ends."""
    x, v = LANE_X[lane], SPEED[lane]
    ready = t_now + (FIRST_TO_LANE_S if first else TO_LANE_S) + DESCEND_S + SETTLE_S
    close = max(ready, passes(lane, k, CATCH_Y[0]))
    yc = part_y(lane, k, close)
    y1 = yc + v * CLOSE_S  # where the part is when the fingers are closed
    dx, dy = DROP_XY[lane, k]
    plan = [
        Waypoint(FIRST_TO_LANE_S if first else TO_LANE_S, (x, yc, TRAVEL_Z), FINGER_OPEN, f"over {name(lane, k)}", YAW),
        Waypoint(DESCEND_S, (x, yc, GRASP_Z), FINGER_OPEN, f"down in front of {name(lane, k)}", YAW),
        Waypoint(close - ready + SETTLE_S, (x, yc, GRASP_Z), FINGER_OPEN, f"wait for {name(lane, k)}", YAW),
        Waypoint(CLOSE_S, (x, y1, GRASP_Z), FINGER_CLOSED, f"close on {name(lane, k)}", YAW),
        Waypoint(LIFT_S, (x, y1, TRAVEL_Z), FINGER_CLOSED, f"lift {name(lane, k)}", YAW),
        Waypoint(CARRY_S, (dx, dy, DROP_Z), FINGER_CLOSED, f"carry {name(lane, k)}", YAW),
        Waypoint(OPEN_S, (dx, dy, DROP_Z), FINGER_OPEN, f"drop {name(lane, k)}", YAW),
    ]
    return plan, close, close + CLOSE_S + LIFT_S + CARRY_S + OPEN_S


def schedule(order=ORDER) -> list[tuple[tuple[str, int], float]]:
    """(part, when the fingers start closing on it) for each part of ``order``, taken one after another."""
    t, out = 0.0, []
    for i, (lane, k) in enumerate(order):
        _, close, t = catch_plan(lane, k, t, first=i == 0)
        out.append(((lane, k), close))
    return out


def demo():
    """Left arm: the fast part 1 (closing at about 5.0 s), the slow part 1 (10.2 s), the fast part 2 (15.7 s) and the
    slow part 2 (20.9 s), each dropped in the bin; done by 24.6 s."""
    plan, t = [], 0.0
    for i, (lane, k) in enumerate(ORDER):
        take, _, t = catch_plan(lane, k, t, first=i == 0)
        plan += take
    dx, dy = DROP_XY[ORDER[-1]]
    plan.append(Waypoint(0.8, (dx, dy, TRAVEL_Z), FINGER_OPEN, "retreat", YAW))
    return {"left": plan}
