"""On the station table, a conveyor line runs across the table from right to left (toward the left arm), 34 cm in front
of the arms: a black track bed (12 cm wide, 1.5 cm tall) with three flat grey carrier plates (10 x 10 x 1.5 cm) riding
on it, 22 cm apart. Each carrier holds one small brown cardboard carton (6 x 6 x 8 cm tall, 150 g) standing upright in
its middle. Each carton has a white shipping label with black barcode bars on one of its four sides, under an orange
band that folds over onto the carton's top. The first carton's label faces the robot's right (back along the line),
the second's faces the robot, the third's faces the robot's left (along the line, the way it moves).

Near the table's left end the line runs through a scanner gate: two dark grey posts either side of the line with a
crossbar over it, and a black scanner head with a red window on the far post (the side away from the robot), looking
across the line at the cartons. It reads each carton's label as the carton passes through.

The line starts with the trial and never stops for the robot: it reaches 4 cm/s within 0.5 s, so a carton comes
through the gate every 5.5 s. The robot must turn each carton on its carrier, before it reaches the gate, so that its
label faces the scanner (away from the robot), leave it standing upright on its own carrier, and have let go of it
before it reaches the gate.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from motionkit import DrivenBox, UsdBody, belt_run, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
# "end": each carton is scored once, by the monitor, as its carrier passes the gate; score() reports that tally.
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Before each carton reaches the scanner gate, turn it on its carrier so that its shipping label faces "
               "the scanner on the far side of the line. Leave each carton standing upright on its own carrier, and "
               "let go of it before it reaches the gate.")


def _f(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab turns np.float64 poses into float64 tensors, which warp rejects."""
    return tuple(float(x) for x in v)


# -- the line ----------------------------------------------------------------------------------------------------------
NAMES = g.NAMES
PATHS = {n: belt_run((g.LINE_X, g.STARTS[n], g.CARRIER_Z), g.SPEED, g.RAMP_S, g.TURN_Y, g.DROP, g.DROP_SPEED)
         for n in NAMES}
CARRIER_OF = {n: f"carrier_{k + 1}" for k, n in enumerate(NAMES)}
BED_BOX = Box("track_bed", pos=_f((g.LINE_X, 0.0, g.TOP + g.BED[2] / 2)), size=_f(g.BED), color=(0.08, 0.08, 0.09))
CARRIERS = [DrivenBox(CARRIER_OF[n], pos=_f((g.LINE_X, g.STARTS[n], g.CARRIER_Z)), size=_f(g.CARRIER),
                      color=(0.5, 0.52, 0.55), path=PATHS[n]) for n in NAMES]
CARTONS = [UsdBody(n, usd="assets/carton.usda", pos=_f((g.LINE_X, g.STARTS[n], g.CARTON_Z + 0.001)),
                   rot_xyzw=_f(g.yaw_quat(yaw)), size=_f(g.SIZE), mass=g.MASS, friction=(0.9, 0.8),
                   torsional_patch=0.01)
           for n, yaw in zip(NAMES, g.START_YAWS, strict=True)]


def carrier_y(name: str, t: float) -> float:
    """Where carton ``name``'s carrier is along the line (its centre's y) at episode time ``t``."""
    return float(PATHS[name](t)[0][1])


def arrival(name: str, y: float) -> float:
    """When carton ``name``'s carrier reaches ``y`` (after the ramp; the line runs at constant speed then)."""
    return float((y - g.STARTS[name]) / g.SPEED + g.RAMP_S / 2)


# -- the scanner gate (fixed) --------------------------------------------------------------------------------------------
GATE_GREY, SCANNER_BLACK, WINDOW_RED = (0.25, 0.26, 0.28), (0.05, 0.05, 0.06), (0.9, 0.08, 0.06)
POSTS = [Box(f"gate_post_{side}", pos=_f((g.LINE_X + sign * g.POST_X, g.GATE_Y, g.TOP + g.POST[2] / 2)),
             size=_f(g.POST), color=GATE_GREY) for side, sign in (("near", -1), ("far", 1))]
BEAM_BOX = Box("gate_beam", pos=_f((g.LINE_X, g.GATE_Y, g.TOP + g.POST[2] + g.BEAM[2] / 2)), size=_f(g.BEAM),
               color=GATE_GREY)
SCANNER_X = g.LINE_X + g.POST_X - g.POST[0] / 2 - g.SCANNER[0] / 2  # against the far post, on the line's side
SCANNER_BOX = Box("scanner_head", pos=_f((SCANNER_X, g.GATE_Y, g.SCANNER_Z)), size=_f(g.SCANNER), color=SCANNER_BLACK)
WINDOW_BOX = Box("scanner_window", pos=_f((SCANNER_X - g.SCANNER[0] / 2 - g.WINDOW[0] / 2, g.GATE_Y, g.SCANNER_Z)),
                 size=_f(g.WINDOW), color=WINDOW_RED)
GATE = [*POSTS, BEAM_BOX, SCANNER_BOX, WINDOW_BOX]

OBJECTS = [BED_BOX, *GATE, *CARRIERS, *CARTONS]

# -- the monitor: each carton is read once, as its carrier's centre passes the gate ------------------------------------
_STATE: dict = {}


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(reads={}, lost={})


reset_state()


def update(cartons: dict, carriers: dict, grasp_points) -> None:
    """One control step. ``cartons`` and ``carriers``: per carton name, its carton's and its carrier's
    (pos, quat_xyzw); ``grasp_points``: both grippers' grasp points. Marks a carton lost for good once its lowest
    corner comes within 1 cm of the table top, and reads each carton once, when its carrier's centre reaches the gate."""
    for name in NAMES:
        pos, quat = cartons[name]
        R = g.rot(quat)
        if name not in _STATE["lost"] and g.lowest_z(pos, R) < g.TOP + g.TABLE_TOUCH:
            _STATE["lost"][name] = "touched the table"
        cpos, _ = carriers[name]
        if name not in _STATE["reads"] and cpos[1] >= g.GATE_Y:
            r = g.check(pos, R, cpos, grasp_points)
            if name in _STATE["lost"]:
                r.update(read=False, lost=True)
            _STATE["reads"][name] = r


@on_step
def _monitor(environment) -> None:
    from motionkit import arm_joints
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.scene import object_states

    names = [*NAMES, *CARRIER_OF.values()]
    s = object_states(environment, names)
    tcps = [ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])[0]
            for side in ("left", "right")]
    update({n: (s[n].pos, s[n].quat_xyzw) for n in NAMES},
           {n: (s[CARRIER_OF[n]].pos, s[CARRIER_OF[n]].quat_xyzw) for n in NAMES}, tcps)


def reads() -> dict:
    """What the scanner made of each carton that has passed the gate so far."""
    return {n: dict(r) for n, r in _STATE["reads"].items()}


def success(objects) -> bool:
    """All three cartons read at the gate."""
    return bool(len(_STATE["reads"]) == len(NAMES) and all(r["read"] for r in _STATE["reads"].values()))


def distance(objects) -> float:
    """The worst label error (radians) among the cartons not yet through the gate; 0 once all have passed."""
    left = [n for n in NAMES if n not in _STATE["reads"]]
    return max((g.label_error(g.rot(objects[n].quat_xyzw)) for n in left), default=0.0)


def score(objects) -> float:
    """A third for each carton read at the gate: as its carrier's centre passed the gate, it stood upright (within 10
    degrees) on its own carrier (centre within 3 cm of the carrier's, underside within 1 cm of its top), with both
    grippers' grasp points 8 cm or more from its centre and its label facing within 20 degrees of straight at the
    scanner; and it never touched the table."""
    return sum(1 for r in _STATE["reads"].values() if r["read"]) / len(NAMES)


# -- the scripted demo ---------------------------------------------------------------------------------------------------
# Each turn: be over the line where the carton will be at its meeting time, then, with one waypoint per control step
# following the carrier, go down, close, lift the carton 1.5 cm, turn it with the wrist, set it back down 3 mm above
# its carrier, let go and go up. The right arm gives carton 2 the first quarter of its half turn and turns carton 3;
# the left arm turns carton 1 and gives carton 2 its second quarter.
DT = 1 / 20
PRE_Z = g.CARRIER_TOP + g.SIZE[2] + 0.025 + PAD_TIP_PAST_TCP   # open pad tips 2.5 cm above the cartons' tops
GRASP_Z = g.CARTON_Z + 0.012                                     # pad tips 3.9 cm above the carrier's top
LIFT_M, SET_M = 0.015, 0.003
DESCEND_S, CLOSE_S, LIFT_S, TURN_S, LOWER_S, OPEN_S, UP_S = 0.6, 0.5, 0.4, 1.4, 0.4, 0.5, 0.5
TURN_TOTAL_S = DESCEND_S + CLOSE_S + LIFT_S + TURN_S + LOWER_S + OPEN_S + UP_S

# (arm, carton, where along the line the gripper meets it, grip yaw before and after the turn, degrees). The fingers
# close across two opposite sides of the carton, so a grip yaw is one of the carton's own yaws modulo 90 degrees;
# turning the grip by d turns the carton by d.
TURNS = (
    ("right", "carton_2", -0.28, 0.0, 90.0),     # label toward the robot (180) -> the robot's right (-90)
    ("right", "carton_3", -0.26, 90.0, 0.0),     # label toward the robot's left (90) -> the scanner (0)
    ("left", "carton_1", 0.16, -90.0, 0.0),      # label toward the robot's right (-90) -> the scanner (0)
    ("left", "carton_2", 0.18, -90.0, 0.0),      # label toward the robot's right (-90) -> the scanner (0)
)
REST = {"right": (0.22, -0.28, g.TOP + 0.16), "left": (0.22, 0.28, g.TOP + 0.16)}
AWAY_Z = g.TOP + 0.15  # straight up to here first (higher is out of reach over the line)


def _smooth(s: float) -> float:
    return s * s * (3 - 2 * s)


def _track(name, t0, seconds, z0, z1, f0, f1, yaw0, yaw1, label):
    """Waypoints one control step apart over carton ``name``'s carrier from ``t0``, while the TCP goes from ``z0`` to
    ``z1``, the fingers from ``f0`` to ``f1`` and the grip yaw from ``yaw0`` to ``yaw1`` (degrees), all smoothly."""
    n = round(seconds / DT)
    out = []
    for k in range(1, n + 1):
        s = _smooth(k / n)
        y = carrier_y(name, t0 + k * DT)
        out.append(Waypoint(DT, (g.LINE_X, y, z0 + s * (z1 - z0)), f0 + s * (f1 - f0), label,
                            math.radians(yaw0 + s * (yaw1 - yaw0))))
    return out


def turn(name: str, t0: float, yaw0: float, yaw1: float) -> list[Waypoint]:
    """One turn of carton ``name`` that starts over it at ``t0`` (the gripper open at PRE_Z, at grip yaw ``yaw0``)."""
    hold, up, seat = GRASP_Z, GRASP_Z + LIFT_M, GRASP_Z + SET_M
    plan, t = [], t0
    for secs, z0, z1, f0, f1, a0, a1, label in (
        (DESCEND_S, PRE_Z, hold, FINGER_OPEN, FINGER_OPEN, yaw0, yaw0, "down"),
        (CLOSE_S, hold, hold, FINGER_OPEN, FINGER_CLOSED, yaw0, yaw0, "close"),
        (LIFT_S, hold, up, FINGER_CLOSED, FINGER_CLOSED, yaw0, yaw0, "lift"),
        (TURN_S, up, up, FINGER_CLOSED, FINGER_CLOSED, yaw0, yaw1, "turn"),
        (LOWER_S, up, seat, FINGER_CLOSED, FINGER_CLOSED, yaw1, yaw1, "set down"),
        (OPEN_S, seat, seat, FINGER_CLOSED, FINGER_OPEN, yaw1, yaw1, "let go"),
        (UP_S, seat, PRE_Z, FINGER_OPEN, FINGER_OPEN, yaw1, yaw1, "up"),
    ):
        plan += _track(name, t, secs, z0, z1, f0, f1, a0, a1, f"{label} {name}")
        t += secs
    return plan


END_S = 30.0  # the plans end here: the last read is at 26.75 s and success holds from about 27.8 s


def demo():
    """Both arms, each turn in TURNS in order: be over the line where the carton will be at its meeting time, then
    turn() it while following its carrier; after its last turn each arm goes up and back to wait."""
    plans: dict[str, list[Waypoint]] = {"left": [], "right": []}
    clock = {"left": 0.0, "right": 0.0}
    for arm, name, meet_y, yaw0, yaw1 in TURNS:
        meet = arrival(name, meet_y)
        span = round((meet - clock[arm]) / DT) * DT
        assert span >= 0.8, f"{arm} arm: only {span:.2f} s to get over {name}"
        plans[arm].append(Waypoint(span, (g.LINE_X, carrier_y(name, clock[arm] + span), PRE_Z), FINGER_OPEN,
                                   f"over {name}", math.radians(yaw0)))
        plans[arm] += turn(name, clock[arm] + span, yaw0, yaw1)
        clock[arm] += span + TURN_TOTAL_S
    for arm, plan in plans.items():
        last = plan[-1]
        plan += [Waypoint(0.6, (last.pos[0], last.pos[1], AWAY_Z), FINGER_OPEN, "up and away", last.yaw),
                 Waypoint(1.4, REST[arm], FINGER_OPEN, "back to wait", 0.0)]
        # yam_scripted ends the rollout 3 s after the plan does: wait there until the last read and success
        plan.append(Waypoint(END_S - sum(w.duration for w in plan), REST[arm], FINGER_OPEN, "wait", 0.0))
    return plans


# Kept for tests: the read moment of each carton.
READ_TIMES = {n: arrival(n, g.GATE_Y) for n in NAMES}
assert np.all(np.diff(list(READ_TIMES.values())) > 0)
