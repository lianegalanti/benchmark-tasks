"""On the station table: a cropped trailer nose (a steel bulkhead panel at the far side with a mounting bracket) carrying
the trailer's two glad hands, bolted face up, the blue (service) one on the robot's left and the red (emergency) one
on its right. Between them and the robot stands the tractor's tender post. From it two air lines run out, a blue one
and a red one, each the end of a coiled air line that pulls back toward the post as it is stretched, ending in a tractor glad hand.
Each tractor glad hand is parked face down on a small stand, in
front of and inboard of its trailer mate.

The robot must couple both lines: bring each tractor glad hand face to face with the trailer glad hand of its colour,
turned 90 degrees to it, press the seals together and turn it 90 degrees about the seal axis, until its flange slides
under the trailer glad hand's lip (and the trailer's under its own) and stops; then let go. The seal ring, compressed
by the lips, holds the coupling, and teeth on the flange tips, in grooves under the lips, keep the line's pull from
sliding it apart.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import Shape, UsdBody, on_reset, on_step  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = "Couple the tractor's service and emergency air lines to the trailer, then let go of both."

# -- objects -------------------------------------------------------------------------------------------------------------
TRAILER = UsdBody("trailer", usd="assets/trailer.usda", pos=tuple(float(v) for v in g.NOSE), size=(0.26, 0.72, 0.30),
                  articulation=True, joint_pos=tuple((f"seal_{c}", 0.0) for c in g.COLOURS),
                  drives=(("seal_.*", g.RING_RATE, g.RING_DAMPING),),
                  drive_targets=tuple((f"seal_{c}", g.RING_TARGET) for c in g.COLOURS),
                  usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001)
TENDER = UsdBody("tender", usd="assets/tender.usda", pos=tuple(float(v) for v in g.TENDER), size=(0.03, 0.03, 0.235),
                 articulation=True, joint_pos=tuple((f"{c}_{j}", 0.0) for c in g.COLOURS for j in g.CHAIN_JOINTS),
                 drives=((".*_(yaw|pitch)", 0.0, 0.02, 0.001), (".*_slide", g.COIL_RATE, g.SLIDE_DAMPING, 0.05),
                         (".*_(roll|bend|twist)", 0.0, 0.004, 0.0005)),
                 usd_materials=True, self_collisions=False, solver_iters=32, contact_offset=0.001,
                 torsional_patch=0.008)
STANDS = [Shape(f"stand_{c}_{i}", pos=tuple(float(v) for v in ctr), size=tuple(float(2 * v) for v in h),
                rot_xyzw=g.quat_xyzw(r), color=(0.40, 0.41, 0.43))
          for c in g.COLOURS for i, (ctr, r, h) in enumerate(g.stand_pads(c))]
OBJECTS = [TRAILER, TENDER, *STANDS]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------------
LOCK_PHI = math.radians(5)      # locked: turned to within 5 degrees of the stop ...
LOCK_TILT = math.radians(5)     # ... faces parallel within 5 degrees ...
LOCK_OFF = 0.006                # ... seal centres within 6 mm (the teeth leave about 2 mm along the pair) ...
LOCK_C = 0.002                  # ... and the seal ring compressed at least 2 mm (only possible with the flanges
#                                 under the lips; the lips hold it at 3 mm)
MEET_TILT = math.radians(10)    # faces together: parallel within 10 degrees, centres within 8 mm, ring pressed 0.5 mm
MEET_OFF = 0.008
MEET_C = 0.0005
LOCKED_CREDIT = 0.45            # per pair: locked, and that glad hand let go (no gripper within CLEAR of it)
HELD_CREDIT = 0.20              # per pair: locked but still held
MEET_CREDIT = 0.10              # per pair: faces together
FALL_V = 0.6                    # m/s: a tractor glad hand no gripper holds falling this fast was dropped (1.8 cm fall)
OFF_TABLE = 0.02                # m: a glad hand's seal face this far below the table top went off the table
SLIDE_END = 0.003               # m: a line within this of its end stop was pulled out to the end
CLEAR = 0.02                    # m: released: every part of both grippers this far from both tractor glad hands
HOLD_R = 0.01                   # m: a gripper may be holding a glad hand when a finger is this close to it
STILL_V, STILL_W = 0.01, 0.1    # m/s and rad/s: tractor glad hands still
STILL_RING = 0.01               # m/s: seal rings still
ENVELOPE = ((-0.062, -0.026, -0.032), (0.052, 0.045, 0.015))   # G frame: the whole casting
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")

_STATE: dict = {}


# -- the grippers as boxes, from each arm's encoders --------------------------------------------------------------------------
def gripper_boxes(tcp, rot, opening: float):
    """The gripper as boxes (centre, rotation, half sizes) in the station frame: each finger's pad (from 3 cm behind the
    TCP to its tip 13 mm past it), each finger's root and the carriage. The fingers close along the TCP's x axis."""
    tcp, rot = np.asarray(tcp, float), np.asarray(rot, float)
    half = opening / 2
    local = []
    for s in (-1.0, 1.0):
        local.append((np.array([s * (half + 0.005), 0.0, -0.0085]), np.array([0.005, 0.020, 0.0215])))
        local.append((np.array([s * (half + 0.010), 0.0, -0.045]), np.array([0.010, 0.020, 0.015])))
    local.append((np.array([0.0, 0.0, -0.1025]), np.array([max(0.051, half + 0.024), 0.034, 0.0425])))
    return [(tcp + rot @ c, rot, h) for c, h in local]


def _rot(quat_xyzw) -> np.ndarray:
    x, y, z, w = np.asarray(quat_xyzw, float) / np.linalg.norm(quat_xyzw)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _arms(environment):
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    from scenekit import arm_joints

    out = {}
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        opening = float(np.abs(arm_joints(environment, side, FINGER_JOINTS)).sum())
        out[side] = {"tcp": tcp, "rot": rot, "opening": opening}
    return out


def envelope(pos, rot):
    return g.place([g.lohi(*ENVELOPE)], pos, rot)[0]


def gap_to(boxes, box) -> float:
    return min(g.boxes_gap(b, box, 5) for b in boxes)


# -- per pair ------------------------------------------------------------------------------------------------------------------
def pair_state(colour_t: str, pos, rot, c_ring: float, trailer_pos=None, trailer_rot=None) -> dict:
    """A tractor glad hand (station pose of its G frame) against trailer glad hand ``colour_t`` whose ring is pressed
    in ``c_ring``: whether the faces are together and whether it is locked."""
    phi, tilt, off, s = g.coupling(colour_t, pos, rot, trailer_pos, trailer_rot)
    meet = tilt <= MEET_TILT and off <= MEET_OFF and c_ring >= MEET_C
    locked = abs(phi) <= LOCK_PHI and tilt <= LOCK_TILT and off <= LOCK_OFF and c_ring >= LOCK_C
    return {"phi_deg": round(math.degrees(phi), 1), "tilt_deg": round(math.degrees(tilt), 1),
            "off_mm": round(off * 1000, 1), "s_mm": round(s * 1000, 1), "c_mm": round(c_ring * 1000, 2),
            "meet": bool(meet), "locked": bool(locked)}


def pair_credit(p: dict, released: bool) -> float:
    if p["locked"]:
        return LOCKED_CREDIT if released else HELD_CREDIT
    return MEET_CREDIT if p["meet"] else 0.0


@on_reset
def reset_state(environment=None) -> None:
    _STATE.clear()
    _STATE.update(why=[], violation=False, level=0.0, locked=False, clear=False, still=False, prev=None, info={})


reset_state()


def _flag(why: str) -> None:
    _STATE["violation"] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def update(trailer_pos, trailer_rot, trailer_joints: dict, glad: dict, slides: dict, arms: dict, dt: float = 0.05) -> None:
    """One control step of the monitor. ``glad``: colour -> (pos, rot) of each tractor glad hand's G frame (station);
    ``slides``: colour -> its line's slide joint (m, 0 at the start); ``arms``: side -> {"tcp", "rot", "opening"} from
    the arm's encoders. Couplings are judged in the trailer's measured frame."""
    S = _STATE
    tp, tr = np.asarray(trailer_pos, float), np.asarray(trailer_rot, float)
    frames = {c: (tp + tr @ (g.TRAILER_GH[c] - g.NOSE), tr @ g.R_TRAILER[c]) for c in g.COLOURS}
    rings = {c: float(trailer_joints.get(f"seal_{c}", 0.0)) for c in g.COLOURS}
    grips = {s: gripper_boxes(a["tcp"], a["rot"], a["opening"]) for s, a in arms.items()}
    env = {c: envelope(*glad[c]) for c in g.COLOURS}
    prev = S["prev"]

    # -- violations (latched for the whole trial) ---------------------------------------------------------------------
    vel, ang = {}, {}
    for c in g.COLOURS:
        p, r = np.asarray(glad[c][0], float), np.asarray(glad[c][1], float)
        if prev is not None:
            p0, r0 = prev["glad"][c]
            vel[c] = (p - p0) / dt
            d = r @ r0.T
            ang[c] = math.acos(max(-1.0, min(1.0, (float(np.trace(d)) - 1) / 2))) / dt
        else:
            vel[c], ang[c] = np.zeros(3), 0.0
        held = any(gap_to(b[:4], env[c]) < HOLD_R for b in grips.values())
        if not held and vel[c][2] < -FALL_V:
            _flag(f"the {c} glad hand was dropped")
        if p[2] < g.TABLE - OFF_TABLE:
            _flag(f"the {c} glad hand went off the table")
        _, _, l0, *_ = g.start_chain(c)
        if float(slides.get(c, 0.0)) >= (g.L_MAX - l0) - SLIDE_END:
            _flag(f"the {c} line was pulled out to its end stop")
        other = "red" if c == "blue" else "blue"
        if pair_state(other, p, r, rings[other], *frames[other])["locked"]:
            _flag(f"the {c} line was coupled to the {other} glad hand (lines crossed)")

    # -- progress --------------------------------------------------------------------------------------------------------
    pairs = {c: pair_state(c, glad[c][0], glad[c][1], rings[c], *frames[c]) for c in g.COLOURS}
    released = {c: all(gap_to(b, env[c]) >= CLEAR for b in grips.values()) for c in g.COLOURS}
    level = sum(pair_credit(pairs[c], released[c]) for c in g.COLOURS)
    locked = all(p["locked"] for p in pairs.values())
    clear = all(released.values())
    ring_v = max(abs(rings[c] - prev["rings"][c]) / dt for c in g.COLOURS) if prev is not None else 0.0
    still = (prev is not None and all(float(np.linalg.norm(vel[c])) < STILL_V and ang[c] < STILL_W for c in g.COLOURS)
             and ring_v < STILL_RING)
    S.update(level=round(level, 4), locked=locked, clear=clear, still=still,
             prev={"glad": {c: (np.asarray(glad[c][0], float).copy(), np.asarray(glad[c][1], float).copy())
                            for c in g.COLOURS}, "rings": dict(rings)},
             info={"pairs": pairs, "slides_mm": {c: round(float(slides.get(c, 0.0)) * 1000, 1) for c in g.COLOURS}})


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    st = object_states(environment, ["trailer", "tender"])
    t, h = st["trailer"], st["tender"]
    glad = {}
    for c in g.COLOURS:
        lp, lq = h.links[f"{c}_gladhand"]
        r = _rot(lq)
        glad[c] = (np.asarray(lp, float) - r @ g.PORT, r)
    slides = {c: float(h.joints.get(f"{c}_slide", 0.0)) for c in g.COLOURS}
    dt = float(getattr(environment.unwrapped, "step_dt", 0.05))
    update(t.pos, _rot(t.quat_xyzw), t.joints, glad, slides, _arms(environment), dt)


def violation() -> bool:
    return bool(_STATE["violation"])


def success(objects) -> bool:
    """No violation so far; both tractor glad hands locked on their own colour's trailer glad hand; every part of both
    grippers at least 2 cm from both; both glad hands and both seal rings still. Held for 1 s."""
    S = _STATE
    return bool(not violation() and S["locked"] and S["clear"] and S["still"])


def score(objects) -> float:
    """0 after any violation: a tractor glad hand dropped (falling faster than 0.6 m/s with no gripper finger within
    1 cm) or off the table, a line pulled out to its end stop, or a line locked onto the other colour's glad hand.
    Otherwise, per pair at the end: 0.45 locked (within 5 degrees of the stop, faces parallel within 5 degrees, centred
    within 6 mm, ring compressed 2 mm or more) and let go (no gripper within 2 cm of that glad hand); 0.2 locked but
    still held; 0.1 faces together (parallel within 10 degrees, centred within 8 mm, ring pressed 0.5 mm or more).
    Success (both locked, released, still, held 1 s) scores 1."""
    return 0.0 if violation() else float(_STATE["level"])
