"""On the station table, a grey plywood track (22 cm wide, 1.08 m long) runs across the front of the arms, 30 cm in
front of them, sloping down 1.2 degrees toward the left and ending 1 cm short of the table's left edge. Two
low-friction rails and a central guide bar run along its top. A blue model cart (12 x 16 x 6 cm, 330 g) sits on the
rails in front of the right arm with its parking brake off: an orange brake lever on the cart's near side (the side
facing the arms) leans out toward the arms. When the trial starts the cart is let go at rest and rolls down the
slope; untouched, it runs off the end of the track and off the table about 3.6 s later.

The robot must stop the cart before it runs off, set its brake by pushing the lever over toward the cart (its rubber
pad then presses on the deck and holds the cart on the slope), and let go. Stopped but left unbraked, the cart rolls
away again: it is only safe once the brake is on.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score \\
           --instruction "<INSTRUCTION below>" ...
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import BodyState, Shape, UsdBody, on_reset, on_step  # noqa: E402
from yam_arena.constants import FINGER_CLOSED, PAD_TIP_PAST_TCP  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"  # where the cart is when the trial ends: a cart stopped and then let roll off scores 0
SUCCESS_HOLD_S = 2.0

INSTRUCTION = ("Stop the rolling cart before it runs off the end of its track, then set its brake by pushing the "
               "orange lever over toward the cart, and let go of the cart.")

GREY, STEEL = (0.62, 0.60, 0.56), (0.45, 0.47, 0.50)


def _floats(v) -> tuple[float, ...]:
    """Plain Python floats: Isaac Lab's Pose.to_tensor turns np.float64 into a float64 tensor, which warp rejects."""
    return tuple(float(x) for x in v)


def _slab(name: str, x0: float, x1: float, below: float, above: float, color, friction, friction_min) -> Shape:
    """A box along the whole track, ``x0..x1`` across it, from ``below`` under the deck top to ``above`` over it."""
    centre = g.deck_point(g.DECK_LEN / 2, (x0 + x1) / 2) + (above - below) / 2 * g.NORMAL
    return Shape(name, pos=_floats(centre), size=_floats((x1 - x0, g.DECK_LEN, above + below)), color=color,
                 rot_xyzw=_floats(g.ROT_XYZW), friction=friction, friction_min=friction_min)


# -- the track -------------------------------------------------------------------------------------------------------
DECK = _slab("deck", *g.DECK_X, g.DECK_T, 0.0, GREY, g.DECK_MU, False)
SLICK = (g.RAIL_MU, g.RAIL_MU)
RAILS = [_slab(f"rail_{side}", s * g.RAIL_X - g.RAIL_W / 2, s * g.RAIL_X + g.RAIL_W / 2, 0.001, g.RAIL_H, STEEL,
               SLICK, True) for side, s in (("near", -1), ("far", 1))]
GUIDE = _slab("guide_bar", -g.GUIDE_W / 2, g.GUIDE_W / 2, 0.001, g.GUIDE_H, STEEL, SLICK, True)

# -- the cart --------------------------------------------------------------------------------------------------------
CART = UsdBody("cart", usd="assets/cart.usda", pos=_floats(g.origin_at(g.Q_START)), rot_xyzw=_floats(g.ROT_XYZW),
               size=(0.20, 0.16, 0.12), articulation=True, free_start=True, joint_pos=(("brake", g.STOWED),),
               joint_damping=0.002, friction=g.CART_MU, self_collisions=False)

OBJECTS = [DECK, *RAILS, GUIDE, CART]

# -- scoring ---------------------------------------------------------------------------------------------------------
BRAKED_MIN = math.radians(8.0)   # the lever at least this far over (its stop is at 12): past the toggle point
STILL_SPEED, STILL_SPIN = 0.005, 0.2
CLEAR = 0.02                     # m: every part of both arms at least this far from the cart's envelope
ARM_R, PAD_R = 0.03, 0.02        # arm links as 3 cm capsules between joints; the gripper as a 2 cm one to the tips
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")

_STATE: dict = {}


@on_reset
def _reset_state(environment=None) -> None:
    _STATE.update(fallen=False, clear=False)


_reset_state()


def placement(cart: BodyState) -> dict:
    """Where the cart is on the track: q (its centre's distance up the slope from the track's end), its sideways and
    upward offsets from where it sits on the rails, and its tilt from the deck's normal (degrees)."""
    q, off = g.to_track(cart.pos)
    up = cart.rotation()[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(up @ g.NORMAL, -1.0, 1.0))))
    return {"q": q, "side": float(off[0]), "up": float(off[2]), "tilt": tilt}


def _gone(p: dict) -> bool:
    """Off the track: its centre more than 1 cm past either end, 3 cm below the rails, or 11 cm to the side."""
    return bool(p["q"] < -0.01 or p["q"] > g.DECK_LEN + 0.01 or p["up"] < -0.03 or abs(p["side"]) > 0.11)


def on_deck(p: dict) -> bool:
    """On the track, upright (within 30 degrees) and not held up off it (within 3 cm of the rails)."""
    return bool(not _gone(p) and 0.0 <= p["q"] <= g.DECK_LEN and abs(p["up"]) < 0.03 and p["tilt"] < 30.0)


def on_rails(p: dict) -> bool:
    """Both ends over the track, square on its rails: within 8 mm of the centre line, 3 mm below to 4 mm above its
    seat (the engaged pad lifts it about 1 mm) and 4 degrees of the deck's slope."""
    return bool(g.CART_HALF_LEN <= p["q"] <= g.DECK_LEN - g.CART_HALF_LEN and abs(p["side"]) < 0.008
                and -0.003 < p["up"] < 0.004 and p["tilt"] < 4.0)


def braked(cart: BodyState) -> bool:
    return bool(cart.joints.get("brake", g.STOWED) >= BRAKED_MIN)


def still(cart: BodyState) -> bool:
    return cart.resting(max_speed=STILL_SPEED, max_spin=STILL_SPIN)


def observe(cart: BodyState, clear: bool) -> None:
    """One control step's facts that scoring needs beyond the cart's state: whether it has ever left the track
    (latched) and whether both arms are clear of it now."""
    if _gone(placement(cart)):
        _STATE["fallen"] = True
    _STATE["clear"] = bool(clear)


def _seg(a, b, n: int = 16) -> np.ndarray:
    return np.asarray(a, float) + np.outer(np.linspace(0.0, 1.0, n), np.asarray(b, float) - np.asarray(a, float))


def cart_distance(cart: BodyState, points) -> np.ndarray:
    """Distance of each station point (rows) to the cart: to the nearer of a box round its body (with runners, shoes
    and bracket) and a box round its brake lever where the lever is now; 0 inside."""
    local = (np.atleast_2d(points) - cart.pos) @ cart.rotation()
    out = []
    for lo, hi in (g.BODY_BOX, g.lever_box(cart.joints.get("brake", g.STOWED))):
        centre, half = (lo + hi) / 2, (hi - lo) / 2
        out.append(np.linalg.norm(np.maximum(np.abs(local - centre) - half, 0.0), axis=1))
    return np.minimum(*out)


def _arms_clear(environment, cart: BodyState) -> bool:
    from scenekit import arm_joints, robot_links
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS

    links = robot_links(environment)
    for side in ("left", "right"):
        tcp, rot = ik.tcp_pose(arm_joints(environment, side, ARM_JOINTS), station.BASES[side])
        chain = [links[side][n] for n in CHAIN if n in links[side]]
        caps = [(a, b, ARM_R) for a, b in zip(chain[:-1], chain[1:])]
        caps.append((chain[-1], tcp + rot[:, 2] * PAD_TIP_PAST_TCP, PAD_R))
        if any(cart_distance(cart, _seg(a, b)).min() - r < CLEAR for a, b, r in caps):
            return False
    return True


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    cart = object_states(environment, ["cart"])["cart"]
    observe(cart, _arms_clear(environment, cart))


def success(objects) -> bool:
    """Square on its rails, its brake on (lever at least 8 degrees over), still, and every part of both arms 2 cm
    clear of it, without ever having left the track; held for 2 s."""
    cart = objects["cart"]
    p = placement(cart)
    return bool(not _STATE["fallen"] and on_rails(p) and braked(cart) and still(cart) and _STATE["clear"])


def distance(objects) -> float:
    """Degrees the brake lever still has to turn to count as set."""
    return max(0.0, math.degrees(BRAKED_MIN - objects["cart"].joints.get("brake", g.STOWED)))


def score(objects) -> float:
    """At the end of the trial (success scores 1.0): 0 if the cart ever left the track (ran off its end, or was
    pushed or lifted off it) or isn't resting upright on it now; 0.25 if it is on the track but still moving or off
    its rails; 0.5 if it is still and on its rails with the brake off (so something is holding it); 0.75 if the brake
    is on and it is still, but an arm is within 2 cm of it (or that has not yet held for 2 s)."""
    cart = objects["cart"]
    p = placement(cart)
    if _STATE["fallen"] or not on_deck(p):
        return 0.0
    if not (on_rails(p) and still(cart)):
        return 0.25
    if not braked(cart):
        return 0.5
    return 0.75  # braked and still: either an arm is still on it, or it has not yet held 2 s (success's 1.0)


# -- the scripted demo -----------------------------------------------------------------------------------------------
# Left arm: closed fingers (closing along y) stand in the cart's path in front of the left arm; the cart rolls into
# their flat side and stops against them, pressed there by the slope. Right arm: closed fingers come down outside the
# leaning brake handle and push it over toward the cart, past straight up, to its stop. Then both arms leave.
FINGER_HALF = 0.0097     # left TCP to where the cart's face stops (measured in Isaac Sim: finger face and the arm's give)
FINGER_EDGE = 0.016      # TCP to the fingers' edge, across the closing direction (YAM tip, stop-ball-before-edge)
STOP_Y = 0.12            # the left TCP: where the cart's leading face stops, plus FINGER_HALF
BLOCK_TIP_Z = 0.025      # fingertips above the rail tops (the cart's face runs from 1 to 6 cm)
BLOCK_X = 0.03           # ... on the far half of the face, away from the right arm at the lever
Q_STOP = g.q_of_y(STOP_Y - FINGER_HALF) + g.CART_HALF_LEN   # the stopped cart's centre
PUSH_H = 0.045           # fingertips above the lever's hinge axis while pushing
PUSH_CLEAR = 0.015       # between the fingers' edge and the stowed handle as they come down
TRAVEL_UP = 0.16         # TCP height above the table top to come in and leave at


def _tcp(q: float, local) -> tuple[float, float, float]:
    """The TCP for fingertips at a track-frame point (a cart centred at q)."""
    return _floats(g.to_station(q, local) + np.array([0.0, 0.0, PAD_TIP_PAST_TCP]))


BLOCK = _tcp(g.q_of_y(STOP_Y), (BLOCK_X, 0.0, BLOCK_TIP_Z))
PUSH_Z = g.PIVOT_Z + PUSH_H
PUSH_FROM_X = float(g.lever_corners(g.STOWED)[:, 0].min()) - PUSH_CLEAR - FINGER_EDGE
PUSH_PAST = 0.023        # past the handle's face at its stop: the arm gives about 2 cm against the lever (Isaac Sim)
PUSH_TO_X = g.handle_face_x(g.ENGAGED, PUSH_H) + PUSH_PAST - FINGER_EDGE
PUSH_FROM = _tcp(Q_STOP, (PUSH_FROM_X, g.LEVER_Y, PUSH_Z))
PUSH_TO = _tcp(Q_STOP, (PUSH_TO_X, g.LEVER_Y, PUSH_Z))
TO_BLOCK_S = 1.2


def arrival_s(mu: float = g.RAIL_MU) -> float:
    """When the cart, rolling untouched from rest, reaches the left fingers."""
    return g.roll_time(g.Q_START - Q_STOP, mu)


def off_end_s(mu: float = g.RAIL_MU) -> float:
    """When the untouched cart's centre passes the end of the track (it tips off and falls)."""
    return g.roll_time(g.Q_START, mu)


def demo():
    above = (PUSH_FROM[0], PUSH_FROM[1], g.TOP + TRAVEL_UP)
    back = (PUSH_TO[0] - 0.07, PUSH_TO[1], PUSH_TO[2])
    right = [
        Waypoint(arrival_s() + 1.2, above, FINGER_CLOSED, "above the lever"),
        Waypoint(0.8, PUSH_FROM, FINGER_CLOSED, "down beside the handle"),
        Waypoint(1.6, PUSH_TO, FINGER_CLOSED, "push the lever over"),
        Waypoint(0.3, PUSH_TO, FINGER_CLOSED, "hold it at its stop"),
        Waypoint(0.5, back, FINGER_CLOSED, "back off"),
        Waypoint(0.8, (back[0], back[1], g.TOP + TRAVEL_UP), FINGER_CLOSED, "up and away"),
    ]
    braked_at = sum(w.duration for w in right[:5])
    bx, by, bz = BLOCK
    left = [
        Waypoint(TO_BLOCK_S, BLOCK, FINGER_CLOSED, "stop in its path"),
        Waypoint(braked_at - TO_BLOCK_S, BLOCK, FINGER_CLOSED, "hold while it is braked"),
        Waypoint(0.5, (bx, by + 0.04, bz), FINGER_CLOSED, "back off"),
        Waypoint(0.8, (bx, by + 0.04, g.TOP + TRAVEL_UP), FINGER_CLOSED, "up and away"),
    ]
    return {"left": left, "right": right}
