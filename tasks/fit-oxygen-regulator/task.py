"""On the station table, ahead of the robot and between its arms: an empty size D aluminium medical oxygen cylinder
(11.1 cm across, 42 cm long, green shoulder) lying along x in a wooden cradle of two V saddles, its base against an
end board and its brass pin-index post valve pointing at the robot, 12 cm above the table. The valve's outlet faces
the robot's right and has the two oxygen pin holes below it, toward the cylinder; a conical recess faces the robot's
left. In front of the right arm a yoke oxygen regulator (blue body, aluminium yoke, a black T-handle screw on the far
side of the yoke) stands on the table the way it goes on: yoke toward the valve's side, inlet and its two pins facing
the robot's left, T-handle backed off.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from regkit import Screw, couple_screw, on_reset, on_step, part, reset_joints  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Fit the regulator onto the oxygen cylinder's valve, hand-tight and ready to use, and let go of it.")

BRASS = (0.78, 0.62, 0.30)
GREEN = (0.15, 0.55, 0.25)
ALUMINIUM = (0.80, 0.81, 0.83)
YOKE = (0.70, 0.72, 0.75)
BLUE = (0.15, 0.35, 0.75)
BLACK = (0.08, 0.08, 0.09)
WHITE = (0.93, 0.93, 0.90)
STEEL = (0.60, 0.62, 0.64)
WOOD = (0.62, 0.45, 0.28)


def _b(lo, hi, color, rot=None):
    """A box from its low and high corners (and optionally a rotation, w x y z)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    spec = (tuple(float(v) for v in (lo + hi) / 2), tuple(float(v) for v in hi - lo), color)
    return spec if rot is None else (*spec, tuple(float(v) for v in rot))


def _c(centre, radius, height, color, axis):
    return (tuple(float(v) for v in centre), float(radius), float(height), color, axis)


# -- the cylinder and its valve (C frame) ------------------------------------------------------------------------------
def _slab(x0, x1, z0, z1, y0, y1, holes):
    """Boxes filling [x0, x1] x [y0, y1] x [z0, z1] except the rectangular ``holes`` ((hx0, hx1, hz0, hz1), inside the
    slab; holes either share an x range or don't overlap in x)."""
    columns: dict = {}
    for hx0, hx1, hz0, hz1 in holes:
        columns.setdefault((hx0, hx1), []).append((hz0, hz1))
    out, x = [], x0
    for (hx0, hx1), spans in sorted(columns.items()):
        out.append(_b((x, y0, z0), (hx0, y1, z1), BRASS))
        z = z0
        for hz0, hz1 in sorted(spans):
            out.append(_b((hx0, y0, z), (hx1, y1, hz0), BRASS))
            z = hz1
        out.append(_b((hx0, y0, z), (hx1, y1, z1), BRASS))
        x = hx1
    out.append(_b((x, y0, z0), (x1, y1, z1), BRASS))
    return out


def _bevel(centre, length, axis, about):
    """A square bar along ``axis`` turned 45 degrees ``about`` that axis: its face between two of its corners is a
    45-degree lead-in whose ends are ``half`` either side of ``centre``."""
    half = g.COUNTERSINK if about != "end" else g.END_CHAMFER
    side = half * math.sqrt(2.0)
    size = {"x": (length, side, side), "y": (side, length, side), "z": (side, side, length)}[axis]
    q = {"x": (math.cos(math.pi / 8), math.sin(math.pi / 8), 0.0, 0.0),
         "y": (math.cos(math.pi / 8), 0.0, math.sin(math.pi / 8), 0.0),
         "z": (math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8))}[axis]
    c = np.asarray(centre, float)
    return _b(c - np.asarray(size) / 2, c + np.asarray(size) / 2, BRASS, q)


def _post():
    hy, hz, L = g.POST_HY, g.POST_HZ, g.POST_LEN
    e = g.END_CHAMFER
    skin = g.POST_HY - g.HOLE_DEPTH              # -y skin holds the pin holes, 6 mm deep
    floor = g.POST_HY - g.RECESS_DEPTH           # +y skin holds the recess, 2 mm deep
    c = g.COUNTERSINK
    holes = [(h[0] - g.HOLE_W / 2, h[0] + g.HOLE_W / 2, h[2] - g.HOLE_W / 2, h[2] + g.HOLE_W / 2) for h in g.HOLES_C]
    mouths = [(x0 - c, x1 + c, z0 - c, z1 + c) for x0, x1, z0, z1 in holes]
    rx0, rx1 = g.PORT_U - g.RECESS_W / 2, g.PORT_U + g.RECESS_W / 2
    rz = g.RECESS_W / 2
    parts = [
        _b((e, -skin, -hz), (L, floor, hz), BRASS),                       # core
        # the free end: 3 mm narrower top and bottom, with 45-degree chamfers up to the full width
        _b((0, -hy, -hz + e), (e, hy, hz - e), BRASS),
        _bevel((e, 0.0, hz - e), 2 * hy, "y", "end"),
        _bevel((e, 0.0, -hz + e), 2 * hy, "y", "end"),
        # outlet face (-y): the two pin holes, with a 0.6 mm 45-degree countersink at their mouths
        *_slab(e, L, -hz, hz, -hy + c, -skin, holes),
        *_slab(e, L, -hz, hz, -hy, -hy + c, mouths),
        # recess face (+y) with the conical recess (a square pocket)
        *_slab(e, L, -hz, hz, floor, hy, [(rx0, rx1, -rz, rz)]),
        # the spindle's square end on the free end
        _b((-0.005, -g.STEM / 2, -g.STEM / 2), (0.0, g.STEM / 2, g.STEM / 2), STEEL),
    ]
    for x0, x1, z0, z1 in holes:                 # countersink faces: bars on the hole's four mouth edges
        y = -hy + c
        zc, xc = (z0 + z1) / 2, (x0 + x1) / 2
        parts += [_bevel((x0 - c, y, zc), z1 - z0 + 2 * c, "z", "hole"),
                  _bevel((x1 + c, y, zc), z1 - z0 + 2 * c, "z", "hole"),
                  _bevel((xc, y, z0 - c), x1 - x0 + 2 * c, "x", "hole"),
                  _bevel((xc, y, z1 + c), x1 - x0 + 2 * c, "x", "hole")]
    return parts


def _cylinder_shapes():
    cyls = [_c((g.POST_LEN + g.NECK_LEN / 2, 0, 0), g.NECK_R, g.NECK_LEN, BRASS, "X")]
    x = g.CYL_X0
    for r, length in g.SHOULDER:
        cyls.append(_c((x + length / 2, 0, 0), r, length, GREEN, "X"))
        x += length
    cyls.append(_c((x + 0.03, 0, 0), g.CYL_R, 0.06, GREEN, "X"))           # green shoulder band
    cyls.append(_c(((x + 0.06 + g.CYL_END) / 2, 0, 0), g.CYL_R, g.CYL_END - x - 0.06, ALUMINIUM, "X"))
    return cyls


CYLINDER = part("cylinder", pos=(g.C_ORIGIN[0], 0.0, g.AXIS_Z + 0.0005), boxes=_post(), cylinders=_cylinder_shapes(), mass=g.CYL_MASS,
                color=ALUMINIUM, friction=0.25)   # polished brass valve, aluminium cylinder


# -- the cradle (fixed; frame at the table under the cylinder's middle) --------------------------------------------------
CRADLE_ORIGIN = np.array([0.58, 0.0, g.TABLE])


def _cradle_boxes():
    out = []
    s45 = math.sin(math.radians(45.0))
    for xs in g.SADDLE_X:
        x0, x1 = xs - g.SADDLE_W / 2, xs + g.SADDLE_W / 2
        out.append(_b((x0, -0.05, g.TABLE), (x1, 0.05, g.V_APEX_Z - 0.002), WOOD))
        for side in (-1.0, 1.0):
            contact = np.array([xs, side * g.CYL_R * s45, g.AXIS_Z - g.CYL_R * s45])
            normal = np.array([0.0, -side * s45, s45])
            centre = contact - normal * g.BOARD[1] / 2
            a = side * math.radians(45.0)
            half = np.array([g.SADDLE_W, g.BOARD[0], g.BOARD[1]]) / 2
            out.append(_b(centre - half, centre + half, WOOD, (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)))
    out.append(_b((g.END_BOARD_X, -0.05, g.TABLE), (g.END_BOARD_X + 0.012, 0.05, g.AXIS_Z), WOOD))
    return [((tuple(float(v) for v in np.asarray(b[0]) - CRADLE_ORIGIN)), *b[1:]) for b in out]


CRADLE = part("cradle", pos=tuple(CRADLE_ORIGIN), boxes=_cradle_boxes(), mass=None, color=WOOD, friction=0.6)


# -- the regulator (R frame) -------------------------------------------------------------------------------------------
def _regulator():
    iy, hz, wz = g.INLET_Y, g.FRAME_HZ, g.WINDOW_HZ
    boxes = [
        _b((g.PLATE_X[0], iy - g.PLATE_T, -hz), (g.PLATE_X[1], iy, hz), YOKE),                     # inlet plate
        _b((g.BAR_X[0], iy, wz), (g.BAR_X[1], g.BAR_IN_Y, hz), YOKE),                              # top bar
        _b((g.BAR_X[0], iy, -hz), (g.BAR_X[1], g.BAR_IN_Y, -wz), YOKE),                            # bottom bar
        _b((g.BAR_X[0], g.BAR_IN_Y, -hz), (g.BAR_X[1], g.BAR_IN_Y + g.BAR_T, hz), YOKE),          # screw bar
    ]
    cyls = [_c((0, iy + g.WASHER_T / 2, 0), g.WASHER_R, g.WASHER_T, BLACK, "Y")]                # sealing washer
    for pin in g.PINS_R:
        cyls.append(_c((pin[0], iy + 0.00225, pin[2]), g.PIN_D / 2, 0.0045, STEEL, "Y"))
        cyls.append(_c((pin[0], iy + 0.005, pin[2]), 0.0015, 0.001, STEEL, "Y"))                 # chamfered end
    y0, y1 = g.BODY_Y
    boxes.append(_b((-g.BODY_W / 2, y0, -g.BODY_HZ), (g.BODY_W / 2, y1, g.BODY_HZ), BLUE))             # body
    cyls.append(_c((0, y0 - g.GAUGE_T / 2, 0), g.GAUGE_R, g.GAUGE_T, WHITE, "Y"))                # contents gauge
    r, length, dz = g.BARB
    cyls.append(_c((0, y0 - g.GAUGE_T - length / 2, dz), r, length, STEEL, "Y"))                  # hose barb
    return boxes, cyls


def _screw():
    tbar = g.TBAR0_Y - g.TIP0_Y
    cyls = [
        _c((0, 0.001, 0), g.TIP_R, 0.002, STEEL, "Y"),                                           # tip
        _c((0, (0.002 + tbar) / 2, 0), g.SHAFT_R, tbar - 0.002, STEEL, "Y"),                     # shaft
        _c((0, tbar, 0), g.TBAR_R, 2 * g.TBAR_HALF, BLACK, "X"),                                 # T-bar
    ]
    return Screw(anchor=(0.0, g.TIP0_Y, 0.0), cylinders=tuple(cyls), mass=g.SCREW_MASS, pitch=g.PITCH,
                 travel=g.TRAVEL, stiffness=g.TIP_STIFFNESS, force=g.TIP_FORCE,
                 turn_damping=g.TURN_TORQUE0 / g.TURN_SPEED, damping_per_n=g.TORQUE_PER_N / g.TURN_SPEED)


_RB, _RC = _regulator()
REGULATOR = part("regulator", pos=g.REG_START, boxes=_RB, cylinders=_RC, mass=g.REG_MASS, color=BLUE,
                 friction=1.0, screw=_screw())

OBJECTS = [CYLINDER, CRADLE, REGULATOR]

# -- monitor and scoring -------------------------------------------------------------------------------------------------
CYL_TIP = math.radians(30.0)     # rad: the cylinder's axis tipped this far from where it lay: tipped out of its cradle
CYL_DROP = 0.02                  # m: ... or its axis this far below its cradled height: fallen out of the saddles
FALL_SPEED = 0.5                 # m/s downward, with no gripper holding it: dropped
HOLD_R = 0.01                    # m: a gripper holds the regulator when its grasp point is this close to its outline
HOLD_OPEN = 0.07                 # m: ... and its fingers are less than this far apart (fully open: 9.4 cm)
CLEAR = 0.03                     # m: let go means every part of both arms at least this far from the regulator
STILL_MM, STILL_DEG = 0.001, 0.5  # still: over the last second it moved less than 1 mm and 0.5 degrees
STILL_STEPS = 20
ARM_R, PAD_R = 0.03, 0.02        # capsule radii of the arm links and the gripper
PAD_TIP = 0.013                  # fingertips past the tool point (yam_arena.constants.PAD_TIP_PAST_TCP)
CHAIN = ("base", "link1", "link2", "link3", "link4", "link5", "gripper")

_STATE: dict = {}


def _fresh() -> None:
    _STATE.clear()
    _STATE.update(cyl0=None, tipped=False, dropped=False, why=[], fit={}, clear=False,
                  still=False, d=0.0, theta=0.0, hist=[])


_fresh()


def _flag(kind: str, why: str) -> None:
    _STATE[kind] = True
    if why not in _STATE["why"]:
        _STATE["why"].append(why)


def _to_box(points, lo, hi) -> np.ndarray:
    """Distance of each point (rows, in the box's frame) to the box [lo, hi]; 0 inside."""
    p = np.atleast_2d(points)
    return np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0), axis=1)


def _seg(a, b, n: int = 16) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a + np.outer(np.linspace(0.0, 1.0, n), b - a)


def arms_clear(caps, reg_pos, reg_R) -> bool:
    """Every arm capsule (end, end, radius; station frame) at least CLEAR from the regulator's outline."""
    lo, hi = g.REG_BOX
    for a, b, r in caps:
        local = (_seg(a, b) - reg_pos) @ reg_R
        if _to_box(local, lo, hi).min() - r < CLEAR:
            return False
    return True


def update(cyl_pos, cyl_R, reg_pos, reg_R, reg_v, reg_w, d, theta, caps, held: bool = False) -> None:
    """One step of the monitor, on measured values (station frame). ``held``: a gripper holds the regulator."""
    cyl_pos, cyl_R = np.asarray(cyl_pos, float), np.asarray(cyl_R, float)
    reg_pos, reg_R = np.asarray(reg_pos, float), np.asarray(reg_R, float)
    if _STATE["cyl0"] is None:
        _STATE["cyl0"] = (cyl_pos.copy(), cyl_R.copy())
    p0, R0 = _STATE["cyl0"]
    # unsafe: the cylinder tipped or fell out of its cradle (sliding or rolling it in the cradle is allowed)
    tip = math.acos(max(-1.0, min(1.0, float(R0[:, 0] @ cyl_R[:, 0]))))
    if tip > CYL_TIP or cyl_pos[2] < p0[2] - CYL_DROP:
        _flag("tipped", "the cylinder tipped or fell out of its cradle")
    # a real drop: falling with no gripper holding it, or off the table (setting it down is allowed)
    if reg_pos[2] < g.TABLE - 0.05:
        _flag("dropped", "the regulator fell off the table")
    if float(reg_v[2]) < -FALL_SPEED and not held:
        _flag("dropped", "the regulator was dropped")

    p_rc = cyl_R.T @ (reg_pos - cyl_pos)
    R_rc = cyl_R.T @ reg_R
    _STATE["fit"] = g.fit(p_rc, R_rc, d, theta)
    _STATE["clear"] = arms_clear(caps, reg_pos, reg_R)
    hist = _STATE["hist"]
    hist.append((reg_pos.copy(), reg_R.copy()))
    del hist[:-STILL_STEPS]
    _STATE["still"] = bool(len(hist) == STILL_STEPS and float(np.linalg.norm(reg_v)) < 0.05 and all(
        np.linalg.norm(p - reg_pos) < STILL_MM and math.degrees(g.angle_between(R, reg_R)) < STILL_DEG
        for p, R in hist))
    _STATE.update(d=float(d), theta=float(theta), reg_pos=reg_pos.copy(), reg_R=reg_R.copy(), cyl_pos=cyl_pos.copy(),
                  cyl_R=cyl_R.copy())


def _arm_capsules(environment):
    """Both arms as capsules (end, end, radius) along their links and grippers, station frame; and each gripper's
    grasp point and finger opening."""
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS

    base = environment.unwrapped
    origin = base.scene.env_origins[0]
    origin = getattr(origin, "torch", origin).detach().cpu().numpy().astype(float)
    caps, grips = [], []
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        pos = getattr(robot.data.body_pos_w, "torch", robot.data.body_pos_w)[0].detach().cpu().numpy() - origin
        links = {n: pos[i] for i, n in enumerate(robot.body_names)}
        ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
        jp = getattr(robot.data.joint_pos, "torch", robot.data.joint_pos)[0].detach().cpu().numpy()
        tcp, rot = ik.tcp_pose(jp[ids], station.BASES[side])
        chain = [links[n] for n in CHAIN if n in links]
        caps += [(a, b, ARM_R) for a, b in zip(chain[:-1], chain[1:])]
        caps.append((chain[-1], tcp + rot[:, 2] * PAD_TIP, PAD_R))
        fids = robot.find_joints(list(FINGER_JOINTS), preserve_order=True)[0]
        grips.append((tcp, float(np.abs(jp[fids]).sum())))
    return caps, grips


def held_by(grips, reg_pos, reg_R) -> bool:
    """Whether a gripper holds the regulator: its grasp point within HOLD_R of the outline, fingers partly closed."""
    lo, hi = g.REG_BOX
    return any(_to_box((tcp - reg_pos) @ reg_R, lo, hi)[0] < HOLD_R and opening < HOLD_OPEN for tcp, opening in grips)


@on_reset
def _reset(environment=None) -> None:
    _fresh()
    if environment is not None:
        reset_joints(environment, "regulator")


@on_step
def _monitor(environment) -> None:
    from yam_arena.scene import object_states

    d, theta = couple_screw(environment, "regulator")
    st = object_states(environment, ["cylinder", "regulator"])
    c, r = st["cylinder"], st["regulator"]
    caps, grips = _arm_capsules(environment)
    update(c.pos, c.rotation(), r.pos, r.rotation(), r.lin_vel, r.ang_vel, d, theta, caps,
           held=held_by(grips, r.pos, r.rotation()))


def violation() -> bool:
    return bool(_STATE["tipped"] or _STATE["dropped"])


def success(objects) -> bool:
    """Seated on both pins, the T-screw hand-tight in the recess (at least 0.75 N m to turn further), every part
    of both arms 3 cm clear of the regulator, the regulator still (it moved under 1 mm and 0.5 degrees over the
    last second), and no violation so far; held for 1 s. The fit is judged relative to the cylinder, wherever it
    lies."""
    return bool(not violation() and _STATE["fit"].get("clamped") and _STATE["clear"] and _STATE["still"])


def score(objects) -> float:
    """At the end of the trial (success scores 1.0): 0 after any violation (the cylinder's axis tipped more than 30
    degrees, or dropped more than 2 cm below its cradled height, i.e. it fell out of the cradle; the regulator falling
    faster than 0.5 m/s with no gripper holding it, or falling off the table). Setting the regulator down, or sliding
    or rolling the cylinder in its cradle, is not a violation. Otherwise, judged in the cylinder's frame: 0.75 seated
    and hand-tight (still held or touched), 0.5 seated on both pins with the washer flat on the outlet, 0.25 the yoke
    over the valve post, else 0."""
    if violation():
        return 0.0
    f = _STATE["fit"]
    if f.get("clamped"):
        return 0.75
    if f.get("seated"):
        return 0.5
    if f.get("on_post"):
        return 0.25
    return 0.0
