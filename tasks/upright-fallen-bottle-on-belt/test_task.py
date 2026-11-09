"""Offline checks (no Isaac Sim): the line's motion and layout, every scoring level and the penalty, the decision at
the guide's mouth, and the scripted demo played against a rigid-grip model of the bottle: reachable, within joint
limits, on time, with every arm link (as the URDF's collision hulls) clear of the table, the line, the guide and the
upright bottles, and scoring 1.0 at the guide.

    PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

import demo
from yam_arena import analysis, ik, station
from yam_arena.constants import ARM_JOINTS, FINGER_OPEN, READY
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).resolve().parent / "task.py")
M = TASK.module
DT = demo.DT
LEFT = np.asarray(station.BASES["left"], float)
FAR = [np.array([0.0, 0.3, 1.2]), np.array([0.0, -0.3, 1.2])]  # grasp points well away from everything
F = M.FALLEN
A, B = M.UPRIGHT
UP = np.eye(3)
LYING = M.rot(M.LYING_XYZW)


def bottle_state(pos, R=UP, vel=(0.0, 0.0, 0.0)):
    pos, R = np.asarray(pos, float), np.asarray(R, float)
    ext = np.abs(R[:, 2]) * M.H / 2 + M.R * np.sqrt(np.clip(1 - R[:, 2] ** 2, 0.0, 1.0))
    return ObjectState(pos, ik.matrix_to_quat_xyzw(R), np.asarray(vel, float), np.zeros(3), pos - ext, pos + ext)


def carrier_state(name, t):
    p, v = M.PATHS[name](t)
    half = np.asarray(M.CARRIER) / 2
    return ObjectState(p, np.array([0.0, 0.0, 0.0, 1.0]), v, np.zeros(3), p - half, p + half)


def top(name, t):
    """The top centre of bottle ``name``'s carrier at time ``t``."""
    return np.array([M.LINE_X, M.carrier_y(name, t), M.CARRIER_TOP])


def upright_on(name, t, dx=0.0, dy=0.0, R=UP):
    return bottle_state(top(name, t) + [dx, dy, M.H / 2], R, M.PATHS[name](t)[1])


def lying_on(name, t):
    return bottle_state(top(name, t) + [0.0, 0.0, M.R], LYING, M.PATHS[name](t)[1])


def scene(t, **override):
    """Every object at time ``t``: the upright bottles on their carriers, the fallen one lying on its carrier, unless
    overridden."""
    out = {n: (lying_on(n, t) if n == F else upright_on(n, t)) for n in M.NAMES}
    out.update({M.CARRIER_OF[n]: carrier_state(n, t) for n in M.NAMES})
    out.update(override)
    return out


def tilt(deg, axis=(1.0, 0.0, 0.0)):
    from yam_arena.urdf import axis_angle_matrix

    return axis_angle_matrix(np.asarray(axis, float), math.radians(deg))


T_DEC = M.arrival(F, M.GUIDE_Y)


def play(frames, tcps=FAR):
    """Run the monitor over (t, objects) frames from a reset; return (score, success) after the last."""
    M.reset_state()
    for _, objects in frames:
        M.update(objects, tcps)
    return M.score(frames[-1][1]), M.success(frames[-1][1])


def decide(objects_at, tcps=FAR):
    """Frames every 0.5 s up to just past the decision, objects from ``objects_at(t)``."""
    ts = list(np.arange(0.0, T_DEC + 0.3, 0.5)) + [T_DEC + 0.06]
    return play([(t, objects_at(t)) for t in ts], tcps)


# -- layout and motion -----------------------------------------------------------------------------------------------


def test_it_loads_with_only_the_bottles_free_and_plain_floats():
    assert TASK.free_objects == M.NAMES
    assert TASK.score_over == "end"
    for o in M.OBJECTS:
        assert all(type(v) is float for v in o.pos), o.name
        if hasattr(o, "rot_xyzw"):
            assert all(type(v) is float for v in o.rot_xyzw), o.name


def test_the_fallen_bottle_lies_along_the_line_cap_downstream():
    R = M.rot(M.LYING_XYZW)
    assert R[:, 2] == pytest.approx([0.0, -1.0, 0.0], abs=1e-9)  # the cap (+z of the bottle) points -y
    # lying along the line, the belt's acceleration (along y) can't roll it
    assert M.tilt_deg(lying_on(F, 0.0)) == pytest.approx(90.0)


def test_the_line_runs_at_2_cm_per_s_toward_the_right_and_gently():
    ts = np.arange(0.0, 40.0, 1 / 240)
    for n in M.NAMES:
        path = M.PATHS[n]
        assert np.linalg.norm(path(0.0)[1]) == 0.0
        assert path(3.0)[1][1] == pytest.approx(-M.SPEED)
        assert M.carrier_y(n, M.arrival(n, -0.15)) == pytest.approx(-0.15)
        vs = np.array([path(t)[1] for t in ts])
        on = ts < M.arrival(n, M.TURN_Y) - 0.05
        assert np.max(np.abs(np.gradient(vs[on, 1], ts[on])[5:-5])) < 0.1  # m/s^2: nothing slides or tips
    # an upright bottle tips only above g * R / (H / 2) = 4.5 m/s^2
    assert 9.81 * M.R / (M.H / 2) > 40 * 0.1


def test_carriers_start_on_the_bed_apart_and_turn_down_off_the_table():
    ys = sorted(M.STARTS.values())
    assert ys[0] - M.CARRIER[1] / 2 > -M.BED[1] / 2 and ys[-1] + M.CARRIER[1] / 2 <= M.BED[1] / 2 + 1e-9
    assert min(np.diff(ys)) - M.CARRIER[1] > 0.09
    assert M.TURN_Y + M.CARRIER[1] / 2 < M.TABLE_RIGHT_Y
    end = M.PATHS[F](80.0)[0]
    assert end[1] == pytest.approx(M.TURN_Y) and end[2] == pytest.approx(M.CARRIER_Z - M.DROP)


def test_the_guide_passes_an_upright_bottle_and_clears_the_carriers():
    assert M.GUIDE_GAP / 2 - M.R >= 0.015
    rail_bottom = M.RAIL_Z - M.RAIL[2] / 2
    assert rail_bottom - M.CARRIER_TOP >= 0.025
    for o in M.GUIDE:  # posts and brackets stand beside the bed, or above the carriers
        low_x, high_x = o.pos[0] - o.size[0] / 2, o.pos[0] + o.size[0] / 2
        beside = high_x < M.LINE_X - M.CARRIER[0] / 2 or low_x > M.LINE_X + M.CARRIER[0] / 2
        above = o.pos[2] - o.size[2] / 2 > M.CARRIER_TOP + 0.02
        assert beside or above, o.name
    # every bottle starts before the guide's mouth
    assert all(y - M.CARRIER[1] / 2 >= M.GUIDE_Y - 1e-9 for y in M.STARTS.values())


def test_the_deadline_leaves_time_but_falls_within_the_run():
    assert T_DEC == pytest.approx(26.75)
    assert T_DEC + 2.0 < 35.0  # run limit 700 steps = 35 s


# -- scoring ---------------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_zero():
    assert decide(lambda t: scene(t)) == (0.0, False)
    assert M._STATE["decided"] == 0.0 and M._STATE["disturbed"] == {}


def test_upright_on_its_carrier_and_let_go_scores_one():
    assert decide(lambda t: scene(t, **{F: upright_on(F, t, 0.012, -0.022, tilt(8))})) == (1.0, True)


def test_upright_on_its_carrier_but_held_scores_half():
    def held(t):
        return scene(t, **{F: upright_on(F, t)})

    M.reset_state()
    for t in list(np.arange(0.0, T_DEC + 0.3, 0.5)):
        M.update(held(t), [top(F, t) + [0.0, 0.01, M.H / 2 + 0.015], FAR[1]])
    assert (M.score({}), M.success({})) == (0.5, False)


def test_off_centre_tilted_or_upside_down_on_the_carrier_is_not_full():
    assert decide(lambda t: scene(t, **{F: upright_on(F, t, 0.02, 0.0)})) == (0.5, False)  # 2 cm across the line
    assert decide(lambda t: scene(t, **{F: upright_on(F, t, 0.0, 0.03)})) == (0.5, False)  # 3 cm along it
    assert decide(lambda t: scene(t, **{F: upright_on(F, t, 0.0, 0.06)}))[0] == 0.0  # centre past the carrier's edge
    assert decide(lambda t: scene(t, **{F: upright_on(F, t, R=tilt(15))}))[0] == 0.0
    assert decide(lambda t: scene(t, **{F: upright_on(F, t, R=tilt(180))}))[0] == 0.0  # cap down
    floating = lambda t: scene(t, **{F: bottle_state(top(F, t) + [0, 0, M.H / 2 + 0.015])})  # noqa: E731
    assert decide(floating)[0] == 0.0  # held 1.5 cm over it: not on the carrier, not lifted clear


def test_taking_the_bottle_off_the_line_scores_a_quarter_and_standing_it_on_the_table_half():
    on_table_lying = bottle_state((0.20, 0.25, M.TOP + M.R), LYING)
    assert decide(lambda t: scene(t, **{F: on_table_lying})) == (0.25, False)
    on_table_upright = bottle_state((0.20, 0.25, M.TOP + M.H / 2))
    assert decide(lambda t: scene(t, **{F: on_table_upright})) == (0.5, False)
    held_high = bottle_state(top(F, T_DEC) + [0.0, 0.0, 0.2], LYING)
    assert decide(lambda t: scene(t, **{F: held_high}))[0] == 0.25
    # standing it back on its carrier beats both
    assert decide(lambda t: scene(t, **{F: upright_on(F, t)}))[0] == 1.0


def test_off_the_table_or_left_lying_on_the_bed_scores_zero():
    assert decide(lambda t: scene(t, **{F: bottle_state((0.3, -0.8, 0.3), LYING)}))[0] == 0.0
    on_bed = bottle_state((M.LINE_X, 0.25, M.TOP + M.BED[2] + M.R), LYING)
    assert decide(lambda t: scene(t, **{F: on_bed}))[0] == 0.0


def test_each_disturbed_upright_bottle_costs_half_and_never_goes_below_zero():
    knocked = bottle_state((M.LINE_X + 0.1, 0.0, M.TOP + M.R), LYING)

    def one(t):  # bottle_b knocked over at 6 s; the fallen one stood up properly
        return scene(t, **{F: upright_on(F, t)}, **({B: knocked} if t >= 6.0 else {}))

    assert decide(one) == (0.5, False)
    assert M._STATE["disturbed"] == {B: "tilted"}

    def two(t):  # also bottle_a nudged 3.5 cm along its carrier for a moment at 1 s
        o = one(t)
        if 1.0 <= t < 1.6:  # before it reaches the guide at 2.75 s
            o[A] = upright_on(A, t, dy=0.035)
        return o

    assert decide(two) == (0.0, False)

    def lifted(t):  # bottle_b picked up 3 cm and put back
        o = scene(t, **{F: upright_on(F, t)})
        if 4.0 <= t < 4.6:
            o[B] = bottle_state(top(B, t) + [0, 0, M.H / 2 + 0.03])
        return o

    assert decide(lifted)[0] == 0.5
    # removing the fallen bottle while knocking one over: nothing
    assert decide(lambda t: scene(t, **{F: bottle_state((0.2, 0.25, M.TOP + M.R), LYING)},
                                  **({B: knocked} if t >= 6 else {})))[0] == 0.0


def test_upright_bottles_are_judged_only_until_their_carrier_reaches_the_guide():
    t_a = M.arrival(A, M.GUIDE_Y)

    def bumped_in_guide(t):  # bottle_a tips after entering the guide: not the robot's doing, not counted
        return scene(t, **{F: upright_on(F, t)}, **({A: upright_on(A, t, R=tilt(30))} if t > t_a + 0.6 else {}))

    assert decide(bumped_in_guide) == (1.0, True)


def test_the_decision_is_kept_after_the_guide():
    frames = [(t, scene(t, **{F: upright_on(F, t)})) for t in np.arange(0.0, T_DEC + 0.3, 0.5)]
    frames.append((T_DEC + 2, scene(T_DEC + 2)))  # later lying again (in the guide): no change
    assert play(frames) == (1.0, True)
    # before the decision, the score is what the state is worth now
    assert play([(5.0, scene(5.0, **{F: upright_on(F, 5.0)}))]) == (1.0, False)


def test_a_gripper_beside_the_bottle_means_not_let_go():
    M.reset_state()
    objs = scene(T_DEC + 0.06, **{F: upright_on(F, T_DEC + 0.06)})
    M.update(objs, [objs[F].pos + [0.0, 0.07, 0.0], FAR[1]])
    assert M.score(objs) == 0.5


# -- the demo, against a rigid-grip bottle ---------------------------------------------------------------------------


class FakeCtx(demo.Ctx):
    """The fallen bottle rides on its carrier until the left fingers close on it, then follows the grip rigidly; once
    they open it drops onto the carrier straight below and rides on with it."""

    def __init__(self):
        self.emb, self.task = None, M
        self.left, self.right = demo.Arm("left"), demo.Arm("right")
        self.k = 0
        self.corr = np.zeros(3)
        self.attached = self.released = False
        self.rel, self.R = np.array([0.0, 0.0, M.R]), LYING.copy()  # the bottle's centre relative to the carrier
        self.offset = None
        self.pos = top(F, 0.0) + self.rel

    def fallen(self):
        return self.pos.copy(), self.R.copy()

    def update(self, t):
        """After the step that ends at ``t``."""
        L = self.left
        if not self.attached and not self.released and L.grip < 0.05:
            self.attached = True
            self.offset = (L.rot.T @ (self.pos - L.pos), L.rot.T @ self.R)
        elif self.attached and L.grip > 0.5:
            self.attached, self.released = False, True
            self.rel = self.pos - top(F, t)
            self.rel[2] = M.H / 2 * self.R[2, 2] + M.R * math.sqrt(max(0.0, 1 - self.R[2, 2] ** 2))
        if self.attached:
            self.pos, self.R = L.pos + L.rot @ self.offset[0], L.rot @ self.offset[1]
        else:
            self.pos = top(F, t) + self.rel


@pytest.fixture(scope="module")
def played():
    ctx = FakeCtx()
    log = []
    gen = demo.stand_up(ctx)
    for k in range(700):
        ctx.k = k
        try:
            next(gen)
        except StopIteration:
            break
        t = (k + 1) * DT
        ctx.update(t)
        L = ctx.left
        log.append(dict(t=t, q=L.q.copy(), grip=L.grip, tcp=L.pos.copy(), rot=L.rot.copy(), pos=ctx.pos.copy(),
                        R=ctx.R.copy(), attached=ctx.attached, released=ctx.released))
    return ctx, log


def test_the_demo_is_reachable_within_limits_and_well_before_the_guide(played):
    ctx, log = played
    assert ctx.left.worst_mm < 0.5 and ctx.left.worst_deg < 0.1, (ctx.left.worst_mm, ctx.left.worst_deg)
    qs = np.array([e["q"] for e in log])
    assert np.all(qs >= ik.LOWER - 1e-9) and np.all(qs <= ik.UPPER + 1e-9)
    released = next(e["t"] for e in log if e["released"])
    assert released < T_DEC - 6.0  # let go 6 s or more before the carrier reaches the guide
    assert log[-1]["t"] < T_DEC - 3.0  # parked before then too
    steps = np.abs(np.diff(qs, axis=0)).max(axis=0) / DT
    assert np.all(steps < 2.0), steps  # rad/s: well within the joints' speed


def test_the_demo_grips_the_lying_bottle_where_planned(played):
    ctx, log = played
    k = next(i for i, e in enumerate(log) if e["attached"])
    e = log[k]
    centre_to_tcp = e["tcp"] - e["pos"]
    assert centre_to_tcp == pytest.approx([0.0, -demo.GRIP_ALONG, demo.GRIP_ABOVE], abs=2e-3)
    # the bottle hadn't moved off its carrier before the grip
    assert e["pos"] - top(F, e["t"]) == pytest.approx([0.0, 0.0, M.R], abs=2e-3)
    # the open pads clear the bottle's sides: 9.4 cm apart around a 5.7 cm chord
    chord = 2 * math.sqrt(M.R ** 2 - demo.GRIP_ABOVE ** 2)
    assert (-FINGER_OPEN * 2 - chord) / 2 >= 0.015


def test_the_bottle_is_turned_cap_up_and_set_on_its_carrier(played):
    ctx, log = played
    e = next(e for e in log if e["released"])
    assert e["R"][2, 2] > math.cos(math.radians(1.0))  # upright, cap up
    off = e["pos"] - top(F, e["t"])
    assert np.hypot(*off[:2]) < 0.003
    # it was let go with its base 3 mm above the carrier
    last_held = [x for x in log if x["attached"]][-1]
    base = last_held["pos"][2] - M.H / 2 * last_held["R"][2, 2] - M.CARRIER_TOP
    assert base == pytest.approx(demo.DROP, abs=0.002)


def test_the_carried_bottle_clears_the_line_and_its_neighbours(played):
    _, log = played
    lifted = lowering = False
    worst = 1.0
    for e in log:
        if not e["attached"]:
            continue
        R, pos = e["R"], e["pos"]
        a = R[:, 2]
        # its rims' lowest point
        rim = M.R * math.sqrt(max(0.0, 1 - a[2] ** 2))
        low = min(pos[2] - M.H / 2 * abs(a[2]) - rim, pos[2] + M.H / 2 * abs(a[2]) - rim)
        lifted = lifted or low - M.CARRIER_TOP > 0.025  # from when it is first lifted 2.5 cm
        lowering = lowering or (a[2] > 0.99 and low - M.CARRIER_TOP < 0.031)  # until it is lowered onto the carrier
        if lifted and not lowering:
            worst = min(worst, low - M.CARRIER_TOP)
        # never nearer than 4 cm to an upright neighbour (axis to axis, minus both radii)
        for n in M.UPRIGHT:
            d = np.hypot(*(pos[:2] - top(n, e["t"])[:2]))
            assert d - 2 * M.R - M.H / 2 * math.sqrt(max(0.0, 1 - a[2] ** 2)) > 0.04, (n, e["t"])
    assert worst > 0.025, worst


def _hulls(q, grip, holding=False):
    qd = dict(zip(ARM_JOINTS, q))
    finger = FINGER_OPEN * grip
    if holding:  # the fingers stop on the bottle
        finger = min(finger, -math.sqrt(M.R ** 2 - demo.GRIP_ABOVE ** 2))
    qd["joint7"] = qd["joint8"] = finger
    return [h for h in analysis.world_hulls(analysis.fk_link_poses_world(qd, LEFT)) if h[0] != "base"]


def _cylinder_points(pos, R, n=20, rings=9):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    zs = np.linspace(-M.H / 2, M.H / 2, rings)
    pts = [pos + R[:, 2] * z + M.R * (R[:, 0] * math.cos(t) + R[:, 1] * math.sin(t)) for z in zs for t in a]
    pts += [pos + R[:, 2] * z for z in (-M.H / 2, M.H / 2)]
    return np.array(pts)


def _box_points(center, size, n=6):
    c, h = np.asarray(center, float), np.asarray(size, float) / 2
    g = np.linspace(-1, 1, n)
    pts = [c + h * [x, y, z] for x in g for y in g for z in (-1, 1)]
    pts += [c + h * [x, s, z] for x in g for z in g for s in (-1, 1)]
    pts += [c + h * [s, y, z] for y in g for z in g for s in (-1, 1)]
    return np.array(pts)


def _dist(points, hulls):
    return min(analysis._signed(points, h).min() for _, h in hulls) if hulls else 1.0


def test_the_arm_stays_clear_of_the_table_line_guide_and_upright_bottles(played):
    _, log = played
    fixed = np.vstack([_box_points(o.pos, o.size) for o in M.GUIDE])
    worst = dict(table=1.0, line=1.0, guide=1.0, upright=1.0, fallen=1.0, fingers=1.0)
    for i, e in enumerate(log):
        if i % 2:
            continue
        t = e["t"]
        gripping = e["attached"] or e["grip"] < 0.999  # closing on it, holding it, or opening off it
        hulls = _hulls(e["q"], e["grip"], gripping)
        worst["table"] = min(worst["table"], min(h.vertices[:, 2].min() for _, h in hulls) - M.TOP)
        carriers = np.vstack([_box_points(M.PATHS[n](t)[0], M.CARRIER) for n in M.NAMES])
        worst["line"] = min(worst["line"], _dist(carriers, hulls),
                            min(h.vertices[:, 2].min() for _, h in hulls) - (M.TOP + M.BED[2]))
        worst["guide"] = min(worst["guide"], _dist(fixed, hulls))
        pts = np.vstack([_cylinder_points(top(n, t) + [0, 0, M.H / 2], UP) for n in M.UPRIGHT])
        worst["upright"] = min(worst["upright"], _dist(pts, hulls))
        mine = _cylinder_points(e["pos"], e["R"])
        body = [h for h in hulls if h[0] not in ("tip_left", "tip_right")]
        worst["fallen"] = min(worst["fallen"], _dist(mine, body))
        if not gripping:
            tips = [h for h in hulls if h[0] in ("tip_left", "tip_right")]
            worst["fingers"] = min(worst["fingers"], _dist(mine, tips))
    print("margins", {k: round(float(v), 4) for k, v in worst.items()})
    assert worst["table"] > 0.02, worst
    assert worst["line"] > 0.012, worst
    assert worst["guide"] > 0.05, worst
    assert worst["upright"] > 0.02, worst
    assert worst["fallen"] > 0.01, worst
    assert worst["fingers"] > 0.01, worst


def test_the_idle_right_arm_is_clear_of_the_guide_and_the_passing_bottles():
    hulls = [h for h in analysis.world_hulls(analysis.fk_link_poses_world(
        dict(zip(ARM_JOINTS, READY), joint7=FINGER_OPEN, joint8=FINGER_OPEN), station.BASES["right"])) if h[0] != "base"]
    fixed = np.vstack([_box_points(o.pos, o.size) for o in M.GUIDE])
    passing = np.vstack([_cylinder_points((M.LINE_X, y, M.CARRIER_TOP + M.H / 2), UP)
                         for y in np.arange(0.2, -0.7, -0.02)])
    assert _dist(fixed, hulls) > 0.05 and _dist(passing, hulls) > 0.03


def test_the_demo_scores_one_at_the_guide(played):
    ctx, log = played
    right = ik.tcp_pose(np.array(READY, float), station.BASES["right"])[0]
    end = log[-1]
    M.reset_state()
    frames = [(e["t"], e["pos"], e["R"], e["tcp"]) for e in log[::4]]
    t = end["t"]
    while t < T_DEC + 0.2:  # parked; the bottle rides on to the guide
        t += 0.25
        frames.append((t, top(F, t) + ctx.rel, end["R"], end["tcp"]))
    for t, pos, R, tcp in frames:
        objs = scene(t, **{F: bottle_state(pos, R, M.PATHS[F](t)[1])})
        M.update(objs, [tcp, right])
    assert M._STATE["disturbed"] == {}
    assert (M.score({}), M.success({})) == (1.0, True)
