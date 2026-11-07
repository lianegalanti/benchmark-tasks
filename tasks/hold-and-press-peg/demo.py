"""Scripted full-score demo for the hold-and-press-peg task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record] [--variant one_arm]

1. The right arm grips the bar near its far end, fingers closing across it, and holds it there.
2. Meanwhile the left arm picks the pin up by its outer 2 cm, carries it behind the mouth on the hole's axis and,
   once the bar is held, pushes it 4 cm in, closed loop on the measured pin and bar poses (it keeps the pin on the
   hole's axis and backs off when it stalls). Then it lets go: the pin's centre of mass is inside the hole, so it
   lies there.
3. The left arm closes its gripper, comes down behind the pin and pushes its end with the fingertips until it is
   flush with the bar's end, closed loop on the measured depth. Then both arms withdraw.

``--variant one_arm`` skips step 1 (the right arm stays at its ready pose), to show the bar being shoved.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.

The runner (Arm, together, run) is adapted from the accepted task ``lock-before-loading``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from yam_arena.constants import PAD_TIP_PAST_TCP

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
DEBUG = bool(os.environ.get("DEMO_DEBUG"))
PAD_HALF_W = 0.020      # the pads reach 2 cm either side of the tool point across their width
LEFT_YAW = -math.pi / 2  # both grippers turned so their fingers close along x, the pads' width along y
RIGHT_YAW = math.pi / 2
GRIP_LEN = 0.020        # the left pads cover the pin's outer 2 cm
INSERT_TO = 0.040       # push the held pin this deep (its centre of mass, at 3.5 cm, is then inside)
STANDOFF = 0.015        # pin tip (or fingertips) this far behind its target before moving in
HOLD_Y = 0.115          # the right tool point this far along the bar from the mouth: over the solid end, 2 cm from
                        # the bar's centre of mass (9.4 cm), so a pad touching first barely turns it; nearer, the arms
                        # come within 8.6 cm
PUSH_LEAN = math.radians(50)  # the pushing fingers lean this far forward, tips first
PUSH_TIP_UP = 0.004     # the fingertips' centre this far above the hole's floor while pushing
PUSH_LEAD = 0.003       # the fingertips' front at most this far ahead of the pin's measured end (more if it sticks)
TIP_FRONT = 0.0038      # leaning so, the fingertips reach this far ahead of their centre (finger collision meshes)
HOLD_GAIN = 0.2         # per control step: how much of the bar's measured drift the holder corrects


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.worst_mm = 0.0
        self.worst_deg = 0.0
        self.q_low = self.q.copy()
        self.q_high = self.q.copy()

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=80)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(rerr))
        self.q_low, self.q_high = np.minimum(self.q_low, q), np.maximum(self.q_high, q)

    def move(self, pos, rot=None, seconds: float = 1.0, grip: float | None = None):
        """Generator: a smooth straight-line move to ``pos``/``rot``; ``grip`` 1 open .. 0 closed."""
        from yam_arena import ik

        rot = self.rot if rot is None else rot
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.aim(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, seconds: float = 0.8):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*gens):
    """Run generators side by side, one step each per control step."""
    gens = [iter(g) for g in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield


class Ctx:
    """The demo's view of the simulation: measured bar and pin poses after the last control step."""

    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")
        self.flags = {"held": False, "done": False}

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, ["bar", "peg"])

    def bar(self):
        """The mouth's centre and the bar's rotation (station frame)."""
        return self.task.bar_frame(self.states()["bar"])

    def peg(self):
        """The pin's centre and its axis (a unit vector, either way)."""
        p = self.states()["peg"]
        return np.asarray(p.pos, float), self.task._rot(p.quat_xyzw) @ np.array([0.0, 1.0, 0.0])

    def fingers(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """The measured origins of an arm's two finger links (station frame)."""
        from scenekit import robot_links

        links = robot_links(self.emb._env)[side]
        return links["tip_left"], links["tip_right"]

    def wrist(self, side: str) -> np.ndarray:
        """The gripper's measured rotation (forward kinematics of the measured arm joints)."""
        from scenekit import arm_joints
        from yam_arena import ik
        from yam_arena.constants import ARM_JOINTS

        arm = getattr(self, side)
        return ik.tcp_pose(arm_joints(self.emb._env, side, ARM_JOINTS), arm.base)[1]

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def hole(ctx):
    """The mouth's centre, the direction into the hole, and the bar's rotation."""
    mouth, rot = ctx.bar()
    return mouth, rot @ np.array([0.0, -1.0, 0.0]), rot


def peg_ends(ctx, mouth, into):
    """The pin's outer and leading ends' centres, by depth along ``into``."""
    c, a = ctx.peg()
    e1, e2 = c + a * 0.035, c - a * 0.035
    return (e1, e2) if np.dot(e1 - mouth, into) < np.dot(e2 - mouth, into) else (e2, e1)


def lateral(p, mouth, into) -> np.ndarray:
    """A point's offset from the hole's axis, across it."""
    v = p - mouth
    return v - np.dot(v, into) * into


def rotation_between(a, b) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return np.eye(3)
    k = v / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def yaw_matrix(a: float) -> np.ndarray:
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def bar_yaw(rot) -> float:
    return math.atan2(rot[1, 0], rot[0, 0])


def centre_fingers(ctx, arm, point, across, grip, rounds: int = 3):
    """Open fingers at ``grip``: shift and turn the arm until its fingers, as measured, close along ``across``
    through ``point``, so closing neither shoves nor turns a loose part."""
    yield from arm.set_grip(grip, 0.6)
    for _ in range(rounds):
        yield from wait(0.4)
        a, b = ctx.fingers(arm.side)
        err = float(np.dot((a + b) / 2 - point, across))
        v = ctx.wrist(arm.side) @ np.array([1.0, 0.0, 0.0])  # the fingers close along the gripper's x
        turn = math.atan2(across[1], across[0]) - math.atan2(v[1], v[0])
        turn = (turn + math.pi / 2) % math.pi - math.pi / 2  # either finger may be on either side
        turn = float(np.clip(turn, -math.radians(5), math.radians(5)))  # a wrist that lags, not a new grasp
        if DEBUG:
            print(f"{arm.side} fingers {err * 1000:+.1f} mm off centre, {math.degrees(turn):+.2f} deg off square",
                  flush=True)
        if abs(err) < 0.0005 and abs(turn) < math.radians(0.3):
            break
        yield from arm.move(arm.pos - across * err, yaw_matrix(turn) @ arm.rot, 0.4)


def hold_bar(ctx, g):
    """Right arm: grip the bar over its solid end and hold until the pin is home, then let go and withdraw."""
    from yam_arena.demo import grip_rotation

    R, rot = ctx.right, grip_rotation(RIGHT_YAW)
    start = hole(ctx)  # where the bar lay before anything touched it
    mouth, into, _ = start
    at = mouth + into * HOLD_Y
    z = g.SHEET_TOP + 0.008 + 0.013  # pad tips 8 mm above the sheet
    yield from R.move((at[0], at[1], g.TOP + 0.10), rot, 2.5, grip=1.0)
    yield from R.move((at[0], at[1], z), rot, 1.5)
    # centre the open fingers on the bar (as measured), then close: the first pad to touch would shove the bar
    across = hole(ctx)[2] @ np.array([1.0, 0.0, 0.0])
    point = np.array([at[0], at[1], z])
    yield from centre_fingers(ctx, R, point, across, 0.6)
    yield from centre_fingers(ctx, R, point, across, 0.42)

    # close, then hold: closed loop on the bar's measured pose, the arm moves and turns to cancel any drift of the
    # bar (the first pad to touch shoves it; the pin's friction drags it; the arm gives under a push)
    def drift():
        m, _, r = hole(ctx)
        return ((m + r @ np.array([0.0, -HOLD_Y, 0.0]) - home) * np.array([1.0, 1.0, 0.0]),
                (bar_yaw(r) - bar_yaw(r0) + math.pi) % (2 * math.pi) - math.pi)

    m0, _, r0 = start
    home = m0 + r0 @ np.array([0.0, -HOLD_Y, 0.0])
    base, rot, corr, turn, n = R.pos.copy(), R.rot.copy(), np.zeros(3), 0.0, round(1.5 * CONTROL_HZ)
    i = 0
    while not ctx.flags["done"]:
        i += 1
        R.grip = max(0.0, 0.42 * (1 - i / n))
        moved, yawed = drift()
        corr = np.clip(corr - HOLD_GAIN * moved, -0.015, 0.015)
        turn = float(np.clip(turn - HOLD_GAIN * yawed, -0.1, 0.1))
        R.aim(base + corr, yaw_matrix(turn) @ rot)
        if i == n + 6:
            if DEBUG:
                print(f"bar gripped: bar_move {ctx.task.max_move() * 1000:.1f} mm", flush=True)
            ctx.flags["held"] = True
        if DEBUG and i % 40 == 0:
            print(f"hold drift {np.round(moved * 1000, 1)} mm {math.degrees(yawed):+.2f} deg, "
                  f"corr {np.round(corr * 1000, 1)} mm {math.degrees(turn):+.2f} deg", flush=True)
        yield
    yield from R.set_grip(1.0, 0.8)
    yield from R.move(R.pos + [0, 0, 0.08], R.rot, 1.2)


def press_pin(ctx, g, one_arm: bool = False):
    """Left arm: take the pin, push it 4 cm in while holding it, let go, and push it flush with the fingertips."""
    from yam_arena.demo import grip_rotation

    L, rot = ctx.left, grip_rotation(LEFT_YAW)
    travel = g.TOP + 0.10

    # pick: the pads over the pin's outer 2 cm (the end toward the left arm), tips 3 mm above the table
    def pick_point():  # measured now, so a pin that has shifted on the table is still picked where it lies
        c, a = ctx.peg()
        outer = c + a * (g.PEG_LEN / 2) * (1.0 if a[1] >= 0 else -1.0)
        return np.array([outer[0], outer[1] - GRIP_LEN + PAD_HALF_W, g.TOP + 0.003 + 0.013])

    pick = pick_point()
    yield from L.move((pick[0], pick[1], travel), rot, 2.5, grip=1.0)
    pick = pick_point()
    yield from L.move((pick[0], pick[1], g.TOP + 0.05), rot, 0.8)
    pick = pick_point()
    if DEBUG:
        print(f"pick at {np.round(pick * 1000, 1)} mm, pin centre {np.round(ctx.peg()[0] * 1000, 1)} mm", flush=True)
    yield from L.move(pick, rot, 1.0)
    yield from centre_fingers(ctx, L, pick_point(), np.array([1.0, 0.0, 0.0]), 0.35, rounds=2)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.3)
    yield from L.move(L.pos + [0, 0, travel - L.pos[2]], rot, 1.2)

    # line up behind the mouth: the measured grip offset puts the pin's axis on the hole's axis
    mouth, into, _ = hole(ctx)
    outer, lead = peg_ends(ctx, mouth, into)
    offset = L.pos - outer  # tool point relative to the pin's outer end, as held
    if DEBUG:
        print(f"held: tool point to outer end {np.round(offset * 1000, 1)} mm, "
              f"bar_move {ctx.task.max_move() * 1000:.1f} mm", flush=True)
    def tcp_for(depth):  # tool point that puts the leading end ``depth`` inside the mouth
        return mouth + into * (depth - g.PEG_LEN) + offset

    start = tcp_for(-STANDOFF)
    yield from L.move((start[0], start[1], travel), rot, 2.0)
    yield from L.move(start, rot, 1.5)

    # the pin droops in a grip on its end: turn the gripper until the pin's axis is on the hole's (measured)
    for _ in range(4):
        yield from wait(0.4)
        mouth, into, _ = hole(ctx)
        outer, lead = peg_ends(ctx, mouth, into)
        turn = rotation_between(lead - outer, into)
        tilt = math.degrees(math.acos(min(1.0, (np.trace(turn) - 1) / 2)))
        offset = L.pos - outer
        if DEBUG:
            print(f"pin {tilt:.1f} deg off the hole's axis; tool point to outer end {np.round(offset * 1000, 1)} mm",
                  flush=True)
        if tilt < 0.4:
            break
        rot = turn @ L.rot
        yield from L.move(tcp_for(-STANDOFF) - (offset - turn @ offset), rot, 0.8)
    yield from wait(0.3)
    outer, lead = peg_ends(ctx, mouth, into)
    offset = L.pos - outer
    while not (ctx.flags["held"] or one_arm):
        yield

    # closed loop: advance at 1.5 cm/s while lined up; correct the pin across the axis; back off when it stalls
    target, corr, stalled = -STANDOFF, np.zeros(3), 0
    for i in range(int(20 * CONTROL_HZ)):
        mouth, into, _ = hole(ctx)
        outer, lead = peg_ends(ctx, mouth, into)
        depth = float(np.dot(lead - mouth, into))
        # before entry, steer the leading end; once in, steer the pin where it crosses the mouth
        if depth < 0.003:
            ref = lead
        else:
            s = depth / g.PEG_LEN
            ref = lead + (outer - lead) * s
        err = lateral(ref, mouth, into)
        corr -= 2.0 * DT * err
        corr = np.clip(corr, -0.010, 0.010)
        lined_up = np.abs(err).max() < (0.0012 if i < 3 * CONTROL_HZ else 0.002)
        if depth > target - 0.004:
            stalled = 0
            if depth > 0.003 or lined_up:
                target = min(target + 0.015 * DT, INSERT_TO)
        else:
            stalled += 1
            if stalled > CONTROL_HZ:
                target, stalled = target - 0.003, 0
        L.aim(tcp_for(target) + corr, rot)
        if DEBUG and i % 10 == 0:
            print(f"insert depth {depth * 1000:.1f} mm target {target * 1000:.1f} err_mm {np.round(err * 1000, 2)} "
                  f"bar_move {ctx.task.max_move() * 1000:.1f} mm", flush=True)
        if depth >= INSERT_TO - 0.002:
            break
        yield

    # let go: the fingers open across the pin, 1 cm clear of the bar's end
    yield from wait(0.3)
    yield from L.set_grip(1.0, 0.8)
    yield from wait(0.3)
    rot = grip_rotation(LEFT_YAW)  # level again for the fingertip push
    yield from L.move(L.pos + [0, 0, 0.05], rot, 1.0)

    # the closed fingertips, leaning 50 deg forward, come down behind the pin's outer end. The finger widens above
    # its tip (from 1.5 cm across at the tip to 2.6 cm 13 mm above it), so a vertical finger would hit the bar's end
    # face above the hole before the pin is flush; leaning, the tip leads and the rest trails behind (test_task.py)
    mouth, into, _ = hole(ctx)
    outer, _ = peg_ends(ctx, mouth, into)
    down = np.array([0.0, 0.0, -1.0])
    lean = rotation_between(down, down * math.cos(PUSH_LEAN) + into * math.sin(PUSH_LEAN))
    rot = lean @ grip_rotation(LEFT_YAW)
    approach = rot @ np.array([0.0, 0.0, 1.0])
    tip_z = g.AXIS_Z - g.HOLE / 2 + PUSH_TIP_UP

    def tcp_for_tip(tip):  # tool point that puts the centre of the fingertips at ``tip``
        return np.asarray(tip, float) - approach * PAD_TIP_PAST_TCP

    def tip_behind(gap):  # the fingertips' front ``gap`` behind the pin's outer end
        p = outer - into * (gap + TIP_FRONT)
        return np.array([p[0], p[1], tip_z])

    behind = tcp_for_tip(tip_behind(STANDOFF))
    yield from L.move((behind[0], behind[1], L.pos[2]), rot, 1.2, grip=0.0)
    yield from L.move(behind, rot, 1.2)

    # closed loop: advance at 1 cm/s until the outer end is within 0.5 mm of flush. The pin bottoms in the hole when
    # it is flush, so overshoot can't push it past the face; give up 2 cm past where the fingertips should be.
    # The fingertips never get more than ``lead`` ahead of the measured outer end, which bounds the push: a hard
    # push would tip the pin and wedge it, and shove the holding arm (it gives about 1 mm per newton).
    travelled, base = 0.0, tip_behind(STANDOFF)
    limit = float(np.dot(mouth - outer, into)) + STANDOFF + 0.020
    lead, best, since = PUSH_LEAD, np.inf, 0
    for i in range(int(20 * CONTROL_HZ)):
        mouth, into, _ = hole(ctx)
        outer, _ = peg_ends(ctx, mouth, into)
        proud = -float(np.dot(outer - mouth, into))
        if proud <= 0.0005 or travelled >= limit:
            break
        if proud < best - 0.0005:
            best, since = proud, 0
        else:
            since += 1
            if since > CONTROL_HZ:  # no progress for 1 s: push a little harder
                lead, since = min(lead + 0.001, 0.008), 0
        ahead = float(np.dot(base - mouth, into)) + TIP_FRONT + travelled + proud  # fingertips' front past the end
        travelled = min(travelled + 0.010 * DT, limit, travelled + lead - ahead)
        tip = base + into * travelled
        tip = tip - (lateral(tip, mouth, into) - lateral(outer, mouth, into)) * np.array([1.0, 1.0, 0.0])
        L.aim(tcp_for_tip((tip[0], tip[1], tip_z)), rot)
        if DEBUG and i % 10 == 0:
            print(f"push proud {proud * 1000:.1f} mm travelled {travelled * 1000:.1f} lead {lead * 1000:.0f} "
                  f"bar_move {ctx.task.max_move() * 1000:.1f} mm", flush=True)
        yield
    yield from wait(0.5)
    yield from L.move(L.pos - into * 0.03, rot, 1.0)
    yield from L.move(L.pos + [0, 0, 0.06], rot, 1.0)
    ctx.flags["done"] = True


def script(ctx, g, variant: str = ""):
    if variant == "one_arm":
        ctx.flags["held"] = False
        yield from press_pin(ctx, g, one_arm=True)
    else:
        yield from together(hold_bar(ctx, g), press_pin(ctx, g))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, variant: str = "",
        cam_width: int = 320, cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    import geometry as g

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb, task)
    gen = script(ctx, g, variant)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info, best = 0, None, False, {}, 0.0
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        best = max(best, float(info.get("score", 0.0) or 0.0))
        snap(res.observation)
        if DEBUG and steps % 20 == 0:
            st = ctx.states()
            print(f"  t {steps * DT:.1f}s pin {np.round(np.asarray(st['peg'].pos) * 1000, 1)} "
                  f"bar {np.round(np.asarray(st['bar'].pos) * 1000, 1)} bar_move {task.max_move() * 1000:.1f} mm",
                  flush=True)
        if steps % 100 == 0:
            print(f"step {steps} score {info.get('score')} bar_move {task.max_move() * 1000:.1f} mm", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    result = {"task_score": 1.0 if success else best, "success": success, "steps": steps,
              "bar_max_move_mm": round(task.max_move() * 1000, 1),
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="", choices=["", "one_arm"])
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, variant=a.variant, cam_width=a.cam_width,
        cam_height=a.cam_height)
