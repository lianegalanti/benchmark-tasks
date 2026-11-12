"""Scripted full-score demo for the fit-oxygen-regulator task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 4800] [--no-record] [--variant no_pins|no_turn]

It reads no object state: it works from the fixed layout and the arms' joint encoders (forward kinematics of the
measured joints).

1. The right arm grips the regulator's flat-sided body from above, over its centre of mass, and lifts it.
2. It carries the yoke 8 cm short of the post's end, then lines it up on the post (pins 2.6 mm clear of the outlet
   face), correcting for the payload's sag from its encoders at each stop, then freezing that correction.
3. It slides the yoke onto the post at 5 mm/s (if its encoders say the hand is held back, the window caught the
   post's end: it backs off and comes again a little higher or lower), then moves it 7.2 mm sideways (0.7 mm past contact) so the two pins
   enter their holes and the washer meets the outlet face. It pushes softly, in steps: when its encoders say the
   hand is held back (pins on the face), it stops advancing and draws circles (0.4 mm, growing to 1 mm) until the
   pins drop in.
4. The left arm, its gripper pointing along -y coaxial with the T-screw, grips the T-bar, turns it about 195 degrees
   with its wrist, lets go, backs off 4 cm, turns back and grips again, following the screw in by 1.75 mm per turn.
   It counts the T-bar's turn from its wrist encoders while it grips, and stops 0.3 turn past where the screw reaches
   the recess (about 6.6 turns in all).
5. Both arms let go and go back to the ready pose; the regulator stays clamped on the valve.

``--variant no_pins`` slides the yoke on but skips the sideways move (pins left 2.6 mm off the face) before turning;
``--variant no_turn`` seats it on the pins and lets go without turning the screw.
Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry as g  # noqa: E402

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
DEBUG = os.environ.get("DEMO_DEBUG", "1") != "0"

SLIDE_DY = -0.0065          # the yoke's y offset while sliding on: pins 2.6 mm off the face, tip 2.5 mm off the recess
PRESS = 0.0007              # m: the seating move goes this far past contact, so the washer stays pressed
GRIP_DZ = 0.006             # tool point above the body's axis when gripping it
TBAR_OUT = 0.003            # left tool point this far outboard of the T-bar's centre
# The right hand holds the 40 mm body with each finger's target 5 mm inside the body's face (about 25 N a finger,
# under the drive's 50 N cap). Fully closed, both finger drives saturate and the pinch can slide sideways as a whole.
GRIP_BODY = (g.BODY_W / 2 - 0.005) / 0.04695
STROKE_A = (68.0, 290.0)    # deg: the T-bar grip's angle range the left wrist reaches
EXTRA_TURN = 0.30           # turns past the recess floor


def _hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _exp(v) -> np.ndarray:
    a = float(np.linalg.norm(v))
    if a < 1e-12:
        return np.eye(3)
    K = _hat(np.asarray(v) / a)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _log(R) -> np.ndarray:
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    a = math.acos(c)
    if a < 1e-9:
        return np.zeros(3)
    return a / (2 * math.sin(a)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it.
    ``learn`` nudges a pose correction so that the measured tool pose (forward kinematics of the measured joints)
    reaches the commanded one; the correction is then kept for the following moves."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.dp, self.dR = np.zeros(3), np.eye(3)
        self.worst_mm, self.worst_deg = 0.0, 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        pos, rot = np.asarray(pos, float), np.asarray(rot, float)
        want_p, want_R = pos + self.dp, self.dR @ rot
        q, err, _ = ik.solve(want_p, ik.matrix_to_quat_xyzw(want_R), q_init=self.q, base_pos=self.base, iters=80)
        got_p, got_R = ik.tcp_pose(q, self.base)
        self.worst_mm = max(self.worst_mm, float(np.linalg.norm(got_p - want_p)) * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(g.angle_between(got_R, want_R)))
        self.q, self.pos, self.rot = q, pos, rot

    def learn(self, measured_q, gain: float = 0.3, limit: float = 0.015) -> None:
        from yam_arena import ik

        p, R = ik.tcp_pose(np.asarray(measured_q, float), self.base)
        self.dp = np.clip(self.dp + gain * (self.pos - p), -limit, limit)
        self.dR = _exp(gain * _log(self.rot @ R.T)) @ self.dR

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
    gens = [iter(x) for x in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield


def settle(ctx, arm, seconds: float = 1.0):
    """Hold the current target and learn the sag correction from the measured joints."""
    for _ in range(round(seconds * CONTROL_HZ)):
        arm.learn(ctx.measured(arm.side))
        arm.aim(arm.pos, arm.rot)
        yield


def down_across_x():
    """Right tool rotation: pointing down, fingers closing along x (they close along the tool's x axis)."""
    from yam_arena.demo import grip_rotation

    return grip_rotation(math.pi / 2)


def tbar_rotation(a_deg: float) -> np.ndarray:
    """Left tool rotation: pointing along -y (into the T-screw), fingers closing along its x axis, (cos a, 0, sin a)."""
    a = math.radians(a_deg)
    z = np.array([0.0, -1.0, 0.0])
    x = np.array([math.cos(a), 0.0, math.sin(a)])
    return np.column_stack([x, np.cross(z, x), z])


class Ctx:
    def __init__(self, emb=None, task=None):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")
        self.flags = {"seated": False, "turned": False}
        self.turned = 0.0       # rad of tightening commanded so far
        self.turned_est = 0.0   # rad the T-bar has turned, from the left wrist's encoders while it grips
        self.hand = None        # the regulator in the right hand: (origin, rotation) in the tool frame

    def measured(self, side: str) -> np.ndarray:
        """The arm's measured joint angles (its encoders)."""
        from yam_arena.constants import ARM_JOINTS

        robot = self.emb._env.unwrapped.scene[f"{side}_robot"]
        ids = robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0]
        q = robot.data.joint_pos
        return getattr(q, "torch", q)[0, ids].detach().cpu().numpy().astype(float)

    def tool(self, side: str):
        """The measured tool pose: forward kinematics of the measured joints."""
        from yam_arena import ik

        arm = self.left if side == "left" else self.right
        return ik.tcp_pose(self.measured(side), arm.base)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def reg_to_tcp(reg_origin) -> np.ndarray:
    """Right tool point for a regulator origin, gripped as planned (body at GRIP_Y, GRIP_DZ above the axis)."""
    return np.asarray(reg_origin, float) + np.array([0.0, g.GRIP_Y, GRIP_DZ])


def hand_to_tcp(ctx, reg_origin, rot) -> np.ndarray:
    """Right tool point that puts the gripped regulator's origin at ``reg_origin`` (the tool keeps ``rot``)."""
    off = ctx.hand[0] if ctx.hand is not None else rot.T @ np.array([0.0, -g.GRIP_Y, -GRIP_DZ])
    return np.asarray(reg_origin, float) - rot @ off


SEAT_STEP = 0.0001          # m per control step (2 mm/s) while the hand keeps up
SEAT_LAG = 0.0012           # m: the hand held back this far means the pins are on the face: stop pushing, circle
CIRCLE = 0.0004             # m: radius of the small circles the hand draws (x, z) while it waits for the pins to drop
SEAT_TIMEOUT = 15.0         # s


SLIDE_STEP = 0.00025        # m per control step (5 mm/s)
SLIDE_LAG = 0.004           # m: the hand held back this far along the slide means the window caught the post's end
                            # (moving freely it trails its target by about 1 mm)
RETRY_DZ = (0.0008, -0.0008, 0.0016, -0.0016, 0.0)


def slide_on(ctx, arm, travel: float):
    """Slide the yoke over the post along +x. If the encoders say the hand is held back (the window caught the post's
    end), back off 3 mm and come again a little higher or lower."""
    start = arm.pos.copy()
    x, dz, tries = 0.0, 0.0, 0
    for k in range(round(30.0 * CONTROL_HZ)):
        got = ctx.tool(arm.side)[0][0] - start[0] if ctx.emb is not None else x
        if x - got > SLIDE_LAG and tries < len(RETRY_DZ):
            if DEBUG:
                print(f"slide caught at {got * 1000:.1f} mm: back off, retry dz {RETRY_DZ[tries] * 1000:.1f} mm",
                      flush=True)
            x = max(0.0, got - 0.003)
            yield from arm.move(start + [x, 0.0, dz], arm.rot, 0.6)
            dz = RETRY_DZ[tries]
            tries += 1
            yield from arm.move(start + [x, 0.0, dz], arm.rot, 0.4)
            continue
        x = min(travel, x + SLIDE_STEP)
        arm.aim(start + [x, 0.0, dz], arm.rot)
        yield
        if x >= travel:
            break


def seat(ctx, arm, travel: float):
    """Push the regulator sideways onto its pins softly and in steps: advance while the hand keeps up with its
    target; when it is held back (pins on the face), stop advancing and draw small circles until the pins drop into
    their holes and the hand moves on. Ends ``PRESS`` past where the washer meets the face. Encoder feedback only."""
    start = arm.pos.copy()
    y, phase = 0.0, 0.0
    for k in range(round(SEAT_TIMEOUT * CONTROL_HZ)):
        got = ctx.tool(arm.side)[0][1] - start[1] if ctx.emb is not None else y
        lag = y - got
        if lag < SEAT_LAG:
            y = min(travel, y + SEAT_STEP)
            wobble = np.zeros(3)
        else:                                     # circles growing from 0.4 to 1.0 mm over 6 s
            phase += math.radians(30.0)
            radius = min(2.5 * CIRCLE, CIRCLE * (1.0 + phase / (2 * math.pi * 10)))
            wobble = radius * np.array([math.cos(phase), 0.0, math.sin(phase)])
        arm.aim(start + [0.0, y, 0.0] + wobble, arm.rot)
        if DEBUG and k % 20 == 0:
            print(f"seat y {y * 1000:.2f} mm, hand {got * 1000:.2f} mm", flush=True)
        if y >= travel and got >= travel - PRESS - 0.0003:
            break
        yield
    arm.aim(start + [0.0, travel, 0.0], arm.rot)
    yield from wait(0.5)


def fit_regulator(ctx, variant: str = ""):
    """Right arm: pick the regulator up, slide its yoke onto the post, seat it on the pins, hold it until the screw is
    tight, let go. Open loop from the layout, with the payload's sag corrected from the joint encoders."""
    R = ctx.right
    rot = down_across_x()
    rest = np.array([g.REG_START[0], g.REG_START[1], g.TABLE + g.FRAME_HZ])   # resting on the table
    grasp = reg_to_tcp(rest)
    yield from R.move(grasp + [0, 0, 0.08], rot, 2.5, grip=1.0)
    yield from R.move(grasp, rot, 1.5)
    yield from R.set_grip(GRIP_BODY, 1.0)
    yield from wait(0.6)
    # where the regulator sits in the hand: the pads centre its flat-sided body on the tool point across the
    # fingers; along and below the tool, it is where it rested on the table
    if ctx.emb is not None:
        p, Rt = ctx.tool("right")
        off = Rt.T @ (rest - p)
        off[0] = 0.0                               # across the fingers (the tool's x axis): centred
        ctx.hand = (off, Rt.T)
    yield from R.move(R.pos + [0, 0, 0.04], rot, 1.2)
    clear = np.array(g.C_ORIGIN) + [g.PORT_U - 0.080, SLIDE_DY, 0.030]
    ready = np.array(g.C_ORIGIN) + [g.PORT_U - 0.048, SLIDE_DY, 0.0]
    yield from R.move(hand_to_tcp(ctx, clear, rot) + [0, -0.06, 0.0], rot, 2.0)
    yield from R.move(hand_to_tcp(ctx, clear, rot), rot, 1.5)
    yield from settle(ctx, R, 1.5)
    yield from R.move(hand_to_tcp(ctx, ready, rot), rot, 2.0)
    yield from settle(ctx, R, 1.5)
    # slide the yoke over the post (sag correction frozen: contact must not wind it up)
    yield from slide_on(ctx, R, 0.048)
    yield from wait(0.5)
    if variant != "no_pins":
        yield from seat(ctx, R, -SLIDE_DY + PRESS)
    ctx.flags["seated"] = True
    if variant == "no_turn":
        ctx.flags["turned"] = True
    while not ctx.flags["turned"]:
        yield
    yield from R.set_grip(1.0, 1.0)
    yield from wait(0.3)
    yield from R.move(R.pos + [0, 0, 0.05], rot, 1.5)
    yield from R.move(R.home[0], R.home[1], 2.5)


def tool_angle(Rt) -> float:
    """The grip angle (rad) of a left tool pointing along -y: the direction its fingers close, in the x-z plane."""
    return math.atan2(Rt[2, 0], Rt[0, 0])


def turn_screw(ctx, total_turns: float | None = None):
    """Left arm: turn the T-screw in by wrist strokes until its encoders say the T-bar has turned ``total_turns``."""
    from yam_arena import ik

    L = ctx.left
    total = 2 * math.pi * (g.turns_to_clamp() + EXTRA_TURN if total_turns is None else total_turns)
    while not ctx.flags["seated"]:
        yield
    axis = np.array([0.0, -1.0, 0.0])                  # the screw advances along -y
    bar0 = np.array(g.SEATED) + [0.0, g.TBAR0_Y + TBAR_OUT, 0.0]

    def tcp(turned: float) -> np.ndarray:
        return bar0 + axis * g.PITCH * turned / (2 * math.pi)

    def grip_angle(turned: float) -> float:
        base = (math.degrees(turned) + 90.0) % 180.0   # fingers across the T-bar (it lies along x at the start)
        return base if base >= STROKE_A[0] else base + 180.0

    def measured_angle() -> float:
        if ctx.emb is None:
            return tool_angle(tbar_rotation(a + math.degrees(ctx.turned - t0)))   # offline: perfect tracking
        return tool_angle(ctx.tool("left")[1])

    out = -axis * 0.04                            # backed off along the screw, fingertips clear of the T-bar
    a, t0 = grip_angle(0.0), 0.0
    yield from L.move(tcp(0.0) + [0, 0.05, 0.03], tbar_rotation(a), 3.0, grip=1.0)
    yield from L.move(tcp(0.0) + out, tbar_rotation(a), 1.5)
    yield from settle(ctx, L, 1.0)
    while ctx.turned_est < total - math.radians(3.0):
        ctx.turned = ctx.turned_est
        a = grip_angle(ctx.turned)
        yield from L.move(tcp(ctx.turned), tbar_rotation(a), 0.6)
        yield from L.set_grip(0.0, 1.0)
        yield from wait(0.4)
        stroke = min(math.radians(STROKE_A[1] - a), total - ctx.turned + math.radians(2.0))
        n = max(1, round(stroke / g.TURN_SPEED * CONTROL_HZ))
        t0 = ctx.turned
        start = measured_angle()
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            ctx.turned = t0 + s * stroke
            L.aim(tcp(ctx.turned), tbar_rotation(a + math.degrees(ctx.turned - t0)))
            yield
        yield from wait(0.3)
        turned = (measured_angle() - start) % (2 * math.pi)   # a stroke turns it forward, less than a turn
        if turned > 1.9 * math.pi:                              # or a hair backward
            turned -= 2 * math.pi
        ctx.turned_est = t0 + turned
        if DEBUG:
            print(f"stroke {math.degrees(stroke):.0f} deg: encoders say {math.degrees(turned):.0f}, "
                  f"total {math.degrees(ctx.turned_est):.0f} of {math.degrees(total):.0f}", flush=True)
        yield from L.set_grip(1.0, 1.0)
        yield from wait(0.5)
        yield from L.move(tcp(ctx.turned) + out, L.rot, 0.6)
        if ctx.turned_est < total - math.radians(3.0):   # turn the open hand back, clear of the T-bar
            a1, a0 = a + math.degrees(stroke), grip_angle(ctx.turned_est)
            m = round(1.0 * CONTROL_HZ)
            for k in range(1, m + 1):
                L.aim(tcp(ctx.turned) + out, tbar_rotation(a1 + ik.smoothstep(k / m) * (a0 - a1)))
                yield
    yield from L.move(L.pos + [0, 0.0, 0.04], L.rot, 1.0)
    ctx.flags["turned"] = True
    yield from L.move(L.home[0], L.home[1], 2.5)


def script(ctx, variant: str = ""):
    turner = wait(0.05) if variant == "no_turn" else turn_screw(ctx)
    yield from together(fit_regulator(ctx, variant), turner)


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 4800, record: bool = True, variant: str = "",
        cam_width: int = 320, cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    sys.modules.setdefault("task", task)

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb, task)
    gen = script(ctx, variant)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 50 == 0 or steps < 3:
            S = task._STATE
            f = S["fit"]
            print(f"step {steps} score {info.get('score')} cyl {np.round(S.get('cyl_pos', np.zeros(3)), 4)} "
                  f"reg {np.round(S.get('reg_pos', np.zeros(3)), 4)} d_mm {S['d'] * 1000:.2f} "
                  f"turns {S['theta'] / (-2 * math.pi):.2f} on_post {f.get('on_post')} pin_depth_mm "
                  f"{f.get('pin_depth', 0) * 1000:.2f} pin_lat_mm {f.get('pin_lateral', 0) * 1000:.2f} face_mm "
                  f"{f.get('face_gap', 0) * 1000:.2f} tilt {math.degrees(f.get('tilt', 0)):.2f} seated "
                  f"{f.get('seated')} tip_in {f.get('tip_in')} over_mm {f.get('overshoot', 0) * 1000:.3f} clamped "
                  f"{f.get('clamped')} clear {S['clear']} still {S['still']} why {S['why']} "
                  f"corrL_mm {np.round(ctx.left.dp * 1000, 2)} corrR_mm {np.round(ctx.right.dp * 1000, 2)} torque "
                  f"{f.get('torque', 0):.2f}",
                  flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE   # for the log and result.json only
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "fit": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in S["fit"].items()},
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=4800)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--variant", default="", choices=["", "no_pins", "no_turn"])
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, variant=a.variant, cam_width=a.cam_width,
        cam_height=a.cam_height)
