"""Scripted full-score demo for the linked S-hooks task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1800] [--no-record]

The left arm does it all; the right arm stays in its ready pose. Hook A hangs on the rail (its plane x-z); hook B
hangs from hook A's bottom curl, facing the robot (its plane y-z), with its top curl's tip toward +y.
1. From the measured pose of hook B, the gripper comes in from the front, pointing forward and 8 degrees down along
   the hook's middle stroke, and closes across the stroke 3 cm below the hook's centre: the fingers close in the
   hook's plane, square to the stroke, so each pad lies along the rod and the grip holds the hook like a vise.
2. It lifts hook B 2.8 cm (hook A's bottom curl drops out of B's curl, clear under B's tip) and moves it 6 cm to
   the right (-y), so A's curl leaves through B's opening: the hooks are apart.
3. It brings hook B in front of the peg's tip with B's top curl centred on the peg's axis (correcting from a fresh
   measurement of B in the grip), slides it 6 cm along the peg, lowers it until the curl's top is 2 mm above the
   peg, opens, backs straight out and returns to the ready pose. Hook A stays on the rail, untouched.

``yam_scripted`` plays only straight-down grips, so this file drives the embodiment itself (as the other tasks with
tilted grips do). Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import geometry as g

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
TILT = math.radians(8.0)        # the grip points forward and this far down along the hook's middle stroke
GRIP_T = 0.03                   # the grip closes on the stroke this far below the hook's centre, along the stroke
APPROACH = 0.06                 # the open gripper starts this far back along its own axis
LIFT = 0.028                    # hook B lifted this much: hook A's curl drops clear under B's tip
SIDE = 0.06                     # then moved this far toward -y: A's curl leaves through B's opening
BACK = 0.045                    # then this far toward the robot, clear of the rail and its brackets
BEFORE_TIP = 0.035              # B's top curl centred on the peg's axis this far in front of the tip ...
ON_PEG = 0.025                  # ... then slid on until its plane is this far behind the tip
ABOVE = 0.002                   # B's curl top this far above the peg when the fingers open
WITHDRAW = 0.08
PART_OPEN = 0.45                # fingers opened only this far (2.1 cm from the rod's centre line each side): wide
                                # open, they would reach the hook's curls, which lie in the fingers' closing plane
VIA_IN = np.array([0.20, 0.24, 0.95])     # passed on the way in and out, clear of hook A and the rail
UPRIGHT_B = g.rz(math.pi / 2)   # hook B's rotation, facing the robot with its top tip toward +y
# the hook's in-plane direction square to its middle stroke (hook frame), up and toward the top curl's tip side
ACROSS = np.array([-g.STROKE_DIR[2], 0.0, g.STROKE_DIR[0]])


def grip_rotation(R) -> np.ndarray:
    """The TCP rotation gripping the hook at rotation ``R`` across its middle stroke from the front: the approach
    axis is the hook plane's normal toward the board, tilted TILT down along the stroke, and the fingers close
    along the in-plane direction square to the stroke (taken as the downward one: the arm's last joint can't turn
    the other way round)."""
    s = R @ g.STROKE_DIR
    a = math.cos(TILT) * -R[:, 1] + math.sin(TILT) * s
    if a[0] < 0:  # the hook's -y axis may point either way; always approach toward the board (+x)
        a = math.cos(TILT) * R[:, 1] + math.sin(TILT) * s
    c = -(R @ ACROSS)
    c = c - a * (c @ a)
    c /= np.linalg.norm(c)
    return np.column_stack([c, np.cross(a, c), a])


def grip_tcp(pos, R) -> np.ndarray:
    return np.asarray(pos, float) + R @ (g.STROKE_DIR * GRIP_T)


def align(a, b) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _closest_rotation(M) -> np.ndarray:
    u, _, vt = np.linalg.svd(M)
    return u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt


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
        self.worst_mm = self.worst_deg = 0.0
        self.qs: list[np.ndarray] = []

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=80)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(rerr))
        self.qs.append(q.copy())

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


def onto_peg(arm: Arm, pos, R, curl_target):
    """Where the TCP must go (and how turned) to bring hook B upright, facing the robot, with its top curl's centre
    at ``curl_target``, the hook held rigidly in the grip as it is now (measured ``pos``, ``R``)."""
    turn = _closest_rotation(UPRIGHT_B @ R.T)
    d = g.world(pos, R, g.CURLS[0]) - arm.pos
    return np.asarray(curl_target, float) - turn @ d, turn @ arm.rot


def separate(ctx):
    """Left arm: grip hook B, lift it off hook A and out through its curl's opening, hang it on the peg, withdraw."""
    L = ctx.left
    pos, R = ctx.hook("hook_b")
    rot = grip_rotation(R)
    tcp = grip_tcp(pos, R)

    yield from L.move(VIA_IN, rot, 2.0, grip=PART_OPEN)
    yield from L.move(tcp - rot[:, 2] * APPROACH, rot, 1.5)
    yield from L.move(tcp, rot, 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)
    yield from L.move(L.pos + [0.0, 0.0, LIFT], L.rot, 1.5)
    yield from L.move(L.pos + [0.0, -SIDE, 0.0], L.rot, 2.0)
    yield from L.move(L.pos + [-BACK, 0.0, 0.0], L.rot, 1.5)

    front = np.array([g.PEG_TIP_X - BEFORE_TIP, g.PEG_Y, g.PEG_Z])
    for seconds in (3.0, 1.0):            # line up, then once more from a fresh measurement
        tcp, rot = onto_peg(L, *ctx.hook("hook_b"), front)
        yield from L.move(tcp, rot, seconds)
        yield from wait(0.3)
    yield from L.move(L.pos + [BEFORE_TIP + ON_PEG, 0.0, 0.0], L.rot, 2.5)
    yield from L.move(L.pos - [0.0, 0.0, g.CHORD - g.ROD_R - g.SUPPORT_R - ABOVE], L.rot, 1.0)
    yield from wait(0.3)
    yield from L.set_grip(PART_OPEN, 0.8)
    yield from wait(0.5)
    yield from L.move(L.pos - L.rot[:, 2] * WITHDRAW, L.rot, 1.5)   # straight back out along the gripper's axis
    yield from L.move(VIA_IN, L.rot, 2.0)
    yield from L.move(L.home[0], L.home[1], 2.0, grip=1.0)


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")

    def hook(self, name):
        from yam_arena.scene import object_states

        b = object_states(self.emb._env, [name])[name]
        return np.asarray(b.pos, float), g.rot(b.quat_xyzw)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1800, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb, task)
    gen = separate(ctx)
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
        if steps % 20 == 0:
            S = task._STATE
            print(f"step {steps} score {info.get('score')} on {S['on']} linked {S['linked']} gap {S['gap']:.3f} "
                  f"held {S['held']} clear {S['clear']} still {S['still']} best {S['best']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "best": S["best"],
              "down": S["down"], "worst_ik_mm": round(ctx.left.worst_mm, 2)}
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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1800)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
