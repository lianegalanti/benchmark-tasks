"""Scripted full-score demo for the drying-peg task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The left arm does it all; the right arm stays in its ready pose.
1. It grips the lying bottle across its middle (at its centre of mass), fingers closing along y on the two side
   staves, with the gripper pointing down and tilted 20 degrees back toward the robot.
2. It lifts the bottle and turns the gripper 90 degrees about the finger axis: the bottle's mouth, which pointed at
   the robot, now points straight down, and the gripper points forward and 20 degrees down (pointing straight
   forward, as the bottle would need with an untilted grip, is out of the arm's reach this low).
3. From the measured bottle pose it lines the bottle's axis up over the left peg, mouth 2.5 cm above the peg's tip,
   correcting any tilt the bottle picked up in the grip; then it lowers the bottle straight down over the peg until
   the inside bottom is 4 mm above the tip.
4. It opens the fingers (the bottle drops 4 mm onto the tip and hangs), backs the gripper out along its own axis and
   returns to the ready pose.

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
BETA = math.radians(20.0)       # the pick grip's tilt back toward the robot (= the hang grip's tilt below level)
ABOVE = 0.025                   # the mouth this far above the peg's tip before lowering
DROP = 0.004                    # the inside bottom this far above the tip when the fingers open
INTERIOR = g.MOUTH_Z - g.INNER_BOTTOM_Z


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def align(a, b) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1 + c)


def grips():
    """(pick rotation, hang rotation) of the TCP."""
    from yam_arena.demo import grip_rotation

    down = grip_rotation(0.0)  # fingers close along world y
    return _ry(BETA) @ down, _ry(BETA - math.pi / 2) @ down


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


def over_peg(arm: Arm, pos, R, peg_xy, inner_bottom_z: float):
    """Where the TCP must go (and how turned) to bring the bottle's axis upright over the peg with its inside bottom
    at ``inner_bottom_z``, the bottle held rigidly in the grip as it is now (measured ``pos``, ``R``)."""
    turn = align(g.mouth_dir(R), (0.0, 0.0, -1.0))
    d = g.point(pos, R, g.INNER_BOTTOM_Z) - arm.pos
    target = np.array([peg_xy[0], peg_xy[1], inner_bottom_z])
    return target - turn @ d, turn @ arm.rot


def hang_bottle(ctx):
    """Left arm: pick, turn mouth-down, lower over the peg, let go, withdraw."""
    L = ctx.left
    pick_rot, hang_rot = grips()
    pos, R = ctx.bottle()
    com = g.point(pos, R, g.COM_Z)
    peg = g.PEGS[g.DEMO_PEG]

    yield from L.move(com + [0.0, 0.0, 0.09], pick_rot, 2.5, grip=1.0)
    yield from L.move(com, pick_rot, 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)
    yield from L.move(com + [0.0, 0.0, 0.10], pick_rot, 1.5)
    yield from L.move((com[0], 0.25, g.TOP + 0.25), hang_rot, 3.0)
    yield from wait(0.5)

    high = g.PEG_TIP + INTERIOR + ABOVE
    for seconds in (2.5, 1.0):            # line up, then once more from a fresh measurement
        tcp, rot = over_peg(L, *ctx.bottle(), peg, high)
        yield from L.move(tcp, rot, seconds)
        yield from wait(0.3)
    yield from L.move(L.pos - [0.0, 0.0, high - (g.PEG_TIP + DROP)], L.rot, 4.0)
    yield from wait(0.3)
    yield from L.set_grip(1.0, 0.8)
    yield from wait(0.5)
    yield from L.move(L.pos - L.rot[:, 2] * 0.12, L.rot, 1.5)   # straight back out along the gripper's axis
    yield from L.move(L.home[0], L.home[1], 2.5)


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")

    def bottle(self):
        from yam_arena.scene import object_states

        b = object_states(self.emb._env, ["bottle"])["bottle"]
        return np.asarray(b.pos, float), g.rot(b.quat_xyzw)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
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
    gen = hang_bottle(ctx)
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
            print(f"step {steps} score {info.get('score')} depth {S['depth']:.3f} tilt {math.degrees(S['tilt']):.1f} "
                  f"lowest {S['lowest'] - g.TOP:.3f} held {S['held']} clear {S['clear']} still {S['still']} "
                  f"best {S['best']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "best": S["best"],
              "fallen": S["fallen"], "worst_ik_mm": round(ctx.left.worst_mm, 2)}
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
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
