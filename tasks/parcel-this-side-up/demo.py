"""Scripted full-score demo for the parcel task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--no-record]

The left arm does it all; the right arm stays in its ready pose.
1. It grips the lying parcel across its middle (at its centre of mass), fingers closing along x on the two long sides
   that face forward and back, with the gripper pointing down and tilted 20 degrees toward the robot's left.
2. It lifts the parcel 12 cm and turns the gripper 90 degrees about the finger axis while carrying it over the pallet
   mark: the end the arrows point to, which pointed at the robot's right, now points straight up, and the gripper
   points to the robot's right and 20 degrees down. (Turning it the other way, which the right arm would do with the
   mirrored motion, would stand it upside down.)
3. From the measured parcel pose it lines the parcel up upright over the mark's centre, its bottom 3 cm above the
   tape, correcting any tilt it picked up in the grip; then it lowers it until the bottom is 4 mm above the tape.
4. It opens the fingers (the parcel drops 4 mm and stands), backs the gripper out along its own axis and returns to
   the ready pose.

Written in the style of the other tilted-grip demos in this repository: ``yam_scripted`` plays only straight-down
grips, so this file drives the embodiment itself. Writes one MP4 per camera, setup.png (the first top-camera frame)
and result.json.
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
BETA = math.radians(20.0)       # the pick grip's tilt toward the robot's left (= the standing grip's tilt below level)
GRIP_YAW = -math.pi / 2         # fingers close along x (the other yaw runs joint 6 into its limit)
ABOVE = 0.03                    # the parcel's bottom this far above the tape before the last descent
DROP = 0.004                    # ... and this far when the fingers open
LIFT = 0.12


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def align(a, b) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1 + c)


def grips():
    """(pick rotation, standing rotation) of the TCP."""
    from yam_arena.demo import grip_rotation

    down = grip_rotation(GRIP_YAW)
    return _rx(BETA) @ down, _rx(BETA - math.pi / 2) @ down


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


def over_mark(arm: Arm, pos, R, bottom_z: float):
    """Where the TCP must go (and how turned) to stand the parcel upright, the right way up, over the mark's centre
    with its bottom at ``bottom_z``, the parcel held rigidly in the grip as it is now (measured ``pos``, ``R``)."""
    turn = align(g.up_dir(R), (0.0, 0.0, 1.0))
    d = g.point(pos, R, (0.0, 0.0, -g.LENGTH / 2)) - arm.pos
    target = np.array([g.MARK_XY[0], g.MARK_XY[1], bottom_z])
    return target - turn @ d, turn @ arm.rot


def stand_parcel(ctx):
    """Left arm: pick across the middle, turn the arrows' end up, stand it on the mark, let go, withdraw."""
    L = ctx.left
    pick_rot, stand_rot = grips()
    pos, _ = ctx.parcel()
    com = np.asarray(pos, float)

    yield from L.move(com + [0.0, 0.0, 0.09], pick_rot, 2.5, grip=1.0)
    yield from L.move(com, pick_rot, 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)
    yield from L.move(com + [0.0, 0.0, LIFT], pick_rot, 1.5)
    high = g.MARK_TOP + ABOVE + g.LENGTH / 2
    yield from L.move((g.MARK_XY[0], g.MARK_XY[1] + 0.02, high + 0.06), stand_rot, 3.0)
    yield from wait(0.4)

    for seconds in (2.0, 1.0):            # line up, then once more from a fresh measurement
        tcp, rot = over_mark(L, *ctx.parcel(), g.MARK_TOP + ABOVE)
        yield from L.move(tcp, rot, seconds)
        yield from wait(0.3)
    yield from L.move(L.pos - [0.0, 0.0, ABOVE - DROP], L.rot, 1.5)
    yield from wait(0.3)
    yield from L.set_grip(1.0, 0.8)
    yield from wait(0.5)
    yield from L.move(L.pos - L.rot[:, 2] * 0.12, L.rot, 1.5)   # straight back out along the gripper's axis
    yield from L.move(L.home[0], L.home[1], 2.5)


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")

    def parcel(self):
        from yam_arena.scene import object_states

        p = object_states(self.emb._env, ["parcel"])["parcel"]
        return np.asarray(p.pos, float), g.rot(p.quat_xyzw)

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
    gen = stand_parcel(ctx)
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
            print(f"step {steps} score {info.get('score')} tilt {math.degrees(S['tilt']):.1f} "
                  f"off {S['centre_off']:.3f} low {S['low'] - g.TOP:.3f} held {S['held']} clear {S['clear']} "
                  f"still {S['still']} best {S['best']}", flush=True)
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
