"""Scripted full-score demo for the fallen-bottle task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 700] [--no-record]

The left arm does it all; the right arm stays in its ready pose. Every move after the first follows the fallen
bottle's carrier along the line (the targets are set relative to the carrier's scripted position, one per control
step), so the bottle never has to be put down anywhere else.
1. Over the line where the carrier will be at 4.25 s (its centre 27 cm left of the midpoint), the gripper pointing
   down and tilted 20 degrees toward the robot's left, fingers closing across the line (along x). It re-measures the
   bottle, follows the carrier down and grips the lying bottle across its body, 1.5 cm toward the cap from its middle
   and 1 cm above its axis (pad tips 1.6 cm over the carrier).
2. It lifts the bottle 3 cm, rises 6 cm more, then turns the gripper a quarter turn about the finger axis: the cap,
   which pointed downstream, now points up, and the gripper points downstream and 20 degrees down.
3. From the measured bottle pose it brings the bottle's axis upright with its base 3 cm over the carrier's centre,
   lowers it until the base is 3 mm above the carrier (nudging the target by the measured offset of the bottle's
   centre from the carrier's), lets go and rises 8 cm, then parks well clear of the line.

``yam_scripted`` plays only straight-down grips, so this file drives the embodiment itself (as the drying-peg task's
demo does; the Arm class is adapted from it). Writes one MP4 per camera, setup.png (the first top-camera frame) and
result.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
TILT = math.radians(20.0)  # the pick grip's tilt toward +y (= the hold grip's tilt below level, pointing -y)
MEET_Y = 0.29  # the carrier's centre when the gripper starts down over the bottle
PRE_UP = 0.07  # the TCP waits this far above its grasp point
GRIP_ALONG = 0.015  # grasp this far from the bottle's middle toward its cap
GRIP_ABOVE = 0.010  # and this far above its axis (the pads close on a chord 5.7 cm wide)
LIFT, RISE = 0.03, 0.06  # off the carrier, then higher before the turn
ABOVE = 0.03  # the base this far above the carrier after the turn
DROP = 0.003  # the base this far above the carrier when the fingers open
K_CORR, MAX_CORR = 0.15, 0.02  # the per-step gain and limit of the centring correction while lowering
PARK = (0.24, 0.26, 0.20)  # where the TCP goes at the end (above the table top), pointing down, away from the line


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def grips():
    """(pick rotation, hold rotation) of the TCP: fingers closing along x, the approach tilted 20 degrees toward +y
    from straight down; then a quarter turn about x, so the approach points -y, 20 degrees below level."""
    from yam_arena.demo import grip_rotation

    down = grip_rotation(math.pi / 2)  # straight down, fingers closing along x
    return _rx(TILT) @ down, _rx(TILT - math.pi / 2) @ down


def align(a, b) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` onto unit vector ``b``."""
    a, b = np.asarray(a, float) / np.linalg.norm(a), np.asarray(b, float) / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K / (1 + c)


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.worst_mm = self.worst_deg = 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=300)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(rerr))

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def move(arm: Arm, pos, rot, seconds: float, grip: float | None = None):
    """Generator: a smooth straight-line move in the station frame; ``grip`` 1 open .. 0 closed."""
    from yam_arena import ik

    p0, r0, g0 = arm.pos.copy(), arm.rot.copy(), arm.grip
    g1 = g0 if grip is None else grip
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        arm.grip = g0 + s * (g1 - g0)
        arm.aim(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
        yield


def follow(ctx, arm: Arm, rel, rot, seconds: float, grip: float | None = None, correct: bool = False):
    """Generator: a smooth move to ``rel`` (TCP minus the carrier's top centre) and ``rot`` while following the
    carrier, one target per control step. With ``correct``, the target is nudged by the measured offset of the
    bottle's centre from the carrier's centre (an integrating correction, so a tracking lag doesn't decentre it)."""
    from yam_arena import ik

    rel0 = arm.pos - ctx.carrier(ctx.t() - DT)
    r0, g0 = arm.rot.copy(), arm.grip
    g1 = g0 if grip is None else grip
    n = max(1, round(seconds * CONTROL_HZ))
    for k in range(1, n + 1):
        s = ik.smoothstep(k / n)
        if correct:
            pos, _ = ctx.fallen()
            err = (ctx.carrier(ctx.t() - DT) - pos)[:2]
            ctx.corr[:2] = np.clip(ctx.corr[:2] + K_CORR * err, -MAX_CORR, MAX_CORR)
        arm.grip = g0 + s * (g1 - g0)
        arm.aim(ctx.carrier(ctx.t()) + rel0 + s * (np.asarray(rel, float) - rel0) + ctx.corr,
                ik.slerp_matrix(r0, rot, s))
        yield


def stand_up(ctx):
    """Left arm: meet the fallen bottle on its carrier, grip it, lift it, turn it cap-up, set it back on the carrier,
    let go and park."""
    M = ctx.task
    L = ctx.left
    pick, hold = grips()

    def grasp_rel():
        pos, R = ctx.fallen()
        return pos + R[:, 2] * GRIP_ALONG + np.array([0.0, 0.0, GRIP_ABOVE]) - ctx.carrier(ctx.t() - DT)

    up = np.array([0.0, 0.0, PRE_UP])
    t_meet = M.arrival(M.FALLEN, MEET_Y)
    yield from move(L, ctx.carrier(t_meet) + grasp_rel() + up, pick, t_meet - ctx.t() + DT, grip=1.0)
    g = grasp_rel()  # measured again, the gripper now just above
    yield from follow(ctx, L, g + up, pick, 0.2)
    yield from follow(ctx, L, g, pick, 1.0)
    yield from follow(ctx, L, g, pick, 0.6, grip=0.0)
    yield from follow(ctx, L, g, pick, 0.3)
    yield from follow(ctx, L, g + [0.0, 0.0, LIFT], pick, 0.6)
    yield from follow(ctx, L, g + [0.0, 0.0, LIFT + RISE], pick, 0.7)
    yield from follow(ctx, L, g + [0.0, 0.0, LIFT + RISE], hold, 1.8)
    yield from follow(ctx, L, L.pos - ctx.carrier(ctx.t() - DT), hold, 0.3)

    for seconds in (0.6, 0.3):  # line up from the measured pose, then once more from a fresh measurement
        pos, R = ctx.fallen()
        turn = align(R[:, 2], (0.0, 0.0, 1.0))
        d = pos - R[:, 2] * M.H / 2 - L.pos  # from the TCP to the bottle's base centre
        rel = np.array([0.0, 0.0, ABOVE]) - turn @ d
        yield from follow(ctx, L, rel, turn @ L.rot, seconds)
    yield from follow(ctx, L, rel - [0.0, 0.0, ABOVE - DROP], L.rot, 1.2, correct=True)
    yield from follow(ctx, L, rel - [0.0, 0.0, ABOVE - DROP], L.rot, 0.2)
    yield from follow(ctx, L, rel - [0.0, 0.0, ABOVE - DROP], L.rot, 0.6, grip=1.0)
    yield from follow(ctx, L, rel - [0.0, 0.0, ABOVE - DROP], L.rot, 0.2)
    yield from follow(ctx, L, rel + [0.0, 0.0, 0.08 - ABOVE + DROP], L.rot, 0.8)  # straight up, fingers beside it
    yield from move(L, (PARK[0], PARK[1], M.TOP + PARK[2]), _rx(-TILT) @ pick, 2.0)  # straight down


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")
        self.k = 0  # control steps taken so far
        self.corr = np.zeros(3)

    def t(self) -> float:
        """The episode time the next command is for (the end of the coming control step)."""
        return (self.k + 1) * DT

    def carrier(self, t: float) -> np.ndarray:
        """The fallen bottle's carrier's top centre at episode time ``t`` (its scripted path)."""
        M = self.task
        return np.array([M.LINE_X, M.carrier_y(M.FALLEN, t), M.CARRIER_TOP])

    def fallen(self):
        """The fallen bottle's centre and rotation, as measured after the last step."""
        from yam_arena.scene import object_states

        M = self.task
        b = object_states(self.emb._env, [M.FALLEN])[M.FALLEN]
        return np.asarray(b.pos, float), M.rot(b.quat_xyzw)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 700, record: bool = True, cam_width: int = 320,
        cam_height: int = 176) -> dict:
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
    gen = stand_up(ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, success, info = 0, False, {}
    while steps < max_steps:
        ctx.k = steps
        try:
            next(gen)
        except StopIteration:
            pass
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 20 == 0:
            S = task._STATE
            pos, R = ctx.fallen()
            print(f"step {steps} t {steps * DT:.1f} score {info.get('score')} level {S['level']} ({S['why']}) "
                  f"tilt {math.degrees(math.acos(np.clip(R[2, 2], -1, 1))):.1f} bottle {np.round(pos, 3).tolist()} "
                  f"carrier_y {task.carrier_y(task.FALLEN, steps * DT):.3f} disturbed {S['disturbed']} "
                  f"decided {S['decided']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "decided": S["decided"], "why": S["why"],
              "disturbed": S["disturbed"], "worst_ik_mm": round(ctx.left.worst_mm, 2),
              "worst_ik_deg": round(ctx.left.worst_deg, 3)}
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
    ap.add_argument("--max-steps", type=int, default=700)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
