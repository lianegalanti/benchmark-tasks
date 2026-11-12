"""Scripted full-score demo for the folding-crate task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

The demo is open loop: the arms work only from the crate's fixed, known start layout, on a fixed schedule. It reads
no simulator state (the crate's joints are read only for the progress printout).

1. The right arm pinches the near long wall's top rail through its handhold, swings the wall up onto its stop, lets
   go and lifts away. The wall stands on its own.
2. The left arm does the same with the far long wall.
3. The right arm pinches the folded right short wall's rail and holds it: this braces the crate. The left arm pinches
   the left short wall's rail and swings the wall up, then pulls it on past upright for a fixed 3 s (8 degrees a
   second of commanded swing) so both its corner catches click out behind it.
4. The left arm now holds its latched wall, bracing the crate, while the right arm swings the right short wall up
   and pulls it through its catches the same way.
5. Both grippers open, rise and move clear beside the crate. The crate stands with all four catches latched.

The grip on a rail is a light pinch (fingers commanded 14 mm apart on the 18 mm rail), so the round rail turns
between the pads as its wall swings: the gripper keeps a fixed tilt instead of turning with the wall.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
G_OPEN, G_PRE, G_PINCH = 1.0, 0.426, 0.15   # normalised grip: 94 mm, 40 mm and 14 mm between the pads
HOVER = 0.06                                # above a rail before going down onto it
PULL_RATE = math.radians(8.0)               # rad/s of extra pull past upright while a short wall clicks in ...
PULL_S = 3.0                                # ... for this long (in Isaac Sim the catches clicked after about 1.5 s)
EASE = math.radians(2.0)                    # pull kept on a latched wall while it braces the crate


def _here():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import geometry as g

    return g


def grip_rot(wall: str, beta: float, flip: float = 0.0) -> np.ndarray:
    """The pinch for a wall's rail: fingers closing across the rail, the approach tilted ``beta`` about the rail."""
    from yam_arena.demo import grip_rotation

    g = _here()
    a = g.WALLS[wall].axis
    a = a if a.sum() > 0 else -a
    yaw = (0.0 if abs(a[0]) > 0.5 else math.pi / 2) + flip
    return g.rot_axis(a, beta) @ grip_rotation(yaw)


# Per wall: the arm that raises it, the tilt (degrees) at q = 0 and from q = 60 degrees on, and the grip's turn.
PLAN = {
    "wall_near": ("right", 0.0, 0.0, 0.0),
    "wall_far": ("left", -20.0, -45.0, math.pi),
    "wall_left": ("left", 0.0, 0.0, 0.0),
    "wall_right": ("right", 0.0, 0.0, 0.0),
}


def tilt(wall: str, q: float) -> float:
    _, b0, b1, _ = PLAN[wall]
    s = min(max(q / math.radians(60.0), 0.0), 1.0)
    return math.radians(b0 + s * (b1 - b0))


def rail_pose(wall: str, q: float):
    g = _here()
    return g.rail_world(wall, q), grip_rot(wall, tilt(wall, q), PLAN[wall][3])


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0
        self.worst_deg = 0.0
        self.trace: list = []

    def aim(self, pos, rot, grip: float | None = None):
        from yam_arena import ik

        if grip is not None:
            self.grip = grip
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(re))
        self.cmd = np.r_[q, self.grip]
        self.trace.append((self.q.copy(), self.grip))

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.aim(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s), g0 + s * (g1 - g0))
            yield

    def hold(self, seconds: float, grip: float | None = None):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def arc(self, wall: str, q0: float, q1: float, duration: float):
        """Carry a pinched rail along its wall's swing from q0 to q1."""
        from yam_arena import ik

        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            q = q0 + ik.smoothstep(k / n) * (q1 - q0)
            self.aim(*rail_pose(wall, q))
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def parallel(*gens):
    """Run generators side by side, one step each per control step, until all are done."""
    gens = list(gens)
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        yield


def take_rail(arm: Arm, wall: str):
    """From wherever the arm is: over the folded wall's rail, down onto it with the fingers part open, pinch."""
    pos, rot = rail_pose(wall, 0.0)
    yield from arm.move(pos + [0, 0, HOVER + 0.04], rot, 2.0, G_PRE)
    yield from arm.move(pos + [0, 0, HOVER], rot, 0.8)
    yield from arm.move(pos, rot, 1.2)
    yield from arm.hold(1.3, G_PINCH)


def let_go(arm: Arm, park):
    """Open, rise straight up 4.5 cm (fingertips 2 cm over the walls' tops), then move to ``park`` (a TCP position
    beside the crate, gripper pointing down)."""
    from yam_arena.demo import grip_rotation

    yield from arm.hold(1.2, G_OPEN)
    yield from arm.move(arm.pos + [0, 0, 0.045], arm.rot, 1.0)
    yield from arm.move(np.asarray(park, float), grip_rotation(0.0), 1.8)


def raise_long(arm: Arm, wall: str, park):
    yield from take_rail(arm, wall)
    yield from arm.arc(wall, 0.0, math.radians(95.0), 3.5)   # 5 degrees past the stop: pressed onto it
    yield from arm.hold(0.5)
    yield from let_go(arm, park)


def click_short(arm: Arm, wall: str):
    """Swing a pinched short wall up, then pull on past upright for a fixed time, through its two catches."""
    g = _here()
    yield from arm.arc(wall, 0.0, g.UP, 4.0)
    q = g.UP
    for _ in range(round(PULL_S * CONTROL_HZ)):
        q += PULL_RATE / CONTROL_HZ
        arm.aim(*rail_pose(wall, q))
        yield
    yield from arm.hold(0.5)
    yield from arm.arc(wall, q, g.UP + EASE, 1.0)


PARK = {"left": (0.18, 0.33, 0.7475 + 0.19), "right": (0.18, -0.33, 0.7475 + 0.19)}


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    L, R = ctx.left, ctx.right

    # 1-2. The long walls, right arm first (near), then the left arm (far).
    yield from raise_long(R, "wall_near", PARK["right"])
    yield from raise_long(L, "wall_far", PARK["left"])
    # 3. Right arm braces on the folded right wall; left arm takes its wall and clicks it in.
    yield from parallel(take_rail(R, "wall_right"), take_rail(L, "wall_left"))
    yield from click_short(L, "wall_left")
    # 4. Left arm holds its latched wall (brace); right arm clicks the right wall in.
    yield from click_short(R, "wall_right")
    yield from wait(0.5)
    # 5. Let go and move clear.
    yield from parallel(let_go(L, PARK["left"]), let_go(R, PARK["right"]))


def offline_plan():
    """The demo's arm commands without the simulator (it is open loop): the two Arms and the step count."""
    from yam_arena import station

    class OfflineCtx:
        def __init__(self):
            self.left = Arm("left", station.BASES["left"])
            self.right = Arm("right", station.BASES["right"])

    ctx = OfflineCtx()
    steps = sum(1 for _ in script(None, ctx))
    return ctx.left, ctx.right, steps


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])

    def crate_joints(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, ["crate"])["crate"].joints

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 4.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
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
        if steps % 40 == 0 or os.environ.get("DEMO_DEBUG"):
            j = ctx.crate_joints()
            st = task._STATE
            hinges = " ".join(f"{w[5:]}={math.degrees(j.get(w + '_hinge', 0)):.1f}" for w in
                              ("wall_near", "wall_far", "wall_left", "wall_right"))
            pawls = " ".join(f"{p[6:]}={1000 * j.get(p + '_slide', 0):.1f}" for p in
                             ("catch_far_left", "catch_far_right", "catch_near_left", "catch_near_right"))
            print(f"step {steps} score {info.get('score')} hinges(deg) {hinges} catches(mm) {pawls} "
                  f"latched {sum(st['latched'].values())} released {st['released']} still {st['still']} "
                  f"why {st['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps,
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
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
