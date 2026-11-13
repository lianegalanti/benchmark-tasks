"""Scripted full-score demo for refitting the insect screen, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

It works from the fixed, known layout (geometry.py) and the arms' own joint encoders only. It never reads where the
screen is, or its plungers' joints, from the simulator (the task's monitor state is only printed, as a log).

Both arms work together throughout; each holds one stile, so the screen moves as one rigid piece between them.

1. Each arm comes in from outside the screen's side, its fingers part open and pointing inward and down at 45
   degrees in the screen's plane, and pinches its stile across the frame's 10 mm thickness, 9 cm above the bottom
   edge (one finger in front, one behind the screen).
2. They lift the screen 12 mm off the table, bring its top edge forward under the top channel (leaning 6 degrees,
   bottom toward the room) and raise it to 8 mm below the channel's mouth.
3. They push it straight up until its top edge is 1.5 mm under the channel's ceiling: the plungers go in 17.5 mm,
   and the bottom edge rises 6.5 mm above the sill's front lip.
4. Holding the top up, they swing the bottom in, about the top edge, until the screen stands in the window's plane,
   right over the bottom channel.
5. They let it down until its bottom edge is 1 mm above the channel's floor, open their fingers (the plungers push it
   down onto the floor), and draw back along their approach, then go back to ready.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arm's own
encoders) takes out the sag from carrying the screen and pushing against the plungers.

:func:`plan` runs the same script without Isaac Sim, for the tests (reach, joint limits, clearances).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import geometry as g  # noqa: E402

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
GRIP_Z = 0.090             # TCP this far up the stile from the bottom edge (S z) ...
GRIP_IN = 0.004            # ... and this far in from the screen's side edge (the fingertips reach 13 mm further)
APPROACH = math.radians(45)    # fingers point inward and down this far below horizontal, in the screen's plane
OPEN = 0.45                # finger opening while coming in and letting go (42 mm across a 10 mm frame)
BACK_OFF = 0.07            # m: come in from, and draw back to, this far along the approach
LIFT = 0.012               # m: first lift off the table
LEAN_IN = math.radians(6)  # lean while the top goes in and up
BELOW_MOUTH = 0.008        # m: top edge this far under the channel's mouth before going up
UNDER_CEIL = 0.0015        # m: pushed up until the top edge is this far under the ceiling
SEAT_ABOVE = 0.001         # m: let down until the bottom edge is this far above the floor, then let go
I_GAIN, I_MAX = 0.08, 0.25
SIGNS = {"left": 1.0, "right": -1.0}


def grip_in_screen(side: str):
    """The TCP's position and rotation in the screen frame S for one arm: fingers closing along the screen's normal,
    approach in its plane, pointing inward (toward the screen's centre line) and down by APPROACH."""
    sy = SIGNS[side]
    a = np.array([0.0, -sy * math.cos(APPROACH), -math.sin(APPROACH)])
    x = np.array([1.0, 0.0, 0.0])
    rot = np.column_stack([x, np.cross(a, x), a])
    pos = np.array([0.0, sy * (g.SCREEN_W / 2 - GRIP_IN), GRIP_Z])
    return pos, rot


def tcp_for_screen(side: str, o, r):
    """Station TCP pose holding the screen at pose (o, R) given in W."""
    p, rr = grip_in_screen(side)
    return g.WINDOW + np.asarray(o) + np.asarray(r) @ p, np.asarray(r) @ rr


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
        self.worst_mm, self.worst_deg = 0.0, 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(rerr))

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

    def set_grip(self, grip: float, seconds: float = 1.0):
        yield from self.move(self.pos, self.rot, seconds, grip)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def together(*gens):
    gens = [iter(x) for x in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield


# -- the screen's waypoints (W frame), from the known layout ----------------------------------------------------------
def waypoints() -> dict:
    o0, r0 = g.start_pose()
    pivot0 = o0 + r0 @ g.TOP
    return {
        "start": (pivot0, g.START_LEAN),
        "lifted": (pivot0 + [0.0, 0.0, LIFT], g.START_LEAN),
        "forward": (np.array([0.0, 0.0, pivot0[2] + LIFT]), LEAN_IN),
        "under": (np.array([0.0, 0.0, g.MOUTH - BELOW_MOUTH]), LEAN_IN),
        "up": (np.array([0.0, 0.0, g.CEIL - UNDER_CEIL]), LEAN_IN),
        "swung": (np.array([0.0, 0.0, g.CEIL - UNDER_CEIL]), 0.0),
        "down": (np.array([0.0, 0.0, g.SCREEN_H + SEAT_ABOVE]), 0.0),
    }


class Script:
    """The demo as per-step arm commands. ``screen`` is the screen pose (o, R in W) the script expects while it
    holds it (for the tests)."""

    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.wp = waypoints()
        self.screen = g.pose_from_pivot(*self.wp["start"])
        self.holding = False
        self.mark: dict = {}
        self.stage = "start"

    def _both(self, fn):
        return together(fn(self.left), fn(self.right))

    def carry(self, name: str, seconds: float):
        """Move the held screen from its current waypoint to ``name``: pivot and lean interpolated together."""
        from yam_arena import ik

        p0, l0 = self.wp[self.stage]
        p1, l1 = self.wp[name]
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            o, r = g.pose_from_pivot(np.asarray(p0) + s * (np.asarray(p1) - p0), l0 + s * (l1 - l0))
            self.screen = (o, r)
            for arm in (self.left, self.right):
                arm.aim(*tcp_for_screen(arm.side, o, r))
            yield
        self.stage = name

    def steps(self):
        o, r = self.screen

        def approach(arm):
            pos, rot = tcp_for_screen(arm.side, o, r)
            a = rot[:, 2]
            yield from arm.move(pos - BACK_OFF * a, rot, 3.0, grip=OPEN)
            yield from arm.move(pos, rot, 1.5)
            yield from arm.set_grip(0.0, 1.0)

        yield from self._both(approach)
        yield from wait(0.6)
        self.holding = True
        self.mark["grasped"] = True
        yield from self.carry("lifted", 1.0)
        yield from self.carry("forward", 2.0)
        yield from self.carry("under", 1.5)
        yield from wait(0.3)
        self.mark["up"] = True
        yield from self.carry("up", 2.0)
        yield from wait(1.5)
        self.mark["swing"] = True
        yield from self.carry("swung", 3.0)
        yield from wait(0.5)
        self.mark["down"] = True
        yield from self.carry("down", 1.5)
        yield from wait(0.5)

        def release(arm):
            yield from arm.set_grip(OPEN, 1.0)
            yield from wait(0.3)
            a = arm.rot[:, 2]
            yield from arm.move(arm.pos - BACK_OFF * a, arm.rot, 1.5)
            yield from arm.move(arm.home[0], arm.home[1], 2.0)

        self.mark["release"] = True
        self.holding = False
        # once let go, the screen drops SEAT_ABOVE onto the floor (the plungers push it down)
        o, r = g.pose_from_pivot(np.array([0.0, 0.0, g.SCREEN_H]), 0.0)
        self.screen = (o, r)
        yield from self._both(release)


def plan(max_steps: int = 2400) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, and the screen pose the
    script assumes."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "screen": (s.screen[0].copy(), s.screen[1].copy()), "holding": s.holding,
                    "stage": s.stage, "mark": dict(s.mark)})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in (("left", s.left), ("right", s.right))}})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.constants import ARM_JOINTS
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    from scenekit import arm_joints

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    script = Script()
    gen = script.steps()
    integ = {"left": np.zeros(6), "right": np.zeros(6)}
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def command(side, arm):
        # proprioceptive correction: the arm's measured joints toward the planned ones
        q = arm_joints(emb._env, side, ARM_JOINTS)
        integ[side] = np.clip(integ[side] + I_GAIN * (arm.q - q), -I_MAX, I_MAX)
        return np.r_[arm.q + integ[side], arm.grip]

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    seen: set = set()
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(Action(data=pack(command("left", script.left), command("right", script.right))))
        steps += 1
        info = res.info
        snap(res.observation)
        new = [k for k in script.mark if k not in seen]
        seen.update(new)
        if steps % 20 == 0 or new:      # a log only: the script never uses it
            S = task._STATE
            from yam_arena import ik as _ik
            lag = {k: np.round((_ik.tcp_pose(arm_joints(emb._env, k, ARM_JOINTS), a.base)[0] - a.pos) * 1000, 1).tolist()
                   for k, a in (("left", script.left), ("right", script.right))}
            print(f"step {steps} {script.stage} {new} tcp_lag_mm {lag} score {info.get('score')} level "
                  f"{S.get('level')} {S.get('info')} clear {S.get('clear')} still {S.get('still')} why {S['why']}",
                  flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "level": S.get("level"), "info": S.get("info"),
              "worst_ik_mm": {"left": round(script.left.worst_mm, 2), "right": round(script.right.worst_mm, 2)}}
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
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
