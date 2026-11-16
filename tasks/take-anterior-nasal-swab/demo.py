"""Scripted full-score demo for the anterior nasal swab, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 3000] [--circles 5] [--no-record]

It works open loop from the fixed, known layout (geometry.py) and the arms' own joint readings only: it never reads
the swab's, the head's or the nose's state from the simulator. Only the right arm moves; the left stays at rest.

1. Fingers level, it takes the swab's cap from behind (across the cap's two flats), lifts the swab straight out of
   its tube and swings it round to below the patient's nose, tip up along the left nostril's axis (65 degrees above
   horizontal).
2. It slides the tip 18 mm up the nostril (the whole 15 mm foam tip inside), then moves it out against the wall and
   round the nostril's oval in eight circles (the commanded path is 5.5 mm outside the free path, so the soft nose is
   pressed up to about 1.4 N); back to the middle and out. The same in the right nostril.
3. It swings the swab back over its tube, lowers it in until the cap's flange sits on the rim, lets go and backs off
   to its ready pose.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arms'
encoders) takes out the soft wrists' sag.

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

READY = [0.0, 1.0, 1.0, -0.6, 0.0, 0.0]

CONTROL_HZ = 20
DT = 1.0 / CONTROL_HZ
I_GAIN, I_MAX = 0.08, 0.12     # joint integral: per control step, and clamp (rad)
DEPTH = 0.018                  # the tip's end this far up the nostril while circling
STANDOFF = 0.025               # ... and this far below the opening before going in
AWAY = np.array([-0.04, 0.0, 0.0])   # coming to and leaving the nose: from 4 cm further forward
PRESS = 0.0055                 # the circle's path this far outside the free path (pressing the soft nose)
CIRCLE_S = 3.2                 # one circle (s)
N_CIRCLES = 8
LIFT = 0.14                    # the swab lifted this far to clear its tube
ROLL = math.radians(-8.0)      # the grip turned this far about the swab's axis at the nose (wrist range)
R_TUBE = np.column_stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])   # TCP at the tube: x, y, z


def nose_rot(roll: float = ROLL) -> np.ndarray:
    """TCP rotation with the swab along the nostrils' axis (tip up), the fingers on the cap's flats."""
    u = g.u_station()
    w = np.array([math.sin(g.THETA), 0.0, -math.cos(g.THETA)])
    z = math.cos(roll) * w + math.sin(roll) * np.cross(u, w)
    return np.column_stack([np.cross(u, z), u, z])


def tcp_at_nostril(side: str, depth: float, offset=(0.0, 0.0), roll: float = ROLL):
    """TCP pose with the tip's end ``depth`` up nostril ``side``'s axis, moved ``offset`` (along F, along y)."""
    r = nose_rot(roll)
    hp, hr = g.head_pose()
    npos, nrot = g.nose_pose(hp, hr)
    end = npos + nrot @ np.array([depth, g.NOSTRIL_Y[side] + offset[1], offset[0]])
    pos = end - r @ g.GRIP_IN_S @ np.array([g.TIP_END, 0.0, 0.0])
    return g.tcp_for_swab(pos, r @ g.GRIP_IN_S)


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

    def solve(self, pos, rot, seeds=()):
        from yam_arena import ik

        best = None
        for s in [self.q, *seeds, *(random_seeds() if seeds else ())]:
            q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=np.asarray(s, float), base_pos=self.base,
                                    iters=120)
            key = err + 0.05 * rerr
            if best is None or key < best[0] - 1e-6:
                best = (key, q, err, rerr)
            if err < 2e-4 and rerr < 1e-3:
                break
        return best[1], best[2], best[3]

    def aim(self, pos, rot) -> None:
        q, err, rerr = self.solve(pos, rot)
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

    def swing(self, pos, rot, seconds: float, seeds=()):
        """Generator: a joint-space move to the pose ``pos``/``rot`` (for turning the swab over)."""
        from yam_arena import ik

        q1, err, rerr = self.solve(pos, rot, seeds)
        self.worst_mm = max(self.worst_mm, err * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(rerr))
        q0 = self.q.copy()
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q = q0 + s * (q1 - q0)
            self.pos, self.rot = ik.tcp_pose(self.q, self.base)
            yield
        self.pos, self.rot = np.asarray(pos, float), np.asarray(rot, float)

    def set_grip(self, grip: float, seconds: float = 1.0):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


_SEEDS: list = []


def random_seeds(n: int = 30) -> list:
    """Fixed pseudo-random joint seeds for the joint-space moves' IK (the same every run)."""
    if not _SEEDS:
        from yam_arena import ik

        rng = np.random.default_rng(7)
        _SEEDS.extend(ik.LOWER + (ik.UPPER - ik.LOWER) * rng.random(6) for _ in range(n))
    return _SEEDS


NOSE_SEEDS = ([1.47, 1.9, 1.15, -0.85, 1.13, -1.55], [1.5, 2.0, 1.2, -1.0, 1.1, -1.4])


class Script:
    """The demo as per-step arm commands. ``swab`` is where the script expects the swab (S pose); ``phase`` and
    ``side`` say what it is doing (for the tests)."""

    def __init__(self, circles: int = N_CIRCLES):
        self.left, self.right = Arm("left"), Arm("right")
        self.n_circles = circles
        self.swab = (g.SWAB_START.copy(), g.R_SWAB0.copy())
        self.holding = False
        self.phase = "start"
        self.side = None

    def _carry(self, gen):
        for _ in gen:
            if self.holding:
                self.swab = g.swab_from_tcp(self.right.pos, self.right.rot)
            yield

    def _nostril(self, side: str):
        A = self.right
        a_f = g.SEMI_F - g.TIP[0] + PRESS
        a_y = g.SEMI_Y - g.TIP[0] + PRESS
        self.side = side
        self.phase = "in"
        p, r = tcp_at_nostril(side, -STANDOFF)
        yield from self._carry(A.move(p, r, 1.5))
        p, r = tcp_at_nostril(side, DEPTH)
        yield from self._carry(A.move(p, r, 2.5))
        self.phase = "circles"
        p, r = tcp_at_nostril(side, DEPTH, (a_f, 0.0))
        yield from self._carry(A.move(p, r, 0.8))
        yield from wait(0.5)
        n = round(CIRCLE_S * CONTROL_HZ)
        for k in range(1, self.n_circles * n + 1):
            phi = 2 * math.pi * k / n
            p, r = tcp_at_nostril(side, DEPTH, (a_f * math.cos(phi), a_y * math.sin(phi)))
            A.aim(p, r)
            self.swab = g.swab_from_tcp(A.pos, A.rot)
            yield
        p, r = tcp_at_nostril(side, DEPTH)
        yield from self._carry(A.move(p, r, 0.8))
        self.phase = "out"
        p, r = tcp_at_nostril(side, -STANDOFF)
        yield from self._carry(A.move(p, r, 2.0))
        self.side = None

    def swab_it(self):
        A = self.right
        tcp0, rot0 = g.tcp_for_swab(g.SWAB_START, g.R_SWAB0)
        back = np.array([-0.03, 0.0, 0.0])     # the arm can't reach much further back with its fingers level
        up = np.array([0.0, 0.0, 1.0])
        self.phase = "pick"
        yield from A.swing(tcp0 + back + 0.06 * up, rot0, 3.0, (A.q,))
        yield from A.move(tcp0 + back, rot0, 1.5)
        yield from A.move(tcp0, rot0, 1.5)
        yield from A.set_grip(0.0, 1.0)
        self.holding = True
        yield from wait(0.5)
        self.phase = "lift"
        yield from self._carry(A.move(tcp0 + LIFT * up, rot0, 3.0))
        self.phase = "swing"
        p, r = tcp_at_nostril("left", -STANDOFF)
        yield from self._carry(A.swing(p + AWAY, r, 4.0, NOSE_SEEDS))
        for side in ("left", "right"):
            yield from self._nostril(side)
        self.phase = "back"
        p, r = tcp_at_nostril("right", -STANDOFF)
        yield from self._carry(A.move(p + AWAY, r, 1.2))
        yield from self._carry(A.swing(tcp0 + LIFT * up, rot0, 4.0, (A.q,)))
        self.phase = "cap"
        yield from self._carry(A.move(tcp0 + 0.03 * up, rot0, 3.0))
        yield from self._carry(A.move(tcp0 + 0.002 * up, rot0, 2.5))
        yield from self._carry(A.move(tcp0 - 0.003 * up, rot0, 1.0))   # press the cap home
        yield from wait(0.3)
        yield from A.set_grip(1.0, 1.0)
        self.holding = False
        self.swab = (g.SWAB_START.copy(), g.R_SWAB0.copy())
        self.phase = "away"
        yield from wait(0.3)
        yield from A.move(tcp0 + 0.06 * up, rot0, 1.5)
        yield from A.move(tcp0 + back + 0.06 * up, rot0, 1.0)
        yield from A.swing(A.home[0], A.home[1], 2.5, (READY,))
        self.phase = "done"

    def steps(self):
        yield from self.swab_it()


def plan(max_steps: int = 3000, circles: int = N_CIRCLES) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip and TCP pose, and where the script
    expects the swab."""
    s = Script(circles)
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "swab": (s.swab[0].copy(), s.swab[1].copy()), "holding": s.holding, "phase": s.phase,
                    "side": s.side})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in (("left", s.left), ("right", s.right))}})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 3000, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 3.0, circles: int = N_CIRCLES) -> dict:
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
    script = Script(circles)
    gen = script.steps()
    integ = {"left": np.zeros(6), "right": np.zeros(6)}
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def command(side, arm):
        q = arm_joints(emb._env, side, ARM_JOINTS)
        err = arm.q - q
        integ[side] = np.clip(integ[side] + I_GAIN * err, -I_MAX, I_MAX)
        return np.r_[arm.q + integ[side], arm.grip]

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    max_disp = {"left": 0.0, "right": 0.0}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(Action(data=pack(command("left", script.left), command("right", script.right))))
        steps += 1
        info = res.info
        snap(res.observation)
        S = task._STATE
        if S["side"]:
            max_disp[S["side"]] = max(max_disp[S["side"]], S["disp_mm"])
        if steps % 20 == 0:
            print(f"step {steps} {script.phase}/{script.side} score {info.get('score')} circles {S['circles']} "
                  f"turned {[round(v / 6.283, 2) for v in S['turned'].values()]} side {S['side']} depth "
                  f"{S['depth']} disp {S['disp_mm']} wall {S['on_wall']} neck {round(math.degrees(S['pitch']), 2)} "
                  f"{round(math.degrees(S['yaw']), 2)} seated {S['seated']} released {S['released']} "
                  f"gaps {S['gaps']} why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "circles": S["circles"], "turns": {k: round(v / (2 * math.pi), 2) for k, v in S["turned"].items()},
              "max_nose_disp_mm": max_disp, "max_neck_deg": round(S["max_neck_deg"], 2),
              "max_depth_mm": round(S["max_depth"] * 1000, 1), "seated": S["seated"], "released": S["released"],
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
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--circles", type=int, default=N_CIRCLES)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height,
        circles=a.circles)
