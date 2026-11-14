"""Scripted full-score demo for coupling both air lines, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

It works from the fixed, known layout (geometry.py) and the arms' own joint encoders only. It never reads where the
glad hands are, or the seal rings' travel, from the simulator (the task's monitor state is only printed, as a log).

Both arms work at once, each on its own line (left arm blue, right arm red), with the same moves:

1. Open, from above, it takes the parked tractor glad hand by its back block (fingers across it, the TCP on the seal
   axis), closes and lifts it off its stand.
2. It carries it over the trailer glad hand of its colour, face down and turned 90 degrees to it (the entry angle,
   which is how it was parked), and lowers it until the seals touch.
3. It presses the seals together (aiming past the ring's full travel; the faces' stop takes it) and turns the glad
   hand 90 degrees about the seal axis, a little past the stop, so the flanges slide under the lips and hit the stops.
4. It eases off to the locked height, so the ring pushes the flanges up against the lips, opens, and lifts away.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arm's own
encoders) takes out the sag from carrying the glad hand and its hose; it is frozen while pressing and turning, so the
press force comes from the arm's own stiffness (about 0.4 N/mm here) and the commanded overshoot.

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
GRASP_D = 0.031            # TCP this far behind the seal face, on the seal axis (pad tips 18 mm behind the face)
GRIP_K = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # TCP axes in G: fingers close along G y,
#                                                                          approach along G z (into the face)
HOVER_S = 0.035            # face separation while coming over the trailer glad hand
TOUCH_S = g.S_REST + 0.002  # just above the seals touching
PRESS_OVER = 0.040         # command the press this far past the ring's full travel (about 16 N from the arm)
TWIST_OVER = math.radians(12)   # and the turn this far past the stop (the wrist lags about 10 degrees against
#                                 the lips' friction; the stop takes the rest)
EASE_S = g.S_LOCK + 0.003  # then back to just above the locked separation
GRIP_HOLD = 0.036 / 0.094  # fingers commanded to a 36 mm gap on the 44 mm back block: about 20 N each, short of
#                            the drive's limit, so the grip stays stiff sideways (fully closed, both finger drives
#                            saturate and the part can slide sideways between them)
I_GAIN, I_MAX = 0.08, 0.25
SIDE_OF = {"blue": "left", "red": "right"}
WRIST = {"blue": 0.3, "red": 0.1}   # wrist roll (rad) to approach the parked glad hand with: the turn then winds it
#                                     toward the other end of its range (blue to about -1.0, red to about -1.9)


GRIP = {"blue": GRIP_K, "red": GRIP_K @ g.rot_z(math.pi)}   # the right hand takes the red one the other way round
#                                                               (the gripper is symmetric), for its wrist's range


def tcp_for(pos, rot, colour: str = "blue"):
    """TCP pose holding a glad hand whose G frame is at (pos, rot)."""
    rot = np.asarray(rot, float)
    return np.asarray(pos, float) + rot @ np.array([0.0, 0.0, -GRASP_D]), rot @ GRIP[colour]


def glad_for(tcp, rot, colour: str = "blue"):
    r = np.asarray(rot, float) @ GRIP[colour].T
    return np.asarray(tcp, float) - r @ np.array([0.0, 0.0, -GRASP_D]), r


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
        self.freeze = False

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

    def goto(self, pos, rot, seconds: float, wrist: float, grip: float | None = None):
        """Generator: a joint-space move to ``pos``/``rot``, solved from a seed whose wrist roll is ``wrist`` (rad), so
        the arm arrives on the wrist branch that leaves room for the turn to come."""
        from yam_arena import ik

        seed = self.q.copy()
        seed[5] = wrist
        q1, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=seed, base_pos=self.base, iters=300)
        self.worst_mm = max(self.worst_mm, err * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(rerr))
        q0, g0 = self.q.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.q, self.grip = q0 + s * (q1 - q0), g0 + s * (g1 - g0)
            self.pos, self.rot = ik.tcp_pose(self.q, self.base)
            yield
        self.pos, self.rot = np.asarray(pos, float), np.asarray(rot, float)

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


class Script:
    """The demo as per-step arm commands. ``glad`` holds the glad hand poses the script expects (for the tests)."""

    def __init__(self):
        self.arms = {"left": Arm("left"), "right": Arm("right")}
        self.glad = {c: g.park_pose(c) for c in g.COLOURS}
        self.holding = {c: False for c in g.COLOURS}
        self.mark: dict = {}

    def couple(self, colour: str):
        A = self.arms[SIDE_OF[colour]]
        park, r_park = g.park_pose(colour)
        tcp, rot = tcp_for(park, r_park, colour)
        up = np.array([0.0, 0.0, 1.0])

        def carry(pos, rot_, secs):
            for _ in A.move(pos, rot_, secs):
                if self.holding[colour]:
                    self.glad[colour] = glad_for(A.pos, A.rot, colour)
                yield

        def at(phi, s):
            return tcp_for(*g.tractor_pose(colour, phi, s), colour)

        yield from A.goto(tcp + 0.08 * up, rot, 2.5, WRIST[colour], grip=1.0)
        yield from A.move(tcp + 0.012 * up, rot, 1.2)
        yield from A.move(tcp, rot, 1.0)
        yield from wait(1.0)                 # settle on the grasp before closing (the integral takes out the lag)
        yield from A.set_grip(GRIP_HOLD, 1.0)
        yield from wait(0.5)
        self.holding[colour] = True
        yield from carry(tcp + 0.06 * up, rot, 1.2)
        yield from carry(*at(g.ENTRY, HOVER_S), 3.0)
        yield from wait(1.5)                 # let the arm settle over the seal axis (the integral takes out the lag)
        yield from carry(*at(g.ENTRY, TOUCH_S), 1.5)
        yield from wait(1.0)
        self.mark[f"{colour}_press"] = True
        A.freeze = True
        yield from carry(*at(g.ENTRY, g.S_MIN - PRESS_OVER), 1.5)
        yield from wait(0.5)
        self.mark[f"{colour}_twist"] = True
        yield from carry(*at(TWIST_OVER, g.S_MIN - PRESS_OVER), 3.0)
        yield from wait(0.5)
        self.mark[f"{colour}_ease"] = True
        yield from carry(*at(TWIST_OVER, EASE_S), 1.2)
        yield from wait(0.5)
        self.glad[colour] = g.tractor_pose(colour, 0.0, g.S_LOCK)
        yield from A.set_grip(1.0, 1.0)
        self.holding[colour] = False
        yield from wait(0.3)
        self.mark[f"{colour}_released"] = True
        yield from A.move(A.pos + 0.07 * up, A.rot, 1.5)
        A.freeze = False
        yield from A.move(A.home[0], A.home[1], 2.5)

    def steps(self):
        yield from together(self.couple("blue"), self.couple("red"))


def plan(max_steps: int = 2400) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, freeze flag, and the glad hand
    poses the script assumes."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({**{k: (a.q.copy(), a.grip, a.pos.copy(), a.rot.copy(), a.freeze) for k, a in s.arms.items()},
                    "glad": {c: (p.copy(), r.copy()) for c, (p, r) in s.glad.items()},
                    "holding": dict(s.holding), "mark": dict(s.mark)})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in s.arms.items()}})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------------
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
        # proprioceptive correction: the arm's measured joints toward the planned ones (held while pressing)
        q = arm_joints(emb._env, side, ARM_JOINTS)
        if not arm.freeze:
            integ[side] = np.clip(integ[side] + I_GAIN * (arm.q - q), -I_MAX, I_MAX)
        return np.r_[arm.q + integ[side], arm.grip]

    snap(obs)
    from yam_arena.scene import object_states as _os0
    _st = _os0(emb._env, ["trailer", "tender"])
    print("trailer root", np.round(_st["trailer"].pos, 4).tolist(), np.round(_st["trailer"].quat_xyzw, 4).tolist(),
          {k: np.round(v[0], 4).tolist() for k, v in _st["trailer"].links.items()},
          "tender root", np.round(_st["tender"].pos, 4).tolist(), {k: np.round(v[0], 4).tolist()
                                                                      for k, v in _st["tender"].links.items()}, flush=True)
    steps, ended, success, info = 0, None, False, {}
    seen: set = set()
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(Action(data=pack(command("left", script.arms["left"]), command("right", script.arms["right"]))))
        steps += 1
        info = res.info
        snap(res.observation)
        new = [k for k in script.mark if k not in seen]
        seen.update(new)
        if steps % 20 == 0 or new or any(a.freeze for a in script.arms.values()) and steps % 4 == 0:      # a log only: the script never uses it
            S = task._STATE
            from yam_arena import ik as _ik
            from yam_arena.scene import object_states as _os
            links = _os(emb._env, ["tender"])["tender"].links
            rel = {}
            for c in ("blue", "red"):      # where each glad hand sits in its gripper (mm, TCP frame), a log only
                a = script.arms[SIDE_OF[c]]
                tcp_m, rot_m = _ik.tcp_pose(arm_joints(emb._env, a.side, ARM_JOINTS), a.base)
                lp, lq = links[f"{c}_gladhand"]
                gp = np.asarray(lp) - task._rot(lq) @ g.PORT
                rel[c] = np.round(rot_m.T @ (gp - tcp_m) * 1000, 1).tolist()
            lag = {k: np.round((_ik.tcp_pose(arm_joints(emb._env, k, ARM_JOINTS), a.base)[0] - a.pos) * 1000, 1).tolist()
                   for k, a in script.arms.items()}
            tj = _os(emb._env, ["tender"])["tender"].joints
            print(f"step {steps} in_grip {rel} tcp_lag_mm {lag} integ {[np.round(v, 2).tolist() for v in integ.values()]} "
                  f"chain {[round(v, 2) for v in tj.values()]}", flush=True)
            print(f"step {steps} {new} score {info.get('score')} level {S.get('level')} {S.get('info')} "
                  f"clear {S.get('clear')} still {S.get('still')} why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "level": S.get("level"), "info": S.get("info"),
              "worst_ik_mm": {k: round(a.worst_mm, 2) for k, a in script.arms.items()}}
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
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
