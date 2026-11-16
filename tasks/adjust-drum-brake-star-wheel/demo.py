"""Scripted full-score demo for adjusting the drum brake, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 4800] [--clicks 25] [--no-record]

It works open loop from the fixed, known layout (geometry.py) and the arms' own joint readings only: it never reads
the spoon's, the drum's or the star wheel's state from the simulator. It does not stop by feel: it makes a fixed
number of lever strokes (``--clicks``, 25: one click each), in the middle of the 22-28 clicks that give a light drag
on this brake. Until the shoes are a click or two onto the drum (22 clicks), the left hand turns the drum to and fro
by a wheel stud, and the demo logs that arm's joint tracking error (its encoders) on each turn, which rises with the
drum's drag; it is printed, not used. From then on the left hand holds the drum still by the stud.

1. Right arm: picks up the spoon by its handle (a top grip, fingers across the handle), lifts it, and brings it in
   front of the adjusting slot, blade along the axle.
2. It slides the blade through the slot, tip down 14 degrees, until its end sits under the lower of the two teeth
   either side of the gap facing the slot.
3. Each stroke: it presses the handle down until the blade is level (2 degrees up), pivoting the spoon on the slot's
   bottom edge, so the blade's end lifts that tooth up past the wheel's axis, past half a tooth: the wheel clicks on
   one notch. It levels the blade, slides it back clear of the teeth, raises the handle and slides it in again under
   the next tooth.
4. Left arm, meanwhile: takes hold of the top wheel stud and turns the drum 20 degrees back and forth, until 22
   strokes are made; then it holds it.
5. After the last stroke the right arm draws the spoon out, lays it on the table where it was and lets go; the left
   hand brings the drum back, lets go of the stud and backs off. Both arms go back to their ready pose.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arms'
encoders) takes out the soft wrists' sag.

:func:`plan` runs the same script without Isaac Sim, for the tests (reach, joint limits, clearances, stroke geometry).
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
I_GAIN, I_MAX = 0.08, 0.12     # joint integral: per control step, and clamp (rad)
TURN = math.radians(20.0)      # how far the left hand turns the drum each way
TURN_UNTIL = 22                # ... while fewer strokes than this are made
STROKE_S = 2.5                 # handle down: the lever stroke (s)
SETTLE_S = 0.4                 # pause with the blade's end under the tooth before each stroke (s)
STUD_DY = 0.030                # left TCP this far outboard of the drum's face: fingertips 17 mm from it
STUD_OPEN = 0.5                # fingers half open (47 mm) round the 12 mm stud
GRIP_IN_S = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # TCP axes in S: across, -x, down


def spoon_tcp(origin, rot):
    """TCP (position, rotation) holding the spoon at S pose (origin, rot): on the handle's top face, centred, the
    fingers closing across the handle."""
    rot = np.asarray(rot, float)
    return np.asarray(origin, float) + rot @ np.array([0.0, 0.0, g.GRASP_Z]), rot @ GRIP_IN_S


def spoon_from_tcp(tcp, rot):
    rs = np.asarray(rot, float) @ GRIP_IN_S.T
    return np.asarray(tcp, float) - rs @ np.array([0.0, 0.0, g.GRASP_Z]), rs


def stroke_pose(alpha: float, d: float):
    """TCP pose with the spoon at lever angle ``alpha`` and its end ``d`` past the fulcrum (station frame)."""
    o, r = g.spoon_pose_b(alpha, d)
    return spoon_tcp(g.B0 + o, r)


def stud_pose(theta: float, dy: float = STUD_DY):
    """Left TCP pose holding the top stud with the drum at ``theta``: approach along -y (toward the drum's face),
    fingers closing across the stud, tangentially."""
    p = g.stud_world(0, theta)
    p[1] = g.B0[1] + g.FACE_Y[1] + dy
    x = -(g.rot_y(theta) @ np.array([1.0, 0.0, 0.0]))
    z = np.array([0.0, -1.0, 0.0])
    return p, np.column_stack([x, np.cross(z, x), z])


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

    def path(self, poses, seconds: float):
        """Generator: through a list of (pos, rot), evenly in time (for arcs)."""
        n = max(1, round(seconds * CONTROL_HZ))
        from yam_arena import ik

        for k in range(n):
            s = ik.smoothstep((k + 1) / n) * (len(poses) - 1)
            i = min(int(s), len(poses) - 2)
            f = s - i
            p = (1 - f) * poses[i][0] + f * poses[i + 1][0]
            self.aim(p, ik.slerp_matrix(poses[i][1], poses[i + 1][1], f))
            yield

    def set_grip(self, grip: float, seconds: float = 1.0):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def wait_for(flags: dict, key: str):
    while not flags.get(key):
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
    """The demo as per-step arm commands. ``spoon`` (S pose or None while it lies on the table untouched), ``alpha``,
    ``d``, ``clicks`` and ``drum`` are what the script expects (for the tests)."""

    def __init__(self, clicks: int = g.DEMO_CLICKS):
        self.left, self.right = Arm("left"), Arm("right")
        self.n_clicks = clicks
        self.flags: dict = {}
        self.spoon = (g.SPOON_START.copy(), g.YAW90.copy())
        self.holding = False
        self.stroke = None        # (alpha, d) while the blade is in the slot
        self.clicks = 0
        self.drum = 0.0
        self.turns: list = []     # (step, theta) when the left hand turns the drum

    # -- the spoon (right arm) ---------------------------------------------------------------------------------------
    def _carry(self, gen):
        for _ in gen:
            if self.holding:
                self.spoon = spoon_from_tcp(self.right.pos, self.right.rot)
            yield

    def _stroke_move(self, alpha, d, seconds):
        a0, d0 = self.stroke
        n = max(1, round(seconds * CONTROL_HZ))
        from yam_arena import ik

        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            a, dd = a0 + s * (alpha - a0), d0 + s * (d - d0)
            self.stroke = (a, dd)
            self.right.aim(*stroke_pose(a, dd))
            self.spoon = spoon_from_tcp(self.right.pos, self.right.rot)
            yield

    def adjust(self):
        A = self.right
        grasp, grot = spoon_tcp(*self.spoon)
        up = np.array([0.0, 0.0, 1.0])
        yield from A.move(grasp + 0.10 * up, grot, 2.5, grip=1.0)
        yield from A.move(grasp + 0.004 * up, grot, 1.5)
        yield from A.move(grasp, grot, 0.5)
        yield from A.set_grip(0.0, 1.0)
        self.holding = True
        yield from wait(0.5)
        yield from self._carry(A.move(grasp + 0.08 * up, grot, 1.2))
        out_p, out_r = stroke_pose(g.ALPHA_LOW, -0.03)      # the blade's end 3 cm short of the slot, in line, tip down
        yield from self._carry(A.move(out_p + 0.04 * up, out_r, 2.5))
        yield from self._carry(A.move(out_p, out_r, 1.2))
        self.stroke = (g.ALPHA_LOW, -0.03)
        yield from self._stroke_move(g.ALPHA_LOW, g.D_BACK, 1.5)    # through the slot, short of the teeth
        self.flags["in_slot"] = True
        yield from self._stroke_move(g.ALPHA_LOW, g.D_ENGAGE, 0.6)  # under the lower tooth
        yield from wait(SETTLE_S)
        for _ in range(self.n_clicks):
            yield from self._stroke_move(g.ALPHA_TOP, g.D_ENGAGE, STROKE_S)  # handle down: the tooth goes up a notch
            self.clicks += 1
            yield from wait(0.2)
            yield from self._stroke_move(0.0, g.D_ENGAGE, 0.4)          # level, clear of the tooth that clicked up
            yield from self._stroke_move(0.0, g.D_BACK, 0.6)            # back off the teeth
            yield from self._stroke_move(g.ALPHA_LOW, g.D_BACK, 1.2)    # handle up
            yield from self._stroke_move(g.ALPHA_LOW, g.D_ENGAGE, 0.7)  # in under the next tooth
            yield from wait(SETTLE_S)
        yield from self._stroke_move(g.ALPHA_LOW, -0.03, 1.5)            # out of the slot
        self.stroke = None
        self.flags["out"] = True
        yield from self._carry(A.move(out_p + 0.04 * up, out_r, 1.0))
        yield from self._carry(A.move(grasp + 0.08 * up, grot, 2.5))
        yield from self._carry(A.move(grasp + 0.002 * up, grot, 1.5))
        yield from A.set_grip(1.0, 1.0)
        self.holding = False
        self.spoon = (g.SPOON_START.copy(), g.YAW90.copy())
        yield from wait(0.3)
        yield from A.move(grasp + 0.08 * up, grot, 1.2)
        yield from A.move(A.home[0], A.home[1], 2.0)

    # -- the drum (left arm) -------------------------------------------------------------------------------------------
    def turn_drum(self):
        L = self.left
        p0, r0 = stud_pose(0.0)
        back = np.array([0.0, 0.06, 0.0])
        yield from wait_for(self.flags, "in_slot")
        yield from L.move(p0 + back + [0.0, 0.0, 0.04], r0, 2.5, grip=STUD_OPEN)
        yield from L.move(p0 + back, r0, 1.0)
        yield from L.move(p0, r0, 1.5)
        yield from L.set_grip(0.0, 1.0)            # closing on the stud centres it between the fingers
        yield from wait(0.5)
        step = 0
        while not self.flags.get("out") and self.clicks < TURN_UNTIL:
            for th in (TURN, 0.0, -TURN, 0.0):
                poses = [stud_pose(t) for t in np.linspace(self.drum, th, 9)]
                start = self.drum
                for _ in L.path(poses, 1.6):
                    step += 1
                    yield
                self.drum = th
                self.turns.append((start, th))
                if (self.flags.get("out") or self.clicks >= TURN_UNTIL) and th == 0.0:
                    break
            yield from wait(0.2)
        yield from wait_for(self.flags, "out")
        yield from L.set_grip(STUD_OPEN, 0.8)
        yield from L.move(p0 + back, r0, 1.2)
        yield from L.move(p0 + back + [0.0, 0.0, 0.06], r0, 1.0)
        yield from L.move(L.home[0], L.home[1], 2.0, grip=1.0)

    def steps(self):
        yield from together(self.adjust(), self.turn_drum())


def plan(max_steps: int = 4800, clicks: int = g.DEMO_CLICKS) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, and what the script expects
    of the spoon, the stroke, the clicks and the drum."""
    s = Script(clicks)
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "spoon": (s.spoon[0].copy(), s.spoon[1].copy()), "holding": s.holding, "stroke": s.stroke,
                    "clicks": s.clicks, "drum": s.drum})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in (("left", s.left), ("right", s.right))}})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 4800, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 3.0, clicks: int = g.DEMO_CLICKS) -> dict:
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
    script = Script(clicks)
    gen = script.steps()
    integ = {"left": np.zeros(6), "right": np.zeros(6)}
    frames: dict[str, list] = {}
    feel: list = []        # per drum turn: the left arm's largest joint tracking error (rad), from its encoders
    turn_err = [0.0, 0]

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def command(side, arm):
        q = arm_joints(emb._env, side, ARM_JOINTS)
        err = arm.q - q
        if side == "left":
            turn_err[0] = max(turn_err[0], float(np.abs(err).max()))
        integ[side] = np.clip(integ[side] + I_GAIN * err, -I_MAX, I_MAX)
        return np.r_[arm.q + integ[side], arm.grip]

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        if len(script.turns) > turn_err[1]:
            feel.append((script.clicks, round(turn_err[0], 4)))
            turn_err[0], turn_err[1] = 0.0, len(script.turns)
        res = emb.step(Action(data=pack(command("left", script.left), command("right", script.right))))
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 50 == 0:
            S = task._STATE
            print(f"step {steps} score {info.get('score')} clicks {S['clicks']} (script {script.clicks}) "
                  f"torque {S['torque']:.3f} forces {S['forces']} clear_mm {S['clear_mm']} "
                  f"down {S['down']} clear {S['clear']} gaps_mm {S['gaps']} why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "clicks": S["clicks"], "script_strokes": script.clicks,
              "torque_nm": round(S["torque"], 3), "peak_torque_nm": round(S["peak_torque"], 3),
              "shoe_forces_n": S["forces"], "clearance_mm": S["clear_mm"], "feel_left_err_rad": feel,
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


CAL_CLICKS = (0, 10, 19, 20, 21, 22, 23, 25, 27, 28, 29, 30, 31, 32)
PROBE_RATE = 1.0               # N m/s: the torque probe's ramp
PROBE_MOVE = math.radians(2.0) # rad/s: the drum has broken away once it turns this fast
PROBE_MAX = 8.0                # N m


def calibrate(task_path: str, out_dir: str) -> list:
    """In Isaac Sim, with the arms parked: set the adjuster to each count in CAL_CLICKS (the star joint written a
    notch at a time), let the shoes settle, read the shoes' measured clearance and normal forces, and the drum's
    turning torque the task computes from them; then ramp a torque probe on the drum joint until the drum breaks away,
    and record that torque. Writes calibration.json."""
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    from scenekit import set_effort, set_target, write_joint

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=str(Path(task_path).resolve()), object_jitter=0, control_hz=CONTROL_HZ,
                               cam_width=320, cam_height=176)
    task = emb.task.module
    emb.reset(Scene(id="calibrate", instruction=task.INSTRUCTION))
    left, right = Arm("left"), Arm("right")
    hold = Action(data=pack(left.command(), right.command()))
    env = emb._env

    def drum():
        from yam_arena.scene import object_states

        return float(object_states(env, ["corner"])["corner"].joints["drum"])

    rows = []
    n_now = 0
    for n in CAL_CLICKS:
        while n_now < n:                     # a notch at a time, as the spoon does
            n_now += 1
            write_joint(env, "corner", "star", g.TIGHTEN * n_now * g.PITCH)
            set_target(env, "corner", "star", g.TIGHTEN * n_now * g.PITCH)
            emb.step(hold)
        for _ in range(40):
            emb.step(hold)
        S = task._STATE
        row = {"clicks": S["clicks"], "clearance_mm": S["clear_mm"], "forces_n": S["forces"],
               "torque_from_forces_nm": round(S["torque"], 3)}
        a0, prev, tau, broke = drum(), drum(), 0.0, None
        while tau < PROBE_MAX:
            tau += PROBE_RATE / CONTROL_HZ
            set_effort(env, "corner", "drum", tau)
            emb.step(hold)
            a = drum()
            if abs(a - prev) * CONTROL_HZ >= PROBE_MOVE:
                broke = tau
                break
            prev = a
        set_effort(env, "corner", "drum", 0.0)
        for _ in range(40):
            emb.step(hold)
        row["probe_breakaway_nm"] = None if broke is None else round(broke, 3)
        row["probe_turn_deg"] = round(math.degrees(drum() - a0), 1)
        row["torque_from_forces_after_nm"] = round(task._STATE["torque"], 3)
        rows.append(row)
        print("CAL", json.dumps(row), flush=True)
    (out / "calibration.json").write_text(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=4800)
    ap.add_argument("--clicks", type=int, default=g.DEMO_CLICKS)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="run the torque-probe sweep instead of the demo")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    flag = HERE / "calibrate.flag"
    if a.calibrate or (flag.exists() and flag.read_text().strip() == "on"):
        calibrate(a.task, a.out)
        sys.exit(0)
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height,
        clicks=a.clicks)
