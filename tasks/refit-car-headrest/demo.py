"""Scripted full-score demo for refitting the car headrest, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

It works from the fixed, known layout (geometry.py) and the arms' own joint encoders only. It never reads where the
headrest is, or the lock's joint, from the simulator (the task's monitor state is only printed, as a log).

1. Each arm comes in from outside the headrest's end, fingers wide open and pointing inward and down at 30 degrees in
   the headrest's plane, and pinches the cushion's thin end (64 mm) front to back, 32 mm above its bottom edge.
2. Together they lift it 15 mm, stand it upright while drawing it back from the seat, raise the post tips 12 mm above
   the collars and carry it forward until both posts are straight over both sleeves.
3. They lower both posts into the sleeves at once and push them down until the pawl clicks into the first notch
   (the notch's flat face would stop it there).
4. The right arm lets go of the cushion, closes its fingers and, pointing them inward and down at 45 degrees from
   outboard, presses the release button on top of the right bezel straight down to its stop. Holding the headrest by
   its left end, the left arm lowers it 45 mm, past the second and third notches, until the posts are 0.5 mm above
   the sleeves' closed ends.
5. The right arm draws back and the button springs up: the pawl springs into the lowest notch. The left arm opens, draws back along
   its approach, and both arms go back to ready.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arm's own
encoders) takes out the sag from carrying the headrest and pushing the posts and the button.

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
GRIP_IN = 0.004            # TCP this far in from the cushion's end face ...
GRIP_UP = 0.032            # ... and this far above its bottom edge (the fingertips reach 13 mm further)
APPROACH = math.radians(30)    # fingers point inward and down this far below horizontal, in the headrest's plane
OPEN = 1.0                 # fingers wide open (94 mm across the 64 mm end)
# Holding, the fingers are commanded 5 mm inside each face of the 64 mm end, not fully shut: each finger then squeezes
# with its position spring (about 25 N, under the fingers' 50 N limit), so the pair centres the cushion between them.
# Fully shut, both fingers would push at their limit and the headrest could slide between them along their travel.
HOLD = (2 * g.EAR_T - 0.010) / 0.094
BACK_OFF = 0.05            # m: come in from, and draw back to, this far along the approach
LIFT = 0.015               # m: first lift off the table
BACK_X = -0.085            # m: posts' axes this far in front of the sleeves while the headrest is stood up
ABOVE = 0.012              # m: post tips this far above the collars' tops before going down
MOUTH = 0.004              # m: tips this far into the lead-ins, then slowly down
CLICK = 0.001              # m: pushed this far past the first notch's depth (the pawl stops it 1.5 mm past)
FLOOR_GAP = 0.0005         # m: lowered with the button held until the tips are this far above the closed ends
PRESS = g.BUTTON_GAP + g.LOCK_TRAVEL + 0.005    # m: fingertips aimed this far below the button's resting top (its
                           # stop, through the latch, is 5 mm short of that; the rest takes up the arm's give)
BUTTON_FROM = 0.015        # m: the fingertips start this far above the button
I_GAIN, I_MAX = 0.08, 0.12
SIGNS = {"left": 1.0, "right": -1.0}


def grip_in_h(side: str):
    """The TCP's position and rotation in the headrest frame H for one arm: fingers closing through the cushion's
    thickness (H x), the approach in its plane, pointing inward and down by APPROACH."""
    sy = SIGNS[side]
    a = np.array([0.0, -sy * math.cos(APPROACH), -math.sin(APPROACH)])
    # the fingers close along H x; the right arm's wrist turned the other way round (its roll joint's range)
    x = np.array([sy, 0.0, 0.0])
    rot = np.column_stack([x, np.cross(a, x), a])
    pos = np.array([0.0, sy * (g.CUSH_W - GRIP_IN), g.POST_LEN + GRIP_UP])
    return pos, rot


def tcp_for(side: str, o, r):
    """Station TCP pose holding the headrest at pose (o, R) given in F."""
    p, rr = grip_in_h(side)
    return g.SEAT + np.asarray(o) + np.asarray(r) @ p, np.asarray(r) @ rr


# the button press: fingers closed, pointing inboard (+y) and down at 45 degrees, from outboard of the cushion's end
BUTTON_TILT = math.radians(45)
_A = np.array([0.0, math.cos(BUTTON_TILT), -math.sin(BUTTON_TILT)])
_X = np.array([-1.0, 0.0, 0.0])            # closing along -x (the right wrist's roll range)
BUTTON_ROT = np.column_stack([_X, np.cross(_A, _X), _A])
# the closed fingertips' lowest edge, from the TCP: 12.8 mm along the approach and 7.7 mm (half the tip's width)
# down and outboard (yam_arena's finger meshes); it meets the button's cap first
LEAD = 0.0128 * _A - 0.0077 * np.cross(_A, _X)
CAP_Y = -g.POST_Y + sum(g.CAP) / 2 - 0.002   # F y where the fingertips press: 2 mm outboard of the cap's centre


def button_tcp(press: float) -> np.ndarray:
    """Station TCP with the fingertips' lowest edge ``press`` m below the cap's resting top (negative: above it), over
    the cap at CAP_Y."""
    return g.SEAT + np.array([0.0, CAP_Y, g.BUTTON_PROUD - press]) - LEAD


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
        if err > 2e-4 or rerr > 1e-3:
            q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=q, base_pos=self.base, iters=400)
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

    def solve_near(self, pos, rot) -> np.ndarray:
        """Joints reaching ``pos``/``rot``: from the current joints and a few other seeds, the solution with the
        smallest error, nearest the current joints."""
        from yam_arena import ik

        best = None
        for seed in (self.q, [0.0, 1.0, 1.0, -0.6, 0.0, 0.0], [0.0, 1.2, 0.8, 0.5, 0.0, 1.5],
                     [0.0, 1.2, 0.8, 0.5, 0.0, -1.5], [0.0, 1.5, 1.5, -0.5, 0.0, 0.0]):
            q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=seed, base_pos=self.base, iters=400)
            ok = np.all(q >= ik.LOWER) and np.all(q <= ik.UPPER)
            key = (not ok, err > 2e-4 or rerr > 1e-3, float(np.abs(q - self.q).sum()))
            if best is None or key < best[0]:
                best = (key, q, err, rerr)
        _, q, err, rerr = best
        self.worst_mm = max(self.worst_mm, err * 1000)
        self.worst_deg = max(self.worst_deg, math.degrees(rerr))
        return q

    def joint_move(self, pos, rot, seconds: float = 1.0, grip: float | None = None):
        """Generator: a smooth move in joint space to ``pos``/``rot`` (for large turns of the wrist, in free space)."""
        from yam_arena import ik

        q0, q1, g0 = self.q.copy(), self.solve_near(pos, rot), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.q = q0 + s * (q1 - q0)
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


# -- the headrest's waypoints (F frame), from the known layout --------------------------------------------------------
def waypoints() -> dict:
    o0, _ = g.start_pose()
    lifted = o0 + [0.0, 0.0, LIFT]
    return {
        "start": (o0, g.LEAN),
        "lifted": (lifted, g.LEAN),
        "upright": (np.array([BACK_X, 0.0, lifted[2]]), 0.0),
        "raised": (np.array([BACK_X, 0.0, ABOVE]), 0.0),
        "over": (np.array([0.0, 0.0, ABOVE]), 0.0),
        "mouth": (np.array([0.0, 0.0, -MOUTH]), 0.0),
        "clicked": (np.array([0.0, 0.0, -(g.DEPTHS[0] + CLICK)]), 0.0),
        "bottom": (np.array([0.0, 0.0, -(g.FLOOR - FLOOR_GAP)]), 0.0),
    }


class Script:
    """The demo as per-step arm commands. ``pose`` is the headrest pose (o, R in F) the script expects, ``holding``
    which arms hold it, ``q`` the lock joint it expects (for the tests)."""

    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.wp = waypoints()
        o, lean = self.wp["start"]
        self.pose = (o, g.rot_y(lean))
        self.holding: tuple = ()
        self.q = self.qb = 0.0
        self.mark: dict = {}
        self.stage = "start"

    @staticmethod
    def pressed(arm) -> float:
        """How far the fingertips push the button down (to the latch's stop), from the commanded TCP."""
        lead_z = (arm.pos + LEAD)[2] - g.SEAT[2]
        return float(np.clip(g.BUTTON_PROUD - lead_z, 0.0, g.BUTTON_GAP + g.LOCK_TRAVEL))

    def carry(self, name: str, seconds: float, arms=None):
        """Move the held headrest from its current waypoint to ``name``: position and lean interpolated together."""
        from yam_arena import ik

        arms = arms or (self.left, self.right)
        p0, l0 = self.wp[self.stage]
        p1, l1 = self.wp[name]
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            o, r = np.asarray(p0) + s * (np.asarray(p1) - p0), g.rot_y(l0 + s * (l1 - l0))
            self.pose = (o, r)
            for arm in arms:
                arm.aim(*tcp_for(arm.side, o, r))
            yield
        self.stage = name

    def steps(self):
        o, r = self.pose

        def approach(arm):
            pos, rot = tcp_for(arm.side, o, r)
            a = rot[:, 2]
            yield from arm.joint_move(pos - BACK_OFF * a, rot, 3.0, grip=OPEN)
            yield from arm.move(pos, rot, 1.5)
            yield from wait(1.0)                     # let the arm settle on the grasp before closing
            yield from arm.set_grip(HOLD, 2.0)       # close slowly on the cushion's end

        yield from together(approach(self.left), approach(self.right))
        yield from wait(0.6)
        self.holding = ("left", "right")
        self.mark["grasped"] = True
        yield from self.carry("lifted", 1.0)
        yield from self.carry("upright", 2.5)
        yield from self.carry("raised", 2.0)
        yield from self.carry("over", 2.5)
        yield from wait(0.3)
        self.mark["insert"] = True
        yield from self.carry("mouth", 1.5)
        yield from self.carry("clicked", 3.0)
        yield from wait(0.5)
        self.mark["clicked"] = True

        # the right arm lets go of the cushion and goes to the button
        right = self.right
        self.holding = ("left",)
        yield from right.set_grip(OPEN, 1.0)
        yield from right.move(right.pos - BACK_OFF * right.rot[:, 2], right.rot, 1.5)
        via = button_tcp(-BUTTON_FROM) + np.array([0.0, -0.04, 0.04])
        yield from right.joint_move(via, BUTTON_ROT, 3.0, grip=0.0)
        yield from right.move(button_tcp(-BUTTON_FROM), BUTTON_ROT, 1.5)
        self.mark["press"] = True
        for _ in right.move(button_tcp(PRESS), BUTTON_ROT, 1.5):
            self.qb = self.pressed(right)
            self.q = g.latch_from_button(self.qb)
            yield
        yield from wait(1.5)

        # holding the button in, the left arm lowers the headrest to the bottom
        self.mark["lower"] = True
        yield from self.carry("bottom", 3.0, arms=(self.left,))
        yield from wait(0.7)

        # let the button up: the pawl drops into the lowest notch
        self.mark["unlock"] = True
        # back off along the approach (up and outboard), clear of the cushion's end above
        for _ in right.move(right.pos - 0.025 * right.rot[:, 2], BUTTON_ROT, 1.0):
            self.qb = self.pressed(right)
            self.q = g.latch_from_button(self.qb)
            yield
        self.q = self.qb = 0.0
        yield from wait(0.5)

        left = self.left
        self.mark["release"] = True
        self.holding = ()
        yield from left.set_grip(OPEN, 1.0)
        # once let go, the headrest settles 0.5 mm onto the closed ends and the pawl
        self.pose = (np.array([0.0, 0.0, -g.FLOOR]), np.eye(3))
        yield from wait(0.3)

        def back(arm):
            yield from arm.move(arm.pos - (BACK_OFF if arm is left else 0.02) * arm.rot[:, 2], arm.rot, 1.5)
            yield from arm.joint_move(arm.home[0], arm.home[1], 2.5)

        yield from together(back(left), back(right))


def plan(max_steps: int = 2400) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, and the headrest pose and
    lock joint the script assumes."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "pose": (s.pose[0].copy(), s.pose[1].copy()), "holding": s.holding, "q": s.q, "qb": s.qb,
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
