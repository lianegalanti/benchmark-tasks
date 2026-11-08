"""Scripted demo for the hang-frame-level task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--variant fix|naive] [--max-steps 2400] [--no-record]

Default: both arms play the task file's ``demo()`` (lift the frame out of its stand, hang it from the notch above its
centre of mass, let go, back away). Then, closed loop on the measured frame pose, the demo checks how it hangs and, if
it is more than 1 degree off level, corrects it (below), up to three times.

``--variant fix`` hangs the frame from the middle notch, as someone who doesn't know about the medal would, and
relies on the correction: it reads the frame's tilt, estimates how many teeth to move it (taking the centre of mass
to be at the frame's centre height, so it may take two passes), takes the tilted top rail again, lifts the frame off
the pin, turns it level, moves it along the teeth and lowers it back on.
``--variant naive`` hangs it from the middle notch and stops: the frame swings 14 degrees crooked onto its corner on
the table, which scores 0.25.

Writes one MP4 per camera, setup.png (the first top-camera frame) and result.json to the output folder.
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


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it."""

    def __init__(self, side: str, q, grip: float):
        from yam_arena import ik, station

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q, self.grip = np.asarray(q, float), grip
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.worst_mm = 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=80)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), np.asarray(rot, float)
        self.worst_mm = max(self.worst_mm, err * 1000)

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

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def together(*gens):
    """Run generators side by side, one step each per control step."""
    gens = [iter(x) for x in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def grasp_roll(side: str, frame_roll: float) -> float:
    """The world roll at which an arm takes the tilted top rail. The pads only clamp the rail's front and back faces,
    so they may sit at any roll to it; each arm takes it at a roll it can reach (the left arm turns its wrist one way,
    the right arm the other), and keeps that grasp while the frame is turned level."""
    return min(frame_roll, 0.0) if side == "left" else max(frame_roll, 0.0)


def arm_targets(task, g, centre, frame_roll: float, side: str, offset: float = 0.0, above: float = 0.0,
                out: float = 0.0):
    """Tool point and rotation for one arm holding the frame (centre, roll about x) by its top rail, the pads turned
    ``offset`` about x from the frame; ``above`` (along the frame's up) and ``out`` (horizontally, past the frame's
    end on that arm's side) move it off the rail."""
    from yam_arena.demo import grip_rotation

    sign = 1.0 if side == "left" else -1.0
    local = np.array([0.0, sign * g.GRIP_Y, g.H / 2 - g.GRIP_BELOW_TOP + above])
    return (np.asarray(centre) + g.rot_x(frame_roll) @ local + [0.0, sign * out, 0.0],
            g.rot_x(frame_roll + offset) @ grip_rotation(task.GRIP_YAW[side]))


REGRIP = -0.005   # the correction takes the rail 5 mm deeper than the first grasp: the raised end is near the reach limit


def correction_poses(g, pos, roll: float, target: int):
    """The frame poses (centre, roll, seconds, grip 1 open .. 0 closed, above, out) of a correction pass.

    The grippers come in from past the frame's ends, open, sliding along the rail at grasp height (the raised end is
    too high to reach from above), and grip. The frame is then turned level about the raised end's grasp, which stays
    put while the rest of the frame swings up: the notch on the pin, near the middle, rises off it. Then the frame is
    moved level until notch ``target`` is 1 cm over the pin, lowered on and let go."""
    hang = g.hang_centre(target, g.APPROACH_GAP)
    pivot_local = np.array([0.0, -math.copysign(g.GRIP_Y, roll), g.H / 2 - g.GRIP_BELOW_TOP + REGRIP])
    pivot = np.asarray(pos, float) + g.rot_x(roll) @ pivot_local
    level = pivot - pivot_local
    return [
        (pos, roll, 2.0, 1.0, REGRIP, 0.10),               # 10 cm past the grasps, fingers either side of the rail
        (pos, roll, 1.5, 1.0, REGRIP, 0.0),                # slide in along it
        (pos, roll, 1.0, 0.0, REGRIP, 0.0),                # grip
        (level, 0.0, 2.0, 0.0, REGRIP, 0.0),               # turn it level about the raised end: off the pin
        (hang + [0, 0, 0.010], 0.0, 2.0, 0.0, REGRIP, 0.0),  # the new notch 1 cm over the pin
        (hang - [0, 0, 0.001], 0.0, 1.5, 0.0, REGRIP, 0.0),  # lower it on
        (hang - [0, 0, 0.001], 0.0, 1.0, 1.0, REGRIP, 0.0),  # let go
    ]


def correction_plan(task, g, pos, rot) -> tuple[int, int, float]:
    """(notch on the pin now, notch to move to, measured roll) from the measured frame pose. The estimate takes the
    centre of mass to be at the frame's centre height (it doesn't know the medal's weight), so it under-shoots large
    tilts; another pass finishes the job."""
    local = (g.pin_points() - pos) @ rot
    inside = (local[:, 0] >= g.HANGER_X[0]) & (local[:, 0] <= g.HANGER_X[1])
    y = float(np.mean(local[inside, 1])) if inside.any() else float(local[:, 1].mean())
    now = int(round(y / g.PITCH))
    roll = g.roll(rot)
    teeth = max(1, int(round(g.rest_z_local() * math.tan(abs(roll)) / g.PITCH)))
    target = int(np.clip(now - math.copysign(teeth, roll), g.NOTCHES[0], g.NOTCHES[-1]))
    return now, target, roll


def correct(ctx, task, g, pos, roll: float, target: int):
    """Both arms: take the (tilted) top rail again, lift the frame off the pin, turn it level while moving it so that
    notch ``target`` is 1 cm above the pin, lower it on, let go and back away. The frame's pose is interpolated as one
    rigid body, so the two grasps never pull against each other."""
    from yam_arena import ik

    arms = (ctx["left"], ctx["right"])
    offsets = {a.side: grasp_roll(a.side, roll) - roll for a in arms}
    centre0, roll0, above0, out0 = np.asarray(pos, float), roll, REGRIP, 0.0
    first = True
    for centre, r, seconds, grip, above, out in correction_poses(g, pos, roll, target):
        if first:  # from where the arms backed away to past the frame's ends: a plain move
            gens = [a.move(*arm_targets(task, g, centre, r, a.side, offsets[a.side], above, out), seconds, grip)
                    for a in arms]
            yield from together(*gens)
            first = False
        else:
            n = max(1, round(seconds * CONTROL_HZ))
            g0 = {a.side: a.grip for a in arms}
            for k in range(1, n + 1):
                s = ik.smoothstep(k / n)
                c = centre0 + s * (np.asarray(centre) - centre0)
                rr, ab, ou = roll0 + s * (r - roll0), above0 + s * (above - above0), out0 + s * (out - out0)
                for a in arms:
                    a.grip = g0[a.side] + s * (grip - g0[a.side])
                    a.aim(*arm_targets(task, g, c, rr, a.side, offsets[a.side], ab, ou))
                yield
        centre0, roll0, above0, out0 = np.asarray(centre, float), r, above, out
    up = np.array([-0.02, 0, 0.035])                     # as task.demo(): over the rail, then away past its ends
    yield from together(*(a.move(a.pos + up, None, 1.0) for a in arms))
    yield from together(*(a.move(a.pos + [-0.10, 0.10 if a.side == "left" else -0.10, 0.015], None, 1.5)
                          for a in arms))


def run(task_path: str, out_dir: str, variant: str = "", max_steps: int = 2400, record: bool = True,
        cam_width: int = 640, cam_height: int = 352) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena import station
    from yam_arena.demo import ArmPlan
    from yam_arena.inspect_embodiment import yam_arena_embodiment
    from yam_arena.scene import object_states

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    import geometry as g

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    notch = g.LEVEL_NOTCH if variant == "" else 0
    plans = {side: ArmPlan(wps[:-1], station.BASES[side]) for side, wps in task.demo(notch).items()}  # no "wait"
    duration = max(p.duration for p in plans.values())
    ctx: dict = {}
    frames: dict[str, list] = {}
    log: list[dict] = []

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def script():
        k = 0
        while k * DT < duration:
            k += 1
            ctx["cmd"] = {s: p.action7(k * DT)[0] for s, p in plans.items()}
            yield
        for side, p in plans.items():
            ctx[side] = Arm(side, p.q, float(ctx["cmd"][side][6]))
        ctx["cmd"] = None
        if variant == "naive":
            return
        for _ in range(3):
            yield from wait(2.0)                       # let it settle
            f = object_states(emb._env, ["frame"])["frame"]
            pos, rot = np.asarray(f.pos, float), g.rotation(f.quat_xyzw)
            if g.tilt_deg(rot) <= 1.0 or not g.on_pin(pos, rot, tol=0.006):
                return
            now, target, roll = correction_plan(task, g, pos, rot)
            log.append({"notch": now, "roll_deg": round(math.degrees(roll), 2), "move_to": target})
            print(f"correcting: notch {now}, roll {math.degrees(roll):.1f} deg -> notch {target}", flush=True)
            yield from correct(ctx, task, g, pos, roll, target)

    def action():
        if ctx.get("cmd") is not None:
            return Action(data=pack(ctx["cmd"]["left"], ctx["cmd"]["right"]))
        return Action(data=pack(ctx["left"].command(), ctx["right"].command()))

    snap(obs)
    gen = script()
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(action())
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 40 == 0:
            m = task.MONITOR
            print(f"step {steps} score {info.get('score')} tilt {m.tilt:.2f} hanging {m.hanging} still {m.still} "
                  f"lifted {m.lifted} dropped {m.dropped}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= 3 * CONTROL_HZ:
            break
    m = task.MONITOR
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "tilt_deg": round(m.tilt, 2),
              "hanging": m.hanging, "dropped": m.dropped, "corrections": log}
    if record:
        import imageio

        for cam, fs in frames.items():
            imageio.mimsave(out / f"demo_{cam}.mp4", fs, fps=CONTROL_HZ, codec="libx264",
                            output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
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
    ap.add_argument("--variant", default="", choices=["", "fix", "naive"])
    ap.add_argument("--max-steps", type=int, default=2400)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.variant, a.max_steps, record=not a.no_record)
