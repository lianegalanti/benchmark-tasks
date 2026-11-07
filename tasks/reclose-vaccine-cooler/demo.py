"""Scripted full-score demo for the reclose-vaccine-cooler task, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record] [--variant leave-open]

1. The left arm takes the lid off by its handle (measured pose), lifts it 4 cm and sets it down, skirt first, on the
   table to the cooler's left.
2. The right arm reaches into the cooler with its fingers closing along x, takes the orange carton by its top
   (measured pose), lifts it clear of the rim and stands it on the green tray.
3. As soon as the carton is out, the left arm takes the lid again, carries it over the cooler, lines it up on the
   measured lid pose (position and yaw) and lowers it onto the rim, then lets go. Both arms return to the ready pose.

``--variant leave-open`` skips step 3 (the lid stays on the table), to show the 0.5 score.
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

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
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

    def set_grip(self, grip: float, seconds: float = 0.8):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def wait_for(flags, key):
    while not flags[key]:
        yield


def together(*gens):
    """Run generators side by side, one step each per control step."""
    gens = [iter(gen) for gen in gens]
    while gens:
        for gen in list(gens):
            try:
                next(gen)
            except StopIteration:
                gens.remove(gen)
        if gens:
            yield


class Ctx:
    def __init__(self, emb, task):
        self.emb, self.task = emb, task
        self.left, self.right = Arm("left"), Arm("right")
        self.flags = {"lid_aside": False, "carton_out": False}

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, ["lid", "orange_carton", "white_carton"])

    def lid(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.states()["lid"]
        return np.asarray(s.pos, float), np.asarray(s.quat_xyzw, float)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.command(), self.right.command()))


def take_lid_off(ctx, g):
    """Left arm: lift the lid off by its handle and set it down on the table to the cooler's left."""
    from yam_arena.demo import grip_rotation

    L = ctx.left
    pos, quat = ctx.lid()
    rot = grip_rotation(g.yaw(quat))
    grasp = g.handle_grasp(pos, quat)
    yield from L.move(grasp + [0, 0, 0.05], rot, 2.5, grip=1.0)
    yield from L.move(grasp, rot, 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)
    yield from L.move((*grasp[:2], g.LID_CARRY_Z), rot, 1.5)
    yield from L.move((*g.LID_ASIDE_XY, g.LID_CARRY_Z), grip_rotation(0.0), 2.5)
    yield from L.move(g.ASIDE_GRASP + [0, 0, 0.003], grip_rotation(0.0), 1.5)
    yield from L.set_grip(1.0, 0.8)
    ctx.flags["lid_aside"] = True
    yield from L.move(L.pos + [0, 0, 0.05], L.rot, 1.0)


def take_carton(ctx, g):
    """Right arm: take the orange carton out by its top, fingers closing along x, and stand it on the tray."""
    from yam_arena.demo import grip_rotation

    R = ctx.right
    rot = grip_rotation(g.CARTON_YAW)
    ajar = g.CARTON_OPEN / g.GRIPPER_OPEN
    yield from wait_for(ctx.flags, "lid_aside")      # the lid passes where this arm would wait
    c = np.asarray(ctx.states()["orange_carton"].pos, float)
    grasp = np.array([c[0], c[1], c[2] + g.CARTON[2] / 2 - g.CARTON_TCP_DOWN])
    yield from R.move((*grasp[:2], g.CARTON_CARRY_Z), rot, 2.5, grip=ajar)
    yield from R.move(grasp, rot, 1.5)
    yield from R.set_grip(0.0, 1.0)
    yield from wait(0.4)
    yield from R.move((*grasp[:2], g.CARTON_CARRY_Z), rot, 1.5)
    ctx.flags["carton_out"] = True
    yield from R.move((*g.TRAY_XY, g.CARTON_CARRY_Z), rot, 2.5)
    yield from R.move(g.TRAY_PLACE, rot, 1.5)
    yield from R.set_grip(ajar, 0.8)
    yield from wait(0.3)
    yield from R.move(R.pos + [0, 0, 0.06], rot, 1.0)
    yield from R.move(*R.home, 2.5, grip=1.0)


def put_lid_back(ctx, g):
    """Left arm: take the lid again, carry it over the cooler, line it up on its measured pose and seat it."""
    from yam_arena.demo import grip_rotation

    L = ctx.left
    yield from wait_for(ctx.flags, "carton_out")
    pos, quat = ctx.lid()
    lid_yaw = g.yaw(quat)
    grasp = g.handle_grasp(pos, quat)
    yield from L.move(grasp + [0, 0, 0.05], grip_rotation(lid_yaw), 1.5, grip=1.0)
    yield from L.move(grasp, grip_rotation(lid_yaw), 1.5)
    yield from L.set_grip(0.0, 1.0)
    yield from wait(0.4)
    yield from L.move((*grasp[:2], g.LID_CARRY_Z), L.rot, 1.5)
    # carry it over the cooler; turning the grip by -lid_yaw brings the lid back square to the cooler
    yaw_cmd = 0.0
    target = np.array([*g.COOLER_XY, g.LID_CARRY_Z])
    yield from L.move(target, grip_rotation(yaw_cmd), 2.5)
    yield from wait(0.5)
    # closed loop on the measured lid pose (position and yaw) while hovering, then while lowering at 1.6 cm/s
    z = g.LID_CARRY_Z
    seat_z = g.SEATED_GRASP[2] + 0.002
    for i in range(int(8 * CONTROL_HZ)):
        p, q = ctx.lid()
        target[:2] += 2.0 * DT * (np.asarray(g.COOLER_XY) - p[:2])
        yaw_cmd -= 2.0 * DT * g.yaw(q)
        lined_up = math.hypot(*(np.asarray(g.COOLER_XY) - p[:2])) < 0.003 and abs(g.yaw(q)) < math.radians(1.5)
        # the skirt's 8 mm gap takes what is left after 2 s of lining up
        if lined_up or z < g.LID_CARRY_Z or i > 2 * CONTROL_HZ:
            z = max(seat_z, z - 0.016 * DT)
        L.aim((target[0], target[1], z), grip_rotation(yaw_cmd))
        if z <= seat_z:
            break
        yield
    yield from wait(0.5)
    yield from L.set_grip(1.0, 0.8)
    yield from wait(0.3)
    yield from L.move(L.pos + [0, 0, 0.05], L.rot, 1.0)
    yield from L.move(*L.home, 2.5, grip=1.0)


def script(ctx, g, variant: str):
    left = [take_lid_off(ctx, g)]
    if variant != "leave-open":
        left.append(put_lid_back(ctx, g))

    def left_chain():
        for gen in left:
            yield from gen
        if variant == "leave-open":
            yield from ctx.left.move(*ctx.left.home, 2.5, grip=1.0)

    yield from together(left_chain(), take_carton(ctx, g))


def nominal_waypoints():
    """The demo's moves as yam_arena Waypoints for the start layout (lid and carton where they start, the lid set
    down and taken again where the demo puts it), for the offline reach and joint-limit test."""
    import geometry as g
    from yam_arena.constants import FINGER_CLOSED, FINGER_OPEN
    from yam_arena.demo import Waypoint

    ajar = FINGER_OPEN * g.CARTON_OPEN / g.GRIPPER_OPEN
    seat, aside, cz = g.SEATED_GRASP, g.ASIDE_GRASP, g.LID_CARRY_Z
    left = [
        Waypoint(2.5, tuple(seat + [0, 0, 0.05]), FINGER_OPEN, "above handle"),
        Waypoint(1.5, tuple(seat), FINGER_OPEN, "to handle"),
        Waypoint(1.0, tuple(seat), FINGER_CLOSED, "grip lid"),
        Waypoint(1.5, (*seat[:2], cz), FINGER_CLOSED, "lift lid"),
        Waypoint(2.5, (*aside[:2], cz), FINGER_CLOSED, "carry aside"),
        Waypoint(1.5, tuple(aside + [0, 0, 0.003]), FINGER_CLOSED, "set lid down"),
        Waypoint(0.8, tuple(aside + [0, 0, 0.003]), FINGER_OPEN, "release lid"),
        Waypoint(1.0, tuple(aside + [0, 0, 0.053]), FINGER_OPEN, "up"),
        Waypoint(1.5, tuple(aside + [0, 0, 0.05]), FINGER_OPEN, "above lid"),
        Waypoint(1.5, tuple(aside), FINGER_OPEN, "to handle again"),
        Waypoint(1.0, tuple(aside), FINGER_CLOSED, "grip lid again"),
        Waypoint(1.5, (*aside[:2], cz), FINGER_CLOSED, "lift lid again"),
        Waypoint(2.5, (*seat[:2], cz), FINGER_CLOSED, "over the cooler"),
        Waypoint(2.0, tuple(seat + [0, 0, 0.002]), FINGER_CLOSED, "lower onto rim"),
        Waypoint(0.8, tuple(seat + [0, 0, 0.002]), FINGER_OPEN, "release"),
        Waypoint(1.0, tuple(seat + [0, 0, 0.052]), FINGER_OPEN, "up"),
    ]
    grasp, carry = g.carton_grasp(g.CARTON_XY["orange_carton"]), g.CARTON_CARRY_Z
    y = g.CARTON_YAW
    right = [
        Waypoint(2.5, (*grasp[:2], carry), ajar, "above carton", y),
        Waypoint(1.5, tuple(grasp), ajar, "to carton", y),
        Waypoint(1.0, tuple(grasp), FINGER_CLOSED, "grip carton", y),
        Waypoint(1.5, (*grasp[:2], carry), FINGER_CLOSED, "lift out", y),
        Waypoint(2.5, (*g.TRAY_XY, carry), FINGER_CLOSED, "to tray", y),
        Waypoint(1.5, tuple(g.TRAY_PLACE), FINGER_CLOSED, "lower", y),
        Waypoint(0.8, tuple(g.TRAY_PLACE), ajar, "release", y),
        Waypoint(1.0, tuple(g.TRAY_PLACE + [0, 0, 0.06]), ajar, "up", y),
    ]
    return {"left": left, "right": right}


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 2400, record: bool = True, variant: str = "",
        cam_width: int = 320, cam_height: int = 176, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    sys.modules.setdefault("task", task)
    import geometry as g

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    ctx = Ctx(emb, task)
    gen = script(ctx, g, variant)
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
        if steps % 50 == 0:
            S = task._STATE
            print(f"step {steps} score {info.get('score')} seated {S['seated']} delivered {S['delivered']} "
                  f"out {S['out']} lid {np.round(S.get('lid_pos', np.zeros(3)), 4)} why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "seated": bool(S["seated"]), "delivered": bool(S["delivered"]),
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
    ap.add_argument("--variant", default="", choices=["", "leave-open"])
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, variant=a.variant, cam_width=a.cam_width,
        cam_height=a.cam_height)
