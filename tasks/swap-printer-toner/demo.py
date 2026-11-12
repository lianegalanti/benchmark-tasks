"""Scripted full-score demo for the toner swap, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

It works from the fixed, known layout (geometry.py) and the arms' own joint readings only: it never reads where the
cartridges or the door are in the simulator.

1. Left arm: takes the old cartridge by its handle, lifts its bosses out of the seat notches, draws it out along the
   guide rails (up their 15-degree slope), carries it clear to the left and sets it down on the table.
2. Right arm, meanwhile: picks up the new cartridge from the table to its right and holds it at the rails' height.
   Once the old one is clear, it brings the new one in front of the bay, slides it in along the rails, lowers its
   bosses into the notches, lets go and backs out.
3. Left arm: pushes the open door up by its top flange, following the hinge's arc, to a few degrees from shut; the
   door, whose weight is then behind the hinge, falls shut. Both arms go back to their ready pose.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arm's own
encoders) takes out the sag from carrying a 0.55-0.7 kg cartridge on the soft wrist joints.

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
PITCH = -30.0            # deg: cartridge grips lean forward 30 degrees (approach toward +x and down)
GRASP_DZ = 0.079         # TCP above the cartridge body's centre when it holds the handle
LINE_LIFT = 0.0027       # bosses centred in the rail slots
UNSEAT = 0.0145          # lift out of the notch (the lip is 12 mm; the slot's ceiling 16-24 mm up)
OUT_BX = -0.045          # printer-frame boss x where the cartridges wait in front of the bay ...
OUT_LIFT = 0.03          # ... this far above the rails' line, well clear of the open door
ENTRY_BX = g.RAIL_FROM - g.FLARE - 0.006   # the bosses just in front of the rails' lead-in flares
PUSH_Y = 0.10            # where the left arm pushes the door (station y)
PUSH_PEN = 0.006         # the fingers press this far into where the door would be
I_GAIN, I_MAX = 0.08, 0.12   # joint integral: per control step, and clamp (rad)


def grip_rot(pitch_deg: float = PITCH) -> np.ndarray:
    """Fingers closing along world y, approach pointing down and tilted ``-pitch_deg`` toward +x."""
    from yam_arena.demo import grip_rotation
    from yam_arena.urdf import axis_angle_matrix

    return axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(pitch_deg)) @ grip_rotation(0.0)


def world(p) -> np.ndarray:
    return g.printer_to_world(p)


def grasp_tcp(centre) -> np.ndarray:
    return np.asarray(centre, float) + [0.0, 0.0, GRASP_DZ]


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

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def wait_for(flags: dict, key: str):
    while not flags.get(key):
        yield


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


class Script:
    """The demo as per-step arm commands. ``door`` is the door angle the script expects (for the tests)."""

    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.flags: dict = {}
        self.door = g.OPEN
        self.holding = {"left": None, "right": None}

    # -- the old cartridge (left arm) --------------------------------------------------------------------------------
    def remove_old(self):
        L, R = self.left, grip_rot()
        seat = world(g.SEAT_C)
        yield from L.move((0.22, 0.30, g.TABLE + 0.25), R, 1.0, grip=1.0)
        yield from L.move((0.20, 0.06, g.TABLE + 0.25), R, 1.5)               # in front of the printer
        yield from L.move(grasp_tcp(seat) + [0, 0, 0.05], R, 1.5)
        yield from L.move(grasp_tcp(seat), R, 1.5)
        yield from L.set_grip(0.0, 1.0)
        self.holding["left"] = "old"
        yield from wait(0.6)
        # bosses up out of the notches, then along the rails (up their slope) until clear of the bay
        yield from L.move(grasp_tcp(seat) + [0, 0, UNSEAT], R, 1.2)
        start = world(g.line_centre(g.LIP_X - 0.007, LINE_LIFT))
        yield from L.move(grasp_tcp(start), R, 1.0)
        yield from L.move(grasp_tcp(world(g.line_centre(ENTRY_BX, LINE_LIFT))), R, 3.0)
        out = world(g.line_centre(OUT_BX, OUT_LIFT))
        yield from L.move(grasp_tcp(out), R, 1.2)
        aside = np.array([out[0], g.TABLE_C["old"][1], out[2]])
        yield from L.move(grasp_tcp(aside), R, 3.0)
        self.flags["old_clear"] = True
        down = np.array([*g.TABLE_C["old"], g.TABLE + g.C_SIZE[2] / 2 + 0.003])
        yield from L.move(grasp_tcp(down), R, 2.0)
        yield from L.set_grip(1.0, 1.0)
        self.holding["left"] = None
        yield from wait(0.3)
        yield from L.move(grasp_tcp(down) + [0, 0, 0.06], R, 1.2)

    # -- the new cartridge (right arm) -------------------------------------------------------------------------------
    def insert_new(self):
        A, R = self.right, grip_rot()
        start = g.NEW_START
        yield from A.move(grasp_tcp(start) + [0, 0, 0.05], R, 3.0, grip=1.0)
        yield from A.move(grasp_tcp(start), R, 1.5)
        yield from A.set_grip(0.0, 1.0)
        self.holding["right"] = "new"
        yield from wait(0.6)
        out = world(g.line_centre(OUT_BX, OUT_LIFT))
        yield from A.move(grasp_tcp([start[0], start[1], out[2]]), R, 2.0)
        yield from wait_for(self.flags, "old_clear")
        yield from A.move(grasp_tcp(out), R, 3.0)
        yield from A.move(grasp_tcp(world(g.line_centre(ENTRY_BX, LINE_LIFT))), R, 1.5)
        # along the rails, down their slope, until the bosses are over the notches; then down into the seat
        over = world(g.line_centre(g.LIP_X + 0.007, LINE_LIFT))
        yield from A.move(grasp_tcp(over), R, 4.5)
        seat = world(g.SEAT_C)
        yield from A.move(grasp_tcp(seat) + [0, 0, 0.002], R, 1.2)
        yield from wait(0.3)
        yield from A.set_grip(1.0, 1.0)
        self.holding["right"] = None
        yield from wait(0.3)
        a = R[:, 2]
        yield from A.move(A.pos - 0.04 * a, R, 1.2)
        yield from A.move(A.pos + [-0.06, 0, 0.0], R, 1.0)
        self.flags["new_in"] = True
        yield from A.move((0.20, -0.30, g.TABLE + 0.24), R, 1.5)
        yield from A.move(A.home[0], A.home[1], 2.0)

    # -- the door (left arm) ---------------------------------------------------------------------------------------
    def close_door(self):
        L = self.left
        yield from wait_for(self.flags, "new_in")
        steps = [math.radians(d) for d in np.arange(math.degrees(g.OPEN), 2.0, -5.0)] + [math.radians(3.0)]
        first = push_pose(steps[0])
        n0 = door_normal(steps[0])
        # round the door's left edge, low and in front of its flange, then along it
        yield from L.move((0.12, 0.25, g.TABLE + 0.14), first[1], 1.5, grip=0.0)
        yield from L.move((0.07, 0.24, g.TABLE + 0.13), first[1], 1.0)
        yield from L.move((0.07, PUSH_Y, g.TABLE + 0.13), first[1], 1.5)
        yield from L.move(first[0] + 0.012 * n0, first[1], 1.0)
        yield from L.move(*first, 1.0)
        for th in steps[1:]:
            self.door = th
            yield from L.move(*push_pose(th), 0.45)
        yield from wait(0.4)
        n = door_normal(steps[-1])
        yield from L.move(L.pos + 0.012 * n, L.rot, 0.6)        # off the flange; the door falls shut
        self.door = 0.0
        yield from L.move(L.pos + [-0.09, 0, -0.02], L.rot, 1.2)
        yield from L.move((0.20, 0.30, g.TABLE + 0.25), L.rot, 1.5)
        yield from L.move(L.home[0], L.home[1], 2.0, grip=1.0)

    def steps(self):
        yield from together(self.remove_old(), self.insert_new())
        yield from together(self.close_door())


def door_normal(theta: float) -> np.ndarray:
    """World direction of the door flange's outer face (door-frame +z)."""
    return g.door_rot(theta) @ np.array([0.0, 0.0, 1.0])


FINGERS_CLOSED = (np.array([0.0, 0.0, -0.034]), np.array([0.048, 0.034, 0.047]))   # TCP frame: centre, half sizes
FINGERS_OPEN = (np.array([0.0, 0.0, -0.034]), np.array([0.071, 0.034, 0.047]))
HOUSING = (np.array([0.0, 0.0, -0.1075]), np.array([0.051, 0.0325, 0.037]))


def gripper_boxes(pos, rot, opening: float = 0.0):
    """The gripper as boxes (centre, rotation, half sizes), world frame: fingers (closed .. open) and housing."""
    fc, fh = FINGERS_CLOSED
    fh = fh + (FINGERS_OPEN[1] - fh) * opening
    return [(pos + rot @ fc, rot, fh), (pos + rot @ HOUSING[0], rot, HOUSING[1])]


def push_pose(theta: float):
    """(TCP, rotation) pressing on the door's top flange with the closed fingertips at door angle ``theta``: the grip
    tilts with the door (between 15 and 30 degrees forward), its tip centred over the flange, and the fingers sit
    just outside the flange's outer face except for PUSH_PEN."""
    pitch = -min(max(math.degrees(theta), 15.0), 30.0)
    rot = grip_rot(pitch)
    n = door_normal(theta)
    p = g.door_to_world((0.060, PUSH_Y, g.PANEL_H + g.DOOR_T), theta)
    p[1] = PUSH_Y
    tip = p - rot @ np.array([0.0, 0.0, 0.013])           # TCP with the tip centre on the face
    c, h = FINGERS_CLOSED
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * h + c
    heights = (tip + corners @ rot.T - p) @ n
    return tip + n * (-heights.min() - PUSH_PEN), rot


def plan(max_steps: int = 2400) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose and the door angle the
    script assumes."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "door": s.door, "holding": dict(s.holding)})
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
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(Action(data=pack(command("left", script.left), command("right", script.right))))
        steps += 1
        info = res.info
        snap(res.observation)
        if steps % 50 == 0:
            S = task._STATE
            print(f"step {steps} score {info.get('score')} door {math.degrees(S.get('door', 0)):.1f} "
                  f"level {S.get('level')} seated {S.get('seated_err')} old {S.get('old_info')} "
                  f"printer_mm {S.get('printer_off_mm')} why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "level": S.get("level"), "worst_ik_mm": {"left": round(script.left.worst_mm, 2),
                                                      "right": round(script.right.worst_mm, 2)}}
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
