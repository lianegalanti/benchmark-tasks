"""Scripted full-score demo for assembling the food processor, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 3600] [--no-record]

It is open-loop from the fixed, known layout (geometry.py), corrected only by each arm's own joint encoders. It never
reads the bowl's or the lid's pose, or their joints, from the simulator (the task's monitor state is only printed, as
a log).

1. Both arms take the bowl: the right hand by its handle's upright bar, the left by pinching the rim opposite (the
   grips on a diameter through its centre of mass). They lift it 19 cm, carry it over the base and lower its centre
   tube over the drive shaft until its floor sits on the boss (its lugs drop between the ledges). Together they turn
   it clockwise about the shaft, aiming 8 degrees past the stop: the lugs slide under the ledges and the detent
   clicks in. Both let go and back away.
2. The right arm takes the lid by its feed tube, lifts it over the bowl and lowers its skirt over the rim (its lugs
   drop between the rim's ledges). It lets go briefly so the lid settles flat, takes the tube again and turns it
   clockwise while holding it 3 mm down on the rim, aiming 15 degrees past the stop: the lugs slide under the
   ledges, the detent clicks in, and the interlock tab rides over the handle and pushes the switch rod down. It lets
   go and both arms go back to their ready pose.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arms' encoders)
takes out the sag from carrying the parts.

:func:`plan` runs the same script without Isaac Sim, for the tests (reach, joint limits, clearances, timing).
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
TILT = 50.0 * g.DEG           # the feed-tube grip's approach, tilted forward from straight down (reach 34 cm up)
RIM_TILT = 45.0 * g.DEG       # the rim grips' (keeps the gripper's body above the rim)
RIM_GRIP_ANGLE = 62.0         # degrees, bowl frame: the left hand pinches the rim here (station 90 at the start), ...
HANDLE_TILT = 60.0 * g.DEG    # ... and the right hand takes the handle's upright bar across its 30 mm width, approach
HANDLE_Z = 0.060              # tilted in toward the bowl, its TCP this high on the bar (bowl frame). The two grips lie
#                               on a diameter through the bowl's centre of mass. (Pinching the 3 mm wall alone gives
#                               only about 7.5 N of grip; the handle bar, like the feed tube, gets the full 50 N.)
RIM_DOWN = 0.012              # TCP this far below the rim (pads 37 mm on the wall)
RIM_MID = (g.BOWL_R_IN + g.BOWL_R_OUT) / 2
TUBE_UP = 0.080               # TCP this far above the lid's underside on the feed tube (its top 10.4 cm up)
SHAFT_CLEAR = 0.012           # the bowl's foot this far over the shaft's tip while it is carried
SKIRT_CLEAR = 0.014           # the lid's lugs this far over the rim while it is carried
OVERTURN = 8.0                # degrees: each turn aims this far past locked (the stops take it; the arms lag the
#                               detents by up to 7 degrees)
LID_PRESS = 0.003             # m: the lid is turned with the arm aiming this far into the rim (the arm lifted it
#                               otherwise, and a lug caught the end of a ledge)
LID_OVERTURN = 15.0           # degrees: the lid's turn aims this far past locked (the stops take it)
LID_DROP = 31.0               # degrees: the lid is set on the rim turned this far from locked (3 degrees more than
#                               it stands on the table), so its lugs clear the ledges even if the bowl is 2.5 short
I_GAIN, I_MAX = 0.08, 0.25


def grip_rot(yaw_deg: float, tilt: float = TILT) -> np.ndarray:
    """Fingers closing along the yawed y axis, approach tilted ``tilt`` toward the yawed +x."""
    from yam_arena.demo import grip_rotation

    return g.rot_z(yaw_deg * g.DEG) @ g.rot_y(-tilt) @ grip_rotation(0.0)


def bowl_grip(side: str, origin, yaw_deg: float):
    """TCP pose of the left hand's rim grip or the right hand's handle grip, for the bowl at ``origin`` (W origin,
    station) turned ``yaw_deg``."""
    rz = g.rot_z(yaw_deg * g.DEG)
    if side == "left":
        a = RIM_GRIP_ANGLE
        p_w = np.array([RIM_MID * math.cos(a * g.DEG), RIM_MID * math.sin(a * g.DEG), g.BOWL_H - RIM_DOWN])
        # fingers closing along the radius, approach tilted along the rim (clockwise seen from above)
        return np.asarray(origin, float) + rz @ p_w, grip_rot(a + yaw_deg - 90.0, RIM_TILT)
    a = g.HANDLE_ANGLE
    r = sum(g.HANDLE_GRIP_R) / 2
    p_w = np.array([r * math.cos(a * g.DEG), r * math.sin(a * g.DEG), HANDLE_Z])
    # fingers closing across the bar (along the rim's tangent), approach tilted in toward the bowl's axis
    return np.asarray(origin, float) + rz @ p_w, grip_rot(a + yaw_deg + 180.0, HANDLE_TILT)


def lid_grip(origin, yaw_deg: float):
    """TCP pose of the feed-tube grip for the lid at ``origin`` (L origin, station) turned ``yaw_deg``."""
    p_l = g.feed_tube_centre_L(TUBE_UP)
    return np.asarray(origin, float) + g.rot_z(yaw_deg * g.DEG) @ p_l, grip_rot(yaw_deg, TILT)


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

    def follow(self, pose_at, seconds: float):
        """Generator: follow ``pose_at(s)`` -> (pos, rot) for s from 0 to 1 (smoothstepped)."""
        from yam_arena import ik

        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            self.aim(*pose_at(ik.smoothstep(k / n)))
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


class Script:
    """The demo as per-step arm commands. ``bowl`` and ``lid`` are the poses (origin, yaw in degrees) the script
    expects, for the tests; ``mark`` names the phases reached."""

    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.flags: dict = {}
        self.bowl = (np.array([g.BOWL0[0], g.BOWL0[1], g.bowl_start_z()]), g.BOWL0_YAW)
        self.lid = (np.array([g.LID0[0], g.LID0[1], g.lid_start_z()]), g.LID0_YAW)
        self.holding = {"bowl": False, "lid": False}
        self.mark: dict = {}

    # -- the bowl (both arms) ----------------------------------------------------------------------------------------
    def bowl_side(self, side: str):
        A = self.left if side == "left" else self.right
        lead = side == "left"           # the left arm's generator updates the expected bowl pose
        o0, y0 = self.bowl
        tcp, rot = bowl_grip(side, o0, y0)
        yield from A.move(tcp + [0.0, 0.0, 0.05], rot, 2.5, grip=1.0)
        yield from A.move(tcp, rot, 1.5)
        yield from A.set_grip(0.0, 1.0)
        yield from wait(0.6)
        if lead:
            self.holding["bowl"] = True

        def carry(origin, yaw, secs):
            for _ in A.move(*bowl_grip(side, origin, yaw), seconds=secs):
                if lead:
                    self.bowl = self._bowl_from(A)
                yield

        lift = g.TABLE + g.SHAFT_TOP + g.SHAFT_CONE - g.SKIRT_Z[0] + SHAFT_CLEAR     # the foot's bottom over the tip
        bx, by = g.BASE_XY
        yield from carry((o0[0], o0[1], lift), y0, 2.5)
        yield from carry((bx, by, lift), y0, 3.5)
        yield from carry((bx, by, g.TABLE + g.SEAT_Z + 0.015), y0, 2.0)
        yield from carry((bx, by, g.TABLE + g.SEAT_Z + 0.0005), y0, 1.5)
        if lead:
            self.mark["bowl_seated"] = True
        yield from wait(0.5)
        origin = np.array([bx, by, g.TABLE + g.SEAT_Z + 0.0005])
        y1 = -OVERTURN

        def turning(s):
            return bowl_grip(side, origin, y0 + s * (y1 - y0))

        for _ in A.follow(turning, 3.5):
            if lead:
                self.bowl = (origin.copy(), max(g.STOP_TURN, self._bowl_from(A)[1]))
            yield
        if lead:
            self.mark["bowl_turned"] = True
        yield from wait(1.5)
        yield from A.set_grip(1.0, 1.0)
        if lead:
            self.holding["bowl"] = False
        yield from wait(0.3)
        if side == "right":         # back off the handle along the approach (out and up)
            yield from A.move(A.pos - 0.04 * A.rot[:, 2], A.rot, 1.0)
        # straight up until the fingertips clear the rim, then out sideways, away from the bowl
        yield from A.move(A.pos + [0.0, 0.0, 0.03], A.rot, 1.0)
        out = A.pos[:2] - np.asarray(g.BASE_XY)
        out = np.r_[out / np.linalg.norm(out), 0.0]
        yield from A.move(A.pos + 0.06 * out, A.rot, 1.0)
        if side == "left":
            self.mark["left_away"] = True
            yield from A.move(A.home[0], A.home[1], 2.0)

    def _bowl_from(self, A):
        """The bowl pose implied by the left gripper (it holds the bowl rigidly)."""
        a = RIM_GRIP_ANGLE
        p_w = np.array([RIM_MID * math.cos(a * g.DEG), RIM_MID * math.sin(a * g.DEG), g.BOWL_H - RIM_DOWN])
        r0 = grip_rot(a + g.BOWL0_YAW - 90.0, RIM_TILT)
        dyaw = math.degrees(g.yaw_of(A.rot @ r0.T))
        yaw = g.BOWL0_YAW + dyaw
        return A.pos - g.rot_z(yaw * g.DEG) @ p_w, yaw

    # -- the lid (right arm) --------------------------------------------------------------------------------------------
    def lid_side(self):
        A = self.right
        o0, y0 = self.lid
        tcp, rot = lid_grip(o0, y0)
        y_drop = LID_DROP
        yield from A.move(tcp + [0.0, 0.0, 0.06], rot, 2.5, grip=1.0)
        yield from A.move(tcp, rot, 1.5)
        yield from A.set_grip(0.0, 1.0)
        yield from wait(0.6)
        self.holding["lid"] = True

        def carry(origin, yaw, secs):
            for _ in A.move(*lid_grip(origin, yaw), seconds=secs):
                self.lid = self._lid_from(A)
                yield

        rim = g.TABLE + g.SEAT_Z + g.BOWL_H
        lift = rim - g.LID_SKIRT_Z[0] + SKIRT_CLEAR
        bx, by = g.BASE_XY
        yield from carry((o0[0], o0[1], lift), y0, 2.0)
        yield from carry((bx, by, lift), y_drop, 2.5)
        yield from carry((bx, by, rim + 0.010), y_drop, 2.0)
        yield from carry((bx, by, rim + 0.0005), y_drop, 1.5)
        # let go for a moment so the lid settles flat on the rim under its own weight, then take the tube again
        yield from A.set_grip(0.6, 0.8)
        self.holding["lid"] = False
        yield from wait(0.7)
        yield from A.set_grip(0.0, 1.0)
        self.holding["lid"] = True
        self.mark["lid_seated"] = True
        yield from wait(0.6)
        # turn it while holding it down on the rim (aiming 3 mm into it), so its lugs stay clear under the ledges
        origin = np.array([bx, by, rim - LID_PRESS])
        yield from A.move(*lid_grip(origin, y_drop), seconds=0.5)
        y1 = -LID_OVERTURN

        def turning(s):
            return lid_grip(origin, y_drop + s * (y1 - y_drop))

        for _ in A.follow(turning, 5.0):
            self.lid = (origin.copy(), max(g.STOP_TURN, self._lid_from(A)[1]))
            yield
        self.mark["lid_turned"] = True
        yield from wait(1.5)
        yield from A.set_grip(1.0, 1.0)
        self.holding["lid"] = False
        yield from wait(0.3)
        # slide the open fingers back off the feed tube, level, along its 70 mm side
        h = np.r_[A.rot[:2, 2], 0.0]
        yield from A.move(A.pos - 0.075 * h / np.linalg.norm(h), A.rot, 1.5)
        self.mark["right_away"] = True
        yield from A.move(A.home[0], A.home[1], 2.0)

    def _lid_from(self, A):
        p_l = np.array([*g.feed_tube_centre_L()[:2], TUBE_UP])
        dyaw = math.degrees(g.yaw_of(A.rot @ grip_rot(0.0, TILT).T))
        return A.pos - g.rot_z(dyaw * g.DEG) @ p_l, dyaw

    def right_side(self):
        yield from self.bowl_side("right")
        yield from self.lid_side()

    def steps(self):
        yield from together(self.bowl_side("left"), self.right_side())


def plan(max_steps: int = 3600) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, and the bowl and lid poses
    the script assumes."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "bowl": (s.bowl[0].copy(), float(s.bowl[1])), "lid": (s.lid[0].copy(), float(s.lid[1])),
                    "holding": dict(s.holding), "mark": dict(s.mark)})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in (("left", s.left), ("right", s.right))}})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def _arm_joints(environment, side: str, names) -> np.ndarray:
    robot = environment.unwrapped.scene[f"{side}_robot"]
    ids = robot.find_joints(list(names), preserve_order=True)[0]
    q = robot.data.joint_pos
    return getattr(q, "torch", q)[0, ids].detach().cpu().numpy().astype(np.float64)


def run(task_path: str, out_dir: str, max_steps: int = 3600, record: bool = True, cam_width: int = 640,
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
        q = _arm_joints(emb._env, side, ARM_JOINTS)
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
        if steps % 40 == 0 or new or (100 <= steps <= 420 and steps % 10 == 0):   # a log only: never used
            S = task._STATE
            print(f"step {steps} {new} score {info.get('score')} level {S.get('level')} {S.get('info')} "
                  f"ready {S.get('ready')} clear {S.get('clear')} still {S.get('still')} why {S['why']}", flush=True)
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
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height)
