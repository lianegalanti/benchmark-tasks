"""Scripted full-score demo for mounting the phone, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

It works from the fixed, known layout (geometry.py) and the arms' own joint encoders only. It never reads where the
phone is, or the holder's joint positions, from the simulator (the task's monitor state is only printed, as a log).

The holder's arms are sprung shut and nothing latches them, so one hand holds them open while the other puts the
phone in:

1. Right arm: takes the leaning phone across its long edges near its top, gripping along the phone's long axis,
   slides it up off the dashboard's edge, swings it upright and waits above and to the right of the holder.
2. Left arm, meanwhile: pointing forward and a little down, it pinches the left arm's pull tab (a fin on the arm's
   front, outside the phone's path) between its fingers and pulls it outward 16 mm; the gearing opens the right arm
   with it (pads 90 mm apart, lips 5 mm clear of the phone's edges). It keeps holding them open.
3. The right arm brings the phone over the cradle, 10 cm up, and lowers it straight down between the open arms, its
   back 2.5 mm in front of the back plate, to 1 mm above the feet, then presses it back flat onto the plate (aiming
   2.5 mm past it) so its screen is behind the arms' lips, and holds it there.
4. The left arm eases the tab back in to 1.5 mm short of where the pad meets the phone's edge, opens its fingers and
   backs away: the springs close both arms on the phone's long edges. Then the right arm opens its fingers, slides
   them up off the phone's top and both arms go back to ready.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arm's own
encoders) takes out the sag from pushing the sprung arms and carrying the phone.

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
GRIP_DOWN = 0.030          # TCP this far below the phone's top along its long axis when holding it
TAB_PITCH = math.radians(15)   # the left gripper points forward, tilted this far down
TAB_TIP_X = -0.026         # H frame: the left fingertips' centre stops here, 10 mm short of the jaw's front
TAB_TIP_Z = 0.070          # ... at the tab's mid-height
TAB_GRIP = 0.45            # the left fingers' opening (fraction of 94 mm) while they come in around the tab
TAB_LET = 0.22             # ... and when they let go of it (10 mm each side of the tab)
PULL = 0.016               # arm travel the left gripper pulls the tab out to (pads 90 mm apart; the stop is at 18)
EASE = 0.0015              # it eases the tab back to this much short of the phone's edge, then lets go
BACK = -0.0025             # phone's back this far in front of the back plate while it is lowered and clamped
PRESS = 0.0025             # then it presses the phone back to the plate, aiming this far past it (the plate stops it)
SEAT = 0.001               # phone's bottom 1 mm above the feet when the arms close
HIGH = 0.093               # phone's bottom this high above the feet while it moves over the cradle (8 mm over the arms)
I_GAIN, I_MAX = 0.08, 0.25


def tab_rot() -> np.ndarray:
    """The left gripper: approach forward (+x) tilted TAB_PITCH down, fingers closing along world y."""
    from yam_arena.demo import grip_rotation

    return g.rot_y(-(math.pi / 2 - TAB_PITCH)) @ grip_rotation(0.0)


def tab_tcp(d: float, back: float = 0.0) -> np.ndarray:
    """The left TCP (station frame) with the fingertips' centre on the left tab at arm travel ``d``, ``back`` m
    further back along the approach."""
    rot = tab_rot()
    tip = H((TAB_TIP_X, (g.TAB_Y[0] + g.TAB_Y[1]) / 2 + d, TAB_TIP_Z))
    return tip - (0.013 + back) * rot[:, 2]


def grip_rot(lean: float = 0.0) -> np.ndarray:
    """Fingers closing along world y, approach pointing down, then turned ``lean`` (rad) about +y (the phone's own
    lean: the approach stays along the phone's long axis)."""
    from yam_arena.demo import grip_rotation

    return g.rot_y(lean) @ grip_rotation(0.0)


def H(p) -> np.ndarray:
    return g.to_station(p)


def grasp_pose():
    """The right gripper's TCP and rotation taking the leaning phone: pointing straight down, fingers across the
    phone's width, the TCP on the phone's mid-plane GRIP_DOWN below its top (along the phone)."""
    c0, r0 = g.phone_start(), g.PHONE_R0
    return c0 + (g.PHONE_L / 2 - GRIP_DOWN) * (r0 @ np.array([0.0, 0.0, 1.0])), grip_rot()


def phone_in_grip():
    """The phone's pose in the TCP frame once gripped (it stays leaning 18 degrees in the fingers)."""
    tcp, rot = grasp_pose()
    return rot.T @ (g.phone_start() - tcp), rot.T @ g.PHONE_R0


def tcp_for_phone(centre, phone_rot=np.eye(3)):
    """TCP pose (position, rotation) that puts the gripped phone's centre at ``centre`` with rotation ``phone_rot``."""
    p_rel, r_rel = phone_in_grip()
    rot = np.asarray(phone_rot) @ r_rel.T
    return np.asarray(centre, float) - rot @ p_rel, rot


def phone_for_tcp(tcp, rot):
    p_rel, r_rel = phone_in_grip()
    return np.asarray(tcp, float) + np.asarray(rot) @ p_rel, np.asarray(rot) @ r_rel


# the phone upright in the cradle: centre in H for a bottom height ``z`` and back position ``back``
def cradle_centre(z: float, back: float) -> np.ndarray:
    return np.array([back - g.PHONE_T / 2, 0.0, z + g.PHONE_L / 2])


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
    """The demo as per-step arm commands. ``phone`` is the phone pose the script expects (for the tests), ``arms`` the
    holder arms' travel it expects."""

    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.flags: dict = {}
        self.phone = (g.phone_start(), g.PHONE_R0)
        self.arms = (0.0, 0.0)
        self.holding = False
        self.pinching = False
        self.pressing = False
        self.mark: dict = {}

    def _carry(self):
        """Keep the expected phone pose with the right gripper while it holds the phone."""
        if self.holding:
            c, r = phone_for_tcp(self.right.pos, self.right.rot)
            if self.pressing:       # the back plate stops the phone's back at its face
                c = c.copy()
                c[0] = min(c[0], g.HOLDER[0] - g.PHONE_T / 2)
            self.phone = (c, r)

    # -- holding the arms open (left arm) ---------------------------------------------------------------------------
    def _slide_tab(self, d0: float, d1: float, seconds: float):
        """Move the pinched tab from arm travel d0 to d1 (the arms follow it, geared)."""
        from yam_arena import ik

        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            d = d0 + ik.smoothstep(k / n) * (d1 - d0)
            self.left.aim(tab_tcp(d), tab_rot())
            self.arms = (d, d)
            yield

    def hold_open(self):
        L, R = self.left, tab_rot()
        yield from L.move(tab_tcp(0.0, 0.10) + [0.0, 0.0, 0.05], R, 2.5, grip=TAB_GRIP)
        yield from L.move(tab_tcp(0.0, 0.05), R, 1.5)
        self.mark["tab_in"] = True
        yield from L.move(tab_tcp(0.0), R, 1.5)                       # fingers either side of the tab
        yield from L.set_grip(0.0, 1.2)
        self.pinching = True
        yield from wait(0.4)
        self.mark["pull"] = True
        yield from self._slide_tab(0.0, PULL, 2.0)                   # pull the tab out: both arms open
        yield from wait(0.8)
        self.flags["open"] = True
        yield from wait_for(self.flags, "seated")
        self.mark["ease"] = True
        ease = g.PHONE_D + EASE
        yield from self._slide_tab(PULL, ease, 2.0)                 # let the arms come in, under control
        self.pinching = False
        yield from L.set_grip(TAB_LET, 0.8)                         # let go: the springs close the last 1.5 mm
        self.arms = (g.PHONE_D, g.PHONE_D)
        self.mark["let_go"] = True
        yield from L.move(tab_tcp(g.PHONE_D, 0.05), R, 1.2)
        self.flags["closed"] = True
        yield from L.move(tab_tcp(g.PHONE_D, 0.10) + [0.0, 0.05, 0.05], R, 1.2)
        yield from L.move(L.home[0], L.home[1], 2.0, grip=1.0)

    # -- the phone (right arm) -----------------------------------------------------------------------------------------
    def mount(self):
        A = self.right
        grasp, R = grasp_pose()
        yield from A.move(grasp + [-0.02, 0.0, 0.065], R, 2.5, grip=1.0)
        yield from A.move(grasp + [0.0, 0.0, 0.045], R, 1.0)
        yield from A.move(grasp, R, 1.5)
        yield from A.set_grip(0.0, 1.0)
        self.holding = True
        yield from wait(0.6)

        def carry(pos, rot, secs):
            for _ in A.move(pos, rot, secs):
                self._carry()
                yield

        # straight up (the phone's back leaves the dashboard's edge), then back toward the robot
        yield from carry(grasp + [0.0, 0.0, 0.04], R, 1.2)
        yield from carry(grasp + [-0.07, 0.0, 0.05], R, 1.5)
        # upright, above the cradle's height and to its right
        above = cradle_centre(HIGH, BACK)
        yield from carry(*tcp_for_phone(H(above) + [0.0, -0.16, 0.0]), 2.5)
        yield from wait_for(self.flags, "open")
        self.mark["over"] = True
        yield from carry(*tcp_for_phone(H(above)), 2.0)                          # over the open cradle
        self.mark["lower"] = True
        yield from carry(*tcp_for_phone(H(cradle_centre(SEAT, BACK))), 2.5)      # straight down between the arms
        self.mark["press"] = True
        self.pressing = True
        yield from carry(*tcp_for_phone(H(cradle_centre(SEAT, PRESS))), 1.0)     # back against the plate
        yield from wait(0.4)
        self.mark["seated"] = True
        self.flags["seated"] = True
        yield from wait_for(self.flags, "closed")
        yield from wait(0.5)
        yield from A.set_grip(1.0, 1.0)
        self.holding = False
        yield from wait(0.3)
        self.mark["release"] = True
        a = A.rot[:, 2]
        yield from A.move(A.pos - 0.07 * a, A.rot, 1.5)            # fingers back up off the phone's top
        yield from A.move(A.pos + [-0.04, -0.10, 0.02], A.rot, 1.2)
        yield from A.move(A.home[0], A.home[1], 2.0)

    def steps(self):
        yield from together(self.hold_open(), self.mount())


def plan(max_steps: int = 2400) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, and the phone pose and holder
    arm travel the script assumes."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "phone": (s.phone[0].copy(), s.phone[1].copy()), "arms": s.arms, "holding": s.holding,
                    "pinching": s.pinching, "pressing": s.pressing,
                    "mark": dict(s.mark)})
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
        if steps % 40 == 0 or new:      # a log only: the script never uses it
            S = task._STATE
            from yam_arena import ik as _ik
            lag = {k: np.round((_ik.tcp_pose(arm_joints(emb._env, k, ARM_JOINTS), a.base)[0] - a.pos) * 1000, 1).tolist()
                   for k, a in (("left", script.left), ("right", script.right))}
            print(f"step {steps} {new} tcp_lag_mm {lag} score {info.get('score')} level {S.get('level')} "
                  f"{S.get('info')} clear {S.get('clear')} still {S.get('still')} why {S['why']}", flush=True)
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
