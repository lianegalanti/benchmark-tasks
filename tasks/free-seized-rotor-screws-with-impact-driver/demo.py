"""Scripted full-score demo for breaking the seized rotor screw loose, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 3600] [--blows 4] [--no-record]

It works open loop from the fixed, known layout (geometry.py) and the arms' own joint readings only: it never reads
the driver's, the hammer's or the screw's state from the simulator. It does not check whether a blow landed or how far
the screw turned: it strikes it a fixed number of times (``--blows``, 4: three good blows break the screw loose,
the fourth is spare).

1. Right arm: takes the impact driver out of its block by the knurled body (approaching from behind, fingers closing
   across it), lifts it clear and brings it upright over the seized screw.
2. It lowers the bit into the recess and on 6 mm further, so the arm keeps it pressed home and square. After each
   blow, while the hammer winds up, it lifts the bit 6 mm and presses it home again (a blow's rebound can lift it out).
   It feels for the recess with the arm's encoders: a bit fully home sits 3.1 mm lower than one on the head's rim, so
   if the hand stays high it lifts and tries small offsets round the spot until it drops in, keeps that offset, and
   only then lets the hammer strike.
3. Left arm, meanwhile: takes the hammer off its stand by the handle, 12 cm from the head (from behind, fingers
   closing on the top and bottom of the handle), and lifts it.
4. It brings the hammer over the driver's cap and winds up from the elbow: the elbow 0.45 rad up from where it is when
   the face rests square on the cap (the face then about 23 cm up). Each blow: the elbow's target jumps at once to
   0.30 rad past that, the other joints held, so the elbow swings the head down its arc (11 degrees off vertical at
   the cap) as fast as the joint goes and the face meets the cap square; after 0.3 s the target comes back to the
   hammer resting on the cap, for 0.3 s; then it winds up again. Four blows.
5. The driver goes back into its block and the hammer back on its stand; both hands let go and return to their
   ready pose.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arms' encoders)
takes out the soft wrists' sag; it is frozen while the bit is pressed home and while the hammer swings, so it never
winds up against a contact.

:func:`plan` runs the same script without Isaac Sim, for the tests (reach, joint limits, clearances, the blows' aim).
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
SHIFT_GAIN = 0.1               # per control step: how fast the held part's measured offset is followed
BLOWS = 4                      # per screw (three good ones break it loose)
SWING_JOINT = 2                    # the blow is swung from the elbow (joint 3): at the cap its arc runs 11 degrees off
WIND = 0.45                        # vertical. Wind-up: the elbow 0.45 rad up from where it is when the face meets the
THROUGH = 0.30                     # cap (face ~23 cm up); the blow's target: 0.30 rad past it, the other joints held
SWING_S = 0.30                     # how long the target stays there before it comes back to the cap
RAISE_Z = 0.08                     # the hammer brought over the cap this high before the wind-up
AIM = np.array([0.018, -0.012, 0.0])  # the swing is aimed this far off the cap (forward, to the right): the soft wrist
                                      # flexes in the fast swing and the face lands 18 mm short and 12 mm toward the hand
                                      # of where it is aimed (measured in Isaac Sim; a fixed calibration, not sensed)
PRESS = 0.006                  # the bit commanded this far past fully home
RESEAT_UP = 0.006              # after each blow the bit is lifted this far and pressed home again
DROP = 0.003                   # between the screws the driver is let go this far above its seat in the block
FEEL_HIGH = 0.0018             # the hand this much higher than at the first seat (encoders): the bit is on the rim
SEARCH = [(5, 0), (3, 3), (3, -3), (8, 0), (0, 3), (0, -3), (-3, 0), (6, 4), (6, -4)]   # mm along (approach, closing)
REST_S, HOLD_S, UP_S = 0.6, 0.3, 1.0

# TCP orientations (columns: the fingers' closing axis, the pads' length, the approach)
GRIP_YAW = math.radians(30.0)  # the right hand comes at the driver from behind and to the right, fingers across it
_A = np.array([math.cos(GRIP_YAW), math.sin(GRIP_YAW), 0.0])
_C = np.array([math.sin(GRIP_YAW), -math.cos(GRIP_YAW), 0.0])
DRIVER_GRIP = np.column_stack([_C, np.cross(_A, _C), _A])
HAMMER_GRIP = np.column_stack([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])   # from behind, top and bottom
TCP_IN_H = g.HAMMER_STRIKE.T @ HAMMER_GRIP         # the hand's frame on the hammer (the grip never changes)


def driver_tcp(tip) -> tuple[np.ndarray, np.ndarray]:
    """TCP holding the upright driver with its bit's tip at ``tip``."""
    return np.asarray(tip, float) + np.array([0.0, 0.0, g.GRIP_Z]), DRIVER_GRIP


def driver_from_tcp(tcp) -> np.ndarray:
    return np.asarray(tcp, float) - np.array([0.0, 0.0, g.GRIP_Z])


def hammer_tcp(origin, rot_h) -> tuple[np.ndarray, np.ndarray]:
    rot_h = np.asarray(rot_h, float)
    return np.asarray(origin, float) + rot_h @ np.array([0.0, g.HAMMER_GRIP_Y, 0.0]), rot_h @ TCP_IN_H


def hammer_from_tcp(tcp, rot):
    rot_h = np.asarray(rot, float) @ TCP_IN_H.T
    return np.asarray(tcp, float) - rot_h @ np.array([0.0, g.HAMMER_GRIP_Y, 0.0]), rot_h


def strike_tcp(cap, above: float = 0.0, roll: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """TCP with the hammer face-down and level, its face's centre ``above`` over the cap's centre; then the hand raised
    by ``above`` too and the wrist rolled by ``roll`` about the hand's approach axis (positive: head up)."""
    face = np.asarray(cap, float)
    origin = face - g.HAMMER_STRIKE @ np.array([0.0, 0.0, g.FACE_TIP])
    tcp, rot = hammer_tcp(origin, g.HAMMER_STRIKE)
    return tcp + np.array([0.0, 0.0, above]), g.rot_x(-roll) @ rot


def seated_tip(screw: str) -> np.ndarray:
    return g.screw_world(screw) - np.array([0.0, 0.0, g.SEAT_DEPTH])


def cap_at(screw: str) -> np.ndarray:
    return seated_tip(screw) + np.array([0.0, 0.0, g.DRIVER_LEN])


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
        self.freeze = False            # integral term frozen (pressing on, or striking, something)
        self.shift = 0.0               # how far the held part sits off-centre between the fingers (finger encoders)

    def aim(self, pos, rot) -> None:
        """Command the TCP so that the held part's centre line (``shift`` along the fingers' closing axis from the TCP)
        goes to ``pos``; ``self.pos`` keeps the target as given."""
        from yam_arena import ik

        rot = np.asarray(rot, float)
        q, err, rerr = ik.solve(np.asarray(pos, float) - self.shift * rot[:, 0], ik.matrix_to_quat_xyzw(rot),
                                q_init=self.q, base_pos=self.base, iters=60)
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

    def set_q(self, q) -> None:
        """Command joint positions directly (``pos``/``rot`` follow by forward kinematics)."""
        from yam_arena import ik

        self.q = np.asarray(q, float).copy()
        p, r = ik.tcp_pose(self.q, self.base)
        self.pos, self.rot = p + self.shift * r[:, 0], r

    def move_q(self, q, seconds: float):
        """Generator: a smooth move in joint space to ``q``."""
        from yam_arena import ik

        q0, q1 = self.q.copy(), np.asarray(q, float)
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            self.set_q(q0 + ik.smoothstep(k / n) * (q1 - q0))
            yield

    def jump_q(self, q, seconds: float):
        """Generator: the joint targets jump to ``q`` at once and stay there."""
        self.set_q(q)
        for _ in range(max(1, round(seconds * CONTROL_HZ))):
            yield

    def jump(self, pos, rot, seconds: float):
        """Generator: the target jumps to ``pos``/``rot`` at once and stays there (the arm moves as fast as it can)."""
        self.aim(pos, rot)
        for _ in range(max(1, round(seconds * CONTROL_HZ))):
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
    """The demo as per-step arm commands. ``driver`` (tip position), ``hammer`` (H pose), ``holding`` and ``strikes``
    are what the script expects (for the tests)."""

    def __init__(self, blows: int = BLOWS):
        self.left, self.right = Arm("left"), Arm("right")
        self.n_blows = blows
        self.flags: dict = {}
        self.driver = g.DRIVER_START.copy()
        self.hammer = (g.HAMMER_START.copy(), g.HAMMER_STRIKE.copy())
        self.holding = {"driver": False, "hammer": False}
        self.seated = None            # the screw the bit is pressed into, if any
        self.strikes: list = []       # (step index, screw) when a blow's target is set
        self.left_stop = None         # during a blow: where the left hand stops (the face on the cap)
        self.feel = None              # the right hand's measured height minus its target (from the encoders), in run()
        self.feel0 = None
        self.offset = np.zeros(3)     # correction for the driver having moved in the hand (found by feel)
        self.searches: list = []
        self.n = 0

    def _carry_driver(self, gen):
        for _ in gen:
            if self.holding["driver"]:
                self.driver = driver_from_tcp(self.right.pos)
            yield

    def _carry_hammer(self, gen):
        for _ in gen:
            if self.holding["hammer"]:
                self.hammer = hammer_from_tcp(self.left.pos, self.left.rot)
            yield

    # -- the driver (right arm) ----------------------------------------------------------------------------------------
    def _press(self, screw: str, extra=(0.0, 0.0), judge: bool = True):
        """Generator: press the bit home on the screw (offset by the correction found so far plus ``extra``, mm along the
        hand's approach and closing axes) and feel whether it went into the recess: True if it did (or if there is no
        feel, as in the offline plan)."""
        A = self.right
        off = self.offset + 0.001 * (extra[0] * _A + extra[1] * _C)
        A.freeze = True
        yield from self._carry_driver(A.move(*driver_tcp(seated_tip(screw) - [0.0, 0.0, PRESS] + off), 0.5))
        yield from wait(0.4)
        if self.feel is None or not judge:
            return True
        if self.feel0 is None:
            self.feel0 = self.feel                     # the first seat on each screw, with a fresh grip, from the known
                                                       # layout, sets the reference
            return True
        return self.feel - self.feel0 <= FEEL_HIGH

    def _lift(self, screw: str):
        A = self.right
        A.freeze = False
        yield from self._carry_driver(A.move(*driver_tcp(seated_tip(screw) + [0.0, 0.0, RESEAT_UP] + self.offset), 0.3))

    def _home(self, screw: str):
        """Generator: press the bit home; if the hand stays high (the bit on the head's rim, not in the recess), lift it
        and try small offsets round the spot until it drops in; keep the offset that worked."""
        ok = yield from self._press(screw)
        for cand in ([] if ok else SEARCH):
            yield from self._lift(screw)
            if (yield from self._press(screw, cand)):
                self.offset = self.offset + 0.001 * (cand[0] * _A + cand[1] * _C)
                self.searches.append((screw, cand))
                break
        self.driver = seated_tip(screw)
        self.seated = screw
        self.flags["ready"] = True

    def _seat(self, screw: str):
        A = self.right
        tip = g.screw_world(screw)
        yield from self._carry_driver(A.move(*driver_tcp(tip + [0.0, 0.0, 0.010] + self.offset), 1.2))
        A.freeze = True
        yield from self._carry_driver(A.move(*driver_tcp(seated_tip(screw) - [0.0, 0.0, 0.0] + self.offset), 1.5))
        yield from self._home(screw)
        yield from wait(0.4)
        self.flags[f"seated_{screw}"] = True

    def _hold(self, screw: str):
        """Hold the driver on the screw until the hammer is done with it; after each blow (while the hammer winds up),
        lift the bit 6 mm and press it home again, feeling for the recess."""
        while not self.flags.get(f"struck_{screw}"):
            if self.flags.pop("reseat", False):
                self.seated = None
                yield from self._lift(screw)
                yield from self._home(screw)
            else:
                yield

    def driver_work(self):
        A = self.right
        grasp, grot = driver_tcp(g.DRIVER_START)
        back = -0.09 * _A
        up = np.array([0.0, 0.0, 1.0])
        yield from A.move(grasp + back + 0.05 * up, grot, 2.5, grip=1.0)
        yield from A.move(grasp + back, grot, 1.0)
        yield from A.move(grasp, grot, 1.5)
        yield from A.set_grip(0.0, 1.0)
        self.holding["driver"] = True
        yield from wait(0.5)
        yield from self._carry_driver(A.move(grasp + 0.08 * up, grot, 1.2))
        travel = g.TABLE + g.FACE + 0.06
        above_block = np.array([g.DRIVER_START[0], g.DRIVER_START[1], travel])
        yield from self._carry_driver(A.move(*driver_tcp(above_block), 1.0))
        yield from self._carry_driver(A.move(*driver_tcp(g.screw_world("a") + [0.0, 0.0, 0.06]), 2.5))
        yield from self._seat("a")
        yield from self._hold("a")
        self.seated = None
        A.freeze = False
        yield from self._carry_driver(A.move(*driver_tcp(g.screw_world("a") + [0.0, 0.0, 0.06]), 1.5))
        yield from self._carry_driver(A.move(*driver_tcp(above_block), 2.5))
        yield from self._carry_driver(A.move(*driver_tcp(g.DRIVER_START + [0.0, 0.0, 0.003]), 2.0))
        yield from A.set_grip(1.0, 1.0)
        self.holding["driver"] = False
        self.driver = g.DRIVER_START.copy()
        yield from wait(0.3)
        yield from A.move(grasp + 0.003 * up + back, grot, 1.2)
        yield from A.move(grasp + back + 0.08 * up, grot, 1.0)
        yield from A.move(A.home[0], A.home[1], 2.0)
        self.flags["driver_back"] = True

    # -- the hammer (left arm) -----------------------------------------------------------------------------------------
    def _strike(self, screw: str):
        L = self.left
        cap = cap_at(screw) + AIM
        yield from self._carry_hammer(L.move(*strike_tcp(cap, RAISE_Z), 1.5))
        L.aim(*strike_tcp(cap))
        contact = L.q.copy()                      # the joints with the face resting square on the cap
        L.set_q(contact)
        L.aim(*strike_tcp(cap, RAISE_Z))
        wind, through = contact.copy(), contact.copy()
        wind[SWING_JOINT] += WIND
        through[SWING_JOINT] -= THROUGH
        yield from self._carry_hammer(L.move_q(wind, 1.0))
        for _ in range(self.n_blows):
            yield from wait(REST_S)
            yield from wait_for(self.flags, "ready")         # the other hand has the bit home
            self.flags["ready"] = False
            L.freeze = True
            self.strikes.append((self.n, screw))
            self.hammer = hammer_from_tcp(*strike_tcp(cap - AIM))        # the face stops on the cap
            self.left_stop = strike_tcp(cap - AIM)                       # and so does the hand holding it
            yield from L.jump_q(through, SWING_S)
            yield from L.jump_q(contact, HOLD_S)                         # resting on the cap; then wind up again
            self.left_stop = None
            self.flags["reseat"] = True                                  # the other hand re-seats the bit meanwhile
            yield from self._carry_hammer(L.move_q(wind, UP_S))
            yield from wait(0.3)
            L.freeze = False
        yield from self._carry_hammer(L.move(*strike_tcp(cap, RAISE_Z + 0.05), 1.0))
        self.flags[f"struck_{screw}"] = True

    def hammer_work(self):
        L = self.left
        grasp, grot = hammer_tcp(g.HAMMER_START, g.HAMMER_STRIKE)
        back = np.array([-0.09, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        yield from L.move(grasp + back + 0.04 * up, grot, 2.5, grip=1.0)
        yield from L.move(grasp + back, grot, 1.0)
        yield from L.move(grasp, grot, 1.5)
        yield from L.set_grip(0.0, 1.0)
        self.holding["hammer"] = True
        yield from wait(0.5)
        yield from self._carry_hammer(L.move(grasp + 0.06 * up, grot, 1.2))
        yield from wait_for(self.flags, "seated_a")
        yield from self._strike("a")
        yield from self._carry_hammer(L.move(grasp + 0.06 * up, grot, 2.5))
        yield from self._carry_hammer(L.move(grasp + 0.003 * up, grot, 1.5))
        yield from L.set_grip(1.0, 1.0)
        self.holding["hammer"] = False
        self.hammer = (g.HAMMER_START.copy(), g.HAMMER_STRIKE.copy())
        yield from wait(0.3)
        yield from L.move(grasp + back + 0.003 * up, grot, 1.2)
        yield from L.move(grasp + back + 0.06 * up, grot, 1.0)
        yield from L.move(L.home[0], L.home[1], 2.0)

    def steps(self):
        for _ in together(self.driver_work(), self.hammer_work()):
            self.n += 1
            yield


def plan(max_steps: int = 3600, blows: int = BLOWS) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, and what the script expects
    of the driver and the hammer."""
    s = Script(blows)
    out = []
    for _ in s.steps():
        lp, lr = s.left_stop if s.left_stop is not None else (s.left.pos, s.left.rot)
        out.append({"left": (s.left.q.copy(), s.left.grip, lp.copy(), lr.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "driver": s.driver.copy(), "hammer": (s.hammer[0].copy(), s.hammer[1].copy()),
                    "holding": dict(s.holding), "seated": s.seated})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in (("left", s.left), ("right", s.right))},
                "strikes": list(s.strikes)})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 3600, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 3.0, blows: int = BLOWS) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
    from yam_arena import ik, station
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    task = emb.task.module
    from scenekit import arm_joints

    obs = emb.reset(Scene(id="demo", instruction=task.INSTRUCTION))
    script = Script(blows)
    gen = script.steps()
    integ = {"left": np.zeros(6), "right": np.zeros(6)}
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def command(side, arm):
        q = arm_joints(emb._env, side, ARM_JOINTS)
        if not arm.freeze:
            integ[side] = np.clip(integ[side] + I_GAIN * (arm.q - q), -I_MAX, I_MAX)
        return np.r_[arm.q + integ[side], arm.grip]

    def sense(side, arm, holding):
        """The held part's offset between the fingers, from the two finger encoders: their midpoint lies (q7 - q8) / 2
        along the closing axis from the TCP. Followed slowly while carrying; held while pressing or striking."""
        f = arm_joints(emb._env, side, FINGER_JOINTS)
        mid = float(f[0] - f[1]) / 2
        if not holding:
            arm.shift = 0.0
        elif not arm.freeze and arm.grip < 0.05:
            arm.shift += SHIFT_GAIN * (mid - arm.shift)
        return mid

    def actual_tcp(side):
        return ik.tcp_pose(arm_joints(emb._env, side, ARM_JOINTS), station.BASES[side])

    snap(obs)
    steps, ended, success, info, seen = 0, None, False, {}, 0
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        mids = {"left": sense("left", script.left, script.holding["hammer"]),
                "right": sense("right", script.right, script.holding["driver"])}
        script.feel = float(actual_tcp("right")[0][2] - script.right.pos[2])     # encoders: the hand's height error
        res = emb.step(Action(data=pack(command("left", script.left), command("right", script.right))))
        steps += 1
        info = res.info
        snap(res.observation)
        S = task._STATE
        for b in S["blows"][seen:]:
            print(f"step {steps} blow {b}", flush=True)
        seen = len(S["blows"])
        if steps % 50 == 0 or (steps % 10 == 0 and script.left.freeze):
            exp_d = script.driver
            exp_h = script.hammer[0]
            gap = S["hist"][-1]["h"] if S["hist"] else None
            print(f"step {steps} max_impulse {S['max_impulse']:.3f} face_gap {gap}", flush=True)
            print(f"step {steps} poses {S.get('poses')} expected driver_tip {np.round(exp_d, 4).tolist()} "
                  f"hammer {np.round(exp_h, 4).tolist()}", flush=True)
            for side, arm in (("left", script.left), ("right", script.right)):
                p, r = actual_tcp(side)
                print(f"step {steps} {side} tcp_err_mm {np.round((p - (arm.pos - arm.shift * arm.rot[:, 0])) * 1000, 1).tolist()} "
                      f"rot_err_deg {round(math.degrees(math.acos(min(1.0, (np.trace(r.T @ arm.rot) - 1) / 2))), 2)} "
                      f"finger_mid_mm {round(mids[side] * 1000, 1)} shift_mm {round(arm.shift * 1000, 1)} "
                      f"tcp {np.round(p, 4).tolist()} closing {np.round(r[:, 0], 3).tolist()}", flush=True)
        if steps % 50 == 0:
            print(f"step {steps} score {info.get('score')} turned_deg "
                  f"{ {k: round(math.degrees(v)) for k, v in S['turned'].items()} } camouts {S['camouts']} "
                  f"off_hub {S['off_hub']} clear {S['clear']} still {S['still']} gaps_mm {S['gaps']} why {S['why']}",
                  flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "turned_deg": {k: round(math.degrees(v), 1) for k, v in S["turned"].items()},
              "camouts": S["camouts"], "blows": S["blows"], "feel_searches": script.searches,
              "worst_ik_mm": {"left": round(script.left.worst_mm, 2), "right": round(script.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2, default=float))
    print(json.dumps(result, default=float), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=3600)
    ap.add_argument("--blows", type=int, default=BLOWS)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height,
        blows=a.blows)
