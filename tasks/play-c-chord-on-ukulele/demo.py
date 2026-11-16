"""Scripted full-score demo for playing a C chord on the ukulele, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 1200] [--depth-mm 2.0] [--no-record]

It works open loop from the fixed, known layout (geometry.py) and the arms' own joint readings only: it never reads
the strings' or the ukulele's state from the simulator.

The ukulele's neck is to the robot's right, so the right arm frets and the left arm strums, and a downstroke (G to
A) moves toward the robot.

1. Right arm (fretting hand): gripper open, its closing axis along the neck, the finger leaning 35 degrees back over
   the nut side. The rounded end of one fingertip's nub (4 mm thick, 15 mm wide) comes down across the A string 5 mm
   behind the 3rd fret wire, its width reaching past the fingerboard's edge on one side and stopping 3.7 mm short of
   the E string on the other, and presses the string toward the fingerboard, which seats it on the 3rd fret's crown.
   The other finger is 80 mm along the neck and 56 mm higher, clear of everything.
2. Left arm (strumming hand): gripper open, closing axis across the strings, the finger leaning 35 degrees with its
   tip pointing away from the robot (trailing, like a fingernail in a downstroke). One fingertip's nub comes down
   beyond the G string over the soundhole's neck edge, below the strings' centres: 0.85 mm at G, deepening to 1.75,
   2.25 and 2.0 mm at C, E and A, because the soft wrist springs up a few millimetres after each string lets go
   (``--depth-mm`` shifts the whole profile; 1.5 is nominal).
3. Once the fret has been held for 2 s, the left hand sweeps 97 mm toward the robot in 0.9 s (smoothly, with a
   30 mm run-up and run-out, so it crosses the four strings, G to A, at a steady speed in about 0.24 s):
   each string rides down the nub's leaning face to its rounded end and snaps off it.
4. Both hands then lift clear.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arms'
encoders) takes out the soft wrists' sag, which also builds the fretting hand's pressing force.

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
I_GAIN, I_MAX = 0.08, 0.15     # joint integral: per control step, and clamp (rad) ...
I_MAX_FRET = 0.30               # ... the fretting arm's clamp: room for the press
CORR_GAIN, CORR_MAX = 0.3, 0.008   # the fretting hand's Cartesian correction after the chord change: per step, clamp

DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])   # TCP x along +x, approach straight down
ACROSS = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # TCP x along +y (the strings), y across them

# fretting hand (right arm)
FRET_ARM = "right"
FRET_TILT = math.radians(35.0)  # the finger leans back over the nut side (about the axis across the strings)
FRET_SPOT = {"C": ("A", 3, 0.0050), "Am": ("G", 2, 0.0065)}   # string, fret, contact this far behind the fret wire
FRET_OUT = 0.002                # the nub's 15 mm edge centred 2 mm outboard of the string (3.4-3.7 mm clear of the next)
PRESS_BELOW = {"C": 0.0030, "Am": 0.0060}   # commanded: the string's underside this far below the fret crowns'
                                # plane, 2-4 mm past the fingerboard's face: the soft wrist stops on the board, pressing
FRET_SIDE = +1                  # gripper turned half a turn: the TCP's +x side tip is the lower one and presses; the
                                # other is 80 mm along the neck and 56 mm higher
# strumming hand (left arm)
STRUM_ARM = "left"
STRUM_TILT = math.radians(35.0)  # the finger leans (about y): its tip points away from the robot, trailing
STRUM_S = 0.222                 # over the soundhole's neck edge, 17 mm past the fingerboard's end
STRUM_DEPTH = 0.0015            # the nub end's centre this far below the strings' centres
STRUM_CLEAR = (0.016, 0.010)    # start this far (nub centre) before G and end this far past A: a run-up and a run-out
SWEEP_S = 0.50                  # one downstroke (smoothstep): it crosses G to A in about 0.2 s at 0.14-0.19 m/s
LIFT = 0.012                    # the hand returns this far above the strings
STRUM_SIDE = +1                 # gripper turned half a turn (the wrist's roll range): the TCP's +x side tip strums,
                                # its outer face leading, toward the robot
# the beat
BEAT = 0.98                     # s: 61 beats per minute ...
CHANGE_BEAT = 0.98              # ... the chord change too
CHANGE_AT = 0.66                # s after the 4th downstroke starts: the fretting hand leaves the A string, which has
                                # rung at least 0.2 s
FRET_SETTLE_S = 1.5             # the C chord is held this long before the first downstroke
# the depth at each string: deeper after the first, because the soft wrist springs up a few millimetres after each
# string lets go (the same for every downstroke)
PROFILE = {"G": 0.00085, "C": 0.00175, "E": 0.00225, "A": 0.0020}
PROFILES = [PROFILE] * len(g.SEQUENCE)
# diagnostics only (empty in the task): extra downstrokes after the scored eight, e.g. to swing G hard
PROBE: list = []


def fret_pose(chord: str = "C", press: float | None = None, lift: float = 0.0):
    """Fretting TCP (position, rotation): the nub's rounded end across the chord's fretted string, behind its fret,
    the string's underside ``press`` below the crowns' plane (``lift`` raises it)."""
    nm, fret, behind = FRET_SPOT[chord]
    press = PRESS_BELOW[chord] if press is None else press
    rot = g.rot_x(FRET_TILT) @ g.rot_z(math.pi) @ ACROSS
    s = g.fret_s(fret) - behind
    d = g.string_spec(nm)[1]
    out = FRET_OUT if g.string_spec(nm)[3] > 0 else -FRET_OUT
    centre = g.u_to_station([float(g.string_x(nm, s)) + out, -s, g.FRET_R - press + d + g.NUB_R + lift])
    return centre - rot @ np.array([FRET_SIDE * g.NUB_X, 0.0, g.NUB_Z]), rot


def strum_pose(x_u: float, depth: float = STRUM_DEPTH, lift: float = 0.0):
    """Strumming TCP (position, rotation) with the nub's end centred at x_u (U frame) across the neck."""
    rot = g.rot_y(-STRUM_TILT) @ g.rot_z(math.pi) @ DOWN
    zc = float(np.mean([g.string_z(nm, STRUM_S) for nm in g.NAMES])) - depth + lift
    centre = g.u_to_station([x_u, -STRUM_S, zc])
    return centre - rot @ np.array([STRUM_SIDE * g.NUB_X, 0.0, g.NUB_Z]), rot


def strum_depth(x_u: float, profile=None) -> float:
    """The planned depth where the nub crosses x_u: ``profile`` at each string, linear between them."""
    if profile is None:
        return STRUM_DEPTH
    xs = [float(g.string_x(nm, STRUM_S)) for nm in g.NAMES]
    return float(np.interp(x_u, xs, [profile[nm] for nm in g.NAMES]))


def strum_span() -> tuple[float, float]:
    return (float(g.string_x("G", STRUM_S)) - STRUM_CLEAR[0], float(g.string_x("A", STRUM_S)) + STRUM_CLEAR[1])


def strum_starts(t0: float) -> list[float]:
    """When each downstroke starts (s)."""
    out = [t0]
    for k in range(1, len(g.SEQUENCE)):
        out.append(out[-1] + (CHANGE_BEAT if g.SEQUENCE[k] != g.SEQUENCE[k - 1] else BEAT))
    return out


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

    def move(self, pos, rot=None, seconds: float = 1.0, grip: float | None = None, ease: bool = True):
        """Generator: a straight-line move to ``pos``/``rot`` (smooth, or at constant speed with ``ease=False``)."""
        from yam_arena import ik

        rot = self.rot if rot is None else rot
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n) if ease else k / n
            self.grip = g0 + s * (g1 - g0)
            self.aim(p0 + s * (np.asarray(pos, float) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def path(self, poses, seconds: float):
        """Generator: through a list of (pos, rot), smoothly in time (a smoothstep over the whole list)."""
        from yam_arena import ik

        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n) * (len(poses) - 1)
            i = min(int(s), len(poses) - 2)
            f = s - i
            p = (1 - f) * poses[i][0] + f * poses[i + 1][0]
            self.aim(p, ik.slerp_matrix(poses[i][1], poses[i + 1][1], f))
            yield

    def command(self) -> np.ndarray:
        return np.r_[self.q, self.grip]


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
    """The demo as per-step arm commands, on a fixed timeline (steps of 1/20 s). ``phase`` names what the script
    expects to be happening (for the tests)."""

    T_PRESSED = 4.7                 # s: the C chord pressed (approach 2.5 s, down 1.2 s, press 1.0 s)

    def __init__(self, depth: float = STRUM_DEPTH):
        self.left, self.right = Arm("left"), Arm("right")
        self.fretter, self.strummer = getattr(self, FRET_ARM), getattr(self, STRUM_ARM)
        self.depth = depth
        self.k = 0                  # control steps so far
        self.phase = {"fret": "ready", "strum": "ready"}
        self.integrate = {"left": True, "right": True}
        self.flags_snapshot = self.flags_restore = False   # the fretting arm's integral: keep, and restore it
        self.starts = strum_starts(self.T_PRESSED + FRET_SETTLE_S)
        self.n_probe = len(PROBE)               # diagnostics: probe downstrokes first, then the piece
        self.starts = [self.starts[0] + BEAT * k for k in range(self.n_probe)] + \
            [s + BEAT * self.n_probe for s in self.starts]

    def until(self, seconds: float):
        while self.k < round(seconds * CONTROL_HZ):
            yield

    def fret(self):
        L = self.fretter
        self.phase["fret"] = "approach"
        yield from L.move(*fret_pose("C", lift=0.025), 2.5, grip=1.0)
        yield from L.move(*fret_pose("C", press=-0.0015), 1.2)       # the nub 1.5 mm over the string's top
        self.flags_snapshot = True                # keep the integral as it is now: the arm's sag, not yet a press
        self.phase["fret"] = "press C"
        yield from L.move(*fret_pose("C"), 1.0)
        self.phase["fret"] = "hold C"
        change = next(i for i in range(1, len(g.SEQUENCE)) if g.SEQUENCE[i] != g.SEQUENCE[i - 1])
        yield from self.until(self.starts[self.n_probe + change - 1] + CHANGE_AT)
        self.phase["fret"] = "change"
        side = FRET_ARM
        self.flags_restore = "sag"                # keep the pressing integral; back to the sag-only one in the air
        self.integrate[side] = False
        yield from L.move(*fret_pose("C", press=-0.006), 0.06)         # off the A string, 6 mm up
        yield from L.move(*fret_pose("Am", press=-0.006), 0.14)        # over the G string, behind the 2nd fret
        self.integrate[side] = True
        yield from wait(0.1)                                            # settle over it
        self.flags_restore = "press"              # the C chord's pressing integral: the same press, 34 mm away
        yield from L.move(*fret_pose("Am"), 0.06)
        self.phase["fret"] = "hold Am"
        yield from self.until(self.starts[-1] + 1.5)
        self.phase["fret"] = "lift"
        yield from L.move(*fret_pose("Am", lift=0.03), 1.0)

    def strum(self):
        R = self.strummer
        x0, x1 = strum_span()
        self.phase["strum"] = "approach"
        yield from R.move(*strum_pose(x0, self.depth, lift=0.03), 2.5, grip=1.0)
        yield from R.move(*strum_pose(x0, self.depth + strum_depth(x0, PROFILES[0]) - STRUM_DEPTH), 1.5)
        profiles = PROBE + PROFILES
        for k, start in enumerate(self.starts):
            prof = profiles[k]
            dd = self.depth - STRUM_DEPTH
            self.phase["strum"] = "poised"
            yield from R.move(*strum_pose(x0, strum_depth(x0, prof) + dd), 0.07)
            self.integrate[STRUM_ARM] = True
            yield from self.until(start)
            self.integrate[STRUM_ARM] = False
            self.phase["strum"] = f"strum {k + 1}"
            sweep = [strum_pose(x, strum_depth(x, prof) + dd) for x in np.linspace(x0, x1, 41)]
            yield from R.path(sweep, SWEEP_S)
            self.phase["strum"] = "return"
            yield from R.move(*strum_pose(x1, strum_depth(x1, prof) + dd, lift=LIFT), 0.07)
            nxt = profiles[k + 1] if k + 1 < len(profiles) else prof
            yield from R.move(*strum_pose(x0, strum_depth(x0, nxt) + dd, lift=LIFT), 0.18)
        self.phase["strum"] = "done"
        yield from R.move(*strum_pose(x1, self.depth, lift=0.03), 1.0)

    def steps(self):
        for _ in together(self.fret(), self.strum()):
            self.k += 1
            yield


def plan(max_steps: int = 1200, depth: float = STRUM_DEPTH) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose and phase."""
    s = Script(depth)
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "phase": dict(s.phase)})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in (("left", s.left), ("right", s.right))}})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 3.0, depth: float = STRUM_DEPTH) -> dict:
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
    M = task.MON
    print("strings:", {nm: task._arrays(emb._env, nm)[0].shape for nm in g.NAMES}, flush=True)
    for nm in g.NAMES:      # diagnostics only: what the sim made of each string's joints and tendons
        try:
            asset = emb._env.unwrapped.scene[f"string_{nm.lower()}"]
            d = asset.data
            info_s = {"joints": len(asset.joint_names), "tendons": asset.num_fixed_tendons}
            for key in ("fixed_tendon_stiffness", "fixed_tendon_damping", "joint_stiffness", "joint_damping",
                        "default_mass", "joint_armature", "joint_vel_limits", "joint_effort_limits",
                        "joint_pos_target", "default_joint_pos"):
                try:
                    v = np.asarray(getattr(getattr(d, key), "torch", getattr(d, key)).detach().cpu().numpy())[0]
                    info_s[key] = [round(float(x), 4) for x in np.ravel(v)[:6]] + ["..."] + \
                        [round(float(x), 4) for x in np.ravel(v)[-3:]]
                except Exception as e:  # pragma: no cover
                    info_s[key] = f"n/a ({type(e).__name__})"
            print(f"string_{nm.lower()}", json.dumps(info_s), flush=True)
        except Exception as e:  # pragma: no cover - diagnostics only
            print("string check failed:", e, flush=True)
    script = Script(depth)
    gen = script.steps()
    integ = {"left": np.zeros(6), "right": np.zeros(6)}
    frames: dict[str, list] = {}
    trace: list = []

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    saved = {}
    corr = np.zeros(3)

    def command(side, arm):
        if side == FRET_ARM and script.flags_snapshot:
            saved["integ"], script.flags_snapshot = integ[side].copy(), False
        if side == FRET_ARM and script.flags_restore == "sag":
            saved["press"] = integ[side].copy()
            integ[side], script.flags_restore = saved.get("integ", np.zeros(6)).copy(), False
        if side == FRET_ARM and script.flags_restore == "press":
            integ[side], script.flags_restore = saved.get("press", integ[side]).copy(), False
        q = arm_joints(emb._env, side, ARM_JOINTS)
        qa = arm.q
        if side == FRET_ARM and script.phase["fret"] in ("change", "hold Am") and script.integrate[side]:
            # the fretting hand's own encoders: where its TCP really is; a fast correction across and along the
            # neck (not up: that is the press) so the nub lands on the G string, not beside it
            from yam_arena import ik, station
            meas, _ = ik.tcp_pose(q, station.BASES[side])
            corr[:2] = np.clip(corr[:2] + CORR_GAIN * (meas - arm.pos)[:2], -CORR_MAX, CORR_MAX)
            qa, _, _ = ik.solve(arm.pos - corr, ik.matrix_to_quat_xyzw(arm.rot), q_init=arm.q,
                                base_pos=station.BASES[side], iters=40)
        if script.integrate[side]:          # frozen while a hand sweeps or changes chord (dynamic lag isn't sag)
            # toward the joints actually commanded (with the correction), so the two never pull against each other
            lim = I_MAX_FRET if side == FRET_ARM else I_MAX
            integ[side] = np.clip(integ[side] + I_GAIN * (qa - q), -lim, lim)
        return np.r_[qa + integ[side], arm.grip]

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
        w = {nm: task._arrays(emb._env, nm)[0] for nm in g.NAMES}
        tcp = {}
        for side in ("left", "right"):          # the arms' own encoders: where the TCPs really are
            from yam_arena import ik, station
            p_, _ = ik.tcp_pose(arm_joints(emb._env, side, ARM_JOINTS), station.BASES[side])
            tcp[side] = [round(float(v) * 1000, 1) for v in p_]
        row = {"step": steps, "phase": dict(script.phase), "stop": dict(M.stop),
               "crowns_N": {nm: M.crowns[nm] for nm in ("G", "A")},
               "tcp_mm": tcp, "cmd_mm": {k: [round(float(v) * 1000, 1) for v in a.pos] for k, a in
                                          (("left", script.left), ("right", script.right))},
               "maxdisp_mm": {nm: round(float(np.abs(w[nm]).max()) * 1000, 2) for nm in g.NAMES}}
        trace.append(row)
        if steps % 10 == 0:
            print(json.dumps(row), "| score", info.get("score"), flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    plucks = [{k: (round(v, 5) if isinstance(v, float) else v) for k, v in p.items() if k != "dir"}
              for p in M.plucks]
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(M.why),
              "report": M.report(), "ring_load_N": {k: round(v, 3) for k, v in M.ring_load.items()},
              "plucks": plucks,
              "peaks_mm": {k: round(v * 1000, 2) for k, v in M.peaks.items()},
              "worst_ik_mm": {"left": round(script.left.worst_mm, 2), "right": round(script.right.worst_mm, 2)}}
    (out / "trace.json").write_text(json.dumps(trace))
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
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--depth-mm", type=float, default=STRUM_DEPTH * 1000)
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--cam-width", type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=352)
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record, cam_width=a.cam_width, cam_height=a.cam_height,
        depth=a.depth_mm / 1000)
