"""Scripted full-score demo for trimming the two fingernails, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 3600] [--no-record]

It works open loop from the fixed, known layout (geometry.py) and the arms' own joint readings only: it never reads
the hand's, the nails' or the clipper's state from the simulator. While it squeezes the clipper it reads the right
gripper's finger encoders (the pads' opening) to keep the clipper's lower jaw where it should be: the pads close on
the lever and the body together, so the body rises toward the TCP by half of what the pads close.

1. Right arm: comes down either side of the clipper standing in the caddy's slot, one pad on the lever's end and one
   under the body, and squeezes it shut (the lever on its stop: the clipper is then rigid in the grip), keeping the
   body still in the slot as the pads close. It lifts it and turns it to point at the hand: along the index finger
   toward the knuckle, tipped 15 degrees down, lever up.
2. Left arm, meanwhile: holds the index finger down on the towel roll, its closed fingertips on the middle segment.
3. Right arm: in front of the nail's tip, eases the squeeze until the jaws are 2.5 mm open, the nail between them, and
   slides them in along the finger until the cutting line is 1.65 mm from the quick at the nail's centre. It squeezes
   the clipper shut over 1.5 s, the nail kept in the middle of the mouth as both jaws close on it,
   holds it shut 0.5 s, opens it to 2.5 mm, slides it back off the nail and squeezes it shut again to carry it. One
   snip takes the whole free edge: the jaws are 13 mm wide, the nails 10.5 and 11.5 mm.
4. The left hand lets go of the index, takes the middle finger; the right arm trims the middle nail the same way.
5. The right arm stands the clipper back in the caddy's slot and lets go; the left hand lets go of the middle finger.
   Both arms go back to their ready pose.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arms' encoders)
takes out the soft wrists' sag.

:func:`plan` runs the same script without Isaac Sim, for the tests (reach, joint limits, clearances, cut geometry).
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
PITCH = math.radians(15.0)     # clipper tipped down toward the knuckle
U_CUT = 1.65 * g.MM            # cutting line, from the quick, at the nail's centre
U_OUT = 6.0 * g.MM             # the lower edge this far past the nail's longest point before sliding in
SQUEEZE_S, SHUT_S, OPEN_S = 1.5, 0.5, 1.0
SLIDE_GAP = g.HOLD_GAP         # the jaws held 2.5 mm open while they slide onto the nail
PRESS = 0.001                  # the left hand's closed pads this far into the finger's top: it holds it on the towel
# Where the jaws actually are relative to each nail, measured in Isaac Sim against where the demo aimed them: the
# clipper turns in the grip as the lever is squeezed (by different amounts held 2.5 mm open and shut), so the demo
# aims the lower edge differently while sliding in (open) and when the jaws meet (shut). Values are the edge
# positions to command, (u, w open, w shut) in the nail's frame, from the logged positions of two runs.
AIM = {"index": (1.56 * g.MM, 2.36 * g.MM, 0.82 * g.MM), "middle": (0.93 * g.MM, 3.06 * g.MM, 4.90 * g.MM)}


def demo_aim(name: str):
    """(u, w open, w shut) the demo commands the lower edge's centre to on ``name``'s nail."""
    return AIM[name]


def approach_dir(name: str) -> np.ndarray:
    """Unit vector along the clipper toward its jaws when cutting ``name``: toward the knuckle, tipped down."""
    xf = g.finger_rot(name)[:, 0]
    return -math.cos(PITCH) * xf + np.array([0.0, 0.0, -math.sin(PITCH)])


def cut_rot(name: str) -> np.ndarray:
    """C's axes when cutting ``name``: x along the clipper (to the jaws), y across, z through the lever (up-ish)."""
    d = approach_dir(name)
    lat = -g.finger_rot(name)[:, 1]
    z = np.cross(d, lat)
    return np.column_stack([d, lat, z / np.linalg.norm(z)])


def nail_point(name: str, u: float, w: float) -> np.ndarray:
    """A point (u, 0, w) of the nail's frame with the finger held, in the station frame."""
    n0, nr = g.nail_pose(name, *g.finger_pose_at(name, g.held_flex(name)))
    return n0 + nr @ np.array([u, 0.0, w])


# How the clipper actually sits in the squeezing grip, measured once in Isaac Sim (it is the same every run: the
# layout is fixed): 1.55 mm further toward its jaws than the nominal grasp, 0.8 mm further from the lever-side pad,
# and turned lever-side-down about its width axis by 1.7 degrees when shut, 4.7 degrees when held 2.5 mm open (the
# lever's spring then yields under the pad). The demo aims the gripper so the clipper lands where it wants it.
GRIP_DX, GRIP_DZ = 1.55e-3, -0.8e-3
GRIP_PITCH = (math.radians(1.7), math.radians(4.7))   # at the jaws shut, and 2.5 mm open


def grip_pose(origin, rc, pg: float, gap: float, calibrated: bool = True):
    """TCP pose that puts the clipper's body at (origin, rc) with the pads ``pg`` apart and the jaws ``gap`` open."""
    if not calibrated:
        return origin + rc @ g.tcp_in_c(pg), g.tcp_rot_from_c(rc)
    a = GRIP_PITCH[0] + (GRIP_PITCH[1] - GRIP_PITCH[0]) * min(max(gap / g.HOLD_GAP, 0.0), 1.0)
    rcmd = rc @ g.rot_y(a)              # the clipper ends up turned by -a about its y axis from the command
    d = -g.tcp_in_c(pg) + np.array([GRIP_DX, 0.0, GRIP_DZ])     # its origin from the TCP, in the commanded axes
    return origin - rcmd @ d, g.tcp_rot_from_c(rcmd)


def tcp_for_edge(name: str, u: float, w: float, pg: float, gap: float = 0.0):
    """TCP pose putting the lower jaw's edge centre at (u, 0, w) on ``name``'s nail with the pads ``pg`` apart."""
    rc = cut_rot(name)
    origin = nail_point(name, u, w) - rc @ np.array([g.E_X, 0.0, g.Z_LE])
    return grip_pose(origin, rc, pg, gap)


def pick_pose(pg: float):
    return g.CLIPPER_START + g.R_STAND @ g.tcp_in_c(pg), g.tcp_rot_from_c(g.R_STAND)


def pinch_pose(name: str):
    """Left TCP over ``name``'s middle segment, at its axis: pointing down, the fingers closing across the finger."""
    p = g.pinch_point(name)
    x = -g.finger_rot(name)[:, 1]
    z = np.array([0.0, 0.0, -1.0])
    return p, np.column_stack([x, np.cross(z, x), z])


def press_pose(name: str):
    """Left TCP pressing ``name`` down onto the towel: pointing down over its middle segment, fingers closed, the pad
    tips PRESS into the top of the finger where it rests on the towel."""
    p, r = pinch_pose(name)
    top = g.Z_F - g.FINGERS[name]["r"][1] - (g.Z_F - g.FINGERS[name]["r"][1] - g.TOWEL_TOP) + 2 * g.FINGERS[name]["r"][1]
    return np.array([p[0], p[1], top + 0.013 - PRESS]), r


def finger_dia(name: str) -> float:
    return 2 * g.FINGERS[name]["r"][1]


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
    """The demo as per-step arm commands. ``clipper`` is where the script expects the clipper (C origin, axes, jaw
    opening) and ``edge`` the lower edge's centre in the nail's frame while cutting (for the tests). ``measured_pg``,
    if set by the runner, is the right pads' measured opening (from the finger encoders)."""

    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.flags: dict = {}
        self.measured_pg: float | None = None
        self.clipper = (g.CLIPPER_START.copy(), g.R_STAND.copy(), g.GAP0)
        self.holding = False
        self.jaws = g.GAP0        # the jaws' planned opening
        self.edge = None          # (nail, u, w) while the jaws are at a nail
        self.snips: list = []     # (nail, step) when the jaws were shut on a nail
        self.step = 0

    def _pg(self, planned: float) -> float:
        """The pads' opening: measured by the finger encoders when the runner gives it, else as planned."""
        if self.measured_pg is None:
            return planned
        return float(np.clip(self.measured_pg, g.pad_gap(0.0) - 0.001, g.pad_gap(g.GAP0) + 0.001))

    def _carry(self, gen):
        for _ in gen:
            if self.holding:
                rc = g.c_rot_from_tcp(self.right.rot)
                self.clipper = (self.right.pos - rc @ g.tcp_in_c(g.pad_gap(self.jaws)), rc, self.jaws)
            yield

    def _hold_clipper(self, origin_fn, gap0, gap1, seconds, grip1=None, calibrated=True):
        """Keep the clipper's body where ``origin_fn(s)`` puts it (C origin and axes, s: 0..1 through the move) while
        the jaws go from gap0 to gap1: the TCP follows the pads' (measured) opening, as the body rides toward the TCP
        by half of what the pads close."""
        from yam_arena import ik

        A = self.right
        g0 = A.grip
        g1 = g.grip_for_gap(gap1) if grip1 is None else grip1
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            gap = gap0 + s * (gap1 - gap0)
            A.grip = g0 + s * (g1 - g0)
            origin, rc = origin_fn(s)
            A.aim(*grip_pose(origin, rc, self._pg(g.pad_gap(gap)), gap, calibrated))
            self.jaws = gap
            self.clipper = (origin, rc, gap)
            yield

    def _edge_move(self, name, u0, w0, u1, w1, gap0, gap1, seconds, grip1=None):
        """Move the lower edge's centre from (u0, w0) to (u1, w1) on ``name``'s nail while the jaws go from gap0 to
        gap1."""
        rc = cut_rot(name)

        def origin(s):
            u, w = u0 + s * (u1 - u0), w0 + s * (w1 - w0)
            self.edge = (name, u, w)
            return nail_point(name, u, w) - rc @ np.array([g.E_X, 0.0, g.Z_LE]), rc

        yield from self._hold_clipper(origin, gap0, gap1, seconds, grip1)

    # -- right arm: the clipper ------------------------------------------------------------------------------------
    def pick(self):
        """Pads either side of the standing clipper, on the lever's end and under the body; they close until both
        touch, then squeeze it shut (the lever on its stop: the clipper is then rigid in the grip) while the TCP keeps
        the body still in its slot."""
        A = self.right
        up = np.array([0.0, 0.0, 1.0])
        p, r = pick_pose(g.pad_gap(g.GAP0))
        wide = (g.pad_gap(g.GAP0) + 0.024) / g.FULL_OPEN
        yield from A.move(p + 0.08 * up, r, 2.5, grip=wide)
        yield from A.move(p + 0.004 * up, r, 1.5)
        yield from A.move(p, r, 0.5)
        yield from A.set_grip((g.pad_gap(g.GAP0) + 0.002) / g.FULL_OPEN, 1.0)      # pads 1 mm off each side
        self.holding = True
        self.jaws = g.GAP0
        start = (g.CLIPPER_START.copy(), g.R_STAND.copy())
        yield from self._hold_clipper(lambda s: start, g.GAP0, 0.0, 2.0, g.squeeze_grip(), calibrated=False)
        yield from wait(0.5)
        yield from self._carry(A.move(A.pos + 0.08 * up, r, 1.5))

    def trim(self, name: str):
        A = self.right
        U_AIM, W_IN, W_CUT = demo_aim(name)
        u_out = float(g.free_edge0(name).max()) + U_OUT
        d = approach_dir(name)
        p_out, r_out = tcp_for_edge(name, u_out, W_IN, g.pad_gap(0.0))          # jaws shut while carried
        back = -0.03 * d + np.array([0.0, 0.0, 0.02])
        yield from self._carry(A.move(p_out + back, r_out, 3.0))
        yield from wait_for(self.flags, f"held_{name}")
        yield from self._carry(A.move(p_out, r_out, 1.5))
        yield from self._edge_move(name, u_out, W_IN, u_out, W_IN, 0.0, SLIDE_GAP, 1.0)        # open the jaws
        yield from self._edge_move(name, u_out, W_IN, U_AIM, W_IN, SLIDE_GAP, SLIDE_GAP, 2.0)   # slide in
        yield from wait(0.4)
        # squeeze: the jaws shut on the nail from both sides; the arm moves with the pads so the body stays put
        yield from self._edge_move(name, U_AIM, W_IN, U_AIM, W_CUT, SLIDE_GAP, 0.0, SQUEEZE_S, g.squeeze_grip())
        self.snips.append((name, self.step))
        yield from self._edge_move(name, U_AIM, W_CUT, U_AIM, W_CUT, 0.0, 0.0, SHUT_S, g.squeeze_grip())
        yield from self._edge_move(name, U_AIM, W_CUT, U_AIM, W_IN, 0.0, SLIDE_GAP, OPEN_S)
        yield from self._edge_move(name, U_AIM, W_IN, u_out, W_IN, SLIDE_GAP, SLIDE_GAP, 1.5)    # slide out
        yield from self._edge_move(name, u_out, W_IN, u_out, W_IN, SLIDE_GAP, 0.0, 1.0, g.squeeze_grip())
        self.edge = None
        yield from self._carry(A.move(A.pos + back, r_out, 1.2))
        self.flags[f"done_{name}"] = True

    def put_back(self):
        A = self.right
        up = np.array([0.0, 0.0, 1.0])
        p, r = pick_pose(g.pad_gap(0.0))
        yield from self._carry(A.move(p + 0.08 * up, r, 3.5))
        yield from self._carry(A.move(p + 0.004 * up, r, 2.0))
        wide = (g.pad_gap(g.GAP0) + 0.024) / g.FULL_OPEN
        yield from A.set_grip(wide, 1.0)
        self.holding = False
        self.clipper = (g.CLIPPER_START.copy(), g.R_STAND.copy(), g.GAP0)
        yield from wait(0.3)
        yield from A.move(p + 0.08 * up, r, 1.5)
        yield from A.move(A.home[0], A.home[1], 2.5, grip=1.0)

    def right_arm(self):
        yield from self.pick()
        yield from self.trim("index")
        yield from self.trim("middle")
        yield from self.put_back()

    # -- left arm: holding the finger ------------------------------------------------------------------------------
    def hold(self, name: str):
        """Hold the finger down on the towel roll: the closed fingertips press on the top of its middle segment."""
        L = self.left
        p, r = press_pose(name)
        up = np.array([0.0, 0.0, 0.025])
        yield from L.move(p + up, r, 2.5, grip=0.0)
        yield from L.move(p + np.array([0.0, 0.0, 0.004]), r, 1.2)
        yield from L.move(p, r, 1.0)
        yield from wait(0.3)
        self.flags[f"held_{name}"] = True
        yield from wait_for(self.flags, f"done_{name}")
        yield from L.move(p + up, r, 1.2)

    def left_arm(self):
        yield from self.hold("index")
        yield from self.hold("middle")
        yield from self.left.move(self.left.home[0], self.left.home[1], 2.5, grip=1.0)

    def steps(self):
        for _ in together(self.right_arm(), self.left_arm()):
            self.step += 1
            yield


def plan(max_steps: int = 3600) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, and what the script expects
    of the clipper and the jaws."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "clipper": (s.clipper[0].copy(), s.clipper[1].copy(), s.clipper[2]), "holding": s.holding,
                    "edge": s.edge, "flags": dict(s.flags)})
        if len(out) >= max_steps:
            break
    out.append({"worst": {k: (a.worst_mm, a.worst_deg) for k, a in (("left", s.left), ("right", s.right))},
                "snips": list(s.snips)})
    return out


# -- running it in Isaac Sim ---------------------------------------------------------------------------------------
def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def run(task_path: str, out_dir: str, max_steps: int = 3600, record: bool = True, cam_width: int = 640,
        cam_height: int = 352, hold_s: float = 3.0) -> dict:
    from inspect_robots import Scene
    from inspect_robots.types import Action
    from inspect_robots_yam.packing import pack
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
    script = Script()
    gen = script.steps()
    integ = {"left": np.zeros(6), "right": np.zeros(6)}
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    def command(side, arm):
        q = arm_joints(emb._env, side, ARM_JOINTS)
        integ[side] = np.clip(integ[side] + I_GAIN * (arm.q - q), -I_MAX, I_MAX)
        return np.r_[arm.q + integ[side], arm.grip]

    snap(obs)
    steps, ended, success, info = 0, None, False, {}
    while steps < max_steps:
        # the right pads' opening, from the gripper's own encoders, while it holds the clipper
        script.measured_pg = (float(np.abs(arm_joints(emb._env, "right", FINGER_JOINTS)).sum())
                              if script.holding else None)
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(Action(data=pack(command("left", script.left), command("right", script.right))))
        steps += 1
        info = res.info
        snap(res.observation)
        S = task._STATE
        if steps % 20 == 0 or script.edge is not None and steps % 4 == 0:
            print(f"step {steps} score {info.get('score')} total {S['total']} gap_mm {S['gap'] * 1000:.2f} "
                  f"leverF {S.get('lever_force')} "
                  f"pg_mm {None if script.measured_pg is None else round(script.measured_pg * 1000, 2)} "
                  f"edge {script.edge and (script.edge[0], round(script.edge[1] * 1000, 2), round(script.edge[2] * 1000, 2))} "
                  f"defl {[round(v, 1) for v in S['maxdef'].values()]} down {S['down']} clear {S['clear']} "
                  f"gaps_mm {S['gaps']} why {S['why']}", flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    S = task._STATE
    score = 1.0 if success else float(info.get("score", 0.0) or 0.0)
    result = {"task_score": score, "success": success, "steps": steps, "violations": list(S["why"]),
              "white_edge_mm": {n: [round(x * 1000, 2) for x in S["L"][n]] for n in S["L"]},
              "finger_max_deflection_deg": {n: round(v, 2) for n, v in S["maxdef"].items()},
              "closures": S["closures"], "total": S["total"],
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
