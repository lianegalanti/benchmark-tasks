"""Scripted full-score demo for the cabin-filter swap, run in the yam_arena embodiment.

    python demo.py task.py demo_out [--max-steps 2400] [--no-record]

It works from the fixed, known layout (geometry.py) and the arms' own joint readings only: it never reads where the
glovebox, its panels or the filters are in the simulator. Every arm target is solved with IK on the URDF; a slow
integral term on the measured joint error (the arms' encoders) takes out the sag from pushing and carrying.

1. Both arms at once: each presses the thumb pad on its side of the open glovebox in (about 15 mm, against about
   10 N), so both stop bumps clear their blocks, and draws the bin down along its arc past the stops. They let go;
   the bin swings down on its hinge to its lowered stop.
2. Left arm: takes the old filter by its pull tab, slides it straight out of the slot, lifts it out over the lowered
   glovebox and sets it down on the table to the left.
3. Right arm, meanwhile: picks the new filter up from the table by its tab. Once the old one is out, it brings the
   new one in front of the slot, slides it in until the tab meets the housing, lets go and backs out.
4. Both arms: reach into the lowered bin and push its back wall up, swinging the glovebox up until both bumps have
   ridden over their blocks (their ramps push the panels in), then withdraw; the bin settles back onto its stops.
5. Both arms return to their ready pose.

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
I_GAIN, I_MAX = 0.10, 0.30       # joint integral: per control step, and clamp (rad)

# squeezing the glovebox
PRESS_PITCH, PRESS_YAW, PRESS_ROLL = -80.0, -45.0, 0.0   # yaw turns the approach inward on each side
PRESS_AT = (0.030, 0.090)        # TCP in the bin frame (x, z) while pressing; it travels with the bin
PRESS_DEPTH = 0.022              # the fingers' inner face this far inside the pad's free outer face (travel 14 mm)
DRAG_TO = math.radians(75.0)     # drawn down this far (the bumps are clear of the fascia from 72 degrees)
# the filters
FILTER_PITCH = -80.0             # the tab grip: approach 10 degrees below horizontal, for picking up and inserting
CARRY_PITCH = -74.0              # outside the dash, the filters are carried tipped 6 degrees further (easier reach)
OUT_TCP_X = 0.180                # TCP x while a filter's back end is just in front of the fascia
NEW_LET_GO = 0.16                # the new filter is let go this far short of seated (4 cm into the slot), then pushed
SET_PITCH = -60.0                # the old filter is tipped back end down by 20 degrees to set it down
PULL = 0.215                     # straight out of the slot until the filter's back end is clear of the face
EXIT_Z = 0.158                   # filter centre above the hinge while it crosses over the lowered bin
EXIT_X = g.FASCIA[0] - 0.012 - g.F_SIZE[0] / 2   # filter centre (dash x) once its back end is past the fascia
OLD_Y = 0.07                     # the old filter leaves toward the left arm, the new one comes in from the right
NEW_Y = -0.07
INSERT_PUSH = 0.002              # pushed this far past the seated position (the tab meets the housing face)
# raising the glovebox
PUSH_Y = 0.10                    # each arm pushes the back wall this far either side of the middle
PUSH_ZB = 0.130                  # ... at this height on it (bin frame)
PUSH_ZB_END = 0.152              # ... and this high at the end of the swing
PUSH_PEN = 0.015                 # the fingertips press this far into where the wall would be
RAISE_TO = g.OPEN_STOP - math.radians(2.5)   # pushed up this far: the bumps have clicked back behind the fascia


def grip_rot(pitch_deg: float, yaw_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Fingers closing along world y, approach pointing down and tilted ``-pitch_deg`` toward +x; then rolled about
    the approach and turned about the vertical."""
    from yam_arena.demo import grip_rotation
    from yam_arena.urdf import axis_angle_matrix

    r = axis_angle_matrix(np.array([0.0, 1.0, 0.0]), math.radians(pitch_deg)) @ grip_rotation(0.0)
    if roll_deg:
        r = r @ axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(roll_deg))
    if yaw_deg:
        r = axis_angle_matrix(np.array([0.0, 0.0, 1.0]), math.radians(yaw_deg)) @ r
    return r


def gripper_boxes(pos, rot, opening: float = 0.0):
    from task import gripper_boxes as gb

    return gb(pos, rot, opening)


# -- poses ---------------------------------------------------------------------------------------------------------
def press_rot(sign: float) -> np.ndarray:
    return grip_rot(PRESS_PITCH, sign * PRESS_YAW, PRESS_ROLL)


_FINGERS: list = []


def finger_points() -> np.ndarray:
    """Points of the closed fingers' collision meshes (from the arm's URDF), in the TCP frame."""
    if _FINGERS:
        return _FINGERS[0]
    import os
    import re
    import struct

    from yam_arena import YAM_URDF
    from yam_arena.constants import ARM_JOINTS, TCP_OFFSET
    from yam_arena.urdf import forward_kinematics

    def stl(path):
        data = open(path, "rb").read()
        n = struct.unpack("<I", data[80:84])[0]
        rows = np.frombuffer(data[84:84 + n * 50], dtype=np.uint8).reshape(n, 50)
        return np.frombuffer(rows[:, 12:48].tobytes(), dtype="<f4").reshape(-1, 3).astype(float)

    def rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]])

    q = {n: 0.0 for n in ARM_JOINTS}
    q.update(joint7=0.0, joint8=0.0)
    links = forward_kinematics(YAM_URDF, q)
    to_grip = np.linalg.inv(links["gripper"])
    text = open(YAM_URDF).read()
    pts = []
    for link in ("tip_left", "tip_right"):
        pts.append([])
        body = re.search(rf'<link name="{link}">(.*?)</link>', text, re.S).group(1)
        pat = r'<collision>\s*<origin xyz="([^"]+)" rpy="([^"]+)"\s*/>\s*<geometry>\s*<mesh filename="([^"]+)"'
        for m in re.finditer(pat, body):
            o = np.array([float(v) for v in m.group(1).split()])
            r = rpy(*[float(v) for v in m.group(2).split()])
            v = stl(os.path.join(os.path.dirname(YAM_URDF), m.group(3))) @ r.T + o
            t = to_grip @ links[link]
            pts[-1].append(v @ t[:3, :3].T + t[:3, 3] - np.array(TCP_OFFSET))
    left, right = (np.vstack(p)[::20] for p in pts)
    _FINGERS.append(np.vstack([left, right]))
    _FINGERS.append((left, right))
    return _FINGERS[0]


def finger_cloud(pos, rot, opening: float = 0.0) -> np.ndarray:
    """The fingers' collision-mesh points in the station frame, the fingers ``opening`` apart (along TCP x)."""
    finger_points()
    left, right = _FINGERS[1]
    shift = np.array([opening / 2, 0.0, 0.0])
    return np.asarray(pos, float) + np.vstack([left + shift, right - shift]) @ np.asarray(rot, float).T


def press_pose(theta: float, sign: float, depth: float) -> np.ndarray:
    """TCP pressing the pad on one side (sign +1 left, -1 right) with the bin at ``theta``: the innermost point of the
    fingers (their real collision meshes) over the pad ``depth`` inside the pad's free outer face (negative: outside)."""
    r = press_rot(sign)
    p = g.bin_point((PRESS_AT[0], 0.0, PRESS_AT[1]), theta)
    w = p + finger_points() @ r.T
    loc = g.bin_rot(theta).T @ (w - g.hinge_world()).T
    over = (loc[0] > g.PAD[0]) & (loc[0] < g.PAD[1]) & (loc[2] > g.PAD[2]) & (loc[2] < g.PAD[3])
    inner = float((w[over, 1] * sign).min())
    p[1] += sign * ((g.HALF_W + g.PAD_OUT - depth) - inner)
    return p


def filter_tcp(centre) -> np.ndarray:
    """TCP holding a level, square filter whose centre is at ``centre`` (station frame) by its tab."""
    return np.asarray(centre, float) + g.GRIP


def dash(p) -> np.ndarray:
    return g.dash_to_world(p)


def back_wall_point(theta: float, y: float, zb: float = PUSH_ZB):
    """A point on the back wall's inner face (station frame) and its inward normal (into the bin), bin at ``theta``."""
    d = np.array([g.BACK_TOP[0] - g.FLOOR_D, g.BACK_TOP[1]])
    lean = math.atan2(d[0], d[1])
    inward = np.array([-math.cos(lean), 0.0, math.sin(lean)])
    outer = np.array([g.back_x(zb), y, zb])
    p_bin = outer + inward * g.WALL_T
    r = g.bin_rot(theta)
    return g.hinge_world() + r @ p_bin, r @ inward


PUSH_ROT = None


def push_rot() -> np.ndarray:
    return grip_rot(-90.0)


def push_pose(theta: float, sign: float, pen: float = PUSH_PEN) -> np.ndarray:
    """TCP whose fingertips' top front edge presses the back wall's inner face from inside the bin, ``pen`` past it."""
    r = push_rot()
    up = r[:, 1] if r[2, 1] > 0 else -r[:, 1]                 # the TCP frame's y axis, whichever way points up
    edge = 0.020 * up + 0.013 * r[:, 2]                       # pads' top front edge, from the TCP
    # low on the wall (away from its rim) while the bin is far down; nearer the rim at the end, clear of the lid
    t = min(1.0, max(0.0, (math.radians(72.0) - theta) / math.radians(17.0)))
    p, n = back_wall_point(theta, sign * PUSH_Y, PUSH_ZB + t * (PUSH_ZB_END - PUSH_ZB))
    return p - n * pen - edge


class Arm:
    """One arm commanded in Cartesian space (IK on the URDF), each move starting where the last command left it."""

    def __init__(self, side: str):
        from yam_arena import ik, station
        from yam_arena.constants import READY

        self.side, self.base = side, np.asarray(station.BASES[side], float)
        self.sign = 1.0 if side == "left" else -1.0
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.home = (self.pos.copy(), self.rot.copy())
        self.worst_mm, self.worst_deg = 0.0, 0.0

    def aim(self, pos, rot) -> None:
        from yam_arena import ik

        q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=60)
        if err > 2e-4 or rerr > 1e-3:      # a hard step: give the solver more iterations from the same start
            q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=self.q, base_pos=self.base, iters=400)
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

    def solve_any(self, pos, rot) -> np.ndarray:
        """An exact IK solution for ``pos``/``rot``, the one closest to the current joints among several starts."""
        from yam_arena import ik
        from yam_arena.constants import READY

        best = None
        for q0 in (self.q, np.array(READY, float), np.array([0.0, 0.5, 0.5, 0.0, 0.0, 0.0]),
                   np.array([0.0, 1.5, 1.5, -1.0, 0.0, 0.0]), np.array([0.0, 2.0, 2.0, 0.0, 0.0, 0.0]),
                   np.array([0.0, 2.5, 1.5, 0.5, 0.0, 0.0])):
            q, err, rerr = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), q_init=q0, base_pos=self.base, iters=400)
            if err < 2e-4 and rerr < 1e-3:
                d = float(np.abs(q - self.q).max())
                if best is None or d < best[0]:
                    best = (d, q)
        if best is None:
            raise ValueError(f"{self.side}: no IK solution for {np.round(pos, 3)}")
        return best[1]

    def move_joint(self, pos, rot, seconds: float = 1.0, grip: float | None = None):
        """Generator: a smooth move in joint space to an IK solution for ``pos``/``rot`` (for moves in free space)."""
        from yam_arena import ik

        q0, q1, g0 = self.q.copy(), self.solve_any(pos, rot), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(seconds * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self.q = q0 + s * (q1 - q0)
            self.pos, self.rot = ik.tcp_pose(self.q, self.base)
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
    """The demo as per-step arm commands. ``theta``/``pins`` are the hinge angle and panel travel the script expects,
    ``holding`` which filter each arm carries (for the tests)."""

    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.flags: dict = {}
        self.theta = g.OPEN_STOP
        self.pins = [0.0, 0.0]
        self.holding = {"left": None, "right": None}
        self.raising = False

    # -- 1. squeeze both walls and draw the bin down past its stops ---------------------------------------------------
    def squeeze(self, A: Arm):
        s = A.sign
        R = press_rot(s)
        th0 = g.OPEN_STOP
        out = press_pose(th0, s, -0.04)
        yield from A.move_joint(out + np.array([-0.06, s * 0.02, 0.04]), R, 2.5, grip=0.0)
        yield from A.move(out, R, 1.0)
        yield from A.move(press_pose(th0, s, -0.004), R, 1.0)
        self.flags[f"at_pad_{A.side}"] = True
        yield from wait_for(self.flags, "at_pad_left")
        yield from wait_for(self.flags, "at_pad_right")
        n = 30
        for k in range(1, n + 1):                                  # press in over 1.5 s
            d = -0.004 + (PRESS_DEPTH + 0.004) * k / n
            self.pins[0 if s > 0 else 1] = min(g.PIN_TRAVEL, max(0.0, d - 0.001))
            A.aim(press_pose(th0, s, d), R)
            yield
        yield from wait(0.5)
        n = 60
        for k in range(1, n + 1):                                  # draw the bin down along its arc over 3 s
            from yam_arena import ik

            th = th0 + ik.smoothstep(k / n) * (DRAG_TO - th0)
            if s > 0:
                self.theta = th
            A.aim(press_pose(th, s, PRESS_DEPTH), R)
            yield
        yield from A.move(press_pose(DRAG_TO, s, -0.025), R, 0.5)  # let go: the panels spring back out
        self.pins[0 if s > 0 else 1] = 0.0
        yield from A.move(A.pos + np.array([-0.03, s * 0.01, 0.03]), R, 0.6)   # up and away as it swings down
        yield from A.move_joint(A.pos + np.array([-0.05, s * 0.03, 0.07]), grip_rot(FILTER_PITCH), 1.0)
        self.flags[f"off_{A.side}"] = True
        yield from wait_for(self.flags, "off_left")
        yield from wait_for(self.flags, "off_right")
        if s > 0:
            self.theta = g.LOWERED                                 # it swings down to its lowered stop

    # -- 2. the old filter out (left arm) ----------------------------------------------------------------------------
    def remove_old(self):
        L, R = self.left, grip_rot(FILTER_PITCH)
        seat = dash(g.SEAT_C)
        a = R[:, 2]
        grasp = filter_tcp(seat)
        yield from L.move_joint(np.array([0.22, 0.20, g.TABLE + 0.33]), R, 1.5, grip=1.0)
        yield from L.move(grasp - 0.06 * a, R, 1.5)
        yield from L.move(grasp, R, 1.2)
        yield from wait(0.8)
        yield from L.set_grip(0.0, 1.0)
        self.holding["left"] = "old"
        yield from wait(0.8)
        out = seat - np.array([PULL, 0.0, 0.0])
        yield from L.move(filter_tcp(out), R, 2.5)                    # straight out of the slot
        over = dash([EXIT_X, OLD_Y, g.HINGE_Z + EXIT_Z])
        yield from L.move(filter_tcp(over), R, 2.0)                   # out over the lowered bin
        self.flags["old_clear"] = True
        R70 = grip_rot(CARRY_PITCH)
        yield from L.move(L.pos + np.array([0.01, 0.0, -0.02]), R70, 1.0)
        yield from L.move(np.array([OUT_TCP_X + 0.005, g.OLD_DOWN[1], g.TABLE + 0.30]), R70, 2.5)   # beside the dash
        # tip it back end down and lower it until its back edge is 8 mm above the table, then let go
        R2 = grip_rot(SET_PITCH)
        rel = R2 @ R.T
        edge = rel @ (np.array([g.F_SIZE[0] / 2, 0.0, -g.F_SIZE[2] / 2]) - g.GRIP)
        down = np.array([g.OLD_DOWN[0] - g.GRIP[0], g.OLD_DOWN[1], g.TABLE + 0.008 - edge[2]])
        yield from L.move(down + np.array([0.0, 0.0, 0.04]), R2, 1.5)
        yield from L.move(down, R2, 1.0)
        yield from L.set_grip(1.0, 0.8)
        self.holding["left"] = None
        self.flags["old_down"] = True
        yield from wait(0.2)
        yield from L.move(L.pos + np.array([0.0, 0.0, 0.07]), R2, 1.0)
        self.flags["left_free"] = True

    # -- 3. the new filter in (right arm) ----------------------------------------------------------------------------
    def insert_new(self):
        """The new filter is carried by its tab, started 4 cm into the slot over the lead-in sill and hood, let go
        (the slot now holds it level), and pushed home by its tab with the closed fingertips."""
        A, R = self.right, grip_rot(FILTER_PITCH)
        start = g.NEW_START
        yield from A.move_joint(filter_tcp(start) + np.array([-0.04, 0.0, 0.16]), R, 2.5, grip=1.0)
        yield from A.move(filter_tcp(start) + np.array([-0.04, 0.0, 0.08]), R, 1.0)
        yield from A.move(filter_tcp(start) + np.array([-0.02, 0.0, 0.04]), R, 1.2)
        yield from A.move(filter_tcp(start), R, 1.2)
        yield from wait(0.5)
        yield from A.set_grip(0.0, 1.0)
        self.holding["right"] = "new"
        yield from wait(0.8)
        lifted = start + np.array([0.0, 0.0, 0.10])
        yield from A.move(filter_tcp(lifted), R, 1.2)
        yield from A.move_joint(np.array([OUT_TCP_X + 0.005, start[1], g.TABLE + 0.30]), grip_rot(CARRY_PITCH), 1.5)
        yield from wait_for(self.flags, "old_clear")
        over = dash([EXIT_X, NEW_Y, g.HINGE_Z + EXIT_Z])
        yield from A.move(filter_tcp(over) + np.array([0.01, 0.0, -0.02]), grip_rot(CARRY_PITCH), 2.5)
        yield from A.move(filter_tcp(over), R, 1.0)
        seat = dash(g.SEAT_C)
        entry = seat - np.array([PULL + 0.01, 0.0, -0.003])
        yield from A.move(filter_tcp(entry), R, 2.0)
        let_go = seat - np.array([NEW_LET_GO, 0.0, -0.002])
        yield from A.move(filter_tcp(let_go), R, 1.5)                 # its far end 4 cm into the slot
        yield from wait(0.3)
        yield from A.set_grip(1.0, 0.8)
        self.holding["right"] = None
        yield from wait(0.2)
        a = R[:, 2]
        yield from A.move(A.pos - 0.04 * a, R, 0.8)
        # push it home by its tab with the closed fingertips
        tab = np.array([let_go[0] + g.TAB[0][0], let_go[1], let_go[2] + 0.012])   # the tab's front face
        tip = 0.013 * a
        yield from A.move(tab - tip + np.array([-0.03, 0.0, 0.0]), R, 0.8, grip=0.0)
        yield from wait(0.4)
        yield from A.move(tab - tip, R, 0.6)
        self.holding["right"] = "new"                                 # the tab moves with the fingertips
        yield from A.move(tab - tip + np.array([NEW_LET_GO + INSERT_PUSH, 0.0, 0.0]), R, 3.5)
        yield from wait(0.3)
        self.holding["right"] = None
        yield from A.move(A.pos - 0.08 * a, R, 1.0)
        yield from A.move(A.pos + np.array([-0.02, -0.05, -0.02]), R, 1.0)
        self.flags["new_in"] = True

    # -- 4. push the glovebox back up past its stops (both arms) -----------------------------------------------------
    def raise_bin(self, A: Arm):
        s = A.sign
        R = push_rot()
        yield from wait_for(self.flags, "new_in")
        yield from wait_for(self.flags, "left_free")
        th = g.LOWERED
        contact = push_pose(th, s, 0.0)
        p, n = back_wall_point(th, s * PUSH_Y)
        below = contact + n * 0.025                                   # 2.5 cm off the wall, inside the bin
        yield from A.move(A.pos + np.array([-0.04, s * 0.06, 0.04]), A.rot, 1.0)
        yield from A.move_joint(np.array([0.22, s * 0.30, g.TABLE + 0.30]), R, 1.5, grip=0.0)
        pre = below + np.array([-0.035, 0.0, -0.015])
        yield from A.move(np.array([pre[0], pre[1], g.TABLE + 0.25]), R, 1.5)       # in front of the bin's rim
        yield from A.move(pre, R, 1.5)
        yield from A.move(below, R, 1.5)
        self.raising = True
        yield from A.move(push_pose(th, s, 0.0), R, 0.8)
        self.flags[f"under_{A.side}"] = True
        yield from wait_for(self.flags, "under_left")
        yield from wait_for(self.flags, "under_right")
        yield from A.move(push_pose(th, s), R, 0.5)
        n_steps = 100
        from yam_arena import ik

        for k in range(1, n_steps + 1):                               # swing it up over 5 s
            t = g.LOWERED + ik.smoothstep(k / n_steps) * (RAISE_TO - g.LOWERED)
            if s > 0:
                self.theta = t
            A.aim(push_pose(t, s), R)
            yield
        yield from wait(0.5)
        self.flags[f"up_{A.side}"] = True
        yield from wait_for(self.flags, "up_left")
        yield from wait_for(self.flags, "up_right")
        _, n = back_wall_point(RAISE_TO, s * PUSH_Y)
        yield from A.move(A.pos + np.array([-0.03, 0.0, 0.012]), R, 0.8)   # off the wall; it settles onto its stops
        self.raising = False
        if s > 0:
            self.theta = g.OPEN_STOP
        yield from A.move(A.pos + np.array([-0.16, 0.0, 0.02]), R, 1.5)
        yield from A.move(np.array([A.pos[0], s * 0.30, A.pos[2] + 0.04]), R, 1.2)
        yield from A.move_joint(A.home[0], A.home[1], 2.0, grip=1.0)

    def steps(self):
        yield from together(self.squeeze(self.left), self.squeeze(self.right))
        yield from together(self.remove_old(), self.insert_new())
        yield from together(self.raise_bin(self.left), self.raise_bin(self.right))
        yield from wait(0.5)


def plan(max_steps: int = 2400) -> list[dict]:
    """The script without Isaac Sim: per control step, each arm's joints, grip, TCP pose, the hinge angle and panel
    travel the script assumes, and what each arm holds."""
    s = Script()
    out = []
    for _ in s.steps():
        out.append({"left": (s.left.q.copy(), s.left.grip, s.left.pos.copy(), s.left.rot.copy()),
                    "right": (s.right.q.copy(), s.right.grip, s.right.pos.copy(), s.right.rot.copy()),
                    "theta": s.theta, "pins": tuple(s.pins), "holding": dict(s.holding), "raising": s.raising})
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
        if steps % 40 == 0 or (880 <= steps <= 1040 and steps % 8 == 0):
            from yam_arena import ik as _ik, station as _st

            S = task._STATE
            err = {}
            for side, arm in (("left", script.left), ("right", script.right)):
                tcp, _ = _ik.tcp_pose(arm_joints(emb._env, side, ARM_JOINTS), _st.BASES[side])
                err[side] = np.round((tcp - arm.pos) * 1000).astype(int).tolist()
            print(f"step {steps} score {info.get('score')} plan_theta {math.degrees(script.theta):.1f} "
                  f"level {S.get('level')} up {S.get('up')} clear {S.get('clear')} {S.get('info')} why {S['why']} "
                  f"tcp_err_mm {err}", flush=True)
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
    (out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)
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
