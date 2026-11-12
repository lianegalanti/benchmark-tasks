"""motionkit: rigid boxes that move along a scripted path, and conveyor belts, for yam_arena task files.

A shared helper module, with :class:`Belt` (a belt whose surface moves while its body stays put),
:func:`speed_schedule` and :func:`episode_time`; its shuttle and stroke paths are general helpers, unused here.

yam_arena's task files spawn objects that are either fixed or free (moved only by gravity and contact). This module
adds :class:`DrivenBox`, a box whose pose and velocity follow ``path(t)`` exactly, like a part of a machine driven
by a motor: a shuttle table, a turntable, a moving tray. Objects resting on it ride along by friction, because its
velocity (not only its pose) is set before every physics step.

Importing this module hooks ``yam_arena`` without changing it:

- ``yam_arena.scene`` spawns a :class:`DrivenBox` as a heavy rigid body with gravity off, and reports its state to
  ``success()`` and ``score()`` like any other object;
- the embodiment wraps its simulation's ``step`` so that, before every physics step (240 Hz), each driven box is
  put where ``path`` says. The path's clock is the episode clock (``observation.state_time``): it stands at 0
  through the reset and its settling, and runs from the first control step on, so a scripted demo can time its
  waypoints against the motion;
- a :class:`Belt` is a driven box that stays where it is but is given its surface's velocity before every physics
  step, so what lies on it is carried along by friction while the belt itself never goes anywhere, as on a real
  belt conveyor (whose top run is a surface moving in place). PhysX moves it by that velocity during the step
  (0.2 mm at 5 cm/s) and the next step puts it back;
- the embodiment runs every function registered with :func:`on_step` after each control step, before the task file
  is scored, and every function registered with :func:`on_reset` after each reset, once the scene has settled. A
  task file uses them for a safety monitor that reads the arms' links (:func:`robot_links`, :func:`arm_joints`).

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["Belt", "DrivenBox", "arm_joints", "dwell_shuttle", "episode_time", "on_reset", "on_step", "park_stroke",
           "robot_links", "speed_schedule"]

DRIVEN_MASS = 50.0  # kg; heavy, so contacts within one physics step barely disturb it before the next write


@dataclass(frozen=True)
class DrivenBox(Box):
    """A box (``size``, ``color``, ``yaw``) whose centre follows ``path(t) -> (pos, vel)`` (world frame, m and m/s)
    from episode time 0. ``pos`` is where it stands at time 0 and must equal ``path(0)[0]``."""

    path: Callable[[float], tuple[np.ndarray, np.ndarray]] | None = None
    friction: tuple[float, float] = (0.8, 0.7)  # static, dynamic

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a DrivenBox has no mass; its path moves it")
        if self.path is None:
            raise ValueError(f"{self.name}: a DrivenBox needs a path")
        if not np.allclose(self.path(0.0)[0], self.pos, atol=1e-6):
            raise ValueError(f"{self.name}: pos must be path(0)")


@dataclass(frozen=True)
class Belt(DrivenBox):
    """A conveyor belt's top run as a box that stays at ``pos`` while its surface moves along the unit vector
    ``direction`` by ``run(t) -> (distance, speed)`` (see :func:`speed_schedule`). Needs no ``path``."""

    run: Callable[[float], tuple[float, float]] | None = None
    direction: tuple[float, float, float] = (0.0, 1.0, 0.0)

    def _check(self) -> None:
        Box._check(self)
        if self.mass is not None:
            raise ValueError(f"{self.name}: a Belt has no mass")
        if self.run is None:
            raise ValueError(f"{self.name}: a Belt needs a run")
        if not np.isclose(np.linalg.norm(self.direction), 1.0):
            raise ValueError(f"{self.name}: direction must be a unit vector")

    def surface_velocity(self, t: float, dt: float) -> np.ndarray:
        """The surface's mean velocity over the physics step from ``t`` (m/s, world frame)."""
        return (self.run(t + dt)[0] - self.run(t)[0]) / dt * np.asarray(self.direction, float)


def speed_schedule(changes, ramp_s: float):
    """A conveyor's run along its line: ``changes`` lists ``(t, speed)`` pairs (s, m/s; the first at t = 0, times
    increasing, each at least ``ramp_s`` after the one before). At each ``t`` the speed goes from what it was to the
    new one in ``ramp_s`` (a cosine ramp, so it never jumps), then holds. Returns ``run(t) -> (distance, speed)``,
    the distance travelled since t = 0 (m) and the speed then (m/s); the belt stands still before t = 0."""
    times = [float(t) for t, _ in changes]
    speeds = [float(v) for _, v in changes]
    if times[0] != 0.0 or any(b - a < ramp_s for a, b in zip(times, times[1:])):
        raise ValueError("changes must start at t = 0 and be at least ramp_s apart")

    def run(t: float) -> tuple[float, float]:
        s, v = 0.0, 0.0
        for i, (t0, target) in enumerate(zip(times, speeds)):
            if t <= t0:
                break
            t1 = min(t, times[i + 1]) if i + 1 < len(times) else t
            u = min((t1 - t0) / ramp_s, 1.0)  # how far through this change's ramp
            s += ramp_s * (v * u + (target - v) * (u - np.sin(np.pi * u) / np.pi) / 2)
            if t1 - t0 > ramp_s:
                s += target * (t1 - t0 - ramp_s)
            v = v + (target - v) * (1 - np.cos(np.pi * u)) / 2
            if u >= 1.0:
                v = target
        return float(s), float(v)

    return run


def dwell_shuttle(start, end, travel_s: float, dwell_s: float):
    """A path that waits ``dwell_s`` at ``start``, moves to ``end`` in ``travel_s`` (a cosine velocity profile, so it
    starts and stops without a jerk), waits ``dwell_s`` there, moves back, and repeats."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    period = 2 * (dwell_s + travel_s)

    def path(t: float):
        t = t % period
        half = dwell_s + travel_s
        p, q = (a, b) if t < half else (b, a)  # the first half-cycle waits at start and goes out; the second returns
        t = t % half
        if t < dwell_s:
            return p.copy(), np.zeros(3)
        u = (t - dwell_s) / travel_s
        s = (1 - np.cos(np.pi * u)) / 2
        ds = np.pi * np.sin(np.pi * u) / (2 * travel_s)
        return p + s * (q - p), ds * (q - p)

    return path


def park_stroke(park, reach, stroke_s: float, park_s: float):
    """A path that goes from ``park`` to ``reach`` and back in one stroke of ``stroke_s`` (a full cosine, so it starts,
    turns and stops without a jerk), waits ``park_s`` at ``park``, and repeats; it starts at ``park``, moving off."""
    a, b = np.asarray(park, float), np.asarray(reach, float)
    period = stroke_s + park_s

    def path(t: float):
        t = t % period
        if t >= stroke_s:
            return a.copy(), np.zeros(3)
        w = 2 * np.pi / stroke_s
        s = (1 - np.cos(w * t)) / 2
        ds = w * np.sin(w * t) / 2
        return a + s * (b - a), ds * (b - a)

    return path


# -- Arena objects -------------------------------------------------------------------------------------------------

_SPECS: dict[str, DrivenBox] = {}
_CLOCK = {"t": 0.0, "running": False}


def _driven_object(spec: DrivenBox):
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from yam_arena.station import linear_rgb

    cfg = sim_utils.CuboidCfg(
        size=tuple(spec.size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, solver_position_iteration_count=16),
        mass_props=sim_utils.MassPropertiesCfg(mass=DRIVEN_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0],
                                                        dynamic_friction=spec.friction[1]),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.5),
    )
    yaw = float(spec.yaw)
    quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=cfg,
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=quat))


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(environment, spec: DrivenBox) -> ObjectState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0]
    pos = pose[:3] - origin
    half = np.asarray(spec.size, float) / 2  # yaw 0 or a multiple of 90 degrees keeps this the bounding box
    return ObjectState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos - half,
                       high=pos + half)


def _drive(base, t: float, dt: float) -> None:
    """Put every driven box where its path is at ``t``, moving as it does over the coming step."""
    import torch

    origin = base.scene.env_origins
    for spec in _SPECS.values():
        asset = base.scene[spec.name]
        if isinstance(spec, Belt):  # back in place, its surface moving
            pos, vel = np.asarray(spec.pos, float), spec.surface_velocity(t, dt)
        else:
            pos, _ = spec.path(t)
            nxt, _ = spec.path(t + dt)
            vel = (nxt - pos) / dt  # the step's mean velocity, so the pose written next step matches where it ends up
        pose = asset.data.default_root_pose.torch.clone()
        pose[:, :3] = torch.tensor(pos, dtype=pose.dtype, device=pose.device) + origin
        twist = torch.zeros((pose.shape[0], 6), dtype=pose.dtype, device=pose.device)
        twist[:, :3] = torch.tensor(vel, dtype=pose.dtype, device=pose.device)
        asset.write_root_pose_to_sim_index(root_pose=pose)
        asset.write_root_velocity_to_sim_index(root_velocity=twist)


def _wrap_sim(base) -> None:
    sim = base.sim
    if getattr(sim, "_motionkit", False) or not _SPECS:
        return
    step = sim.step
    dt = float(sim.get_physics_dt()) if hasattr(sim, "get_physics_dt") else float(base.physics_dt)

    def step_hook(*a, **k):
        _drive(base, _CLOCK["t"], dt)
        out = step(*a, **k)
        if _CLOCK["running"]:
            _CLOCK["t"] += dt
        return out

    sim.step = step_hook
    sim._motionkit = True


def robot_links(environment) -> dict[str, dict[str, np.ndarray]]:
    """Each arm's link origins by link name (station frame, env 0), for proximity checks."""
    base = environment.unwrapped
    origin = _np(base.scene.env_origins[0])
    out = {}
    for side in ("left", "right"):
        robot = base.scene[f"{side}_robot"]
        pos = _np(robot.data.body_pos_w)[0] - origin
        out[side] = {name: pos[i] for i, name in enumerate(robot.body_names)}
    return out


def arm_joints(environment, side: str, names) -> np.ndarray:
    """One arm's measured joint positions, in the order of ``names``."""
    robot = environment.unwrapped.scene[f"{side}_robot"]
    ids = robot.find_joints(list(names), preserve_order=True)[0]
    return _np(robot.data.joint_pos)[0, ids]


def episode_time() -> float:
    """The episode clock (s): the time the driven boxes and belts are at, after the last physics step."""
    return float(_CLOCK["t"])


_STEP: list = []
_RESET: list = []


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn(env)`` after every reset, once the scene has settled."""
    _RESET.append(fn)
    return fn


# -- hooking yam_arena -----------------------------------------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_motionkit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, DrivenBox):
            _SPECS[spec.name] = spec
            return _driven_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._motionkit = True


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_motionkit", False):
        return
    ensure_sim, reset, task_step = cls._ensure_sim, cls.reset, cls._task_step

    def ensure_sim_hook(self, *a, **k):
        out = ensure_sim(self, *a, **k)
        _wrap_sim(self._env.unwrapped)
        return out

    def reset_hook(self, *a, **k):
        _CLOCK.update(t=0.0, running=False)  # driven boxes stand at path(0) while the scene resets and settles
        obs = reset(self, *a, **k)
        _CLOCK.update(t=0.0, running=True)  # the embodiment's own clock (state_time) restarts here too
        for fn in _RESET:
            fn(self._env)
        return obs

    def task_step_hook(self, *a, **k):
        for fn in _STEP:
            fn(self._env)
        return task_step(self, *a, **k)

    cls._ensure_sim, cls.reset, cls._task_step = ensure_sim_hook, reset_hook, task_step_hook
    cls._motionkit = True


class _Finder(importlib.abc.MetaPathFinder):
    """Patches yam_arena's scene and embodiment modules as they are imported (they need Isaac Sim, which starts after
    the task file is loaded)."""

    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        exec_module = loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module)

        loader.exec_module = exec_and_patch
        return spec


def install() -> None:
    """Hook yam_arena now for modules already imported, and on import for the rest. Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
