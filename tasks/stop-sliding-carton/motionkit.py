"""motionkit: moving and slippery boxes for yam_arena task files.

yam_arena's task files spawn objects that are either fixed or free (moved only by gravity and contact), all with
the station's rubber-like friction, and free ones start at rest. This module adds three object kinds:

- :class:`DrivenBox`, a box whose pose and velocity follow ``path(t)`` exactly, like a part of a machine driven
  by a motor: a shuttle table, a turntable, a moving tray. Objects resting on it ride along by friction, because
  its velocity (not only its pose) is set before every physics step.
- :class:`LaunchedBox`, a free box with its own friction that starts the episode moving at ``velocity``, like a
  parcel pushed off a chute.
- :class:`SlickBox`, a fixed box with low friction that wins over whatever touches it (PhysX's "min" combine
  mode), like a ball-transfer deck or a polished slide plate.
- :class:`Marker`, a fixed box that is only seen, never touched (no collider), like paint or tape on a surface.
  Mark a spot on a surface with it rather than by tiling the surface: a box sliding across the seam between two
  flush static boxes can catch on the next one's edge and tip over it.

Importing this module hooks ``yam_arena`` without changing it:

- ``yam_arena.scene`` spawns a :class:`DrivenBox` as a heavy rigid body with gravity off, and reports its state to
  ``success()`` and ``score()`` like any other object; a :class:`LaunchedBox` as a rigid body and a
  :class:`SlickBox` as a static collider, each with its own physics material, and a :class:`Marker` as a
  visual-only box (yam_arena reports their state);
- the embodiment wraps its simulation's ``step`` so that, before every physics step (240 Hz), each driven box is
  put where ``path`` says. The path's clock is the episode clock (``observation.state_time``): it stands at 0
  through the reset and its settling, and runs from the first control step on, so a scripted demo can time its
  waypoints against the motion;
- the embodiment's reset, once it has put everything back and settled, sets each launched box's velocity, so it
  is moving from the first control step on (the episode clock's 0);
- the embodiment runs every function registered with :func:`on_reset` after each reset (once the scene has settled
  and the launched boxes are moving), and every function registered with :func:`on_step` after each control step,
  before the task file is scored. A task file uses them for a monitor that latches a violation for the whole trial.

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

__all__ = ["DrivenBox", "LaunchedBox", "Marker", "SlickBox", "dwell_shuttle", "on_reset", "on_step", "slide"]

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
class LaunchedBox(Box):
    """A free box (it needs a ``mass``) that starts moving at ``velocity`` (world frame, m/s) when the episode clock
    starts, and has its own ``friction`` (static, dynamic) instead of the station's rubber-like one."""

    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: tuple[float, float] = (0.5, 0.4)  # static, dynamic

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a LaunchedBox is free; give it a mass")
        if len(self.velocity) != 3 or not np.all(np.isfinite(self.velocity)):
            raise ValueError(f"{self.name}: velocity must be 3 finite numbers")


@dataclass(frozen=True)
class SlickBox(Box):
    """A fixed box whose ``friction`` (static, dynamic) rules every contact on it: PhysX combines it with the other
    body's by taking the lower ("min"), where yam_arena's own objects average theirs."""

    friction: tuple[float, float] = (0.01, 0.01)

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a SlickBox is fixed (no mass)")
        if not 0 <= self.friction[1] <= self.friction[0]:
            raise ValueError(f"{self.name}: friction must be (static, dynamic) with 0 <= dynamic <= static")


@dataclass(frozen=True)
class Marker(Box):
    """A fixed box that is only rendered: it has no collider, so nothing touches it (paint or tape on a surface)."""

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a Marker is fixed (no mass)")


def slide(v0: float, mu: float, t: float, g: float = 9.81) -> tuple[float, float]:
    """A box sliding freely on a flat surface with friction ``mu``, launched at ``v0`` (m/s): (distance travelled,
    speed) after ``t`` seconds. It slows at ``mu * g`` and stays stopped once stopped."""
    a = mu * g
    t = min(t, v0 / a)
    return v0 * t - a * t * t / 2, v0 - a * t


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


# -- Arena objects -------------------------------------------------------------------------------------------------

_SPECS: dict[str, DrivenBox] = {}
_LAUNCHED: dict[str, LaunchedBox] = {}
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


def _material(friction, combine: str):
    import isaaclab.sim as sim_utils

    return sim_utils.RigidBodyMaterialCfg(static_friction=friction[0], dynamic_friction=friction[1],
                                          friction_combine_mode=combine)


def _launched_object(spec: LaunchedBox):
    """A free box like yam_arena's own, but with its own friction and no damping, so it slides as friction says."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from yam_arena.station import linear_rgb

    cfg = sim_utils.CuboidCfg(
        size=tuple(spec.size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(linear_damping=0.0, solver_position_iteration_count=16),
        mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        physics_material=_material(spec.friction, "average"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.8),
    )
    yaw = float(spec.yaw)
    quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
    return Object(name=spec.name, object_type=ObjectType.RIGID, spawner_cfg=cfg,
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=quat))


def _slick_object(spec: SlickBox):
    """A static box collider (no rigid body), like yam_arena's fixed boxes, with a "min"-combined material."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from yam_arena.station import linear_rgb

    cfg = sim_utils.CuboidCfg(
        size=tuple(spec.size),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_material(spec.friction, "min"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.3),
    )
    yaw = float(spec.yaw)
    quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
    return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg,
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=quat))


def _marker_object(spec: Marker):
    """A visual-only box: no collision properties, so no collider is made."""
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from yam_arena.station import linear_rgb

    cfg = sim_utils.CuboidCfg(
        size=tuple(spec.size),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.3),
    )
    yaw = float(spec.yaw)
    quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
    return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg,
                  initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=quat))


def _launch(base) -> None:
    """Set every launched box moving at its velocity (and not spinning)."""
    import torch

    for spec in _LAUNCHED.values():
        asset = base.scene[spec.name]
        twist = torch.zeros((base.scene.num_envs, 6), dtype=torch.float32, device=base.device)
        twist[:, :3] = torch.tensor(spec.velocity, dtype=twist.dtype, device=twist.device)
        asset.write_root_velocity_to_sim_index(root_velocity=twist)


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


_STEP: list = []
_RESET: list = []


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn(env)`` after every reset, once the scene has settled and the launched boxes are moving."""
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
        if isinstance(spec, LaunchedBox):
            _LAUNCHED[spec.name] = spec
            return _launched_object(spec)
        if isinstance(spec, SlickBox):
            return _slick_object(spec)
        if isinstance(spec, Marker):
            return _marker_object(spec)
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
        _launch(self._env.unwrapped)  # after the settling, before the first control step
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
