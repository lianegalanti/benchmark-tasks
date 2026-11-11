"""motionkit: scripted moving boxes, low-friction fixed boxes and an episode clock, for yam_arena task files.

yam_arena's task files spawn objects that are either fixed or free (moved only by gravity and contact), all with the
station's rubber-like friction. This module adds:

- :class:`DrivenBox`, a box whose pose and velocity follow ``path(t)`` exactly, like a part of a machine driven
  by a motor (here, the lift's deck). Objects resting on it ride along, because its velocity (not only its pose)
  is set before every physics step. Its ``friction`` can win over whatever touches it (``combine="min"``);
- :class:`SlickBox`, a fixed box whose low friction wins over whatever touches it (PhysX's "min" combine mode),
  like a steel bench top a carton slides on;
- :func:`dwell_stops`, a path that visits a list of stops in turn, waiting at each (here, the lift's three levels);
- :func:`clock`, the episode time, and :func:`on_reset`, callbacks run at the start of every episode, so a task
  file's ``score()`` can keep its own state over a trial (here, whether the carton has tipped).

Importing this module hooks ``yam_arena`` without changing it:

- ``yam_arena.scene`` spawns a :class:`DrivenBox` as a heavy rigid body with gravity off, and reports its state to
  ``success()`` and ``score()`` like any other object; a :class:`SlickBox` as a static collider with its own
  material (yam_arena reports its state);
- the embodiment wraps its simulation's ``step`` so that, before every physics step (240 Hz), each driven box is
  put where ``path`` says. The path's clock is the episode clock (``observation.state_time``): it stands at 0
  through the reset and its settling, and runs from the first control step on, so a scripted demo can time its
  waypoints against the motion. Callbacks registered with :func:`on_reset` run when it restarts.

Extended from the motionkit of the unload-paused-shuttle task (DrivenBox), with SlickBox from stop-sliding-carton and
clock/on_reset from stop-ball-before-edge; dwell_stops and DrivenBox's friction combine mode are new here.

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

__all__ = ["DrivenBox", "SlickBox", "clock", "dwell_stops", "dwell_windows", "on_reset"]

DRIVEN_MASS = 50.0  # kg; heavy, so contacts within one physics step barely disturb it before the next write


@dataclass(frozen=True)
class DrivenBox(Box):
    """A box (``size``, ``color``, ``yaw``) whose centre follows ``path(t) -> (pos, vel)`` (world frame, m and m/s)
    from episode time 0. ``pos`` is where it stands at time 0 and must equal ``path(0)[0]``."""

    path: Callable[[float], tuple[np.ndarray, np.ndarray]] | None = None
    friction: tuple[float, float] = (0.8, 0.7)  # static, dynamic
    combine: str = "average"  # PhysX friction combine mode; "min" lets a low friction here rule its contacts

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a DrivenBox has no mass; its path moves it")
        if self.combine not in ("average", "min", "multiply", "max"):
            raise ValueError(f"{self.name}: unknown friction combine mode {self.combine!r}")
        if self.path is None:
            raise ValueError(f"{self.name}: a DrivenBox needs a path")
        if not np.allclose(self.path(0.0)[0], self.pos, atol=1e-6):
            raise ValueError(f"{self.name}: pos must be path(0)")


@dataclass(frozen=True)
class SlickBox(Box):
    """A fixed box whose ``friction`` (static, dynamic) rules every contact on it: PhysX combines it with the other
    body's by taking the lower ("min"), where yam_arena's own objects average theirs."""

    friction: tuple[float, float] = (0.4, 0.3)

    def _check(self) -> None:
        super()._check()
        if self.mass is not None:
            raise ValueError(f"{self.name}: a SlickBox is fixed (no mass)")
        if not 0 <= self.friction[1] <= self.friction[0]:
            raise ValueError(f"{self.name}: friction must be (static, dynamic) with 0 <= dynamic <= static")


def dwell_stops(stops, travel_s: float, dwell_s: float):
    """A path that waits ``dwell_s`` at each of ``stops`` in turn, moving from one to the next in ``travel_s`` (a cosine
    velocity profile, so it starts and stops without a jerk), and from the last back to the first, forever. It starts
    at ``stops[0]``, at the beginning of its wait there. Stops can lie in any direction: a lift's levels, a shuttle's
    ends."""
    pts = [np.asarray(p, float) for p in stops]
    if len(pts) < 2:
        raise ValueError("dwell_stops needs at least two stops")
    leg = dwell_s + travel_s
    period = len(pts) * leg

    def path(t: float):
        t = t % period
        k = int(t // leg) % len(pts)
        p, q = pts[k], pts[(k + 1) % len(pts)]
        t -= k * leg
        if t < dwell_s:
            return p.copy(), np.zeros(3)
        u = min((t - dwell_s) / travel_s, 1.0)
        s = (1 - np.cos(np.pi * u)) / 2
        ds = np.pi * np.sin(np.pi * u) / (2 * travel_s)
        return p + s * (q - p), ds * (q - p)

    return path


def dwell_windows(stops, travel_s: float, dwell_s: float, stop: int, until: float) -> list[tuple[float, float]]:
    """The (start, end) times, up to ``until``, of every wait at ``stops[stop]`` on a :func:`dwell_stops` path."""
    leg = dwell_s + travel_s
    period = len(stops) * leg
    out, k = [], 0
    while k * period + stop * leg < until:
        t0 = k * period + stop * leg
        out.append((t0, t0 + dwell_s))
        k += 1
    return out


# -- Arena objects -------------------------------------------------------------------------------------------------

_SPECS: dict[str, DrivenBox] = {}
_CLOCK = {"t": 0.0, "running": False}
_RESET: list[Callable[[], None]] = []


def on_reset(fn: Callable[[], None]) -> Callable[[], None]:
    """Run ``fn()`` at the start of every episode, when the clock restarts (after the scene has reset and settled)."""
    _RESET.append(fn)
    return fn


def clock() -> float:
    """The episode time (s): 0 through the reset and settling, then the time since the first control step."""
    return float(_CLOCK["t"])


def _material(friction, combine: str):
    import isaaclab.sim as sim_utils

    return sim_utils.RigidBodyMaterialCfg(static_friction=float(friction[0]), dynamic_friction=float(friction[1]),
                                          friction_combine_mode=combine)


def _driven_object(spec: DrivenBox):
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from yam_arena.station import linear_rgb

    cfg = sim_utils.CuboidCfg(
        size=tuple(float(v) for v in spec.size),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, solver_position_iteration_count=16),
        mass_props=sim_utils.MassPropertiesCfg(mass=DRIVEN_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        physics_material=_material(spec.friction, spec.combine),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.5),
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
        size=tuple(float(v) for v in spec.size),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=_material(spec.friction, "min"),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.3),
    )
    yaw = float(spec.yaw)
    quat = (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))
    return Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg,
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


# -- hooking yam_arena -----------------------------------------------------------------------------------------------


def _patch_scene(scene) -> None:
    if getattr(scene, "_motionkit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, DrivenBox):
            _SPECS[spec.name] = spec
            return _driven_object(spec)
        if isinstance(spec, SlickBox):
            return _slick_object(spec)
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
    ensure_sim, reset = cls._ensure_sim, cls.reset

    def ensure_sim_hook(self, *a, **k):
        out = ensure_sim(self, *a, **k)
        _wrap_sim(self._env.unwrapped)
        return out

    def reset_hook(self, *a, **k):
        _CLOCK.update(t=0.0, running=False)  # driven boxes stand at path(0) while the scene resets and settles
        obs = reset(self, *a, **k)
        _CLOCK.update(t=0.0, running=True)  # the embodiment's own clock (state_time) restarts here too
        for fn in _RESET:
            fn()
        return obs

    cls._ensure_sim, cls.reset = ensure_sim_hook, reset_hook
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
