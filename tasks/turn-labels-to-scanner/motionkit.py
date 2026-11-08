"""motionkit: rigid boxes that move along a scripted path, and free bodies loaded from USD, for yam_arena task files.

yam_arena's task files spawn objects that are either fixed or free (moved only by gravity and contact). This module
adds :class:`DrivenBox`, a box whose pose and velocity follow ``path(t)`` exactly, like a part of a machine driven
by a motor: here, the carrier plates of a conveyor line (:func:`belt_run`). Objects resting on it ride along by
friction, because its velocity (not only its pose) is set before every physics step. It also adds :class:`UsdBody`,
a free rigid body loaded from a USD file next to the task file (here, a carton with its shipping label as visual-only
prims on one face).

Importing this module hooks ``yam_arena`` without changing it:

- ``yam_arena.scene`` spawns a :class:`DrivenBox` as a heavy rigid body with gravity off, and reports its state to
  ``success()`` and ``score()`` like any other object;
- the embodiment wraps its simulation's ``step`` so that, before every physics step (240 Hz), each driven box is
  put where ``path`` says. The path's clock is the episode clock (``observation.state_time``): it stands at 0
  through the reset and its settling, and runs from the first control step on, so a scripted demo can time its
  waypoints against the motion;
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
from pathlib import Path

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["DrivenBox", "UsdBody", "arm_joints", "belt_run", "on_reset", "on_step", "robot_links"]

HERE = Path(__file__).resolve().parent

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
class UsdBody(Box):
    """A free rigid body from ``usd`` (relative to this folder), its origin placed at ``pos`` turned ``rot_xyzw``.
    ``size`` is its box collider's edge lengths (body frame), used for its bounds. It subclasses ``Box`` only so the
    task-file checks accept it."""

    usd: str = ""
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    friction: tuple[float, float] = (0.9, 0.8)  # static, dynamic
    solver_iters: int = 16
    torsional_patch: float | None = None  # m: lets the finger pads resist twisting of a body held in a pinch

    def _check(self) -> None:
        super()._check()
        if self.mass is None:
            raise ValueError(f"{self.name}: a UsdBody here is free; give it a mass")
        if not (HERE / self.usd).is_file():
            raise FileNotFoundError(f"{self.name}: {HERE / self.usd} not found")


def belt_run(start, speed: float, ramp_s: float, turn_y: float, drop: float, drop_speed: float):
    """A conveyor carrier's path: it starts at ``start`` at rest, speeds up to ``speed`` (m/s, toward +y) over
    ``ramp_s`` (a cosine speed ramp, so it starts without a jerk), and runs on at that speed. Once its centre reaches
    ``turn_y`` (past the end of the line) it goes straight down ``drop`` metres at ``drop_speed``, as a carrier turning
    down round a conveyor's end, and stays there."""
    a = np.asarray(start, float)
    run_to_turn = turn_y - a[1]

    def run(t: float) -> tuple[float, float]:
        """Distance travelled along the line by ``t``, and the speed then."""
        if t <= 0.0:
            return 0.0, 0.0
        if t < ramp_s:
            return speed * (t / 2 - ramp_s / (2 * np.pi) * np.sin(np.pi * t / ramp_s)), \
                speed * (1 - np.cos(np.pi * t / ramp_s)) / 2
        return speed * (ramp_s / 2 + t - ramp_s), speed

    def path(t: float):
        s, v = run(t)
        if s <= run_to_turn:
            return a + np.array([0.0, s, 0.0]), np.array([0.0, v, 0.0])
        down = (s - run_to_turn) / speed * drop_speed  # time past the turn, at the drop speed
        if down >= drop:
            return a + np.array([0.0, run_to_turn, -drop]), np.zeros(3)
        return a + np.array([0.0, run_to_turn, -down]), np.array([0.0, 0.0, -drop_speed])

    return path


# -- Arena objects -------------------------------------------------------------------------------------------------

_SPECS: dict[str, DrivenBox] = {}  # driven every physics step
_BODIES: dict[str, UsdBody] = {}
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


def _usd_object(spec: UsdBody):
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    addon = {
        "physics_material": sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0],
                                                           dynamic_friction=spec.friction[1]),
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=spec.solver_iters),
        "mass_props": sim_utils.MassPropertiesCfg(mass=spec.mass),
    }
    if spec.torsional_patch:
        addon["collision_props"] = sim_utils.CollisionPropertiesCfg(
            torsional_patch_radius=spec.torsional_patch, min_torsional_patch_radius=0.8 * spec.torsional_patch)
    obj = Object(name=spec.name, usd_path=str(HERE / spec.usd), object_type=ObjectType.RIGID, spawn_cfg_addon=addon)
    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos),
                              rotation_xyzw=tuple(float(v) for v in spec.rot_xyzw)))
    return obj


def _rotation(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def _state(environment, spec: DrivenBox | UsdBody) -> ObjectState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0]
    pos = pose[:3] - origin
    half = np.asarray(spec.size, float) / 2  # yaw 0 or a multiple of 90 degrees keeps this the bounding box
    if isinstance(spec, UsdBody):  # free: any orientation; its world-aligned bounding box
        half = np.abs(_rotation(pose[3:7])) @ half
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
        if isinstance(spec, UsdBody):
            _BODIES[spec.name] = spec
            return _usd_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        mine = {**_SPECS, **_BODIES}
        plain = [n for n in names if n not in mine]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, mine[n]) for n in names if n in mine})
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
