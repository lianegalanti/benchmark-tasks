"""scenekit: USD bodies, articulations and posed fixed shapes for a yam_arena task file.

Adapted from an earlier scene helper in this repository (its license applies). This copy adds: ``UsdBody.self_collisions``; writing articulations' ``joint_pos`` into the sim after every reset (Arena itself
resets only the arms); per-joint drives (``UsdBody.drives``: springs as well as dampers) whose targets are the start
positions or, for sprung joints, ``UsdBody.drive_targets``; a collision ``contact_offset``; per-link poses in
:class:`BodyState` (``links``); putting free rigid bodies back at their spawn pose once the scene has settled; an
optional effort cap per drive group; and helpers to set a joint's drive target, its damping, or its position while
the trial runs (:func:`set_target`, :func:`set_damping`, :func:`write_joint`); and callbacks after every physics
substep (:func:`on_substep`, 240 Hz), for events shorter than a control step, such as a hammer blow; per-substep reads
of an articulation's links and joints (:func:`links_np`); and PhysX contact forces between two bodies
(:class:`ContactPair`, from PhysX's tensor contact view; both bodies carry the contact-report API in their USD).

yam_arena's task files spawn single primitives (yaw only) and library assets. This module adds two object kinds
(here :class:`UsdBody` is used for the hub and rotor, the driver's block, the hammer's stand, the impact driver and the
hammer):

- :class:`UsdBody`: a body loaded from a USD file next to the task file. With ``articulation=True`` it is an
  articulation (here, the hub and rotor: a fixed base with the two screws on revolute joints) whose joints start at
  ``joint_pos``.
- :class:`Shape`: a fixed box or capsule in any orientation.

They subclass ``yam_arena.task_file.Box`` only so the task-file checks accept them. Importing this module hooks
``yam_arena.scene`` and the embodiment's step and reset, without changing yam_arena:

- ``success()`` and ``score()`` receive a :class:`BodyState` for these objects: the body origin's pose (not a
  bounding-box centre) and, for articulations, the joint positions;
- callbacks registered with :func:`on_step` run after every control step, before the task file is scored, and
  ones registered with :func:`on_reset` run after every reset; both get the Isaac Lab environment.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from yam_arena.task_file import Box, ObjectState

__all__ = ["BodyState", "ContactPair", "Shape", "UsdBody", "arm_joints", "env", "links_np", "on_reset", "on_step",
           "on_substep", "pose_np", "set_target_fast",
           "robot_links", "set_damping",
           "set_target", "write_joint", "write_start_joints", "write_start_poses"]

HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class UsdBody(Box):
    """A body from ``usd`` (relative to this folder). ``pos``/``rot_xyzw`` place the file's origin. ``size`` is only
    nominal. With ``mass=None`` a rigid body is kinematic; an articulation keeps the fixed base its file declares."""

    usd: str = ""
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    articulation: bool = False
    joint_pos: tuple[tuple[str, float], ...] = ()
    joint_damping: float = 0.03
    friction: tuple[float, float] = (0.8, 0.7)  # static, dynamic
    friction_min: bool = False  # combine friction with "min", so this surface's friction rules its contacts
    solver_iters: int = 16
    self_collisions: bool = True  # articulations: whether its links collide with each other
    torsional_patch: float | None = None  # m, for bodies held in a pinch (lets the pads resist twisting)
    # articulations: (joint-name regex, stiffness, damping[, effort cap]) per group; empty: every joint free with
    # joint_damping
    drives: tuple[tuple, ...] = ()
    drive_targets: tuple[tuple[str, float], ...] = ()  # joint -> spring rest point (default: its start position)
    contact_offset: float | None = None  # m, for every collider of the body (rest offset 0)
    usd_materials: bool = False  # keep the physics materials bound in the USD file (``friction`` is then unused)

    def _check(self) -> None:
        if not (HERE / self.usd).is_file():
            raise FileNotFoundError(f"{self.name}: {HERE / self.usd} not found")


@dataclass(frozen=True)
class Shape(Box):
    """A fixed box (``size``) or capsule (``radius``, ``height`` of the cylindrical part, along ``axis``), posed
    with ``rot_xyzw``. It collides unless ``collide`` is False."""

    kind: str = "box"
    radius: float = 0.01
    height: float = 0.02
    axis: str = "Z"
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    collide: bool = True

    def _check(self) -> None:
        if self.kind not in ("box", "capsule"):
            raise ValueError(f"{self.name}: kind must be 'box' or 'capsule'")
        if self.mass is not None:
            raise ValueError(f"{self.name}: a Shape is fixed (no mass)")


@dataclass(frozen=True)
class BodyState(ObjectState):
    """The body origin's pose, velocities, bounds and joint positions (rad or m)."""

    joints: dict = field(default_factory=dict)
    links: dict = field(default_factory=dict)  # articulations: link name -> (position, quat_xyzw), station frame

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- Arena objects -------------------------------------------------------------------------------------------------

_SPECS: dict[str, UsdBody | Shape] = {}


def _usd_object(spec: UsdBody):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    material = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction[0], dynamic_friction=spec.friction[1],
                                              friction_combine_mode="min" if spec.friction_min else "average")
    addon = {} if spec.usd_materials else {"physics_material": material}
    if spec.articulation:
        addon["articulation_props"] = sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=spec.solver_iters, enabled_self_collisions=spec.self_collisions)
    else:
        addon["rigid_props"] = sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=spec.solver_iters,
                                                                kinematic_enabled=spec.mass is None)
        if spec.torsional_patch:
            addon["collision_props"] = sim_utils.CollisionPropertiesCfg(
                torsional_patch_radius=spec.torsional_patch, min_torsional_patch_radius=0.8 * spec.torsional_patch)
    if spec.contact_offset is not None:
        extra = {}
        if spec.torsional_patch:
            extra = dict(torsional_patch_radius=spec.torsional_patch, min_torsional_patch_radius=0.8 * spec.torsional_patch)
        addon["collision_props"] = sim_utils.CollisionPropertiesCfg(contact_offset=spec.contact_offset, rest_offset=0.0,
                                                                    **extra)
    if not spec.articulation and spec.mass is not None:
        addon["mass_props"] = sim_utils.MassPropertiesCfg(mass=spec.mass)

    class UsdObject(Object):
        def _generate_articulation_cfg(self):
            cfg = super()._generate_articulation_cfg()
            joints = dict(spec.joint_pos)
            cfg.init_state.joint_pos = joints
            if spec.drives:
                cfg.actuators = {}
                for i, (expr, k, c, *cap) in enumerate(spec.drives):
                    extra = {"joint_effort_limit": float(cap[0])} if cap else {}
                    cfg.actuators[f"group{i}"] = ImplicitActuatorCfg(joint_names_expr=[expr], stiffness=k, damping=c,
                                                                     friction=0.0, **extra)
            else:
                cfg.actuators = {"joints": ImplicitActuatorCfg(joint_names_expr=list(joints) or [".*"],
                                                               stiffness=0.0, damping=spec.joint_damping,
                                                               friction=0.0)}
            return cfg

    obj = UsdObject(name=spec.name, usd_path=str(HERE / spec.usd),
                    object_type=ObjectType.ARTICULATION if spec.articulation else ObjectType.RIGID,
                    spawn_cfg_addon=addon)
    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(spec.rot_xyzw)))
    return obj


def _shape_object(spec: Shape):
    import isaaclab.sim as sim_utils
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    from yam_arena.station import linear_rgb

    body = {"visual_material": sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(spec.color), roughness=0.6)}
    if spec.collide:
        body["collision_props"] = sim_utils.CollisionPropertiesCfg()
    if spec.kind == "box":
        cfg = sim_utils.CuboidCfg(size=tuple(spec.size), **body)
    else:
        cfg = sim_utils.CapsuleCfg(radius=spec.radius, height=spec.height, axis=spec.axis, **body)
    obj = Object(name=spec.name, object_type=ObjectType.BASE, spawner_cfg=cfg)
    obj.set_initial_pose(Pose(position_xyz=tuple(float(v) for v in spec.pos), rotation_xyzw=tuple(spec.rot_xyzw)))
    return obj


def _np(x) -> np.ndarray:
    return getattr(x, "torch", x).detach().cpu().numpy().astype(np.float64)


def pose_np(environment, name: str) -> np.ndarray:
    """A body's root pose (x, y, z, qx, qy, qz, qw) in the station frame, env 0 (cheap: for substep callbacks)."""
    base = environment.unwrapped
    pose = _np(base.scene[name].data.root_link_pose_w)[0]
    pose[:3] -= _np(base.scene.env_origins[0])
    return pose


def _state(environment, spec: UsdBody | Shape) -> BodyState:
    if isinstance(spec, Shape):
        half = np.asarray(spec.size) / 2
        pos = np.asarray(spec.pos, float)
        return BodyState(pos=pos, quat_xyzw=np.asarray(spec.rot_xyzw, float), lin_vel=np.zeros(3),
                         ang_vel=np.zeros(3), low=pos - half, high=pos + half)
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0]
    joints, links = {}, {}
    if spec.articulation:
        q = _np(asset.data.joint_pos)[0]
        joints = {n: float(q[i]) for i, n in enumerate(asset.joint_names)}
        poses = _np(asset.data.body_link_pose_w)[0]
        links = {n: (poses[i, :3] - origin, poses[i, 3:7]) for i, n in enumerate(asset.body_names)}
    pos = pose[:3] - origin
    return BodyState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6], low=pos, high=pos,
                     joints=joints, links=links)


def links_np(environment, name: str):
    """An articulation's link poses (link name -> (x, y, z, qx, qy, qz, qw), station frame) and joint positions
    (joint name -> value), env 0 (cheap: for substep callbacks)."""
    base = environment.unwrapped
    asset = base.scene[name]
    origin = _np(base.scene.env_origins[0])
    poses = _np(asset.data.body_link_pose_w)[0]
    poses[:, :3] -= origin
    q = _np(asset.data.joint_pos)[0]
    return ({n: poses[i] for i, n in enumerate(asset.body_names)}, {n: float(q[i]) for i, n in enumerate(asset.joint_names)})


class ContactPair:
    """The contact force between two rigid bodies, read from PhysX every physics step: ``sensor`` and ``other`` are the
    bodies' prim names under their objects (e.g. "hammer" and "driver/.../body"), found in env 0's stage."""

    def __init__(self, environment, sensor: str, other: str):
        from isaaclab_physx.physics import PhysxManager

        self.paths = (_find_body(sensor), _find_body(other))
        globs = [p.replace("/env_0/", "/env_*/") for p in self.paths]
        self.view = PhysxManager.get_physics_sim_view().create_rigid_contact_view(
            [globs[0]], filter_patterns=[[globs[1]]], max_contact_data_count=256)
        self.dt = float(environment.unwrapped.physics_dt)

    def force(self) -> np.ndarray:
        """The force (N, station frame) the ``other`` body puts on the ``sensor`` body in the last physics step."""
        f = self.view.get_contact_force_matrix(dt=self.dt)
        f = f.numpy() if hasattr(f, "numpy") and not hasattr(f, "cpu") else _np(f)
        return np.asarray(f, float).reshape(-1, 3)[0]


def _find_body(suffix: str) -> str:
    """The path of the rigid body in env 0 whose path ends with ``suffix``."""
    import omni.usd
    from pxr import Usd, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath("/World/envs/env_0")
    hits = [str(p.GetPath()) for p in Usd.PrimRange(root)
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and str(p.GetPath()).endswith("/" + suffix)]
    if len(hits) != 1:
        raise RuntimeError(f"rigid body '{suffix}': found {hits}")
    return hits[0]


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


def _joint_ids(environment, name: str, joint: str):
    asset = environment.unwrapped.scene[name]
    return asset, asset.find_joints([joint], preserve_order=True)[0]


def _value(environment, n: int, v: float):
    import torch

    base = environment.unwrapped
    return torch.full((base.num_envs, n), float(v), dtype=torch.float32, device=base.device)


def set_target(environment, name: str, joint: str, value: float) -> None:
    """Set one joint's drive target (rad or m)."""
    asset, ids = _joint_ids(environment, name, joint)
    asset.actuators.target_command.set_position_index(value=_value(environment, len(ids), value), joint_ids=ids)


def set_target_fast(environment, asset_name: str, joint_ids, value: float) -> None:
    """Set one joint's drive target from a substep callback (``joint_ids`` from ``find_joints``)."""
    asset = environment.unwrapped.scene[asset_name]
    asset.actuators.target_command.set_position_index(value=_value(environment, len(joint_ids), value),
                                                      joint_ids=joint_ids)


def set_damping(environment, name: str, joint: str, value: float) -> None:
    """Set one joint's drive damping in the sim (N m s/rad or N s/m)."""
    asset, ids = _joint_ids(environment, name, joint)
    asset.write_joint_damping_to_sim_index(damping=_value(environment, len(ids), value), joint_ids=ids)


def write_joint(environment, name: str, joint: str, value: float) -> None:
    """Put one joint at ``value`` with zero velocity (a hard stop)."""
    import torch

    asset, ids = _joint_ids(environment, name, joint)
    base = environment.unwrapped
    q = torch.tensor(_np(asset.data.joint_pos), dtype=torch.float32, device=base.device)
    v = torch.tensor(_np(asset.data.joint_vel), dtype=torch.float32, device=base.device)
    q[:, ids] = float(value)
    v[:, ids] = 0.0
    env_ids = torch.arange(base.num_envs, device=base.device)
    # all joints written at once, as after a reset (the other joints keep their measured state)
    asset.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=v, env_ids=env_ids)


# -- hooking yam_arena -----------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_SUBSTEP: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Run ``fn(env)`` after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Run ``fn(env)`` after every reset, once the scene has settled."""
    _RESET.append(fn)
    return fn


def on_substep(fn):
    """Run ``fn(env)`` after every physics substep (the scene's ``update`` inside the env's decimation loop)."""
    _SUBSTEP.append(fn)
    return fn


def _hook_substeps(environment) -> None:
    scene = environment.unwrapped.scene
    if not _SUBSTEP or getattr(scene, "_scenekit_substep", False):
        return
    update = scene.update

    def update_hook(*a, **k):
        update(*a, **k)
        if _ENV["env"] is not None:
            for fn in _SUBSTEP:
                fn(_ENV["env"])

    scene.update = update_hook
    scene._scenekit_substep = True


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_scenekit", False):
        return
    task_object, object_states = scene.task_object, scene.object_states

    def task_object_hook(spec):
        if isinstance(spec, UsdBody | Shape):
            _SPECS[spec.name] = spec
            return _usd_object(spec) if isinstance(spec, UsdBody) else _shape_object(spec)
        return task_object(spec)

    def object_states_hook(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = object_states(environment, plain) if plain else {}
        out.update({n: _state(environment, _SPECS[n]) for n in names if n in _SPECS})
        return {n: out[n] for n in names}

    scene.task_object, scene.object_states = task_object_hook, object_states_hook
    scene._scenekit = True


def write_start_joints(environment) -> bool:
    """Put every articulated UsdBody's joints at its ``joint_pos`` (zero velocity); True if any were written."""
    import torch

    base = environment.unwrapped
    wrote = False
    for spec in _SPECS.values():
        if not (isinstance(spec, UsdBody) and spec.articulation and spec.joint_pos):
            continue
        asset = base.scene[spec.name]
        names = [n for n, _ in spec.joint_pos]
        ids = asset.find_joints(names, preserve_order=True)[0]
        q = _np(asset.data.joint_pos).copy()
        for i, (_, v) in zip(ids, spec.joint_pos):
            q[:, i] = v
        dev = base.device
        q = torch.tensor(q, dtype=torch.float32, device=dev)
        env_ids = torch.arange(base.num_envs, device=dev)
        asset.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        # and make them the joints' drive targets, so a sprung joint (stiffness > 0) holds its start position
        # (Isaac Lab otherwise leaves the targets of an articulation nothing commands at zero)
        target = q.clone()
        if spec.drive_targets:
            tids = asset.find_joints([n for n, _ in spec.drive_targets], preserve_order=True)[0]
            for i, (_, v) in zip(tids, spec.drive_targets):
                target[:, i] = v
        asset.set_joint_position_target_index(target=target, env_ids=env_ids, full_data=True)
        asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)
        wrote = True
    return wrote


def write_start_poses(environment) -> bool:
    """Put every free rigid UsdBody back at its spawn pose, at rest; True if any were written. Run after the start
    joints have settled, so a body nudged while the scene settled starts the trial exactly where the task places it."""
    import torch

    base = environment.unwrapped
    dev = base.device
    wrote = False
    for spec in _SPECS.values():
        if not isinstance(spec, UsdBody) or spec.articulation or spec.mass is None:
            continue
        asset = base.scene[spec.name]
        pose = torch.tensor([[*spec.pos, *spec.rot_xyzw]], dtype=torch.float32, device=dev).repeat(base.num_envs, 1)
        pose[:, :3] += base.scene.env_origins
        env_ids = torch.arange(base.num_envs, device=dev)
        asset.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(base.num_envs, 6, device=dev),
                                               env_ids=env_ids)
        wrote = True
    return wrote


def _patch_embodiment(module) -> None:
    cls = module.YamArenaEmbodiment
    if getattr(cls, "_scenekit", False):
        return
    task_step, reset = cls._task_step, cls.reset

    def task_step_hook(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return task_step(self, *a, **k)

    def reset_hook(self, *a, **k):
        _ENV["env"] = None          # no substep callbacks while Arena resets and settles the scene
        obs = reset(self, *a, **k)
        # Arena resets only the arms' joints (reset_arms_to_default); an articulation's ``init_state.joint_pos``
        # is never written to the sim, so it would start at its USD pose. Write the start
        # joints now, then settle and render again so the first observation shows them.
        if write_start_joints(self._env):
            for _ in range(max(1, round(module.SETTLE_S / self._base.step_dt))):
                self._env.step(self._ready)
            if write_start_poses(self._env):
                for _ in range(max(1, round(0.25 / self._base.step_dt))):
                    self._env.step(self._ready)
            for _ in range(module.RESET_RENDERS):
                self._base.sim.render()
            obs = self._observation()
        _hook_substeps(self._env)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = task_step_hook, reset_hook
    cls._scenekit = True


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
