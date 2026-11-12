"""cratekit: the folding crate as one articulation (base, four hinged walls, four sprung catches) for a yam_arena
task file, built at spawn time with pxr from the dimensions in geometry.py.

Adapted from a partkit.py helper used by other tasks in this repository (rigid parts from boxes with spring-loaded
sliders): the spawner here authors a tree of links joined by revolute (wall hinges) and prismatic (catches) joints,
with capsules for the round handhold rails, and turns the articulation's self-collisions on so the walls, the catches
and the folded walls collide with each other.

Importing this module hooks ``yam_arena.scene`` and the embodiment's step and reset without modifying yam_arena:

- ``success()`` and ``score()`` receive a :class:`CrateState` for the crate: the base's pose and every joint's
  position (wall hinges in rad, 0 folded and pi/2 upright; catches in m, 0 out and negative pushed in);
- callbacks registered with :func:`on_step` run after every control step, before the task file is scored, and ones
  registered with :func:`on_reset` after every reset; both get the Isaac Lab environment.

Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

import geometry as g

__all__ = ["Crate", "CrateState", "arm_joints", "env", "on_reset", "on_step", "robot_links"]


@dataclass(frozen=True)
class Crate(Box):
    """The crate (see geometry.py). It subclasses Box only so yam_arena's task-file checks accept it; ``pos`` is the
    base's origin (the middle of its underside) and ``size`` its outer box."""

    hinge_damping: float = 0.02     # N m s/rad, light
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if self.mass is None or self.mass <= 0:
            raise ValueError(f"{self.name}: the crate must be loose (give it a mass)")


@dataclass(frozen=True)
class CrateState(ObjectState):
    """The base origin's pose (``pos``, ``quat_xyzw``), velocities, bounds and the joints by name."""

    joints: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# -- spawning -------------------------------------------------------------------------------------------------------

_SPECS: dict[str, Crate] = {}


def _spawn_crate(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the crate under ``prim_path`` (pxr only): a plain Xform holding the links as siblings."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Crate = _SPECS[cfg.crate_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=g.PP_FRICTION, dynamic_friction=g.PP_FRICTION,
                                        friction_combine_mode="min")
    pm.func(phys, pm)
    looks: dict = {}

    def look(color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.6)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path: str, pos, rot: np.ndarray, mass: float):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(float(v) for v in pos)))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(*g.quat_wxyz(rot)))
        prim = xform.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(mass))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        return prim

    def collide(prim, color):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(0.001)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), phys, stage=stage)
        bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def shapes(path: str, boxes, capsules=()):
        for i, (centre, size, color, rot) in enumerate(boxes):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in centre)))
            if tuple(rot) != (1.0, 0.0, 0.0, 0.0):
                cube.AddOrientOp().Set(Gf.Quatf(*(float(v) for v in rot)))
            cube.AddScaleOp().Set(Gf.Vec3f(*(float(v) for v in size)))
            collide(cube.GetPrim(), color)
        for i, (centre, radius, length, color, axis) in enumerate(capsules):
            cap = UsdGeom.Capsule.Define(stage, f"{path}/round_{i}")
            cap.CreateAxisAttr(axis)
            cap.CreateRadiusAttr(float(radius))
            cap.CreateHeightAttr(float(length))
            cap.AddTranslateOp().Set(Gf.Vec3d(*(float(v) for v in centre)))
            collide(cap.GetPrim(), color)

    def joint(kind, path, parent, child, pos0, rot0, axis, lower, upper):
        j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
        j.CreateAxisAttr(axis)
        j.CreateBody0Rel().SetTargets([Sdf.Path(parent)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(child)])
        j.CreateLocalPos0Attr(Gf.Vec3f(*(float(v) for v in pos0)))
        j.CreateLocalRot0Attr(Gf.Quatf(*g.quat_wxyz(rot0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLowerLimitAttr(float(lower))
        j.CreateUpperLimitAttr(float(upper))
        return j

    base = f"{prim_path}/base"
    body(base, (0.0, 0.0, 0.0), np.eye(3), g.M_BASE)
    shapes(base, g.base_boxes())
    for name, wall in g.WALLS.items():
        path = f"{prim_path}/{name}"
        flat = wall.frame(0.0)
        body(path, wall.hinge, flat, wall.mass)
        shapes(path, *g.wall_shapes(wall))
        # Revolute about the body's x axis; 0 is the authored (folded) pose, pi/2 upright. Limits in degrees.
        joint("revolute", f"{path}/{g.hinge_joint(name)}", base, path, wall.hinge, flat, "X", 0.0, 90.0)
    for name, pawl in g.PAWLS.items():
        wall = g.WALLS[pawl.wall]
        path = f"{prim_path}/{name}"
        body(path, wall.point(pawl.rest_local, 0.0), wall.frame(0.0), g.M_PAWL)
        shapes(path, pawl.boxes())
        j = joint("prismatic", f"{path}/{g.pawl_joint(name)}", f"{prim_path}/{pawl.wall}", path, pawl.rest_local,
                  np.eye(3), "Y", -g.PAWL_TRAVEL, 0.0)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(g.PAWL_RATE)
        drive.CreateDampingAttr(2.0)
        drive.CreateTargetPositionAttr(0.0)
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(base))
    art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(base))
    art.CreateEnabledSelfCollisionsAttr(True)
    art.CreateSolverPositionIterationCountAttr(32)
    art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _arena_object(spec: Crate):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class CrateSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_crate)
        crate_name: str = ""

    class CrateObject(Object):
        def _generate_articulation_cfg(self):
            # Hinges: free, lightly damped. Catches: springs whose default target (0) is the pawl's rest stop.
            actuators = {
                "hinges": ImplicitActuatorCfg(joint_names_expr=[".*_hinge"], stiffness=0.0,
                                              damping=spec.hinge_damping, friction=0.0),
                "catches": ImplicitActuatorCfg(joint_names_expr=[".*_slide"], stiffness=g.PAWL_RATE, damping=2.0),
            }
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            cfg.init_state.joint_pos = {".*": 0.0}
            return self._add_initial_pose_to_cfg(cfg)

    return CrateObject(
        name=spec.name, object_type=ObjectType.ARTICULATION, spawner_cfg=CrateSpawnerCfg(crate_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                          rotation_xyzw=(0.0, 0.0, math.sin(spec.yaw / 2), math.cos(spec.yaw / 2))))


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _crate_state(environment, spec: Crate) -> CrateState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    q = _np(asset.data.joint_pos)[0]
    joints = {n: float(q[i]) for i, n in enumerate(asset.joint_names)}
    pos = pose[:3] - origin
    half = np.array(spec.size) / 2
    return CrateState(pos=pos, quat_xyzw=pose[3:7], lin_vel=vel[:3], ang_vel=vel[3:6],
                      low=pos - [half[0], half[1], 0.0], high=pos + [half[0], half[1], 2 * half[2]], joints=joints)


def write_start_joints(environment) -> None:
    """Fold every wall and let every catch out (all joints 0, at rest) on the crates."""
    import torch

    base = environment.unwrapped
    for spec in _SPECS.values():
        asset = base.scene[spec.name]
        q = torch.zeros_like(_t(asset.data.joint_pos))
        env_ids = torch.arange(base.num_envs, device=q.device)
        asset.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
        asset.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q), env_ids=env_ids)


def robot_links(environment) -> dict[str, dict[str, np.ndarray]]:
    """Each arm's link origins by link name (station frame, env 0)."""
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


# -- hooking yam_arena -------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset."""
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_cratekit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Crate):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _crate_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._cratekit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_cratekit_patched", False):
        return
    base_task_step, base_reset = cls._task_step, cls.reset

    def _task_step(self, *a, **k):
        _ENV["env"] = self._env
        for fn in _STEP:
            fn(self._env)
        return base_task_step(self, *a, **k)

    def reset(self, *a, **k):
        obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for fn in _RESET:
            fn(self._env)
        return obs

    cls._task_step, cls.reset = _task_step, reset
    cls._cratekit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
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
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena (now for modules already imported, otherwise when they are). Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


install()
assert task_file
