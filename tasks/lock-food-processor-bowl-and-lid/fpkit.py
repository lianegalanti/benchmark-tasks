"""fpkit: rigid parts built from boxes, cylinders and cones, optionally carrying sprung sliders, for yam_arena task files.

Adapted from a parts helper in another task folder of this repository (its license applies). Changes: the T-screw is replaced by any number of :class:`Slider`
links (a prismatic joint along the part's z with a spring: a stiff drive toward a rest point beyond its limit, which
gives a preload), cones are added, and the sliders' joint positions and spring targets are written after every reset.

A ``Part`` is one rigid body with several box, cylinder and cone colliders. Without a mass it is kinematic (fixed).
With ``sliders`` it is an articulation with a floating root (the part) whose links collide with other bodies but not
with the part itself (self-collisions off), so a plunger may sit inside the lug that houses it.

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena.
Pure Python at import time; everything that needs Isaac Sim is imported lazily.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from dataclasses import dataclass, field

import numpy as np
from yam_arena import task_file
from yam_arena.task_file import Box, ObjectState

__all__ = ["Part", "PartState", "Slider", "env", "install", "on_reset", "on_step", "part", "reset_sliders"]

Color = tuple[float, float, float]
# box: (centre, size, colour[, rotation as (w, x, y, z)]); cylinder: (centre, radius, height, colour[, axis]);
# cone: (centre, radius, height, colour) along +z, apex up
BoxSpec = tuple


@dataclass(frozen=True)
class Slider:
    """A link on a prismatic joint along the part's z. Its ``boxes`` are in the part's frame at joint position 0.
    The spring pulls it toward ``rest`` (which may lie beyond a limit: then it is preloaded against that limit)."""

    name: str
    boxes: tuple[BoxSpec, ...]
    lower: float
    upper: float
    stiffness: float
    damping: float
    rest: float
    mass: float = 0.01


@dataclass(frozen=True)
class Part(Box):
    """A rigid part made of ``boxes``, ``cylinders`` and ``cones``. It subclasses Box only so yam_arena's task-file
    checks accept it; ``size`` is its bounding box and ``pos`` is the body origin. Without a mass it is kinematic."""

    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple = ()
    cones: tuple = ()
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    sliders: tuple[Slider, ...] = ()
    friction: float = 0.8
    contact_offset: float = 0.0005
    torsional_patch: float | None = None   # m: for parts held in a pinch (lets the pads resist twisting)
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not (self.boxes or self.cylinders):
            raise ValueError(f"{self.name}: a Part needs at least one shape")
        if self.sliders and self.mass is None:
            raise ValueError(f"{self.name}: a part with sliders must be loose")


def _quat_matrix(wxyz) -> np.ndarray:
    w, x, y, z = wxyz
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _box(spec):
    centre, size, color = spec[:3]
    return centre, size, color, (tuple(spec[3]) if len(spec) > 3 else (1.0, 0.0, 0.0, 0.0))


def _cyl(spec):
    centre, radius, height, color = spec[:4]
    return centre, radius, height, color, (spec[4] if len(spec) > 4 else "Z")


def part(name, pos, boxes=(), mass=None, rot_xyzw=(0.0, 0.0, 0.0, 1.0), color=(0.5, 0.5, 0.5), friction=0.8,
         cylinders=(), cones=(), sliders=(), torsional_patch=None) -> Part:
    """Build a Part; ``size`` is filled in from the shapes' extent (part frame)."""
    half = [np.abs(_quat_matrix(q)) @ (np.array(s) / 2) for _, s, _, q in map(_box, boxes)]
    half += [np.array([r, r, h / 2]) for _, r, h, _, _ in map(_cyl, cylinders)]
    half += [np.array([c[1], c[1], c[2] / 2]) for c in cones]
    centres = [np.array(b[0]) for b in boxes] + [np.array(c[0]) for c in cylinders] + [np.array(c[0]) for c in cones]
    lo = np.min([c - h for c, h in zip(centres, half)], axis=0)
    hi = np.max([c + h for c, h in zip(centres, half)], axis=0)
    return Part(name=name, pos=tuple(float(v) for v in pos), size=tuple(float(v) for v in hi - lo), mass=mass,
                color=color, boxes=tuple(boxes), cylinders=tuple(cylinders), cones=tuple(cones),
                rot_xyzw=tuple(float(v) for v in rot_xyzw), sliders=tuple(sliders), friction=friction,
                torsional_patch=torsional_patch)


@dataclass(frozen=True)
class PartState(ObjectState):
    """The body origin's pose (``pos``, ``quat_xyzw``), velocities, bounds, and joint positions by name."""

    joints: dict = field(default_factory=dict)

    def rotation(self) -> np.ndarray:
        x, y, z, w = self.quat_xyzw
        return _quat_matrix((w, x, y, z))


# -- spawning -----------------------------------------------------------------------------------------------------

_SPECS: dict[str, Part] = {}


def _spawn_part(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author the part under ``prim_path`` (pxr only): a plain Xform holding the body (and its sliders' links)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec: Part = _SPECS[cfg.part_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))

    phys = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(phys, pm)
    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.4)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path: str, mass: float | None):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        rbody = UsdPhysics.RigidBodyAPI.Apply(prim)
        if mass is None:
            rbody.CreateKinematicEnabledAttr(True)
        else:
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateAngularDampingAttr(0.5)
        return prim

    def collide(prim, color):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(spec.contact_offset)  # below the fits' clearances
        cp.CreateRestOffsetAttr(0.0)
        if spec.torsional_patch:
            cp.CreateTorsionalPatchRadiusAttr(spec.torsional_patch)
            cp.CreateMinTorsionalPatchRadiusAttr(0.8 * spec.torsional_patch)
        bind_physics_material(str(prim.GetPath()), phys, stage=stage)
        bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def shapes(path: str, box_specs, cyl_specs=(), cone_specs=()):
        for i, (centre, size, color, rot) in enumerate(map(_box, box_specs)):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
            if tuple(rot) != (1.0, 0.0, 0.0, 0.0):
                cube.AddOrientOp().Set(Gf.Quatf(*rot))
            cube.AddScaleOp().Set(Gf.Vec3f(*size))
            collide(cube.GetPrim(), color)
        for i, (centre, radius, height, color, axis) in enumerate(map(_cyl, cyl_specs)):
            cyl = UsdGeom.Cylinder.Define(stage, f"{path}/cyl_{i}")
            cyl.CreateAxisAttr(axis)
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.AddTranslateOp().Set(Gf.Vec3d(*centre))
            collide(cyl.GetPrim(), color)
        for i, (centre, radius, height, color) in enumerate(cone_specs):
            cone = UsdGeom.Cone.Define(stage, f"{path}/cone_{i}")
            cone.CreateAxisAttr("Z")
            cone.CreateRadiusAttr(radius)
            cone.CreateHeightAttr(height)
            cone.AddTranslateOp().Set(Gf.Vec3d(*centre))
            collide(cone.GetPrim(), color)

    main = f"{prim_path}/body"
    body(main, spec.mass)
    shapes(main, spec.boxes, spec.cylinders, spec.cones)
    for s in spec.sliders:
        link = f"{prim_path}/{s.name}_link"
        body(link, s.mass)
        shapes(link, s.boxes)
        j = UsdPhysics.PrismaticJoint.Define(stage, f"{link}/{s.name}")
        j.CreateAxisAttr("Z")
        j.CreateBody0Rel().SetTargets([Sdf.Path(main)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(link)])
        j.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(float(s.lower))
        j.CreateUpperLimitAttr(float(s.upper))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(s.stiffness))
        drive.CreateDampingAttr(float(s.damping))
        drive.CreateTargetPositionAttr(float(s.rest))
    if spec.sliders:
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(main))
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(main))
        art.CreateEnabledSelfCollisionsAttr(False)
        art.CreateSolverPositionIterationCountAttr(32)
        art.CreateSolverVelocityIterationCountAttr(4)
    return root.GetPrim()


def _arena_object(spec: Part):
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class PartSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_part)
        part_name: str = ""

    class PartObject(Object):
        def _generate_articulation_cfg(self):
            actuators = {s.name: ImplicitActuatorCfg(joint_names_expr=[s.name], stiffness=s.stiffness,
                                                     damping=s.damping, friction=0.0)
                         for s in spec.sliders}
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    kind = ObjectType.ARTICULATION if spec.sliders else ObjectType.RIGID
    return PartObject(name=spec.name, object_type=kind, spawner_cfg=PartSpawnerCfg(part_name=spec.name),
                      initial_pose=Pose(position_xyz=tuple(float(v) for v in spec.pos),
                                        rotation_xyzw=tuple(float(v) for v in spec.rot_xyzw)))


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _part_state(environment, spec: Part) -> PartState:
    base = environment.unwrapped
    asset = base.scene[spec.name]
    origin = _np(base.scene.env_origins[0])
    pose = _np(asset.data.root_link_pose_w)[0]
    vel = _np(asset.data.root_vel_w)[0] if hasattr(asset.data, "root_vel_w") else np.zeros(6)
    pos, quat = pose[:3] - origin, pose[3:7]
    R = _quat_matrix((quat[3], quat[0], quat[1], quat[2]))
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * np.array(spec.size) / 2
    pts = pos + corners @ R.T
    joints = {}
    if spec.sliders:
        q = _np(asset.data.joint_pos)[0]
        joints = {n: float(q[i]) for i, n in enumerate(asset.joint_names)}
    return PartState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pts.min(0), high=pts.max(0),
                     joints=joints)


def reset_sliders(environment, name: str) -> None:
    """Put every slider of part ``name`` at 0 (at rest), with its spring's target at its rest point."""
    import torch

    spec = _SPECS.get(name)
    if spec is None or not spec.sliders:
        return
    base = environment.unwrapped
    asset = base.scene[name]
    dev = base.device
    names = list(asset.joint_names)
    q = torch.zeros((base.num_envs, asset.num_joints), dtype=torch.float32, device=dev)
    target = q.clone()
    for s in spec.sliders:
        target[:, names.index(s.name)] = float(s.rest)
    env_ids = torch.arange(base.num_envs, device=dev)
    asset.write_joint_position_to_sim_index(position=q, env_ids=env_ids)
    asset.write_joint_velocity_to_sim_index(velocity=q.clone(), env_ids=env_ids)
    asset.set_joint_position_target_index(target=target, env_ids=env_ids, full_data=True)


# -- hooking yam_arena -------------------------------------------------------------------------------------------

_STEP: list = []
_RESET: list = []
_ENV: dict = {"env": None}


def on_step(fn):
    """Register ``fn(env)`` to run after every control step, before the task file is scored."""
    _STEP.append(fn)
    return fn


def on_reset(fn):
    """Register ``fn(env)`` to run after every reset, once objects have settled."""
    _RESET.append(fn)
    return fn


def env():
    return _ENV["env"]


def _patch_scene(scene) -> None:
    if getattr(scene, "_fpkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Part):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        for n in names:
            if n in _SPECS:
                out[n] = _part_state(environment, _SPECS[n])
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._fpkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_fpkit_patched", False):
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
    cls._fpkit_patched = True


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
