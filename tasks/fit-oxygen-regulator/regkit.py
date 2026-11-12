"""regkit: rigid parts built from boxes and cylinders, optionally carrying a T-screw, for yam_arena task files.

Adapted from ``partkit.py`` of this repository's ``tasks/din-rail-mount`` (its license applies). Changes: parts take any
orientation (``rot_xyzw``), cylinders any axis, a part without a mass is kinematic (fixed), the spring sliders are
replaced by a :class:`Screw` (a slide and a turn joint coupled by :func:`couple_screw`), and states carry joint
positions.

A ``Part`` is one rigid body with several box and cylinder colliders. With a ``screw`` it is an articulation with a
floating root (the part) and two joints along the screw's axis (part y):

- ``<name>_slide``: a prismatic joint from the part to a light carriage, driven by a stiff position drive whose effort
  is capped (the clamp force);
- ``<name>_turn``: a revolute joint from the carriage to the screw body (shaft, tip and T-bar), free, with a little
  friction and a damping that rises with the clamp force (the loaded thread).

:func:`couple_screw`, run after every control step, sets the slide's target from the measured turn: pitch x turns.
That is the thread. Turning the T-bar moves the tip; a blocked tip stops moving while the handle keeps turning, and the
drive's force rises with the turn past first contact up to the cap, and with it the torque needed to turn the
handle.

Importing this module hooks ``yam_arena.scene`` (and the embodiment's step and reset) without modifying yam_arena.
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

__all__ = ["Part", "PartState", "Screw", "couple_screw", "env", "install", "on_reset", "on_step", "part",
           "reset_joints"]

Color = tuple[float, float, float]
# box: (centre, size, colour[, rotation as (w, x, y, z)]); cylinder: (centre, radius, height, colour[, axis "X"/"Y"/"Z"])
BoxSpec = tuple
CylSpec = tuple


@dataclass(frozen=True)
class Screw:
    """A screw along the part's y axis through ``anchor`` (part frame). Its ``boxes``/``cylinders`` are in a frame at
    the anchor (axes as the part's) and advance along -y by ``pitch`` per turn about -y."""

    anchor: tuple[float, float, float]
    cylinders: tuple[CylSpec, ...]
    boxes: tuple[BoxSpec, ...] = ()
    mass: float = 0.03
    pitch: float = 0.00175
    travel: float = 0.0125         # how far it can advance (m)
    stiffness: float = 2.0e5       # N/m, slide drive
    damping: float = 200.0         # N s/m, slide drive
    force: float = 60.0            # N, slide drive cap (the clamp force)
    turn_friction: float = 0.05    # PhysX joint friction of the turn joint, unloaded
    turn_damping: float = 0.005    # N m s/rad, unloaded
    damping_per_n: float = 0.0     # N m s/rad of extra turn damping per N of clamp force (the loaded thread)


@dataclass(frozen=True)
class Part(Box):
    """A rigid part made of ``boxes`` and ``cylinders``. It subclasses Box only so yam_arena's task-file checks accept
    it; ``size`` is its bounding box and ``pos`` is the body origin. Without a mass it is kinematic (fixed)."""

    boxes: tuple[BoxSpec, ...] = ()
    cylinders: tuple[CylSpec, ...] = ()
    rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    screw: Screw | None = None
    friction: float = 0.8
    contact_offset: float = 0.0003
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not (self.boxes or self.cylinders):
            raise ValueError(f"{self.name}: a Part needs at least one shape")
        if self.mass is not None and self.mass <= 0:
            raise ValueError(f"{self.name}: mass must be positive (or None for a fixed part)")
        if self.screw is not None and self.mass is None:
            raise ValueError(f"{self.name}: a part with a screw must be loose")


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


def _cyl_half(radius, height, axis) -> np.ndarray:
    return np.array([height / 2 if axis == "X" else radius, height / 2 if axis == "Y" else radius,
                     height / 2 if axis == "Z" else radius])


def part(name, pos, boxes=(), mass=None, rot_xyzw=(0.0, 0.0, 0.0, 1.0), color=(0.5, 0.5, 0.5), friction=0.8,
         cylinders=(), screw=None) -> Part:
    """Build a Part; ``size`` is filled in from the shapes' extent (part frame)."""
    half = [np.abs(_quat_matrix(q)) @ (np.array(s) / 2) for _, s, _, q in map(_box, boxes)]
    half += [_cyl_half(r, h, a) for _, r, h, _, a in map(_cyl, cylinders)]
    centres = [np.array(b[0]) for b in boxes] + [np.array(c[0]) for c in cylinders]
    lo = np.min([c - h for c, h in zip(centres, half)], axis=0)
    hi = np.max([c + h for c, h in zip(centres, half)], axis=0)
    return Part(name=name, pos=tuple(float(v) for v in pos), size=tuple(float(v) for v in hi - lo), mass=mass,
                color=color, boxes=tuple(boxes), cylinders=tuple(cylinders),
                rot_xyzw=tuple(float(v) for v in rot_xyzw), screw=screw, friction=friction)


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
    """Author the part under ``prim_path`` (pxr only): a plain Xform holding the body (and the screw's bodies)."""
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
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    def body(path: str, pos, mass: float | None, inertia=None):
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*pos))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(1, 0, 0, 0))
        prim = xform.GetPrim()
        rbody = UsdPhysics.RigidBodyAPI.Apply(prim)
        if mass is None:
            rbody.CreateKinematicEnabledAttr(True)
        else:
            m = UsdPhysics.MassAPI.Apply(prim)
            m.CreateMassAttr(mass)
            if inertia is not None:
                m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(1.0)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateAngularDampingAttr(1.0)  # light damping: small stiff parts otherwise chatter in tight fits
        return prim

    def collide(prim, color):
        UsdPhysics.CollisionAPI.Apply(prim)
        cp = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        cp.CreateContactOffsetAttr(spec.contact_offset)  # below the fits' clearances
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(str(prim.GetPath()), phys, stage=stage)
        bind_visual_material(str(prim.GetPath()), look(color), stage=stage)

    def shapes(path: str, box_specs, cyl_specs):
        for i, (centre, size, color, rot) in enumerate(map(_box, box_specs)):
            cube = UsdGeom.Cube.Define(stage, f"{path}/box_{i}")
            cube.CreateSizeAttr(1.0)
            cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
            if rot != (1.0, 0.0, 0.0, 0.0):
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

    main = f"{prim_path}/body"
    body(main, (0.0, 0.0, 0.0), spec.mass)
    shapes(main, spec.boxes, spec.cylinders)
    s = spec.screw
    if s is not None:
        carriage, screw = f"{prim_path}/carriage", f"{prim_path}/screw"
        body(carriage, s.anchor, 0.005, inertia=(2e-7, 2e-7, 2e-7))
        body(screw, s.anchor, s.mass)
        shapes(screw, s.boxes, s.cylinders)
        joint_name = cfg.part_name
        slide = UsdPhysics.PrismaticJoint.Define(stage, f"{carriage}/{joint_name}_slide")
        slide.CreateAxisAttr("Y")
        slide.CreateBody0Rel().SetTargets([Sdf.Path(main)])
        slide.CreateBody1Rel().SetTargets([Sdf.Path(carriage)])
        slide.CreateLocalPos0Attr(Gf.Vec3f(*s.anchor))
        slide.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        slide.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        slide.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        slide.CreateLowerLimitAttr(-s.travel)
        slide.CreateUpperLimitAttr(0.0005)
        drive = UsdPhysics.DriveAPI.Apply(slide.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(s.stiffness)
        drive.CreateDampingAttr(s.damping)
        drive.CreateMaxForceAttr(s.force)
        drive.CreateTargetPositionAttr(0.0)
        turn = UsdPhysics.RevoluteJoint.Define(stage, f"{screw}/{joint_name}_turn")
        turn.CreateAxisAttr("Y")
        turn.CreateBody0Rel().SetTargets([Sdf.Path(carriage)])
        turn.CreateBody1Rel().SetTargets([Sdf.Path(screw)])
        turn.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
        turn.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        turn.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        turn.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
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
            s = spec.screw
            actuators = {
                "slide": ImplicitActuatorCfg(joint_names_expr=[f"{spec.name}_slide"], stiffness=s.stiffness,
                                             damping=s.damping, joint_effort_limit=s.force),
                "turn": ImplicitActuatorCfg(joint_names_expr=[f"{spec.name}_turn"], stiffness=0.0,
                                            damping=s.turn_damping, friction=s.turn_friction),
            }
            cfg = ArticulationCfg(prim_path=self.prim_path, spawn=self.spawner_cfg, actuators=actuators)
            return self._add_initial_pose_to_cfg(cfg)

    kind = ObjectType.ARTICULATION if spec.screw is not None else ObjectType.RIGID
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
    if spec.screw is not None:
        q = _np(asset.data.joint_pos)[0]
        joints = {n: float(q[i]) for i, n in enumerate(asset.joint_names)}
    return PartState(pos=pos, quat_xyzw=quat, lin_vel=vel[:3], ang_vel=vel[3:6], low=pts.min(0), high=pts.max(0),
                     joints=joints)


def unwrap(last: float, theta: float, raw: float) -> float:
    """The unwrapped angle after a wrapped reading ``raw`` that follows ``last`` (it moves less than half a turn)."""
    return theta + (raw - last + math.pi) % (2 * math.pi) - math.pi


_TURNS: dict[str, tuple[float, float]] = {}
_DAMPING: dict[str, float] = {}   # per screw: (last measured angle, unwrapped angle)


def couple_screw(environment, name: str) -> tuple[float, float]:
    """The thread: set the slide's target to pitch x the measured turn (clamped to its travel). Returns the measured
    slide and the turn, unwrapped (rad, counting whole turns)."""
    import torch

    spec = _SPECS[name]
    s = spec.screw
    asset = environment.unwrapped.scene[name]
    names = list(asset.joint_names)
    q = _np(asset.data.joint_pos)[0]
    d, raw = float(q[names.index(f"{name}_slide")]), float(q[names.index(f"{name}_turn")])
    # PhysX reports a free revolute joint's angle wrapped to (-pi, pi]: count the turns (it moves < pi per step)
    last, theta = _TURNS.get(name, (raw, raw))
    theta = unwrap(last, theta, raw)
    _TURNS[name] = (raw, theta)
    target = min(0.0, max(-s.travel, s.pitch * theta / (2 * math.pi)))
    ids = asset.find_joints([f"{name}_slide"])[0]
    origin = _t(environment.unwrapped.scene.env_origins)
    value = torch.tensor([[target]], dtype=origin.dtype, device=origin.device)
    asset.actuators.target_command.set_position_index(value=value, joint_ids=ids)
    # the loaded thread: turning gets harder as the clamp force rises (as damping on the turn joint)
    force = min(s.force, max(0.0, s.stiffness * (d - target)))
    damping = turn_damping(s, force)
    if abs(damping - _DAMPING.get(name, -1.0)) > 0.01 * damping:
        value = torch.tensor([[damping]], dtype=origin.dtype, device=origin.device)
        asset.write_joint_damping_to_sim_index(damping=value, joint_ids=asset.find_joints([f"{name}_turn"])[0])
        _DAMPING[name] = damping
    return d, theta


def turn_damping(s: Screw, force: float) -> float:
    """Damping of the turn joint (N m s/rad) with the tip pressing with ``force`` (N)."""
    return s.turn_damping + s.damping_per_n * force


def reset_joints(environment, name: str) -> None:
    """Put the screw back at its start (both joints and the slide's target at 0)."""
    import torch

    _TURNS.pop(name, None)
    _DAMPING.pop(name, None)
    asset = environment.unwrapped.scene[name]
    origin = _t(environment.unwrapped.scene.env_origins)
    zeros = torch.zeros((1, asset.num_joints), dtype=origin.dtype, device=origin.device)
    asset.write_joint_position_to_sim_index(position=zeros)
    asset.write_joint_velocity_to_sim_index(velocity=zeros.clone())
    asset.actuators.target_command.set_position_index(value=zeros.clone())


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
    if getattr(scene, "_regkit_patched", False):
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
    scene._regkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_regkit_patched", False):
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
    cls._regkit_patched = True


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
