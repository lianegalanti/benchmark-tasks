"""Writes the task's USD assets from geometry.py:

- assets/holder.usda: the one-touch phone holder, a fixed-base articulation in the holder frame H (see geometry.py).
  Root link ``body`` (back plate, feet, housing, vent clip) is fixed to the world by ``root_joint``. Child links:
  ``arm_left`` and ``arm_right`` on prismatic joints along y (positive opens; ``arm_right`` follows ``arm_left``
  through a mimic constraint, standing in for the rack and pinion that gears them together) and ``trigger`` on a
  prismatic joint along +x (positive presses it in). Links joined to the root don't collide with it (as in any
  articulation); the arms and the trigger do collide with each other, which is how the tails meet the catches.
- assets/phone.usda: the phone, one rigid body (aluminium frame, glass faces; a dark screen panel marks the front).

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

BLACK = (0.08, 0.08, 0.09)       # holder plastic
GREY = (0.35, 0.36, 0.38)        # silicone pads
INNER = (0.85, 0.85, 0.80)       # POM internals (catch, tails)
FRAME = (0.75, 0.74, 0.70)       # phone: natural aluminium
SCREEN = (0.02, 0.03, 0.05)

FRICTION = {"abs": (0.4, 0.35), "silicone": (1.0, 0.9), "pom": (0.2, 0.18)}


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, static: float, dynamic: float):
    from pxr import UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    return mat


def _box(stage, path: str, centre, size, yaw: float, rgb, look, physics=None, collide: bool = True) -> None:
    """A box in its parent's frame, turned ``yaw`` (rad) about z."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if yaw:
        cube.AddRotateZOp().Set(float(math.degrees(yaw)))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
    binding.Bind(look)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        binding.Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")


def _lohi(lo, hi):
    return [(a + b) / 2 for a, b in zip(lo, hi)], [b - a for a, b in zip(lo, hi)]


def _stage(root: str):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, f"/{root}")
    stage.SetDefaultPrim(prim.GetPrim())
    UsdGeom.Scope.Define(stage, f"/{root}/Looks")
    return stage


def _mass(prim, mass, com, size) -> None:
    from pxr import Gf, UsdPhysics

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    a, b, c = size
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(mass / 12 * (b * b + c * c), mass / 12 * (a * a + c * c),
                                         mass / 12 * (a * a + b * b)))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _mirror(lo, hi):
    return (lo[0], -hi[1], lo[2]), (hi[0], -lo[1], hi[2])


def _prismatic(stage, name: str, child: str, axis: str, flip: bool, upper: float, rate: float, damping: float):
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.PrismaticJoint.Define(stage, f"/holder/Physics/{name}")
    j.CreateBody0Rel().SetTargets([Sdf.Path("/holder/Geometry/body")])
    j.CreateBody1Rel().SetTargets([Sdf.Path(f"/holder/Geometry/body/{child}")])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # the right arm's joint frame is turned 180 degrees about z on both sides, so its +y is the station's -y
    rot = Gf.Quatf(0.0, 0.0, 0.0, 1.0) if flip else Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    j.CreateLocalRot0Attr(rot)
    j.CreateLocalRot1Attr(rot)
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(upper))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(rate))
    drive.CreateDampingAttr(float(damping))
    drive.CreateMaxForceAttr(100.0)
    return j


def build_holder(path: Path) -> None:
    from pxr import Sdf, UsdGeom, UsdPhysics

    stage = _stage("holder")
    look = {"black": _material(stage, "/holder/Looks/black", BLACK), "grey": _material(stage, "/holder/Looks/grey", GREY),
            "inner": _material(stage, "/holder/Looks/inner", INNER)}
    phys = {k: _physics_material(stage, f"/holder/Looks/{k}_physics", *v) for k, v in FRICTION.items()}
    UsdGeom.Scope.Define(stage, "/holder/Geometry")

    body = UsdGeom.Xform.Define(stage, "/holder/Geometry/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    _mass(body, g.BODY_MASS, (0.02, 0.0, 0.045), (0.045, 0.088, 0.10))
    parts = [("plate", *g.PLATE), ("foot_left", *g.FEET[0]), ("foot_right", *g.FEET[1]), ("housing", *g.HOUSING),
             ("clip", *g.CLIP)]
    for name, lo, hi in parts:
        c, s = _lohi(lo, hi)
        _box(stage, f"/holder/Geometry/body/{name}", c, s, 0.0, BLACK, look["black"], phys["abs"])

    for side, sign in (("left", 1.0), ("right", -1.0)):
        link = UsdGeom.Xform.Define(stage, f"/holder/Geometry/body/arm_{side}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        _mass(link, g.ARM_MASS, (0.0, sign * 0.034, 0.07), (0.024, 0.01, 0.03))
        for name, lo, hi in g.ARM_BOXES:
            if sign < 0:
                lo, hi = _mirror(lo, hi)
            c, s = _lohi(lo, hi)
            key = {"pad": "silicone", "lip": "silicone", "tail": "pom", "bar": "pom"}.get(name, "abs")
            rgb, lk = {"silicone": (GREY, look["grey"]), "pom": (INNER, look["inner"])}.get(key, (BLACK, look["black"]))
            _box(stage, f"/holder/Geometry/body/arm_{side}/{name}", c, s, 0.0, rgb, lk, phys[key])

    trig = UsdGeom.Xform.Define(stage, "/holder/Geometry/body/trigger").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(trig)
    _mass(trig, g.TRIG_MASS, (0.01, 0.0, 0.044), (0.027, 0.02, 0.02))
    for name, lo, hi in g.TRIGGER_BOXES:
        c, s = _lohi(lo, hi)
        _box(stage, f"/holder/Geometry/body/trigger/{name}", c, s, 0.0,
             BLACK if name == "pad" else INNER, look["black"] if name == "pad" else look["inner"],
             phys["abs"] if name == "pad" else phys["pom"])
    for side, sign in (("left", 1.0), ("right", -1.0)):
        for name, c, s, yaw in g.catch_boxes(sign):
            _box(stage, f"/holder/Geometry/body/trigger/{name}_{side}", c, s, yaw, INNER, look["inner"], phys["pom"])

    UsdGeom.Scope.Define(stage, "/holder/Physics")
    _prismatic(stage, "arm_left", "arm_left", "Y", False, g.ARM_STOP, g.ARM_RATE, g.ARM_DAMPING)
    right = _prismatic(stage, "arm_right", "arm_right", "Y", True, g.ARM_STOP, g.ARM_RATE, g.ARM_DAMPING)
    _prismatic(stage, "trigger", "trigger", "X", False, g.TRIG_STOP, g.TRIG_RATE, g.TRIG_DAMPING)
    # the gearing: arm_right = arm_left (both positive opening). NewtonMimicAPI is not in usd-core: author it raw.
    prim = right.GetPrim()
    prim.SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["PhysicsDriveAPI:linear", "NewtonMimicAPI"]))
    prim.CreateAttribute("newton:mimicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    prim.CreateRelationship("newton:mimicJoint").SetTargets([Sdf.Path("/holder/Physics/arm_left")])
    prim.CreateAttribute("newton:mimicCoef0", Sdf.ValueTypeNames.Float).Set(0.0)
    prim.CreateAttribute("newton:mimicCoef1", Sdf.ValueTypeNames.Float).Set(1.0)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/holder/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/holder")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/holder/Geometry/body")])
    stage.GetRootLayer().Export(str(path))


def build_phone(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("phone")
    frame = _material(stage, "/phone/Looks/frame", FRAME)
    screen = _material(stage, "/phone/Looks/screen", SCREEN)
    glass = _physics_material(stage, "/phone/Looks/phone_physics", 0.5, 0.4)
    body = stage.GetPrimAtPath("/phone")
    UsdPhysics.RigidBodyAPI.Apply(body)
    size = (g.PHONE_T, g.PHONE_W, g.PHONE_L)
    _mass(body, g.PHONE_MASS, (0.0, 0.0, 0.0), size)
    _box(stage, "/phone/body", (0.0, 0.0, 0.0), size, 0.0, FRAME, frame, glass)
    # the screen (front, -x in the phone's frame): a thin dark panel, visual only
    _box(stage, "/phone/screen", (-g.PHONE_T / 2 - 0.0002, 0.0, 0.0), (0.0004, g.PHONE_W - 0.004, g.PHONE_L - 0.006),
         0.0, SCREEN, screen, collide=False)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_holder(OUT / "holder.usda")
    build_phone(OUT / "phone.usda")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.usda"))))
