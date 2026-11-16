"""Writes the task's USD assets from geometry.py:

- assets/hub.usda: the hub and brake rotor on their bench stand, one USD articulation fixed to the world by
  ``root_joint``. Root link ``base``: the stand, the rotor's friction ring and hat, the hat's face (with an opening
  round each screw head), the hub pilot and the four wheel studs. Child link ``screw_a`` (the other screw is already out; its hole is open):
  countersunk screw's head with its PH3 recess, on a revolute joint about its own axis; the task drives each joint to
  the angle the blows have turned it (a stiff drive: the screw is seized).
- assets/block.usda: the hardwood block the driver stands in (kinematic).
- assets/stand.usda: the stepped hardwood stand the hammer stands on (kinematic).
- assets/driver.usda: the hand impact driver, a free articulation: root link ``body`` (the knurled body and the striking
  cap); ``slider`` on the prismatic joint ``stroke`` along the driver's axis (3 mm, the return spring is its drive);
  ``anvil`` (nose, bit holder and PH3 bit) on the revolute joint ``cam`` about the axis, which the task drives from the
  stroke (the helical cam).
- assets/hammer.usda: the 16 oz ball-peen hammer, one rigid body.

The hammer and the driver's body carry PhysX's contact-report API, so the task can read the contact impulse between
the hammer and the driver's cap every physics step.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

IRON = (0.33, 0.32, 0.31)        # cast-iron rotor
RUST = (0.45, 0.28, 0.18)        # the hat's rusty face
STEEL = (0.40, 0.41, 0.43)       # stand
ZINC = (0.72, 0.71, 0.66)        # studs, pilot, screw heads
DARK = (0.08, 0.08, 0.08)        # the recess's cross
BLACK = (0.12, 0.12, 0.13)       # the driver's black-oxide body
CHROME = (0.78, 0.79, 0.80)      # the driver's cap and nose, the bit, the hammer's head
HICKORY = (0.78, 0.60, 0.36)     # hammer handle
WOOD = (0.62, 0.45, 0.28)        # bench block

FRICTION = {"steel": (0.40, 0.35), "iron": (0.45, 0.40), "knurl": (0.9, 0.8), "grip": (0.9, 0.8),
            "wood": (0.6, 0.5), "hard": (0.20, 0.15)}   # hard: the hammer's face and the striking cap (hardened, polished)


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


def _bind(prim, look, phys) -> None:
    from pxr import UsdPhysics, UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(look)
    if phys is not None:
        UsdPhysics.CollisionAPI.Apply(prim)
        api.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, centre, size, rot, rgb, look, phys) -> None:
    """A box in its link's frame (rotation matrix ``rot``); colliding unless ``phys`` is None."""
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if not np.allclose(rot, np.eye(3)):
        cube.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot)]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _bind(cube.GetPrim(), look, phys)


def _cyl(stage, path: str, centre, radius, length, axis, rgb, look, phys) -> None:
    """A colliding cylinder along ``axis`` ("x"/"y"/"z")."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(length))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if axis == "y":
        cyl.AddRotateXOp().Set(90.0)
    elif axis == "x":
        cyl.AddRotateYOp().Set(90.0)
    _bind(cyl.GetPrim(), look, phys)


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


def _mass(prim, mass, com, inertia) -> None:
    from pxr import Gf, UsdPhysics

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _contact_report(prim) -> None:
    """PhysX contact reporting on a rigid body (the schema is PhysX's; usd-core only writes its name and attribute)."""
    from pxr import Sdf

    prim.AddAppliedSchema("PhysxContactReportAPI")
    prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.0)


def _link(stage, path: str, pos=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    return x.GetPrim()


def build_hub(path: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("hub")
    looks = {k: _material(stage, f"/hub/Looks/{k}", v) for k, v in
             (("iron", IRON), ("rust", RUST), ("steel", STEEL), ("zinc", ZINC), ("dark", DARK))}
    phys = {k: _physics_material(stage, f"/hub/Looks/{k}_physics", *FRICTION[k]) for k in ("steel", "iron")}
    UsdGeom.Scope.Define(stage, "/hub/Geometry")

    base = _link(stage, "/hub/Geometry/base")
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(base, 9.0, (0.0, 0.0, 0.07), (0.05, 0.05, 0.08))
    for name, c, s, r in g.fixture_boxes():
        if name == "foot":
            rgb, look = STEEL, looks["steel"]
        elif name.startswith("face") or name.startswith("csk"):
            rgb, look = RUST, looks["rust"]
        else:
            rgb, look = IRON, looks["iron"]
        _box(stage, f"/hub/Geometry/base/{name}", c, s, r, rgb, look, phys["iron"])
    for name, c, rad, length, ax in g.fixture_cylinders():
        if name in ("post",):
            rgb, look = STEEL, looks["steel"]
        elif name == "hat":
            rgb, look = IRON, looks["iron"]
        else:
            rgb, look = ZINC, looks["zinc"]
        _cyl(stage, f"/hub/Geometry/base/{name}", c, rad, length, ax, rgb, look, phys["steel"])

    UsdGeom.Scope.Define(stage, "/hub/Physics")
    for key, pos in g.SCREWS.items():
        link = _link(stage, f"/hub/Geometry/base/screw_{key}", pos)
        _mass(link, 0.02, (0.0, 0.0, -0.003), (2e-5, 2e-5, 2e-5))
        for name, c, s, r in g.screw_boxes():
            _box(stage, f"/hub/Geometry/base/screw_{key}/{name}", c, s, r, ZINC, looks["zinc"], phys["steel"])
        for name, c, s, r in g.screw_marks():
            _box(stage, f"/hub/Geometry/base/screw_{key}/{name}", c, s, r, DARK, looks["dark"], None)
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/hub/Physics/screw_{key}")
        j.CreateBody0Rel().SetTargets([Sdf.Path("/hub/Geometry/base")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/hub/Geometry/base/screw_{key}")])
        j.CreateAxisAttr("Z")
        j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos]))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(SCREW_K * math.pi / 180))
        drive.CreateDampingAttr(float(SCREW_D * math.pi / 180))
        drive.CreateMaxForceAttr(float(SCREW_CAP))
        drive.CreateTargetPositionAttr(0.0)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/hub/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/hub")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/hub/Geometry/base")])
    stage.GetRootLayer().Export(str(path))


# The screws' drive: stiff, and capped far above anything a hand can put through a driver (they are seized)
SCREW_K, SCREW_D, SCREW_CAP = 50.0, 1.0, 100.0


def build_block(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("block")
    look = _material(stage, "/block/Looks/wood", WOOD)
    phys = _physics_material(stage, "/block/Looks/wood_physics", *FRICTION["wood"])
    body = stage.GetPrimAtPath("/block")
    UsdPhysics.RigidBodyAPI.Apply(body)
    for name, c, s, r in g.block_boxes():
        _box(stage, f"/block/{name}", c, s, r, WOOD, look, phys)
    stage.GetRootLayer().Export(str(path))


def build_stand(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("stand")
    look = _material(stage, "/stand/Looks/wood", WOOD)
    phys = _physics_material(stage, "/stand/Looks/wood_physics", *FRICTION["wood"])
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath("/stand"))
    for name, c, s, r in g.stand_boxes():
        _box(stage, f"/stand/{name}", c, s, r, WOOD, look, phys)
    stage.GetRootLayer().Export(str(path))


def build_driver(path: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("driver")
    looks = {"black": _material(stage, "/driver/Looks/black", BLACK),
             "chrome": _material(stage, "/driver/Looks/chrome", CHROME)}
    phys = {"knurl": _physics_material(stage, "/driver/Looks/knurl_physics", *FRICTION["knurl"]),
            "steel": _physics_material(stage, "/driver/Looks/steel_physics", *FRICTION["steel"]),
            "hard": _physics_material(stage, "/driver/Looks/hard_physics", *FRICTION["hard"])}
    UsdGeom.Scope.Define(stage, "/driver/Geometry")

    body = _link(stage, "/driver/Geometry/body")
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _contact_report(body)
    m = g.BODY_MASS
    _mass(body, m, (0.0, 0.0, 0.11), (m * (3 * 0.018 ** 2 + 0.12 ** 2) / 12, m * (3 * 0.018 ** 2 + 0.12 ** 2) / 12,
                                       m * 0.018 ** 2 / 2))
    for name, c, rad, length, ax in g.body_cylinders():
        if name == "body":       # the knurled body: a 12-sided prism of facets round a core
            for fname, fc, fs, fr in g.driver_body_boxes():
                _box(stage, f"/driver/Geometry/body/{fname}", fc, fs, fr, BLACK, looks["black"], phys["knurl"])
            _cyl(stage, "/driver/Geometry/body/core", c, rad - 0.006, length, ax, BLACK, looks["black"], None)
            continue
        _cyl(stage, f"/driver/Geometry/body/{name}", c, rad, length, ax, CHROME, looks["chrome"],
             phys["hard" if name == "cap" else "steel"])

    slider = _link(stage, "/driver/Geometry/body/slider")
    _mass(slider, g.SLIDER_MASS, (0.0, 0.0, 0.04), (2e-6, 2e-6, 2e-6))
    anvil = _link(stage, "/driver/Geometry/body/slider/anvil")
    _mass(anvil, g.ANVIL_MASS, (0.0, 0.0, 0.035), (2e-5, 2e-5, 8e-6))
    for name, c, rad, length, ax in g.anvil_cylinders():
        _cyl(stage, f"/driver/Geometry/body/slider/anvil/{name}", c, rad, length, ax, CHROME, looks["chrome"],
             phys["steel"])

    UsdGeom.Scope.Define(stage, "/driver/Physics")
    j = UsdPhysics.PrismaticJoint.Define(stage, "/driver/Physics/stroke")
    j.CreateBody0Rel().SetTargets([Sdf.Path("/driver/Geometry/body")])
    j.CreateBody1Rel().SetTargets([Sdf.Path("/driver/Geometry/body/slider")])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(g.STROKE))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(g.SPRING_K))
    drive.CreateDampingAttr(float(SPRING_D))
    drive.CreateMaxForceAttr(1000.0)
    drive.CreateTargetPositionAttr(float(SPRING_REST))

    j = UsdPhysics.RevoluteJoint.Define(stage, "/driver/Physics/cam")
    j.CreateBody0Rel().SetTargets([Sdf.Path("/driver/Geometry/body/slider")])
    j.CreateBody1Rel().SetTargets([Sdf.Path("/driver/Geometry/body/slider/anvil")])
    j.CreateAxisAttr("Z")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(CAM_K * math.pi / 180))
    drive.CreateDampingAttr(float(CAM_D * math.pi / 180))
    drive.CreateMaxForceAttr(50.0)
    drive.CreateTargetPositionAttr(0.0)
    stage.GetRootLayer().Export(str(path))


# The return spring: a drive toward a rest point 4 mm beyond the stroke's open end (20 N preload at the stop); the cam:
# a stiff drive the task points at CAM x stroke every physics step.
SPRING_REST = -g.SPRING_PRELOAD / g.SPRING_K
SPRING_D = 5.0
CAM_K, CAM_D = 20.0, 0.2


def build_hammer(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("hammer")
    looks = {"head": _material(stage, "/hammer/Looks/head", CHROME),
             "handle": _material(stage, "/hammer/Looks/handle", HICKORY)}
    phys = {"hard": _physics_material(stage, "/hammer/Looks/hard_physics", *FRICTION["hard"]),
            "grip": _physics_material(stage, "/hammer/Looks/grip_physics", *FRICTION["grip"])}
    body = stage.GetPrimAtPath("/hammer")
    UsdPhysics.RigidBodyAPI.Apply(body)
    _contact_report(body)
    L = g.HANDLE[1]
    ixx = g.HEAD_MASS * 0.1 ** 2 / 12 + g.HANDLE_MASS * L ** 2 / 3
    _mass(body, g.HAMMER_MASS, g.HAMMER_COM, (ixx, g.HEAD_MASS * 0.0145 ** 2 / 2 + 0.0005, ixx))
    for name, c, s, r in g.hammer_boxes():
        _box(stage, f"/hammer/{name}", c, s, r, HICKORY, looks["handle"], phys["grip"])
    for name, c, rad, length, ax in g.hammer_cylinders():
        _cyl(stage, f"/hammer/{name}", c, rad, length, ax, CHROME, looks["head"], phys["hard"])
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_hub(OUT / "hub.usda")
    build_block(OUT / "block.usda")
    build_stand(OUT / "stand.usda")
    build_driver(OUT / "driver.usda")
    build_hammer(OUT / "hammer.usda")
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
