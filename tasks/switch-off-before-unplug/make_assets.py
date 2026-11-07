"""Writes assets/strip.usda (the power strip and its rocker switch, a USD articulation with one hinge) and
assets/plug.usda (the heater's plug, one rigid body).

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed, so only to rebuild

The strip follows the layout of the lock-before-loading task's platform asset (as close-door-past-hotplate's
make_assets.py does): a root rigid body (the housing) carrying the articulation root and fixed to the world by
``root_joint``, a child rigid body (the rocker paddle) nested under it, and a revolute joint ``rocker`` between them,
about +x through the paddle's centre. Its origin is the centre of the strip's footprint on the table; the task file
places it there, unrotated. The joint's 0 is the USD pose and is ON; it stops at 0 and 24 degrees (OFF). The paddle's
centre of mass is on the hinge axis, so gravity doesn't turn it; the task applies the over-centre spring as a joint
torque. The housing and the paddle don't collide with each other (the task turns self-collisions off).
"""

from __future__ import annotations

import math
from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

WHITE = (0.92, 0.92, 0.90)     # strip housing, white ABS
RED = (0.85, 0.08, 0.06)       # rocker paddle, translucent red
BLACK = (0.08, 0.08, 0.08)     # plug body and cord
BRASS = (0.80, 0.70, 0.40)     # nickel-plated brass pins


def _stage(root: str):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, root)
    stage.SetDefaultPrim(prim.GetPrim())
    UsdGeom.Scope.Define(stage, f"{root}/Looks")
    return stage


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _bind(prim, rgb, material) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)


def _box(stage, path: str, lo, hi, rgb, material, rot_x_deg: float = 0.0) -> None:
    """A colliding box from corner ``lo`` to ``hi`` in its parent's frame, turned ``rot_x_deg`` about +x through
    the parent's origin."""
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    if rot_x_deg:
        cube.AddRotateXOp().Set(rot_x_deg)
    cube.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[b - a for a, b in zip(lo, hi)]))
    _bind(cube.GetPrim(), rgb, material)


def _cylinder(stage, path: str, centre, radius: float, height: float, axis: str, rgb, material) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.AddTranslateOp().Set(Gf.Vec3d(*centre))
    _bind(cyl.GetPrim(), rgb, material)


def build_strip(path: Path = OUT / "strip.usda") -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("/strip")
    m_white = _material(stage, "/strip/Looks/housing", WHITE)
    m_red = _material(stage, "/strip/Looks/rocker", RED)
    UsdGeom.Scope.Define(stage, "/strip/Geometry")

    body = UsdGeom.Xform.Define(stage, "/strip/Geometry/housing").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(0.5)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.005, 0.005, 0.005))
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    for name, lo, hi in g.housing_boxes():
        _box(stage, f"/strip/Geometry/housing/{name}", lo, hi, WHITE, m_white)

    # the paddle (child link): its frame sits on the hinge axis, the box turned to its ON tilt (I end, +y, down)
    rocker = UsdGeom.Xform.Define(stage, "/strip/Geometry/housing/rocker")
    rocker.AddTranslateOp().Set(Gf.Vec3d(*map(float, g.PIVOT)))
    prim = rocker.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    rm = UsdPhysics.MassAPI.Apply(prim)
    rm.CreateMassAttr(g.PADDLE_MASS)
    rm.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    rm.CreateDiagonalInertiaAttr(Gf.Vec3f(2e-6, 2e-6, 2e-6))
    rm.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    hx, hy, hz = g.PADDLE_HALF
    _box(stage, "/strip/Geometry/housing/rocker/paddle", (-hx, -hy, -hz), (hx, hy, hz), RED, m_red,
         rot_x_deg=-math.degrees(g.ROCK))

    UsdGeom.Scope.Define(stage, "/strip/Physics")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/strip/Physics/rocker")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/strip/Geometry/housing")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/strip/Geometry/housing/rocker")])
    hinge.CreateAxisAttr("X")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(*map(float, g.PIVOT)))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    hinge.CreateLowerLimitAttr(math.degrees(g.Q_ON))
    hinge.CreateUpperLimitAttr(math.degrees(g.Q_OFF))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(g.HINGE_DAMPING * math.pi / 180)  # per degree, as USD drives take it
    drive.CreateMaxForceAttr(10.0)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/strip/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/strip")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/strip/Geometry/housing")])
    stage.GetRootLayer().Export(str(path))


def build_plug(path: Path = OUT / "plug.usda") -> None:
    from pxr import UsdPhysics

    stage = _stage("/plug")
    prim = stage.GetPrimAtPath("/plug")
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(g.PLUG_MASS)
    m_black = _material(stage, "/plug/Looks/body", BLACK)
    m_pin = _material(stage, "/plug/Looks/pins", BRASS)
    bx, by = g.BODY_HALF
    _box(stage, "/plug/body", (-bx, -by, 0.0), (bx, by, g.BODY_H), BLACK, m_black)
    for k, sy in enumerate((-1, 1)):
        _cylinder(stage, f"/plug/pin{k}", (0.0, sy * g.PIN_DY, -g.PIN_LEN / 2), g.PIN_R, g.PIN_LEN, "Z", BRASS,
                  m_pin)
    _cylinder(stage, "/plug/cord", (bx + g.CORD_LEN / 2, 0.0, g.CORD_Z), g.CORD_R, g.CORD_LEN, "X", BLACK, m_black)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_strip()
    build_plug()
    print(f"wrote {OUT / 'strip.usda'} and {OUT / 'plug.usda'}")
