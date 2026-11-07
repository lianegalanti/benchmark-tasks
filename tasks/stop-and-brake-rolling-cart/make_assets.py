"""Writes assets/cart.usda: the cart and its brake lever, a free-floating USD articulation with one hinge.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda is committed, so only to rebuild

The file's origin is the cart's track-frame origin (geometry.py): on the plane of its runners' undersides, under its
centre; +y points down the slope, +z away from the deck. The task file places it on the tilted track. The layout
follows the close-door-past-hotplate and lock-before-loading assets (a root rigid body carrying the articulation
root, a child rigid body nested under it, a revolute joint between them), but with no joint fixing the root to the
world, so the cart is free. The joint ``brake`` turns about +y through the lever's hinge axis; its stops are
geometry.STOWED (-40 degrees) and geometry.ENGAGED (+12 degrees). The lever is authored at its stowed angle, so the
file's rest pose is the start pose. The cart's links don't collide with each other (the task turns self-collisions
off); the lever never reaches the body in its range anyway (test_task.py checks).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "cart.usda"

BODY = (0.15, 0.45, 0.80)      # blue plastic
DARK = (0.18, 0.18, 0.20)      # runners, shoes, bracket, pad
LEVER = (0.95, 0.45, 0.08)     # orange


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _box(stage, path: str, lo, hi, rgb, material) -> None:
    """A colliding box from corner ``lo`` to ``hi`` in its parent's frame."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(a + b) / 2 for a, b in zip(lo, hi)]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(b - a) for a, b in zip(lo, hi)]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def _box_inertia(mass: float, parts) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass and diagonal inertia (about it) of ``mass`` spread uniformly over the boxes ``parts``."""
    boxes = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in parts]
    vols = np.array([np.prod(hi - lo) for lo, hi in boxes])
    masses = mass * vols / vols.sum()
    com = sum(m * (lo + hi) / 2 for m, (lo, hi) in zip(masses, boxes)) / mass
    inertia = np.zeros(3)
    for m, (lo, hi) in zip(masses, boxes):
        d = hi - lo
        c = (lo + hi) / 2 - com
        inertia += m / 12 * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
        inertia += m * np.array([c[1] ** 2 + c[2] ** 2, c[0] ** 2 + c[2] ** 2, c[0] ** 2 + c[1] ** 2])
    return com, inertia


def build(path: Path = OUT) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/cart")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/cart/Looks")
    m_body = _material(stage, "/cart/Looks/body", BODY)
    m_dark = _material(stage, "/cart/Looks/dark", DARK)
    m_lever = _material(stage, "/cart/Looks/lever", LEVER)
    UsdGeom.Scope.Define(stage, "/cart/Geometry")

    # body (root link, free): the tray, two runners, two guide shoes and the lever's bracket
    body = UsdGeom.Xform.Define(stage, "/cart/Geometry/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    parts = g.body_parts()
    com, inertia = _box_inertia(g.BODY_MASS, [parts["body"]])  # the tray carries nearly all the mass
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(g.BODY_MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    for name, (lo, hi) in parts.items():
        rgb, mat = (BODY, m_body) if name == "body" else (DARK, m_dark)
        _box(stage, f"/cart/Geometry/body/{name}", lo, hi, rgb, mat)

    # lever (child link): its frame is on the hinge axis, authored turned to the stowed angle about +y
    lever = UsdGeom.Xform.Define(stage, "/cart/Geometry/body/lever")
    lever.AddTranslateOp().Set(Gf.Vec3d(g.PIVOT_X, g.LEVER_Y, g.PIVOT_Z))
    lever.AddRotateYOp().Set(math.degrees(g.STOWED))
    lp = lever.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(lp)
    lparts = g.lever_parts()
    lcom, linertia = _box_inertia(g.LEVER_MASS, lparts.values())
    lm = UsdPhysics.MassAPI.Apply(lp)
    lm.CreateMassAttr(g.LEVER_MASS)
    lm.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in lcom]))
    lm.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in linertia]))
    lm.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    for name, (lo, hi) in lparts.items():
        rgb, mat = (DARK, m_dark) if name == "pad" else (LEVER, m_lever)
        _box(stage, f"/cart/Geometry/body/lever/{name}", lo, hi, rgb, mat)

    UsdGeom.Scope.Define(stage, "/cart/Physics")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/cart/Physics/brake")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/cart/Geometry/body")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/cart/Geometry/body/lever")])
    hinge.CreateAxisAttr("Y")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(g.PIVOT_X, g.LEVER_Y, g.PIVOT_Z))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))  # lever frame turned STOWED: the joint starts there
    hinge.CreateLowerLimitAttr(math.degrees(g.STOWED))
    hinge.CreateUpperLimitAttr(math.degrees(g.ENGAGED))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(0.002 * math.pi / 180)  # per degree, as USD drives take it
    drive.CreateMaxForceAttr(10.0)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    build()
    print(f"wrote {OUT}")
