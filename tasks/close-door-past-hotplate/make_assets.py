"""Writes assets/cabinet.usda: the cabinet and its door, a USD articulation with one vertical hinge.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda is committed, so only to rebuild

The file's origin is the hinge axis at table level; the task file places it there, unrotated. The layout follows
the lock-before-loading task's platform asset: a root rigid body (the carcass) carrying the articulation root and
fixed to the world by ``root_joint``, a child rigid body (the door) nested under it, and a revolute joint ``hinge``
between them. The hinge turns about +z: 0 is closed, positive opens the door toward the robot, with stops at 0 and
90 degrees. The carcass and the door don't collide with each other (the task turns self-collisions off); the
hinge's stop at 0 stands in for the carcass's front edge.
"""

from __future__ import annotations

import math
from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "cabinet.usda"

CARCASS = (0.88, 0.87, 0.83)   # white melamine
DOOR = (0.72, 0.55, 0.36)      # birch plywood


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
    cube.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[b - a for a, b in zip(lo, hi)]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def build(path: Path = OUT) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/cabinet")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/cabinet/Looks")
    m_carcass = _material(stage, "/cabinet/Looks/carcass", CARCASS)
    m_door = _material(stage, "/cabinet/Looks/door", DOOR)
    UsdGeom.Scope.Define(stage, "/cabinet/Geometry")

    # carcass (root link): an open-fronted box behind the closed door, its front face at x = DOOR_T / 2
    body = UsdGeom.Xform.Define(stage, "/cabinet/Geometry/carcass").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(5.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.05, 0.05, 0.05))
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    x0, x1 = g.DOOR_T / 2, g.DOOR_T / 2 + g.CAB_D
    w, h, p = g.DOOR_W, g.CAB_H, g.PANEL
    for name, lo, hi in (("bottom", (x0, 0.0, 0.001), (x1, w, p)),
                         ("top", (x0, 0.0, h - p), (x1, w, h)),
                         ("back", (x1 - p, 0.0, p), (x1, w, h - p)),
                         ("side_right", (x0, 0.0, p), (x1 - p, p, h - p)),
                         ("side_left", (x0, w - p, p), (x1 - p, w, h - p)),
                         ("shelf", (x0 + 0.02, p, h / 2 - p / 2), (x1 - p, w - p, h / 2 + p / 2))):
        _box(stage, f"/cabinet/Geometry/carcass/{name}", lo, hi, CARCASS, m_carcass)

    # door (child link): its frame is the hinge axis; closed, it spans y 0..DOOR_W in front of the carcass
    door = UsdGeom.Xform.Define(stage, "/cabinet/Geometry/carcass/door").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(door)
    dm = UsdPhysics.MassAPI.Apply(door)
    z0, z1 = g.DOOR_Z
    dh = z1 - z0
    dm.CreateMassAttr(g.DOOR_MASS)
    dm.CreateCenterOfMassAttr(Gf.Vec3f(0.0, w / 2, (z0 + z1) / 2))
    dm.CreateDiagonalInertiaAttr(Gf.Vec3f(g.DOOR_MASS / 12 * (w * w + dh * dh),
                                          g.DOOR_MASS / 12 * (g.DOOR_T ** 2 + dh * dh),
                                          g.DOOR_MASS / 12 * (w * w + g.DOOR_T ** 2)))
    dm.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    _box(stage, "/cabinet/Geometry/carcass/door/panel", (-g.DOOR_T / 2, 0.0, z0), (g.DOOR_T / 2, w, z1), DOOR,
         m_door)

    UsdGeom.Scope.Define(stage, "/cabinet/Physics")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/cabinet/Physics/hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/cabinet/Geometry/carcass")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/cabinet/Geometry/carcass/door")])
    hinge.CreateAxisAttr("Z")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    hinge.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    hinge.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(math.degrees(g.MAX_OPEN))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(g.HINGE_DAMPING * math.pi / 180)  # per degree, as USD drives take it
    drive.CreateMaxForceAttr(100.0)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/cabinet/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/cabinet")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/cabinet/Geometry/carcass")])
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    build()
    print(f"wrote {OUT}")
