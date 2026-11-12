"""Writes the task's USD assets from geometry.py:

- assets/printer.usda: the printer, a free-standing (not fixed) USD articulation: the body (case, bay, guide rails,
  seat notches, transfer roller) as the root link, and the front door on a revolute joint ``hinge`` about the bay
  opening's bottom front edge. 0 is shut, positive opens the door toward the robot, with stops at 0 and 80 degrees.
  The door and the body don't collide with each other (the task turns self-collisions off); the hinge's stop at 0
  stands in for the door's seat on the case.
- assets/cartridge_old.usda, assets/cartridge_new.usda: the two toner cartridges, one rigid body each (body, handle,
  two keyed side bosses).

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

CASE = (0.86, 0.86, 0.84)       # light grey ABS
DOOR = (0.70, 0.71, 0.70)
RAIL = (0.25, 0.26, 0.28)       # dark grey guides inside the bay
ROLLER = (0.05, 0.05, 0.05)
CART = (0.06, 0.06, 0.07)       # black cartridge body
HANDLE = {"old": (0.45, 0.46, 0.48), "new": (0.95, 0.45, 0.05)}   # grey handle (old), orange handle (new)


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, static: float, dynamic: float, combine: str):
    """A physics material (friction, and how it combines with the other surface's)."""
    from pxr import Sdf, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def _bind_physics(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, centre, size, pitch: float, rgb, material) -> None:
    """A colliding box in its parent's frame, turned ``pitch`` (rad) about +y."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if pitch:
        cube.AddRotateYOp().Set(float(math.degrees(pitch)))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


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


def build_printer(path: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("printer")
    m_case = _material(stage, "/printer/Looks/case", CASE)
    m_rail = _material(stage, "/printer/Looks/rail", RAIL)
    m_roll = _material(stage, "/printer/Looks/roller", ROLLER)
    m_door = _material(stage, "/printer/Looks/door", DOOR)
    # physics: smooth ABS (rails, case, door; "min", so it rules contacts with the cartridges and grippers) and rubber
    # feet ("max", so they grip the table)
    p_abs = _physics_material(stage, "/printer/Looks/abs_physics", *g.ABS_FRICTION, "min")
    p_rubber = _physics_material(stage, "/printer/Looks/rubber_physics", *g.FOOT_FRICTION, "max")
    UsdGeom.Scope.Define(stage, "/printer/Geometry")

    body = UsdGeom.Xform.Define(stage, "/printer/Geometry/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(body, g.P_MASS, (g.P_DEPTH * 0.6, 0.0, 0.09), (g.P_DEPTH, 2 * g.P_HALF_W, g.P_HEIGHT))
    for name, c, s, p in g.printer_boxes():
        inside = name.startswith(("rail", "flare", "seat", "stop"))
        _box(stage, f"/printer/Geometry/body/{name}", c, s, p, RAIL if inside else CASE, m_rail if inside else m_case)
        _bind_physics(stage.GetPrimAtPath(f"/printer/Geometry/body/{name}"),
                      p_rubber if name.startswith("foot") else p_abs)
    roller = UsdGeom.Cylinder.Define(stage, "/printer/Geometry/body/transfer_roller")
    # a z-axis cylinder turned onto y by its transform (the physics engine's cylinder collider follows the prim's
    # transform, not every authoring of the axis attribute)
    roller.CreateAxisAttr("Z")
    roller.CreateRadiusAttr(float(g.ROLLER[2]))
    roller.CreateHeightAttr(float(2 * g.ROLLER[3]))
    roller.CreateDisplayColorAttr([Gf.Vec3f(*ROLLER)])
    roller.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in g.ROLLER_C]))
    roller.AddRotateXOp().Set(90.0)
    UsdPhysics.CollisionAPI.Apply(roller.GetPrim())
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(roller.GetPrim()).Bind(m_roll)
    _bind_physics(roller.GetPrim(), p_abs)

    # the door: its frame is on the hinge axis; shut, it closes the bay's front and the front of its top
    door = UsdGeom.Xform.Define(stage, "/printer/Geometry/body/door")
    door.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(g.HINGE_Z)))
    UsdPhysics.RigidBodyAPI.Apply(door.GetPrim())
    _mass(door.GetPrim(), g.DOOR_MASS, g.door_com(), (0.06, 2 * g.DOOR_HALF_W, g.PANEL_H))
    for name, c, s, p in g.door_boxes():
        _box(stage, f"/printer/Geometry/body/door/{name}", c, s, p, DOOR, m_door)
        _bind_physics(stage.GetPrimAtPath(f"/printer/Geometry/body/door/{name}"), p_abs)

    UsdGeom.Scope.Define(stage, "/printer/Physics")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/printer/Physics/hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/printer/Geometry/body")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/printer/Geometry/body/door")])
    hinge.CreateAxisAttr("Y")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(g.HINGE_Z)))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # the joint frame turned 180 degrees about z on both sides: its +Y axis is the station's -y, so a positive angle
    # swings the door's top toward the robot
    flip = Gf.Quatf(0.0, 0.0, 0.0, 1.0)
    hinge.CreateLocalRot0Attr(flip)
    hinge.CreateLocalRot1Attr(flip)
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(float(math.degrees(g.OPEN)))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(g.DOOR_DAMPING * math.pi / 180))
    drive.CreateMaxForceAttr(100.0)
    stage.GetRootLayer().Export(str(path))


def build_cartridge(path: Path, which: str) -> None:
    from pxr import UsdPhysics

    stage = _stage("cartridge")
    m_body = _material(stage, "/cartridge/Looks/body", CART)
    m_handle = _material(stage, "/cartridge/Looks/handle", HANDLE[which])
    body = stage.GetPrimAtPath("/cartridge")          # the rigid body is the file's root prim
    UsdPhysics.RigidBodyAPI.Apply(body)
    _mass(body, g.MASS[which], (0.0, 0.0, -0.005), g.C_SIZE)
    for name, c, s, p in g.cartridge_boxes():
        handle = name == "handle"
        _box(stage, f"/cartridge/{name}", c, s, p, HANDLE[which] if handle else CART,
             m_handle if handle else m_body)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_printer(OUT / "printer.usda")
    for which in ("old", "new"):
        build_cartridge(OUT / f"cartridge_{which}.usda", which)
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
