"""Writes the task's USD asset from geometry.py:

- assets/screen.usda: the insect screen, a floating-base articulation in the screen frame S (see geometry.py). Root
  link ``frame``: the four aluminium rails and the mesh panel (one rigid body). Child links ``plunger_left`` and
  ``plunger_right``: the two spring plunger pins on the top rail, each on a prismatic joint along S's z axis
  (0 = at rest against its stop, negative = pushed in, down to flush). A drive with stiffness on each joint is the
  plunger's spring. Links joined to the root don't collide with it (as in any articulation).

The window frame section is built from fixed boxes in task.py (it doesn't move).

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda file is committed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

FRAME = (0.80, 0.80, 0.78)       # natural anodised aluminium
MESH = (0.12, 0.12, 0.13)        # charcoal fibreglass mesh
PIN = (0.92, 0.92, 0.88)         # POM pins

FRICTION = {"alu": (0.6, 0.5), "mesh": (0.6, 0.5), "pom": (0.2, 0.15)}


def _material(stage, path: str, rgb, opacity: float = 1.0):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    if opacity < 1.0:
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
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


def _box(stage, path: str, lo, hi, rgb, look, physics=None, collide: bool = True) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in (lo + hi) / 2]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in hi - lo]))
    binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
    binding.Bind(look)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        binding.Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")


def frame_inertia():
    """Mass, centre of mass and diagonal inertia of the frame link (rails and mesh, by volume share)."""
    boxes = [(np.asarray(lo, float), np.asarray(hi, float)) for lo, hi in g.RAILS.values()]
    vols = [np.prod(hi - lo) for lo, hi in boxes]
    # the mesh weighs about 60 g (fibreglass mesh, 0.21 m2 at ~120 g/m2 plus spline); the rest is the rails
    mesh_m = 0.06
    rail_m = g.FRAME_MASS - mesh_m
    parts = [(rail_m * v / sum(vols), lo, hi) for v, (lo, hi) in zip(vols, boxes)]
    parts.append((mesh_m, np.asarray(g.MESH[0], float), np.asarray(g.MESH[1], float)))
    m = sum(p[0] for p in parts)
    com = sum(p[0] * (p[1] + p[2]) / 2 for p in parts) / m
    inertia = np.zeros(3)
    for mi, lo, hi in parts:
        a, b, c = hi - lo
        d = (lo + hi) / 2 - com
        inertia += mi / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
        inertia += mi * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return m, com, inertia


def _mass(prim, mass, com, inertia) -> None:
    from pxr import Gf, UsdPhysics

    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def build_screen(path: Path) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/screen")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Scope.Define(stage, "/screen/Looks")
    look = {"alu": _material(stage, "/screen/Looks/alu", FRAME),
            "mesh": _material(stage, "/screen/Looks/mesh", MESH, opacity=0.55),
            "pom": _material(stage, "/screen/Looks/pom", PIN)}
    phys = {k: _physics_material(stage, f"/screen/Looks/{k}_physics", *v) for k, v in FRICTION.items()}
    UsdGeom.Scope.Define(stage, "/screen/Geometry")

    frame = UsdGeom.Xform.Define(stage, "/screen/Geometry/frame").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(frame)
    UsdPhysics.ArticulationRootAPI.Apply(frame)
    frame.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    m, com, inertia = frame_inertia()
    _mass(frame, m, com, inertia)
    for name, (lo, hi) in g.RAILS.items():
        _box(stage, f"/screen/Geometry/frame/{name}", lo, hi, FRAME, look["alu"], phys["alu"])
    _box(stage, "/screen/Geometry/frame/mesh", *g.MESH, MESH, look["mesh"], phys["mesh"])
    # the plungers' housings on the top rail's face (visual only: they are part of the rail)
    for sign, side in ((1.0, "left"), (-1.0, "right")):
        y = sign * g.PIN_Y
        _box(stage, f"/screen/Geometry/frame/housing_{side}", (-g.FRAME_T / 2 - 0.002, y - 0.012, g.SCREEN_H - 0.019),
             (-g.FRAME_T / 2, y + 0.012, g.SCREEN_H - 0.001), PIN, look["pom"], collide=False)

    UsdGeom.Scope.Define(stage, "/screen/Physics")
    for sign, side in ((1.0, "left"), (-1.0, "right")):
        link = UsdGeom.Xform.Define(stage, f"/screen/Geometry/frame/plunger_{side}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        lo, hi = g.pin_box(sign)
        c = (np.asarray(lo) + np.asarray(hi)) / 2
        s = np.asarray(hi) - np.asarray(lo)
        _mass(link, g.PIN_MASS, c, g.PIN_MASS / 12 * np.array([s[1] ** 2 + s[2] ** 2, s[0] ** 2 + s[2] ** 2,
                                                                s[0] ** 2 + s[1] ** 2]))
        _box(stage, f"/screen/Geometry/frame/plunger_{side}/pin", lo, hi, PIN, look["pom"], phys["pom"])

        j = UsdPhysics.PrismaticJoint.Define(stage, f"/screen/Physics/plunger_{side}")
        j.CreateBody0Rel().SetTargets([Sdf.Path("/screen/Geometry/frame")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/screen/Geometry/frame/plunger_{side}")])
        j.CreateAxisAttr("Z")
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLowerLimitAttr(float(-g.PIN_TRAVEL))
        j.CreateUpperLimitAttr(0.0)
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(g.PIN_RATE))
        drive.CreateDampingAttr(float(g.PIN_DAMPING))
        drive.CreateTargetPositionAttr(float(g.PIN_TARGET))
        drive.CreateMaxForceAttr(50.0)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_screen(OUT / "screen.usda")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.usda"))))
