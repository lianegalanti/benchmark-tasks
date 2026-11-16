"""Writes the task's USD assets from geometry.py:

- assets/corner.usda: the rear drum-brake corner, one USD articulation fixed to the world by ``root_joint``. Root link
  ``base``: the bench stand and axle stub, the backing plate (with its adjusting slot and dust lip), the wheel
  cylinder, anchor pin, and the adjusting screw and socket. Child links:
  ``drum`` on the revolute joint ``drum`` about the axle (B's y axis), free but for the hub's drag (a damping);
  ``shoe_primary`` and ``shoe_secondary`` on prismatic joints along B's x (outward), driven by the task toward the
  adjuster's setting; each lining touches the drum through three contact spheres (lining friction 0.40);
  ``star`` (the star wheel: hub and ten teeth) on the revolute joint ``star`` about the adjusting screw's axis (B's
  x axis), limited to the screw's travel; the task drives it toward its detent notches.
- assets/spoon.usda: the brake adjusting spoon, one rigid body: a moulded handle and a flat steel blade.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

STEEL = (0.42, 0.43, 0.45)        # backing plate, stand (painted dark grey steel)
IRON = (0.30, 0.29, 0.28)         # cast-iron drum
LINING = (0.55, 0.45, 0.33)       # shoe linings
BRIGHT = (0.75, 0.73, 0.68)       # adjuster parts, studs (bright zinc)
BRASS = (0.80, 0.62, 0.25)        # star wheel (as on many adjusters: easy to see through the slot)
HANDLE = (0.85, 0.12, 0.10)       # red moulded handle
BLADE = (0.70, 0.72, 0.74)

FRICTION = {"steel": (0.40, 0.35), "iron": (0.45, 0.40), "grip": (1.0, 0.9),
            "lining": (g.MU_LINING, g.MU_LINING)}   # the lining's own friction rules its contacts ("min")


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, static: float, dynamic: float, combine_min: bool = False):
    from pxr import Sdf, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    if combine_min:   # PhysX: "min" outranks the default "average", so the pair's friction is this material's
        prim = mat.GetPrim()
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("min")
    return mat


def _bind(prim, look, phys) -> None:
    """Bind the look; with a physics material, the prim also collides (``phys`` None: looks only)."""
    from pxr import UsdPhysics, UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(look)
    if phys is not None:
        UsdPhysics.CollisionAPI.Apply(prim)
        api.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, centre, size, rot, rgb, look, phys) -> None:
    """A colliding box in its link's frame (rotation matrix ``rot``)."""
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
    """A colliding cylinder along ``axis`` ("x"/"y"/"z"): authored along z and turned onto its axis by its transform
    (the physics engine's cylinder collider follows the prim's transform)."""
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


def _sphere(stage, path: str, centre, radius, rgb, look, phys) -> None:
    from pxr import Gf, UsdGeom

    sp = UsdGeom.Sphere.Define(stage, path)
    sp.CreateRadiusAttr(float(radius))
    sp.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    sp.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    _bind(sp.GetPrim(), look, phys)


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


def _link(stage, path: str, pos=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    return x.GetPrim()


def build_corner(path: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("corner")
    looks = {k: _material(stage, f"/corner/Looks/{k}", v) for k, v in
             (("steel", STEEL), ("iron", IRON), ("lining", LINING), ("bright", BRIGHT), ("brass", BRASS))}
    phys = {k: _physics_material(stage, f"/corner/Looks/{k}_physics", *v, combine_min=(k == "lining"))
            for k, v in FRICTION.items()}
    UsdGeom.Scope.Define(stage, "/corner/Geometry")

    base = _link(stage, "/corner/Geometry/base")
    UsdPhysics.ArticulationRootAPI.Apply(base)
    # self-collisions on: the shoes' linings must touch the drum (parent-child pairs stay filtered)
    base.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    _mass(base, 12.0, (0.0, -0.02, -0.03), (0.15, 0.25, 0.15))
    for name, c, s, r in g.plate_boxes() + g.stand_boxes():
        _box(stage, f"/corner/Geometry/base/{name}", c, s, r, STEEL, looks["steel"], phys["steel"])
    for name, c, s, r in g.base_boxes():
        _box(stage, f"/corner/Geometry/base/{name}", c, s, r, STEEL, looks["steel"], phys["steel"])
    for name, c, rad, length, ax in g.shoe_cylinders():
        _cyl(stage, f"/corner/Geometry/base/{name}", c, rad, length, ax, BRIGHT, looks["bright"], phys["steel"])
    for name, c, rad, length, ax in g.stand_cylinders():
        _cyl(stage, f"/corner/Geometry/base/{name}", c, rad, length, ax, STEEL, looks["steel"], phys["steel"])

    drum = _link(stage, "/corner/Geometry/base/drum")
    r_avg = (g.DRUM_R_IN + g.DRUM_R_OUT) / 2
    iyy = g.DRUM_MASS * r_avg ** 2
    _mass(drum, g.DRUM_MASS, (0.0, 0.045, 0.0), (iyy / 2 + g.DRUM_MASS * 0.07 ** 2 / 12, iyy,
                                                  iyy / 2 + g.DRUM_MASS * 0.07 ** 2 / 12))
    for name, c, s, r in g.drum_boxes():
        _box(stage, f"/corner/Geometry/base/drum/{name}", c, s, r, IRON, looks["iron"], phys["iron"])
    for name, c, rad, length, ax in g.drum_cylinders():
        stud = name.startswith("stud")
        _cyl(stage, f"/corner/Geometry/base/drum/{name}", c, rad, length, ax, BRIGHT if stud else IRON,
             looks["bright" if stud else "iron"], phys["steel" if stud else "iron"])

    # the shoes: each a link on a prismatic joint along B's x (outward), its lining touching the drum through three
    # contact spheres; lining, table and web are looks only
    for side, _, _, _ in g.SHOES:
        shoe = _link(stage, f"/corner/Geometry/base/{side}")
        _mass(shoe, g.SHOE_MASS, (0.0, 0.037, 0.0), (2e-3, 2e-3, 2e-3))
        for name, c, s, r in g.shoe_visual_boxes(side):
            lining = name.startswith("lining")
            _box(stage, f"/corner/Geometry/base/{side}/{name}", c, s, r, LINING if lining else STEEL,
                 looks["lining" if lining else "steel"], None)
        for name, c, rad in g.shoe_pads(side):
            _sphere(stage, f"/corner/Geometry/base/{side}/{name}", c, rad, LINING, looks["lining"], phys["lining"])

    star = _link(stage, "/corner/Geometry/base/star", g.SW)
    _mass(star, g.SW_MASS, (0.0, 0.0, 0.0), (4e-6, 4e-6, 4e-6))
    for name, c, s, r in g.star_boxes():
        _box(stage, f"/corner/Geometry/base/star/{name}", c, s, r, BRASS, looks["brass"], phys["steel"])
    for name, c, rad, length, ax in g.star_cylinders():
        _cyl(stage, f"/corner/Geometry/base/star/{name}", c, rad, length, ax, BRASS, looks["brass"], phys["steel"])

    UsdGeom.Scope.Define(stage, "/corner/Physics")
    j = UsdPhysics.RevoluteJoint.Define(stage, "/corner/Physics/drum")
    j.CreateBody0Rel().SetTargets([Sdf.Path("/corner/Geometry/base")])
    j.CreateBody1Rel().SetTargets([Sdf.Path("/corner/Geometry/base/drum")])
    j.CreateAxisAttr("Y")
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(g.BEAR_DAMPING * math.pi / 180))
    drive.CreateMaxForceAttr(1.0e6)

    for side, _, sign, _ in g.SHOES:
        j = UsdPhysics.PrismaticJoint.Define(stage, f"/corner/Physics/{side}")
        j.CreateBody0Rel().SetTargets([Sdf.Path("/corner/Geometry/base")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/corner/Geometry/base/{side}")])
        j.CreateAxisAttr("X")
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        lo, hi = -0.001, 0.003            # outward travel limits (m): the screw's full travel is 2 mm
        j.CreateLowerLimitAttr(float(min(sign * lo, sign * hi)))
        j.CreateUpperLimitAttr(float(max(sign * lo, sign * hi)))
        drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(g.K_SHOE))
        drive.CreateDampingAttr(float(g.D_SHOE))
        drive.CreateMaxForceAttr(float(g.SHOE_FORCE_MAX))
        drive.CreateTargetPositionAttr(0.0)

    j = UsdPhysics.RevoluteJoint.Define(stage, "/corner/Physics/star")
    j.CreateBody0Rel().SetTargets([Sdf.Path("/corner/Geometry/base")])
    j.CreateBody1Rel().SetTargets([Sdf.Path("/corner/Geometry/base/star")])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in g.SW]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # tightening is a negative angle about +x (the slot side's teeth move up); the screw's travel limits it
    j.CreateLowerLimitAttr(float(-math.degrees((g.N_MAX + 0.5) * g.PITCH)))
    j.CreateUpperLimitAttr(float(math.degrees(0.5 * g.PITCH)))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(g.DETENT_K * math.pi / 180))
    drive.CreateDampingAttr(float(g.DETENT_D * math.pi / 180))
    drive.CreateMaxForceAttr(float(g.CLICK_TORQUE))
    drive.CreateTargetPositionAttr(0.0)

    fixed = UsdPhysics.FixedJoint.Define(stage, "/corner/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/corner")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/corner/Geometry/base")])
    stage.GetRootLayer().Export(str(path))


def build_spoon(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("spoon")
    looks = {"handle": _material(stage, "/spoon/Looks/handle", HANDLE),
             "blade": _material(stage, "/spoon/Looks/blade", BLADE)}
    phys = {"grip": _physics_material(stage, "/spoon/Looks/grip_physics", *FRICTION["grip"]),
            "steel": _physics_material(stage, "/spoon/Looks/steel_physics", *FRICTION["steel"])}
    body = stage.GetPrimAtPath("/spoon")          # the rigid body is the file's root prim
    UsdPhysics.RigidBodyAPI.Apply(body)
    m_handle, m_blade = g.SPOON_MASS - 0.02, 0.02
    com_x = m_blade * (g.HANDLE[0] / 2 + g.BLADE[0] / 2) / g.SPOON_MASS
    L = g.SPOON_LEN
    _mass(body, g.SPOON_MASS, (com_x, 0.0, -0.001),
          (g.SPOON_MASS * (g.HANDLE[1] ** 2 + g.HANDLE[2] ** 2) / 12, g.SPOON_MASS * L ** 2 / 14,
           g.SPOON_MASS * L ** 2 / 14))
    (hc, _, hh), (bc, _, bh) = g.spoon_boxes_s()
    _box(stage, "/spoon/handle", hc, 2 * hh, np.eye(3), HANDLE, looks["handle"], phys["grip"])
    _box(stage, "/spoon/blade", bc, 2 * bh, np.eye(3), BLADE, looks["blade"], phys["steel"])
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_corner(OUT / "corner.usda")
    build_spoon(OUT / "spoon.usda")
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
