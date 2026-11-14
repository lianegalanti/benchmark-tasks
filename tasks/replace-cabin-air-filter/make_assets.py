"""Writes the task's USD assets from geometry.py:

- assets/dash.usda: the dashboard section, a fixed-base USD articulation. Root link ``body`` (fascia, cavity, the two
  stop blocks and the filter housing with its slot) is fixed to the world by ``root_joint``. Child links: ``bin``
  (the glovebox) on a revolute joint ``hinge`` about the opening's bottom edge (0 shut, positive swings its top toward
  the robot, stops at 0 and 120 degrees), and on the bin the two flexing upper side-wall panels ``pin_left`` and
  ``pin_right`` on prismatic joints along the bin's y axis (positive pushes the panel in toward the bin's middle, up
  to 14 mm). Links joined to each other don't collide; the panels' stop bumps do collide with the dash's stop blocks,
  which is what holds the bin at its normal open position.
- assets/filter_old.usda, assets/filter_new.usda: the two cabin air filters, one rigid body each (frame, pull tab,
  key rib).

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

DASH = (0.16, 0.16, 0.17)        # dark grey dash plastic
BIN = (0.22, 0.22, 0.23)         # the glovebox, a shade lighter
PAD = (0.30, 0.30, 0.32)         # thumb pads
STOP = (0.55, 0.55, 0.52)        # stop bumps and blocks (POM)
HOUSING = (0.42, 0.40, 0.36)     # HVAC housing (unpainted PP)
FRAME = {"old": (0.36, 0.33, 0.28), "new": (0.93, 0.93, 0.90)}   # used: grey-brown with dust; new: white

FRICTION = {"abs": (0.4, 0.35), "pom": (0.15, 0.12), "pp": (0.35, 0.3)}


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, static: float, dynamic: float, combine: str | None = None):
    from pxr import Sdf, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    if combine:
        mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
        mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def _quat(rot):
    from pxr import Gf

    m = np.asarray(rot, float)
    w = math.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    if w > 1e-6:
        x, y, z = (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)
    else:   # 180-degree turns (not used here, kept for safety)
        i = int(np.argmax(np.diag(m)))
        v = np.zeros(3)
        v[i] = math.sqrt(max(0.0, 1 + 2 * m[i, i] - np.trace(m))) / 2
        j, k = (i + 1) % 3, (i + 2) % 3
        v[j] = (m[j, i] + m[i, j]) / (4 * v[i])
        v[k] = (m[k, i] + m[i, k]) / (4 * v[i])
        x, y, z = v
        w = (m[k, j] - m[j, k]) / (4 * v[i])
    return Gf.Quatf(float(w), float(x), float(y), float(z))


def _box(stage, path: str, centre, size, rot, rgb, look, physics, collide: bool = True) -> None:
    """A box in its parent's frame."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if not np.allclose(rot, np.eye(3)):
        cube.AddOrientOp().Set(_quat(rot))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
    binding.Bind(look)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        binding.Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")


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


def _link(stage, path: str, translate=(0.0, 0.0, 0.0)):
    from pxr import Gf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    if any(translate):
        x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in translate]))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    return x.GetPrim()


def build_dash(path: Path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage = _stage("dash")
    look = {k: _material(stage, f"/dash/Looks/{k}", v) for k, v in
            (("dash", DASH), ("bin", BIN), ("pad", PAD), ("stop", STOP), ("housing", HOUSING))}
    # POM stops: "min", so their low friction rules where the bumps slide on the fascia
    phys = {k: _physics_material(stage, f"/dash/Looks/{k}_physics", *v, "min" if k == "pom" else None)
            for k, v in FRICTION.items()}
    from pxr import UsdGeom

    UsdGeom.Scope.Define(stage, "/dash/Geometry")
    body = _link(stage, "/dash/Geometry/body")
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    _mass(body, 6.0, (0.1, 0.0, 0.1), (0.4, 0.5, 0.4))
    for name, c, s, r in g.dash_boxes():
        if name.startswith("stop"):
            lk, ph = "stop", "pom"
        elif name.startswith(("face", "tunnel")):
            lk, ph = "housing", "pp"
        else:
            lk, ph = "dash", "abs"
        _box(stage, f"/dash/Geometry/body/{name}", c, s, r, {"dash": DASH, "stop": STOP, "housing": HOUSING}[lk],
             look[lk], phys[ph])

    # the glovebox bin: its frame on the hinge axis
    b = _link(stage, "/dash/Geometry/body/bin", (0.0, 0.0, g.HINGE_Z))
    _mass(b, g.BIN_MASS, g.BIN_COM, (0.15, 2 * g.HALF_W, 0.2))
    for name, c, s, r in g.bin_boxes():
        _box(stage, f"/dash/Geometry/body/bin/{name}", c, s, r, BIN, look["bin"], phys["abs"])
    for sign, side in ((1.0, "left"), (-1.0, "right")):
        p = _link(stage, f"/dash/Geometry/body/bin/pin_{side}")
        _mass(p, g.PIN_MASS, (0.05, sign * (g.HALF_W - 0.003), 0.11), (0.12, 0.006, 0.11))
        for name, c, s, r in g.pin_boxes(sign):
            stop = name.startswith(("bump", "ramp"))
            pad = name.startswith("pad")
            _box(stage, f"/dash/Geometry/body/bin/pin_{side}/{name}", c, s, r,
                 STOP if stop else PAD if pad else BIN, look["stop" if stop else "pad" if pad else "bin"],
                 phys["pom" if stop else "abs"])

    UsdGeom.Scope.Define(stage, "/dash/Physics")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/dash/Physics/hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/dash/Geometry/body")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/dash/Geometry/body/bin")])
    hinge.CreateAxisAttr("Y")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(g.HINGE_Z)))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # joint frames turned 180 degrees about z on both sides: +Y is the station's -y, so a positive angle swings the
    # bin's top toward the robot
    flip = Gf.Quatf(0.0, 0.0, 0.0, 1.0)
    hinge.CreateLocalRot0Attr(flip)
    hinge.CreateLocalRot1Attr(flip)
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(float(math.degrees(g.LOWERED)))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(g.HINGE_DAMPING * math.pi / 180))
    drive.CreateMaxForceAttr(100.0)

    for sign, side in ((1.0, "left"), (-1.0, "right")):
        j = UsdPhysics.PrismaticJoint.Define(stage, f"/dash/Physics/pin_{side}")
        j.CreateBody0Rel().SetTargets([Sdf.Path("/dash/Geometry/body/bin")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/dash/Geometry/body/bin/pin_{side}")])
        j.CreateAxisAttr("Y")
        j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        # positive pushes the panel inward: -y for the left panel (joint frame turned about z), +y for the right
        rot = Gf.Quatf(0.0, 0.0, 0.0, 1.0) if sign > 0 else Gf.Quatf(1.0, 0.0, 0.0, 0.0)
        j.CreateLocalRot0Attr(rot)
        j.CreateLocalRot1Attr(rot)
        j.CreateLowerLimitAttr(0.0)
        j.CreateUpperLimitAttr(float(g.PIN_TRAVEL))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(float(g.PIN_RATE))
        d.CreateDampingAttr(float(g.PIN_DAMPING))
        d.CreateMaxForceAttr(200.0)
        d.CreateTargetPositionAttr(float(g.PIN_TARGET))

    fixed = UsdPhysics.FixedJoint.Define(stage, "/dash/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/dash")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/dash/Geometry/body")])
    stage.GetRootLayer().Export(str(path))


def build_filter(path: Path, which: str) -> None:
    from pxr import UsdPhysics

    stage = _stage("filter")
    look = _material(stage, "/filter/Looks/frame", FRAME[which])
    media = _material(stage, "/filter/Looks/media", tuple(0.85 * v for v in FRAME[which]))
    phys = _physics_material(stage, "/filter/Looks/pp_physics", *FRICTION["pp"])
    body = stage.GetPrimAtPath("/filter")
    UsdPhysics.RigidBodyAPI.Apply(body)
    _mass(body, g.F_MASS, (-0.005, 0.0, 0.0), g.F_SIZE)
    for name, c, s, r in g.filter_boxes():
        _box(stage, f"/filter/{name}", c, s, r, FRAME[which], look, phys)
    # the pleated media seen through the frame (visual only): darker strips on both faces
    for k in range(9):
        y = -0.088 + k * 0.022
        for zf in (-1, 1):
            _box(stage, f"/filter/pleat_{k}_{'top' if zf > 0 else 'bottom'}", (0.0, y, zf * (g.F_SIZE[2] / 2 + 0.0003)),
                 (g.F_SIZE[0] - 0.02, 0.012, 0.0006), np.eye(3), FRAME[which], media, phys, collide=False)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_dash(OUT / "dash.usda")
    for which in ("old", "new"):
        build_filter(OUT / f"filter_{which}.usda", which)
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
