"""Writes the task's USD assets from geometry.py:

- assets/hand.usda: the resident's right hand, one USD articulation fixed to the world by ``root_joint``. Root link
  ``palm``: the palm, the forearm stub, the thumb and the ring and little fingers (already done). Child links ``index``
  and ``middle``, each on a sprung revolute knuckle joint (``index``, ``middle``; about the finger's y axis, + flexes)
  with segments, nail bed and the attached nail plate; under each, eight free-edge strips ``<finger>_b0`` ..
  ``<finger>_b7`` on prismatic joints along the finger (the task slides a strip back into the plate by what a cut takes
  off it, so the white edge left stands out of the plate exactly that long).
- assets/clipper.usda: the lever nail clipper, one free-floating USD articulation. Root link ``body`` (lower arm, lower
  jaw, rivet spacer, the pin), ``upper`` on the revolute joint ``jaw`` at the rivet (sprung open against its stop; + closes
  the jaws) and ``lever`` on the revolute joint ``lever`` at the pin's head (- presses it down). The lever's heel pushes
  the upper arm down by contact: only those two links collide with each other.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

SKIN = (0.80, 0.62, 0.52)
NAIL = (0.86, 0.66, 0.62)        # nail plate over the pink nail bed
WHITE = (0.95, 0.93, 0.86)       # the free edge
STEEL = (0.72, 0.73, 0.76)       # chrome-plated clipper
DARK = (0.45, 0.46, 0.48)        # the edges and the pin

FRICTION = {"skin": (0.9, 0.8), "nail": (0.4, 0.35), "steel": (0.5, 0.45), "lever": (0.9, 0.8)}


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


def _bind(prim, look, phys, collide: bool = True) -> None:
    from pxr import UsdPhysics, UsdShade

    if collide:
        UsdPhysics.CollisionAPI.Apply(prim)
    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(look)
    if collide:
        api.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path, centre, size, rot, rgb, look, phys, collide=True) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if not np.allclose(rot, np.eye(3)):
        cube.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot)]))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    _bind(cube.GetPrim(), look, phys, collide)


def _capsule(stage, path, a, b, radius, rgb, look, phys, collide=True) -> None:
    """A capsule from point a to point b (in its link's frame): authored along z and turned onto a-b."""
    from pxr import Gf, UsdGeom

    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    z = d / length
    x = np.cross([0.0, 0.0, 1.0], z) if abs(z[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    rot = np.column_stack([x, np.cross(z, x), z])
    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr("Z")
    cap.CreateRadiusAttr(float(radius))
    cap.CreateHeightAttr(length)
    cap.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cap.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in (a + b) / 2]))
    cap.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot)]))
    _bind(cap.GetPrim(), look, phys, collide)


def _cyl_y(stage, path, centre, radius, length, rgb, look, phys, collide=True) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Y")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(length))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    _bind(cyl.GetPrim(), look, phys, collide)


def _cyl_z(stage, path, centre, radius, length, rgb, look, phys, collide=True) -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Z")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(length))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    _bind(cyl.GetPrim(), look, phys, collide)


def _stage(root: str):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, f"/{root}")
    stage.SetDefaultPrim(prim.GetPrim())
    UsdGeom.Scope.Define(stage, f"/{root}/Looks")
    UsdGeom.Scope.Define(stage, f"/{root}/Geometry")
    UsdGeom.Scope.Define(stage, f"/{root}/Physics")
    return stage


def _mass(prim, mass, com, inertia) -> None:
    from pxr import Gf, UsdPhysics

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(max(v, 2e-8)) for v in inertia]))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _link(stage, path: str, pos=(0.0, 0.0, 0.0), rot=None, gravity: bool = True):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    if rot is not None:
        x.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot)]))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    if not gravity:
        x.GetPrim().CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
    return x.GetPrim()


def _revolute(stage, path, body0, body1, pos0, rot0, axis, lower_deg, upper_deg, k, d, target, max_force=1.0e3):
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.RevoluteJoint.Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
    j.CreateLocalRot0Attr(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot0)]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLowerLimitAttr(float(lower_deg))
    j.CreateUpperLimitAttr(float(upper_deg))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(k * math.pi / 180))
    drive.CreateDampingAttr(float(d * math.pi / 180))
    drive.CreateMaxForceAttr(float(max_force))
    drive.CreateTargetPositionAttr(float(math.degrees(target)))


def build_hand(path: Path) -> None:
    from pxr import Gf, Sdf, UsdPhysics

    stage = _stage("hand")
    looks = {k: _material(stage, f"/hand/Looks/{k}", v) for k, v in (("skin", SKIN), ("nail", NAIL), ("white", WHITE))}
    phys = {k: _physics_material(stage, f"/hand/Looks/{k}_physics", *FRICTION[k]) for k in ("skin", "nail")}
    root = "/hand/Geometry/palm"
    palm = _link(stage, root, gravity=False)
    UsdPhysics.ArticulationRootAPI.Apply(palm)
    palm.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(palm, 0.4, (-0.04, 0.0, 0.0), (5e-4, 5e-4, 8e-4))
    h0 = g.HAND0
    c, r, h = g.palm_box()
    _box(stage, f"{root}/palm", c - h0, 2 * h, r, SKIN, looks["skin"], phys["skin"])
    for i, (a, b, rad) in enumerate(g.static_capsules_world()):
        _capsule(stage, f"{root}/part{i}", a - h0, b - h0, rad, SKIN, looks["skin"], phys["skin"])
    for name in g.NAILS:
        f = g.FINGERS[name]
        fp = f"{root}/{name}"
        link = _link(stage, fp, g.finger_origin(name) - h0, g.finger_rot(name), gravity=False)
        lp = f["seg"][0] + f["seg"][1] / 2
        _mass(link, g.FINGER_MASS, (lp, 0.0, 0.0), (2e-6, g.FINGER_MASS * lp ** 2 / 3, g.FINGER_MASS * lp ** 2 / 3))
        for seg, x0, x1, rad in g.finger_capsules(name):
            _capsule(stage, f"{fp}/{seg}", (x0, 0.0, 0.0), (x1, 0.0, 0.0), rad, SKIN, looks["skin"], phys["skin"])
        for part, cen, size in g.finger_boxes_f(name):
            key = "nail" if part == "plate" else "skin"
            _box(stage, f"{fp}/{part}", cen, size, np.eye(3), NAIL if part == "plate" else SKIN, looks[key], phys[key])
        lo, hi = g.FINGER_LIMITS
        _revolute(stage, f"/hand/Physics/{name}", root, fp, g.finger_origin(name) - h0, g.finger_rot(name), "Y",
                  lo, hi, g.FINGER_K, g.FINGER_D, 0.0)
        n0 = g.nail_in_f(name)
        bw = g.bin_width(name)
        for b, (vb, l0) in enumerate(zip(g.bin_v(name), g.free_edge0(name))):
            bp = f"{fp}/{name}_b{b}"
            pos = n0 + np.array([0.0, vb, 0.0])
            bl = _link(stage, bp, pos, gravity=False)
            _mass(bl, g.BIN_MASS, (l0 / 2, 0.0, 0.0), (2e-8, 2e-8, 2e-8))
            _box(stage, f"{bp}/strip", (l0 / 2, 0.0, 0.0), (l0, bw - 0.00005, g.BIN_T), np.eye(3), WHITE,
                 looks["white"], phys["nail"])
            j = UsdPhysics.PrismaticJoint.Define(stage, f"/hand/Physics/{name}_b{b}")
            j.CreateBody0Rel().SetTargets([Sdf.Path(fp)])
            j.CreateBody1Rel().SetTargets([Sdf.Path(bp)])
            j.CreateAxisAttr("X")
            j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos]))
            j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(float(-l0 - 0.0005))
            j.CreateUpperLimitAttr(0.0)
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(g.BIN_K))
            drive.CreateDampingAttr(float(g.BIN_D))
            drive.CreateMaxForceAttr(100.0)
            drive.CreateTargetPositionAttr(0.0)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/hand/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/hand")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(root)])
    stage.GetRootLayer().Export(str(path))


def build_clipper(path: Path) -> None:
    from pxr import Sdf, UsdPhysics

    stage = _stage("clipper")
    looks = {"steel": _material(stage, "/clipper/Looks/steel", STEEL), "dark": _material(stage, "/clipper/Looks/dark", DARK)}
    phys = {k: _physics_material(stage, f"/clipper/Looks/{k}_physics", *FRICTION[k]) for k in ("steel", "lever")}
    eye = np.eye(3)
    B = "/clipper/Geometry/body"
    body = _link(stage, B)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    _mass(body, g.BODY_MASS, (0.03, 0.0, 0.001), (1e-7, 3.5e-6, 3.5e-6))
    # lower arm, its jaw (strips whose fronts follow the curved edge), the rivet spacer, the pin (not a collider:
    # it passes through a hole in the upper arm)
    _box(stage, f"{B}/arm", (g.JAW_X0 / 2, 0.0, g.LOW_T / 2), (g.JAW_X0, g.ARM_W, g.LOW_T), eye, STEEL,
         looks["steel"], phys["lever"])
    ys = np.linspace(-g.JAW_HW, g.JAW_HW, 8)
    for i in range(7):
        yc = (ys[i] + ys[i + 1]) / 2
        xf = float(g.x_edge(yc))
        _box(stage, f"{B}/jaw{i}", ((g.JAW_X0 + xf) / 2, yc, g.Z_LE / 2), (xf - g.JAW_X0, ys[1] - ys[0], g.Z_LE), eye,
             DARK, looks["dark"], phys["steel"])
    for s in (-1, 1):   # the arm's edges past the jaws' 13 mm
        _box(stage, f"{B}/jawside{'lr'[s > 0]}", ((g.JAW_X0 + g.E_X) / 2, s * (g.JAW_HW + 0.00025), g.LOW_T / 2),
             (g.E_X - g.JAW_X0, 0.0005, g.LOW_T), eye, STEEL, looks["steel"], phys["steel"])
    _box(stage, f"{B}/spacer", ((g.X_H + 0.0035) / 2 + 0.0, 0.0, (g.LOW_T + g.Z_H) / 2),
         (0.007, 0.010, g.Z_H - g.LOW_T), eye, STEEL, looks["steel"], phys["steel"])
    _cyl_z(stage, f"{B}/pin", (g.P_X, 0.0, (g.LOW_T + g.Z_P) / 2), 0.0011, g.Z_P - g.LOW_T, DARK, looks["dark"],
           phys["steel"], collide=False)
    _cyl_z(stage, f"{B}/pinhead", (g.P_X, 0.0, g.Z_P + 0.0004), 0.0018, 0.0008, DARK, looks["dark"], phys["steel"],
           collide=False)

    U = f"{B}/upper"
    up = _link(stage, U, (g.X_H, 0.0, g.Z_H))
    _mass(up, g.UPPER_MASS, (0.025, 0.0, 0.0008), (1e-7, 2e-6, 2e-6))
    xj0 = g.JAW_X0 - g.X_H
    _box(stage, f"{U}/arm", ((xj0 - 0.0035) / 2, 0.0, g.UP_T / 2), (xj0 + 0.0035, g.ARM_W, g.UP_T), eye, STEEL,
         looks["steel"], phys["steel"])
    for i in range(7):
        yc = (ys[i] + ys[i + 1]) / 2
        xf = float(g.x_edge(yc)) - g.X_H
        _box(stage, f"{U}/jaw{i}", ((xj0 + xf) / 2, yc, g.UP_T / 2), (xf - xj0, ys[1] - ys[0], g.UP_T), eye, DARK,
             looks["dark"], phys["steel"])

    L = f"{B}/lever"
    lev = _link(stage, L, (g.P_X, 0.0, g.Z_P))
    _mass(lev, g.LEVER_MASS, (-0.02 * math.cos(g.PHI0), 0.0, 0.02 * math.sin(g.PHI0)), (1e-7, 1.2e-6, 1.2e-6))
    r0 = g.lever_rot(g.PHI0)
    _box(stage, f"{L}/plate", r0 @ np.array([-(g.LEVER_L - 0.003) / 2, 0.0, 0.0]), (g.LEVER_L + 0.003, g.LEVER_W, g.LEVER_T),
         r0, STEEL, looks["steel"], phys["lever"])
    _box(stage, f"{L}/nub", r0 @ np.array([g.HEEL[0], 0.0, (g.HEEL[2] - g.LEVER_T / 2) / 2]),
         (0.0026, 0.008, -g.HEEL[2] - g.LEVER_T / 2), r0, STEEL, looks["steel"], phys["steel"])
    _cyl_y(stage, f"{L}/heel", r0 @ g.HEEL, g.HEEL_R, 0.008, STEEL, looks["steel"], phys["steel"])

    # only the lever and the upper arm touch each other (the heel on the arm)
    for a, b in ((B, U), (B, L)):
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(a)).CreateFilteredPairsRel().AddTarget(Sdf.Path(b))
    # the cam: the upper arm's hinge is geared to the lever's (a PhysX mimic joint, both ways: a nail between the
    # jaws stops the lever too); the arms' spring acts at the lever
    _revolute(stage, "/clipper/Physics/jaw", B, U, (g.X_H, 0.0, g.Z_H), eye, "Y", 0.0, math.degrees(g.THETA_SHUT),
              0.0, g.JAW_D, 0.0)
    jp = stage.GetPrimAtPath("/clipper/Physics/jaw")
    jp.AddAppliedSchema("PhysxMimicJointAPI:rotY")
    # jaw + gearing x lever + offset = 0, with the lever's angle negative when pressed
    jp.CreateAttribute("physxMimicJoint:rotY:gearing", Sdf.ValueTypeNames.Float).Set(float(g.CAM_GEAR))
    jp.CreateAttribute("physxMimicJoint:rotY:offset", Sdf.ValueTypeNames.Float).Set(0.0)
    jp.CreateRelationship("physxMimicJoint:rotY:referenceJoint").SetTargets([Sdf.Path("/clipper/Physics/lever")])
    jp.CreateAttribute("physxMimicJoint:rotY:referenceJointAxis", Sdf.ValueTypeNames.Token,
                       variability=Sdf.VariabilityUniform).Set("rotY")
    _revolute(stage, "/clipper/Physics/lever", B, L, (g.P_X, 0.0, g.Z_P), eye, "Y", math.degrees(g.LEVER_MIN), 0.0,
              g.LEVER_K_EQ, g.LEVER_D, g.LEVER_PRE, 5.0)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_hand(OUT / "hand.usda")
    build_clipper(OUT / "clipper.usda")
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
