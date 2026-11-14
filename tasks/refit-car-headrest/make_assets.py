"""Writes the task's USD assets from geometry.py:

- assets/headrest.usda: the headrest, one rigid body in the headrest frame H (see geometry.py): the cushion (a 105 mm
  thick middle and two 64 mm thick ends) and two 14 mm steel posts. Each post collides as boxes that carry its
  notches (flat upper faces, 45-degree lower faces, a chamfered tip); it is drawn as a round tube with dark bands at
  the notches (the collision boxes are guide geometry, not rendered).
- assets/seat.usda: the cropped seat back section, a fixed-base articulation in the seat frame F. Root link ``seat``
  (the upholstered block with the two bores through it, the bores' closed ends, the two collars with their lead-ins,
  the button's housing) is fixed to the world by ``root_joint``. Child link ``lock`` (the latch: the pawl tooth, the
  follower with its 45-degree face, and the two bars joining them past the post) slides on a prismatic joint along +y
  (0 = at rest, positive = drawn back); a drive with stiffness on it is its spring. Links joined to the root don't
  collide with it (as in any articulation).
- assets/button.usda: the release button, a separate fixed-base articulation in F (a mount without colliders and the
  button: cap and stem, on a prismatic joint pointing down, 0 = at rest, positive = pressed, with its return spring).
  As a separate articulation it collides with the seat's latch: its stem pushes the latch's 45-degree face.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

FABRIC = (0.16, 0.16, 0.17)      # headrest cover (charcoal fabric)
STEEL = (0.78, 0.79, 0.80)       # chromed posts
NOTCH = (0.30, 0.30, 0.32)       # the notches, drawn as dark bands
TRIM = (0.24, 0.23, 0.22)        # seat upholstery
BEZEL = (0.06, 0.06, 0.07)       # collars (black plastic)
BUTTON = (0.42, 0.42, 0.44)      # the lock button (grey plastic)

FRICTION = {"fabric": (0.7, 0.6), "steel": (0.25, 0.2), "trim": (0.6, 0.5), "plastic": (0.3, 0.25),
            "pom": (0.2, 0.15)}


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


def _quat(r: np.ndarray):
    """Rotation matrix -> Gf.Quatf (w, x, y, z)."""
    from pxr import Gf
    from yam_arena.ik import matrix_to_quat_xyzw

    x, y, z, w = (float(v) for v in matrix_to_quat_xyzw(np.asarray(r, float)))
    return Gf.Quatf(w, x, y, z)


def _box(stage, path: str, box, rgb, look, physics=None, collide: bool = True, guide: bool = False) -> None:
    """A (centre, rotation, half sizes) box in its parent's frame."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    c, r, h = box
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
    if not np.allclose(r, np.eye(3)):
        cube.AddOrientOp().Set(_quat(r))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(2 * v) for v in h]))
    if guide:
        cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
    binding.Bind(look)
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        binding.Bind(physics, UsdShade.Tokens.weakerThanDescendants, "physics")


def _cylinder(stage, path: str, centre, radius: float, height: float, rgb, look) -> None:
    """A visual-only cylinder along z."""
    from pxr import Gf, UsdGeom, UsdShade

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateAxisAttr("Z")
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(look)


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

    api = UsdPhysics.MassAPI.Apply(prim)
    api.CreateMassAttr(float(mass))
    api.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    api.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def headrest_inertia():
    """Diagonal inertia about the centre of mass: the cushion as a solid block, each post as a rod."""
    com = g.COM_H
    a, b, c = 2 * g.CUSH_T, 2 * g.CUSH_W, g.CUSH_H
    d = np.array([0.0, 0.0, g.POST_LEN + g.CUSH_H / 2]) - com
    m = g.CUSH_MASS
    inertia = m / 12 * np.array([b * b + c * c, a * a + c * c, a * a + b * b])
    inertia += m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    length = g.POST_LEN + g.POST_HIDDEN
    for side in (1.0, -1.0):
        d = np.array([0.0, side * g.POST_Y, length / 2]) - com
        mp = g.POST_MASS
        inertia += mp / 12 * np.array([length ** 2, length ** 2, 0.0])
        inertia += mp * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    return inertia


def build_headrest(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("headrest")
    look = {"fabric": _material(stage, "/headrest/Looks/fabric", FABRIC),
            "steel": _material(stage, "/headrest/Looks/steel", STEEL),
            "notch": _material(stage, "/headrest/Looks/notch", NOTCH)}
    phys = {k: _physics_material(stage, f"/headrest/Looks/{k}_physics", *FRICTION[k]) for k in ("fabric", "steel")}
    body = stage.GetPrimAtPath("/headrest")
    UsdPhysics.RigidBodyAPI.Apply(body)
    _mass(body, g.MASS, g.COM_H, headrest_inertia())
    for name, box in g.HEADREST_H.items():
        if name.startswith("post"):
            _box(stage, f"/headrest/{name}", box, STEEL, look["steel"], phys["steel"], guide=True)
        else:
            _box(stage, f"/headrest/{name}", box, FABRIC, look["fabric"], phys["fabric"])
    length = g.POST_LEN + g.POST_HIDDEN
    for side, sign in (("left", 1.0), ("right", -1.0)):
        y = sign * g.POST_Y
        _cylinder(stage, f"/headrest/tube_{side}", (0.0, y, length / 2 + 0.0015), g.POST_R, length - 0.003, STEEL,
                  look["steel"])
        _cylinder(stage, f"/headrest/tube_{side}_end", (0.0, y, 0.0015), g.POST_R - 0.0015, 0.003, STEEL,
                  look["steel"])
        for k, n in enumerate(g.NOTCHES):
            # the notch, seen from the inboard side: a dark band across the post's inboard face
            u0 = g.POST_R - 0.0005
            ya, yb = sign * (g.POST_Y - g.POST_R - 0.0002), sign * (g.POST_Y - u0)
            _box(stage, f"/headrest/notch_{side}_{k}",
                 g.lohi((-0.006, min(ya, yb), n - g.NOTCH_H / 2), (0.006, max(ya, yb), n + g.NOTCH_H / 2)),
                 NOTCH, look["notch"], collide=False)
    stage.GetRootLayer().Export(str(path))


def build_seat(path: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("seat")
    look = {k: _material(stage, f"/seat/Looks/{k}", rgb)
            for k, rgb in (("trim", TRIM), ("bezel", BEZEL), ("button", BUTTON))}
    phys = {k: _physics_material(stage, f"/seat/Looks/{k}_physics", *FRICTION[k]) for k in ("trim", "plastic", "pom")}
    UsdGeom.Scope.Define(stage, "/seat/Geometry")
    body = UsdGeom.Xform.Define(stage, "/seat/Geometry/seat").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(body, 3.0, (0.0, 0.0, -g.C / 2), (0.01, 0.01, 0.01))
    for name, box in g.SEAT_F.items():
        if name.startswith("block"):
            _box(stage, f"/seat/Geometry/seat/{name}", box, TRIM, look["trim"], phys["trim"])
        else:
            _box(stage, f"/seat/Geometry/seat/{name}", box, BEZEL, look["bezel"], phys["plastic"])

    lock = UsdGeom.Xform.Define(stage, "/seat/Geometry/seat/lock").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(lock)
    _mass(lock, g.LOCK_MASS, (0.0, -g.POST_Y, g.PAWL_Z), (2e-6, 2e-6, 2e-6))
    for name, box in g.lock_boxes_f(0.0, visual=True).items():
        collide = not name.startswith("bar")
        _box(stage, f"/seat/Geometry/seat/lock/{name}", box, BEZEL, look["bezel"], phys["pom"], collide=collide)

    UsdGeom.Scope.Define(stage, "/seat/Physics")
    _prismatic(stage, "seat", "seat", "lock", "Y", Gf.Quatf(1.0, 0.0, 0.0, 0.0), g.LOCK_TRAVEL, g.LOCK_RATE,
               g.LOCK_DAMPING, g.LOCK_TARGET)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/seat/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/seat")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/seat/Geometry/seat")])
    stage.GetRootLayer().Export(str(path))


def _prismatic(stage, root, base, name, axis, rot, upper, rate, damping, target):
    """A prismatic joint from link ``base`` to its child link ``name``, with a spring drive (its stiffness) toward
    ``target``; limits 0 .. ``upper``."""
    from pxr import Gf, Sdf, UsdPhysics

    j = UsdPhysics.PrismaticJoint.Define(stage, f"/{root}/Physics/{name}")
    j.CreateBody0Rel().SetTargets([Sdf.Path(f"/{root}/Geometry/{base}")])
    j.CreateBody1Rel().SetTargets([Sdf.Path(f"/{root}/Geometry/{base}/{name}")])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    j.CreateLocalRot0Attr(rot)
    j.CreateLocalRot1Attr(rot)
    j.CreateLowerLimitAttr(0.0)
    j.CreateUpperLimitAttr(float(upper))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(rate))
    drive.CreateDampingAttr(float(damping))
    drive.CreateTargetPositionAttr(float(target))
    drive.CreateMaxForceAttr(100.0)


def build_button(path: Path) -> None:
    """The release button as its own fixed-base articulation in F (a mount without colliders, fixed to the world, and
    the button on a prismatic joint pointing down: its joint frame is turned 180 degrees about x, so positive is
    pressed). Being a separate articulation, it collides with the seat's latch, whose 45-degree face its stem drives."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("button")
    look = _material(stage, "/button/Looks/button", BUTTON)
    phys = _physics_material(stage, "/button/Looks/pom_physics", *FRICTION["pom"])
    UsdGeom.Scope.Define(stage, "/button/Geometry")
    mount = UsdGeom.Xform.Define(stage, "/button/Geometry/mount").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(mount)
    UsdPhysics.ArticulationRootAPI.Apply(mount)
    mount.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(mount, 0.05, (0.0, -g.POST_Y + sum(g.CAP) / 2, -0.010), (1e-5, 1e-5, 1e-5))
    button = UsdGeom.Xform.Define(stage, "/button/Geometry/mount/button").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(button)
    _mass(button, g.BUTTON_MASS, (0.0, -g.POST_Y + sum(g.CAP) / 2, 0.0), (1e-6, 1e-6, 1e-6))
    for name, box in g.button_boxes_f(0.0).items():
        _box(stage, f"/button/Geometry/mount/button/{name}", box, BUTTON, look, phys)
    UsdGeom.Scope.Define(stage, "/button/Physics")
    _prismatic(stage, "button", "mount", "button", "Z", Gf.Quatf(0.0, 1.0, 0.0, 0.0), g.BUTTON_TRAVEL,
               g.BUTTON_RATE, g.BUTTON_DAMPING, g.BUTTON_TARGET)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/button/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/button")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/button/Geometry/mount")])
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_headrest(OUT / "headrest.usda")
    build_seat(OUT / "seat.usda")
    build_button(OUT / "button.usda")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.usda"))))
