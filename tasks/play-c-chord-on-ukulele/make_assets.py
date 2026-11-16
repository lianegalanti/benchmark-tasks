"""Writes the task's USD assets from geometry.py:

- assets/ukulele.usda: the soprano ukulele in its cradle, one kinematic (fixed) rigid body in the U frame: the
  body (two bouts), neck, fingerboard, fifteen nickel-silver fret wires (capsules across the fingerboard), nut,
  bridge and saddle, headstock, the cradle and the neck rest; visual only: soundhole, rosette, position dots,
  tuners and the strings' tails past the nut and the saddle.
- assets/string_g.usda ... string_a.usda: each string's speaking length (nut to saddle) as one articulation fixed to
  the world. Root link ``anchor`` (at the U origin, no collider); per node i: link ``lat_ii`` on the prismatic joint
  ``lat_ii`` (across the neck, x) from the previous node (or the anchor), and link ``node_ii`` (the string's capsule
  for that stretch) on the prismatic joint ``vert_ii`` (z) from ``lat_ii``. Each joint has a linear drive (stiffness
  T / h, damping DAMP_TAU times that, target 0). The last span's pull toward the saddle (T / h_last on the last node's
  displacement) is applied by the task as a force on the last node after every physics step (task.py); the last node
  is heavy so that this explicit spring is well inside the stable range. Gravity is off on the string's links (a real
  string's sag is under 0.01 mm).

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

MAHOGANY = (0.50, 0.28, 0.14)     # body, neck and headstock (all-mahogany soprano)
ROSEWOOD = (0.20, 0.12, 0.08)     # fingerboard and bridge
NICKEL = (0.80, 0.80, 0.76)       # fret wire
BONE = (0.93, 0.90, 0.80)         # nut and saddle
DARK = (0.04, 0.03, 0.03)         # soundhole
PEARL = (0.92, 0.90, 0.86)        # position dots, rosette
FOAM = (0.12, 0.12, 0.13)         # cradle pads
CHROME = (0.70, 0.70, 0.72)       # tuners
STRING = (0.95, 0.94, 0.88)       # fluorocarbon strings

# static, dynamic friction, combine mode
FRICTION = {"wood": (0.45, 0.40, "average"), "fret": (0.30, 0.25, "average"), "foam": (0.9, 0.8, "average"),
            "string": (0.22, 0.18, "min")}


def _material(stage, path: str, rgb, roughness: float = 0.5):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, static: float, dynamic: float, combine: str):
    from pxr import Sdf, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    if combine != "average":
        prim = mat.GetPrim()
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def _bind(prim, look, phys=None) -> None:
    from pxr import UsdPhysics, UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(look)
    if phys is not None:
        UsdPhysics.CollisionAPI.Apply(prim)
        api.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def _xform_ops(prim_geom, centre, rot=None, scale=None) -> None:
    from pxr import Gf

    prim_geom.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if rot is not None and not np.allclose(rot, np.eye(3)):
        prim_geom.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot)]))
    if scale is not None:
        prim_geom.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))


def _box(stage, path, centre, size, rgb, look, phys=None) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _xform_ops(cube, centre, scale=size)
    _bind(cube.GetPrim(), look, phys)


def _cyl(stage, path, centre, radius, height, rgb, look, phys=None, axis="Z") -> None:
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis)
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(height))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _xform_ops(cyl, centre)
    _bind(cyl.GetPrim(), look, phys)


def _capsule(stage, path, centre, half_len, radius, axis, rgb, look, phys=None, rot=None) -> None:
    from pxr import Gf, UsdGeom

    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr(axis)
    cap.CreateRadiusAttr(float(radius))
    cap.CreateHeightAttr(float(2 * half_len))
    cap.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _xform_ops(cap, centre, rot=rot)
    _bind(cap.GetPrim(), look, phys)


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


def _mass(prim, mass, inertia=(1e-8, 1e-8, 1e-8)) -> None:
    from pxr import Gf, UsdPhysics

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _rot_y_to(d) -> np.ndarray:
    """A rotation taking the y axis onto unit vector ``d`` (for capsules authored along Y)."""
    y = np.array([0.0, 1.0, 0.0])
    d = np.asarray(d, float) / np.linalg.norm(d)
    v = np.cross(y, d)
    c = float(y @ d)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else g.rot_x(np.pi)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


# -- the ukulele ---------------------------------------------------------------------------------------------------
def build_ukulele(path: Path) -> None:
    from pxr import Sdf, UsdPhysics

    stage = _stage("ukulele")
    looks = {k: _material(stage, f"/ukulele/Looks/{k}", v, r) for k, v, r in
             (("mahogany", MAHOGANY, 0.45), ("rosewood", ROSEWOOD, 0.6), ("nickel", NICKEL, 0.25),
              ("bone", BONE, 0.5), ("dark", DARK, 0.9), ("pearl", PEARL, 0.3), ("foam", FOAM, 0.95),
              ("chrome", CHROME, 0.2), ("string", STRING, 0.35))}
    phys = {k: _physics_material(stage, f"/ukulele/Looks/{k}_physics", *v) for k, v in FRICTION.items()}
    body = stage.GetPrimAtPath("/ukulele")
    UsdPhysics.RigidBodyAPI.Apply(body)
    body.CreateAttribute("physics:kinematicEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    _mass(body, 0.35, (0.004, 0.0005, 0.004))      # a soprano weighs about 350 g (it is clamped: kinematic)

    for name, c, size in g.neck_boxes():
        if name.startswith("fingerboard") or name == "bridge":
            key, rgb = "rosewood", ROSEWOOD
        elif name in ("nut", "saddle"):
            key, rgb = "bone", BONE
        else:
            key, rgb = "mahogany", MAHOGANY
        _box(stage, f"/ukulele/{name}", c, size, rgb, looks[key], phys["wood"])
    for name, c, rad, h in g.body_cylinders():
        _cyl(stage, f"/ukulele/{name}", c, rad, h, MAHOGANY, looks["mahogany"], phys["wood"])
    for name, c, half, rad in g.fret_capsules():
        _capsule(stage, f"/ukulele/{name}", c, half, rad, "X", NICKEL, looks["nickel"], phys["fret"])
    for name, c, size in g.cradle_parts():
        _box(stage, f"/ukulele/{name}", c, size, FOAM, looks["foam"], phys["foam"])

    # visual only
    hs, hr = g.SOUNDHOLE
    _cyl(stage, "/ukulele/rosette", (0.0, -hs, g.TOP_Z + 0.0002), hr + 0.006, 0.0004, PEARL, looks["pearl"])
    _cyl(stage, "/ukulele/soundhole", (0.0, -hs, g.TOP_Z + 0.0004), hr, 0.0004, DARK, looks["dark"])
    for n in (5, 7, 10, 12):
        s = 0.5 * (g.fret_s(n - 1) + g.fret_s(n))
        xs = (-0.006, 0.006) if n == 12 else (0.0,)
        for k, x in enumerate(xs):
            _cyl(stage, f"/ukulele/dot{n}_{k}", (x, -s, 0.00005), 0.003, 0.0002, PEARL, looks["pearl"])
    hs0, hs1, hw, ht = g.HEAD
    posts = []
    for k, (name, d, _t, x0, _x1, _o, _f) in enumerate(g.STRINGS):
        side = -1.0 if k < 2 else 1.0
        sy = -0.035 if k in (0, 3) else -0.075
        p = np.array([side * 0.016, -sy, -0.002])
        posts.append(p)
        _cyl(stage, f"/ukulele/post_{name}", p + [0, 0, 0.006], 0.003, 0.012, CHROME, looks["chrome"])
        # the string's tail from the nut to its tuner post, and from the saddle to the bridge's tie block
        a = np.array([x0, 0.0, float(g.string_z(name, 0.0))])
        b = p + [0.0, 0.0, 0.008]
        _capsule(stage, f"/ukulele/tail_nut_{name}", (a + b) / 2, np.linalg.norm(b - a) / 2, d / 2, "Y", STRING,
                 looks["string"], rot=_rot_y_to(b - a))
        a = np.array([float(g.string_x(name, g.SCALE)), -g.SCALE, float(g.string_z(name, g.SCALE))])
        b = np.array([a[0], -(g.BRIDGE[1] - 0.002), g.TOP_Z + g.BRIDGE[3] + d / 2])
        _capsule(stage, f"/ukulele/tail_saddle_{name}", (a + b) / 2, np.linalg.norm(b - a) / 2, d / 2, "Y",
                 STRING, looks["string"], rot=_rot_y_to(b - a))
    stage.GetRootLayer().Export(str(path))


# -- a string ------------------------------------------------------------------------------------------------------
def _no_gravity(prim) -> None:
    from pxr import Sdf

    prim.AddAppliedSchema("PhysxRigidBodyAPI")
    prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)


def _link(stage, path: str, pos):
    from pxr import Gf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    _no_gravity(x.GetPrim())
    return x.GetPrim()


def build_string(name: str, path: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root = f"string_{name.lower()}"
    stage = _stage(root)
    look = _material(stage, f"/{root}/Looks/string", STRING, 0.35)
    phys = _physics_material(stage, f"/{root}/Looks/string_physics", *FRICTION["string"])
    UsdGeom.Scope.Define(stage, f"/{root}/Geometry")
    UsdGeom.Scope.Define(stage, f"/{root}/Physics")
    d = g.string_spec(name)[1]
    rest = g.node_rest(name)
    k = g.joint_stiffness(name)

    nut = np.zeros(3)      # the anchor sits at the file's origin (the fixed root joint's frame), under the nut
    anchor = _link(stage, f"/{root}/Geometry/anchor", nut)
    UsdPhysics.ArticulationRootAPI.Apply(anchor)
    anchor.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(anchor, 0.001)

    fixed = UsdPhysics.FixedJoint.Define(stage, f"/{root}/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path(f"/{root}")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(f"/{root}/Geometry/anchor")])

    parent, ppos = f"/{root}/Geometry/anchor", nut
    for i in range(g.N_NODES):
        p = rest[i]
        lat = _link(stage, f"/{root}/Geometry/lat_{i:02d}", p)
        _mass(lat, g.LINK_MASS)
        node = _link(stage, f"/{root}/Geometry/node_{i:02d}", p)
        _mass(node, g.LAST_MASS if i == g.N_NODES - 1 else g.LINK_MASS)
        lo, hi = g.node_extent(i)
        a = np.array([g.string_x(name, lo), -lo, g.string_z(name, lo)])
        b = np.array([g.string_x(name, hi), -hi, g.string_z(name, hi)])
        half = max(np.linalg.norm(b - a) / 2 - d / 2, 0.0005)
        _capsule(stage, f"/{root}/Geometry/node_{i:02d}/cap", (a + b) / 2 - p, half, d / 2, "Y", STRING, look,
                 phys, rot=_rot_y_to(b - a))

        for jname, axis, b0, b1, off in ((f"lat_{i:02d}", "X", parent, lat.GetPath(), p - ppos),
                                         (f"vert_{i:02d}", "Z", lat.GetPath(), node.GetPath(), np.zeros(3))):
            j = UsdPhysics.PrismaticJoint.Define(stage, f"/{root}/Physics/{jname}")
            j.CreateBody0Rel().SetTargets([Sdf.Path(str(b0))])
            j.CreateBody1Rel().SetTargets([Sdf.Path(str(b1))])
            j.CreateAxisAttr(axis)
            j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in off]))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
            j.CreateLowerLimitAttr(-g.JOINT_LIMIT)
            j.CreateUpperLimitAttr(g.JOINT_LIMIT)
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(float(k[i]))
            drive.CreateDampingAttr(float(g.DAMP_TAU * k[i]))
            drive.CreateMaxForceAttr(1.0e4)
            drive.CreateTargetPositionAttr(0.0)
        parent, ppos = str(node.GetPath()), p
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_ukulele(OUT / "ukulele.usda")
    for nm in g.NAMES:
        build_string(nm, OUT / f"string_{nm.lower()}.usda")
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
