"""Writes the task's USD assets from geometry.py:

- assets/manikin.usda: the head-and-shoulders manikin, one USD articulation fixed to the world by ``root_joint``.
  Root link ``base``: the bust and neck. ``neck`` on the revolute joint ``neck_pitch`` (about y, at the top of the
  neck); ``head`` on ``neck_yaw`` (about z): the head's convex collider, lips, eyes, ears and the bony bridge of the
  nose. ``nosef`` on the prismatic joint ``nose_f`` and ``nose`` on ``nose_y``: the soft lower nose (the lining of both
  nostrils, columella, alae, tip and sill), sprung in the plane across the nostrils. The task drives the neck back to
  straight (and makes it flinch) and the nose back to rest.
- assets/swab.usda: the swab, one rigid body: its cap (grip section with two flats, flange, plug), shaft and foam tip.
- assets/tube.usda: the transport tube and its rack, one rigid body (the task makes it kinematic: it stays put).

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

SKIN = (0.86, 0.70, 0.60)
LIP = (0.72, 0.42, 0.40)
DARK = (0.12, 0.08, 0.07)
GOWN = (0.45, 0.62, 0.72)
CAP = (0.15, 0.35, 0.75)
WHITE = (0.93, 0.93, 0.90)
FOAM = (0.97, 0.97, 0.95)
TUBE = (0.80, 0.84, 0.86)
RACK = (0.92, 0.92, 0.92)

FRICTION = {"skin": (0.6, 0.5), "grip": (1.0, 0.9), "plastic": (0.4, 0.35), "foam": (0.6, 0.5)}


def _material(stage, path: str, rgb, opacity: float = 1.0):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
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


def _bind(prim, look, phys=None) -> None:
    from pxr import UsdPhysics, UsdShade

    api = UsdShade.MaterialBindingAPI.Apply(prim)
    api.Bind(look)
    if phys is not None:
        UsdPhysics.CollisionAPI.Apply(prim)
        api.Bind(phys, UsdShade.Tokens.weakerThanDescendants, "physics")


def _xform_ops(prim, centre, rot=None, scale=None) -> None:
    from pxr import Gf, UsdGeom

    x = UsdGeom.Xformable(prim)
    x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if rot is not None and not np.allclose(rot, np.eye(3)):
        x.AddOrientOp().Set(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot)]))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in scale]))


def _box(stage, path, centre, size, rot, rgb, look, phys=None, invisible=False) -> None:
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _xform_ops(cube.GetPrim(), centre, rot, size)
    _bind(cube.GetPrim(), look, phys)
    if invisible:
        cube.MakeInvisible()


def _cyl(stage, path, centre, radius, length, axis, rgb, look, phys=None) -> None:
    """A cylinder along ``axis`` ("x"/"y"/"z")."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr(axis.upper())
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(length))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _xform_ops(cyl.GetPrim(), centre)
    _bind(cyl.GetPrim(), look, phys)


def _capsule(stage, path, centre, radius, length, axis, rgb, look, phys=None) -> None:
    """A capsule along ``axis``; ``length`` is its cylindrical part."""
    from pxr import Gf, UsdGeom

    cap = UsdGeom.Capsule.Define(stage, path)
    cap.CreateAxisAttr(axis.upper())
    cap.CreateRadiusAttr(float(radius))
    cap.CreateHeightAttr(float(length))
    cap.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _xform_ops(cap.GetPrim(), centre)
    _bind(cap.GetPrim(), look, phys)


def _sphere(stage, path, centre, radius, rgb, look) -> None:
    from pxr import Gf, UsdGeom

    s = UsdGeom.Sphere.Define(stage, path)
    s.CreateRadiusAttr(float(radius))
    s.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _xform_ops(s.GetPrim(), centre)
    _bind(s.GetPrim(), look)


def ellipsoid_mesh(centre, axes, n_lat: int = 16, n_lon: int = 24, u_min: float | None = None):
    """Points and faces of a lat-long ellipsoid (closed at both poles). With ``u_min`` (a fraction of the first
    axis), only the part with x >= u_min * axes[0] is kept and closed with a flat cap (a dome)."""
    c, a = np.asarray(centre, float), np.asarray(axes, float)
    pts, faces = [], []
    if u_min is None:
        pts.append(c + a * [0, 0, -1])
        rings = [-math.pi / 2 + math.pi * i / n_lat for i in range(1, n_lat)]
        for lat in rings:
            for j in range(n_lon):
                lon = 2 * math.pi * j / n_lon
                pts.append(c + a * [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
        pts.append(c + a * [0, 0, 1])
        top = len(pts) - 1
        for j in range(n_lon):
            faces.append((0, 1 + (j + 1) % n_lon, 1 + j))
        for i in range(len(rings) - 1):
            b0, b1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
            for j in range(n_lon):
                k = (j + 1) % n_lon
                faces.append((b0 + j, b0 + k, b1 + k, b1 + j))
        b = 1 + (len(rings) - 1) * n_lon
        for j in range(n_lon):
            faces.append((b + j, b + (j + 1) % n_lon, top))
        return np.array(pts), faces
    # dome along +x from x = u_min * a0 to the apex
    s0 = math.asin(max(-1.0, min(1.0, u_min)))
    pts.append(c + a * [u_min, 0, 0])                                      # base centre
    rings = [s0 + (math.pi / 2 - s0) * i / n_lat for i in range(n_lat)]
    for lat in rings:
        for j in range(n_lon):
            lon = 2 * math.pi * j / n_lon
            pts.append(c + a * [math.sin(lat), math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon)])
    pts.append(c + a * [1, 0, 0])
    top = len(pts) - 1
    for j in range(n_lon):
        faces.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 1):
        b0, b1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            k = (j + 1) % n_lon
            faces.append((b0 + k, b0 + j, b1 + j, b1 + k))
    b = 1 + (n_lat - 1) * n_lon
    for j in range(n_lon):
        faces.append((b + (j + 1) % n_lon, b + j, top))
    return np.array(pts), faces


def hull_mesh():
    """The head's collider: geometry.hull_triangles (PhysX takes its convex hull)."""
    return g.hull_triangles()


def _mesh(stage, path, pts, faces, rgb, look, phys=None, hull=False, invisible=False) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in pts]))
    m.CreateFaceVertexCountsAttr([len(f) for f in faces])
    m.CreateFaceVertexIndicesAttr([int(i) for f in faces for i in f])
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    _bind(m.GetPrim(), look, phys)
    if hull:
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
    if invisible:
        m.MakeInvisible()


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


def _link(stage, path: str, pos=(0.0, 0.0, 0.0), rot=None, gravity=True):
    from pxr import Sdf, UsdGeom, UsdPhysics

    x = UsdGeom.Xform.Define(stage, path)
    _xform_ops(x.GetPrim(), pos, rot)
    UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
    if not gravity:
        x.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
        x.GetPrim().CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
    return x.GetPrim()


def _joint(stage, path, kind, body0, body1, axis, pos0=(0, 0, 0), rot0=None, limits=None, drive=None):
    from pxr import Gf, Sdf, UsdPhysics

    j = (UsdPhysics.RevoluteJoint if kind == "revolute" else UsdPhysics.PrismaticJoint).Define(stage, path)
    j.CreateBody0Rel().SetTargets([Sdf.Path(body0)])
    j.CreateBody1Rel().SetTargets([Sdf.Path(body1)])
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    if rot0 is not None:
        j.CreateLocalRot0Attr(Gf.Quatf(*[float(v) for v in g.quat_wxyz(rot0)]))
    j.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    if limits is not None:
        j.CreateLowerLimitAttr(float(limits[0]))
        j.CreateUpperLimitAttr(float(limits[1]))
    if drive is not None:
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular" if kind == "revolute" else "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(float(drive[0]))
        d.CreateDampingAttr(float(drive[1]))
        d.CreateMaxForceAttr(float(drive[2]))
        d.CreateTargetPositionAttr(0.0)


def build_manikin(path: Path) -> None:
    from pxr import Sdf, UsdGeom, UsdPhysics

    stage = _stage("manikin")
    looks = {k: _material(stage, f"/manikin/Looks/{k}", v) for k, v in
             (("skin", SKIN), ("lip", LIP), ("dark", DARK), ("gown", GOWN))}
    skin = _physics_material(stage, "/manikin/Looks/skin_physics", *FRICTION["skin"])
    UsdGeom.Scope.Define(stage, "/manikin/Geometry")
    root = "/manikin/Geometry/base"

    base = _link(stage, root)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(base, 8.0, (0.0, 0.0, 0.05), (0.1, 0.06, 0.12))
    for name, c, s in g.bust_boxes():
        _box(stage, f"{root}/{name}", c, s, np.eye(3), GOWN, looks["gown"], skin)
    _capsule(stage, f"{root}/shoulders", g.SHOULDER_C, g.SHOULDER_R, g.SHOULDER_LEN, "y", GOWN, looks["gown"], skin)
    _cyl(stage, f"{root}/neck", g.NECK_C + [0, 0, sum(g.NECK_Z) / 2], g.NECK_R, g.NECK_Z[1] - g.NECK_Z[0], "z",
         SKIN, looks["skin"], skin)

    neck = _link(stage, f"{root}/neck_link", g.PIVOT)
    _mass(neck, 0.1, (0.0, 0.0, 0.0), (1e-4, 1e-4, 1e-4))
    head = _link(stage, f"{root}/neck_link/head")
    hp = f"{root}/neck_link/head"
    a, b, c = g.HEAD_AX
    m = g.HEAD_MASS
    _mass(head, m, g.HEAD_C, (m * (b * b + c * c) / 5, m * (a * a + c * c) / 5, m * (a * a + b * b) / 5))
    pts, faces = hull_mesh()
    _mesh(stage, f"{hp}/skull_collider", pts, faces, SKIN, looks["skin"], skin, hull=True, invisible=True)
    pts, faces = ellipsoid_mesh(g.HEAD_C, g.HEAD_AX)
    _mesh(stage, f"{hp}/skin", pts, faces, SKIN, looks["skin"])
    for name, cen, r, length in g.LIPS:
        _capsule(stage, f"{hp}/{name}", cen, r, length, "y", LIP, looks["lip"], skin)
    for s in (-1, 1):
        _sphere(stage, f"{hp}/eye_{'l' if s < 0 else 'r'}", g.HEAD_C + [-0.083, 0.032 * s, 0.014], 0.010, DARK,
                looks["dark"])
        pts, faces = ellipsoid_mesh(g.HEAD_C + [0.005, 0.073 * s, -0.008], (0.018, 0.010, 0.030), 8, 12)
        _mesh(stage, f"{hp}/ear_{'l' if s < 0 else 'r'}", pts, faces, SKIN, looks["skin"])
        _box(stage, f"{hp}/brow_{'l' if s < 0 else 'r'}", g.HEAD_C + [-0.090, 0.030 * s, 0.030],
             (0.010, 0.034, 0.006), np.eye(3), DARK, looks["dark"])
    # the bony bridge of the nose, from the top of the soft lower nose to between the eyes (visual)
    top = g.N0 + 0.028 * g.U
    nasion = g.HEAD_C + np.array([-0.092, 0.0, 0.030])
    d = nasion - top
    ax = d / np.linalg.norm(d)
    rot = np.column_stack([np.cross([0.0, 1.0, 0.0], ax), [0.0, 1.0, 0.0], ax])
    _box(stage, f"{hp}/bridge", (top + nasion) / 2, (0.014, 0.016, np.linalg.norm(d) + 0.006), rot, SKIN,
         looks["skin"])

    np_ = f"{hp}/nosef"
    nosef = _link(stage, np_, g.N0, g.R_N, gravity=False)
    _mass(nosef, 0.01, (0.0, 0.0, 0.0), (1e-6, 1e-6, 1e-6))
    nose = _link(stage, f"{np_}/nose", gravity=False)
    nn = f"{np_}/nose"
    _mass(nose, g.NOSE_MASS, (0.012, 0.0, 0.0), (3e-6, 3e-6, 3e-6))
    for side in g.NOSTRIL_Y:
        for name, cen, size, rot in g.lumen_staves(side):
            _box(stage, f"{nn}/{name}", cen, size, rot, SKIN, looks["skin"], skin, invisible=True)
    for name, cen, size in g.nose_outer_boxes():
        _box(stage, f"{nn}/{name}", cen, size, np.eye(3), SKIN, looks["skin"], skin, invisible=True)
    # what you see: a rounded lower nose, open at its base, with the two dark nostrils (visual)
    pts, faces = ellipsoid_mesh((0.0, 0.0, 0.002), (0.030, 0.019, 0.016), 8, 20, u_min=0.0)
    _mesh(stage, f"{nn}/lower_nose", pts, faces, SKIN, looks["skin"])
    for side, yc in g.NOSTRIL_Y.items():
        pts, faces = ellipsoid_mesh((-0.0004, yc, 0.0), (0.0006, g.SEMI_Y, g.SEMI_F), 2, 16)
        _mesh(stage, f"{nn}/nostril_{side}", pts, faces, DARK, looks["dark"])

    UsdGeom.Scope.Define(stage, "/manikin/Physics")
    lim = math.degrees(g.NECK_LIMIT)
    k_deg = g.NECK_K * math.pi / 180
    d_deg = g.NECK_D * math.pi / 180
    _joint(stage, "/manikin/Physics/neck_pitch", "revolute", root, f"{root}/neck_link", "Y", g.PIVOT,
           limits=(-lim, lim), drive=(k_deg, d_deg, g.NECK_CAP))
    _joint(stage, "/manikin/Physics/neck_yaw", "revolute", f"{root}/neck_link", hp, "Z",
           limits=(-lim, lim), drive=(k_deg, d_deg, g.NECK_CAP))
    _joint(stage, "/manikin/Physics/nose_f", "prismatic", hp, np_, "Z", g.N0, g.R_N,
           limits=(-g.NOSE_LIMIT, g.NOSE_LIMIT), drive=(g.NOSE_K, g.NOSE_D, 20.0))
    _joint(stage, "/manikin/Physics/nose_y", "prismatic", np_, nn, "Y",
           limits=(-g.NOSE_LIMIT, g.NOSE_LIMIT), drive=(g.NOSE_K, g.NOSE_D, 20.0))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/manikin/Physics/root_joint")
    fixed.CreateBody0Rel().SetTargets([Sdf.Path("/manikin")])
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(root)])
    stage.GetRootLayer().Export(str(path))


def build_swab(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("swab")
    looks = {k: _material(stage, f"/swab/Looks/{k}", v) for k, v in (("cap", CAP), ("white", WHITE), ("foam", FOAM))}
    phys = {k: _physics_material(stage, f"/swab/Looks/{k}_physics", *FRICTION[k]) for k in ("grip", "plastic", "foam")}
    body = stage.GetPrimAtPath("/swab")
    UsdPhysics.RigidBodyAPI.Apply(body)
    L = g.SWAB_LEN
    _mass(body, g.SWAB_MASS, (0.006, 0.0, 0.0), (g.SWAB_MASS * 0.016 ** 2 / 8, g.SWAB_MASS * L ** 2 / 20,
                                                 g.SWAB_MASS * L ** 2 / 20))
    _box(stage, "/swab/grip", (0.0, 0.0, 0.0), g.GRIP, np.eye(3), CAP, looks["cap"], phys["grip"])
    for name, (r, x0, x1) in (("flange", g.FLANGE), ("plug", g.PLUG)):
        _cyl(stage, f"/swab/{name}", ((x0 + x1) / 2, 0.0, 0.0), r, x1 - x0, "x", CAP, looks["cap"], phys["plastic"])
    r0, r1, x0, x1 = g.CONE                      # the plug's tapered lead-in: a convex frustum
    n = 16
    pts = [(x, r * math.cos(2 * math.pi * k / n), r * math.sin(2 * math.pi * k / n)) for x, r in ((x0, r0), (x1, r1))
           for k in range(n)]
    faces = [tuple(range(n))[::-1], tuple(range(n, 2 * n))] + [(k, (k + 1) % n, n + (k + 1) % n, n + k) for k in range(n)]
    _mesh(stage, "/swab/lead_in", pts, faces, CAP, looks["cap"], phys["plastic"], hull=True)
    r, x0, x1 = g.SHAFT
    _cyl(stage, "/swab/shaft", ((x0 + x1) / 2, 0.0, 0.0), r, x1 - x0, "x", WHITE, looks["white"], phys["plastic"])
    r, x0, x1 = g.TIP
    _capsule(stage, "/swab/tip", ((x0 + x1) / 2, 0.0, 0.0), r, x1 - x0 - 2 * r, "x", FOAM, looks["foam"],
             phys["foam"])
    stage.GetRootLayer().Export(str(path))


def build_tube(path: Path) -> None:
    from pxr import UsdPhysics

    stage = _stage("tube")
    looks = {"tube": _material(stage, "/tube/Looks/tube", TUBE, 0.55), "rack": _material(stage, "/tube/Looks/rack",
                                                                                         RACK)}
    phys = _physics_material(stage, "/tube/Looks/plastic_physics", *FRICTION["plastic"])
    body = stage.GetPrimAtPath("/tube")
    UsdPhysics.RigidBodyAPI.Apply(body)
    _mass(body, 0.15, (0.0, 0.0, -0.12), (1e-3, 1e-3, 1e-3))
    for name, c, s, a in g.tube_staves():
        _box(stage, f"/tube/{name}", c, s, g.rot_z(a), TUBE, looks["tube"], phys)
    _cyl(stage, "/tube/bottom", (0.0, 0.0, -g.TUBE_LEN + g.TUBE_BOTTOM / 2), g.TUBE_R_OUT, g.TUBE_BOTTOM, "z", TUBE,
         looks["tube"], phys)
    for name, c, s in g.rack_boxes():
        _box(stage, f"/tube/{name}", c, s, np.eye(3), RACK, looks["rack"], phys)
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_manikin(OUT / "manikin.usda")
    build_swab(OUT / "swab.usda")
    build_tube(OUT / "tube.usda")
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
