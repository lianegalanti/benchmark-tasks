"""Writes the task's USD assets from geometry.py:

- assets/trailer.usda: the cropped trailer nose, a fixed-base articulation in the nose frame (station axes, origin
  at ``geometry.NOSE``). Root link ``nose``: bulkhead panel, mounting bracket, two pedestals and the two trailer glad
  hands' castings (bolted face up). Child links ``seal_blue`` / ``seal_red``: each trailer glad hand's seal ring, on a
  prismatic joint along the seal axis (positive = pressed in), sprung outward against its outer stop.
- assets/tender.usda: the tractor's tender post and its two air lines, a fixed-base articulation in the tender frame
  (station axes, origin at ``geometry.TENDER``). Root link ``post``. Per line (``blue_*`` / ``red_*``): the hose guide
  pivots on the post (``yaw`` about z, ``pitch`` about y), the hose slides through it (``slide``, prismatic along
  the hose: it pays out and takes up slack between L_MIN and its end stop L_MAX), and the glad hand turns on a
  three-axis swivel at the hose end (``roll``, ``bend``, ``twist``) standing in for the hose's flex. Every link is
  authored at the start pose, so every joint starts at 0.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

COLOUR = {"blue": (0.10, 0.25, 0.70), "red": (0.72, 0.08, 0.07)}
RUBBER = (0.05, 0.05, 0.05)
STEEL = (0.45, 0.46, 0.48)
PAINT = (0.86, 0.86, 0.83)       # the trailer nose's painted steel
FRICTION = {"cast": (0.6, 0.5), "rubber": (0.9, 0.8), "steel": (0.5, 0.4)}


def _quat(r):
    from pxr import Gf

    x, y, z, w = g.quat_xyzw(r)
    return Gf.Quatf(float(w), float(x), float(y), float(z))


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


class Kit:
    def __init__(self, root: str):
        from pxr import Usd, UsdGeom, UsdPhysics

        self.root = root
        self.stage = stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
        prim = UsdGeom.Xform.Define(stage, f"/{root}")
        stage.SetDefaultPrim(prim.GetPrim())
        UsdGeom.Scope.Define(stage, f"/{root}/Looks")
        UsdGeom.Scope.Define(stage, f"/{root}/Geometry")
        UsdGeom.Scope.Define(stage, f"/{root}/Physics")
        self.looks, self.phys = {}, {k: _physics_material(stage, f"/{root}/Looks/{k}_physics", *v)
                                     for k, v in FRICTION.items()}

    def look(self, name, rgb):
        if name not in self.looks:
            self.looks[name] = _material(self.stage, f"/{self.root}/Looks/{name}", rgb)
        return self.looks[name]

    def body(self, name, pos=(0, 0, 0), rot=np.eye(3)):
        from pxr import Gf, UsdGeom, UsdPhysics

        x = UsdGeom.Xform.Define(self.stage, f"/{self.root}/Geometry/{name}")
        x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos]))
        x.AddOrientOp().Set(_quat(rot))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        return x.GetPrim()

    def box(self, body: str, name: str, box, rgb, look: str, phys: str | None):
        from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

        c, r, h = box
        cube = UsdGeom.Cube.Define(self.stage, f"/{self.root}/Geometry/{body}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in c]))
        cube.AddOrientOp().Set(_quat(r))
        cube.AddScaleOp().Set(Gf.Vec3f(*[float(2 * v) for v in h]))
        binding = UsdShade.MaterialBindingAPI.Apply(cube.GetPrim())
        binding.Bind(self.look(look, rgb))
        if phys:
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            binding.Bind(self.phys[phys], UsdShade.Tokens.weakerThanDescendants, "physics")

    def capsule(self, body: str, name: str, centre, radius, height, axis, rgb, look):
        from pxr import Gf, UsdGeom, UsdShade

        cap = UsdGeom.Capsule.Define(self.stage, f"/{self.root}/Geometry/{body}/{name}")
        cap.CreateRadiusAttr(float(radius))
        cap.CreateHeightAttr(float(height))
        cap.CreateAxisAttr(axis)
        cap.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        cap.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
        UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(self.look(look, rgb))

    def joint(self, kind, name, body0, body1, axis, pos0=(0, 0, 0), rot0=np.eye(3), rot1=np.eye(3), lower=None,
              upper=None, linear=False):
        from pxr import Gf, Sdf, UsdPhysics

        cls = UsdPhysics.PrismaticJoint if kind == "prismatic" else UsdPhysics.RevoluteJoint
        j = cls.Define(self.stage, f"/{self.root}/Physics/{name}")
        j.CreateBody0Rel().SetTargets([Sdf.Path(f"/{self.root}/Geometry/{body0}")])
        j.CreateBody1Rel().SetTargets([Sdf.Path(f"/{self.root}/Geometry/{body1}")])
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in pos0]))
        j.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot0Attr(_quat(rot0))
        j.CreateLocalRot1Attr(_quat(rot1))
        conv = (lambda v: float(v)) if kind == "prismatic" else (lambda v: float(math.degrees(v)))
        if lower is not None:
            j.CreateLowerLimitAttr(conv(lower))
            j.CreateUpperLimitAttr(conv(upper))
        UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear" if kind == "prismatic" else "angular")
        return j

    def fix_root(self, body: str):
        from pxr import Sdf, UsdPhysics

        fixed = UsdPhysics.FixedJoint.Define(self.stage, f"/{self.root}/Physics/root_joint")
        fixed.CreateBody0Rel().SetTargets([Sdf.Path(f"/{self.root}")])
        fixed.CreateBody1Rel().SetTargets([Sdf.Path(f"/{self.root}/Geometry/{body}")])

    def export(self, path: Path):
        self.stage.GetRootLayer().Export(str(path))


def _mass(prim, mass, com, inertia) -> None:
    from pxr import Gf, UsdPhysics

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(*[float(v) for v in inertia]))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def _box_inertia(mass, size):
    a, b, c = size
    return (mass / 12 * (b * b + c * c), mass / 12 * (a * a + c * c), mass / 12 * (a * a + b * b))


def _casting(kit: Kit, body: str, colour: str, pos, rot, prefix: str = "") -> None:
    """A glad hand's casting boxes, G placed at (pos, rot) in the body's frame."""
    rgb = COLOUR[colour]
    for name, box in g.casting_boxes():
        kit.box(body, f"{prefix}{name}", g.place([box], pos, rot)[0], rgb, f"cast_{colour}", "cast")


def build_trailer(path: Path) -> None:
    from pxr import Sdf, UsdPhysics

    kit = Kit("trailer")
    nose = kit.body("nose")
    UsdPhysics.ArticulationRootAPI.Apply(nose)
    nose.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(nose, 20.0, (-0.05, 0.0, 0.1), (1.0, 1.0, 1.0))
    kit.box("nose", "bulkhead", g.lohi(*g.BULKHEAD), PAINT, "paint", "steel")
    kit.box("nose", "bracket", g.lohi(*g.BRACKET), STEEL, "steel", "steel")
    for colour in g.COLOURS:
        t = g.TRAILER_GH[colour] - g.NOSE
        rt = g.R_TRAILER[colour]
        _casting(kit, "nose", colour, t, rt, prefix=f"{colour}_")
        hx, hy = g.PEDESTAL_HALF
        top = t[2] + g.BACK[0][2]
        kit.box("nose", f"{colour}_pedestal", g.lohi((t[0] - hx, t[1] - hy, g.BRACKET[1][2]), (t[0] + hx, t[1] + hy, top)),
                STEEL, "steel", "steel")
        link = f"seal_{colour}"
        seal = kit.body(link, t, rt)
        _mass(seal, g.RING_MASS, (0.0, 0.0, 0.0), (2e-6, 2e-6, 3e-6))
        for name, box in g.ring_boxes(0.0):
            kit.box(link, name, box, RUBBER, "rubber", "rubber")
        flip = g.rot_x(math.pi)
        j = kit.joint("prismatic", link, "nose", link, "Z", pos0=t, rot0=rt @ flip, rot1=flip, lower=0.0,
                      upper=g.RING_TRAVEL)
        d = UsdPhysics.DriveAPI.Get(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(float(g.RING_RATE))
        d.CreateDampingAttr(float(g.RING_DAMPING))
        d.CreateMaxForceAttr(200.0)
    kit.fix_root("nose")
    kit.export(path)


def build_tender(path: Path) -> None:
    from pxr import Sdf, UsdPhysics

    kit = Kit("tender")
    post = kit.body("post")
    UsdPhysics.ArticulationRootAPI.Apply(post)
    post.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(post, 5.0, (0.0, 0.0, 0.1), (0.1, 0.1, 0.1))
    kit.box("post", "post", g.lohi(*g.POST), STEEL, "steel", "steel")
    tiny = (2e-6, 2e-6, 2e-6)
    for colour in g.COLOURS:
        psi, theta, length, a, b, c = g.start_chain(colour)
        guide = g.GUIDE[colour] - g.TENDER
        r_yaw = g.rot_z(psi)
        r_hose = r_yaw @ g.rot_y(theta)
        port = guide + r_hose[:, 0] * length
        r_roll = r_hose @ g.rot_x(a)
        r_bend = r_roll @ g.rot_y(b)
        r_gh = r_bend @ g.rot_z(c)
        n = {k: f"{colour}_{k}" for k in ("yaw", "pitch", "hose", "roll", "bend", "gladhand")}
        for key, pos, rot in (("yaw", guide, r_yaw), ("pitch", guide, r_hose), ("roll", port, r_roll),
                              ("bend", port, r_bend)):
            _mass(kit.body(n[key], pos, rot), g.LINK_MASS, (0, 0, 0), tiny)
        hose = kit.body(n["hose"], port, r_hose)
        _mass(hose, g.HOSE_MASS, (-0.12, 0.0, 0.0), (1e-5, 6e-4, 6e-4))
        kit.capsule(n["hose"], "hose", (-g.HOSE_VIS / 2, 0.0, 0.0), g.HOSE_R, g.HOSE_VIS - 2 * g.HOSE_R, "X", RUBBER,
                    "rubber")
        kit.capsule(n["pitch"], "guide", (0.0, 0.0, 0.0), 0.014, 0.012, "X", STEEL, "steel")
        gh = kit.body(n["gladhand"], port, r_gh)
        _mass(gh, g.GH_MASS, g.GH_COM - g.PORT, _box_inertia(g.GH_MASS, g.GH_SIZE))
        rgb = COLOUR[colour]
        for name, box in g.tractor_boxes():
            phys = "rubber" if name.startswith("seal") else "cast"
            kit.box(n["gladhand"], name, g.place([box], -g.PORT, np.eye(3))[0],
                    RUBBER if phys == "rubber" else rgb, "rubber" if phys == "rubber" else f"cast_{colour}", phys)
        lim = math.radians(175)
        kit.joint("revolute", f"{colour}_yaw", "post", n["yaw"], "Z", pos0=guide, rot0=r_yaw,
                  lower=-math.radians(120), upper=math.radians(120))
        kit.joint("revolute", f"{colour}_pitch", n["yaw"], n["pitch"], "Y", rot0=g.rot_y(theta),
                  lower=g.PITCH_LIM[0] - theta, upper=g.PITCH_LIM[1] - theta)
        kit.joint("prismatic", f"{colour}_slide", n["pitch"], n["hose"], "X", pos0=(length, 0.0, 0.0),
                  lower=g.L_MIN - length, upper=g.L_MAX - length)
        kit.joint("revolute", f"{colour}_roll", n["hose"], n["roll"], "X", rot0=g.rot_x(a), lower=-lim, upper=lim)
        kit.joint("revolute", f"{colour}_bend", n["roll"], n["bend"], "Y", rot0=g.rot_y(b),
                  lower=-g.BEND_MAX - b, upper=g.BEND_MAX - b)
        kit.joint("revolute", f"{colour}_twist", n["bend"], n["gladhand"], "Z", rot0=g.rot_z(c), lower=-lim, upper=lim)
    kit.fix_root("post")
    kit.export(path)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_trailer(OUT / "trailer.usda")
    build_tender(OUT / "tender.usda")
    print("wrote", ", ".join(p.name for p in sorted(OUT.glob("*.usda"))))
