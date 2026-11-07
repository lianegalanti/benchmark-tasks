"""Writes assets/bottle.usda: the baby bottle as one rigid body made of simple colliders.

    python make_bottle.py        # needs usd-core (pip install usd-core)

The open tube is 16 flat wall staves (boxes) around the axis, the last 15 mm of them amber (the screw collar at the
mouth), closed at the bottom by a 4 mm disc. The body's origin is on the axis halfway along the tube, and its +z points
out of the mouth. The centre of mass is set explicitly (uniform wall and bottom); PhysX computes the inertia from the
colliders.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import geometry as g

HERE = Path(__file__).resolve().parent
CLEAR = (0.88, 0.92, 0.96)   # clear-ish plastic
AMBER = (0.95, 0.68, 0.15)   # the collar


def _box(stage, path, centre, size, yaw, color):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xf = UsdGeom.XformCommonAPI(cube)
    xf.SetTranslate(Gf.Vec3d(*centre))
    xf.SetRotate(Gf.Vec3f(0.0, 0.0, math.degrees(yaw)))
    xf.SetScale(Gf.Vec3f(*size))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def build(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Bottle")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(g.MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, g.COM_Z))

    r_mid = g.R_OUT - g.WALL / 2
    body_len = g.LENGTH - g.COLLAR
    for k in range(g.STAVES):
        a = 2 * math.pi * k / g.STAVES  # stave 0 faces +x of the bottle frame
        c, s = math.cos(a), math.sin(a)
        _box(stage, f"/Bottle/wall_{k:02d}", (r_mid * c, r_mid * s, -g.LENGTH / 2 + body_len / 2),
             (g.WALL, g.STAVE_W, body_len), a, CLEAR)
        _box(stage, f"/Bottle/collar_{k:02d}", (r_mid * c, r_mid * s, g.MOUTH_Z - g.COLLAR / 2),
             (g.WALL, g.STAVE_W, g.COLLAR), a, AMBER)

    bottom = UsdGeom.Cylinder.Define(stage, "/Bottle/bottom")
    bottom.CreateAxisAttr("Z")
    bottom.CreateRadiusAttr(g.R_IN + 0.001)  # 1 mm into the staves: no gap at the edge
    bottom.CreateHeightAttr(g.BOTTOM_T)
    bottom.CreateDisplayColorAttr([Gf.Vec3f(*CLEAR)])
    UsdGeom.XformCommonAPI(bottom).SetTranslate(Gf.Vec3d(0.0, 0.0, -g.LENGTH / 2 + g.BOTTOM_T / 2))
    UsdPhysics.CollisionAPI.Apply(bottom.GetPrim())
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    out = HERE / "assets" / "bottle.usda"
    out.parent.mkdir(exist_ok=True)
    build(out)
    print(f"wrote {out}")
