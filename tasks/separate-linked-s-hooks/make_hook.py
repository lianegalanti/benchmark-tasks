"""Writes assets/s_hook.usda: one S-hook as one rigid body made of capsules (both hooks in the scene use it).

    python make_hook.py        # needs usd-core (pip install usd-core)

The rod's centre line (geometry.hook_polyline: two 160-degree curls of 8 chords each and the straight middle stroke)
becomes 17 capsules of the rod's 5 mm radius, end to end, so the rod is smooth at every joint. The body's origin is
the hook's centre, which is also its centre of mass (the hook is point-symmetric about it); PhysX computes the
inertia from the colliders.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import geometry as g

HERE = Path(__file__).resolve().parent
ALUMINIUM = (0.80, 0.82, 0.85)


def build(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/SHook")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(g.MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))

    for k, (a, b) in enumerate(g.SEGMENTS):
        d = b - a
        cap = UsdGeom.Capsule.Define(stage, f"/SHook/rod_{k:02d}")
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(g.ROD_R)
        cap.CreateHeightAttr(float(math.hypot(d[0], d[2])))  # the cylindrical part; the caps reach the joints
        cap.CreateDisplayColorAttr([Gf.Vec3f(*ALUMINIUM)])
        xf = UsdGeom.XformCommonAPI(cap)
        xf.SetTranslate(Gf.Vec3d(*[float(v) for v in (a + b) / 2]))
        xf.SetRotate(Gf.Vec3f(0.0, -math.degrees(math.atan2(d[2], d[0])), 0.0))  # +x onto the chord
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    stage.GetRootLayer().Save()


if __name__ == "__main__":
    out = HERE / "assets" / "s_hook.usda"
    out.parent.mkdir(exist_ok=True)
    build(out)
    print(f"wrote {out}")
