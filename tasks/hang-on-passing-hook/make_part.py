"""Writes assets/hanger_part.usda: the hanger part (a block with a rectangular eye and a grip tab on top) as one
rigid body made of five boxes.

    python make_part.py        # needs usd-core (pip install usd-core)

The body's origin is at the centre of the eye's opening; +x points through the opening, +z up (see geometry.py).
The mass and centre of mass (uniform material) are set explicitly; PhysX computes the inertia from the colliders.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import geometry as g

HERE = Path(__file__).resolve().parent
ORANGE = (0.95, 0.45, 0.10)


def build(path: Path) -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Part")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(root.GetPrim())
    mass.CreateMassAttr(g.MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*(float(v) for v in g.COM)))
    for name, (centre, size) in g.PART_BOXES.items():
        cube = UsdGeom.Cube.Define(stage, f"/Part/{name}")
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*ORANGE)])
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(*(float(v) for v in centre)))
        xf.SetScale(Gf.Vec3f(*(float(v) for v in size)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Export(str(path))  # overwrites an older copy


if __name__ == "__main__":
    out = HERE / "assets" / "hanger_part.usda"
    out.parent.mkdir(exist_ok=True)
    build(out)
    print(f"wrote {out}")
