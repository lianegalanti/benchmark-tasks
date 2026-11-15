"""Writes assets/strut_plate.usda: the 3-hole steel strut plate as one rigid body made of ten boxes (the material
around its three holes, each hole a 14.3 mm square opening).

    python make_part.py        # needs usd-core (pip install usd-core)

The body's origin is at the centre of hole 0; +x points through the holes, +z along the plate (see geometry.py).
The mass (the real plate's, with round holes) and centre of mass (the plate's centre) are set explicitly; PhysX computes the inertia from the colliders.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import geometry as g

HERE = Path(__file__).resolve().parent
ZINC = (0.72, 0.74, 0.77)


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
        cube.CreateDisplayColorAttr([Gf.Vec3f(*ZINC)])
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(*(float(v) for v in centre)))
        xf.SetScale(Gf.Vec3f(*(float(v) for v in size)))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Export(str(path))  # overwrites an older copy


if __name__ == "__main__":
    out = HERE / "assets" / "strut_plate.usda"
    out.parent.mkdir(exist_ok=True)
    build(out)
    print(f"wrote {out}")
