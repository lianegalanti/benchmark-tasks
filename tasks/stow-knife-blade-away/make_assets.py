"""Write assets/knife.usda: the knife as one compound rigid body (a wooden handle box and a two-part steel blade, all
colliders of the same body), from the dimensions in geometry.py. Needs usd-core (``pxr``).

    python make_assets.py
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "knife.usda"
COLORS = {"handle": (0.36, 0.20, 0.10), "blade": (0.78, 0.80, 0.83), "tip": (0.78, 0.80, 0.83)}


def build(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Knife")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(g.KNIFE_MASS)
    for name, (centre, size) in g.KNIFE_PARTS.items():
        cube = UsdGeom.Cube.Define(stage, f"/Knife/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
        cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
        cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*COLORS[name])]))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    stage.GetRootLayer().Save()
    return path


if __name__ == "__main__":
    print(build())
