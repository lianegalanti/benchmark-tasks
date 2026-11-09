"""Write assets/basket.usda: the basket as one rigid body (a floor, four walls and the handle tab, each a box
collider), from the dimensions in task.py. Needs usd-core (``pip install usd-core``); run from this folder:

    python make_basket.py
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "basket.usda"


def main() -> None:
    spec = importlib.util.spec_from_file_location("catch_task", HERE / "task.py")
    task = importlib.util.module_from_spec(spec)  # imported directly: loading it as a task checks this file exists
    spec.loader.exec_module(task)
    OUT.parent.mkdir(exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Basket")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(task.BASKET_MASS))
    for name, (centre, size) in task.basket_boxes().items():
        cube = UsdGeom.Cube.Define(stage, f"/Basket/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*centre))
        cube.AddScaleOp().Set(Gf.Vec3f(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*task.BLUE)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    OUT.write_text(stage.GetRootLayer().ExportToString())
    print(f"wrote {OUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()
