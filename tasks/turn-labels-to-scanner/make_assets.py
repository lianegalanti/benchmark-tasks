"""Write assets/carton.usda: one shipping carton as a single rigid body (one box collider) with its shipping label as
visual-only prims, from the dimensions in geometry.py. Needs usd-core (``pxr``).

    python make_assets.py

The carton's own frame has its origin at the box's centre, +z up and the label on its +x face. The label is a white
patch with black barcode bars on that face, under an orange header band; the band folds over the top edge onto the
top face, so the top camera sees which side the label is on. All marks sit 0.5 mm proud of the faces and have no
collision, so the fingers grip the plain box.
"""

from __future__ import annotations

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "carton.usda"
KRAFT = (0.66, 0.50, 0.33)   # brown corrugated cardboard
WHITE = (0.96, 0.96, 0.94)
BLACK = (0.03, 0.03, 0.03)
ORANGE = (0.98, 0.45, 0.05)


def _cube(stage, path, centre, size, color, collide=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def build(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Carton")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(g.MASS)
    _cube(stage, "/Carton/box", (0.0, 0.0, 0.0), g.SIZE, KRAFT, collide=True)

    hx, hz, d = g.SIZE[0] / 2, g.SIZE[2] / 2, g.DECAL
    w, h = g.LABEL                                   # across the face (y), up the face (z)
    band = g.BAND                                    # the orange header band's height on the side face
    label_bottom = hz - band - h
    # white label on the +x face, under the band
    _cube(stage, "/Carton/label", (hx + d / 2, 0.0, label_bottom + h / 2), (d, w, h), WHITE)
    # black barcode bars on the label
    for k, (y0, bw) in enumerate(g.BARS):
        _cube(stage, f"/Carton/bar_{k}", (hx + 1.5 * d, y0 + bw / 2, label_bottom + h / 2),
              (d, bw, h - 2 * g.BAR_MARGIN), BLACK)
    # the orange header band: on the side face above the label, and folded over the top edge onto the top face
    _cube(stage, "/Carton/band_side", (hx + d / 2, 0.0, hz - band / 2), (d, w, band), ORANGE)
    _cube(stage, "/Carton/band_top", (hx - g.BAND_TOP / 2, 0.0, hz + d / 2), (g.BAND_TOP, w, d), ORANGE)
    stage.GetRootLayer().Save()
    return path


if __name__ == "__main__":
    print(build())
