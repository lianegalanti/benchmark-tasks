"""Write assets/parcel.usda: the parcel as one rigid body (a single box collider) with its printed marks as visual-only
prims, from the dimensions in geometry.py. Needs usd-core (``pxr``).

    python make_assets.py

The marks: on each of the four long faces, a white label with one bold black arrow (a box for the shaft, a triangle
mesh for the head) pointing to the up end (+z of the body); the up end's square face is a red panel. They sit 0.5 mm
proud of the faces and have no collision, so the fingers grip the plain box.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets" / "parcel.usda"
KRAFT = (0.66, 0.50, 0.33)   # brown corrugated cardboard
WHITE = (0.95, 0.95, 0.93)
BLACK = (0.04, 0.04, 0.04)
RED = (0.85, 0.12, 0.10)

# The four long faces: (name, outward normal, the face's "across" direction), body frame.
FACES = (("px", (1, 0, 0), (0, 1, 0)), ("nx", (-1, 0, 0), (0, -1, 0)),
         ("py", (0, 1, 0), (-1, 0, 0)), ("ny", (0, -1, 0), (1, 0, 0)))


def _cube(stage, path, centre, size, color, collide=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, centre)))
    cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    if collide:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def _triangle(stage, path, points, color):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in points]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    return mesh


def _thin(normal, across, w_across, h_up, t):
    """A box size (body frame) for a flat patch on a face: ``w_across`` wide, ``h_up`` tall, ``t`` thick."""
    n, a = np.abs(normal), np.abs(across)
    return n * t + a * w_across + np.array([0.0, 0.0, h_up])


def build(path: Path = OUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Parcel")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(g.MASS)
    _cube(stage, "/Parcel/box", (0.0, 0.0, 0.0), g.SIZE, KRAFT, collide=True)

    half_w = g.WIDTH / 2
    for name, normal, across in FACES:
        n, a = np.array(normal, float), np.array(across, float)
        label_c = n * (half_w + g.DECAL / 2) + np.array([0.0, 0.0, g.LABEL_Z])
        _cube(stage, f"/Parcel/label_{name}", label_c, _thin(n, a, g.LABEL[0], g.LABEL[1], g.DECAL), WHITE)
        sw, sl, s0 = g.SHAFT
        shaft_c = n * (half_w + 1.5 * g.DECAL) + np.array([0.0, 0.0, s0 + sl / 2])
        _cube(stage, f"/Parcel/arrow_shaft_{name}", shaft_c, _thin(n, a, sw, sl, g.DECAL), BLACK)
        base_z, (hw, hh) = s0 + sl, g.HEAD
        off = n * (half_w + 2.2 * g.DECAL)
        pts = [off + a * (-hw / 2) + [0, 0, base_z], off + a * (hw / 2) + [0, 0, base_z],
               off + [0, 0, base_z + hh]]
        _triangle(stage, f"/Parcel/arrow_head_{name}", pts, BLACK)
    # the red top panel on the up end
    _cube(stage, "/Parcel/top_panel", (0.0, 0.0, g.LENGTH / 2 + g.DECAL / 2),
          (g.WIDTH - 0.004, g.WIDTH - 0.004, g.DECAL), RED)
    stage.GetRootLayer().Save()
    return path


if __name__ == "__main__":
    print(build())
