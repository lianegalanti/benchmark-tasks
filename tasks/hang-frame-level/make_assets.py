"""Write assets/frame.usda, the picture frame as one compound rigid body, from the dimensions in geometry.py.

    python make_assets.py            # rewrites assets/frame.usda
    python make_assets.py --check    # exits 1 if the file on disk differs from what this would write

Plain text, no USD library needed. The body's mass, centre of mass and inertia are set explicitly (the medal makes
the frame heavier on its lower left); its colliders are boxes: the front (glass, picture and backing), the four rear
rails of the moulding, the hanger bar and its 12 teeth (boxes turned 45 degrees about x). The mat, the medal
and ribbon on the front are visual only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import geometry as g

OUT = Path(__file__).resolve().parent / "assets" / "frame.usda"

WOOD = (0.28, 0.17, 0.09)
STEEL = (0.62, 0.63, 0.66)
MAT = (0.93, 0.92, 0.88)
BRONZE = (0.72, 0.50, 0.20)
RIBBON = (0.70, 0.10, 0.12)


def _f(v) -> str:
    return "(" + ", ".join(f"{float(x):.6g}" for x in v) + ")"


def _cube(name, centre, size, color, collide=True, rot_x_deg=0.0) -> str:
    api = '\n    (\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )' if collide else ""
    ops = ['"xformOp:translate"'] + (['"xformOp:rotateX"'] if rot_x_deg else []) + ['"xformOp:scale"']
    rot = f"        float xformOp:rotateX = {rot_x_deg:.6g}\n" if rot_x_deg else ""
    return (f'    def Cube "{name}"{api}\n    {{\n'
            f"        color3f[] primvars:displayColor = [{_f(color)}]\n"
            f"        double size = 1\n"
            f"        double3 xformOp:translate = {_f(centre)}\n{rot}"
            f"        float3 xformOp:scale = {_f(size)}\n"
            f"        uniform token[] xformOpOrder = [{', '.join(ops)}]\n    }}\n")


def _disc(name, centre, radius, height, color) -> str:
    return (f'    def Cylinder "{name}"\n    {{\n'
            f'        uniform token axis = "X"\n'
            f"        color3f[] primvars:displayColor = [{_f(color)}]\n"
            f"        double height = {height:.6g}\n        double radius = {radius:.6g}\n"
            f"        double3 xformOp:translate = {_f(centre)}\n"
            f'        uniform token[] xformOpOrder = ["xformOp:translate"]\n    }}\n')


def _quat_wxyz(rot: np.ndarray) -> np.ndarray:
    m = rot
    w = np.sqrt(max(0.0, 1.0 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) / 2, m[2, 1] - m[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] + m[1, 1] - m[2, 2])) / 2, m[0, 2] - m[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - m[0, 0] - m[1, 1] + m[2, 2])) / 2, m[1, 0] - m[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def principal() -> tuple[np.ndarray, np.ndarray]:
    """Principal moments and the rotation whose columns are the principal axes (a proper rotation)."""
    inertia, _ = g.inertia()
    moments, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    return moments, axes


def frame_usda() -> str:
    moments, axes = principal()
    parts = [_cube(name, c, s, STEEL if name.startswith("hanger") else WOOD) for name, c, s in g.collision_boxes()]
    side = g.PITCH / np.sqrt(2.0)
    parts += [_cube(f"tooth{i:02d}", c, (g.HANGER_T, side, side), STEEL, rot_x_deg=45.0)
              for i, c in enumerate(g.teeth())]
    front = -g.D / 2
    parts += [
        _cube("mat", (front - 0.0005, 0.0, 0.0), (0.001, g.W - 2 * g.RAIL - 0.01, g.H - 2 * g.RAIL - 0.01), MAT,
              collide=False),
        _disc("medal", (front - 0.0015, g.MEDAL_AT[1], g.MEDAL_AT[2]), g.MEDAL_R, 0.002, BRONZE),
        _cube("ribbon", (front - 0.0012, g.MEDAL_AT[1], g.MEDAL_AT[2] + g.MEDAL_R + 0.012), (0.0014, 0.014, 0.026),
              RIBBON, collide=False),
    ]
    return ('#usda 1.0\n(\n    defaultPrim = "Frame"\n    metersPerUnit = 1\n    kilogramsPerUnit = 1\n'
            '    upAxis = "Z"\n)\n\n'
            'def Xform "Frame" (\n    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n)\n{\n'
            f"    float physics:mass = {g.MASS:.6g}\n"
            f"    point3f physics:centerOfMass = {_f(g.COM)}\n"
            f"    float3 physics:diagonalInertia = {_f(moments)}\n"
            f"    quatf physics:principalAxes = {_f(_quat_wxyz(axes))}\n\n"
            + "\n".join(parts) + "}\n")


if __name__ == "__main__":
    text = frame_usda()
    if "--check" in sys.argv:
        sys.exit(0 if OUT.read_text() == text else 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT}")
