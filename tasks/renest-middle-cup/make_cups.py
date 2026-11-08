"""Write the three cups as USD rigid bodies: assets/cup_small.usda, cup_medium.usda and cup_large.usda.

Each cup is one rigid body made of five box colliders: a square floor and four walls, open at the top. The body's
origin is the centre of its underside, so its pose gives the cup's bottom directly. Plain text USDA, so no USD
library is needed:  python make_cups.py
"""

from __future__ import annotations

from pathlib import Path

import cups

HERE = Path(__file__).resolve().parent


def _cube(name: str, centre, size) -> str:
    cx, cy, cz = (float(v) for v in centre)
    sx, sy, sz = (float(v) for v in size)
    r, g, b = cups.COLOR
    return f"""
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        color3f[] primvars:displayColor = [({r}, {g}, {b})]
        double size = 1
        double3 xformOp:translate = ({cx:.5f}, {cy:.5f}, {cz:.5f})
        float3 xformOp:scale = ({sx:.5f}, {sy:.5f}, {sz:.5f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
"""


def cup_usda(key: str) -> str:
    c = cups.CUPS[key]
    o, h, t = c.outer, c.height, cups.WALL
    parts = [_cube("floor", (0, 0, t / 2), (o, o, t))]
    wall_h, wall_z = h - t, t + (h - t) / 2
    for name, (x, y), size in (("wall_near", (-(o - t) / 2, 0), (t, o, wall_h)),
                               ("wall_far", ((o - t) / 2, 0), (t, o, wall_h)),
                               ("wall_right", (0, -(o - t) / 2), (o - 2 * t, t, wall_h)),
                               ("wall_left", (0, (o - t) / 2), (o - 2 * t, t, wall_h))):
        parts.append(_cube(name, (x, y, wall_z), size))
    return f"""#usda 1.0
(
    defaultPrim = "Cup"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Cup" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {c.mass}
{"".join(parts)}}}
"""


def main() -> None:
    (HERE / "assets").mkdir(exist_ok=True)
    for key in cups.CUPS:
        (HERE / "assets" / f"cup_{key}.usda").write_text(cup_usda(key))


if __name__ == "__main__":
    main()
