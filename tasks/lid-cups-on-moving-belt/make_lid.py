"""Write assets/lid.usda: the lid as one rigid body of six boxes (see geometry.py). Plain text, no USD library needed.

    python make_lid.py
"""

from __future__ import annotations

from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
COLOR = (0.93, 0.93, 0.9)  # white polypropylene
TAB_COLOR = (0.15, 0.35, 0.8)  # the blue grip fin


def usda() -> str:
    parts = []
    for name, centre, size in g.lid_parts():
        color = TAB_COLOR if name == "tab" else COLOR
        parts.append(f'''
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
        double size = 1
        double3 xformOp:translate = ({centre[0]:.6f}, {centre[1]:.6f}, {centre[2]:.6f})
        float3 xformOp:scale = ({size[0]:.6f}, {size[1]:.6f}, {size[2]:.6f})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
''')
    return f'''#usda 1.0
(
    defaultPrim = "Lid"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Lid" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {g.LID_MASS}
{"".join(parts)}}}
'''


if __name__ == "__main__":
    (HERE / "assets").mkdir(exist_ok=True)
    (HERE / "assets" / "lid.usda").write_text(usda())
    print("wrote assets/lid.usda")
