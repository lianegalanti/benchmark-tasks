"""Write assets/tray.usda: a tray as one rigid body of six boxes (see geometry.py). Plain text, no USD library needed.
Adapted from make_lid.py in this repository's lid-cups-on-moving-belt task.

    python make_tray.py
"""

from __future__ import annotations

from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
COLOR = (0.15, 0.32, 0.8)  # blue polypropylene
RIB_COLOR = (0.95, 0.85, 0.1)  # the yellow handle rib


def usda() -> str:
    parts = []
    for name, centre, size in g.tray_parts():
        color = RIB_COLOR if name == "rib" else COLOR
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
    defaultPrim = "Tray"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Tray" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{{
    float physics:mass = {g.TRAY_MASS}
{"".join(parts)}}}
'''


if __name__ == "__main__":
    (HERE / "assets").mkdir(exist_ok=True)
    (HERE / "assets" / "tray.usda").write_text(usda())
    print("wrote assets/tray.usda")
