"""Writes assets/probe.usda: the test probe as one rigid body, a square handle with a thin square tip under it.

The body's origin is the handle's centre; the tip hangs below it along -z. The body carries PhysX contact
reporting, so task.py's contact sensor can read the probe's contacts with the pads. Plain text (no USD library
needed). The sizes come from task.py, so run this again after changing them:

    python make_probe.py
"""

from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HANDLE_COLOR = (0.1, 0.1, 0.12)  # black plastic
TIP_COLOR = (0.78, 0.6, 0.2)  # brass


def cube(name: str, size, centre, color) -> str:
    sx, sy, sz = (float(v) for v in size)
    cx, cy, cz = (float(v) for v in centre)
    return f'''
    def Cube "{name}" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {{
        color3f[] primvars:displayColor = [({color[0]}, {color[1]}, {color[2]})]
        double size = 1
        double3 xformOp:translate = ({cx}, {cy}, {cz})
        float3 xformOp:scale = ({sx}, {sy}, {sz})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
'''


def probe_usda(handle, tip, mass: float) -> str:
    tip_z = -(handle[2] / 2 + tip[2] / 2)
    return f'''#usda 1.0
(
    defaultPrim = "Probe"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Probe" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxContactReportAPI"]
)
{{
    float physics:mass = {mass}
    float physxContactReport:threshold = 0
{cube("handle", handle, (0.0, 0.0, 0.0), HANDLE_COLOR)}{cube("tip", tip, (0.0, 0.0, tip_z), TIP_COLOR)}}}
'''


def main() -> None:
    import task

    out = HERE / "assets" / "probe.usda"
    out.write_text(probe_usda(task.HANDLE, task.TIP, task.PROBE_MASS))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
