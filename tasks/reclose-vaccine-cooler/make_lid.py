"""Writes assets/lid.usda, the cooler's lid as one rigid body of box colliders (plate, four skirt strips, handle).

    python make_lid.py

Plain text, no USD library needed; test_task.py checks the file is up to date and, if usd-core is installed, that it
parses with the expected bounds. Origin: the centre of the plate's underside; z up.
"""

from __future__ import annotations

from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
LID_COLOR = (0.30, 0.52, 0.78)      # sRGB, light blue moulded plastic, like the cooler
HANDLE_COLOR = (0.20, 0.22, 0.25)


def boxes() -> list[tuple[str, tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    """(name, centre, size, colour) of each collider, in the lid frame."""
    hx, hy = g.LID_HALF
    t, h = g.SKIRT_T, g.SKIRT_H
    return [
        ("plate", (0.0, 0.0, g.PLATE_T / 2), (2 * hx, 2 * hy, g.PLATE_T), LID_COLOR),
        ("skirt_front", (hx - t / 2, 0.0, -h / 2), (t, 2 * hy, h), LID_COLOR),
        ("skirt_back", (-hx + t / 2, 0.0, -h / 2), (t, 2 * hy, h), LID_COLOR),
        ("skirt_left", (0.0, hy - t / 2, -h / 2), (2 * hx - 2 * t, t, h), LID_COLOR),
        ("skirt_right", (0.0, -hy + t / 2, -h / 2), (2 * hx - 2 * t, t, h), LID_COLOR),
        ("handle", (0.0, 0.0, g.PLATE_T + g.HANDLE[2] / 2), g.HANDLE, HANDLE_COLOR),
    ]


def _f(v) -> str:
    return "(" + ", ".join(repr(round(float(x), 6)) for x in v) + ")"


def usda() -> str:
    out = ['#usda 1.0', '(', '    defaultPrim = "Lid"', '    metersPerUnit = 1', '    upAxis = "Z"', ')', '',
           'def Xform "Lid" (', '    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]', ')', '{',
           f'    float physics:mass = {g.LID_MASS}']
    for name, centre, size, color in boxes():
        out += ['', f'    def Cube "{name}" (', '        prepend apiSchemas = ["PhysicsCollisionAPI"]', '    )', '    {',
                f'        color3f[] primvars:displayColor = [{_f(color)}]', '        double size = 1',
                f'        double3 xformOp:translate = {_f(centre)}', f'        float3 xformOp:scale = {_f(size)}',
                '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]', '    }']
    out += ['}', '']
    return "\n".join(out)


if __name__ == "__main__":
    (HERE / "assets").mkdir(exist_ok=True)
    (HERE / "assets" / "lid.usda").write_text(usda())
    print("wrote", HERE / "assets" / "lid.usda")
