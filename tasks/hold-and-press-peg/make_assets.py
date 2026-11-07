"""Write assets/bar.usda and assets/peg.usda from geometry.py. Plain text USD, so no USD library is needed.

    python make_assets.py          # (re)write both files
    python make_assets.py --check  # exit 1 if either file differs from what geometry.py gives

The bar is one rigid body made of five boxes (floor, roof and two side walls around the hole, and the solid far
end), so the hole is a real cavity. The peg is one cylinder along y.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
ORANGE = (0.95, 0.45, 0.10)
STEEL = (0.62, 0.63, 0.66)


def _header(prim: str) -> str:
    return f'#usda 1.0\n(\n    defaultPrim = "{prim}"\n    metersPerUnit = 1\n    upAxis = "Z"\n)\n\n'


def _f(v: float) -> str:
    return f"{v:.6f}".rstrip("0").rstrip(".") if v else "0"


def _vec(v) -> str:
    return "(" + ", ".join(_f(float(x)) for x in v) + ")"


def bar_usda() -> str:
    out = [_header("Bar"), 'def Xform "Bar" (\n    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n)\n{\n']
    out.append(f"    float physics:mass = {_f(g.BAR_MASS)}\n")
    for name, (centre, size) in g.bar_boxes().items():
        out.append(
            f'\n    def Cube "{name}" (\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )\n    {{\n'
            f"        color3f[] primvars:displayColor = [{_vec(ORANGE)}]\n"
            f"        double size = 1\n"
            f"        double3 xformOp:translate = {_vec(centre)}\n"
            f"        float3 xformOp:scale = {_vec(size)}\n"
            f'        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]\n'
            f"    }}\n"
        )
    out.append("}\n")
    return "".join(out)


def peg_usda() -> str:
    return (
        _header("Peg")
        + 'def Xform "Peg" (\n    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]\n)\n{\n'
        + f"    float physics:mass = {_f(g.PEG_MASS)}\n\n"
        + '    def Cylinder "shaft" (\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )\n    {\n'
        + '        uniform token axis = "Y"\n'
        + f"        color3f[] primvars:displayColor = [{_vec(STEEL)}]\n"
        + f"        double height = {_f(g.PEG_LEN)}\n"
        + f"        double radius = {_f(g.PEG_R)}\n"
        + "    }\n}\n"
    )


FILES = {"assets/bar.usda": bar_usda, "assets/peg.usda": peg_usda}


def main(check: bool = False) -> int:
    bad = []
    for rel, make in FILES.items():
        path, text = HERE / rel, make()
        if check:
            if not path.is_file() or path.read_text() != text:
                bad.append(rel)
        else:
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
    if bad:
        print("out of date:", ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    sys.exit(main(check="--check" in sys.argv))
