"""Writes the task's USD assets from geometry.py:

- assets/printer.usda: the printer, a free-standing (not fixed) USD articulation: the body (case, bay, guide rails,
  seat notches, transfer roller) as the root link, and the front door on a revolute joint ``hinge`` about the bay
  opening's bottom front edge. 0 is shut, positive opens the door toward the robot, with stops at 0 and 80 degrees.
  The door and the body don't collide with each other (the task turns self-collisions off); the hinge's stop at 0
  stands in for the door's seat on the case.
- assets/cartridge_old.usda, assets/cartridge_new.usda: the two toner cartridges, one rigid body each (body, carry
  handle loop, two keyed side bosses).

Collisions are the boxes of geometry.py. The rest is visual only (no collision): on the printer, the output tray with
a printed sheet, the control panel, the paper cassette front, side vents, the door's finger recess and the bay's dark
lining; on each cartridge, its moulded shell (rounded toner hopper, chamfered drum edge) over the body's collision
box, which is hidden, plus end caps, the green drum showing under its edge, and labels.

    python make_assets.py        # needs usd-core (pip install usd-core); the .usda files are committed
"""

from __future__ import annotations

import math
from pathlib import Path

import geometry as g

HERE = Path(__file__).resolve().parent
OUT = HERE / "assets"

CASE = (0.86, 0.86, 0.84)       # light grey ABS
DOOR = (0.70, 0.71, 0.70)
RAIL = (0.25, 0.26, 0.28)       # dark grey guides inside the bay
ROLLER = (0.05, 0.05, 0.05)
CART = (0.06, 0.06, 0.07)       # black cartridge body
CAP = (0.16, 0.16, 0.17)        # its end caps
DRUM = (0.10, 0.42, 0.22)       # the green photoconductor drum under the cartridge's back edge
LABEL = {"old": (0.62, 0.62, 0.60), "new": (0.93, 0.93, 0.92)}
HANDLE = {"old": (0.45, 0.46, 0.48), "new": (0.95, 0.45, 0.05)}   # grey handle (old), orange handle (new)
TRIM = (0.20, 0.21, 0.22)       # dark grey trim: output tray, control panel, recesses, vents, bay lining
BAY = (0.13, 0.13, 0.14)
LCD = (0.16, 0.30, 0.42)
BUTTON = (0.75, 0.76, 0.77)
LED = (0.20, 0.85, 0.30)
PAPER = (0.97, 0.97, 0.95)
INK = (0.35, 0.35, 0.37)


def _material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _physics_material(stage, path: str, static: float, dynamic: float, combine: str):
    """A physics material (friction, and how it combines with the other surface's)."""
    from pxr import Sdf, UsdPhysics, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(static)
    api.CreateDynamicFrictionAttr(dynamic)
    api.CreateRestitutionAttr(0.0)
    mat.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
    mat.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(combine)
    return mat


def _bind_physics(prim, material) -> None:
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _box(stage, path: str, centre, size, pitch: float, rgb, material) -> None:
    """A colliding box in its parent's frame, turned ``pitch`` (rad) about +y."""
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if pitch:
        cube.AddRotateYOp().Set(float(math.degrees(pitch)))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def _visual(stage, path: str, centre, size, rgb, pitch: float = 0.0, invisible: bool = False):
    """A box that is only seen (no collision), in its parent's frame."""
    from pxr import Gf, UsdGeom

    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cube.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))
    if pitch:
        cube.AddRotateYOp().Set(float(math.degrees(pitch)))
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    return cube


def _prism(stage, path: str, profile, y0: float, y1: float, rgb) -> None:
    """A visual mesh: the convex x-z ``profile`` (counter-clockwise seen from +y) extruded from y0 to y1."""
    from pxr import Gf, UsdGeom, Vt

    n = len(profile)
    pts = [Gf.Vec3f(float(x), float(y), float(z)) for y in (y0, y1) for x, z in profile]
    counts, idx = [n, n], list(range(n)) + list(range(2 * n - 1, n - 1, -1))
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [i, n + i, n + j, j]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _cylinder_y(stage, path: str, centre, radius: float, length: float, rgb) -> None:
    """A visual cylinder along y."""
    from pxr import Gf, UsdGeom

    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateAxisAttr("Y")
    cyl.CreateRadiusAttr(float(radius))
    cyl.CreateHeightAttr(float(length))
    cyl.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    cyl.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in centre]))


def shell_profile(n_arc: int = 6):
    """The cartridge's moulded cross-section (x forward-back, z up; inside the body's collision box): flat bottom
    and front, a rounded toner hopper at the top back, and a chamfer along the bottom back over the drum."""
    dx, _, dz = (v / 2 for v in g.C_SIZE)
    r = 0.040
    arc = [(dx - r + r * math.cos(a), dz - r + r * math.sin(a))
           for a in (math.pi / 2 * k / n_arc for k in range(n_arc + 1))]
    return [(-dx + 0.004, -dz), (0.02, -dz), (dx, -0.02), *arc, (-dx + 0.004, dz), (-dx, dz - 0.004),
            (-dx, -dz + 0.004)]


def _printer_details(stage, root: str) -> None:
    """What makes the case read as a desktop laser printer; visual only, printer frame."""
    d, w, h = g.P_DEPTH, g.P_HALF_W, g.P_HEIGHT
    v = f"{root}/details"
    # output tray on top, with a printed sheet in it, and the paper exit slot behind it
    _visual(stage, f"{v}/output_tray", (0.235, 0.0, h + 0.0005), (0.20, 0.30, 0.001), TRIM)
    _visual(stage, f"{v}/paper", (0.235, 0.0, h + 0.0012), (0.17, 0.21, 0.0004), PAPER)
    for i, z in enumerate((0.005, 0.012, 0.019, 0.026)):           # lines of print on it
        _visual(stage, f"{v}/print{i}", (0.18 + 4 * z, 0.0, h + 0.0015), (0.004, 0.15, 0.0002), INK)
    _visual(stage, f"{v}/exit_slot", (0.34, 0.0, h + 0.0005), (0.012, 0.32, 0.001), BAY)
    # control panel on the fixed top's front right: display, buttons, power light
    _visual(stage, f"{v}/panel", (0.132, -0.13, h + 0.0005), (0.036, 0.12, 0.001), TRIM)
    _visual(stage, f"{v}/display", (0.132, -0.155, h + 0.0012), (0.020, 0.045, 0.0006), LCD)
    for i, y in enumerate((-0.115, -0.095, -0.08)):
        _visual(stage, f"{v}/button{i}", (0.132, y, h + 0.0014), (0.010, 0.010, 0.001), BUTTON)
    _visual(stage, f"{v}/power_led", (0.132, -0.185, h + 0.0012), (0.004, 0.004, 0.0006), LED)
    # main paper cassette across the bottom of the front, with its pull recess
    _visual(stage, f"{v}/cassette", (-0.0004, 0.0, 0.029), (0.0008, 2 * w - 0.02, 0.038), (0.80, 0.80, 0.79))
    _visual(stage, f"{v}/cassette_grip", (-0.0010, 0.0, 0.032), (0.0008, 0.10, 0.012), TRIM)
    # cooling vents on both sides, toward the back
    for sy in (1, -1):
        for i in range(6):
            _visual(stage, f"{v}/vent_{'lr'[sy < 0]}{i}", (0.28, sy * (w + 0.0004), 0.10 + 0.018 * i),
                    (0.10, 0.0008, 0.005), TRIM)
    # the toner bay's dark lining: side walls, floor and the print engine's face behind the cartridge
    for sy in (1, -1):
        _visual(stage, f"{v}/bay_wall_{'lr'[sy < 0]}", (g.REAR_FROM / 2, sy * (g.WALL_IN - 0.0006), 0.133),
                (g.REAR_FROM, 0.0008, g.TOP_UNDER - g.BASE_TOP - 0.001), BAY)
    _visual(stage, f"{v}/bay_floor", (g.REAR_FROM / 2, 0.0, g.BASE_TOP + 0.0004),
            (g.REAR_FROM, 2 * g.WALL_IN - 0.002, 0.0008), BAY)
    _visual(stage, f"{v}/bay_back", (g.REAR_FROM - 0.0004, 0.0, 0.133),
            (0.0008, 2 * g.WALL_IN - 0.002, g.TOP_UNDER - g.BASE_TOP - 0.001), BAY)


def _stage(root: str):
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    prim = UsdGeom.Xform.Define(stage, f"/{root}")
    stage.SetDefaultPrim(prim.GetPrim())
    UsdGeom.Scope.Define(stage, f"/{root}/Looks")
    return stage


def _mass(prim, mass, com, size) -> None:
    from pxr import Gf, UsdPhysics

    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(float(mass))
    m.CreateCenterOfMassAttr(Gf.Vec3f(*[float(v) for v in com]))
    a, b, c = size
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(mass / 12 * (b * b + c * c), mass / 12 * (a * a + c * c),
                                         mass / 12 * (a * a + b * b)))
    m.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))


def build_printer(path: Path) -> None:
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    stage = _stage("printer")
    m_case = _material(stage, "/printer/Looks/case", CASE)
    m_rail = _material(stage, "/printer/Looks/rail", RAIL)
    m_roll = _material(stage, "/printer/Looks/roller", ROLLER)
    m_door = _material(stage, "/printer/Looks/door", DOOR)
    # physics: smooth ABS (rails, case, door; "min", so it rules contacts with the cartridges and grippers) and rubber
    # feet ("max", so they grip the table)
    p_abs = _physics_material(stage, "/printer/Looks/abs_physics", *g.ABS_FRICTION, "min")
    p_rubber = _physics_material(stage, "/printer/Looks/rubber_physics", *g.FOOT_FRICTION, "max")
    UsdGeom.Scope.Define(stage, "/printer/Geometry")

    body = UsdGeom.Xform.Define(stage, "/printer/Geometry/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdPhysics.ArticulationRootAPI.Apply(body)
    body.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(False)
    _mass(body, g.P_MASS, (g.P_DEPTH * 0.6, 0.0, 0.09), (g.P_DEPTH, 2 * g.P_HALF_W, g.P_HEIGHT))
    for name, c, s, p in g.printer_boxes():
        inside = name.startswith(("rail", "flare", "seat", "stop"))
        _box(stage, f"/printer/Geometry/body/{name}", c, s, p, RAIL if inside else CASE, m_rail if inside else m_case)
        _bind_physics(stage.GetPrimAtPath(f"/printer/Geometry/body/{name}"),
                      p_rubber if name.startswith("foot") else p_abs)
    roller = UsdGeom.Cylinder.Define(stage, "/printer/Geometry/body/transfer_roller")
    # a z-axis cylinder turned onto y by its transform (the physics engine's cylinder collider follows the prim's
    # transform, not every authoring of the axis attribute)
    roller.CreateAxisAttr("Z")
    roller.CreateRadiusAttr(float(g.ROLLER[2]))
    roller.CreateHeightAttr(float(2 * g.ROLLER[3]))
    roller.CreateDisplayColorAttr([Gf.Vec3f(*ROLLER)])
    roller.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in g.ROLLER_C]))
    roller.AddRotateXOp().Set(90.0)
    UsdPhysics.CollisionAPI.Apply(roller.GetPrim())
    from pxr import UsdShade

    UsdShade.MaterialBindingAPI.Apply(roller.GetPrim()).Bind(m_roll)
    _bind_physics(roller.GetPrim(), p_abs)

    # the door: its frame is on the hinge axis; shut, it closes the bay's front and the front of its top
    door = UsdGeom.Xform.Define(stage, "/printer/Geometry/body/door")
    door.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, float(g.HINGE_Z)))
    UsdPhysics.RigidBodyAPI.Apply(door.GetPrim())
    _mass(door.GetPrim(), g.DOOR_MASS, g.door_com(), (0.06, 2 * g.DOOR_HALF_W, g.PANEL_H))
    for name, c, s, p in g.door_boxes():
        _box(stage, f"/printer/Geometry/body/door/{name}", c, s, p, DOOR, m_door)
        _bind_physics(stage.GetPrimAtPath(f"/printer/Geometry/body/door/{name}"), p_abs)
    # the finger recess at the top of the door's front, the usual way to pull it open
    _visual(stage, "/printer/Geometry/body/door/finger_recess", (-0.0004, 0.0, g.PANEL_H - 0.022),
            (0.0008, 0.12, 0.016), TRIM)
    _printer_details(stage, "/printer/Geometry/body")

    UsdGeom.Scope.Define(stage, "/printer/Physics")
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/printer/Physics/hinge")
    hinge.CreateBody0Rel().SetTargets([Sdf.Path("/printer/Geometry/body")])
    hinge.CreateBody1Rel().SetTargets([Sdf.Path("/printer/Geometry/body/door")])
    hinge.CreateAxisAttr("Y")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, float(g.HINGE_Z)))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # the joint frame turned 180 degrees about z on both sides: its +Y axis is the station's -y, so a positive angle
    # swings the door's top toward the robot
    flip = Gf.Quatf(0.0, 0.0, 0.0, 1.0)
    hinge.CreateLocalRot0Attr(flip)
    hinge.CreateLocalRot1Attr(flip)
    hinge.CreateLowerLimitAttr(0.0)
    hinge.CreateUpperLimitAttr(float(math.degrees(g.OPEN)))
    drive = UsdPhysics.DriveAPI.Apply(hinge.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(g.DOOR_DAMPING * math.pi / 180))
    drive.CreateMaxForceAttr(100.0)
    stage.GetRootLayer().Export(str(path))


def build_cartridge(path: Path, which: str) -> None:
    from pxr import UsdPhysics

    stage = _stage("cartridge")
    m_body = _material(stage, "/cartridge/Looks/body", CART)
    m_handle = _material(stage, "/cartridge/Looks/handle", HANDLE[which])
    body = stage.GetPrimAtPath("/cartridge")          # the rigid body is the file's root prim
    UsdPhysics.RigidBodyAPI.Apply(body)
    _mass(body, g.MASS[which], (0.0, 0.0, -0.005), g.C_SIZE)
    for name, c, s, p in g.cartridge_boxes():
        handle = name == "handle" or name.startswith("post")
        _box(stage, f"/cartridge/{name}", c, s, p, HANDLE[which] if handle else CART,
             m_handle if handle else m_body)
    # the body collides as its box, but is seen as its moulded shell (inside the box): hide the box
    from pxr import UsdGeom

    UsdGeom.Imageable(stage.GetPrimAtPath("/cartridge/body")).MakeInvisible()
    half, cap = g.C_SIZE[1] / 2, 0.012
    prof = shell_profile()
    _prism(stage, "/cartridge/shell", prof, -half + cap, half - cap, CART)
    for sy in (1, -1):
        _prism(stage, f"/cartridge/end_cap_{'lr'[sy < 0]}", prof, *sorted((sy * half, sy * (half - cap))), CAP)
    _cylinder_y(stage, "/cartridge/drum", (0.036, 0.0, -0.034), 0.010, 2 * (half - cap) - 0.01, DRUM)
    _visual(stage, "/cartridge/front_label", (-g.C_SIZE[0] / 2 - 0.0004, 0.06, -0.008), (0.0008, 0.12, 0.05),
            LABEL[which])
    _visual(stage, "/cartridge/top_label", (0.0, -0.10, g.C_SIZE[2] / 2 + 0.0004), (0.016, 0.09, 0.0008),
            LABEL[which])
    stage.GetRootLayer().Export(str(path))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    build_printer(OUT / "printer.usda")
    for which in ("old", "new"):
        build_cartridge(OUT / f"cartridge_{which}.usda", which)
    print("wrote", ", ".join(str(p.name) for p in sorted(OUT.glob("*.usda"))))
