"""The YAM fingertips' collision points (from the arm's own collision meshes), for offline clearance checks in the
tests: each tip's convex pieces' vertices in the TCP frame, for a given finger opening. Pure Python (NumPy)."""

from __future__ import annotations

import glob
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def _read_stl(path: str) -> np.ndarray:
    b = Path(path).read_bytes()
    if b[:5] == b"solid" and b"facet" in b[:400]:
        return np.array([[float(v) for v in ln.split()[1:]] for ln in b.decode().splitlines()
                         if ln.strip().startswith("vertex")])
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3).astype(float)


def _rpy(r, p, y) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def tip_points(grip: float = 1.0) -> dict[str, np.ndarray]:
    """{"tip_left": points, "tip_right": points}: collision-mesh vertices in the TCP frame, the gripper ``grip``
    open (1 open .. 0 closed, as the demo commands it)."""
    import yam_arena
    from yam_arena.constants import FINGER_OPEN, TCP_OFFSET

    root = Path(yam_arena.__file__).resolve().parent / "assets" / "yam"
    urdf = ET.parse(root / "yam.urdf").getroot()
    q = grip * FINGER_OPEN
    out = {}
    for j in urdf.findall("joint"):
        if j.get("name") not in ("joint7", "joint8"):
            continue
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()])
        rot = _rpy(*[float(v) for v in o.get("rpy").split()])
        axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
        tip = j.find("child").get("link")
        link = next(lk for lk in urdf.findall("link") if lk.get("name") == tip)
        vo = np.array([float(v) for v in link.find("visual/origin").get("xyz").split()])
        pts = np.vstack([_read_stl(f) for f in sorted(glob.glob(str(root / "collision" / tip / "*.stl")))])
        out[tip] = (rot @ (pts + vo).T).T + xyz + rot @ (axis * q) - np.asarray(TCP_OFFSET)
    return out


def to_world(points: np.ndarray, tcp, rot) -> np.ndarray:
    return np.asarray(tcp, float) + (np.asarray(rot, float) @ np.asarray(points, float).T).T


def point_seg_dist(pts, a, b) -> np.ndarray:
    pts, a, b = np.asarray(pts, float), np.asarray(a, float), np.asarray(b, float)
    d = b - a
    t = np.clip(((pts - a) @ d) / max(float(d @ d), 1e-18), 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * d), axis=1)
