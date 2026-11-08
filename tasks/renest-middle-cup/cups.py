"""The three nesting cups and where they stand on the station table. Pure Python, so the task file, the asset
script and the tests can import it before Isaac Sim starts.

Each cup is a square, open-topped box with 4 mm walls and a 4 mm floor. Nested, every cup has 1.5 cm of clearance
to the next larger cup's walls on each side. The smaller cups are taller, so each nested cup stands at least 3.4 cm
above the rim of the cup around it: a gripper holds it there, with its fingertips clear of the outer rim.
"""

from __future__ import annotations

from dataclasses import dataclass

TOP = 0.7475               # yam_arena.task_file.TABLE_TOP["station"]
WALL = 0.004               # wall and floor thickness, m
GAP = 0.015                # wall-to-wall clearance between nested cups, per side
COLOR = (0.18, 0.62, 0.52)  # all three are the same teal, like one set


@dataclass(frozen=True)
class Cup:
    outer: float   # outside width (square), m
    height: float  # overall height, m
    mass: float    # kg

    @property
    def inner(self) -> float:
        return self.outer - 2 * WALL


SMALL = Cup(0.050, 0.095, 0.040)
MEDIUM = Cup(SMALL.outer + 2 * GAP + 2 * WALL, 0.065, 0.060)   # 8.8 cm
LARGE = Cup(MEDIUM.outer + 2 * GAP + 2 * WALL, 0.032, 0.080)   # 12.6 cm
CUPS = {"small": SMALL, "medium": MEDIUM, "large": LARGE}
NEXT_LARGER = {"small": "medium", "medium": "large"}

# -- the layout (station frame: x forward, y to the robot's left; z of a cup is its underside) -----------------------
LARGE_XY = (0.22, 0.04)     # the nested set: the small cup stands inside the large one
MEDIUM_XY = (0.24, -0.20)   # the loose cup, in front of the right arm
START = {
    "large": (*LARGE_XY, TOP),
    "small": (*LARGE_XY, TOP + WALL),
    "medium": (*MEDIUM_XY, TOP),
}


def rim_when_nested(key: str) -> float:
    """Height above the table of a cup's rim in the full nest (large on the table, medium in it, small in that)."""
    floors = {"large": 0, "medium": 1, "small": 2}[key]
    return floors * WALL + CUPS[key].height
