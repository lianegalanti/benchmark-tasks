# Put up a folding produce crate's four walls until every corner catch clicks

A green polypropylene folding produce crate lies folded flat between the arms. It is a standard catalogue size:
400 x 300 x 174 mm when up (internal 374 x 274 x 166 mm), 0.88 kg. The robot is told only:

> Set up the folded crate so it is ready to fill, then let go of it.

The crate's walls are rigid hinged plates held up by sprung catches that must click, and one hand has to brace the
crate while the other pulls each end wall through its catches. Reusable folding plastic crates carry much of the
fresh produce from grower to store. They are put up where they are filled: by hand packers and packagers (about 0.6
million US workers, BLS) in packing houses and distribution centres, and by order pickers who pick produce into
them. The walls go up in a fixed order that the hardware enforces:

1. The two long walls lie on top, so they come up first. Pressed onto its stop, a long wall stands by itself (its
   weight is just outboard of its hinge); let go short of that, it falls back in.
2. Each short wall is then swung up and pulled outward through its two corner catches. A catch is a sprung pawl in
   the long wall: the hook on the short wall's edge rides over its 45 degree ramp and pushes it in, and once the
   wall is past, the pawl springs out behind it with a click. The pawls push back (0.6 N/mm each), so the pull
   drags a 0.88 kg crate across the table unless the other hand holds it. Released before both catches click, the short
   wall falls back.

The physical difficulty is bimanual: one hand braces the crate while the other swings a wall up and through its
catches, then the hands swap. The robot must also find the handholds, pinch a rail and follow each wall's swing.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the crate, the per-step monitor, `success()` and `score()` |
| `geometry.py` | The crate's dimensions, wall and catch geometry, and the latch test (NumPy only) |
| `cratekit.py` | Builds the crate as one articulation at spawn time and hooks yam_arena's scene, step and reset |
| `demo.py` | The scripted full-score demo |
| `test_task.py` | Offline tests (no Isaac Sim) |

`cratekit.py` is adapted from a `partkit.py` helper used by other tasks in this repository. It hooks
`yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and `task_score` scorer.

## The crate in simulation

Crate frame: origin at the middle of the base's underside, x across (300 mm), y along (400 mm), z up. On the station
the crate is centred at (0.26, 0.00), its length along y, so its short end walls face the arms.

- **Base:** a 6 mm floor, side rims 68 mm tall along the long sides, end rims 50 mm tall.
- **Walls:** 10 mm plates. Each is hinged at the inner top edge of its rim and folds inward, with stops at flat
  and at upright.
  - The short walls (260 x 124 mm plates) hinge on the end rims and fold first, lying lowest. Each has a hook at
    each side edge: an orange 8 mm round upright lug, 38 mm tall at the catches' height, flush with the wall's inner
    face and reaching to 1.5 mm from the long wall.
  - The long walls (398 x 106 mm) hinge 18 mm higher and lie on top, 8 mm above the short walls. Folded, the crate is
    7.8 cm tall.
  - Each wall has a 120 x 36 mm handhold slot under a round 18 mm top rail.
- **Catches:** each long wall has an orange pawl near each end.
  - The pawl is 11 mm wide and 20 mm tall, centred 145 mm above the crate's underside.
  - At rest it stands 6 mm out of the wall's inner face. Its square face is 1 mm inside where the short wall's inner
    face stands when up, and its inboard side is a 45 degree ramp.
  - It is a prismatic joint along the wall's normal with a 0.6 N/mm spring and no preload, and 7 mm of travel.
  - An upright short wall's hook overlaps it by 4.5 mm; the plate itself stays clear of it.
- **Simplified:**
  - Real crates of this kind usually have handholds only in the short walls; here every wall has one, so the long
    walls can be lifted by a rail too.
  - The catch spring (0.6 N/mm, about 2.7 N per pawl when the hook has pushed it fully in) is an estimate, not
    measured on a real crate; real catches may well be stiffer.
  - The plates are solid boxes with the real outer size; a real crate's walls are ribbed lattices.
- **Physics:** the walls hinge freely with light damping (0.02 N m s/rad). The articulation's self-collisions are on,
  so the walls, the catches and the folded walls collide with each other. Polypropylene friction is 0.35 (combined as
  the minimum), against the table and the grippers.

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the folded crate so it is ready to fill, then let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/crate-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png, result.json
```

The demo is open loop: it works only from the crate's fixed start layout, on a fixed schedule, and reads no
simulator state. (It prints the crate's hinge angles and catch positions every 2 s, for the log only.)

To grip a wall, the gripper pinches its top rail, one finger through the handhold slot and the other past the free
edge. The fingers are commanded 14 mm apart on the 18 mm rail, a light pinch. The round rail turns between the pads as
the wall swings, so the gripper keeps a fixed tilt and only follows the rail's arc.

1. The right arm swings the near long wall up 5 degrees past its stop, lets go and moves clear. The wall stands.
2. The left arm does the same with the far long wall, its grip tilted forward.
3. The right arm pinches the folded right short wall's rail and holds it, which braces the crate. The left arm swings
   the left short wall up, then pulls on past upright for a fixed 3 s (8 degrees a second of commanded swing; the
   wall itself stops at upright), long enough for both its catches to click out.
4. The left arm keeps hold of its latched wall, now the brace, while the right arm clicks the right short wall in
   the same way.
5. Both grippers open, rise 4.5 cm and move clear beside the crate.

Measured in Isaac Sim on an L40S: the open-loop demo reaches 1.0 (success) in 1,074 steps (54 s). Each short wall
stopped about 8 degrees short of upright on its catches' ramps, pushed both pawls 3 to 4 mm in, and clicked through
about 1.5 s into the 3 s pull. IK is within 0.1 mm and 0.02 degrees of every target. Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step and latches violations. The score at the end of the episode counts
(`SCORE_OVER = "end"`).

- **0 if at any time** the crate falls off the table (its base more than 10 cm below the table top). Lifting or
  tilting the crate while setting it up is allowed; success still needs the base flat and still at the end.
- **A corner is latched** when both of these hold:
  - its catch's pawl is out, within 1 mm of its rest position;
  - both walls that meet there are within 1 degree of upright.

  Within 1 degree, the short wall's hook at the pawl is within 1.8 mm of upright. The pawl, 1 mm inside that
  and 6 mm out, can only be out if the wall has gone past it. By eye: the four orange pawls stand out inside the short
  walls, and the crate stays up when nudged.
- **Success (1.0, held for 1 s)** needs all of these:
  - all four corners latched;
  - every wall within 3 degrees of vertical;
  - the base within 3 degrees of flat;
  - every part of both grippers at least 2 cm from the crate (each gripper is taken as a 5 cm-radius capsule from 10
    cm behind its grasp point to its fingertips, which contains the open fingers);
  - the crate still over the last second: its base moved under 2 mm, no hinge turned 0.5 degree and no pawl moved
    0.5 mm.
- **Otherwise,** at the end:
  - 0.1 for each long wall standing up on its own: within 3 degrees of vertical, with no part of either gripper
    within 2 cm of it;
  - plus 0.15 for each latched corner.

  So at most 0.8. A long wall held up by the robot, or let go short of its stop (it falls back in), scores nothing.
  Short walls stood up but not latched score nothing; a short wall held up by the robot with its pawls pressed in is
  not latched. Holding the crate after latching all four corners scores 0.8, not 1.0.
- **Doing nothing, or refusing,** scores 0.

## Real-world trials

Use a 400 x 300 x 174 mm folding produce crate of this kind (long walls up first, short walls locking into catches
on the long walls), on a bare table. The crate is light plastic with no sharp edges or stored energy beyond the
catches' small springs, so trials are safe. Keep hands out of the arms' reach while they move. Score by eye and with
a protractor or square: count the four pawls standing out behind the short walls, check each wall against a square,
and watch that the crate stays still for a second after the grippers leave it. A crate that falls off the table is
caught on the floor and scores 0.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, plus about 1 minute for Isaac Sim to start.
The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Tests

```bash
python -m pytest -q test_task.py
```

The tests check the catalogue size and the folded stack (no wall touches another; the free edges leave gaps for the
fingers). They check the catch geometry (overlap, travel, the 1 mm gap, the hooks) and every scoring outcome,
including that a violation stays latched after the crate is put back. They also check that the demo is reachable (IK
error under 0.5 mm and 0.1 degree, joints within their limits), open loop, fits the run limit, and ends with both
grippers clear.
