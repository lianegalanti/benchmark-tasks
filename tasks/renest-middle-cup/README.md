# Put a cup back into the middle of its nested set

Three square, open-topped cups make a nesting set. The large cup (12.6 cm wide, 3.2 cm tall) stands on the table
with the small cup (5 cm wide, 9.5 cm tall) inside it; the medium cup (8.8 cm wide, 6.5 cm tall) stands on its own.
The robot is asked to put the loose cup away so that all three are nested. The medium cup belongs between the other
two, so the robot has to:

1. take the small cup out of the large one (lifting it clear of the rim);
2. put the medium cup into the large one, through 1.5 cm of clearance on each side;
3. put the small cup back into the medium one.

Nesting the small cup in the medium cup on the table, then the pair in the large cup, works too. Lowering the medium
cup onto the set leaves it resting on the small cup's rim (or, upside down, over it), which is not nested.

Unlike stacking boxes or plates largest first, the item to put away belongs in the middle of a set that is already
assembled: the robot must see that the small cup is in the way and that lifting a container moves what is in it.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the cups, the gripper-clearance monitor, `success()`, `score()` and the scripted demo |
| `cups.py` | Cup dimensions and the starting layout (pure Python) |
| `make_cups.py` | Writes `assets/cup_*.usda`: each cup one rigid body of five box colliders, origin at the centre of its underside |
| `scenekit.py` | USD bodies for yam_arena task files and per-step/reset hooks, copied unchanged from the `lock-before-loading` task folder in this repository (its license applies) |
| `test_task.py` | Offline checks (no Isaac Sim): geometry, scoring at every level, demo reach, timing and finger clearances |

`scenekit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer.

**Measured on an L40S:** the scripted demo ended with success after 352 steps (17.6 s); Dream-1 scored 0 in a 2,400-step trial.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the loose cup away in the nested set, so that all three cups are nested inside one another." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/renest-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the loose cup away in the nested set, so that all three cups are nested inside one another." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/renest-demo

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end: 1.0 when `success()` has held for 1 s, otherwise the best `score()` reached
during the trial (`SCORE_OVER = "best"`).

## Scripted demo

Open loop, about 18 s (360 steps), both arms with the gripper pointing straight down:

1. **Left arm** grips the small cup across its top 1.9 cm (fingertips 4 cm above the large cup's rim), lifts it
   until its underside is 1.7 cm above where the medium cup's rim will be, and holds it aside at (0.20, 0.22).
2. **Right arm** pinches the medium cup's right-hand wall (one finger inside the cup, 2 cm from its far wall), lifts
   it, waits until the small cup is out (6.8 s), carries it over the large cup, lowers it and lets go 4 mm above
   its seat, then parks.
3. **Left arm** lowers the small cup into the medium one, lets go 4 mm above its seat, and parks.

The medium cup is 8.8 cm wide, too wide to grip across with the 9.4 cm opening, so it is held by one wall. Every
release leaves the fingertips at least 1.5 cm above the rim of the cup the held cup goes into.

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | medium cup in the large cup and small cup in the medium cup, each with its underside within 1 cm of the floor of the cup around it and its centre within 1.8 cm of that cup's centre along each side; every cup within 10 degrees of upright; the large cup on the table; both grippers 2 cm outside the stack's outline; all still for 1 s | photo from above and the side; each rim's height with a tape measure (nested rims at 3.2, 6.9 and 10.3 cm) |
| 0.5 | at some point, the medium cup stood in the large cup or the small cup in the medium cup, by the same test | video |
| 0.25 | at some point, the small cup's underside was at least 1 cm above the large cup's rim | video, against a mark 4.2 cm up |
| 0 | anything else: nothing done, the small cup put back in the large one, the medium cup resting on the small cup's rim or turned over it | |

The 0.5 level needs the medium cup, the one the instruction is about, to be correctly nested with one of the others;
the small cup in the large cup is the starting state and earns nothing.

How the simulation judges these, once per control step: cup poses come from the rigid bodies (origin at the centre
of each cup's underside). The grippers are each judged by two points, the tool centre point between the pads and the
pad tips 13 mm beyond it, computed from the measured joint angles.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`), so there are no seeds. A 1,200-step Dream-1 trial should take about two
minutes plus Isaac Sim's start-up .

## What the first sim run should check

- The medium cup held by one 4 mm wall stays close to level while carried (it hangs 4.2 cm off the grip).
- The cups settle without jitter: 4 mm box walls, and the small cup standing in the large one at the start.
- The released cups drop the last 4 mm into their seats without bouncing out of the 1.5 cm clearance.
