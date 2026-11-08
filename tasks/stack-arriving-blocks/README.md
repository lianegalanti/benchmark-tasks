# Stack blocks into a tower as they arrive on a moving conveyor

A conveyor line runs across the table, 38 cm in front of the arms, from the right toward the left arm. Three grey
carrier plates ride on it, 28 cm apart, each with a painted wooden cube (5 cm): red, yellow, blue. The line starts
with the trial and never stops for the robot: at 4 cm/s a new block comes past the left arm every 7 s, and a red end
line, taped across the table 48 cm left of the midpoint, marks where it's too late. The robot must build a tower of
the three blocks on a fixed green pad (8 x 8 x 2 cm) between the line and the left arm, each block squarely on the one
before, taking each block off the line before it passes the end line. A block that passes the line, or touches the
table, is lost and never counts.

Stacking static blocks is common in benchmarks; here the blocks arrive on their own schedule, so the robot has to
take a moving block, set it down precisely on a growing tower, and be back at the line before the next one passes.
At the end of packing lines, palletizers (people or robots) stack cartons coming off a conveyor that doesn't wait:
a carton missed runs off the end, and a stack set crooked falls.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the line's motion, the lost-block monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files (a box whose pose and velocity follow a scripted path, set before every physics step; here the conveyor's carriers, `belt_run`), and `on_step` / `on_reset` hooks with `robot_links()` / `arm_joints()` for a per-step monitor |
| `test_task.py` | Offline checks (no Isaac Sim): the line's motion, the lost-block rule, the tower scoring, and that the demo is reachable, on time and clear of the blocks |

`motionkit.py` is copied from this repository's `reach-between-sweeps` task, with its path helpers replaced by
`belt_run`. It hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without modifying
yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer.
The line's clock is the episode clock (`observation.state_time`): the carriers stand still through the reset and start
with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stack the three blocks into a tower on the green pad as they come along the conveyor, each one squarely on the one before. Take each block before it passes the red end line; a block that passes the line or touches the table is lost." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/stack-arriving-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stack the three blocks into a tower on the green pad as they come along the conveyor, each one squarely on the one before. Take each block before it passes the red end line; a block that passes the line or touches the table is lost." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/stack-arriving-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/stack-arriving-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
`score()` at the end of the trial (`SCORE_OVER = "end"`): the tower as it stands then.

## Scoring

A block is **lost** for the rest of the trial once its centre passes the end line, or its underside comes within 1 cm
of the table top (it touched the table: dropped, knocked off the line, or set down there). Lost blocks never count,
even if they are later put on the tower. A block **stands on the pad** if its centre is over the pad and its
underside is within 1 cm of the pad's top; it **stands on a block** if its centre is over that block's top face and
its underside is within 1 cm of that block's top; either way tilted no more than 10 degrees.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | all three blocks stand in one tower on the pad, each centre within 1.5 cm (horizontally) of the centre of the block below, all still for 1 s, with both grippers' grasp points 8 cm or more from every block's centre | a photo from the side and one from above with a ruler; video for the lost rule |
| 0.75 | at the end, three blocks stand in a tower on the pad (not aligned within 1.5 cm, still held, or still moving) | photo |
| 0.5 | at the end, two blocks stand in a tower on the pad | photo |
| 0.25 | at the end, one block stands on the pad | photo |
| 0 | anything else, including every block lost | |

The monitor (`update()`, run after every control step through `motionkit.on_step`) tracks lost blocks and both
arms' grasp points; `tower()` finds the tallest stack on the pad among the blocks not lost, in any order.

## Runtime and seeds

- The scripted demo uses the left arm. For each block it is over the line where the block will be at its meeting
  time (2.75 s, 9.75 s, 16.75 s, at 22 cm left of the midpoint), follows the carrier down with one waypoint per
  control step, closes, lifts the block 4 cm off the carrier still following it, rises, goes across to the pad,
  lowers the block, lets go 3 mm above its seat and goes back up. The fingers close across the line (along x), so a
  lag behind the moving carrier only shifts the grip along the block's face. Each block is off its carrier about 20 cm
  before the end line; the third is let go at about 21.5 s and the plan ends at 25.0 s.
- Measured on an L40S: the scripted demo ended with success after 469 steps (23.5 s); Dream-1 scored 0 in a
  1,200-step trial.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The line always starts from rest
  with the trial.

## Fidelity

The carriers are driven exactly along their path, like a stiff servo-driven pallet conveyor: the arms can't push them
or stop them (a real line's drive would fault if blocked). They float 2 mm above the track bed. Past the table's
left edge they simply move straight down out of reach, where a real line would carry the blocks on off its end into
a bin. The blocks are rigid cubes with sharp edges and the station's rubber-like friction (1.2 static, 1.0 dynamic),
more than painted wood on aluminium; the line's acceleration (under 0.15 m/s²) is far too low to slide them.
