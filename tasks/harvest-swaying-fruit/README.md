# Pick the ripe fruit from a plant that keeps swaying, leaving the unripe one

Two mock tomato plants stand 30 cm in front of the arms, 18 cm apart. Each is a green stem topped by a small calyx
cup, and in each cup rests a tomato (a 6 cm ball, 150 g): red (ripe) on the plant toward the right arm, green
(unripe) on the one toward the left arm. Both plants keep swaying, as in wind or on a shaken greenhouse trellis: each
cup draws a level figure of eight, 7 cm across and 4 cm deep, every 4.8 s, never stopping, a quarter of a sway apart.
The robot must pick the red tomato, following it as it sways, put it in the brown basket at the right, and leave the
green tomato in its cup.

Unlike the repository's shuttle task (grab during a pause of a 30 cm move) and its moving-tote task, nothing here
ever waits: the fruit moves continuously along a small 2-D path at 3 to 7 cm/s, wider (3.5 cm either side) than the
open gripper forgives (about 1.7 cm), so a grasp at a fixed point catches it only by chance; the robot has to track
it while descending and closing, and choose by ripeness.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: plants, tomatoes and basket, the sway, the cup monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step; `on_step`/`on_reset` hooks for per-step monitors; and `lissajous`, the never-stopping sway path |
| `test_task.py` | Offline checks (no Isaac Sim): the sway, the plants' geometry and clearances, the monitor and every scoring level, and that the demo is reachable and follows the fruit |

`motionkit.py` is copied from this repository's `../hand-to-moving-receiver/` (same authors), with `lissajous`
added. It hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena, so the
task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The sway's clock is
the episode clock (`observation.state_time`): the plants stand at their start through the reset and start with the
first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

**Measured on an L40S:** the scripted demo ended with success after 276 steps (13.8 s); Dream-1 scored 0 in a
2,400-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
INSTR="Pick the ripe red tomato and put it in the brown basket. The plants keep swaying and won't hold still for you, so follow the tomato as you pick it. Leave the green tomato on its plant."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/harvest-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/harvest-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/harvest-*/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
`score()` at the end of the trial (`SCORE_OVER = "end"`), which reads flags the cup monitor keeps over the whole trial.

## Scoring

Every control step (0.05 s) a monitor compares each tomato with its own cup, wherever the cup has swayed to. A tomato
is *out of its cup* when its centre is more than 1.5 cm sideways from the cup's centre line or more than 1.5 cm above
or below its seated height; the red tomato is *picked* once its centre is 3 cm or more above its seated height.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | the red tomato rests in the basket (centre inside the walls, underside within 1 cm of the floor, still for 1 s) and the green tomato was never out of its cup | a photo from above; video of the green plant |
| 0.5 | at the end the red tomato rests in the basket, but the green one was out of its cup at some point (taken, or knocked off) | photo and video |
| 0.25 | the red tomato was picked at some point (centre 3 cm above its seat), but doesn't end in the basket | video, against a 3 cm mark |
| 0 | anything else: the red tomato never picked (left, knocked off its cup without a lift), or only the green one taken | |

## Runtime and seeds

- The scripted demo (right arm) is over the red tomato by 2.5 s and then puts a waypoint on its sway every control
  step, aimed one step ahead: it descends around the tomato (3.0 to 4.4 s), closes (4.6 to 5.2 s) and lifts it
  6.7 cm (5.4 to 6.4 s), all while following, then carries it to the basket, lowers it to 1 cm above the floor, lets
  go at about 10 s and parks by 11 s. In Isaac Sim it ended with success after 276 steps (13.8 s).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The sway always starts from rest at
  the same points.

## Fidelity

The plants are rigid and translate along their paths exactly, like a stiff drive under the table: the arms can't push
them or stop them, and the stems don't bend or tilt (a real stem swings about its base). Each plant is six boxes
(stem, cup floor, four walls) 1 mm apart, moving as one. The tomatoes are rigid balls resting loose on the four top
edges of their cup's square opening, not attached by a stalk, with the station's rubber-like friction (1.2 static,
1.0 dynamic); real tomatoes are softer and must be snapped off. The sway's acceleration stays under 0.15 m/s², far
below the 5.7 m/s² a seated tomato would need to climb out of its cup, so it rides along until touched.
