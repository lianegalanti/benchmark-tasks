# Sort parts off a belt that stops dead, surges to double speed and has a part shoved aside mid-run

A conveyor crosses the table 34 cm in front of the arms and carries four painted cubes (4 cm) on steel carriers from
the robot's right to its left: blue, red, blue, red. At its left end, 42 cm left of the midpoint between the arms, it
runs into a tunnel too low to reach into. A red bin stands in front of the right arm, a blue bin in front of the left
arm. The robot must put each cube in the bin of its colour before it goes into the tunnel.

The difficulty is the belt, not the sorting. Its speed changes abruptly on a fixed schedule: 4 cm/s, then a dead stop
for 5.5 s (a jam downstream), then 8 cm/s for 5 s (clearing the backlog), then 4 cm/s again. During the stop a pusher
bar behind the belt shoves the first blue cube 4 cm across its carrier toward the robot. A grasp timed from the
cruising speed arrives where a cube no longer is: too late after the surge, too early into the stop, and 4 cm off
for the shoved cube, which the open fingers (9.4 cm apart around a 4 cm cube) then land on. The robot has to watch
each cube up to the moment it closes and re-aim when the line changes.

The board's other conveyor tasks sort or pick from a belt at one steady speed (sorting coloured blocks, packing
biscuit packs, sampling every fourth part from an indexing line). Here the speed and the parts' positions change
under the robot mid-run, which is what makes real lines hard to work beside.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the conveyor, carriers, cubes, pusher, tunnel and bins, the belt's schedule, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction. Copied from our unload-paused-shuttle task, with `speed_schedule()` / `belt_path()` (a belt whose speed changes on a schedule) and `stroke()` (a pusher going out and back once) in place of its shuttle path |
| `test_task.py` | Offline checks (no Isaac Sim): the belt's schedule and smoothness, the pusher (only while the belt stands, only the one cube, low enough not to tip it, far enough to spoil an unchanged grasp), what the tunnel hides, every scoring level, and that the demo is reachable, keeps the arms apart and takes every cube in time |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The belt's
clock is the episode clock (`observation.state_time`): it stands still through the reset and starts with the first
control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured on an L40S:** the scripted
demo ended with success after 515 steps (25.8 s); Dream-1 scored 0 in a 1,200-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sort the cubes riding on the belt before they go into the tunnel: red cubes in the red bin, blue cubes in the blue bin. The belt changes speed without warning." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/surging-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sort the cubes riding on the belt before they go into the tunnel: red cubes in the red bin, blue cubes in the blue bin. The belt changes speed without warning." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/surging-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/surging-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 once `success()` has held for 1 s. Otherwise it
scores `score()` at the end of the trial (`SCORE_OVER = "end"`): what lies in the bins then.

## The belt

Every change of speed takes 0.3 s (a cosine ramp, peak acceleration 0.42 m/s²).

| Time (s) | Belt | What happens |
|---|---|---|
| 0 to 9 | 4 cm/s | the four cubes come along; red_1 passes in front of the right arm and leaves its reach at about 6.7 s |
| 9 to 14.5 | stopped | the jam. 9.4 to 10.9 s: the pusher bar (8 cm left of the midpoint) shoves blue_1 4 cm toward the robot and pulls back. red_2 stands in front of the right arm |
| 14.5 to 19.5 | 8 cm/s | the surge: red_2 leaves the right arm's reach about 1.7 s after the restart; blue_1, if nobody took it, goes into the tunnel at about 18.9 s |
| 19.5 to 28.5 | 4 cm/s | the other cubes, if nobody takes them, go into the tunnel: red_1 at about 20.9 s, blue_2 at 23.7 s, red_2 at 26.4 s (cube centres at the mouth) |
| 28.5 on | stopped | every carrier is inside the tunnel, 5 cm or more under the roof |

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| +0.25 per cube | lying in the bin of its own colour: centre between the bin's inner walls and below the top of its walls, still | a photo from above, and that no cube sits above the rim |
| −0.25 per cube | lying in the other bin (same test) | the same photo |
| 0 | a cube anywhere else: on its carrier, in the tunnel, on the table or floor, in a gripper | |

The total is never below 0; all four in their own bins is 1.0 and ends the trial with success. Putting every cube in
one bin, or alternating at random, scores 0 on average, so sorting by colour is the only way to score.

## Runtime and seeds

- The scripted demo, open loop and timed against the schedule: the right arm takes red_1 at 2.5 s while the belt
  cruises (following it down, closing and lifting while it moves) and red_2 at 9.8 s while the belt stands still; the
  left arm waits high over the pusher, comes down at 11.9 s (1 s after the bar is back at rest), aims 4 cm nearer than
  the line the cube rode on, takes blue_1 and has it lifted clear at 13.5 s, 1 s before the belt restarts; it takes
  blue_2 at 19.9 s, once the belt is back at cruising speed. Each cube is lifted 4 cm off its carrier 1.6 s after the
  arm starts down, at least 4 cm before the tunnel. The last cube is let go at 24.3 s, and in Isaac Sim the trial
  ended with success after 515 steps (25.8 s).
- Dream-1 scored 0 in a 1,200-step trial.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.

## Layout notes

- The bins' far walls are 5 cm from the carriers' near edge, and their near walls 2.5 cm in front of the arm bases'
  front faces; open fingers clear the bins' walls by 1.5 cm when letting go.
- The tunnel runs 25 cm past the table's left edge on a post. It has to hold all four carriers (nose to tail 11 cm
  apart, the closest they can ride) once they have passed the mouth, and the mouth has to stay inside the left arm's
  reach; ending the line on the table would need parts to fall off the belt's end, which a scripted carrier can't
  model cleanly (carriers turning down at the end would pile up on each other).

## Fidelity

The carriers and the pusher bar follow their paths exactly, like servo-driven axes: the arms can't push or stop them
(a real line would stop on a jam). The carriers ride 2 mm above the rails, the bar 2 mm above its guide block; the
bar's drive (an air cylinder behind it on a real line) isn't modelled. The cubes have the station's rubber-like
friction (1.2 static, 1.0 dynamic), more than painted wood on steel; at a peak acceleration of 0.42 m/s² they don't
slide on the carriers. The bar pushes a cube only in its lowest 1.3 cm, well under the 2 cm or so at which a push
would tip a 4 cm cube.
