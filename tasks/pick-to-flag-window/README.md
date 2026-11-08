# Pick from each bin only while its signal flag is up

Four shallow grey bins (15 x 15 cm inside, 1.5 cm walls) stand in a row 33 cm in front of the arms, two in front of
each arm, each holding one orange wooden cube (4 cm). Behind the row a black screen (10 cm tall) hides four green
flags, one behind each bin. Like the lights on a pick-to-light rack, each flag pops up once, in a fixed order (bin 3 at
2 s, bin 1 at 4 s, bin 4 at 10 s, bin 2 at 12 s; bins numbered from the robot's left), stays up 4 s and drops again.
The robot must take each cube while its own bin's flag is up and drop it in a grey tote nearer the arms. Taking from
a bin whose flag is down (too early, too late, or the wrong bin) is a wrong pick and costs points.

A policy has to watch the flags and react: from the moment a flag counts as up it has about 4.2 s to get to the bin,
grasp and lift the cube out, and two windows (one per arm) overlap for about 2 s each time, so a slow policy, or one
that runs a fixed routine without looking, misses picks or makes wrong ones.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the flags' schedule, the pick monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files (a box whose pose and velocity follow a scripted path, set before every physics step), the `pop_up` path, the episode `clock()`, and `on_step` / `on_reset` hooks for a monitor |
| `test_task.py` | Offline checks (no Isaac Sim): the flags' timing, the monitor and the scoring of every outcome, the layout's clearances, and that the demo is reachable and lifts each cube out while its flag is up |

`motionkit.py` is copied from the reach-between-sweeps task in this repository, with a `pop_up`
path and `clock()` added and the arm-link helpers removed. It hooks `yam_arena.scene` and the embodiment's reset,
simulation step and task scoring without modifying yam_arena, so the task runs with the standard `yam_arena`
embodiment, `yam_scripted` policy and `task_score` scorer. The flags' clock is the episode clock
(`observation.state_time`): it stands still through the reset and starts with the first control step. Written against
yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured on an L40S:** the scripted demo ended with success after 384 steps (19.2 s); Dream-1 scored 0 in an
800-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
INSTR="Each bin has a green flag behind it that pops up for a few seconds when its part is wanted. Take the part from a bin only while that bin's flag is up, and drop it in the grey tote. Leave a bin alone while its flag is down."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/flags-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/flags-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/flags-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
`score()` at the end of the trial (`SCORE_OVER = "end"`). How each cube left its bin is latched by the monitor the
first control step it is out, so later events can't change it; whether it is in the tote counts at the end.

## Scoring

A cube has **left its bin** the first control step its underside is 3 cm or more above the bin's floor, or its centre
is outside the bin's inside walls. Its flag is **up** while the flag's top is 5 cm or more above the screen's top
(of the 7.2 cm it rises above it): about 4.2 s per flag. Per bin:

| Points | Condition | Check on a real table |
|---|---|---|
| 1 | the cube was lifted out (underside 3 cm above the floor) while its bin's flag was up, and ends lying in the tote: centre inside the tote's walls, underside within 1 cm of its floor, still | video against a 3 cm mark on the bin and a 5 cm mark above the screen; a photo of the tote |
| 0.5 | lifted out while its bin's flag was up, but it ends anywhere else (dropped, left in the gripper) | video |
| -0.5 | a wrong pick: it left the bin in any way (lifted, pushed, knocked) while its bin's flag was down, before or after its window, even during another bin's window | video |
| 0 | it never left the bin, or it was pushed out over the wall without being lifted while its flag was up | video |

The score is the sum over the four bins divided by 4, clipped to 0..1 (steps of 0.125). Touching or nudging a cube
inside its bin is not a pick. Grabbing every cube without watching the flags scores 0; dropping every cube on the
table after picking it in its window scores 0.5.

## Runtime and seeds

- The scripted demo moves each arm to a waiting point above its two bins and starts each pick 0.3 s after that bin's
  flag counts as up, never earlier, so it uses nothing a camera couldn't show. It lifts each cube out 2.35 s after
  the flag counts as up (1.86 s before it stops counting). It then sets the cube down on its own spot in the tote
  (12 x 43 cm inside): the four spots lie along the tote's centre line 9.5 cm apart, 4.75 and 14.25 cm either side
  of the midpoint, each arm using its own half, with the two cubes the arms carry at the same time going to spots
  19 cm apart. It lowers the cube slowly (1.2 s), lets go with its underside 1 cm above the floor, opens fully and
  only then rises. The last cube is let go at 18.3 s; in Isaac Sim the trial ended with success after 384 steps
  (19.2 s).
- `PICK_DEBUG=1 inspect-robots run ...` prints each bin's outcome (`taken`, `wrong`, `pushed_out` or `None`) and
  every cube's position (cm: x, y, underside above the table) and whether it counts as in the tote, whenever an
  outcome changes and once a second.
- `test_task.py` checks the lift-out times on the commanded TCP, not the simulated one: tracking lag in the simulator
  eats into the 1.86 s margin.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The flags always start down and run
  the same schedule.

## Fidelity

The flags are driven exactly along their path, like stiff linear actuators: the arms can't push or stop them, and
they stand behind the screen, out of the way. A flag's guide or actuator is not modelled: it floats 1.2 cm above
the table. At the grasp the open fingers are 1.8 cm inside the bin's walls and their tips 1.5 cm
above the walls' tops. The cubes are plain boxes with the station's rubber-like friction (1.2 static, 1.0 dynamic), more than
painted wood on plastic. The monitor judges each pick at the 20 Hz control rate from the flags' measured heights; on
a real table the same rule is read from video against marks on the bins and the screen.
