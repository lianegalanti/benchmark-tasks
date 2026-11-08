# Pick parts off a slow and a fast lane in the order they will leave reach

Two conveyor lanes run side by side across the table in front of the left arm, from the robot's right to its left:
the near lane at 10 cm/s, the far lane at 2 cm/s. Grey sheet-steel covers hide both lanes except a 34 cm window in
front of the left arm; parts (bare aluminium 4 cm cubes, each on its own steel carrier) come out from under the feed
cover on the right and ride under the run-out cover on the left, past which they are out of reach. Each lane brings
two parts. The robot must put all four in a grey bin nearer the arms before each one reaches the end line.

The order that works is not the order the parts come out. Each time, a slow far-lane part comes out first, and about
2 s later a fast near-lane part comes out behind it (further from the end line), yet the fast part reaches the end
line about 12 s sooner. A robot that takes the first part to come out (or the one nearest the end) is still busy with
it when the fast part runs out, and scores 0.5; one that waits for the fast part and takes the slow one after it has
more than 5 s to spare. Pickers working several lanes or a merge make this call all the time: what matters is time
left in reach, which depends on speed, not on arrival order or distance.

| | comes out | reaches the end line |
|---|---|---|
| far (slow) part 1 | 1.75 s | 19.75 s |
| near (fast) part 1 | 3.85 s | 7.45 s |
| far (slow) part 2 | 12.5 s | 30.5 s |
| near (fast) part 2 | 14.6 s | 18.2 s |

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: lanes, carriers' paths, covers, bin, the straight-to-the-bin rule, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files (a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction), `release_cruise` paths for carriers on a constant-speed lane, and `on_step` / `on_reset` hooks for the rule's monitor. Copied from this repository's reach-between-sweeps task (itself extending unload-paused-shuttle's) |
| `test_task.py` | Offline checks (no Isaac Sim): lane speeds and smoothness, that parts start and end hidden, that taking parts in arrival order loses both fast parts while the deadline order fits, the scoring and the rule, and that the demo is reachable, on time and clear of the bin, the other lane and the covers |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without modifying
yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer.
The carriers' clock is the episode clock (`observation.state_time`): they stand still through the reset and the
first release timer starts with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac
Lab-Arena 0.3.1). **Measured on an L40S:** the scripted demo ended with success after 499 steps (25 s); Dream-1 scored 0 in a 900-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Four parts will come out on the two lanes, two on each. Put every part in the grey bin before it rides under the cover on the left. The near lane runs five times as fast as the far lane." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/lanes-dream

# The scripted demo, meant to reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Four parts will come out on the two lanes, two on each. Put every part in the grey bin before it rides under the cover on the left. The near lane runs five times as fast as the far lane." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/lanes-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/lanes-*/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end: 1.0 when `success()` has held for 1 s (which ends the trial), otherwise
`score()` at the end of the trial (`SCORE_OVER = "end"`).

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 0.25 per cube | the cube lies still in the bin when the trial ends: centre inside the walls, underside below the rim (on the floor or on another cube), and it was not lost | a photo from above |
| 1.0 | all four cubes lie in the bin, none lost, for 1 s | a photo from above |
| lost (scores 0) | for 0.5 s or more in a row, the cube was neither on its own carrier, nor held in a gripper (centre within 5 cm of the point between the finger pads), nor over the inside of the bin: it was put down or dropped elsewhere (table, a cover, another carrier or cube) or knocked off its lane | video |

A cube that rides under the run-out cover is out of reach and scores 0. The "lost" rule stops a robot beating the
clock by knocking parts off the lanes, or parking them, and picking them up later.

## The scripted demo

Open loop, left arm only (the right arm can't reach the window). The fingers close across the lane (grip yaw -90
degrees), so a coming cube rides in between the open pads, with 2.7 cm to spare each side. For each part the arm moves
over the lane at a point the part has not yet reached, lowers to grasp height, waits until the cube's centre passes
under the TCP, and closes in 0.6 s while moving along with the cube (within 6 mm), then lifts straight up (the
carrier slides on from under it), carries it to the bin and lets go 1.9 cm above the rim. Order and closing times:
near part 1 at 4.95 s, far part 1 at 10.15 s, near part 2 at 15.7 s, far part 2 at 20.9 s; the plan ends at 24.6 s,
so the trial should end with success at about 25.6 s (about 510 steps). Each fast part is closed on 1.2 s before it
would leave the demo's catching zone (9 cm short of the end line), each slow part 5.5 s before.

## Runtime and seeds

- One fixed layout: `-E object_jitter=0`; the seed doesn't change the start or the timing.
- The demo should end with success after about 510 steps; a 900-step Dream-1 trial takes a few minutes on an L40S.

## Fidelity

The carriers follow their paths exactly, like servo-driven carriers: the arms can't push or stop them (a real drive
would fault if blocked). They ride 2 mm above the rails without touching them. The cubes have the station's
rubber-like friction (1.2 static, 1.0 dynamic), more than aluminium on steel; the carriers' peak acceleration
(0.31 m/s²) is far below what that friction holds, so cubes don't slip on their carriers. A real line would use
belts and a timed escapement under the feed cover; here each cube rides its own carrier released at a fixed time.
