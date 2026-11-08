# Pull every fourth part off an indexing line for QC, counting identical parts as they pass

An indexing conveyor crosses the table 32 cm in front of the arms and carries parts from the robot's right to its
left. A grey steel cover hides it except a 10 cm open window, just left of the midpoint between the arms. Eight
steel carriers ride on it 9 cm apart, each with one bare aluminium cube (4 cm). The eight cubes are identical. Every
4 s the line moves 9 cm, so each carrier in turn stops in the window for 2.8 s and then goes on under the cover, out
of reach. The robot must take every fourth part off the line into a blue QC tray, counting the first part to arrive as
part 1 (so parts 4 and 8), and let the others ride past.

The parts are identical and only one is visible at a time, so the count is the only cue. The robot must keep it in
memory across about 30 s and several stops it should do nothing in. Sampling every nth part from a running line is
how plants catch process drift; the board's other conveyor tasks pick parts by how they look (colour, a cracked band)
or by need, not by their place in the sequence.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the conveyor, carriers, cover and tray, the line's timing, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction. Copied from our unload-paused-shuttle task, with `index_line()` added (a carrier on an indexing conveyor) |
| `test_task.py` | Offline checks (no Isaac Sim): the line's motion and what the cover hides, every scoring level, and that the demo is reachable and takes parts 4 and 8 during their stops |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena.
The task therefore runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The
line's clock is the episode clock (`observation.state_time`). It stands still through the reset and starts with the
first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). On an L40S the scripted
demo ended with success after 777 steps (38.9 s); Dream-1 scored 0 in a 2,400-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Parts will ride past on the conveyor one at a time. Take every fourth part off the line and put it in the blue QC tray, counting the first part to arrive as part 1 (so the 4th and the 8th), and let all the others ride past." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/sample-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Parts will ride past on the conveyor one at a time. Take every fourth part off the line and put it in the blue QC tray, counting the first part to arrive as part 1 (so the 4th and the 8th), and let all the others ride past." \
  --scorer task_score --max-steps 1000 --store-frames --log-dir ~/yam-eval/sample-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/sample-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 once `success()` has held for 1 s. Otherwise it
scores `score()` at the end of the trial (`SCORE_OVER = "end"`): what is in the tray and what has left the line then.

## The line

| Time (s) | What happens |
|---|---|
| 0 to 3.0 | the line stands still; the window is empty, carriers 1 to 8 wait under the right tunnel |
| 3.0 + 4(k-1) | move k (k = 1 to 9): every carrier moves 9 cm to the left in 1.2 s (cosine profile, peak 11.8 cm/s, under 0.4 m/s²) |
| 4.2 + 4(k-1) to 7.0 + 4(k-1) | part k stands in the window (part 4: 16.2 to 19.0 s; part 8: 32.2 to 35.0 s) |
| from 36.2 | the line has stopped; every carrier is under the left tunnel and the window is empty |

While a part stands in the window, its neighbours' cubes are 2 cm inside the tunnels, under a roof 2 cm above the
cubes' tops. Before and after its stop, every part is out of reach and out of sight, so a missed part can't be
fetched later, and a policy can't count the parts waiting or the parts gone. Parts are still scored by identity
(their number), so a part taken at any time counts as itself.

## Scoring

| Score | Condition at the end of the trial | Check on a real table |
|---|---|---|
| 1.0 | cubes 4 and 8 lying in the tray, and every other cube lying on a carrier | read the numbers under the cubes in the tray; lift the left cover and count the cubes on carriers |
| 0.5 per sample | cube 4 or 8 lying in the tray: centre inside the tray's walls, underside within 1 cm of the tray floor, still | a photo from above, the numbers underneath |
| -0.25 per wrong cube | any cube other than 4 and 8 not lying on a carrier (in the tray, on the table, in a gripper) | as above |
| floor 0 | the total never goes below 0 | |

Examples: 4 and 8 in the tray, 1.0. Only 4 (8 missed), 0.5. Cubes 3, 4 and 8 taken, 0.75. Every second part (2, 4,
6, 8), 0.5. Off by one (3 and 7), 0. Every part into the tray, 0. Nothing taken, 0. A sample dropped on the table
earns nothing but costs nothing.

## Runtime and seeds

- The scripted demo (left arm, open loop, timed against the line) waits over the window with its fingers closing
  across the line. Parts 1 to 3 pass under it. It grabs part 4 at 16.4 to 17.6 s, lays it in the tray, comes back,
  lets parts 5 to 7 pass, grabs part 8 at 32.4 to 33.6 s and lays it beside part 4. It lets go at 38.0 s, so the
  trial should end with success at about 40 s (800 steps).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.

## Fidelity

The carriers follow their path exactly, like a stiff servo-driven indexing conveyor: the arms can't push them or stop
them, where a real line would stop on a jam. They ride 2 mm above the rails. The cubes have the station's
rubber-like friction (1.2 static, 1.0 dynamic), more than aluminium on steel. At a peak acceleration under 0.4 m/s²,
they don't slide on the carriers. The conveyor is longer than the table: its left end overhangs the table's left edge
by 23 cm on a post to the floor, and its right end overhangs the right edge by 3 cm.
