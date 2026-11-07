# Relay a part between two cycling shuttle lines

Two shuttle carriers run toward and away from the arms on their own rails, one 18 cm outside each arm's base. They
never stop for the robot. Carrier A (outside the right arm) brings an orange aluminium block (5 x 5 x 3 cm) in from
its far stop, waits 2 s at its near stop and goes back, on a 9 s cycle. Carrier B (outside the left arm) has a green
nest, four 5 mm lips around a 7 x 7 cm pocket, and runs the same 16 cm stroke on its own 12 s cycle (3 s waits, 3 s
moves). Neither arm reaches the other arm's carrier, so the block has to pass from the right arm to the left arm,
here by way of a fixed yellow handover pad halfway between them. The left arm must then set it in B's nest while B
waits, and be clear before B moves on. Both arms work to two machines' rhythms that don't line up with each other.

Robot cells between two indexing lines or machines (a press and a conveyor, two shuttle tables) move parts from one
line's cycle to the other's; when the lines are too far apart for one arm, a two-robot cell passes the part across a
fixed transfer nest. A fixed pad is the usual industrial choice over a direct hand-to-hand pass: each robot works
alone in its own zone and never has to match the other's pose in mid-air, which is also how the scripted demo works.
The task accepts any relay that ends with the block in B's nest.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the carriers' paths, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction |
| `test_task.py` | Offline checks (no Isaac Sim): the carriers' paths and the nest riding with B, the scoring, that neither arm reaches the other's carrier, and that the demo is reachable for both arms, grabs and releases during the pauses and keeps the arms apart |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The
carriers' clock is the episode clock (`observation.state_time`): it stands still at its start through the reset and
starts with the first control step. B's nest lips are separate driven boxes on B's path, 1 mm above B's top.
Tested offline with yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the orange block from the right carrier into the green nest on the left carrier. Neither carrier stops for you, and each arm reaches only its own carrier." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/relay-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the orange block from the right carrier into the green nest on the left carrier. Neither carrier stops for you, and each arm reaches only its own carrier." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/relay-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/relay-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
the best `score()` reached during the trial (`SCORE_OVER = "best"`).

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | the block lies in B's nest: its centre within 1 cm of the nest's centre along and across the line (between the lips), its underside within 1 cm of B's top, and it moves with B (within 1 cm/s of B's velocity, turning under 0.5 rad/s) for 1 s, whether B is waiting or moving | a photo from above, and that it rides out with B |
| 0.5 | at some point the block's centre was 81 cm or more across the table from the right arm's base, past that arm's reach (about 78 cm), with its underside 3 cm or more above the table (the left arm carried it toward B) | video: the block lifted over B's line or just short of it |
| 0.25 | at some point the block's underside was 3 cm or more above carrier A's top (it was taken off A) | video, against a 3 cm mark on A's end stop |
| 0 | anything else: the block never lifted, pushed off A, or knocked to the table | |

The partial credits are the best the trial reached, so a block carried toward B and then dropped keeps 0.5. The right arm
alone can't earn 0.5: its TCP reaches at most about 78 cm from its base (sampled from its joint limits, in the tests). The hooks
don't see the grippers: a block held still in the nest while B waits also counts, but one held while B moves doesn't.

## Runtime and seeds

- The scripted demo grabs the block during A's first wait at its near stop (4.5 to 6.5 s), sets it on the pad at
  10 s, the left arm takes it at 13.7 s and, during B's second wait at its near stop (18 to 21 s), lowers it slowly
  over the lips, pauses and lets go at 19.7 s. In Isaac Sim (yam_scripted, 20 Hz) it ends with success after 415
  steps (20.75 s of simulated time).
- The demo grips with the fingers commanded 4 cm apart on the 5 cm block rather than fully closed: fully closed, both
  fingers saturate at their 50 N limit, balance each other anywhere, and the block slid 2 to 3 cm along the closing
  axis in the left hand and landed on a lip of the nest.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. Both carriers start at their far
  stops, at the beginning of their waits.

## Fidelity

The carriers are driven exactly along their paths, like stiff servo axes: the arms can't push them off their paths or
stop them (on a real machine, blocking a carrier would trip its drive; here it simply carries on). They ride 2 mm
above their rails, and the nest lips 1 mm above B, without touching. The block and the carriers' tops are plain boxes
with sharp edges; the block has the station's rubber-like friction (1.2 static, 1.0 dynamic), more than aluminium on
steel, so it doesn't slide on a carrier as it starts and stops (peak acceleration under 0.5 m/s²). Each arm's reach
across the table (about 0.78 m from its base in any wrist orientation, 1 to 15 cm above the table, sampled from its
joint limits) falls at least 3 cm short of the other arm's carrier, which keeps the relay necessary.
