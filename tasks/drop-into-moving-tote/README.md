# Put a part into a tote that keeps moving

An orange aluminium block (5 x 5 x 3 cm) lies on the table 20 cm in front of the left arm. A blue open tote (14 x 14
cm inside, 6 cm tall) glides back and forth across the front of both arms, 34 cm out, and never pauses: it runs 50 cm
at 10 cm/s, eases into each end over 0.8 s, turns straight back, and repeats (5.8 s each way). The robot must put the
block in the tote while it moves. Unlike picking from a conveyor, the part is still and the *target* moves: the robot
has to carry the block to where the tote will be, match its motion, and let go so the block lands inside and rides
along.

Kitting and packing lines work this way: totes, trays and cartons run past a station on a conveyor that doesn't stop,
and the person or robot loading them drops each part in on the fly.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the tote's path, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction. `back_and_forth()` builds the tote's non-stop path; `offset_path()` moves the tote's floor and walls together |
| `test_task.py` | Offline checks (no Isaac Sim): the tote's path, the scoring (including a block riding in the moving tote), that the demo is reachable, and that it lets go over the inside of the moving tote |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The
tote's clock is the episode clock (`observation.state_time`): it stands still at its start through the reset and
starts with the first control step. Tested with yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the orange block in the blue tote. The tote keeps moving and won't stop for you." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/tote-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the orange block in the blue tote. The tote keeps moving and won't stop for you." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/tote-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/tote-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
the best `score()` reached during the trial (`SCORE_OVER = "best"`).

## Scoring

All positions are taken relative to where the tote is at that moment.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | the block's centre is inside the tote's walls, its underside within 1 cm of the tote floor, and it moves with the tote (velocity within 1 cm/s of the tote's, turning under 0.5 rad/s) for 1 s | video: it lies on the floor and doesn't slide or tumble for 1 s |
| 0.5 | at some point the block's centre was inside the tote's walls, with its underside less than 5 cm above the rim (it was brought to the tote but didn't end up lying in it) | video, against a 5 cm mark above the rim |
| 0.25 | at some point the block's underside was 3 cm or more above the table (it was picked up) | video, against a 3 cm mark |
| 0 | anything else: the block never lifted, or was pushed along the table | |

The partial credits are the best the trial reached, so a block lifted and then dropped on the table keeps 0.25.

## Runtime and seeds

- The tote starts at rest at its right end and reaches its left end at 5.8 s. The scripted demo picks the block
  (lifted by 3.9 s), meets the tote just after it turns at its left end (6.0 s), follows it in 0.2 s steps while
  lowering, opens at 7.0 to 7.4 s with the block's underside 2 cm above the rim while the tote cruises at 10 cm/s,
  and rises. On an L40S the demo trial ended with success after 174 steps (8.7 s of simulated time).
- Dream-1 (`dreamlabs/dream-1-beta`) scored 0 in its 2,400-step trial (2 minutes). A full trial with Dream-1 takes
  about 8 minutes on an L40S.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The tote always starts at rest at its
  right end.

## Fidelity

The tote is driven exactly along its path, like a stiff servo axis or a conveyor: the arms can't push it off its path
or stop it. It glides 2 mm above the table without touching it. In the simulation it is five boxes (a floor and four
walls, 1 mm apart so the driven parts never touch each other) that move as one; a real tote is one moulded piece. The
block and the tote use plain boxes with sharp edges; the block has the station's rubber-like friction (1.2 static, 1.0
dynamic) and the tote 0.8 static, 0.7 dynamic, more than aluminium on plastic, so the block doesn't slide in the tote
as it turns (its peak acceleration is 0.2 m/s²).
