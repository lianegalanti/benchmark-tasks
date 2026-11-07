# Unload a part from a cycling shuttle table during its pause

A grey shuttle plate (12 x 12 x 2 cm) runs on two rails across the table, 34 cm in front of the arms, carrying an
orange aluminium block (5 x 5 x 3 cm). The shuttle never stops for the robot: it waits 2.5 s at its right end, slides
30 cm to its left end in 2.5 s, waits there 2.5 s, slides back, and repeats. The robot must take the block off and put
it in a blue tray in front of the left arm. It has to time its grasp: the block is only in reach of the left arm, and
only still, during the 2.5 s pause at the left end, and the fingers must be clear before the shuttle leaves.

Shuttle tables, indexing transfer lines and rotary index tables feed parts to machines on a fixed cycle; the
operator (or robot) that unloads them works to the machine's rhythm, not the other way round.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the shuttle's path, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction |
| `test_task.py` | Offline checks (no Isaac Sim): the shuttle's path, the scoring, and that the demo is reachable and grabs during the pause |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The
shuttle's clock is the episode clock (`observation.state_time`): it stands still at its start through the reset and
starts with the first control step. Tested with yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the orange block off the moving shuttle and put it in the blue tray. The shuttle keeps cycling and won't stop for you." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/shuttle-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the orange block off the moving shuttle and put it in the blue tray. The shuttle keeps cycling and won't stop for you." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/shuttle-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/shuttle-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
the best `score()` reached during the trial (`SCORE_OVER = "best"`).

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | the block's centre is inside the tray's walls, its underside within 1 cm of the tray floor, and it stays still for 1 s | a photo from above, and that it lies on the floor |
| 0.5 | at some point the block's centre was inside the tray's walls, with its underside less than 5 cm above the rim (it was carried to the tray but didn't end up lying in it) | video |
| 0.25 | at some point the block's underside was 3 cm or more above the shuttle's top (it was taken off the shuttle) | video, against a 3 cm mark on the shuttle's end stop |
| 0 | anything else: the block never lifted, pushed off, or knocked to the table | |

The partial credits are the best the trial reached, so a block lifted and then dropped on the table keeps 0.25.

## Runtime and seeds

- The scripted demo waits above the shuttle's left stop, grabs the block during its first pause there (5.0 to 7.5 s),
  and lets go in the tray at 10.7 s; on an L40S the trial ended with success after 232 steps (11.6 s of simulated
  time). The whole demo run, including starting Isaac Sim, took about 2 minutes.
- Dream-1 (`dreamlabs/dream-1-beta`) reached full score in its trial: it took the block during a pause and the trial
  ended with success after about 1,030 steps (52 s), so no demo video is submitted. A 2,400-step trial with
  Dream-1 takes up to about 8 minutes on an L40S.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The shuttle always starts at its right
  end, at the beginning of its 2.5 s wait.

## Fidelity

The shuttle is driven exactly along its path, like a stiff servo axis: the arms can't push it off its path or stop it
(on a real machine, blocking the shuttle would trip its drive; here it simply carries on). It rides 2 mm above its
rails without touching them. The block and the shuttle's top use plain boxes with sharp edges; the block has the
station's rubber-like friction (1.2 static, 1.0 dynamic), more than aluminium on steel, so it doesn't slide on the
shuttle as it starts and stops (its peak acceleration is under 0.5 m/s²).
