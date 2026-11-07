# Catch a part as it comes out of a covered tunnel on a moving carrier

A grey carrier plate (12 x 12 x 2 cm) runs on two rails across the table, 32 cm in front of the arms, carrying an
orange aluminium block (5 x 5 x 3 cm). Halfway along, the rails pass through a dark tunnel, open at both ends, whose
roof hides the block from the overhead camera and sits only 2.3 cm above it, too low for the fingers to reach in.
The carrier never stops for the robot: it waits 2 s at its right end, runs 44 cm at 10 cm/s through the tunnel to its
left end, waits there 2 s, runs back into the tunnel, and repeats. The robot must take the block off the carrier
after it comes out of the tunnel and put it on a green pad in front of the left arm. It has to predict where and when
the block will come out from how it went in: the block is in reach only in a 4.4 s window after it leaves the
tunnel (1.2 s arriving, 2 s stopped, 1.2 s heading back), and then disappears again.

Lines that run parts through a wash or dry tunnel, an oven or an inspection hood hide them for a while; the operator
(or robot) unloading at the exit must be ready where and when each part comes out, on the line's schedule.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the carrier's path, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction; `cruise_shuttle` gives the constant-speed back-and-forth path |
| `test_task.py` | Offline checks (no Isaac Sim): the carrier's path, that the tunnel hides the block and keeps fingers out, the scoring, and that the demo is reachable, grabs while the block is out and never puts its fingers into the tunnel |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The
carrier's clock is the episode clock (`observation.state_time`): it stands still at its start through the reset and
starts with the first control step. Tested with yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the orange block off the carrier when it comes out of the tunnel and put it on the green pad. The carrier keeps moving and won't stop for you." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/tunnel-dream

# The scripted demo, which reaches full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the orange block off the carrier when it comes out of the tunnel and put it on the green pad. The carrier keeps moving and won't stop for you." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/tunnel-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/tunnel-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
the best `score()` reached during the trial (`SCORE_OVER = "best"`).

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | the block's centre is over the pad, its underside within 1 cm of the pad's top, and it stays still for 1 s | a photo from above, and that it lies on the pad |
| 0.5 | at some point the block's centre was over the pad, with its underside less than 5 cm above the pad's top (it was carried to the pad but didn't end up lying on it) | video, against a 5 cm mark beside the pad |
| 0.25 | at some point the block's underside was 3 cm or more above the carrier's top (it was taken off the carrier) | video, against a 3 cm mark on the carrier's end stop |
| 0 | anything else: the block never lifted, pushed off, or knocked to the table | |

The partial credits are the best the trial reached, so a block lifted and then dropped on the table keeps 0.25.

## Runtime and seeds

- The carrier's first pass: the block goes under the roof at 2.8 s, is wholly hidden from 3.3 to 5.2 s, is wholly out
  of the tunnel's left end from 5.7 s, stops at 6.9 s, leaves at 8.9 s and starts back under the roof at 10.1 s. The
  cycle repeats every 13.8 s, so a missed block comes back out every 13.8 s.
- The scripted demo waits above the carrier's left stop, grabs the block during its first stop there (descending at
  7.05 s, lifting at 8.25 s), and lets go on the pad at 12.25 s; the trial should end with success at about 13.5 s
  of simulated time (about 270 steps). On an L40S it ended with success after 263 steps (13.2 s).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The carrier always starts at its right
  end, at the beginning of its 2 s wait.

## Fidelity

The carrier is driven exactly along its path, like a stiff servo axis: the arms can't push it off its path or stop it
(on a real machine, blocking the carrier would trip its drive; here it simply carries on). It rides 2 mm above its
rails without touching them. The tunnel is three fixed boxes (two walls and a roof) with 1.5 cm of side clearance to
the carrier. The block and the carrier's top use plain boxes with sharp edges; the block has the station's
rubber-like friction (1.2 static, 1.0 dynamic), more than aluminium on steel, so it doesn't slide on the carrier as it
starts and stops (its peak acceleration is 0.31 m/s²).
