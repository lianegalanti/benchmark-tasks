# Slide a carton onto a cycling lift's deck while it waits level with the bench

A brown carton (12 x 12 x 6 cm, 300 g) stands on a grey steel-topped bench (8 cm tall) in front of the right arm.
Beyond the bench's far edge, across a 1.5 cm gap, is the yellow deck (18 x 18 cm) of a small automatic lift. The lift
never stops for the robot: it serves three levels on an 18 s cycle, waiting 3 s at each stop and taking 1.5 s between
them: low (deck top 5.4 cm below the bench top), level (4 mm below it), high (4.6 cm above it), level again, and back
to low. It starts at its low stop. The carton is wider than the open gripper (9.4 cm), so it has to be slid across.

The robot has to watch the deck and push only while it waits level with the bench: pushed while the deck is low, the
carton tips over the bench edge into the gap; pushed while it is high, it runs into the deck's side; left half across
when the deck leaves, it is levered over. A 22 cm push takes about 2 s of the 3 s wait, so it also has to start
promptly and get its fingers off the deck before the deck moves.

Category: **Impact on jobs.** Vertical lifts and lift tables in production and pallet-return lines serve several
conveyor levels on a fixed cycle; material handlers slide totes and cartons across at a landing, timed to the lift.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the lift's path, `success()`, `score()` (with a monitor that remembers a tip) and the scripted demo |
| `motionkit.py` | Adds to yam_arena task files a `DrivenBox` (a box whose pose and velocity follow a scripted path: the lift's deck), a `SlickBox` (a fixed box whose low friction rules its contacts: the bench), `dwell_stops` (a path visiting stops in turn, waiting at each: here, vertical), the episode clock and episode-start callbacks |
| `test_task.py` | Offline checks (no Isaac Sim): the lift's cycle, the layout, every scoring level, and that the demo is reachable, pushes only while the deck waits level, and stays clear of the bench and the deck |

`motionkit.py` is extended from the unload-paused-shuttle task's copy (`DrivenBox`), with `SlickBox` from
stop-sliding-carton and `clock`/`on_reset` from stop-ball-before-edge; `dwell_stops` and the deck's friction
combine mode are new. It hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying
yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer.
The lift's clock is the episode clock (`observation.state_time`): it stands still at the low stop through the reset
and starts with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

**Measured on an L40S:** the scripted demo ended with success after 152 steps (7.6 s). Dream-1 refused the instruction
as unsafe on its first inference request (`DA-4000 ... unsafe instruction, refused`), before taking a step, so there is
no model-run video.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide the carton onto the lift's deck while the deck is level with the bench. The lift keeps cycling and won't stop for you." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/lift-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Slide the carton onto the lift's deck while the deck is level with the bench. The lift keeps cycling and won't stop for you." \
  --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/lift-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/lift-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
`score()` at the end of the trial (`SCORE_OVER = "end"`), which keeps the best milestone reached unless the carton
has tipped or fallen, which zeroes it for good. `"best"` can't take credit away, so the task keeps that memory itself,
cleared at every episode start.

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | all four bottom corners of the carton over the deck, its underside within 1 cm of the deck's top, tilted 15° or less, moving with the deck (under 2 cm/s relative), for 1 s | a photo from above, and that it stands flat on the deck |
| 0.5 | at some point it stood upright partly on the deck: at least 2 cm of its underside over the deck, within 1 cm above the deck's top | video, against a 2 cm mark on the deck's near edge |
| 0.25 | at some point it stood upright on the bench with its centre at least 5 cm closer to the lift than at the start | video, against a 5 cm mark on the bench |
| 0 | it tilted more than 15° at any time, or its underside went more than 1 cm below the deck's low stop (on the table or in the gap), or it never moved 5 cm | video |

Upright means tilted 15° or less; the 4 mm step down onto the level deck tilts it about 2° while it crosses. A carton
overhanging the gap while the deck is low or high isn't over the deck's top within 1 cm, so it earns no 0.5; if the
deck then rises under an overhang (or leaves with the carton half across), the carton is levered over and scores 0.

## Runtime and seeds

- The deck waits level from 4.5 to 7.5 s, 13.5 to 16.5 s, and every 9 s after; a 1,200-step trial has 6 such waits.
- The scripted demo (right arm, closed fingers turned to close along the push, pad tips 6 mm above the bench) is
  behind the carton by 3.2 s, pushes it 21.5 cm across from 4.6 to 6.6 s (peak 0.16 m/s), leaving its centre 5 mm
  past the deck's centre with about 3 cm of deck on either side, backs its fingers off behind the bench edge by 7.3 s
  and lifts clear by 8.1 s. The carton should then ride the deck and success should hold by about 7.7 s.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.

## Fidelity

The deck is driven exactly along its path, like a stiff servo axis: the arms can't push it or stop it (on a real rig,
a low-force lead-screw stage that stalls when obstructed, so trials are safe). The lift's mechanism under the deck is
not modelled: the deck moves on its own and the gap below it is open. The bench top and the deck use steel-on-cardboard
friction (0.4 static, 0.3 dynamic, "min"-combined so it overrides the carton's own); the carton is a rigid box with
sharp edges. At its level stop the deck is 4 mm below the bench top, a landing tolerance, so the carton's leading edge
steps down onto it rather than catching on a flush edge (a carton pushed across a flush seam in Isaac Sim can catch and
tip).

## What the first Isaac Sim run should check

- That the carton crosses the 1.5 cm gap and 4 mm step without catching or tipping, and doesn't yaw more than a few
  degrees under the single-finger push (the contact offset of 1.8 cm is the one measured in the sliding-carton task).
- That it stays put on the deck as the deck rises (success is expected during the level wait or the ride up).
- That the gripper housing doesn't touch the carton's top edge during the push (the carton is only 6 cm tall).
