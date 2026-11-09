# Push two reject cartons off a moving line onto the reject table, leaving the good ones on it

A conveyor crosses the table 26 cm in front of the arms and carries four cartons (12 x 10 x 5 cm) on steel carriers
from the robot's right to its left at a steady 5 cm/s: brown, red, brown, red, with 3 cm between cartons. The red
ones are rejects. Along the far side of the line runs a reject table, 5 mm lower than the carriers and 1 cm from them.
At the left end the line runs into a tunnel too low to reach into. The robot must push each red carton across the line
onto the reject table before it reaches the tunnel, and leave both brown cartons standing on their carriers.

The cartons are wider than the open gripper (9.4 cm) both ways, so one hand can't pick them: each reject has to be
pushed sideways off a carrier that keeps moving, while the brown cartons 3 cm in front of it and behind it keep riding
past. The arm has to come down in the gap in front of the carton, move along the line with it while it pushes it
across, and do this within the few seconds the carton spends in its reach. Pushing everything off, or sweeping the line
clear, scores nothing.

Packing lines work this way where there is no automatic diverter: a carton that fails a check (weight, seal, label)
is pushed off the running conveyor onto a reject table by the operator beside it, without stopping the line or
disturbing the cartons around it.

The board's other conveyor tasks pick parts off a belt (sorting coloured blocks, packing biscuit packs, sampling every
fourth part, sorting off a belt that changes speed) or place parts onto moving totes and carriers. Here nothing can be picked: the skill
is pushing a moving object sideways across its carrier, timed to it and to its neighbours.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the conveyor, carriers, cartons, reject table and tunnel, the belt's run, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction. Copied unchanged from our sort-surging-belt task (itself from unload-paused-shuttle); this task uses its `speed_schedule()` / `belt_path()` |
| `test_task.py` | Offline checks (no Isaac Sim): the belt, the line, the reject table and the tunnel, every scoring level (including pushing or sweeping everything off), that a low push slides a carton rather than tipping it, and that the demo is reachable, comes down clear of the cartons and carriers, keeps each arm in its own half, and leaves each reject on the table before the tunnel whatever the exact finger contact point |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The belt's
clock is the episode clock (`observation.state_time`): it stands still through the reset and starts with the first
control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured on an L40S:** the scripted
demo ended with success after 327 steps (16.4 s); Dream-1 scored 0 in a 1,200-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The two red cartons on the moving line are rejects. Push each one off the line onto the grey reject table behind the line before it reaches the tunnel, and leave the brown cartons on the line." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/rejects-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The two red cartons on the moving line are rejects. Push each one off the line onto the grey reject table behind the line before it reaches the tunnel, and leave the brown cartons on the line." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/rejects-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/rejects-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 once `success()` has held for 1 s. Otherwise it
scores `score()` at the end of the trial (`SCORE_OVER = "end"`).

## The line

The belt speeds up to 5 cm/s over the first 0.3 s, runs at 5 cm/s, and stops at 19.4 s (over 0.3 s), when every
carrier is inside the tunnel. Untouched, each carton's leading face reaches the tunnel's mouth at 8.8 s (brown 1),
11.4 s (red 1), 14.0 s (brown 2) and 16.5 s (red 2).

## Scoring

At the end of the trial:

| Score | Condition | Check on a real table |
|---|---|---|
| +0.5 per red carton | lying on the reject table: its centre 3 cm or more past the table's near edge and inside its fence, less than 8 cm above the table top (not held up), still | a photo from above with a mark 3 cm in from the table's near edge |
| +0.125 per red carton | off the line but not on the reject table: its centre outside its carrier's 14 x 12 cm footprint, or 8 cm or more above it (on the table top, the floor, in a gripper) | the same photo |
| 0 per red carton | still on its carrier in any pose (gone into the tunnel, or knocked over on it) | look into the tunnel |
| −0.5 per brown carton | not standing upright on its own carrier (pushed off, knocked over, lifted): its centre outside the carrier's footprint, or more than 1 cm off the height it rides at | the photo, and a look into the tunnel |

The total is never below 0. Both reds on the table and both browns on their carriers is 1.0 (success). Pushing every
carton onto the table, or sweeping the whole line onto the table top or the floor, scores 0; pushing a brown carton
off costs as much as one red carton pushed off correctly gains.

## Runtime and seeds

- The scripted demo, open loop and timed against the belt, with the fingers closed throughout and closing across the
  line (each carton meets the flat outer side of a finger). The right arm comes down in front of red 1 at 1.4 s, 1.5 cm
  clear of it and in front of its carrier, with the pad tips 6 mm above the carriers' tops, follows it along the line,
  closes in and pushes it 16 cm across onto the table from 2.4 to 4.4 s (peak 12 cm/s), still following the line, then
  rises and goes back. The left arm does the same for red 2 from 11.8 s (once it is 5 cm past the midpoint), pushing
  from 12.8 to 14.8 s, 8 cm before the tunnel. The last arm is back at 17.0 s; in Isaac Sim the trial ended with success
  after 327 steps (16.4 s).
- Dream-1 scored 0 in a 1,200-step trial.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.

## Layout notes

- The reject table's top is 5 mm below the carriers' tops and 1 cm from their far edge: a carton pushed across drops
  slightly onto it rather than butting into its edge (in our stop-sliding-carton task a carton pushed across a flush
  seam caught on it and tipped).
- A push at the carton's top edge, the worst case, needs friction above 1.2 to tip a 12 cm deep, 5 cm tall carton; the
  carton-on-carrier friction is about 1.0 static, so it slides.
- The tunnel overhangs the table's left edge by 27 cm on a post: it has to hold all four carriers once they pass its
  mouth, and the mouth has to stay in the left arm's reach.

## Fidelity

The carriers follow their path exactly, like a servo-driven conveyor: the arms can't push or stop them. They ride
2 mm above the rails. The cartons are rigid boxes with the station's rubber-like friction (1.2 static, 1.0 dynamic),
more than cardboard on steel; the carriers have 0.8 / 0.7. Real reject cartons usually carry a red label rather
than being red all over; one colour per rigid body keeps the simulation simple.
