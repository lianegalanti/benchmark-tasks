# Stop a sliding carton and push it onto the packing square

A grey ball-transfer deck (35 x 78 x 3 cm, almost frictionless) lies across the table in front of the arms, its left
end an open 3 cm drop. A brown carton (10 x 10 x 6 cm, 150 g) starts on its right end already sliding left at
0.42 m/s. Untouched, it coasts past a green 16 x 16 cm square marked on the deck and slides off the left end after
2.3 s. The robot must stop it before it goes off, then push it onto the square and leave it there, upright and at
rest. It has to act at once (the carton reaches the square's line about 1.8 s in), put something in the carton's
path rather than chase it, and push it gently enough that it doesn't coast past the square: on this deck, a carton
pushed and let go keeps sliding. The carton is wider than the open gripper, so it can only be pushed.

At packing and sortation stations, parcels come off chutes and pushers onto ball-transfer tables and keep coasting;
the worker catches each one before it runs off the end and steers it to the working spot.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the deck, the carton and its launch, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds object kinds to yam_arena task files: `DrivenBox` (follows a scripted path), `LaunchedBox` (a free box with its own friction that starts the episode moving), `SlickBox` (a fixed low-friction box whose friction wins over what touches it) and `Marker` (a visual-only box, no collider); plus `slide()`, the analytic slide used by the tests |
| `test_task.py` | Offline checks (no Isaac Sim): the deck and its painted square, the untouched slide (analytic, deceleration = friction x g), the scoring, and that the demo is reachable, in place before the carton arrives, and pushes gently enough |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The deck is
one static box with friction 0.01 and PhysX's "min" friction combine mode, so the deck's friction, not the carton's
(0.5 / 0.4, cardboard), rules their contact. The green square is a 1 mm visual-only marker on top of it (no
collider): an earlier deck tiled around the square made the carton, pushed across a tile seam, catch on the next
tile's edge and tip over it. The carton's velocity is
written once at the end of the embodiment's reset, after the scene has settled, so it is moving from the first
control step on (`observation.state_time` 0). Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena
0.3.1).

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stop the sliding carton before it falls off the deck and push it into the green square." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/carton-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stop the sliding carton before it falls off the deck and push it into the green square." \
  --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/carton-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/carton-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
`score()` at the end of the trial (`SCORE_OVER = "end"`), so a carton stopped and then pushed off the deck scores 0.

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | the carton stands upright on the deck (top face within 15 degrees of vertical, underside within 1 cm of the deck top), its whole footprint inside the green square, still for 1 s | a photo from above |
| 0.5 | at the end: upright on the deck, its centre inside the green square (but sticking out of it, or still moving) | a photo from above, with the carton's centre marked on its lid |
| 0.25 | at the end: upright on the deck anywhere else | a photo |
| 0 | anything else: it slid or was pushed off the deck, tipped over, or is held off the deck | |

## Runtime and seeds

- The scripted demo (left arm) stands its closed fingers in the carton's path, in line with the square, by 1.1 s;
  the carton runs into them at about 1.75 s (at 0.25 m/s) and stops. The arm then lifts over it, turns the fingers a
  quarter turn, comes down behind its far side and pushes it 16 cm toward the arms over 4 s (never slowing faster
  than 0.06 m/s², below the 0.1 m/s² at which friction slows the carton, so it stays against the fingers and stops
  with them), and backs off at 10.2 s. In Isaac Sim the carton stops 1 cm off the square's centre line and the push leaves it
  within 1 mm of the square's centre; the trial ends with success at about 9.9 s of simulated time (about 200
  steps).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The carton always starts at the same
  place with the same velocity.

## Fidelity

The ball-transfer deck is a flat plate with friction 0.01, which is how a light carton coasts on ball transfers
(slowing at about 0.1 m/s²), without the balls' bumps or their slightly direction-dependent drag. The analytic slide
in the tests (constant deceleration of friction x g) puts the untouched carton off the deck's end with 15 cm to
spare, and still off with 25 % more friction. The carton is a rigid box with cardboard friction (0.5 / 0.4) against
the fingers. Its launch is a velocity set on it at the start, standing in for a pusher or chute. The demo's finger
contact positions use 1.8 cm from the closed fingers' centre line to where they touch the carton, measured in Isaac
Sim; the square leaves 3 cm of play on each side.
