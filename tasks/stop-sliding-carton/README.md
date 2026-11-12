# Stop a sliding carton and push it onto the packing square

A grey ball-transfer deck (35 x 78 x 3 cm, almost frictionless) lies across the table in front of the arms, its left
end an open 3 cm drop. A brown carton (10 x 10 x 6 cm, 150 g) starts on its right end already sliding left at
0.42 m/s. Untouched, it coasts past a green 16 x 16 cm square marked on the deck and slides off the left end after
2.3 s. The robot must stop it before it goes off, then push it onto the square and leave it there, upright and at
rest. It has to act at once (untouched, the carton's centre comes level with the square's right edge 1.5 s in and crosses its centre line 1.8 s in), put something in the carton's
path rather than chase it, and push it gently enough that it doesn't coast past the square: on this deck, a carton
pushed and let go keeps sliding. The carton is wider than the open gripper and must not be lifted: it is pushed along the deck.

At packing and sortation stations, parcels come off chutes and pushers onto ball-transfer tables and keep coasting;
the worker catches each one before it runs off the end and steers it to the working spot.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the deck, the carton and its launch, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds object kinds to yam_arena task files: `DrivenBox` (follows a scripted path), `LaunchedBox` (a free box with its own friction that starts the episode moving), `SlickBox` (a fixed low-friction box whose friction wins over what touches it) and `Marker` (a visual-only box, no collider); `on_step` / `on_reset` hooks for the whole-trial monitor; plus `slide()`, the analytic slide used by the tests |
| `test_task.py` | Offline checks (no Isaac Sim): the deck and its painted square, the untouched slide (analytic, deceleration = friction x g), the scoring, the whole-trial monitor (letting the carton fall onto the table, or over any edge, and putting it back, or lifting and carrying it, scores 0; the demo's path stays 5 cm or more inside the deck), and that the demo is reachable, in place before the carton arrives, and pushes gently enough |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset, control step and simulation step without modifying yam_arena,
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
  --instruction "Stop the sliding carton before it falls off the deck and push it into the green square, without lifting it off the deck." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/carton-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Stop the sliding carton before it falls off the deck and push it into the green square, without lifting it off the deck." \
  --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/carton-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/carton-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The carton is tracked through the whole trial: after every control step a
monitor (`_monitor`, registered with motionkit's `on_step`) checks whether it has left the deck (its centre no longer
over the deck, or its underside more than 1 cm below the deck top) or been lifted (its underside more than 2 cm above
the deck top) and, if so, latches that for the rest of the
trial (only the next reset clears it); `success()` and `score()` then both fail. So letting the carton fall onto the
table and putting it back on the square, or picking it up and carrying it there, scores 0. Otherwise the trial ends with 1.0 when `success()` has held for
1 s, or scores `score()` at the end of the trial (`SCORE_OVER = "end"`).

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 0 | at any moment of the trial the carton left the deck (its centre no longer over the deck, or its underside more than 1 cm below the deck top) or was lifted (its underside more than 2 cm above the deck top), even if it is put back afterwards | a side video (or the operator watching, with a 2 cm mark on a ruler standing by the deck): did it ever go over an edge or come up off the deck? |
| 1.0 | it never left the deck nor was lifted, and it stands upright on the deck (its top face's normal within 15 degrees of vertical, underside within 1 cm of the deck top), its whole footprint inside the green square, still for 1 s | a photo from above, and a phone level on its lid |
| 0.5 | it never left the deck nor was lifted, and at the end it stands upright on the deck with its centre inside the green square (but sticking out of it, or still moving) | a photo from above, with the carton's centre marked on its lid |
| 0.25 | it never left the deck nor was lifted, and at the end it stands upright on the deck anywhere else | a photo |
| 0 | anything else: it tipped over (its top face's normal more than 15 degrees from vertical) or is held off the deck at the end | |

## Runtime and seeds

- The scripted demo (left arm) stands its closed fingers in the carton's path, in line with the square, by 1.1 s;
  the carton runs into them at about 1.75 s (at 0.25 m/s) and stops. The arm then lifts over it, turns the fingers a
  quarter turn, comes down behind its far side and pushes it 16 cm toward the arms over 4 s (never slowing faster
  than 0.06 m/s², below the 0.1 m/s² at which friction slows the carton, so it stays against the fingers and stops
  with them), and backs off at 10.2 s. In Isaac Sim the carton stops 1 cm off the square's centre line and the push leaves it
  within 1 mm of the square's centre; with the whole-trial monitor on, the carton never leaves the deck
  (its underside never rises above the deck top, against the 2 cm allowed) and the trial ends with success (task_score 1.0) at 9.9 s of simulated time (198 steps).
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
