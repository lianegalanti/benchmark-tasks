# Catch a ball flying out of a chute in a basket before it touches the table

A grey chute stands on two legs 34 cm in front of the arms, sloping down toward the left at 10 degrees. Its lower
end is open, 15 cm above the table and 20 cm left of the midpoint between the arms. A cover lies over its lower
25 cm. An orange rubber ball (5 cm across) waits behind a gate in the chute's uncovered top, in front of the right
arm. At 7.5 s the gate lifts. The ball rolls down inside the covered chute, flies out of the open end at about
0.85 m/s and comes down 13 cm past it, 0.16 s later. Untouched, it lands on the bare table and rolls on off the
table's left edge. A light blue basket (11 x 16 cm inside, 6.5 cm deep, with a handle block on its near wall)
stands on the table in front of the left arm, off the ball's line. The robot must catch the ball in the basket
before the ball touches the table.

The cover stops the robot from taking the ball out of the chute. The robot can only predict where the ball will
come down and get the basket there before it does. Under the open end is the wrong place: a basket against the
chute's leg is hit by the ball or overshot. About 9 cm of basket positions along the ball's line catch it.

Category: **Impact on jobs.** Parts leaving a chute or a feeder outlet have to land in a container, not on the
bench, where they get damaged or roll off onto the floor. When the bin has been moved aside, a cell attendant has to
get it under the outlet, where the part will come down, before the next part comes out.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the gate's path, the analytic run and flight of the ball, the catch window, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds to yam_arena task files a `DrivenBox` (a box following a scripted path: the gate), a `Shape` (a fixed box in any orientation: the tilted chute, rails and cover), a `UsdBody` (a free rigid body from a USD file: the basket), the episode clock and episode-start callbacks |
| `assets/basket.usda` | The basket: one rigid body of six box colliders (floor, four walls, handle block) |
| `make_basket.py` | Writes `assets/basket.usda` from the numbers in `task.py` (needs `usd-core`) |
| `test_task.py` | Offline checks (no Isaac Sim): the layout, the ball's run and flight, the catch window, the scoring, and that the demo is reachable, on time and out of the ball's way |

`motionkit.py` is our ball task's helper (`stop-ball-before-edge`). It is extended with `UsdBody`, whose spawning
code is copied from `scenekit.py` in the `lock-before-loading` task folder; that folder's license applies to that
part. It hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena, so
the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The gate's
clock is the episode clock (`observation.state_time`). It stands still through the reset and settling, while the
ball rests against the gate, and starts with the first control step. Written against yam-isaac-lab-arena at
`bd62771` (Isaac Lab-Arena 0.3.1).

**Measured on an L40S:** the scripted demo ended with success after 252 steps (12.6 s); Dream-1 scored 0 in a
600-step trial. The flight numbers below are analytic, and the tests check them offline; in the sim the ball came out
at about 0.84 m/s and landed in the basket as predicted.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "In a few seconds a ball will fly out of the end of the covered chute. Catch it in the blue basket before it touches the table." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/catch-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "In a few seconds a ball will fly out of the end of the covered chute. Catch it in the blue basket before it touches the table." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/catch-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/catch-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 once `success()` has held for 1 s. Otherwise it
scores `score()` at the end of the trial (`SCORE_OVER = "end"`). The score reads a monitor of what the ball has done,
which is cleared at every episode start.

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | caught, then lying still in the basket for 1 s | video, then a photo |
| 0.75 | caught, but not in the basket at the end (it bounced or spilled out, or the basket was dropped) | video |
| 0.5 | it came out of the chute's lower end and is in the upright basket at the end, but wasn't caught (it touched the table first, or got there more than 1 s after leaving the cover) | video, then a photo |
| 0.25 | it came out of the chute's lower end and hasn't gone over an edge of the table | video |
| 0 | it is still in the chute, never went through it (taken off the open top, say), or fell off the table (its centre 5 cm or more below the table top) | video |

- **Caught:** the ball's centre is inside the basket's walls, above its floor's underside and below its rim, with
  the basket tilted at most 30 degrees. This must happen within 1 s after the ball was last inside the covered part
  of the chute, and before the ball touched the table. The flight takes 0.16 s. In the basket's own frame, so a held
  basket counts too.
- **Touched the table:** the ball's underside (its centre's height less 2.5 cm) came within 3 mm of the table top
  while its centre was outside the basket's footprint. In the basket it lies 8 mm up, on the basket's floor. The
  centre is used because yam_arena reports a spinning sphere's bounds as its turned box's, up to 4.3 cm below the
  centre. An earlier version that used the box counted a ball lying in the basket as touching the table.
- An untouched ball lands on the table and goes over the left edge about 0.4 s later, scoring 0. A ball taken off the
  open top before the gate lifts never went through the chute, so putting it in the basket scores 0.

## Runtime and timing

- Analytic run (no slip, no rolling resistance): released as the gate lifts (7.5 to 7.8 s), the ball leaves the
  chute 0.70 s later at 0.85 m/s, so at 8.20 to 8.50 s. It comes down onto the basket's floor 0.16 s after that.
  PhysX's default angular damping slows it by about 1 % on the chute. It leaves faster than sqrt(gR), so it flies
  off the floor's edge instead of following it round.
- Catch window: a basket standing on the ball's line catches it cleanly with its centre 28.1 to 36.9 cm left of the
  midpoint. Cleanly means the ball passes over the right wall without touching it and lands at least a radius
  inside both end walls. The window still contains the demo's spot (32.5 cm) for a ball 15 % slower or faster. A
  basket pushed up against the chute's leg (centre at 24 cm) is hit by the ball.
- Scripted demo (left arm, fingers closing across the handle):
  - 0.0 to 2.8 s: takes the basket by its handle, pad tips 1 cm above the rim, gripping 4.7 cm of the handle.
  - 2.8 to 6.4 s: lifts it slowly, carries it to (34, 32.5) cm in 1.5 s, sets it down from 4 mm and lets go.
  - by 7.4 s: lifts off over the handle and moves back to 15 cm in front of the bases, 19 cm off the ball's line.

  The basket is down 1.8 s before the ball can come out. The demo needs about 10.5 s (success 1 s after the ball comes
  to rest), well inside the 30 s run limit.
- One fixed layout: `-E object_jitter=0`. With jitter, the ball would spawn off the gate and the chute's centre line.
- Positions handed to Isaac Lab are plain Python floats (`np.float64` ones fail at reset in warp).


## Debug prints

Run with `BALL_DEBUG=1` in the environment. Every 0.5 s, and every step from the gate's lift until
the ball has landed, a `[ball-debug]` line gives:
- the basket's origin, tilt, yaw and offset from the catch spot;
- the left pads' gap and the left fingers' position;
- after the gate lifts, the ball's position and velocity;
- the monitor's flags.

## Fidelity

The chute, cover and legs are fixed boxes. The gate follows its path exactly and can't be pushed; on a real rig it is
a hand-lifted or solenoid card at the top of the chute, there only to start every trial the same way. The ball is a
rigid sphere with the station's rubber-like friction (1.2 static, 1.0 dynamic), no restitution and no rolling
resistance. A real rubber ball bounces a little in the basket and slows slightly on the table. The basket is one
rigid body of box colliders (0.8 static, 0.7 dynamic friction). A real basket is a moulded plastic box with a handle, about the same weight.
