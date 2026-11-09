# Swap the tray on a passing cart in one pass, without stopping it

A grey cart rides across the table on a guide track, 33 cm in front of the arms, from the right toward the left arm,
at a constant 6 cm/s. In the nest on its top (a funnel of four guide plates leaning outward at 55 degrees round a pocket
1.25 cm wider than a tray on every side at the bottom, 2.6 cm at the top) sits a full tray: a blue tray (11 x 11 cm, 3.5 cm deep) with a yellow handle rib across its middle and an aluminium
block in each of its two compartments. An identical empty tray stands on a grey stand in front of the left arm, and a
grey outfeed table stands in front of the right arm. The cart never stops: its centre passes the right arm 4.0 s into
the trial, the left arm 14.2 s in, and a red end line, taped 46 cm left of the midpoint, 16.75 s in. The robot must
swap the trays within that one pass: take the full tray off onto the outfeed table, blocks still in it, and put the
empty tray in the nest, seated and let go, before the cart reaches the line.

Tray and fixture exchange with carts, AGVs and pallet carriers is routine line-side work, and on a line that can't
stop the carrier the swap happens on the move. Unlike a static swap (take one out, put one in), the order is forced
and the time is shared: the nest must be emptied while the cart is in reach of one arm, and refilled while it is
still in reach of the other, each by matching the cart's motion while gripping or seating a tray. Dropping a tray,
spilling its blocks or missing the window each cost their half of the score.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the cart's motion, the monitor, `success()`, `score()` and the scripted demo |
| `geometry.py` | The tray's, cart's and nest's shapes |
| `make_tray.py` | Writes `assets/tray.usda` from `geometry.py` (plain text; no USD library needed) |
| `assets/tray.usda` | A tray: one rigid body of six boxes (floor, four walls, handle rib) |
| `motionkit.py` | Adds `DrivenBox` (a box whose pose and velocity follow a scripted path, set before every physics step; here the cart and its tilted guide plates, `belt_run`; a `rot_xyzw` field added here gives a driven box a fixed tilt) and `UsdBody` (a free rigid body from a USD file; here the trays) to yam_arena task files, and `on_step` / `on_reset` hooks with `arm_joints()` for a per-step monitor |
| `test_task.py` | Offline checks (no Isaac Sim): the cart's motion, the tray and nest shapes, every scoring rule, and that the demo is reachable, on time, clear of the cart and trays, and scores 1.0 when replayed kinematically |

`motionkit.py` is copied from this repository's `lid-cups-on-moving-belt` task (itself from `stack-arriving-blocks`,
with a `UsdBody` loader adapted from `scenekit.py` in `lock-before-loading`); `make_tray.py` is adapted from that
task's `make_lid.py`. It hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without
modifying yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and
`task_score` scorer. The cart's clock is the episode clock (`observation.state_time`): it stands still through the
reset and starts with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena
0.3.1). **Measured on an L40S:** the scripted demo ended with success after 355 steps
(17.8 s); Dream-1 scored 0 in a 600-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "A cart with a full tray in its nest is passing in front of you and won't stop. Take the full tray off the cart and set it on the grey outfeed table, and put the empty tray from the stand into the cart's nest before the cart reaches the red line." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/tray-swap-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "A cart with a full tray in its nest is passing in front of you and won't stop. Take the full tray off the cart and set it on the grey outfeed table, and put the empty tray from the stand into the cart's nest before the cart reaches the red line." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/tray-swap-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/tray-swap-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s (both halves done,
about 18 s in); otherwise it scores `score()` at the end of the trial (`SCORE_OVER = "end"`; the nest's half is
latched when the cart reaches the end line).

## Scoring

Two halves of 0.5, added.

| Half | Score | Condition | Check on a real table |
|---|---|---|---|
| Nest, judged once when the cart's centre reaches the end line | 0.5 | the empty tray seated in the nest: centre within 1.5 cm (horizontally) of the cart's centre, underside within 5 mm of the cart's top, level within 5 degrees, moving with the cart (within 2 cm/s), and let go (both grasp points 5 cm or more from the middle of its rib, 2.75 cm above the rim) | a photo from above as the cart crosses the line |
| | 0.25 | the empty tray let go and riding on the cart but not seated: centre over the cart's top, underside within 3 cm above it (perched on a guide plate, say) | the same photo |
| | 0 | anything else: not on the cart, still held, or placed only after the line | |
| Full tray, judged at the end of the trial | 0.5 | on the outfeed table: centre over its top, underside within 1 cm of it, level within 10 degrees, still, both blocks inside (centres between the walls, above the floor, below the rim), let go | a photo |
| | 0.25 | taken off the cart at some point (underside 4 cm or more above the cart's top), and at the end level within 10 degrees with both blocks inside, off the cart, but not set down on the outfeed table and let go (still held, or on the bare table) | video, against a 4 cm mark on the cart |
| | 0 | anything else: left on the cart, dropped and tipped, or a block spilled | |

1.0 needs both halves. Placing the empty tray after the line, or anything that happens to it afterwards, doesn't
change the nest's half; the full tray's half is its state at the end.

## The scripted demo

Each arm pinches a tray's rib from above, fingers closing along the track, pad tips 1.5 cm above the rim (2.5 cm of
rib between the pads). The gripper takes about 1 s to close or open (measured in Isaac Sim), so each arm holds 1.3 s
after its close command before lifting and 1.2 s after its open command before rising. Following the cart, each arm
first speeds up to the cart's 6 cm/s over 1 s (3 cm) and slows down over 1 s afterwards, so there are no jumps in
speed; then it follows the cart one waypoint per control step.

- Right arm: waits over the track, matches the cart from 2.25 s, and from 3.25 s (35 cm right of the midpoint) goes
  down to the full tray (0.8 s), closes (0.4 s), holds (1.3 s) and lifts it clear of the funnel (1.0 s); it then
  sets it on the outfeed table, opens (released at 10.85 s) and waits 1.2 s before rising.
- Left arm: takes the empty tray off the stand (same close and hold), waits with it beside the track until the right
  arm has left it, moves over the track, matches the cart from 10.1 s, and from 11.1 s (12 cm left of the midpoint)
  lowers the tray into the pocket (1.0 s), holds (0.3 s), opens 3 mm above the cart's top (0.4 s), keeps following
  the cart 1.2 s while the fingers open (done at 14.0 s, 2.75 s before the end line), then rises, slows down and
  moves back over the table, where it stays until 22 s (past the end line and the 1 s success hold).

The arms' grasp points never come within 20 cm of each other; open finger tips stay 1.5 cm or more above the full
tray's rib and the funnel, and a held tray's underside 2.5 cm or more above the funnel's top, except while following
the cart. `SWAP_DEBUG=1` prints, every 0.5 s, the cart's position, both trays' poses, tilts and offsets from the
cart's centre, and the blocks' positions; every step a gripper is within 6 cm of a tray's grip point, its TCP
relative to it, its pad gap and its offset from the cart along the track; and the nest's judgment.

## Runtime and seeds

- The cart reaches the end line 16.75 s into the trial; a successful trial ends about 18 s in. The run limit is 600
  steps (30 s at 20 Hz).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.

## Fidelity

The cart and its guide plates follow their path exactly, like a stiff servo-driven carrier on a guide track: the arms
can't push, stop or tip it (a real drive would fault if blocked). The cart floats 2 mm above the bed and past the
table's edge moves straight down. Each tray is one rigid body of six boxes (floor, four walls, handle rib) with sharp
edges and no draft, friction 0.8/0.7; the blocks have the station's rubber-like friction. The cart accelerates only
in its first 0.5 s (under 0.2 m/s²), too gently to slide a tray or a block.
