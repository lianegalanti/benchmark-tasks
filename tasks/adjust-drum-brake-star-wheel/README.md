# Adjust a rear drum brake with a brake spoon until the drum turns with a light drag

A cropped rear drum-brake corner of a car stands on a bench stand on the table, its axle running across the table
18 cm up. The backing plate faces the robot's right arm, and the 9 in drum, with its five wheel studs, faces the left.
New shoes have just been fitted and the self-adjuster is run in, so the shoes sit 1.5 mm inside the drum. A brake
adjusting spoon lies on the table. The robot must:

1. put the spoon's blade through the adjusting slot in the backing plate (its rubber plug is out) onto the star
   wheel's teeth, behind the plate and out of sight;
2. lever the star wheel round a click at a time, pivoting the spoon on the slot's edge, while the other hand turns the
   drum to feel the drag;
3. stop when the drum turns with a light drag (shoes 0.10 to 0.40 mm from the drum), put the spoon down and let go.

Instruction: *"Adjust the rear brake so the drum turns with a light drag, then put the spoon down and let go of everything."*

Adjusting rear drum brakes after a shoe job is routine brake service for automotive service technicians and mechanics
(about 0.8 million US jobs, BLS Occupational Outlook Handbook).

What makes it hard:
- **Blind tool use through a slot.** The 12 mm blade must pass a 22 x 16 mm slot and land between two teeth of a
  30 mm star wheel 42 mm behind it.
- **A lever, not a push.** About 9.5 N at the blade's end turns the wheel one tooth. Pivoting on the slot's bottom
  edge, the hand needs about a fifth of that. Pushed straight, the soft wrist sags and the blade slips.
- **Stopping at the right click.** The window is 7 clicks (22 to 28) out of 30 before the shoes lock the drum. The
  self-adjuster's lever acts as a pawl, so the wheel can't be turned back: over-adjusting is final, as on a real car
  without a second tool to hold the lever off.
- **Feel with the other hand.** The drum's drag rises from 0.15 N m (free) to 0.31-0.8 N m in the window and over
  4 N m one click past it. Turning the drum shows where the adjuster is; counting clicks from the known start
  (as the demo does) is also a valid way to stop in the window.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor (adjuster clicks, pawl, drum drag), `success()` and `score()` |
| `geometry.py` | Dimensions of the corner, star wheel, slot and spoon; the adjuster model; the 2-D stroke geometry and box maths |
| `make_assets.py` | Writes `assets/corner.usda` (one articulation: fixed base, drum and star wheel joints) and `assets/spoon.usda` |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files (hooks yam_arena without modifying it) |
| `demo.py` | The scripted demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: mechanism and stroke geometry, every scoring level and violation, the demo's reach and clearances |

`scenekit.py` is adapted from the scene helper of this repository's `lock-before-loading` task folder (its license
applies).

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env` elsewhere):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Adjust the rear brake so the drum turns with a light drag, then put the spoon down and let go of everything." \
  --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/drum-brake-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is **open loop**. It works from the fixed, known layout and the arms' own joint readings only, and never reads
the spoon's, drum's or star wheel's state from the simulator. **It does not stop by feel: it makes a fixed number of
lever strokes** (25, one click each, 0.25 mm, in the middle of the 22-28 click window, so it tolerates three missed or
three doubled clicks). While it levers, the left hand turns the drum to and fro by a stud. The demo logs that arm's
largest joint tracking error on each turn (from its encoders) in `result.json` (`feel_left_err_rad`). The error rises
with the drum's drag, but the demo only reports it and does not use it.

1. **Spoon in (right arm).** A top grip across the handle picks the spoon up and brings it in line with the slot,
   blade along the axle, tip down 14 degrees. It slides the blade through the slot until its end sits under the lower
   of the two teeth either side of the gap facing the slot, 12.4 mm from the wheel's axis.
2. **Strokes (right arm).** The handle goes down about the slot's bottom edge until the blade is just past level
   (1.5 degrees up), which lifts the blade's end about 10 mm. It lifts the tooth above it up past the wheel's axis,
   past half a tooth, and the wheel clicks on to the next notch; mid-stroke the blade's end reaches the root of the
   tooth and slides up the hub. Lifting the lower tooth (rather than the upper one) makes the tooth push the blade
   into the slot, not out of the teeth, so the soft arm giving way doesn't let the blade slip off: the wheel turns
   29-34 degrees whether the blade's end is 1 mm deep or 4 mm short (a click is past 18 degrees, two would take 54).
   Then the blade levels, slides back 11 mm, clear of the teeth, the handle comes up and the blade goes in under the
   next tooth. That is 6 s per click.
3. **Drum (left arm).** The left hand closes on the top wheel stud, from outboard, along the axle. It turns the drum
   20 degrees each way, over and over, until the last stroke.
4. **Finish.** The blade comes out of the slot and the spoon goes back to its place on the table. The left hand brings
   the drum back and lets go of the stud. Both arms return to their ready pose.

The plan takes 3,488 steps (174 s). A slow integral term on the measured joint error (the arms' encoders) takes out
the soft wrists' sag.

**Measured on an L40S in Isaac Sim:** the demo scored 1.0 (25 clicks, 0.25 mm; success held at step 3,431). Dream-1 scored 0 in a 4,800-step trial (the run limit).

## Scoring

The monitor runs after every control step. The score at the end of the episode counts (`SCORE_OVER = "end"`).
The adjuster's position is counted in clicks of the star wheel. Each click takes 0.05 mm off the shoe-to-drum
clearance, which starts at 1.5 mm. Everything is judged in the corner's frame.

**0 if at any time:**
- the shoes are driven onto the drum: 30 clicks or more (clearance 0, the drum locked; the pawl keeps it there);
- the spoon leaves the table (its centre 5 cm below the table top).

Setting the spoon down to regrip it, dropping it onto the table and turning the drum are all allowed.

**Otherwise, at the end:**

| Score | Clicks | Clearance | Drum drag (hand torque) |
|---|---|---|---|
| 0 | 0 | 1.5 mm | 0.15 N m |
| 0.5 x clicks / 22 | 1-21 | 1.45-0.45 mm | 0.15-0.23 N m (spins freely) |
| 0.75 | 22-28, success conditions not all met for 1 s | 0.40-0.10 mm | 0.31-0.8 N m (light drag) |
| 0.25 | 29 | 0.05 mm | over 4 N m (dragging brake) |
| 1.0 | 22-28, success held 1 s | 0.40-0.10 mm | 0.31-0.8 N m |

**Success** needs all of these:
- 22 to 28 clicks, never 30;
- the spoon lying on the table: every part of it within 3 cm of the table top;
- every part of both grippers at least 2 cm from the spoon and from the drum and its studs, so nothing still holds them;
- the spoon still (under 1 cm/s) and the drum still (under 4 degrees/s);
- all of the above held for 1 s.

**Doing nothing, or refusing,** scores 0.

On a real corner, the clearance is read with a drum-and-shoe gauge (drum bore minus the shoes' diameter, halved), or
by counting clicks from a known start. The drag is read with a dial torque wrench on the hub nut.

## Simulation model

- **The corner** is one articulation fixed to the world:
  - the base: stand, axle stub, backing plate with slot and lip, both shoes, wheel cylinder, adjusting screw and
    socket;
  - the drum, on a revolute joint about the axle;
  - the star wheel (hub and ten box teeth), on a revolute joint about the screw, limited to the screw's 40-click
    travel.
- **The adjuster** (task.py, every control step):
  - The star wheel's drive pulls it toward its current notch, with its torque capped at 0.10 N m: that is the detent.
  - Turned past half a tooth in the tightening sense, the wheel counts a click, and the drive's target moves to the
    next notch, which the wheel snaps into.
  - Turned back more than 0.6 tooth behind its notch, the wheel is put back there: that is the pawl.
  - The shoes don't move in the model: 0.05 mm per click is invisible inside the drum. The clearance is computed from
    the clicks.
- **The drum's drag** is the drum joint's damping, set from the clearance. It is sized so that turning the drum at
  30 degrees/s takes the torque in the table above, and at zero clearance it is 2,000 N m s/rad, so the drum is
  locked. The drag-versus-clearance curve is an estimate: the linings' high spots start to brush the drum at about
  0.5 mm. It is not measured on a car.
- **Not simulated:** shoe-to-drum friction, the shoes' return springs, and the wheel getting stiffer as the shoes
  load it.
- **The spoon** is one rigid body: a grippy handle (friction 1.0) and a steel blade (0.4). It is held in a top grip
  across the handle.

## Real-world trials

- Use a cropped corner (a scrap axle end or a brake training board) bolted to a bench stand.
- The drum turns slowly by hand and nothing is under power, so there is no pinch hazard beyond the drum's own
  rotation.
- The spoon is light (90 g) and blunt.

## Runtime and seeds

A full 4,800-step trial with Dream-1 takes about 8 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
