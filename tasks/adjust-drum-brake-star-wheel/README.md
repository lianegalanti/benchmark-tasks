# Adjust a rear drum brake with a brake spoon until the drum turns with a light drag

A cropped rear drum-brake corner of a car stands on a bench stand on the table, its axle running across the table
18 cm up. It is modelled on a **1979-1993 Ford Mustang 9 in duo-servo rear drum brake** (9.000 in drum bore, 1.75 in
wide linings; see "Calibration"). The backing plate faces the robot's right arm, and the drum, with its five wheel
studs, faces the left. New shoes have just been fitted and the self-adjuster is run in, so the linings sit about 1 mm
(0.975 mm) inside the drum. A brake adjusting spoon lies on the table. The robot must:

1. put the spoon's blade through the adjusting slot in the backing plate (its rubber plug is out) onto the star
   wheel's teeth, behind the plate and out of sight;
2. lever the star wheel round a click at a time, pivoting the spoon on the slot's edge, while the other hand turns the
   drum to feel the drag;
3. stop when the drum turns with a light drag (the linings pressed on the drum so that turning it takes 1.0 to
   3.9 N m, a 17-69 N pull at a wheel stud), put the spoon down and let go.

Instruction: *"Adjust the rear brake so the drum turns with a light drag, then put the spoon down and let go of everything."*

Adjusting rear drum brakes after a shoe job is routine brake service for automotive service technicians and mechanics
(about 0.8 million US jobs, BLS Occupational Outlook Handbook).

What makes it hard:
- **Blind tool use through a slot.** The 12 mm blade must pass a 22 x 16 mm slot and land between two teeth of a
  30 mm star wheel 42 mm behind it.
- **A lever, not a push.** About 9.5 N at the blade's end turns the wheel one tooth. Pivoting on the slot's bottom
  edge, the hand needs about a fifth of that. Pushed straight, the soft wrist sags and the blade slips.
- **Stopping at the right click.** The linings reach the drum at click 20; from there each click presses them on
  harder and adds 0.43 N m. Light drag is clicks 22 to 28 of a 40-click screw. The self-adjuster's lever acts as a
  pawl, so the wheel can't be turned back: over-adjusting is final, as on a real car without a second tool to hold
  the lever off.
- **Feel with the other hand.** The drum turns at 0.15 N m (the hub alone) until the linings touch, then the drag
  rises click by click: 1.2 N m at 22, 3.7 N m at 28, 5 N m at 31. Turning the drum shows where the adjuster is;
  counting clicks from the known start (as the demo does) is also a valid way to stop in the window.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor (adjuster clicks, pawl, shoe forces, drum torque), `success()` and `score()` |
| `geometry.py` | Dimensions of the corner, star wheel, slot and spoon; the adjuster and brake calibration; the 2-D stroke geometry and box maths |
| `make_assets.py` | Writes `assets/corner.usda` (one articulation: fixed base; drum, two shoes and star wheel joints) and `assets/spoon.usda` |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files (hooks yam_arena without modifying it) |
| `demo.py` | The scripted demo; `plan()` replays it without Isaac Sim for the tests; `calibrate()` is the torque-probe sweep below |
| `calibration.json` | That sweep's output from Isaac Sim (clicks, measured clearance, shoe forces, torque, probe breakaway) |
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
python demo.py task.py cal_out --calibrate   # the click-to-torque sweep: writes calibration.json
```

The demo is **open loop**. It works from the fixed, known layout and the arms' own joint readings only, and never reads
the spoon's, drum's or star wheel's state from the simulator. **It does not stop by feel: it makes a fixed number of
lever strokes** (25, one click each: 2.5 N m, in the middle of the 22-28 click window, so it tolerates three missed
or three doubled clicks). Until 22 strokes are made, the left hand turns the drum to and fro by a stud; then it holds
the drum still. The demo logs that arm's largest joint tracking error on each turn (from its encoders) in
`result.json` (`feel_left_err_rad`). The error rises with the drum's drag, but the demo only reports it and does not
use it.

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
   20 degrees each way, over and over, until 22 strokes are made, then holds it.
4. **Finish.** The blade comes out of the slot and the spoon goes back to its place on the table. The left hand brings
   the drum back and lets go of the stud. Both arms return to their ready pose.

The plan takes 3,488 steps (174 s). A slow integral term on the measured joint error (the arms' encoders) takes out
the soft wrists' sag.

**Measured on an L40S in Isaac Sim:** the demo scored 1.0 (25 clicks; the linings on the drum with 25.2 N each, so
turning it takes 2.46 N m; success held at step 3,430, 171.5 s). Dream-1 scored 0 in a 4,800-step trial (the run limit).

## Scoring

The monitor runs after every control step. The score at the end of the episode counts (`SCORE_OVER = "end"`).
Light drag is judged by the **drum's turning torque**, measured in the sim: each shoe's normal force on the drum is
read from its joint (the stretch of its spring, commanded minus measured travel: what the drum pushes back with), and
the torque is 0.40 (lining friction) x 114.3 mm (drum radius) x the two forces, plus 0.15 N m for the hub. The star
wheel's clicks only drive the adjuster; the score doesn't count them. Everything is judged in the corner's frame.

**0 if at any time:**
- the shoes are driven onto the drum: over 4.8 N m to turn it for 0.5 s (31 clicks or more; the pawl keeps it there);
- the spoon leaves the table (its centre 5 cm below the table top).

Setting the spoon down to regrip it, dropping it onto the table and turning the drum are all allowed.

**Otherwise, at the end:**

| Score | Drum turning torque | (clicks, from the sim) |
|---|---|---|
| 0 | 0.15 N m, shoes not moved | 0 |
| 0.5 x progress (under 0.5) | under 1.0 N m: spins freely or barely touching | 1-21 |
| 0.75 | 1.0-3.9 N m (light drag), success conditions not all met for 1 s | 22-28 |
| 0.25 | 3.9-4.8 N m (a dragging brake) | 29-30 |
| 0 | over 4.8 N m at the end (driven on), even if for less than 0.5 s | 31+ |
| 1.0 | 1.0-3.9 N m, success held 1 s | 22-28 |

Progress is how far the shoes have been set toward light drag: their measured outward travel, plus (once they touch
the drum) the spring stretch that the measured drag shows, over the 1.075 mm that gives 1.0 N m.

**Success** needs all of these:
- the drum's turning torque 1.0 to 3.9 N m, never over 4.8 N m for 0.5 s;
- the spoon lying on the table: every part of it within 3 cm of the table top;
- every part of both grippers at least 2 cm from the spoon and from the drum and its studs, so nothing still holds them;
- the spoon still (under 1 cm/s) and the drum still (under 4 degrees/s);
- all of the above held for 1 s.

**Doing nothing, or refusing,** scores 0.

On a real corner, the turning torque is read with a spring balance hooked on a wheel stud (newtons x 0.05715 m), as
service manuals measure brake drag, or with a dial torque wrench on the hub nut.

## Calibration

The task is calibrated against published data for a specified real assembly, the Ford Mustang (1979-1993) 9 in
duo-servo rear drum brake. No source we found gives a drum's turning torque across adjuster clicks, so the window
is built from the published pieces below, and the sim's click-to-torque mapping is measured (table after the list).

| Quantity | Value used | Source |
|---|---|---|
| Drum bore, wear limit, runout | 228.6 mm (9.000 in); 9.060 in max; 0.007 in max runout | Ford 1993 Mustang Workshop Manual, Section 06-02, Specifications [1] |
| Linings | 1.75 in wide; primary 6.10 in, secondary 8.65 in long | same [1] |
| Self-adjuster | one tooth of the star wheel per adjuster stroke; shoes set to a drum gauge | Ford manual, Section 06-02 [2], [3] |
| Procedure | turn the star wheel to "heavy drag", then back "until a very light drag" is obtained; start clearance 0.05 in (1.27 mm). A generic drum-brake procedure, not Ford's (Ford sets the shoes with a gauge [2]) | US Army TM 9-2330-357-14&P [4] |
| Running clearance | "about 10 to 20 thousands of an inch" (0.25-0.5 mm) | US Patent 4,401,195 (Kelsey-Hayes adjuster) [5] |
| Lining friction | 0.40: edge code "FF" (F = 0.35-0.45) | SAE J866 code letters [6], [7] |
| Most drag accepted | "Standard value: 69 N or less", front disc brake, spring balance on a hub bolt | Mitsubishi workshop manual, brake drag check [8] |
| A drag worth noticing | "1 Nm drag torque at each wheel can cause up to 2 g CO2 per km" | EuroBrake 2014 paper EB2014-DF-017 [9] |
| Wheel studs | five on a 114.3 mm circle (57.15 mm radius) | Ford 5 x 4.5 in bolt pattern |

**The light-drag window: 1.0 to 3.9 N m at the drum.**
- The upper bound is the 69 N drag limit [8], read as a pull at a wheel stud: 69 N x 57.15 mm = 3.9 N m. More than
  that is a dragging brake. Caveat: the 69 N limit is a different maker's front disc brake, measured with a spring
  balance at its own hub-bolt radius, hub drag included. Ford publishes no drag figure (its manual sets the shoes
  with a gauge [2]), so this is used as the closest published limit, applied at the Mustang's 57.15 mm stud radius.
- The lower bound, 1.0 N m (17.5 N at a stud), is the least drag clearly felt above the free drum (0.15 N m, the
  hub's drag in the sim): over six times it, so the shoes are clearly on the drum. EuroBrake [9] confirms that 1 N m
  is a measurable drag.
- The demo's 25 clicks give 2.5 N m, in the middle.

**From clicks to torque.** Each click moves each lining 0.05 mm toward the drum. From 0.975 mm at the start, they
touch at click 20 (about 1 mm, between the manual's 1.27 mm start [4] and the running clearance [5]). Past that,
the drum stops the shoe and the adjuster's further travel stretches the shoe's compliance, K = 93 N/mm per shoe.
That force presses the lining on the drum. K stands for the shoe, its lining and its return springs together. It is
a model parameter, set so that the window spans 7 clicks; it is not measured. The score reads torque, not clicks:
a real shoe with a different stiffness changes only how many clicks the window spans, not what counts as light drag.
On a real corner the torque is checked the same way, with a spring balance on a wheel stud.

**Measured in Isaac Sim** (`demo.py` `calibrate()`, `calibration.json`). With the arms parked, the adjuster is set
a click at a time and the shoes settle. Then the shoes' clearance and forces are read from their joints, and a torque
probe on the drum joint ramps at 1 N m/s until the drum turns (2 deg/s). With the shoes clear, the probe reads
0.1 N m from the ramp's lag.

| Clicks | Clearance per shoe (mm, measured) | Normal force per shoe (N) | Turning torque, scorer (N m) | Probe breakaway (N m) | Score band |
|---|---|---|---|---|---|
| 0 | 0.975 | 0 | 0.15 | 0.10 | partial |
| 10 | 0.475 | 0 | 0.15 | 0.10 | partial |
| 19 | 0.025 | 0 | 0.15 | 0.10 | partial |
| 20 | -0.008 (touching) | 1.6 | 0.30 | 0.25 | partial |
| 21 | -0.005 | 6.6 | 0.75 | 0.70 | partial |
| 22 | -0.003 | 11.4 | 1.19 | 1.15 | light drag |
| 23 | -0.002 | 16.1 | 1.62 | 1.60 | light drag |
| 25 | -0.002 | 25.4 | 2.47 | 2.40 | light drag |
| 27 | -0.005 | 34.5 | 3.30 | 3.20 | light drag |
| 28 | -0.008 | 38.8 | 3.69 | 3.55 | light drag |
| 29 | -0.012 | 43.1 | 4.09 | 4.15 | dragging (0.25) |
| 30 | -0.009 | 48.0 | 4.54 | 4.60 | dragging (0.25) |
| 31 | -0.006 | 52.9 | 4.99 | 5.00 | driven on (0) |
| 32 | -0.005 | 57.6 | 5.42 | 5.40 | driven on (0) |

The probe, which turns the drum against the linings' contact friction in the physics engine, agrees with the
scorer's force-based torque to within 0.15 N m. The negative clearances are the contact's few microns of
penetration.

Sources:
1. Ford, 1993 Mustang Workshop Manual, Section 06-02 Brakes, Rear Drum, Specifications:
   https://www.fordservicecontent.dealerconnection.com/pubs/content/~WSPB/~MUS~LEN/19/SPB62012.HTM
2. Same, Adjustments: https://www.fordservicecontent.dealerconnection.com/pubs/content/~WSPB/~MUS~LEN/19/SPB62011.HTM
3. Same, self-adjuster assembly: https://www.fordservicecontent.dealerconnection.com/pubs/content/~WSPB/~MUS~LEN/19/SPB62004.HTM
4. TM 9-2330-357-14&P, brake shoe adjustment:
   https://operatormanuals.tpub.com/TM-9-2330-357-14-P/TM-9-2330-357-14-P0115.htm
5. US 4,401,195, Automatic brake adjuster: https://patents.google.com/patent/US4401195
6. Brake linings, fact and friction (SAE J661/J866 edge codes):
   https://www.truckpartsandservice.com/home/article/14973450/the-break-shop-brake-liningsfact-and-friction
7. Edge code ranges (E = 0.25-0.35, F = 0.35-0.45): https://www.forabodiesonly.com/mopar/posts/1970026837/printfriendly
8. Mitsubishi Motors workshop manual 35C, brake drag force check:
   https://kitchen.manualsonline.com/manuals/mfg/mitsubishi_1/35c.html?p=20
9. EuroBrake 2014, EB2014-DF-017: https://go.fisita.com/store/papers/EuroBrake2014/EB2014-DF-017

## Simulation model

- **The corner** is one articulation fixed to the world, with self-collisions on (so the linings touch the drum):
  - the base: stand, axle stub, backing plate with slot and lip, wheel cylinder, anchor pin, adjusting screw and
    socket;
  - the drum, on a revolute joint about the axle. Its braking band is 240 flat segments, round to 0.01 mm. The hub's
    drag is the joint's damping: 0.15 N m at 30 degrees/s;
  - the two shoes, each on a prismatic joint moving outward across the drum (the primary toward +x, the secondary
    toward -x). Lining, table and web are visual. Each lining touches the drum through three 5 mm contact spheres on
    its face, with friction 0.40 (combine mode "min", so the lining's friction rules the pair);
  - the star wheel (hub and ten box teeth), on a revolute joint about the screw, limited to the screw's 40-click
    travel.
- **The adjuster** (task.py, every control step):
  - The star wheel's drive pulls it toward its current notch, with its torque capped at 0.10 N m: that is the detent.
  - Turned past half a tooth in the tightening sense, the wheel counts a click, and the drive's target moves to the
    next notch, which the wheel snaps into.
  - Turned back more than 0.6 tooth behind its notch, the wheel is put back there: that is the pawl.
  - Each shoe's drive is a spring (93 N/mm, damped) toward 0.05 mm x clicks of outward travel. Short of the drum, the
    shoe gets there; on the drum, the contact stops it and the spring's stretch is its normal force.
- **The drum's drag** is the physics engine's contact friction between the linings and the drum: turning the drum
  takes the force's torque (see the probe column above). The scorer reads it from the shoes' measured joint
  positions.
- **Simplified:** the shoes slide straight out instead of pivoting on the anchor pin; the drum is perfectly round
  (the 0.007 in runout allowed [1] is not modelled); the wheel's 10 coarse teeth and 0.05 mm per click are chosen
  for the spoon's stroke (Ford doesn't publish the tooth count or the screw's thread); the wheel doesn't get stiffer
  as the shoes load it (at 40 N of screw thrust it would add about 0.03 N m to the 0.10 N m click).
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
