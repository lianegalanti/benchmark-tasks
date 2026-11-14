# Assemble a food processor so it can run: lock the bowl on the base, lock the lid, close the interlock

A 9-cup food processor's motor base stands on the table ahead of the robot. Its empty work bowl stands on the table
in front of the base, handle on its right, and its lid (feed tube up) stands to the right. The robot is told:

> Assemble the food processor so it is ready to run: lock the work bowl onto the base and the lid onto the bowl, then
> let go of both.

Food processors are in most kitchens, and they come apart for washing after every use, so the person using one
assembles it every time: bowl onto the motor base, turned until it locks, then the lid, turned until it locks. The
motor will not start unless both are fully home, because a safety interlock in the bowl's handle is only pushed
closed by the lid's tab when the lid is turned all the way. A half-turned bowl or lid is the usual reason a
processor "won't start".

What makes it hard (physically, not in recognising anything):
- **Two stacked bayonets, each with a blind drop and a turn.** The bowl's three lugs are hidden under its foot. They
  must pass down between the three ledges on the base's coupling boss (2.5 mm play to the boss, 6 mm skirt
  clearance), with the bowl's centre tube over the drive shaft (2 mm play), and only then can the bowl be turned 28
  degrees clockwise so the lugs slide under the ledges to their stops. The lid's four lugs then drop between the four
  ledges on the outside of the bowl's rim (2.5 mm play to the wall, 5 mm skirt clearance) and it too is turned 28 degrees. A lug lowered onto
  a ledge leaves the part sitting 15-18 mm high. Locked, the lid's lugs sit 3.5 mm under the rim's ledges.
- **A large, light, round part.** The bowl is 18 cm across and 0.55 kg; a 9.4 cm gripper can take it only by its
  handle (cantilevered 13 cm from its centre) or by pinching its 3 mm wall, which holds little. Carried by one, it
  swings. With one hand on the handle and the other on the rim opposite, it stays square while it is lifted 19 cm
  over the shaft, lowered onto it and turned, both hands travelling round the shaft together (6 cm at the handle).
- **Turning against detents.** Each bayonet clicks into a sprung detent at its stop (3 N plungers), and the last
  degrees of the lid's turn also push the interlock rod down (1.0 N, rising to 1.4 N): the lid must be held down and
  turned by its feed tube, 6 cm off its axis, through this resistance. Stopping short leaves the rod up: the processor will not run.
- **Both locks, and the right way round.** The motor's switch is under the handle only when the bowl is locked in
  the one position (of three) that puts the handle over it, and the rod is down only when the lid is turned home.
  The scorer needs both, whatever the order they were done in.

How this differs from a single bayonet cap pushed and turned into a socket: here there are two bayonets stacked on
one axis, on full-size kitchen parts (an 18 cm bowl, a 21 cm lid), the second locks onto the first, and the goal is
a third mechanism, the interlock, which closes only when both are home. Gripping a large, light, round bowl by its
rim with two hands and carrying it over a shaft is a different physical problem from turning a small cap by its
head.

## Real objects (full size)

- **Motor base:** white ABS housing 20 x 17 x 9.5 cm (4.2 kg, rubber feet; fixed in the simulation), two control
  paddles on its front. On top: a grey coupling boss 110 mm across and 20 mm tall with three 25-degree ledges (9 mm
  thick, 12 mm deep) round its top edge and a stop post at each ledge's clockwise end; a steel drive shaft hub 25 mm
  across standing 50 mm above the boss, with a 10 mm conical tip.
- **Work bowl:** clear polycarbonate, 180 mm outside diameter, 3 mm wall, 125 mm tall (about 2.8 l to the brim,
  2.1 l working capacity, "9 cup"), 0.55 kg. A centre tube (29 mm bore, 105 mm tall) slides over the drive shaft. Its
  foot is a 148 mm skirt, 17.5 mm deep, with three 22-degree lugs inside it. A grey handle (30 mm wide, 24 mm deep)
  on its right front when locked; a red 6 mm switch rod runs down the inside of the handle, its top 9 mm proud of the
  handle's top, sprung up (1.0 N preload, 60 N/m; 1.4 N pressed 7 mm), 114.5 mm from the bowl's axis (5.5 mm outside the
  lid's skirt). Four 25-degree ledges round the outside of the rim, 6.5-15.5 mm below its
  top, each with a stop at its clockwise end and an 8 mm lead-in ramp at its other end (where the lid's lug enters).
- **Lid:** clear polycarbonate, 212 mm across, its 24 mm skirt fitting over the rim with four 22-degree lugs inside
  it; 0.25 kg. A feed tube 40 x 70 mm outside (3 mm walls) rises 10 cm from the lid on its right, its centre 58 mm off
  the axis. A grey interlock tab sticks out of the skirt over the handle: its underside is flat, with a 30-degree
  ramp on its clockwise (leading) side.
- **Detents:** in one bowl lug, a sprung plunger (a 4 mm bar turned 45 degrees, 3 N preload, 300 N/m) stands 1.5 mm into
  the ledge above it and clicks into a 6 mm notch in that ledge at the stop. In one rim ledge, a plunger (an 8 mm bar turned 45
  degrees, 3 N preload, 300 N/m) does the same into a notch in the lid lug below it.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the layout, the parts as lists of boxes, the mechanism's numbers, outlines for the let-go check |
| `fpkit.py` | Rigid parts from boxes, cylinders and cones, with sprung sliders (an articulation with a floating root), for yam_arena task files. Adapted from a helper in another task folder of this repository (its license applies). It hooks yam_arena without modifying it. |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: sizes, both bayonets' plays, the detents and interlock, every scoring level and violation, the demo's reach, limits and clearances |

## Run a trial

From this folder, in a shell with the yam_arena environment active. New shells on the walkthrough's Brev instance
have it; elsewhere, `source <ARENA_DIR>/yam.env`.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Assemble the food processor so it is ready to run: lock the work bowl onto the base and the lid onto the bowl, then let go of both." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/food-processor-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is open-loop from the fixed, known layout, corrected only by each arm's own joint encoders. It never reads
the bowl's or the lid's pose, or their joints, from the simulator (the monitor's state is only printed, as a log).

1. **Bowl (both arms).** The right hand takes the handle's upright bar across its 30 mm width (approach tilted 60
   degrees in toward the bowl); the left pinches the rim directly opposite (approach tilted 45 degrees along the
   rim). The two grips lie on a diameter through the bowl's centre of mass. Together they lift the bowl 19 cm, carry
   it over the base and lower its tube over the shaft until its floor rests on the boss. They turn it clockwise about
   the shaft, aiming 8 degrees past the stop (the arms lag the detent). Both let go and back away.
2. **Lid (right arm).** The right arm takes the lid by its feed tube (fingers across its 40 mm width, approach tilted
   50 degrees), lifts it over the bowl, turns it 3 degrees further from locked, and lowers its skirt over the rim. It
   lets go for a moment so the lid settles flat under its own weight, takes the tube again, and turns the lid
   clockwise while holding it 3 mm down on the rim, aiming 15 degrees past the stop: the lugs slide under the ledges,
   the detent clicks and the tab rides over the rod and pushes it down 7 mm. It lets go and slides its fingers off
   the tube.
3. Both arms go back to their ready pose.

The left hand does not brace the bowl's handle during the lid turn: with the lid on, the handle (front right) is out
of the left arm's reach at a usable grip angle. The bowl's own detent and stops hold it against the lid's turn, which
pushes it the same way (clockwise).

A slow integral term on the measured joint error (the arms' encoders) takes out the sag from carrying and turning. The
plan takes 1,068 steps (53 s).

Why these grips: the gripper is position-controlled toward closed, so a 3 mm wall stops each finger only 1.5 mm short
and is held with about 7.5 N, while a 30-40 mm handle or feed tube gets the full 50 N. Two opposite rim pinches let
the bowl swing about the line between them and slip (seen in Isaac Sim); the handle grip resists that swing.

Measured on an L40S, the demo reaches 1.0 (success) in 1,029 steps (51 s): the bowl locked 0.96 degrees from its
stop with its detent clicked in, the lid locked 1.28 degrees from its stop, the switch rod pushed down 5.8 mm, both
grippers let go. Dream-1 scored 0 in a 3,600-step trial (the run limit).

## Scoring

The monitor runs after every control step. It latches violations for the whole trial. Otherwise the stage at the end
of the episode counts (`SCORE_OVER = "end"`). The bowl is judged in the base's frame and the lid in the bowl's.

**0 if at any time:**
- the bowl or the lid falls off the table (its origin more than 5 cm below the table top);
- the bowl or the lid is forced more than 3 degrees past its stops while seated (a lug or stop would break);
- the switch rod or a detent plunger is forced 2 mm past the end of its travel.

Lifting, tilting, setting a part down on the table, dropping it on the table, regripping and moving it round are
all allowed.

**Otherwise, the stage at the end:**

| Score | Stage |
|---|---|
| 0 | none of the below |
| 0.25 | **Bowl seated:** its axis within 4 mm of the shaft's, tilted under 3 degrees, its floor from 2 mm below to 3 mm above the boss's top (its lugs down beside the boss, under the ledges' level) |
| 0.5 | **Bowl locked:** seated and turned to within 3 degrees of its stops (-2 to +2.5 degrees from locked; the stops are at -0.5): every lug wholly under its ledge |
| 0.75 | **Lid seated** on the locked bowl: its axis within 4 mm of the bowl's, tilted under 3 degrees relative to it, its top from 2 mm below to 3 mm above the rim |
| 1.0 | **Success, held 1 s:** the bowl locked with its handle over the base's switch, the lid locked (within 3 degrees of its stops), the switch rod pushed down at least 5 mm, every part of both grippers at least 2 cm from the bowl (its rim ledges, handle and rod included) and the lid (feed tube and tab included), both parts still (under 1 cm/s and 0.1 rad/s), no violation |

- **Doing nothing, or refusing,** scores 0.
- **Holding the lid** in its locked place scores 0.75: only the latches holding it scores 1.
- The bowl fits the boss three ways round and the lid the rim four ways; any of them counts as locked for 0.5 and
  0.75, but only the one with the handle over the switch (and the tab over the rod) can score 1.

On a real table, with a ruler, a pen mark and a photo:
- turn marks: a mark on the base's boss and on the bowl's foot, and on the rim and the lid's skirt, line up within 3
  degrees (5 mm at the bowl's rim) when locked;
- the bowl's floor sits flat on the boss and the lid flat on the rim (no gap over 3 mm);
- the rod's pen mark has gone 5 mm into the handle (or: the processor's pulse button runs the motor);
- nothing touches the processor, and it stays put for a second.

How the simulation judges these, once per control step:
- The base's, bowl's and lid's poses and the bowl's joint positions (rod and plungers) come from the simulator.
- The grippers are boxes placed from each arm's encoders: each finger's pad and root, and the carriage. Distances
  are to the bowl's outline (a cylinder to its rim ledges, from its foot to its rim, plus the handle and rod) and
  the lid's (a cylinder to its skirt, plus the feed tube and the tab).

## Real-world trials

- Use a real 9-cup food processor, unplugged (the interlock can be checked by its continuity across the base's
  switch, or by plugging it in only after the robot has let go and finished).
- No blade or disc is fitted: the bowl is empty.
- Everything happens at table height: a dropped bowl or lid falls at most about 35 cm onto the table, which a
  polycarbonate bowl survives.

## Runtime and seeds

A full 3,600-step trial with Dream-1 takes about 6 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

Simulation details:
- 240 Hz physics, 20 Hz control.
- The base is a fixed body: boxes for the housing, ledges and stops, cylinders for the boss and shaft, a cone for
  the shaft's tip. The bowl is one articulation with a floating root (139 boxes, including the 32-panel
  wall) and three sprung slider links (rod and two detent plungers) that collide with the base and the lid but not
  with the bowl itself. The lid is one rigid body of boxes.
- Round walls and rings are panels (32 round the bowl's wall and foot, 40 round the lid's skirt); each ledge is
  several boxes along its arc. The panels' corners stick out at most 0.4 mm past the nominal radius, which the
  stated plays allow for.
- The bowl's slight taper and the processor's blade, discs and food pusher are left out (the feed tube is open).
- Spring forces are estimates, not measured on a product. The switch rod is light (1.4 N pressed): in the simulation
  a 3-4 N rod lifted the lid within its lugs' 3.5 mm play once the hand let go, and the rod rose again. The
  detents are 3 N. Friction: polycarbonate 0.4, the gripper's pads their own.

## Origin

`fpkit.py` is adapted from another task folder in this repository (see Files). Everything else was written for this
task.
