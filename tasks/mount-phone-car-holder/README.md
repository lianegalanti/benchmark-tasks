# Mount a phone in a spring-clamp car holder, holding its sprung arms open while the phone goes in

A car dashboard section stands on the table with a spring-clamp phone holder clipped to its air vent, cradle toward
the robot. The holder's two side arms are closed and empty, as left after the last phone was taken out. A phone leans
back against the dashboard's top front edge, to the right of the holder, screen toward the robot. The robot is told:

> Mount the phone in the car phone holder so that the holder's arms grip it, then let go of the phone.

Most drivers who navigate by phone use a dash or vent holder and mount the phone at the start of a trip. Spring-clamp
cradles with two geared side arms are among the most common kinds. Unlike setting a phone on a passive stand, the
skill here is working against a spring. The arms are sprung shut and nothing latches them open: the moment they are
let go, they snap back. The usual way to use the holder:

1. **Hold the arms open.** The two arms are geared together. Pulling one arm out (by the tab on its front, or by
   pushing the phone's edge against it) opens both. They stay open only while something holds them.
2. **Put the phone in.** While the arms are held open, bring the phone in between them, its back square to the back
   plate, onto the cradle's two feet.
3. **Let the arms close.** Let go of the arm: the springs close both arms on the phone's long edges, and the front
   lips go round its edges, not onto the screen.
4. **Let go of the phone.** The holder, not the robot, must hold the phone.

What makes it hard (physically, not in recognising anything):
- **Holding against a spring, with one hand busy.** The arms pull shut at every opening (about 3 N each when open,
  and both are felt at one tab through the gearing). With no catch, the robot either keeps them open with one hand
  while the other brings the phone in, or uses the phone's edge to push one arm open and swings the phone in
  against it. Let go too early and the arms close on the phone's screen or shut in front of it.
- **Seating.** The phone has to come in between the open arms (6-8 mm clear at each lip, depending on how far
  they are held) and stand on two feet that are 18 mm deep, square to the back plate. Tilted or held out from the
  plate when the arms close, it ends up gripped by the lips or crooked instead of clamped.
- **Ending matters.** Arms held open around the phone score nothing. A gripper still holding the phone or the arm
  scores 0.75, not 1. Dropping the phone scores 0.

## Real objects (full size)

- **Phone:** an iPhone 15-class phone, 147.6 x 71.6 x 7.8 mm, 171 g (aluminium frame, glass faces).
- **Holder:** a spring-clamp vent-mount cradle of the common type. Its arm span is 58 mm closed, with a stop at
  94 mm, so it takes phones 58-88 mm wide (the lips must clear them). The back plate is 64 x 102 mm. The two feet are 20 mm wide and 18 mm deep.
  The arms' silicone pads are 30 mm tall, centred 70 mm above the feet, with 3 mm lips at the front. Each arm has a
  pull tab on its front: a fin 24 mm deep, 4 mm thick and 30 mm tall, outboard of the phone's edges. The holder head
  weighs about 200 g. It has no catch or detent: the arms are always sprung shut.
- **Dashboard section:** a moulded 60 x 15 x 9 cm section (dark grey ABS) with an air-vent grille on its face. It is
  clamped to the table, and the holder's clip grips its vent louvre.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the holder's link boxes and springs, the phone's start, box maths |
| `make_assets.py` | Writes `assets/holder.usda` (the holder: a fixed-base articulation) and `assets/phone.usda` |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files. It hooks yam_arena without modifying it. Adapted from the scene helper in this repository's `lock-before-loading` task folder (its license applies). |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: the mechanism (no catch), every scoring level and violation, the demo's reach and clearances |

## Run a trial

From this folder, in a shell with the yam_arena environment active. New shells on the walkthrough's Brev instance
have it; elsewhere, `source <ARENA_DIR>/yam.env`.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount the phone in the car phone holder so that the holder's arms grip it, then let go of the phone." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/phone-holder-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is planned from the fixed, known layout and uses only each arm's own joint encoders. It never reads the
phone's pose or the holder's joints from the simulator. The monitor's state is only printed, as a log. One hand holds
the arms open while the other puts the phone in:

1. **Taking the phone (right arm).** The right arm comes down from above with its fingers open across the phone's
   width. It closes on the phone's long edges 3 cm below its top and lifts it straight up, off the dashboard's edge.
   It turns the phone upright and waits above and to the right of the holder.
2. **Holding the arms open (left arm, meanwhile).** Pointing forward and 15 degrees down, the left arm brings its
   fingers in on either side of the left arm's pull tab and pinches it. It pulls the tab out 16 mm; the gearing opens
   the right arm with it (pads 90 mm apart, lips 5 mm clear of the phone's edges). It keeps holding them there.
3. **Mounting.** The right arm brings the phone over the cradle, its bottom 9 cm up, and lowers it straight down
   between the open arms, its back 2.5 mm in front of the back plate, to 1 mm above the feet. It then presses the
   phone back flat onto the plate (aiming 2.5 mm past it, to take up the phone's give in the fingers), so its screen
   sits behind the arms' lips, and holds it there.
4. **Closing.** The left arm eases the tab back in to 1.5 mm short of the phone's edge, opens its fingers and backs
   away; the springs close both arms on the phone's long edges. The right arm then opens its fingers and slides them
   up off the top of the phone. Both arms go back to their ready pose.

A slow integral term on the measured joint error (the arms' encoders) takes out the sag from pulling against the
springs and carrying the phone. The plan takes about 565 steps (28 s).

Measured on an L40S, the demo reaches a score of 1.0 in 513 steps (26 s). Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step. It latches violations for the whole trial. Otherwise the stage at the
end of the episode counts (`SCORE_OVER = "end"`): clamping the phone and then pulling it out, or pulling the arms open
again, keeps nothing. Everything about the cradle is measured in the holder's own frame.

**0 if at any time:**
- the phone falls faster than 0.5 m/s while no gripper is holding it (no finger within 1 cm of it), a free fall of
  about 1.3 cm;
- the phone's centre goes more than 2 cm below the table top (it left the table).

**Otherwise, the stage at the end. Nothing counts unless the holder's arms are closed on the phone:**

| Score | Stage |
|---|---|
| 0 | none of the below: for example the arms closed and empty, closed on the phone's screen (lips on the glass), or held open around the phone by a gripper, however well the phone stands on the feet |
| 0.5 | **Gripped:** both arms stopped on the phone's long edges. Each arm is within 2 mm of the travel at which its pad meets the phone's edge (so the phone, not the arm's closed stop or its lip on the glass, stopped it), and its pad is within 1 mm of the phone (touching it). The phone is centred within 4 mm and upright within 20 degrees. Where it sits on the cradle is not checked |
| 0.75 | **Clamped:** gripped, and seated: the phone centred within 2 mm, upright and square within 6 degrees, its lowest corner from 2 mm below to 10 mm above the feet's tops (the pads may hold it a little above them), its back within 6 mm of the plate (inside the lips) |
| 1.0 | **Success, held 1 s:** clamped, every part of both grippers at least 2 cm from the phone and from the holder's arms (pads, lips and pull tabs: let go of both), the phone still (under 1 cm/s and 0.1 rad/s) and the arms still (under 1 cm/s), no violation |

- **Doing nothing, or refusing,** scores 0.
- **Holding the phone in the clamp, or holding an arm on it by its tab,** scores 0.75. Only the holder's springs
  carrying it score 1.

On a real table, with a ruler and a photo:
- the arms' pads touch the phone's two long edges, and their gap equals the phone's width;
- the phone's bottom edge is on, or less than 1 cm above, both feet and its back lies flat on the back plate;
- nothing touches the phone or the holder's arms, and the phone stays put for a second.

How the simulation judges these, once per control step:
- The phone's pose and the holder's joint positions come from the simulator.
- The grippers are boxes placed from each arm's encoders: each finger's pad and root, and the carriage.
- A gripper counts as holding the phone (for the drop rule) while any of its fingers is within 1 cm of the phone,
  whichever way it grips it.

## Real-world trials

- Use a dummy phone (an aluminium or acrylic block 147.6 x 71.6 x 7.8 mm, weighted to 171 g), or a real phone in a
  thin case, so a slip costs nothing.
- Clamp the dashboard section to the table and check the holder's clip is tight on its vent louvre.
- Everything happens at table height: a dropped phone falls at most about 15 cm, onto the table.
- The holder's springs are weak (about 1.2-3.4 N per arm; see below). The only pinch point is between the arms and
  their tabs, as when a person uses the holder.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

Simulation details:
- 240 Hz physics, 20 Hz control.
- The holder is a fixed-base articulation built from boxes. Each arm is a prismatic joint with a spring (120 N/m,
  1.2 N preload when closed, so 2.0 N on the phone and 3.4 N fully open) that pulls it shut at every travel; nothing
  else acts on the arms, so there is no catch or detent. The right arm follows the left through a mimic constraint,
  which stands in for the rack and pinion.
- The spring forces are estimates, not measured on a product, and are on the soft side of real holders (whose arms
  can take several newtons to pull open). The pads' friction (1.0) still lets them carry the phone on their own.
- Friction: silicone pads 1.0, phone 0.5, the holder's plastic 0.4.
- The dashboard section is a fixed box.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
