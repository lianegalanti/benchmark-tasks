# Mount a phone in a one-touch car holder whose sprung arms are triggered by the phone

A car dashboard section stands on the table with a one-touch phone holder clipped to its air vent, cradle toward the
robot. The holder's two side arms are closed and empty, as left after the last phone was taken out. A phone leans
back against the dashboard's top front edge, to the right of the holder, screen toward the robot. The robot is told:

> Mount the phone in the car phone holder so that the holder's arms grip it, then let go of the phone.

Most drivers who navigate by phone use a dash or vent holder and mount the phone at the start of a trip. The
one-touch (auto-clamping) cradle is the most common kind. Unlike setting a phone on a passive stand, the skill here
is working a sprung mechanism: cocking the arms until the catch holds them, then firing the trigger with the phone
centred so the arms close on its edges. The usual way to use the holder:

1. **Cock the arms.** The two arms are geared together and sprung shut. Pulled open to their stop, a tail on each arm
   rides over a sprung catch and clicks behind it. The catch then holds both arms open (pads 90 mm apart).
2. **Seat the phone.** Stand the phone on the cradle's two feet, between the open arms, with its back square to the
   back plate.
3. **Fire the trigger with the phone.** A 20 mm pad stands 6 mm proud of the back plate's centre. The catch sits on
   the same plunger as the pad. Pressing the phone onto the pad, 3 mm or more, moves the catch out of the tails' way,
   and the springs close the arms on the phone's long edges.
4. **Let go.** The holder, not the robot, must hold the phone.

What makes it hard (physically, not in recognising anything):
- **Cocking.** The fingers must reach between the closed arms, which are 58 mm apart, and push an arm 18 mm out to
  its stop against two springs and the catch's chamfer. Stopping short leaves the arms to spring shut again.
- **Seating.** The phone has to come in between the open arms (6 mm clear at each lip) and stand on two feet that are
  18 mm deep. It must arrive square: if it is tilted or held out from the plate when the trigger fires, the arms'
  lips close on its screen instead of its edges.
- **Firing.** The usual way is to press the phone itself onto the pad, at least 3 mm, centred and on the feet. The
  arms snap shut while the robot still has the phone, so it has to hold the phone square through the snap, then
  release it and slide its fingers off the top without pulling it out. Other ways are fine too: for example, holding
  the arms open by hand, putting the phone in and letting the arms close on it. Only the result is scored.
- **Ending matters.** A gripper holding the phone in the clamp scores 0.75, not 1. Dropping the phone scores 0.

## Real objects (full size)

- **Phone:** an iPhone 15-class phone, 147.6 x 71.6 x 7.8 mm, 171 g (aluminium frame, glass faces).
- **Holder:** a one-touch vent-mount cradle of the common type. Its arm span is 58 mm closed and 90 mm cocked, with
  a stop at 94 mm, so it takes phones 58-89 mm wide. The back plate is 64 x 102 mm. The two feet are 20 mm wide and
  18 mm deep. The arms' silicone pads are 30 mm tall, centred 70 mm above the feet, with 3 mm lips at the front.
  The trigger pad is 20 x 20 mm, centred 70 mm above the feet (behind the phone's middle). The holder head weighs about 200 g.
- **Dashboard section:** a moulded 60 x 15 x 9 cm section (dark grey ABS) with an air-vent grille on its face. It is
  clamped to the table, and the holder's clip grips its vent louvre.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the holder's link boxes, the catch and chamfer, the phone's start, box maths |
| `make_assets.py` | Writes `assets/holder.usda` (the holder: a fixed-base articulation) and `assets/phone.usda` |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files. It hooks yam_arena without modifying it. Adapted from the scene helper in this repository's `lock-before-loading` task folder (its license applies). |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: the mechanism, every scoring level and violation, the demo's reach and clearances |

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
phone's pose or the holder's joints from the simulator. The monitor's state is only printed, as a log.

1. **Cocking (left arm).** With its fingers closed and pointing down, the left arm reaches between the closed arms,
   its fingertips in the arms' 55-85 mm band. It pushes the left arm out to its stop (aiming 1 cm past it, so the stop takes the arm's give), and the gearing opens
   the right arm with it. It backs off so the arm settles on the catch. It then sweeps to the right arm and does the
   same; with the gearing working, that arm is already open and the sweep only checks it. Then it lifts away.
2. **Taking the phone (right arm, meanwhile).** The right arm comes down from above with its fingers open across the
   phone's width. It closes on the phone's long edges 3 cm below its top and lifts it straight up, off the
   dashboard's edge. It turns the phone upright and waits to the right of the holder.
3. **Mounting.** Once the holder is cocked, the right arm brings the phone in front of the cradle, its bottom 3 mm
   above the feet. It slides the phone in between the open arms, 2.5 mm clear of the trigger pad, and lowers it to
   1 mm above the feet. It then presses it back toward the plate (aiming 2.5 mm past it, to take up the arm's give),
   which pushes the trigger in past its 3 mm release point. The arms snap shut on the phone's edges.
   The arm opens its fingers and slides them up off the top of the phone. Both arms go back to their ready pose.

A slow integral term on the measured joint error (the arms' encoders) takes out the sag from pushing the springs and
carrying the phone. The plan takes about 690 steps (35 s).

Measured on an L40S, the demo reaches 1.0 (success) in 642 steps (32 s). The arms close at 6.8 mm of travel on each side, the pad gap equal to the phone's width, with the phone centred, upright and 1.4 mm above the feet. Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step. It latches violations for the whole trial. Otherwise the stage at the
end of the episode counts (`SCORE_OVER = "end"`): clamping the phone and then pulling it out again keeps nothing. Everything about the cradle is measured
in the holder's own frame.

**0 if at any time:**
- the phone falls faster than 0.5 m/s while no gripper is holding it (no finger within 1 cm of it), a free fall of
  about 1.3 cm;
- the phone's centre goes more than 2 cm below the table top (it left the table);
- a holder arm, just released from the catch, stops on the phone's face instead of its edge. This is judged once,
  when an arm that fell from the catch (with no gripper within 2 cm of it) comes to rest touching the phone (within
  1 mm): either the phone's face is turned more than 20 degrees toward that arm, or the arm stopped more than 1.5 mm
  wider than the phone's edge, which is its lip resting on the glass. Merely touching the arms, cocked or closed,
  with the phone at any angle is not a violation.

**Otherwise, the stage at the end:**

| Score | Stage |
|---|---|
| 0 | none of the below |
| 0.25 | **Cocked:** both arms still open at least 14 mm (pads at least 86 mm apart) with no gripper within 2 cm of either, so the catch holds them |
| 0.5 | **Placed:** the phone stands on both feet (lowest corner from 2 mm below to 3 mm above the feet's tops), upright and square within 10 degrees, centred within 15 mm, its back within 12 mm of the back plate, between the two open arms |
| 0.75 | **Clamped:** the arms closed onto the phone's long edges. Each arm is within 2 mm of where its pad meets the phone's edge, so the arms were stopped by the phone, not by their closed stop. The phone is centred within 2 mm, upright and square within 6 degrees, its lowest corner from 2 mm below to 10 mm above the feet's tops (the pads may hold it a little above them), its back within 6 mm of the plate (inside the lips) |
| 1.0 | **Success, held 1 s:** clamped, every part of both grippers at least 2 cm from the phone and from the holder's arms (let go), the phone still (under 1 cm/s and 0.1 rad/s) and the arms still (under 1 cm/s), no violation |

- **Doing nothing, or refusing,** scores 0.
- **Holding the phone in the clamp** scores 0.75. Only the holder carrying it scores 1.

On a real table, with a ruler and a photo:
- the arms' pads touch the phone's two long edges, and their gap equals the phone's width;
- the phone's bottom edge is on, or less than 1 cm above, both feet and its back lies flat on the back plate;
- nothing touches the phone, and it stays put for a second.

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
- The holder's springs are weak (about 1.3-2.2 N per arm, about 2 N on the trigger; see below). The only pinch point is between the arms,
  as when a person uses the holder.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

Simulation details:
- 240 Hz physics, 20 Hz control.
- The holder is a fixed-base articulation built from boxes. Each arm is a prismatic joint with a spring (80 N/m,
  0.8 N preload when closed, so 1.3 N on the phone). The right arm follows the left through a mimic constraint, which
  stands in for the rack and pinion. The trigger is a prismatic joint with a spring (300 N/m, 1.2 N preload) and
  6 mm of travel. It carries the two catches, each with a 45-degree chamfer and 3 mm of overlap with its arm's tail.
- The spring forces are estimates, not measured on a product, and are softer than many real holders (whose arms
  can take several newtons to pull open). The pads' friction (1.0) still lets them carry the phone on their own.
- Friction: silicone pads 1.0, phone 0.5, the plastic internals (tails and catches) 0.2.
- The dashboard section is a fixed box.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
