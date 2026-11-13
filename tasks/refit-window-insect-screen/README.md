# Put an insect screen back into a window: up into the top channel against its spring plungers, swing in, let down

A cropped section of a small window frame is fixed upright on the table, straight ahead of the robot. Its head has a
deep channel (24 mm) facing down and its sill a shallow one (8 mm front lip) facing up. The window's insect screen
stands on the table in front of it, leaning back against the jambs, as left after it was taken out and cleaned. The
robot is told:

> Put the insect screen back in the window, then let go of it.

Many households take screens out to wash or store them and refit them themselves. A screen
held by spring plungers can't simply be pressed into place: its overall height is more than the opening between the
channels' lips. The usual way:

1. **Top in, pushed up.** Tilt the screen's top toward the window and put its top edge up into the top channel,
   then push the whole screen up. The two spring plungers on its top edge bear on the channel's ceiling and go in.
2. **Swing in.** With the screen held up, its bottom edge now clears the sill's front lip. Swing the bottom in, about
   the top edge, until the screen stands in the window's plane, over the bottom channel.
3. **Let down.** Lower it into the bottom channel. The plungers push it down and hold it there.
4. **Let go.** The window, not the robot, must hold the screen.

What makes it hard (physically, not in recognising anything):
- **Pushing up while swinging.** The bottom edge clears the 8 mm lip only while the screen is pushed up against both
  plungers (about 2.6 N each at full push, plus the screen's 4.5 N weight). Let it sag and the bottom edge catches
  on the lip. It's a large, light panel held near its bottom, so both hands are needed: one per side.
- **Alignment far from the hands.** The top edge, 50 cm above the hands, has to enter a 22 mm channel and stay in it
  while the screen swings; the bottom edge has to drop into a 15 mm channel (2.5 mm play each side).
- **Ending matters.** Holding the screen in place scores 0.75, not 1.

## Real objects (full size)

- **Insect screen:** an aluminium window insect screen for a small window, 450 mm wide x 600 mm tall (about
  18 x 24 in, a stock screen size), about 450 g. Frame profile 10 mm thick x 20 mm face, anodised aluminium; charcoal fibreglass mesh. Two spring
  plungers (POM pins, 8 x 10 mm) on the top edge, 150 mm either side of the centre. Each pin stands 19 mm proud at
  rest and can be pushed in flush (19 mm); its spring gives 0.5 N at rest, 0.9 N fitted (3 mm in), 2.6 N at 17.5 mm.
- **Window frame section** (white PVC/aluminium, cropped): 470 mm long, fixed to the table. Sill 100 mm deep and
  50 mm tall (the frame's bottom member with the wall below it, cropped). In it, the bottom channel: 15 mm wide,
  front (room-side) lip 8 mm tall, back lip 15 mm, lips 3 mm thick. Head 30 mm thick, its underside 616 mm above the
  bottom channel's floor. In it, the top channel: 22 mm wide, 24 mm deep, lips 3 mm thick. Fitted, the screen's top
  edge is 8 mm up in the top channel and 16 mm below its ceiling. Two jambs, 40 mm wide and 35 mm deep, whose
  room-side faces stand 40 mm behind the screen's plane, join head and sill. A 6 mm glass pane fills the opening
  between them. Each channel is closed at both ends by an end stop (lip to lip, 7 mm wide), 3 mm beyond the
  screen's sides, so the screen's width is bounded as in a real frame and it can't be slid in from the side.
- **Crop:** the section ends 10 mm beyond the screen's sides, at the head and sill, and at the jambs' outer faces, so
  the screen's side edges stand free (the wall around the window is cut away). Every part that touches the screen
  (both channels, lips, plungers) is full size.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the window frame's boxes, the screen's start pose, box maths |
| `make_assets.py` | Writes `assets/screen.usda` (the screen: a floating-base articulation with two sprung plungers) |
| `assets/` | The generated USD file (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files; hooks yam_arena without modifying it. Adapted from an earlier scene helper in this repository (its license applies); this copy adds floating-base articulations. |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: channels and plungers, every scoring level and violation, the demo's reach and clearances |

## Run a trial

From this folder, in a shell with the yam_arena environment active. New shells on the walkthrough's Brev instance
have it; elsewhere, `source <ARENA_DIR>/yam.env`.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the insect screen back in the window, then let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/insect-screen-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is planned from the fixed, known layout and uses only each arm's own joint encoders. It never reads the
screen's pose or its plungers' joints from the simulator. The monitor's state is only printed, as a log.

1. **Grip.** Each arm comes in from outside its side of the screen, fingers part open, pointing inward and down at
   45 degrees in the screen's plane. It pinches its stile across the frame's 10 mm thickness, 9 cm above the bottom
   edge: one finger in front of the screen, one behind it.
2. **Bring the top under the channel.** Together, the arms lift the screen 12 mm off the table, bring its top edge
   forward under the top channel (leaning 6 degrees, bottom toward the room) and raise it to 8 mm below the channel's
   mouth.
3. **Push up.** They push the screen straight up until its top edge is 1.5 mm under the ceiling. The plungers go in
   17.5 mm, and the bottom edge rises 6.5 mm above the sill's front lip.
4. **Swing in.** Holding it up, they swing the bottom in about the top edge until the screen stands in the window's
   plane, over the bottom channel.
5. **Let down and let go.** They lower it until its bottom edge is 1 mm above the channel's floor, open their
   fingers (the plungers push it down onto the floor), draw back along their approach and return to ready.

A slow integral term on the measured joint error (the arms' encoders) takes out the sag from carrying the screen and
pushing against the plungers. The plan takes about 494 steps (25 s).

Measured on an L40S, the demo reaches 1.0 (success) in 492 steps (25 s). Pushed up, the plungers went in about 18 mm and the bottom edge rose 6 mm above the sill's lip; fitted, the screen stands within 0.3 degrees of the window's plane, both plungers 3.0 mm in, the bottom edge on the floor and the top edge 8 mm up in the channel. Dream-1 scored TBD in a 2,400-step trial (the run limit). Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step, in the window frame's own frame (the frame is fixed to the table). It
latches violations for the whole trial. Otherwise the stage at the end of the episode counts (`SCORE_OVER = "end"`).

**0 if at any time:**
- the screen is dropped: its centre falls faster than 0.5 m/s (a free fall of about 1.3 cm) while every part of both
  grippers is at least 2 cm from it (a screen that tips over and falls onto the table counts);
- any corner of the screen goes more than 2 cm below the table top (it fell off the table);
- any part of the screen's frame is forced more than 4 mm into the window frame (through a lip, the sill or the head).

Lifting, tilting, carrying it, setting it down gently on the table or leaning it against the window, and retrying
are all fine.

**Otherwise, the stage at the end:**

| Score | Stage |
|---|---|
| 0 | none of the below |
| 0.25 | **Top in, pushed up:** all four corners of the screen's top edge at least 4 mm above the top channel's mouth and within 12 mm of its centre line (inside its lips), both plungers pushed in at least 8 mm, the screen within 15 degrees of the window's plane |
| 0.5 | **Bottom over the lip:** the top edge in the channel as above, and all four corners of the bottom edge within 8.5 mm of the bottom channel's centre line (inside its lips) and from 3 mm below to 24 mm above its floor; the screen within 6 degrees of the window's plane |
| 0.75 | **Fitted:** top edge in the channel as above; all four bottom-edge corners within 8.5 mm of the channel's centre line and from 3 mm below to 4 mm above its floor (at least 4 mm below the lip's top); the screen within 3 degrees of the window's plane, centred on the opening within 15 mm; both plungers bearing on the ceiling (pushed in at least 1.5 mm) |
| 1.0 | **Success, held 1 s:** fitted, every part of both grippers at least 2 cm from the screen (let go), the screen still (under 1 cm/s and 0.1 rad/s) and both plungers still (under 1 cm/s), no violation |

- **Doing nothing, or refusing,** scores 0.
- **Holding the screen in place** scores 0.75. Only the window holding it scores 1.

On a real table, with a ruler:
- the bottom edge sits on the bottom channel's floor, behind its front lip, and the top edge is inside the top
  channel (its lips cover it by at least 4 mm);
- the screen stands in the window's plane (within 3 degrees, i.e. 3 cm over its 60 cm height) and centred;
- nothing touches the screen, and it stays put for a second.

How the simulation judges these, once per control step:
- The screen's pose and the plungers' joint positions come from the simulator.
- The grippers are boxes placed from each arm's encoders: each finger's pad and root, and the carriage.

## Real-world trials

- Clamp the window section to the table. The screen weighs 450 g. A dropped or toppled screen ends the trial with
  0; it falls at most 60 cm onto the table or floor, and an aluminium screen is not a hazard (the mesh may need
  re-tensioning).
- The plungers' springs are weak (under 3 N). The only pinch point is between the screen and the channels, as when a
  person fits it.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

Simulation details:
- 240 Hz physics, 20 Hz control.
- The screen is a floating-base articulation built from boxes: four rails and the mesh panel (rigid; the mesh's flex
  plays no part in the mechanism) in one link, and the two plunger pins, each a prismatic joint with a spring drive
  (120 N/m, 0.5 N preload against its stop, 19 mm travel).
- The window frame section is fixed boxes.
- Friction: aluminium frame 0.6, mesh 0.6, POM pins 0.2. The spring rates are estimates, not measured on a product.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
