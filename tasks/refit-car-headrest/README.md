# Refit a car headrest at its lowest notch: both posts into their sleeves, then hold the release button and push it down

The top of a car seat back (cropped) is fixed to the table, straight ahead of the robot, with the two guide sleeves
of its headrest. The headrest leans back against the seat, its post tips on the table, as left after it was taken
out. The robot is told:

> Put the headrest back into the seat, locked at its lowest position, then let go of it.

Drivers take headrests out to fit seat covers or child seats, or to fold the seats, and refit them. The lock works
the common way described in owner's manuals: the headrest goes down only while the release button is pressed, and
rises freely. Inside, a sprung pawl in one sleeve clicks into each notch of the post and the notch's flat face stops
it going down; going up, the notches' sloped faces push the pawl back (the first notch is square, so the headrest
can't be pulled out without the button). The usual way:

1. **Both posts in at once.** Hold the headrest upright, line both posts up with both sleeves (200 mm apart) and
   push them in. The pawl clicks into the first notch and the headrest stops at its highest setting.
2. **Button down, push down.** Hold the release button down with one hand and push the headrest down with the
   other, past the second and third notches, to the bottom.
3. **Button out.** Let the button out: the pawl drops into the lowest notch.
4. **Let go.** The seat, not the robot, must hold the headrest.

What makes it hard (physically, not in recognising anything):
- **Parallel two-point insertion.** Two 14 mm posts must enter two 16 mm bores at the same time; the lead-ins catch
  them only within 5 mm, and a headrest held tilted jams one post.
- **Bimanual hold-and-push.** Below the first notch the pawl blocks the post unless the button on top of the bezel
  is held down (about 7 N at its stop). One arm holds the button while the other lowers the 1 kg headrest; pushing harder without
  the button does nothing (forcing a post past the pawl ends the trial with 0).
- **Ending matters.** The pawl must be in the lowest notch with the button let out, and the headrest let go.
  Holding the button at the bottom scores 0.25; holding the locked headrest scores 0.75.

## Real objects (full size)

- **Headrest:** a front-seat headrest with a contoured cushion: 260 mm wide, 170 mm tall, 105 mm thick across its
  middle, thinning to 64 mm over its outer 32 mm at each end; charcoal fabric over foam on a steel frame; 1.0 kg.
  Two chromed steel posts (14 mm tube), 200 mm apart, standing 130 mm out of the cushion's bottom. Each post has
  four height notches 15 mm apart on its inboard face, 8 mm tall and 3 mm deep: the first (nearest the tip) square,
  the others with a flat upper face and a 45-degree lower face. Each tip's inboard edge is chamfered (5.5 mm over
  9.5 mm), so the post pushes the pawl aside on the way in.
- **Seat back, top section** (cropped 105 mm below its top, at the table): the upholstered top of the seat back,
  100 mm thick and 440 mm wide (its full width; the shoulders are squared off), its top 77 mm above the table, fixed
  to the table. Two guide sleeves 200 mm apart, each with a black plastic bezel 28 x 28 mm standing 28 mm above the
  upholstery and a square bore of 16 mm (1 mm play around the post), widening to 26 mm over the top 6 mm (the
  lead-in); the bores are closed 96.5 mm below the bezels' tops, which stops the posts at the lowest notch. The
  sleeves stand vertical here; in a car they recline with the seat back.
- **Lock** (in the right sleeve): a sprung latch. Its pawl tooth (8 mm wide, 5 mm tall, centred 12 mm below the
  bezel's top) stands 4.5 mm into the bore from its inboard side, through a slot. The grey release button (cap
  14 x 16 mm, 6 mm proud) sits on top of the bezel's outboard extension (a housing 20 mm wide reaching 48 mm
  outboard of the sleeve's axis), its centre 36 mm outboard of the axis, and is pressed straight down: its stem bears
  on a 45-degree face on the latch, so 5.8 mm down draws the pawl clear of the post. Springs: the latch 2 N at rest,
  3.1 N drawn back; the button 1.5 N.
- **Depths:** the notches' depths (post tips below the bezels' tops with the pawl centred in the notch) are 50, 65,
  80 and 95 mm. Let down onto the pawl, the tips sit 1.5 mm deeper (51.5 ... 96.5 mm; at 96.5 mm they also rest on
  the sleeves' closed ends). The cushion's bottom edge is 130 mm above the tips: at the lowest notch, 35 mm above
  the bezels' tops (33.5 mm resting on the pawl).
- **Crop:** only the seat back's lower part (below the table) is cut away. Everything the hands and the headrest
  touch (bores, lead-ins, pawl, button, closed ends, the upholstery round the sleeves) is full size.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the seat's and the headrest's boxes, the notches and the lock, the start pose, box maths |
| `make_assets.py` | Writes `assets/headrest.usda` (a rigid body), `assets/seat.usda` (a fixed-base articulation with the sprung latch) and `assets/button.usda` (the release button, its own fixed-base articulation) |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files; hooks yam_arena without modifying it. Adapted from an earlier scene helper in this repository (its license applies). |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: notches, pawl and button, every scoring level and violation, the demo's reach and clearances |

## Run a trial

From this folder, in a shell with the yam_arena environment active. New shells on the walkthrough's Brev instance
have it; elsewhere, `source <ARENA_DIR>/yam.env`.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the headrest back into the seat, locked at its lowest position, then let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/headrest-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is planned from the fixed, known layout and uses only each arm's own joint encoders. It never reads the
headrest's pose or the lock's joint from the simulator. The monitor's state is only printed, as a log.

1. **Grip.** Each arm comes in from outside the headrest's end, fingers wide open, pointing inward and down at 30
   degrees in the headrest's plane, and pinches the cushion's thin end (64 mm) front to back, 32 mm above its bottom.
   The fingers are commanded 5 mm inside each face, not fully shut, so their position springs (about 25 N each,
   under the fingers' force limit) keep the cushion centred between them.
2. **Up and over.** Together the arms lift the headrest 15 mm, stand it upright while drawing it back from the
   seat, raise the post tips 12 mm above the bezels and carry it forward until both posts are over both sleeves.
3. **In to the first click.** They lower both posts into the sleeves at once and push until the pawl clicks into the
   first notch.
4. **Button down, headrest down.** The right arm lets go, closes its fingers and, pointing them inward and down at
   45 degrees from outboard of the cushion's end, presses the release button straight down to its stop. The left arm, holding the headrest by its left end, lowers it 45 mm, past the
   second and third notches, to 0.5 mm above the sleeves' closed ends.
5. **Button up, let go.** The right arm draws back along its approach and the button springs up (the pawl springs
   into the lowest notch); the left arm opens and draws back, and both return to ready. Every part of both grippers
   stays above the seat back's top throughout.

A slow integral term on the measured joint error (the arms' encoders) takes out the sag from carrying the headrest
and pushing the posts and the button. The plan takes about 838 steps (42 s).

Measured on an L40S, the demo reaches 1.0 (success) in 816 steps (41 s). The pawl clicked into the first notch
(post tips 51.5 mm down); pressed from above, the button went down 5.4 mm and drew the latch to its 5.5 mm stop; the
headrest was lowered to the closed ends (96.5 mm) and, with the button let up, the pawl sprang into the lowest notch;
let go, the headrest stands within 0.1 degrees of upright. Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step, in the seat's own frame (the seat is fixed to the table). It latches
violations for the whole trial. Otherwise the stage at the end of the episode counts (`SCORE_OVER = "end"`).

**0 if at any time:**
- the headrest leaves the table: any corner of it more than 2 cm below the table top;
- the headrest is dropped: its centre falls faster than 0.5 m/s (a free fall of about 1.3 cm) while every part of
  both grippers is at least 2 cm from it (letting go with the button held down, so that it falls down the sleeves,
  counts; setting it down gently does not);
- any part of the headrest is forced more than 4 mm into the seat (through a bezel, a bore's wall or its closed end);
- a post is forced past the pawl: more than 1.5 mm into it, or carried more than 1.5 mm past a notch's flat face
  between two control steps while the button was up at both (so a push that gets past between samples counts too).

Lifting, tilting, carrying the headrest, setting it down on the table or leaning it against the seat, pulling it
back out (with the button) and retrying are all fine.

**Otherwise, the stage at the end:**

| Score | Stage |
|---|---|
| 0 | none of the below |
| 0.25 | **Posts in:** both post tips at least 20 mm down their sleeves, their axes within 9 mm of the sleeves' axes |
| 0.5 | **Clicked in:** posts in, the pawl engaged in a notch (button out: the pawl within 1.5 mm of its rest stop; the right post's tips within 2.5 mm of that notch's depth: 50, 65, 80 or 95 mm), the posts within 6 degrees of the sleeves' axes |
| 0.75 | **Locked at the lowest position:** the pawl engaged in the lowest notch (tips within 2.5 mm of 95 mm down), the posts within 3 degrees of the sleeves' axes |
| 1.0 | **Success, held 1 s:** locked at the lowest position, every part of both grippers at least 2 cm from the headrest (let go), the headrest still (under 1 cm/s and 0.1 rad/s) and the lock still, no violation |

- **Doing nothing, or refusing,** scores 0.
- **Holding the button in at the bottom** scores 0.25 (the pawl is not in); **holding the locked headrest** scores 0.75.

On a real table, with a ruler:
- the posts are in both sleeves and the headrest is down: the post tips 95 mm (within 2.5 mm) below the bezels'
  tops, i.e. the cushion's bottom edge 35 mm (within 2.5 mm) above them, standing straight (within 3 degrees, 9 mm
  over its 170 mm height);
- the button is up (at its rest position) and the headrest does not move when pushed down by hand;
- nothing touches the headrest, and it stays put for a second.

How the simulation judges these, once per control step:
- The headrest's pose and the latch's and button's joint positions come from the simulator.
- The grippers are boxes placed from each arm's encoders: each finger's pad and root, and the carriage.

## Real-world trials

- Clamp the seat section to the table. The headrest weighs 1.0 kg; leaning against the seat it can only fall over
  onto the table (a trial with it off the table ends with 0).
- The lock's springs are weak (under 4 N); the button takes about 7 N to hold down. The only pinch points are between the posts and the bezels, as when a
  person fits a headrest.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

Simulation details:
- 240 Hz physics, 20 Hz control.
- The headrest is one rigid body built from boxes: the cushion as three blocks (the foam's give plays no part in
  the mechanism) and each post as boxes that carry its notches, their 45-degree lower faces and the tip's chamfer
  (collision is a 14 mm square section; the posts are drawn as round tubes). The posts are rigid (real ones flex a
  little).
- The seat section is a fixed-base articulation: the block, bores, closed ends, lead-ins and the button's housing as
  fixed boxes; the latch a prismatic joint across the bore with a spring drive (200 N/m, 2 N preload against its
  stop, 5.5 mm travel). The button is a separate fixed-base articulation (so that it collides with the latch), on a
  prismatic joint pointing down (150 N/m, 1.5 N preload); its stem pushes the latch's 45-degree face.
  The spring rates are estimates, not measured on a product.
- The sleeves are vertical; in a car they recline with the seat back.
- The bores have 1 mm play on each side of the posts (real guides are tighter, about 0.3 mm).
- Friction: steel posts 0.25, plastic bezels 0.3, the pawl 0.2, fabric 0.7, upholstery 0.6.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
