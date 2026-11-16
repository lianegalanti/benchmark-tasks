# Take an anterior nasal swab from a manikin: four light-pressure circles in each nostril, then cap the swab

A life-size adult head-and-shoulders patient manikin faces the robot on the table. Its head sits on a sprung neck and
its lower nose is soft. A sterile foam-tipped anterior nasal swab hangs in its transport tube, in a small rack to the
robot's right: the swab is fixed in the tube's cap, and the cap is its handle. The robot must:

1. take the swab out of its tube by the cap;
2. put the whole 15 mm foam tip into one nostril (its end 15 to 22 mm in, never more than 25 mm) and rub it round
   the nostril's wall in four full circles, pressing gently (0.3 to 2 N);
3. do the same in the other nostril with the same swab;
4. never let the foam tip touch anything but the nostrils' linings (face, lips, the outside or rim of the nose, the
   table, the tube's outside, its own fingers): a touched tip spoils the specimen;
5. put the swab back in its tube with the cap seated, and let go of it only there.

Instruction: *"Take an anterior nasal swab from the manikin: four circles against the wall of each nostril, then put
the swab back in its tube and let go."*

The skill is **sustained force-controlled circling** with a thin tool: not insert-and-hold, but keeping light,
steady contact while moving round the inside of a small cavity, four full circles in each nostril. Collecting specimens is a core
task of medical assistants (O\*NET 31-9092.00, "Collect blood, tissue, or other laboratory specimens"; 833,900 US jobs)
and of nurses, and anterior nasal swabs for rapid flu, COVID-19 and RSV tests are taken many times a day in clinics and
pharmacies. The steps and limits follow the CDC's instructions for an anterior nasal specimen: insert the entire
collection tip (usually 1 to 1.5 cm), rotate it against the nasal wall at least four times, repeat in the other
nostril with the same swab, and return the swab to its tube.

What makes it hard:
- **Light, steady contact inside a small cavity.** The nostril is a 15 x 10 mm oval and the tip is 5.5 mm across:
  the tip has 2-5 mm of free play, and a circle only counts while the tip presses the wall it is on with 0.3 to 2 N
  (1.2 to 8 mm of the soft nose's give) and is 15 to 22 mm deep. Circling in the air, scraping too hard, going round
  too shallow, or pushing the nose from outside scores nothing. The arm's wrists are soft (about 150 N/m at the tip,
  the same order as the nose), so the pressure has to be held by feel or by a well-judged path, not by position alone.
- **A thin tool, 13 cm out of the hand.** The swab is held by its cap; every millimetre at the tip is a fraction of a
  degree at the wrist, and the tip must never touch the nostril's rim or the lip just below it.
- **A patient's head, not a fixture.** The head sits on a sprung neck that gives way past 0.2 N m: pushing too hard
  moves the head, and turning it more than 15 degrees is a fail.
- **Back in the tube.** The swab has to be turned over and lowered tip-first down a 12.5 mm bore until its cap's
  10 mm plug (tapered at its end) drops in and the flange sits on the rim; the swab may be let go of only there.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor (contamination, depth, nostril-wall force and circles, letting go, the neck), `success()` and `score()` |
| `geometry.py` | Dimensions and frames of the manikin, nose and nostrils, swab, tube and rack; distance maths |
| `make_assets.py` | Writes `assets/manikin.usda` (one articulation: bust, sprung neck, head, soft lower nose), `assets/swab.usda` and `assets/tube.usda` |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files (hooks yam_arena without modifying it) |
| `demo.py` | The scripted demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: sizes, the nostrils' geometry, every scoring level and violation, the demo's reach, joint limits and clearances, and the whole plan replayed through the scorer |

`scenekit.py` is adapted from the scene helper of another task folder in this repository (its license applies). All geometry is generated in code from primitives; no third-party meshes or textures are used.

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env` elsewhere):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take an anterior nasal swab from the manikin: four circles against the wall of each nostril, then put the swab back in its tube and let go." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/nasal-swab-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is **open loop**. It works from the fixed, known layout and the arm's own joint readings only, and never
reads the swab's, the head's or the nose's state from the simulator. Only the right arm moves.

1. **Swab out.** Fingers level, it closes across the cap's two flats and lifts the swab 14 cm straight up, out of the
   tube.
2. **Turn it over.** A joint-space move turns the swab tip-up and brings it 2.5 cm below the patient's left nostril,
   in line with its axis (65 degrees above horizontal).
3. **Each nostril.** The tip's end goes 18 mm up the nostril along its axis, then out against the wall and round the
   oval eight times (3.2 s a circle). The commanded path is 5.5 mm outside the tip's free path, which would press the
   soft nose about 1.4 N with a stiff arm; the soft wrists give some of it back, so not every part of every circle is
   in the force band (eight are made for the four that count). Then back to the middle and out, and the same in the
   right nostril.
4. **Cap it.** A joint-space move brings the swab back over its tube, tip down; it is lowered in until the flange sits
   on the rim and pressed 3 mm further, the fingers open and the arm backs off up and away to its ready pose.

The plan takes 2,004 steps (100.2 s). A slow integral term on the measured joint error (the arms' encoders) takes out
the soft wrists' sag.

**Measured on an L40S in Isaac Sim:** the demo scored 1.0 (success held at step 1,936, 96.8 s): 5.86 and 5.16 net
turns counted in the left and right nostrils (4 needed), the soft nose pushed at most 6.3 mm (1.6 N), the tip's end
at most 20.7 mm deep, the head turned at most 0.4 degrees, the cap seated, no violation. Dream-1 scored 0 in a 3,000-step trial (the run limit).

## Scoring

The monitor runs after every control step; the score at the end of the episode counts (`SCORE_OVER = "end"`).
Everything is measured in the patient's frame: the head's and the lower nose's link poses, the neck's joint angles and
the soft nose's displacement.

- **Nostril-wall force:** the lower nose rests on a spring of 250 N/m each way across the nostrils, so the force on
  the wall is 250 N/m times its displacement (zeroed at the start). 0.3 to 2 N is 1.2 to 8 mm.
- **Depth:** the foam tip's end, measured along the nostril's axis from its opening.
- **Circles:** while the tip's end is in a nostril 15 to 22 mm deep (the whole foam tip inside), the nose is pushed
  0.3 to 2 N, and the tip is within 1 mm of the lining on the side the nose is pushed toward (so pressing the nose
  from outside doesn't count), the direction of the push is followed step by step; its net turn, in full turns, is
  the circle count (0 to 4) for that nostril. Pauses don't reset it; back-and-forth cancels out.

**0 if at any time:**
- the foam tip comes within 1 mm of anything other than a nostril's lining (or the inside of its own tube's bore):
  the face, lips, the outside or rim of the nose, the bust, the table, the rack or the tube's outside, or either
  gripper (until the swab is back with its cap seated);
- the tip's end goes more than 2.5 cm up a nostril (past the vestibule toward the turbinates);
- the head is turned more than 15 degrees from straight (forced);
- once the swab has left its tube, both grippers are 1 cm or more from it while it is not seated in its tube (let
  go of anywhere else), or it falls off the table (its centre 5 cm below the table top).

Lifting, tilting and turning the swab, regripping it, pausing, and holding the head are all allowed.

**Otherwise, at the end:**

| Part | Credit |
|---|---|
| Each nostril | 0.35 x circles / 4 (up to 4 circles) |
| Swab back in its tube, cap seated (after at least one circle) | 0.2 |
| ... and both grippers 2 cm from the swab, the tube and the head | 0.1 |

**Cap seated:** the flange's underside within 2 mm of the rim and within 3 mm of the tube's axis, the swab within
8 degrees of the axis, tip down.

**Success (1.0)**, held for 1 s: four circles in each nostril, the swab capped, both grippers 2 cm clear, the swab
still (under 1 cm/s) and the head still (under 2 degrees/s). Otherwise the score is at most 0.9.

**Doing nothing, or refusing,** scores 0; so does taking the swab out and putting it back unused.

On a real manikin the wall force can be read from a thin-film force sensor in each nostril's soft lining (or the
lower nose's displacement on a calibrated spring mount), the depth from marks 1.5 and 2.2 cm from the tip's end, and the
circles, the touches and the cap from video.

## Simulation model

- **The manikin** is one articulation fixed to the world:
  - the base: bust (400 mm wide, 105 mm high at the shoulders) and neck;
  - the head (1.3 kg; a 192 x 152 x 230 mm ellipsoid as a convex hull, with lips, eyes, ears and the nose's bony
    bridge), on two revolute joints at the top of the neck (pitch, yaw). Each is sprung to straight (20 N m/rad,
    capped at 0.2 N m: the patient holds still but gives way to a push) and limited to 30 degrees;
  - the soft lower nose (columella, alae, tip, sill and both nostrils' 2 mm linings, 16 staves round each oval), on
    two prismatic joints across the nostrils, sprung at 250 N/m with 10 mm travel, gravity off.
- **The swab** is one rigid body (8 g): the cap's grip section (grippy plastic, friction 1.0), flange, plug with its
  tapered lead-in, shaft and a foam capsule tip (0.6).
- **The tube and rack** are one fixed body: 12 staves and a bottom; the rack's four walls.
- **Not simulated:** the lining's local deformation (the whole lower nose moves instead), the specimen itself, the
  swab's bend, the cap's push fit (the cap seats under gravity).

## Real-world trials

- Use a nasal-swab training manikin (or a head-and-shoulders nursing manikin with soft nostrils) clamped to the
  table, and real swab-in-tube collection kits (one per trial).
- The swab is light and blunt, the manikin is not a person: no hazard beyond the arms' own motion.

## Runtime and seeds

A full 3,000-step trial with Dream-1 takes about 6 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
