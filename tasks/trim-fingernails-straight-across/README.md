# Trim two fingernails straight across with a lever nail clipper to 1-2 mm of white edge while steadying each finger

A resident's right hand lies palm down on a foam hand rest, reaching in from the robot's far left, fingers spread and
held out past the rest's edge over a rolled towel. The index and middle fingernails are long: 3-4 mm of white free edge, rounded. A lever
nail clipper stands jaws down in the slot of a nail-care caddy in front of the right arm. The robot must:

1. take the clipper, lever up, and bring its open jaws onto a nail, the nail between them;
2. hold that finger steady with the other hand: the knuckles are relaxed and springy, so the clipper's push moves an
   unheld finger;
3. squeeze the clipper shut with the cutting line 1-2 mm from the quick, square across the nail, without the jaws
   ever closing at the quick or on the skin;
4. do the same for the other nail, then put the clipper down and let go.

Instruction: *"Trim the resident's index and middle fingernails straight across with the nail clipper, leaving 1 to 2 mm
of white nail, without nicking the skin and keeping each finger steady; then put the clipper down and let go of
everything."*

Nursing assistants (1,505,900 US jobs, O*NET 31-1131) and home health and personal care aides (4,677,100, O*NET
31-1122; 2025 employment) trim residents' and clients' fingernails as routine personal care where the care plan allows:
hold the finger, cut straight across, don't cut into the quick.

What makes it hard:
- **Placing the jaws to half a millimetre.** Full credit needs 1.0-2.0 mm of white edge on every 1.3-1.4 mm strip of
  the nail, and the edge straight across to 0.75 mm. The clipper's jaws are curved (their ends reach 0.6 mm further
  than their middle), so the jaws must be square to the finger within a few degrees and at the right depth. The
  quick is 1-2 mm further in: a closure within 0.5 mm of it is a nick and scores 0.
- **A real tool, used the real way.** Nothing is cut unless the clipper's jaws actually close on the nail: the
  gripper squeezes the lever's end toward the clipper's body (9 mm of travel, up to 7 N), the lever rocks on the
  pin's head and its heel pushes the sprung upper arm down. A parallel gripper squeezing the clipper also moves the
  clipper's body by half the squeeze: the jaws drift onto the finger unless the arm makes up for it.
- **Two hands.** The knuckles are relaxed springs: about 0.7 N at the nail turns a finger 10 degrees. The other hand
  holds the finger down on the rolled towel under it, at its middle segment; the steadiness credit counts only if a
  gripper other than the one on the clipper is within 1 cm of that finger's middle segment at every cut.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor (closures, cuts, nicks, finger steadiness), `success()` and `score()` |
| `geometry.py` | Dimensions of the hand, the nails and the clipper; the clipper's lever, heel and jaw kinematics; the cut model; box maths |
| `make_assets.py` | Writes `assets/hand.usda` (fixed articulation: two sprung fingers, 16 free-edge strips) and `assets/clipper.usda` (free articulation: jaw and lever) |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files (hooks yam_arena without modifying it) |
| `demo.py` | The scripted demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: mechanism, cut model, every scoring level and violation, the demo's reach and clearances |

`scenekit.py` is adapted from the scene helper of this repository's `lock-before-loading` task folder (its license
applies).

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env` elsewhere):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Trim the resident's index and middle fingernails straight across with the nail clipper, leaving 1 to 2 mm of white nail, without nicking the skin and keeping each finger steady; then put the clipper down and let go of everything." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/fingernails-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is **open loop** from the fixed, known layout and the arms' own joint readings. It never reads the hand's,
the nails' or the clipper's state from the simulator. While squeezing it reads the right gripper's finger encoders (the
pads' opening) to keep the clipper's lower jaw in place as the pads close.

1. **Clipper out (right arm).** The pads come down either side of the standing clipper, one on the lever's end and
   one under the body, and squeeze it shut: the lever on its stop makes the clipper rigid in the grip. While the pads
   close, the arm keeps the clipper's body still in the slot. It lifts the clipper out and turns it to point along the
   index finger toward the knuckle, tipped 15 degrees down, lever up.
2. **Hold (left arm).** It holds the index finger down on the towel roll, its closed fingertips pressing 1 mm into
   the top of the finger's middle segment.
3. **Snip (right arm).** The open jaws come in front of the nail's tip, the nail between them, and slide in along the
   finger until the cutting line is 1.65 mm from the quick at the nail's centre. (In front of the nail it first eases
   the squeeze until the jaws are 2.5 mm open.) Over 1.5 s the pads close past the
   jaws' shut position while the arm moves down by half the pads' measured closing, so the clipper's body stays put
   and both jaws close on the nail in the middle of the mouth. It holds 0.5 s, opens to 2.5 mm, slides back off the
   nail and squeezes the clipper shut again to carry it. One snip trims the whole nail: the jaws are 13 mm wide, the nails 10.5 and 11.5 mm.
4. **Middle finger.** The left hand lets go of the index and holds the middle finger down; the right arm trims its nail
   the same way.
5. **Finish.** The clipper goes back into the caddy's slot, jaws down, and the right hand opens. The left hand lets go
   of the middle finger. Both arms return to their ready pose.

The plan takes 1,055 steps (53 s). Its aim points for the jaws on each nail (`AIM` in demo.py) are constant per-nail
aim corrections, tuned once from the scorer's cut log of two earlier runs, the same every trial (separately for the jaws
held open and shut: the clipper turns slightly in the squeezing grip); at run time the demo reads only the arms'
encoders. A slow integral term on the measured joint error takes out the soft wrists' sag.

**Measured on an L40S in Isaac Sim:** the demo scored 1.0 (success held at step 1,000, 50 s). White edge left: index
1.32-1.68 mm on all 8 strips (straight to 0.36 mm), middle 1.48-1.91 mm (straight to 0.43 mm); no nick; the fingers
stayed within 0.4 and 0.7 degrees of rest. Dream-1 refused the instruction on its first inference request, before taking any action ("Clipping human fingernails involves potential pinch or cutting hazards to a person."), so there is no model run video. Real-world trials use a nursing-skills training hand, not a person.

## Scoring

The monitor runs after every control step. The score at the end of the episode counts (`SCORE_OVER = "end"`).
Everything is judged in each nail's own frame (it moves with its finger).

**The cut model.** Each nail's free edge is 8 strips across its width. A strip's white edge is its length from the
hyponychium (the quick: where the nail leaves the skin under it) to its tip. A **closure** is the clipper's jaws
measured, from the jaw's joint angle, at most 0.8 mm open (the nail's 0.5 mm plus 0.3 mm) for two control steps in a
row, with at least 6 N on the lever's pivot (the squeeze; 6 N is about the arms' spring force with the jaws on the nail; the nail's own cutting resistance is not modelled), after having been at least 1.5 mm open since the last one. At a closure, the cutting line (where the two edges meet, curved like the jaws) cuts wherever the nail's mid-plane
lies inside the shut jaws' blades, from 2 mm below the line (the lower jaw) to 1.5 mm above it (the upper jaw): there
the closing jaws went through the nail. Each strip whose centre the line crosses there, between the quick and the
strip's tip, is cut back to the line. Jaws closing beside the nail, above or below it, or hovering open cut nothing.

**0 if at any time:**
- a closure's cutting line, where it cuts as above, lies within 0.5 mm of the quick, or behind it over the nail, or up to
  0.5 mm past the nail's sides (a nick);
- the jaws are within 2 mm of shut with the fingertip's soft tissue (the end of the finger and the nail bed) 0.3 mm or
  more between them (the skin pinched);
- the clipper falls off the table (its centre 5 cm below the table top).

Setting the clipper down, regripping it, snipping a nail several times, and moving the fingers are all allowed.

**Otherwise, at the end, per nail (up to 0.5 each):**

| Credit | When |
|---|---|
| 0.3 x fraction of the 8 strips | white edge 1.0 to 2.0 mm |
| 0.1 | every strip cut, and the longest at most 0.75 mm longer than the shortest (straight across) |
| 0.1 | the nail cut at all, its finger never turned more than 10 degrees from rest at its knuckle, and at every cut the other gripper (not the one touching the clipper) within 1 cm of the finger's middle segment |

The sum over both nails is capped at 0.9 unless the clipper conditions below hold.

**Success (1.0)** needs all of these, held for 1 s:
- both nails at full marks (every strip 1-2 mm, straight, finger steady) and no violation;
- the clipper put down: its lowest point within 4 cm of the table top (standing in the caddy or lying on the table);
- the clipper still (under 1 cm/s);
- every part of both grippers at least 2 cm from the clipper and from both fingers.

**Doing nothing, or refusing,** scores 0.

On a real hand (a training hand with clip-on nail tips), the white edge is read per strip with a feeler gauge or from
a photo through a loupe against a millimetre scale; a nick is a cut mark within 0.5 mm of the quick.

## Simulation model

- **The hand** is one articulation fixed to the world: the palm with the forearm stub, thumb, ring and little fingers;
  the index and middle fingers, each on a revolute knuckle joint (flexion) driven toward rest by a spring of
  0.40 N m/rad. Gravity is off for the hand (the fingers are held out by tone, which the spring models). Each finger
  has its segments (capsules), the nail bed and the attached nail plate (12 mm, 0.5 mm thick).
- **The towel roll** is a fixed box under the fingers' middle segments (its top 0.5 mm below them); the fingertips
  are free over its front.
- **The free edge** is 8 strips per nail, each a link with a collider on a prismatic joint along the finger. The jaws
  press on them. After a cut, the task slides each cut strip back into the plate by what was cut off it, so the
  white edge left stands out exactly that far. The clipping is not simulated as a falling piece.
- **The clipper** is a free-floating articulation: the body (lower arm and jaw, rivet spacer, the pin), the upper arm
  on a revolute joint at the rivet (its stop holds the jaws 4 mm open), and the lever on a revolute joint at the pin's
  head. The lever's heel, 8 mm behind the pin, pushes the upper arm down: in the simulation this cam is a PhysX mimic
  (gear) joint that ties the upper arm's hinge to the lever's at the heel's ratio (0.28, within 0.1 degree of the
  exact cam over the stroke). It acts both ways: a nail between the jaws also stops the lever. The arms' spring is
  felt at the lever: the lever joint's spring gives the torque the arms' spring (0.12 N m preload, 11 N m/rad) gives
  through the cam, 1.1 N at the lever's end at rest and 7 N with the jaws shut. The lever's own stop is where the
  jaws meet.
- **Cutting force.** The jaws close onto the nail's strips and stop there. A closure counts only when the force the
  lever's pivot joint carries (read from PhysX, about the squeeze on the lever) is at least 6 N while the jaws are
  shut on the nail; the cut strips then slide back and the jaws finish closing. 6 N is about the arms' spring
  force with the jaws on the nail; the nail's own cutting resistance is not modelled.
- **Not simulated:** the nail's curve across its
  width (it is flat), the fingers' other joints, and the fingertip's soft pad (cut back square under the quick).

## Real-world trials

- Use a nursing-skills training hand with clip-on nail tips, fixed to the hand rest at the wrist.
- The clipper is small and light (25 g). Its jaws can pinch a gripper pad but cannot harm the robot; the training hand
  takes the place of a person.

## Runtime and seeds

A full 3,600-step trial with Dream-1 takes about 6 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
