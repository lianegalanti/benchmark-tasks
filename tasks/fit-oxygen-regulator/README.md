# Fit a yoke regulator onto a portable oxygen cylinder's pin-index valve, hand-tight

An empty size D aluminium medical oxygen cylinder lies in a wooden cradle, its brass pin-index post valve (CGA 870,
the US form of ISO 407) pointing at the robot. A yoke oxygen regulator stands on the table in front of the right arm.
The robot must fit the regulator onto the valve, hand-tight, and let go of it.

More than 1.5 million adults in the US use supplemental oxygen at home (American Thoracic Society workshop report,
Ann Am Thorac Soc 2018;15:1369). A portable cylinder is what lets them leave the house, and every fresh cylinder needs
its regulator fitted like this. Many of them have limited hand strength or dexterity. They wait for a family
caregiver or a supplier's visit, or go out without oxygen. A robot that does this reliably helps them stay
independent and eases their carers' load.

What makes it hard is physical, and the geometry enforces it:

- **Blind keyed insertion.** The yoke's window must pass over the post, with 0.5 mm play top and bottom; the post's
  end is chamfered 3 mm to lead it on. Then the yoke moves sideways so that two 4 mm pins, standing 5.5 mm proud of
  the inlet, enter two 4.75 mm holes in the outlet face (0.375 mm a side, with a 0.6 mm countersink at each mouth).
  The yoke hides the holes once it is on. A yoke turned end for end puts the pins on solid metal, 2.7 cm from the
  holes. One turned upside down or back to front puts its inlet on the wrong side. Neither can seat.
- **Hold and turn.** Until the screw bites, nothing holds the yoke square on its pins: one hand must hold it while the
  other turns the T-handle about 6.3 turns (1.75 mm pitch) to bring the tip into the recess, then tightens it until it
  resists 0.75 N m. A wrist turns at most about 200 degrees, so this takes about a dozen regrips.
- **Keep the cylinder in its cradle.** The cylinder lies loose in its cradle, and the yoke is pushed against it.
  Sliding or rolling it is harmless (the fit is judged on the cylinder, wherever it lies), but tipping it out of the
  cradle is a violation.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the monitor, `success()` and `score()` |
| `geometry.py` | Dimensions (ISO 407 basic dimensions), the placement on the station, and the fit test |
| `regkit.py` | Parts built from boxes and cylinders, with a T-screw articulation, for yam_arena task files |
| `demo.py` | The scripted full-score demo (the known layout and the arms' joint encoders only) |
| `test_task.py` | Offline tests: dimensions, clearances, every scoring level and violation, the demo's plan |

`regkit.py` is adapted from `partkit.py` in this repository's `tasks/din-rail-mount` (its license applies). It hooks
`yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit the regulator onto the oxygen cylinder's valve, hand-tight and ready to use, and let go of it." \
  --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/oxygen-regulator-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out                          # full score: demo_*_cam.mp4, setup.png, result.json
python demo.py task.py out --no-record --variant no_pins # slides the yoke on but never seats the pins, then turns
python demo.py task.py out --no-record --variant no_turn # seats it on the pins and lets go without the screw
```

The demo reads no object state. It works from the fixed layout and the arms' joint encoders: forward kinematics of
the measured joints gives where each hand really is.

1. **Pick.** The right arm grips the regulator's flat-sided body from above, its fingers' targets 5 mm inside the
   body (about 25 N each, so both finger drives stay centred). The pads square and centre the body, so the arm knows where the regulator sits in its hand from where it rested on the table.
2. **Line up.** It carries the yoke 8 cm short of the post's end, then lines it up on the post, pins 2.6 mm clear of
   the outlet face. At each of these stops it waits 1.5 s and corrects for the payload's sag, closed loop on its own
   encoders. It then freezes that correction, so contact cannot wind it up.
3. **Slide on and seat.** It slides the yoke over the post at 5 mm/s.
   - If its encoders say the hand is held back (the window caught the post's end), it backs off 3 mm and comes again,
     up to 1.6 mm higher or lower.
   - It then pushes the yoke 7.2 mm sideways (0.7 mm past contact), softly and in steps, so the pins enter their
     holes and the washer meets the outlet face.
   - When its encoders say the hand is held back (pins on the face), it stops advancing and draws circles
     (0.4 mm, growing to 1 mm) until the pins drop in.
   - It holds the regulator there.
4. **Tighten.** The left arm points its gripper along the screw's axis and grips the T-bar.
   - Each stroke turns it about 195 degrees with the wrist, opens, backs off 4 cm, turns back and grips again.
   - It follows the screw in by 1.75 mm per turn.
   - It counts how far the T-bar has turned from its wrist encoders while it grips.
   - It stops 0.3 turn past where the screw reaches the recess (6.6 turns, 14 strokes). The last strokes turn
     against the rising torque.
5. **Let go.** Both arms let go and return to the ready pose. The regulator stays clamped.

**Isaac Sim:** the encoder-only demo above reaches 1.0 (success) in 3,016 steps (151 s; the run limit is 4,800). It
did so in two runs; the submitted video is from the second. Dream-1 scored 0 in a 4,800-step trial (the run limit).

- The window caught the post's end twice. On the third approach, 1.6 mm higher, the yoke slid on.
- In 14 strokes the T-bar turned 6.56 turns, counted by the left wrist's encoders.
- There were no violations.

At the end of that run, the scorer measured (`result.json` of the demo):

| Measure | Value | Needed for full credit |
|---|---|---|
| Both pins into their holes | 3.9 mm (fully seated), 0.37 mm off the holes' axes | at least 3 mm |
| Washer to outlet face, all round | 0.01 mm (flat) | within 1 mm |
| Inlet axis to outlet axis | 0.05 degrees | within 3 degrees |
| Clamp force | 60 N (the cap) | |
| Torque to turn the T-bar further | 1.55 N m | at least 0.75 N m |

Earlier runs found and fixed several problems:

- With the fingers fully closed, both finger drives saturate and the grip can slide sideways, so the right hand now
  holds the body below the drives' cap.
- With 1.5 mm of window play the yoke hung low on the post, so the window is now snug and the post's end chamfered.
- Pushing hard on misaligned pins shifted the yoke in the grip, so the seat push is now soft and stepwise.

## Scoring

The monitor runs after every control step and latches violations. The score at the end of the episode counts
(`SCORE_OVER = "end"`).

- **0 if at any time something unsafe happens:**
  - the cylinder tips out of its cradle: its axis turns more than 30 degrees from where it lay, or drops more than
    2 cm below its cradled height;
  - the regulator is dropped (it moves down faster than 0.5 m/s while no gripper holds it) or falls off the table.
- **Allowed:** setting the regulator down to regrip it, and sliding or rolling the cylinder in its cradle.
- **Otherwise, from the state at the end, judged relative to the cylinder wherever it lies:**
  - **0.25, on the post:** the post passes through the yoke's window and the yoke's whole 32 mm depth is over the post.
  - **0.5, seated:**
    - both pins at least 3 mm into their holes (seated fully they reach 3.9 mm; a pen mark on each pin);
    - the washer within 1 mm of the outlet face all round (feeler gauge);
    - the inlet's axis within 3 degrees of the outlet's, so the yoke sits square on the valve face (turning about
      that axis, which the pins limit, doesn't count).
  - **0.75, hand-tight:** seated, and also:
    - the screw tip in the recess, within 1.2 mm of its centre;
    - a torque gauge on the T-bar reads at least 0.75 N m before the T-bar turns further.
  - **1.0, success:** seated and hand-tight, every part of both arms at least 3 cm from the regulator, and the
    regulator still (it moved less than 1 mm and 0.5 degrees over the last second), held for 1 s with no violation.
- **Doing nothing, or refusing,** scores 0.

Unintended strategies don't score:

- A screw tightened on a yoke whose pins sit on the face clamps a cocked yoke. It is not seated, so it scores 0.25.
- Holding the regulator against the valve without the screw scores at most 0.5.
- A hand-tight regulator still in the hand scores 0.75.
- Nudging the cylinder doesn't help: the fit is judged in the cylinder's own frame, and tipping it out of the
  cradle scores 0.

How the simulation judges these, once per control step:

- **Pin depth, washer gap and squareness** come from the regulator's pose relative to the cylinder's.
- **Held:** a gripper holds the regulator when its grasp point is within 1 cm of the regulator's outline and its
  fingers are less than 7 cm apart.
- **Hand-tight** comes from the screw's joints: the tip's slide position and the T-handle's turn.
  - Past first contact, the thread's target runs ahead of the stopped tip. That lead, times the slide drive's
    stiffness, is the clamp force.
  - The torque to turn further is 0.05 N m plus 0.025 N m per newton of clamp force: 0.75 N m at 28 N (29 degrees
    past contact), and 1.55 N m at the 60 N cap (62 degrees).
- **Arms clear:** the arms are checked as 3 cm capsules between their joints, and the grippers as 2 cm capsules out to
  the fingertips. The regulator's outline includes the T-bar's sweep.

## Real-world build and safety

The cylinder is a size D aluminium medical oxygen cylinder, empty, with its valve closed. A decommissioned, empty
cylinder with its valve fitted works too, so no oxygen is ever released. It lies in a cradle made of two plywood V
saddles and an end board, screwed to a base clamped to the table. The regulator is any CGA 870 yoke oxygen regulator
(about 0.3 to 0.45 kg). Its pins are the standard oxygen positions 2 and 5. Use a new sealing washer if the old one
is worn. Scoring needs a pen mark on each pin at 3 mm, a feeler gauge, and a small torque gauge on the T-bar. A spring
scale also works: hooked on the end of the T-bar, 3 cm from its axis, hand-tight is 25 N.

## Fidelity

- **Physics:** rigid bodies, 240 Hz physics, 20 Hz control.
- **Modelled as real colliders, at ISO 407 basic dimensions:**
  - the valve post (25 x 28 mm);
  - the two pin holes (4.75 mm, 6 mm deep, 0.6 mm 45-degree countersink, on a 14.3 mm radius at 18 degrees either
    side of the line toward the cylinder);
  - the recess (7 mm, 2 mm deep);
  - the yoke's window (0.5 mm play top and bottom) and the post end's 3 mm 45-degree chamfer;
  - the pins (4 mm, 5.5 mm proud, with a 0.5 mm chamfer);
  - the 1.6 mm washer.
- **Simplified:**
  - the round holes and the conical recess are square pockets;
  - the screw tip is flat;
  - the inlet's 3 mm stub into the outlet is omitted, so the pins alone locate the yoke.
- **The screw:**
  - The regulator is a floating articulation, and the T-screw is a turn joint on a slide joint along its axis.
  - PhysX has no screw joint, so the thread is a coupling written after every control step: the slide's position
    target is 1.75 mm per measured turn of the T-handle. The coupling counts whole turns, since PhysX reports the
    angle wrapped to half a turn either way.
  - A stiff drive (200 N/mm) moves the slide to its target, capped at 60 N. A tip stopped by the recess stays put
    while the handle turns on, and the clamp force rises.
  - The tip can't be pushed back: like a real screw, the thread doesn't back-drive.
  - Turning gets harder as it tightens. The turn joint's damping rises with the clamp force, sized to give the
    torques above at 100 degrees per second (the demo's turning speed): 0.05 N m free and 1.55 N m at the cap.
    Turned more slowly, the simulated resistance is lower, so hand-tight is scored from the clamp force.
- **The cylinder** rests loose in its saddles (friction 0.5). Its contact with the cradle uses PhysX's cylinder
  collider.
- **Simulated:** no gas, pressure or seal leakage.

## Runtime and seeds

Physics is slow with this many small colliders and a 32-iteration solver. The 3,016-step demo, recording 640x352
frames, took about 40 minutes on a shared L40S, so a full 4,800-step trial takes about an hour. The layout is fixed
(`-E object_jitter=0`), so there are no seeds to choose.
