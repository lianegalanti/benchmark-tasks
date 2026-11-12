# Swap the toner cartridge in a desktop laser printer and shut its door

A compact desktop laser printer (40 x 36 x 23 cm, 7.5 kg, standing free on the table) has its front door open. In its
toner bay sits the used cartridge; the new one lies on the table to its right. The robot must:

1. take the old cartridge out and put it down on the table;
2. slide the new one in until it is seated;
3. shut the door.

It must not touch the transfer roller inside the bay.

The cartridge is full size: a 34 cm long, 10 x 9 cm black body (0.55 kg used, 0.70 kg new) with a single handle on
top. Two side bosses run in guide rails on the bay's walls, and the rails slope down 15 degrees toward the back with
3 mm of play. At their end, each boss drops 12 mm into a seat notch, under the slot's ceiling, which runs on over the notch to the back stop (so the notches can be reached only along the slots, not from above), and the cartridge's front settles on two support ribs on the bay floor. The two bosses differ in height (10 and 18 mm),
so a cartridge turned end for end does not fit.

What makes it hard:
- **Extraction.** The old cartridge must first be lifted out of the notches, then drawn up the slope, square and
  level, without jamming its 34 cm length between the rails.
- **Insertion.** The new one must enter both rails at once and follow them down until both bosses drop into the
  notches.
- **The door.** It is an L-shaped cover (front panel and top flange) on a hinge along the bay's bottom edge. It
  only shuts on a seated cartridge: a cartridge left on the rails' end lip, or anywhere short of the notches, meets
  the door's front panel or its inside rib before the door is within 2 degrees of shut.
- **Closing it.** The door must be pushed up, following its hinge's arc, from 80 degrees open until its weight
  carries it shut.

The swap (out first, then in) and the door interlock make this task more than a single "slide a module into a bay".

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions of the printer, door, rails, notches and cartridges, and the box maths the scorer and tests use |
| `make_assets.py` | Writes `assets/printer.usda` (the printer and its door, one articulation) and the two cartridge files |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files (hooks yam_arena without modifying it) |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: the mechanism, every scoring level and violation, and the demo's reach and clearances |

Tested with yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

## Run a trial

From this folder, in a shell with the yam_arena environment active. New shells on the walkthrough's Brev instance
have it; elsewhere, `source <ARENA_DIR>/yam.env`.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Replace the toner cartridge in the printer with the new one from the table, put the old one down on the table, and shut the printer's front door. Don't touch the black roller inside the printer." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/toner-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo works from the fixed, known layout and the arms' own joint readings only. It never reads the cartridges' or
the door's pose from the simulator. Every grip leans 30 degrees forward, which is how the short arms reach into the
bay at handle height.

1. **Old cartridge out (left arm).** The arm closes on the old cartridge's handle and lifts the bosses 16 mm, clear
   of the notches. It draws the cartridge up the rails' slope with the bosses centred in the slots, then on in front
   of the printer. It lifts the cartridge 3 cm, well clear of the open door, carries it to the left and sets it down on the table 3 mm above it, then lets go.
2. **New cartridge in (right arm).** Meanwhile the right arm picks up the new cartridge and holds it at the rails'
   height. Once the old one is clear, it brings the new one in front of the bay, 3 cm above the rails' line, lowers it onto the line just ahead of the rails' lead-in flares and slides it down the rails until
   the bosses are over the notches. It lowers them in, lets go and backs out.
3. **Door (left arm).** The left arm goes round the door's left edge, low and in front of its flange. With its closed
   fingertips on the flange's outer face, it pushes the door up along the hinge's arc in 5-degree steps to 3 degrees
   from shut, then lifts off. The door's weight then carries it shut.
4. Both arms return to their ready pose.

After every reset, once the door's start angle has settled, `scenekit.py` puts the printer and both cartridges back
exactly at their start poses, at rest, and lets them settle for 0.25 s. The trial starts from the layout in the
setup, not from wherever the settle nudged them.

Carrying a cartridge makes the soft wrist joints sag. A slow integral term on the measured joint error (the arms'
encoders) takes this out. The plan takes about 1,000 steps (50 s).

Measured on an L40S, the demo reaches 1.0 (success) in 1,025 steps (51 s). The new cartridge ends 1 mm from its seated position and 0.4 degrees from level, and the printer moves 0.3 mm. Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step and latches violations; the score at the end of the episode counts
(`SCORE_OVER = "end"`).

**0 if at any time:**
- the printer is moved more than 1 cm, or turned or tipped more than 2 degrees, from its marked place (a taped
  outline on a real table);
- any part of either arm touches the transfer roller (comes within 3 mm of it);
- a cartridge that no gripper holds falls faster than 0.5 m/s, a free fall of about 1.3 cm (setting it down, or
  pushing a let-go cartridge down the rails, is slower);
- a cartridge tilts more than 30 degrees (toner spills, or it fell);
- a cartridge leaves the table.

**Otherwise, at the end:**

| Score | Condition |
|---|---|
| 0 | the old cartridge is still in the printer, held, or not resting on the table |
| 0.2 | the old cartridge is wholly outside the printer's footprint, resting on the table (lowest corner within 8 mm of it, level within 5 degrees), let go |
| 0.5 | and the new cartridge is seated and let go: both bosses down in the notches, its body within 4 mm of the seated position, level and square within 2 degrees |
| 0.75 | and the door is shut: within 2 degrees, so the top of its front panel is within 6 mm of the case front |
| 1.0 | success, held for 1 s: all of the above, every part of both arms at least 3 cm from the door and both cartridges, and the cartridges and the door still |

- **Let go** means no gripper is closed on the cartridge's handle. A gripper holds it while its grasp point is within
  3 cm of the handle's centre and its fingers are less than 3 cm apart.
- **Only the new cartridge counts as seated.** Putting the old one back in scores 0.
- **Doing nothing, or refusing,** scores 0.

How the simulation judges these, as from a video, once per control step:
- The arms are checked as capsules around their links: 3.5 cm radius for the links and 4.5 cm for the gripper.
- **Seated, out of the printer's footprint and the door** are judged in the printer's measured frame, so a printer
  nudged within its 1 cm limit costs nothing. **Seated** is judged from the cartridge's pose relative to the printer. On a real table, the door shutting is itself the check: it cannot
  come within 2 degrees of shut unless both bosses are down in the notches. `test_task.py` checks this for a
  cartridge on the lip and at points up the rails.

## Real-world trials

- Use an unplugged printer and dummy cartridges: empty shells weighted to 0.55 and 0.70 kg, so nothing can spill.
- Tape the printer's outline on the table for the "moved" check.
- The cartridges are light and the robot works at table height, so a dropped cartridge falls at most about 15 cm,
  onto the table.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
