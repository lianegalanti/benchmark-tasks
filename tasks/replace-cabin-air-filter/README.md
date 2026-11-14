# Lower a car's glovebox past its stops with a two-handed squeeze, swap the cabin air filter behind it, and close it back past the stops

A section of a car's passenger-side dashboard is fixed to the table with its glovebox open, resting on its two stops.
Behind it, in the face of the HVAC housing, the used cabin air filter sits in its slot. The new filter lies on an
upturned parts bin to the right. The robot is told:

> Take the glovebox down, replace the cabin air filter behind it with the new one from the table, and put the glovebox back up.

What is new here is the glovebox: on most cars it only comes down past its sprung stops while both of its side walls
are squeezed in at once, and it has to be pushed back up until the stops snap behind the dash. In between, the filter
is drawn from and slid into a keyed 200 mm slot reached through the opening. This is a different skill from swapping a
filter under a cover (one hand lifts a cover, the other swaps the element).

This is the usual procedure on most current cars, done by automotive service technicians and mechanics (BLS 49-3023,
about 680,000 workers) and quick-lube technicians at most services. Keyed or stepped frames, like the rib here,
are common on OEM filters:

1. **Lower the glovebox past its stops.** Each side wall of the bin carries a stop bump that catches on the back of
   the dash beside the opening. The bin only drops further while both walls are squeezed in, so the bumps clear the
   dash. This takes both hands: about 10 N on each wall, 8 mm in, while the bin swings down. One side alone does not
   let it pass.
2. **Take the old filter out.** Grip it by its pull tab and slide it straight out of the slot (200 mm), then lift it
   out over the lowered glovebox and put it down on the table.
3. **Slide the new filter in.** The slot has 3.75 mm of side play. A key rib on the filter's top face runs in a
   channel in the slot's ceiling, so it only goes in rib up, tab first toward the cabin: the airflow direction is set
   by the part, not by reading an arrow. Push it in until the tab meets the housing.
4. **Swing the glovebox back up.** Push the bin up until its bumps pass the dash again. Their backs are chamfered, so
   pushing the bin up squeezes the walls in, and they snap back out behind the fascia. Let go: the bin settles back
   onto its stops.

What makes it hard (physically, not in recognising anything):
- **The squeeze.** Both walls have to be pushed in at once and held in while the bin swings 24 degrees down (from 48 to 72 degrees). One
  wall alone, or not far enough, leaves the stop holding it. Forcing it through counts as breaking the stop.
- **Reaching into the dash.** The slot is 14 cm behind the opening, at the height of the opening's middle. The
  grippers must reach in level, under the top of the opening, and draw a 200 mm filter straight out of a 3.6 cm
  tall slot. Then it has to be lifted over the lowered glovebox and clear of the dash's sides.
- **Insertion.** The new filter must enter the slot square, rib up, and travel 200 mm without jamming.
- **Raising the glovebox.** It hangs at 110 degrees. It has to be pushed up from inside, against its weight and the
  two ramps, until both bumps click behind the dash.

## Real objects (full size)

- **Cabin air filters:** the common 215 x 200 x 30 mm size, 0.15 kg, with a rigid plastic frame, a key rib and a pull tab.
- **Glovebox:** 36 cm wide, with a 21 cm tall lid. It is 6 cm deep at the floor and 15.6 cm at the top, weighs
  0.85 kg and has a damper. The flexing walls are modelled as sprung panels: 900 N/m with 2.7 N preload and 14 mm of
  travel. Each has a moulded thumb pad (26 mm proud) and a POM stop bump whose square-faced part stands 36 mm proud of the wall, 6 mm past the edge of the opening. Behind it, each bump has a 20-degree chamfer that rises from 25 mm to 38.1 mm, 2.1 mm above the square part, so the dash's edge steps down onto the flat instead of catching on its edge.
- **Dashboard section:** a cropped passenger-side dash, 54 cm wide. The opening is 42 cm wide and 31.5 cm tall, and the filter
  housing face is 14 cm behind the glovebox hinge, with moulded 45-degree lead-ins at the slot's mouth (a sill 16 mm deep below, 10 mm cheeks at the sides, a hood 10 mm
  high above its outer parts). The section is mounted low on the table (hinge 13.5 cm above it) so
  the arms can reach. The parts that are handled are full size. On some cars the slot is closed by a clip-on cover; it
  is left out here (the slot is open), and so is the glovebox latch (it starts open).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions of the dash, glovebox, stops, slot and filters; box maths for the scorer and tests |
| `make_assets.py` | Writes `assets/dash.usda` (the dash section: a fixed-base articulation with the hinged glovebox and its two sprung wall panels) and the two filter files |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files. It hooks yam_arena without modifying it. Adapted from helpers in other task folders of this repository (their license applies). |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: the mechanism, every scoring level and violation, the demo's reach and clearances |

## Run a trial

From this folder, in a shell with the yam_arena environment active. New shells on the walkthrough's Brev instance
have it; elsewhere, `source <ARENA_DIR>/yam.env`.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the glovebox down, replace the cabin air filter behind it with the new one from the table, and put the glovebox back up." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/cabin-filter-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo runs open loop from the fixed, known layout. It uses only the arms' own joint readings and never reads the
glovebox's or the filters' poses from the simulator. A slow integral term on the measured joint error takes out the
sag from pushing and carrying.

1. **Squeeze and lower (both arms).** Each gripper, nearly level and turned 45 degrees inward, presses its closed
   fingertips onto the thumb pad on its side. The pose is placed from the fingers' real collision meshes. It pushes
   the pad in 22 mm (the panel stops at 14 mm), then
   draws the bin down along its arc from 48 to 75 degrees. Both arms let go, and the bin swings down onto its hinge
   stop.
2. **Old filter out (left arm).** The arm grips the tab level (approach 10 degrees below horizontal) and slides the
   filter 215 mm straight out. It lifts the filter over the lowered glovebox and carries it to the left of the dash.
   There it tips the filter back end down and lets go just above the table.
3. **New filter in (right arm).** Meanwhile the right arm picks the new filter up from the parts bin by its tab and
   waits beside the dash. When the old one is out, it brings the new one in front of the slot. It starts its far end
   4 cm in over the lead-in sill, so the slot now holds the filter level, and lets go. Then it pushes the filter home
   by its tab with the closed fingertips until the tab meets the housing, and backs out.
4. **Raise (both arms).** Both grippers reach level into the lowered bin's mouth. The tops of their fingertips push
   up on the bin's back wall (low on it at first, near its rim at the end), 10 cm either side of the middle, and swing the bin up to 2.5 degrees past
   its stop angle. The ramps squeeze the walls in on the way, and the bumps click back behind the fascia. The arms
   withdraw and the bin settles onto its stops.
5. Both arms return to their ready pose and open their fingers.

After every reset `scenekit.py` writes the hinge's start angle (just short of the stops; it settles onto them) and
the panels' spring targets. It then puts both filters back exactly at their start poses.

Measured on an L40S, the demo reaches 1.0 (success) in 1,112 steps (56 s). The squeeze pushes both panels in 14 mm
and the bin drops to its 110-degree stop. The new filter ends 0.0 mm deep and 2.1 mm to the side of its seated
position. Raised, the bumps ride up to 8.1 mm, click behind the fascia, and the bin settles on its stops at
47.6 degrees with both panels back out. Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step and latches violations. The score at the end of the episode counts
(`SCORE_OVER = "end"`).

**0 if at any time:**
- the glovebox is forced past a stop it was not squeezed clear of: a bump is more than 3 mm into the fascia. The
  scorer knows, for every hinge angle, how far each panel must be pushed in for its bump to clear (on a real car,
  forcing it breaks the stop);
- the glovebox is forced more than 3 degrees past its hinge stop;
- a filter falls off the table (its centre 5 cm below the table top). Dropping a filter onto the table is not a
  violation.

**Otherwise, at the end:**

| Score | Condition |
|---|---|
| 0 | the glovebox was never lowered past its stops |
| 0.25 | the glovebox has been lowered past both stops (72 degrees or more, at any time) |
| 0.5 | and the old filter is out of the dash section's footprint, resting on the table (lowest point within 1 cm of it), let go |
| 0.75 | and the new filter is fully in: front face at most 3 mm proud of seated, within 4 mm of the slot's centre line and floor, level and square within 3 degrees, let go |
| 1.0 | success, held for 1 s: all of the above, the glovebox back up on its stops (within 1.5 degrees of the stop angle or above, both panels sprung back out within 2 mm), every part of both grippers at least 2 cm from the glovebox and both filters, everything still |

- **Let go** means every part of both grippers is at least 2 cm from that filter.
- **Only the new filter counts as seated.** Putting the old one back in scores 0.25.
- **Doing nothing, or refusing,** scores 0.

On a real table these are all measurable: the hinge angle with a protractor, the filter's depth with a ruler against
the housing face, the stop bumps' clicking back by eye, and the 2 cm release distance from a photo.

## Simulation details

Rigid bodies at 240 Hz physics and 20 Hz control. The dash section is a fixed-base USD articulation. The glovebox
hangs on a damped revolute hinge (0.25 N m s/rad) with stops at 0 and 110 degrees. Each flexing upper side wall is a
sprung prismatic joint (900 N/m, 2.7 N preload, 14 mm travel). The stops are collisions between the POM bumps
(friction 0.15, combined "min") and the fascia. Grippers are checked as boxes from the arms' encoders, and the demo's
pad pose uses the fingers' real collision meshes.

## Real-world trials

- Use a dash section from a scrapyard (or a 3D-printed copy of one), clamped to the table, and new filters for both
  parts (one marked "old").
- Everything is light (filters 0.15 kg, glovebox 0.85 kg) and at table height. A dropped filter falls at most 30 cm
  onto the table.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
