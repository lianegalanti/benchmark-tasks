# Hold a loose bar still while pressing a dowel pin into it

An orange bar (31 x 31 mm, 17 cm long, 171 g) lies loose on a PTFE sheet. Its left end has a square blind hole, 19 mm
across and 7 cm deep. A steel dowel pin (16 mm across, 7 cm long, 110 g) lies on the table in front of the left arm.
The robot must press the pin in until it is flush with the bar's end, without the bar moving.

The bar slides on the sheet at the slightest push. A pin pushed in by one arm drags the bar along:
- its friction on the hole's floor is more than three times the bar's friction on the sheet;
- once the pin bottoms, every further push moves the bar.

So one gripper has to hold the bar while the other presses. The instruction says so, and the scorer checks it: the
bar may never move more than 1 cm at any time in the trial. Pushing the bar back afterwards doesn't help.

The pin goes in flush, so no gripper can hold it to the end. The pads have to let go part way and push its end with
the fingertips, as a fitter finishes a dowel with a thumb or a drift. The pin then rests on the hole's floor.

The insertion is horizontal, along the bar. A vertical press would only push a lying part into the table, not
sideways, so a hole in the top face would not need holding.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the bar-movement monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, masses, friction and layout (station frame) |
| `make_assets.py` | Writes `assets/bar.usda` and `assets/peg.usda` from `geometry.py` (plain-text USD; `--check` compares) |
| `assets/` | The bar (one rigid body of five boxes around the cavity) and the pin (one cylinder) |
| `scenekit.py` | USD bodies and posed fixed shapes for yam_arena task files |
| `demo.py` | The scripted demo, closed loop on the measured pin and bar poses |
| `test_task.py` | Offline checks (no Isaac Sim) |

`scenekit.py` and the runner in `demo.py` are copied from the accepted task `lock-before-loading` and extended:
- `Shape` takes its own friction, for the low-friction sheet;
- free USD bodies get the scene's 5 mm contact offset.

`scenekit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena. The task runs
with the standard `yam_arena` embodiment and `task_score` scorer. Written against yam-isaac-lab-arena `bd62771`.

**Status: the scripted demo scores 1.0 in Isaac Sim** (`python demo.py task.py <out>`): the pin seated flush in 629
control steps (31 s) of the 2,400 allowed, with the bar's largest movement 3.5 mm (limit 1 cm). The Dream-1 run is
done: Dream-1 scored 0 in a 2,400-step trial. The `one_arm` variant has not been run in the simulator.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Press the steel pin all the way into the hole in the end of the orange bar, until it is flush with the end. The bar is loose on a slippery sheet and must not move: hold it still with one gripper while the other presses the pin in." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/hold-press-dream
```

`task_score` prints the trial's score. It is 1.0 when the trial ends in success, otherwise the best `score()` reached
(`SCORE_OVER = "best"`).

## Scripted demo

```bash
python demo.py task.py demo_out                       # writes demo_*.mp4, setup.png, result.json
python demo.py task.py one_arm_out --variant one_arm   # no holding arm: the bar should be shoved away
DEMO_DEBUG=1 python demo.py task.py demo_out --no-record   # prints the insertion and push loops
```

1. **Hold.** The right arm comes down over the bar's solid end, 11.5 cm from the mouth (2 cm from its centre of
   mass), and closes its fingers across it:
   - before closing, it centres and squares its open fingers on the bar's measured pose, from the measured finger
     positions and wrist rotation;
   - while closing and holding, it moves and turns to cancel any measured drift or turn of the bar (closing turns it
     by about 4 degrees, and the arm gives about 1 mm per newton of push).
2. **Insert.** Meanwhile the left arm picks up the pin by its outer end, where it measures it just before closing,
   with its open fingers centred on it. It carries the pin behind the mouth. The pin droops about 6 degrees in the
   grip, so the arm turns the gripper until the measured pin axis is on the hole's axis. It lines the pin up on the
   axis using the grip offset it measured. Once the bar is held, the arm pushes the pin in at 1.5 cm/s, closed loop:
   - it steers the leading end onto the axis before entry, then the point where the pin crosses the mouth;
   - it waits for alignment within 1.2 mm before entry;
   - it backs off 3 mm when stalled.

   At 4 cm deep, the pin's centre of mass is inside the hole and the pads are 1 cm from the bar, so the arm lets go.
3. **Press flush.** The left arm closes its fingers, leans them 50 degrees forward and comes down 1.5 cm behind the
   pin's end. The finger widens above its tip (from 1.5 cm across at the tip to 2.6 cm 13 mm above it), so a vertical
   finger hits the bar's end face above the hole about 6 mm before the pin is flush. Leaning, the tips lead and the
   rest of the finger trails at least 5 mm behind the face (checked against the finger collision meshes in
   `test_task.py`). The tips push the pin's end at up to 1 cm/s until it is within 0.5 mm of flush. They stay at
   most 3 mm (up to 8 mm if the pin sticks) ahead of its measured end, so the push stays gentle: a hard push tips the
   pin, wedges it, and shoves the holding arm.
4. **Withdraw.** The left arm backs off; then the right arm lets go and lifts.

In Isaac Sim the demo takes 629 control steps (31 s) and scores 1.0, with the bar moving at most 3.5 mm. Offline, in
a kinematic stand-in, it takes 589 control steps. Its IK error stays under 0.08 mm and 0.02 degrees, and the two
arms' link axes stay at least 9.6 cm apart.

## Scoring

Checked after every control step:
- **The bar's movement** is the larger displacement of the centres of its two end faces from where they lay after
  the reset. The scorer keeps the largest value reached during the trial. On a real table, mark the bar's outline
  on the sheet and check the video.
- **Seated:** the centres of both ends of the pin are inside the hole's cross-section, and its outer end is at most
  2 mm proud of the bar's end face. Either end of the pin may lead.

| Score | Condition |
|---|---|
| 1.0 | seated, pin and bar still (under 1 cm/s) for 1 s, and the bar never moved more than 1 cm |
| 0.5 | at some point seated while the bar had not moved more than 5 cm |
| 0.25 | at some point the pin's leading end at least 5 mm inside the hole |
| 0 | anything else (the pin never entered, or only by less than 5 mm) |

Some outcomes and how they score:
- One-arm pushing usually shoves the bar along with the pin. It scores 0.25, or 0.5 if the pin still got home within
  5 cm.
- Lifting the bar to insert the pin in the air moves it, so it can't score 1.0.
- A pin lying on top of the bar or beside it is not inside the hole's cross-section, so it scores 0.

## Simulation details

- The sheet's friction is 0.08 static and 0.06 dynamic, combined as the minimum, so it governs the bar's contact.
  The bar and pin have friction 1.0 static and 0.9 dynamic. The pin has a 8 mm torsional patch so the pads resist it
  pivoting while held by one end.
- PhysX has no rolling resistance, so the lying pin rolls slowly toward the robot (about 3 mm/s). It has an angular
  damping of 5/s as a stand-in for a real table's rolling resistance. The demo picks it where it is measured.
- Measured in the demo run in Isaac Sim:
  - the held pin droops 6 to 7 degrees in its grip; the demo corrects it to 0.2 degrees by turning the gripper;
  - the pin enters with 1.5 mm clearance per side and no chamfer, within 0.6 mm of the axis;
  - let go at 4 cm deep, it drops 1.5 mm onto the hole's floor and lies flat;
  - squared and centred before closing, the holder moves the bar 2.9 mm as it closes;
  - pushed with the fingertips, the pin slides home freely while the push stays gentle.
- Not yet checked in the simulator: the `one_arm` variant, which should shove the bar more than 1 cm.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose. A full 2,400-step Dream-1 trial should
take about 4 to 5 minutes on an RTX-class GPU, plus Isaac Sim's start-up. This estimate comes from comparable tasks;
it has not been measured.
