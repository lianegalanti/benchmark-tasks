# Hang a washed baby bottle mouth-down on a drying-rack peg

A washed baby bottle (a straight wide-neck bottle without its teat and cap: an open tube 6 cm across and 12 cm long,
closed at one end, amber screw collar at the mouth) lies on its side on a dish towel in front of the left arm, its
mouth toward the robot. Ahead of it stands a drying rack: a white base with two upright pegs, 12 mm thick and 15 cm
tall, 12 cm apart. The robot is asked to hang the bottle mouth-down on a peg to dry.

To do it, the robot has to:
1. pick the bottle up across its side;
2. turn it a quarter turn in the hand, so the mouth that pointed at the robot points straight down;
3. line its 5.4 cm bore up over a 1.2 cm peg and lower it until the peg's tip carries the bottle by its inside
   bottom; the peg disappears inside the bottle, so the last 12 cm of the insertion is out of sight;
4. let go and move the gripper away, leaving the bottle hanging still with its rim clear of the base.

Category: **Societal relevance.** Feeding bottles are dried upside down on a rack's pegs, so water drains out and
air reaches the inside without the rim sitting on a wet surface; homes, nurseries and hospital milk rooms do this
many times a day. It is a household chore that needs an in-hand reorientation and an insertion by a part of the object
the robot can't see at the end, rather than a top-down pick and place.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the layout, and the geometric tests the scorer uses (peg depth, tilt, the bottle's lowest point) |
| `scenekit.py` | USD bodies and posed fixed shapes for yam_arena task files (copied unchanged from the `lock-before-loading` task folder in this repository; its license applies) |
| `make_bottle.py`, `assets/bottle.usda` | The bottle: one rigid body of 16 wall staves and a bottom disc (needs `usd-core` to regenerate) |
| `demo.py` | The scripted demo (left arm), closed loop on the measured bottle pose |
| `test_task.py` | Offline checks: layout and clearances, every scoring level, and the demo played against a rigid-grip bottle |

`scenekit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the washed baby bottle mouth-down on a peg of the drying rack to dry." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/bottle-dream

# Offline checks (no Isaac Sim)
PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the trial
(`SCORE_OVER = "end"`; the score itself keeps the best milestone reached, cleared at every reset, so that a bottle
knocked off the table can still zero it).

## Scripted demo

```bash
python demo.py task.py demo_out     # writes demo_*_cam.mp4, setup.png and result.json
```

`yam_scripted` plays only straight-down grips, so `demo.py` drives the embodiment itself. The left arm:
1. comes down over the bottle's centre of mass with the gripper pointing down and tilted 20 degrees back toward the
   robot, fingers closing along y on the two side staves (1.7 cm clearance each side while open), and grips;
2. lifts it and turns the gripper 90 degrees about the finger axis while carrying it up to 25 cm: the mouth now
   points down and the gripper points forward and 20 degrees down. (With an untilted grip the gripper would have to
   point straight forward at the end, which the arm can't do this close to the table: joint 4 runs out.)
3. measures the bottle's pose and lines its axis up over the left peg, mouth 2.5 cm above the tip, correcting any
   tilt the bottle took in the grip; measures again and corrects once more;
4. lowers it 14 cm straight down over the peg in 4 s, until the inside bottom is 4 mm above the tip; opens; backs the
   gripper 12 cm out along its own axis; returns to the ready pose.

The plan is 482 steps (24 s); success needs 1 s more. Offline (`test_task.py`), against a bottle held rigidly in
the grip: IK error under 0.5 mm and 0.1 degree, joints within limits, and the arm's URDF collision hulls stay 1.7 cm
above the towel, 2.3 cm from the pegs and 2.5 cm from the bottle (open fingers 1.6 cm from it); while the mouth is
below the peg's tip the peg is within 0.6 cm of the bore's axis (2.1 cm of clearance all round).

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | hanging on either peg (below), no gripper holding it, both grippers at least 5 cm from its surface, still (under 2 cm/s) for 1 s | photo from the side, ruler |
| 0.5 | at some point on a peg: a peg's tip at least 3 cm inside the bottle through its mouth, the bottle's axis within 30 degrees of vertical, mouth down (held or not) | video; a 3 cm mark on the peg below its tip is hidden |
| 0.25 | at some point the bottle's lowest point was at least 3 cm above the table (lifted) | video, against a 3 cm mark |
| 0 | otherwise, or the bottle fell off the table (its centre 5 cm below the table top) at any time | video |

**Hanging** means on a peg with the bottle's lowest point (the mouth rim) at least 5 mm above the rack base. Resting
on the base with the peg inside is impossible (the peg is 2.2 cm longer than the bore is deep), so a bottle on a peg
either hangs from the tip or leans on the peg; leaning is fine as long as the rim is clear. A bottle stood mouth-up,
or laid on its side, on the rack is never on a peg: the peg rises from below and can only enter a mouth facing down.

How the simulation judges these, once per control step: the peg's tip is transformed into the bottle's frame (inside
the 2.7 cm bore radius, between the mouth plane and the bottom); the rim's lowest point from 24 points round each end;
a gripper **holds** the bottle when its grasp point is within 5 cm of the bottle's axis and its fingers are less
than 6.8 cm apart; a gripper's **centre line** runs from its wrist (the gripper link) to its fingertips.

## Runtime and seeds

A 1,200-step trial with Dream-1 takes a few minutes on an RTX GPU, plus Isaac Sim's start. The layout is fixed
(`-E object_jitter=0`), so there are no seeds to choose. The demo measures the bottle's pose before lining it up, so
it does not depend on exact timing.

On an L40S the scripted demo ended with success after 445 steps (22.3 s); Dream-1 scored 0 in a 2,400-step trial. The first run should check that the bottle lies still on the towel
after settling, that the pinch holds it through the turn without it pivoting about the finger axis (it is gripped at
its centre of mass, and `torsional_patch` is set), and that, let go 4 mm above the tip, it settles hanging still within
a second.

## Fidelity

Rigid bodies, 240 Hz physics. The bottle is a 16-sided tube 6.0 cm across its flats, so it lies on a flat without
rolling; a real round bottle can roll a little on the towel. Its friction is 0.9 static and 0.8 dynamic; the rack and
pegs are fixed. Flex, water and drips are not simulated. Positions handed to Isaac Lab are plain Python floats.
