# Stand a parcel on the pallet mark the way its THIS SIDE UP arrows point

A small brown cardboard parcel (7 x 7 x 14 cm, 300 g) lies on its side in front of the left arm, its length running
across the table. Each of its four long sides carries a white label with a bold black "this way up" arrow (ISO 780),
pointing along the parcel to one end, and that end's face is a red top panel. The arrows point to the robot's right.
The robot is told only: "Put the parcel on the green pallet mark." The green mark is a 12 x 12 cm square of
floor tape, in front of the middle of the robot.

To earn full score the robot has to:
1. read the handling marks: the parcel must end up standing on its end, arrows pointing up;
2. stand it up with a quarter turn, the right way round. Turning it with the left arm, gripper ending up pointing
   to the robot's right, raises the arrows' end; the same move mirrored with the right arm stands it upside down;
3. set it down standing on the mark and let go.

Category: **Impact on jobs.** Warehouse pickers, packers and loaders are told where a package goes, not which way up:
the package carries that instruction itself in its handling marks, and ignoring "this way up" spills liquids and
crushes contents. This is why the instruction does not mention the orientation; reading it off the parcel is part of
the job. A parcel set on the mark the wrong way up scores 0.5: it is in the right place, but it was mishandled.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the layout, the printed marks, and the geometric tests the scorer uses |
| `scenekit.py` | USD bodies and posed fixed shapes for yam_arena task files (copied unchanged from the `lock-before-loading` task folder in this repository; its license applies) |
| `make_assets.py`, `assets/parcel.usda` | The parcel: one rigid box collider with its labels, arrows and red top panel as visual-only prims (needs `usd-core` to regenerate) |
| `demo.py` | The scripted demo (left arm), closed loop on the measured parcel pose (its structure follows the `hang-bottle-on-drying-peg` demo in this repository) |
| `test_task.py` | Offline checks: layout, marks, every scoring level, and the demo played against a rigid-grip parcel |

`scenekit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the parcel on the green pallet mark." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/parcel-dream

# Offline checks (no Isaac Sim)
PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the trial
(`SCORE_OVER = "end"`; the score keeps its own memory of the 0.25 milestone, cleared at every reset, so that a parcel
knocked off the table can still zero it).

## Scripted demo

```bash
python demo.py task.py demo_out     # writes demo_*_cam.mp4, setup.png and result.json
```

`yam_scripted` plays only straight-down grips, so `demo.py` drives the embodiment itself. The left arm:
1. comes down over the parcel's centre with the gripper pointing down and tilted 20 degrees toward the robot's left,
   fingers closing along x on the two long sides facing forward and back (1.2 cm clearance each side while open), and
   grips;
2. lifts it 12 cm and turns the gripper 90 degrees about the finger axis while carrying it over the mark: the arrows'
   end, which pointed to the robot's right, now points up, and the gripper points right and 20 degrees down;
3. measures the parcel's pose and lines it up upright over the mark's centre, its bottom 3 cm above the tape,
   correcting any tilt it took in the grip; measures again and corrects once more;
4. lowers it until the bottom is 4 mm above the tape, opens, backs the gripper 12 cm out along its own axis, and
   returns to the ready pose.

The plan is 420 steps (21 s); success needs 1 s more. Offline (`test_task.py`), against a parcel held rigidly
in the grip: IK error under 0.5 mm and 0.1 degree, joints within limits with no branch jumps, and the arm's URDF
collision hulls stay 2.0 cm above the tape and table, 2.2 cm from the parcel (open fingers 1.0 cm from it); the
parcel ends centred on the mark, red end up.

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | standing on the mark the right way up (below), let go, still for 1 s | photo from above and from the side |
| 0.5 | at the end, resting on the mark, let go and still, any other way: upside down, lying, or leaning more than 15 degrees | photo |
| 0.25 | at some point lifted (its lowest point 3 cm or more above the table) or stood upright, either end up, anywhere | video, against a 3 cm mark |
| 0 | otherwise, or the parcel went off the table (its centre 5 cm below the table top) at any time | video |

- **On the mark:** the parcel's centre, seen from above, within 3 cm of the mark's centre, and its lowest point within
  1 cm of the tape's top. Standing, that is the centre of its footprint.
- **The right way up:** its arrows' direction (the line from the plain end to the red end) within 15 degrees of
  straight up.
- **Let go and still:** no gripper holds it, every gripper's centre line (wrist to fingertips) is at least 2 cm from
  it, and it moves under 2 cm/s.

How the simulation judges these, once per control step: the parcel's pose gives its eight corners; a gripper
**holds** it when its grasp point is within 1.5 cm of the box and its fingers are less than 7.8 cm apart.

## Runtime and seeds

A 1,200-step trial with Dream-1 takes a few minutes on an RTX GPU, plus Isaac Sim's start. The layout is fixed
(`-E object_jitter=0`), so there are no seeds to choose. The demo measures the parcel's pose before lining it up, so
it does not depend on exact timing.

On an L40S the scripted demo ended with success after 378 steps (18.9 s); Dream-1 scored 0 in a
2,400-step trial.

## Fidelity

Rigid bodies, 240 Hz physics. The parcel is one rigid box with sharp edges and an even mass; a real carton flexes a
little in the grip and has rounded edges. Its labels, arrows and red panel are visual-only prims 0.5 mm proud of its
faces. Friction 0.9 static, 0.8 dynamic. The tape is a fixed 2 mm slab. Contents, crushing and spills are not
simulated. Positions handed to Isaac Lab are plain Python floats.
