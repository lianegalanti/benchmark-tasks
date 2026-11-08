# Hang an off-balance picture frame level on a pin by its sawtooth hanger

A dark wooden shadow-box frame (30 x 15 x 3.5 cm, 0.5 kg) stands upright in a grey stand in front of the robot. A
bronze medal is mounted behind its glass in the lower left, so the frame's centre of mass is 1.6 cm left of its
middle. On its back, under the top rail, is a sawtooth hanger: a bar with 11 V-notches, 8 mm apart. At the far edge
of the arms' reach, a narrow white board stands on the table with a 4 mm steel pin sticking 13 mm out of it, 15 cm up.
The robot must hang the frame on the pin by the hanger so that it hangs level by itself.

A frame on a single pin settles with its centre of mass straight under the pin. Hung from the hanger's middle notch,
the natural first try, this frame swings about 13 degrees crooked and comes to rest on its low corner on the table.
One notch off, it hangs 7.3 degrees crooked. Only the notch two teeth toward the medal (1.6 cm left of the middle)
hangs it level. A good policy hangs it, looks at how it hangs, and moves it along the teeth, as people do.
This differs from leaning a frame on a ledge: here the frame hangs from one point and swings to its own balance.

The board is narrower (10 cm) than the frame, so the grippers, which take the frame's top rail between pads on its
front and back faces, can let go with their rear fingers beside the board rather than trapped between frame and wall.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the monitor, `success()`, `score()` and the scripted demo (`demo()`) |
| `geometry.py` | Dimensions, where everything stands, how the frame hangs from each notch, and the tests on its pose |
| `make_assets.py` | Writes `assets/frame.usda` from `geometry.py` (`--check` verifies it is up to date) |
| `assets/frame.usda` | The frame as one compound rigid body: box colliders for the front, rails, hanger bar and teeth; explicit mass, centre of mass and inertia |
| `scenekit.py` | USD bodies and posed fixed shapes for yam_arena task files, with step and reset hooks |
| `demo.py` | Runs the demo, then closed-loop corrections on the measured frame pose; variants `fix` and `naive` |
| `test_task.py` | Offline checks (no Isaac Sim) |

`scenekit.py` is copied from the lock-before-loading task folder and extended so fixed shapes can set their surface
friction (that folder's license applies to it). It hooks `yam_arena.scene` and the embodiment's step and reset without
modifying yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and
`task_score` scorer. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

**Measured on an L40S:** the scripted demo ended with success after 306 steps (15.3 s); Dream-1 scored 0 in a 2,400-step trial.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the picture frame on the pin in the white board by the sawtooth hanger on its back, so that it hangs level by itself." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/frame-dream

# The scripted demo (task.py's demo(), both arms in step), which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the picture frame on the pin in the white board by the sawtooth hanger on its back, so that it hangs level by itself." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/frame-demo

# The same demo with closed-loop checks, and the variants; writes MP4s, setup.png and result.json
python demo.py task.py demo_out                     # demo(), then corrects if more than 1 degree off level
python demo.py task.py fix_out --variant fix        # hangs from the middle notch, then looks and corrects
python demo.py task.py naive_out --variant naive    # hangs from the middle notch and stops: 0.25

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The trial ends with 1.0 once `success()` has held for 1 s; otherwise it scores `score()` at the end
(`SCORE_OVER = "end"`). The monitor keeps its own milestones (lifted to the board, dropped) for the whole episode.

## Scripted demo

`demo()` in `task.py` (16 s, then a 3 s wait):
1. Both arms point down with their fingers closing front to back and take the frame's top rail 12 cm either side of
   its centre, 1 cm below its top (the pads cover its top 2.3 cm).
2. They lift it 4 cm out of the stand and carry it level to 4 cm in front of the board, the level notch (1.6 cm left
   of the hanger's middle) right above the pin with its teeth about 7 mm over the pin's top.
3. They move it in until its rails are 3 mm off the board (the pin now 8 mm under the hanger), lower it 1.1 cm so the
   notch takes the pin, and open.
4. The fingers rise over the rail and back away, out past the frame's ends.

`demo.py` plays the same plan, waits 2 s, and if the frame is more than 1 degree off level, corrects it, closed loop
on the measured pose (up to three passes). A correction pass:
- estimates how many teeth to move from the measured roll, taking the centre of mass to be at the frame's centre
  height (it does not know the medal's weight, so it under-shoots large tilts and may need a second pass);
- brings the open grippers in horizontally from past the frame's ends and grips the tilted rail. The raised end is
  too high to reach from above. Each arm grips at a wrist roll it can reach (the pads only clamp the rail's faces);
- turns the frame level about the raised end's grasp, so the notch near the middle rises off the pin;
- moves it level until the new notch is 1 cm over the pin, lowers it on, lets go and backs away.

The frame's pose is interpolated as one rigid body, so the two grasps never pull against each other.
`--variant fix` starts from the middle notch: it rests at about 13 degrees on its corner, the first pass moves one
tooth (to 7.3 degrees) and the second one more (level). `--variant naive` stops after the middle-notch hang.

## Scoring

- **0** if at any time the frame is dropped (its centre falls faster than 0.6 m/s, about a 2 cm free fall) or falls
  below the table top.
- **Hanging** means:
  - the pin sits in one of the hanger's notches: some point of the pin's axis within the hanger's thickness is
    within 2.5 mm of the notches' rest line (2.8 mm under the notch tops, where a 4 mm pin sits) and inside the row
    of notches;
  - nothing else holds it: every part of both arms is at least 2 cm from the frame, and the frame's lowest corner is
    at least 5 mm above the table.
- **1.0 (success):** hanging, the top edge within 2 degrees of horizontal (the top corners within 1.05 cm of the same
  height), and still (under 1 cm/s and 3 degrees/s) for 1 s.
- **Otherwise, at the end:**
  - 0.5 if hanging within 10 degrees of level;
  - else 0.25 if at some point it was lifted to the board: its lowest corner at least 2 cm above the table while its
    rearmost point was within 3 cm of the board's face;
  - else 0.

From each notch, frictionless, the frame hangs at:

| Notch (teeth from the level one) | Tilt | Score |
|---|---|---|
| level notch | 0 | 1.0 |
| 1 off | 7.3 degrees | 0.5 |
| 2 off (the middle notch is one) | 13-14 degrees, comes to rest on its corner at 10.4-13 | 0.25 |

Friction can hold the frame off its balance by at most about 2 degrees. That comes from the rails against the board
(friction 0.1) and the pin turning in its notch (0.3). It is too little to make a wrong notch look level, but it means
a frame let go tilted on the right notch may stay up to about 2 degrees off: let go of it level.

How the simulation judges these, once per control step: arm links are capsules of radius 3 cm between their joints,
and each gripper is a 3.5 cm capsule from its wrist joint to the fingertips. Tilt is the angle of the frame's top edge
(its local width axis) from horizontal, so leaning back against the board does not count.

## Runtime and seeds

A full 2,400-step trial with Dream-1 should take about 4 minutes on an RTX GPU, plus about 1 minute for Isaac Sim to
start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## What the first sim run should check

- The frame stands in its slot without toppling, and the two-arm grasp lifts it without slipping or twisting.
- Lowered on, the level notch catches the pin and the frame stays within 2 degrees once let go.
- The middle-notch hang (`--variant naive`) swings to its corner, and `--variant fix` finishes level.
- The frame's mass properties from `assets/frame.usda` survive spawning (centre of mass 1.6 cm left of the centre).
