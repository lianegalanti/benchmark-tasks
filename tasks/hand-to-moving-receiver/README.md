# Hand a part into a receiving cup that keeps moving, without dropping it in

A bare aluminium cylinder (4 cm across, 7 cm tall) stands on the table between the arms, toward the right arm. A
blue open receiving cup (10 x 10 cm inside, rim 2.5 cm above its floor) is carried by a two-axis drive in front of
the right arm, like a coworker's hand or another machine's gripper held out for a part: it wanders slowly and
unevenly over an 8 x 18 cm area, in moves of 2.5 to 5 s at up to 6 cm/s, with brief stops of 0 to 0.8 s, and never
waits for the robot. The robot must hand the cylinder over: carry it to the cup, move along with it, set the
cylinder down inside and let go only when its base is at most 2 cm above the cup's floor, with the gripper moving
with the cup (within 3 cm/s). Dropping or tossing it in, or letting go while the gripper and cup move differently,
earns only partial credit.

Handing a part to a receiver that doesn't hold still is common wherever people and machines share work: a person
reaching for a part, another robot's gripper, a fixture on a moving axis. A part dropped from height gets damaged or
bounces out, and a gripper that doesn't follow the receiver knocks it. The skill tested is matching the receiver's
motion closely enough to place, not throw.

This differs from `../drop-into-moving-tote/`, where the tote cruises at a constant 10 cm/s on a straight, regular
back-and-forth and dropping the block in from up to 5 cm above the rim scores full. Here the receiver moves slowly
but irregularly in two directions (no constant speed to lock onto, and its stops last at most 0.8 s), and the
score depends on how the part was let go, not only on where it ends.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the cup's path, the handover monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind (a box whose pose and velocity follow a scripted path, set before every physics step), `on_step` / `on_reset` hooks with `arm_joints()` for a monitor, `stop_and_go()` (the cup's path), `offset_path()` and `clock()` |
| `test_task.py` | Offline checks (no Isaac Sim): the cup's path and reach, the handover monitor and every scoring level, and that the demo is reachable and follows the cup at every control step through the release |

`motionkit.py` is copied from this repository's `reach-between-sweeps` task folder (with `offset_path()` from
`drop-into-moving-tote`) and extended with `stop_and_go()` and `clock()`. It hooks `yam_arena.scene` and the
embodiment's reset, simulation step and task scoring without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The cup's clock is the episode clock
(`observation.state_time`): it stands still at its start through the reset and starts with the first control step.
Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured on an L40S:** the scripted demo
ended with success after 206 steps (10.3 s); Dream-1 scored 0 in a 2,400-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hand the aluminium cylinder over into the blue cup. The cup keeps moving and won't wait for you: move along with it, set the cylinder down inside, and let go only when its base is no more than 2 cm above the cup's floor. Don't drop or toss it in." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/handover-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hand the aluminium cylinder over into the blue cup. The cup keeps moving and won't wait for you: move along with it, set the cylinder down inside, and let go only when its base is no more than 2 cm above the cup's floor. Don't drop or toss it in." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/handover-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/handover-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
`score()` at the end (`SCORE_OVER = "end"`). Use `-E control_hz=20`: the demo puts a waypoint on the cup at every
0.05 s control step.

## Scoring

A **release** is the moment a gripper that was holding the cylinder stops holding it. All positions and speeds of the
cylinder and gripper are taken relative to where the cup is at that moment.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | the cylinder rests in the cup (centre inside the walls, base within 1 cm of the floor, moving with the cup to within 1 cm/s and turning under 0.5 rad/s, for 1 s, no gripper holding it), and its **last** release was gentle: the cylinder's centre over the cup's inside, its base 2 cm or less above the cup floor, and the gripper moving within 3 cm/s of the cup | video against a 2 cm mark on the cup's inner wall; relative speed from the arm's logged pose and the drive's logged position (or frame-by-frame video) |
| 0.5 | at the end the cylinder rests in the cup, but it got there otherwise: dropped or tossed from higher, let go while moving differently from the cup, or let go over a wall | a photo at the end |
| 0.25 | at some point the cylinder's base was 3 cm or more above the table (it was picked up) | video, against a 3 cm mark |
| 0 | anything else: never lifted, pushed over, or left outside the cup | |

Lying on its side in the cup still counts as in the cup (0.5, or 1.0 after a gentle release). Holding the cylinder
in the cup without letting go isn't a handover (0.25). A cylinder dropped in and then taken out and handed over
gently scores 1.0: only the last release counts.

### The handover monitor

After every control step (20 Hz), before scoring, the monitor reads each gripper's TCP (forward kinematics of the
measured arm joints) and the gap between its finger pads (from the finger joints). A gripper **holds** the cylinder
while its pads are closed on it (gap 3.0 to 4.6 cm; the cylinder is 4 cm across) and the cylinder's centre is within
3.5 cm of its TCP. The first step at which it no longer holds it is the release. The monitor then records, from the
last step it held: the cylinder's base height above the cup floor and whether its centre was over the cup's inside;
and the gripper's speed relative to the cup over the 0.05 s before that step and the 0.05 s after it (the larger of
the two), from the TCP's and cup floor's measured positions. A free fall of 2 cm takes 0.064 s and reaches 63 cm/s,
so tossing the part in from higher is far from the limit; so is following the cup only roughly.

## Runtime and seeds

- The cup starts at rest and loops every 16.8 s. The scripted demo (right arm) picks the cylinder from the top
  (lifted by 3.9 s), carries it over the cup by 6.0 s, then aims its TCP at where the cup's centre will be 0.05 s
  later at every control step (the arm's joint PD trails a moving target by about 60 ms). It sets the cylinder down
  on the cup floor while following (6.5 to 7.8 s), rides along, opens the fingers slowly from 8.5 s while the cup
  moves at 5.6 cm/s near its steady peak (the grip fades from 50 N to 0 by 8.95 s; the monitor sees the release
  about 9.1 s), rides along open until 9.8 s, rises while following and parks (11.4 s in all). In Isaac Sim it ended with
  success after 206 steps (10.3 s).
- Why set down rather than let go just above the floor: the arm's gravity compensation doesn't include the 240 g
  cylinder, so while holding it the TCP sags about 5 mm and springs back within a few control steps when the
  cylinder leaves the fingers in mid-air (some 10 cm/s, well over 3 cm/s). Set down on the floor (commanded as if
  there were no sag), the floor carries the cylinder and the arm has nothing to spring back from; a grip higher or
  lower on the cylinder than planned only moves its base a few mm, far under the 2 cm limit.
- `HANDOVER_DEBUG=1` in the environment prints, every held step, the gripper's pad gap, TCP relative to the cup and
  the cylinder's base above the cup floor, and each release the monitor records.
- Dream-1 scored 0 in a 2,400-step trial (2 minutes, about seven loops of the cup).
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start.

## Fidelity

The cup is driven exactly along its path, like a stiff servo axis: the arms can't push it off its path or stop it,
where a real drive would fault. It glides 2 mm above the table without touching it. In the simulation it is five
boxes (a floor and four walls, 1 mm apart so the driven parts never touch each other) that move as one; a real cup
is one moulded piece on a carriage. Holding and releasing are read from joint positions every 0.05 s, so the release
moment is known to within one control step. The cylinder has the station's rubber-like friction (1.2 static, 1.0
dynamic), more than aluminium on plastic; the cup's peak acceleration (under 0.1 m/s²) can neither slide nor tip it.
