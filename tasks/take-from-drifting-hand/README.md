# Take a part from a person's hand that keeps drifting while they hold it out

A person across the table holds out a yellow tool block (10 x 2.5 x 5.5 cm, 140 g) for the robot on their flat
palm, between two low stops. While they wait, their hand drifts the way a held-out hand does: small loops that keep
shifting, about 7 x 10 cm across and 0.9 cm up and down, at 2 to 5.6 cm/s, never pausing and never retracing its
path. The robot must take the block from the hand, lifting it straight up off the palm without knocking it off or
bumping the hand, and put it in the grey tray at the right.

This is the reverse handover, from an irregular, human-like motion. It differs from this repository's
`../hand-to-moving-receiver/` (the robot *gives* a part, setting it into a cup that moves on a fixed stop-and-go
loop) and `../harvest-swaying-fruit/` (picking from a periodic figure-of-eight sway, which repeats every 4.8 s): here
the target's motion is not periodic, so it can't be learned or anticipated as a cycle and must be tracked, in three
dimensions, and the part must leave the hand in one way only (straight up), with a distance rule standing in for
"don't bump the person".

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the hand, the block and the tray, the drift, the monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step; `on_step`/`on_reset` hooks for per-step monitors; and `drift`, the never-stopping, non-repeating wander |
| `test_task.py` | Offline checks (no Isaac Sim): the drift (smooth, never pausing, never repeating), the hand's geometry, the monitor and every scoring level, and that the demo is reachable, follows the hand and keeps clear of it |

`motionkit.py` is copied from this repository's `../harvest-swaying-fruit/` (same authors), with `drift` added. It
hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena, so the task runs
with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The drift's clock is the
episode clock (`observation.state_time`): the hand stands at its start through the reset and starts with the first
control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1).

**Measured on an L40S:** the scripted demo ended with success after 214 steps (10.7 s); Dream-1 scored 0 in a 2400-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
INSTR="The person is holding out the yellow block for you on their palm. Take it from their hand and put it in the grey tray. Their hand keeps drifting while they wait, so follow it, lift the block straight up off the palm, and don't bump their hand."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/drifthand-dream

# The scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/drifthand-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/drifthand-*/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s; otherwise it scores
`score()` at the end of the trial (`SCORE_OVER = "end"`), which reads flags the monitor keeps over the whole trial.

## The hand's drift

The middle of the palm's top face starts at (30, -17, 8) cm (in front of the bases' line, toward the left arm from
their midpoint, above the table). Its offset from there, with `u` the hand's clock in seconds and angles in degrees:

    x = 1.8 sin(360 u / 3.8) + 1.5 sin(360 u / 14.6)
    y = 2.5 (1 - cos(360 u / 3.8)) + 2.0 sin(360 u / 16.6) + 0.8 sin(360 u / 6.7)
    z = 0.3 sin(360 u / 6.8) + 0.2 sin(360 u / 3.4)

The 3.8 s loop keeps the hand moving (it never drops below 2 cm/s); the slower terms move the loop around and bob it.
The periods share no common multiple shorter than years, and the tests check that within the 2-minute trial no delay
brings any 40 s stretch of the path back to within 2 cm of itself. The hand starts at rest and its clock's pace
grows as a smoothstep over the first 2 s. Acceleration stays under 0.1 m/s², far below what would slide the block on
the palm, so it rides along until touched.

## Scoring

Every control step (0.05 s) a monitor reads the palm's measured position and checks:

- **How the block leaves the palm.** It is in its seat while its centre is within 1.5 cm sideways of where it sits
  seated and within 1.5 cm above or below that height. The first time it is out, it was *taken up* if its centre was
  more than 1.5 cm above its seat and still within 1.5 cm sideways, and *knocked off* otherwise (pushed, dragged or
  tipped off). It is *taken* once taken up and its centre has been 3 cm or more above its seat.
- **Whether the robot bumps the hand**: any part of either arm within 1 cm of the palm, the stops or the forearm. The
  robot is modelled conservatively as capsules: each arm link between consecutive link origins (radius 4 cm), the
  gripper housing down to where the fingers start (5 cm), the wrist camera (2.5 cm) and each finger at its measured
  opening (8 mm). A real gripper flagged here passes within a few cm of the hand, never closer than 1 cm.

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | taken up, the robot never within 1 cm of the hand, and the block rests in the tray (centre inside the walls, underside within 1 cm of the floor, still for 1 s) | side video of the hand against a 1 cm mark; a photo of the tray |
| 0.5 | at the end the block rests in the tray, having been taken up, but the robot came within 1 cm of the hand | the same |
| 0.25 | taken (lifted straight up, 3 cm above its seat), but not in the tray at the end: dropped, or still held | video against a 3 cm mark |
| 0 | anything else: the block knocked, pushed or tipped off the palm (even if it then ends in the tray), never taken, or lifted less than 3 cm and left | |

Knocking the part off the hand and picking it up from the table scores 0, so the only way to credit is to take it
from the moving hand.

## Runtime and seeds

- The scripted demo (right arm) is over the block by 2.5 s and then puts a waypoint on the hand's drift every control
  step, aimed one step ahead, in all three axes: it descends around the block (3.0 to 4.2 s), closes (4.4 to 5.0 s)
  and lifts it 4.5 cm straight up (5.2 to 6.0 s), all while following, then moves up and away, carries it to the tray,
  lowers it to 1 cm above the floor, lets go at about 10 s and parks by 11 s. Planned, its closest approach to the
  hand is 2.9 cm (the fingertips over the palm while gripping), 1.9 cm beyond the 1 cm rule and more than three
  times the hand's 0.5 cm bob, so a height lag of a whole bob still leaves 1.4 cm to spare.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The drift always starts from rest at
  the same point.

## Fidelity

The hand is rigid and follows its path exactly, like a stiff drive: the arms can't push it or stop it, and it
doesn't flinch, yield or close as a person's hand would. Bumping it is judged by distance, not force (the sim's
driven hand can't feel a touch). The hand is four boxes (palm, two stops, forearm) 1 mm apart, moving as one; the
block is a rigid box with the station's rubber-like friction (1.2 static, 1.0 dynamic). A real trial needs the mock
hand on a three-axis stage that can play the path back; a person can't repeat it exactly.
