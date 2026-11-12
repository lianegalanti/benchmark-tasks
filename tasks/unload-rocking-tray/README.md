# Unload sample bottles from a rocker that keeps rocking

A laboratory rocker stands between the arms, 25 cm in front of them: a black rubber-matted platform (18 x 44 x
1.5 cm, long side from left to right) that rocks like a see-saw about its centre line, 10 degrees each way, one full
rock every 4 s, and never stops. Three square plastic sample bottles (4 x 4 x 6 cm) stand on the mat in a row,
orange at the left end, green in the middle, purple at the right end; the mat holds them, so they ride with it: the
end ones rise and fall 3 cm each way and lean 10 degrees with the platform. The robot must take all three off and
leave them standing upright in two grey bins on the table beside the rocker's ends, one in front of each arm, without
knocking any off the platform or dropping one on the table. Knocking a bottle off scores the whole trial 0, even if
the robot then picks it up and puts it in a bin.

Rockers and platform shakers keep samples moving through long incubations, staining and washing steps, and the
bottles on them are often taken off without stopping the run; parts and samples on a moving vehicle or a ship's deck
pose the same problem. The difficulty is timing: a grasp aimed where a bottle was meets it 1 to 3 cm higher or lower
and tilted, and a gripper that comes down while an end is rising hits the bottle or the mat. The platform pauses for
an instant at each turn of the rock, with one end high; that is the moment to close on a bottle.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the rocker's motion, the monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds a `DrivenBox` object kind to yam_arena task files: a box whose pose and velocity follow a scripted path, set before every physics step, so objects on it ride along by friction; and `on_step` / `on_reset` hooks. A path can also tilt the box (`rocking()`, with its angular velocity written too) |
| `test_task.py` | Offline checks (no Isaac Sim): the rocking motion and clearances, the scoring of every outcome (including a bottle knocked off and then put in a bin, pushed or flung off unheld, dropped on the table, and standing in a bin but not let go), and that the demo is reachable, grips each bottle over a turn, carries it past the platform's edge with its fingers closed, keeps clear of the mat, the other bottles and the bins, and ends with both grippers clear of every bottle |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without
modifying yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and
`task_score` scorer. The rocker's clock is the episode clock (`observation.state_time`): it stands level through the
reset and starts rocking with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac
Lab-Arena 0.3.1). **Measured on an L40S:** the scripted demo ended with success after 389 steps (19.5 s);
Dream-1 scored 0 in a 2,400-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the three bottles off the rocking platform and leave them standing upright in the grey bins. The platform keeps rocking and won't stop for you; don't knock any bottle off it or drop one on the table." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/rocker-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the three bottles off the rocking platform and leave them standing upright in the grey bins. The platform keeps rocking and won't stop for you; don't knock any bottle off it or drop one on the table." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/rocker-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/rocker-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` (no bottle knocked off, and all
three standing in the bins, let go and still) has held for 1 s; otherwise it scores `score()` at the end of the trial (`SCORE_OVER = "end"`).

## Scoring

**Knocked off: the whole trial scores 0.** If at any moment a bottle is knocked off, the score is 0 for the whole
trial, even if the robot later picks the bottle up and stands it in a bin. A bottle is knocked off when

- its centre passes beyond an edge of the platform (seen square to the platform) while no gripper is closed on it, or
- it touches the table anywhere outside the bins (its underside within 5 mm of the table top, its centre outside the
  bins' walls): knocked off the platform, or dropped while being carried.

A gripper is closed on a bottle only when its fingers are stopped by it: they are 3.5 to 5.2 cm apart (the bottle
is 4 cm across) and its grasp point (midway between the fingertips) is within 1 cm of the bottle's centre line,
between its ends. The bottle is pinched between the fingers, not pushed along or flicked off by them; a fully closed
gripper pressed against its side does not count as holding it. On video this is judged frame by frame: a bottle must leave the
platform pinched between a gripper's fingers; one that slides, tips, rolls or is pushed or flung past an edge, or
that lands on the table, ends the trial at 0. A monitor applies the same test after every control step and latches
it for the rest of the trial.

**Otherwise** the score is the mean of the three bottles' credits:

| Credit | Condition (for one bottle, at the end of the trial) | Check on a real table |
|---|---|---|
| 0 | it was never taken off the platform (below) | video |
| 1 | it stands in a bin and is let go: its centre inside the bin's walls, its underside within 1 cm of the bin floor, its axis within 20 degrees of vertical, and both grippers' centre lines (wrist to fingertips) 8 cm or more from its axis, so the open fingers, 4.7 cm either side of that line, are 3 cm or more clear of it | a photo from above and one from the side |
| 0.5 | its centre is inside a bin's walls and its underside less than 5 cm above the rim: lying or leaning in a bin, still held there, or standing with a gripper within 8 cm of its axis | photo |
| 0.25 | anywhere else (still held, or put back on the platform) | video |

"Taken off the platform": at some moment every corner of the bottle was 3 cm or more above the mat (extended as a
plane) directly beneath it. The trial ends with 1.0 once all three bottles are standing in the bins, let go and
still (under 1 cm/s), for 1 s.

## Runtime and seeds

- The platform's tilt is 10 degrees x sin(2 pi t / 4 s), positive raising its left end: the left end is highest at
  1, 5, 9 s ..., the right end at 3, 7, 11 s .... At those turns it stands still for an instant.
- The scripted demo (open loop, timed against the rock): the left arm waits over the orange bottle, comes down around
  it 0.5 s before the turn at 5 s, closes over the turn, lifts 0.25 s after it and stands the bottle in the left bin
  (let go at 9.8 s); then it takes the green bottle the same way over the turn at 13 s (let go at 17.8 s). The right
  arm takes the purple bottle over the turn at 7 s and stands it in the right bin (let go at 11.8 s). Each bottle is
  gripped across its front and back faces, which stay vertical however the platform tilts; it is let go leaning as
  it was gripped (10 degrees), its lowest corner 1 cm above the bin floor, and settles upright (it would tip only past
  34 degrees). After each let-go the gripper rises 11.7 cm straight up, its fingertips ending 9.7 cm above the
  bottle's top, clear of it by the 8 cm the full credit needs. In Isaac Sim it ended with success after 389 steps
  (19.5 s).
- `test_task.py` checks, on commanded positions: the IK of both arms (under 0.5 mm and 0.1 degrees, joints within
  limits); that the grip point moves under 3 mm from the moment the closing pads reach the bottle until the lift;
  that the pad tips stay 2 cm or more above the mat and the waiting gripper's tips 2 cm above the bottle's top; that a
  carried bottle hangs 3 cm or more above the platform's highest point and passes over no bottle still standing; that
  the arms' tool points stay more than 20 cm apart; that each bottle passes the platform's edge during the carry, its
  gripper closed on it, never earlier than 0.5 s after the lift starts or later than 0.5 s before the lowering; that
  at the end both grippers' centre lines are more than 9 cm from every bottle's axis; and that the tilted platform
  never touches its base, its pivot block or the table.
- A 2,400-step trial with Dream-1 takes up to about 8 minutes on an L40S.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The rocker always starts level, its
  left end about to rise.

## Fidelity

The platform is driven exactly along its motion, like a stiff motor: the arms can't push it or stop it (a real
rocker's small motor would stall or slip if blocked). It pivots about its centre line with no visible bearing: a
grey drive base and pivot block stand under its middle, clear of it by 8.5 mm or more. The bottles are plain boxes
(no caps or rounded corners) with the station's rubber-like friction (1.2 static, 1.0 dynamic); the mat's friction is
1.0 static, 0.9 dynamic. Either is far above what a 10 degree tilt needs (0.18), so nothing slides on the mat, as on
a real rocker's rubber mat.

## Real trials

A bench-top laboratory rocker (15 rocks a minute, 10 degrees each way) is slow and low-powered; its platform is 6 cm
above the table at its pivot and about 10 cm at a raised end, so a bottle knocked off falls 10 cm or less onto the
table and stays within the station. Use capped empty (or water-filled) bottles, and stop the rocker from its switch
if an arm jams against the platform.
