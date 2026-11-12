# Merge a parcel into the gap in a running belt's line, square and clear of its neighbours, before the scanner

A belt conveyor crosses the table 32 cm in front of the arms, running from the robot's right to its left at 6 cm/s.
Four cardboard parcels (9 x 6 x 5 cm) ride on it in a line, 6 cm apart, with one empty gap of 24 cm between the
second and the third. A fifth parcel waits on a small steel side table between the belt and the left arm, its top
level with the belt. The robot must merge it into the gap while the belt runs, so that when it passes the scanner
arch 52 cm left of the midpoint it has at least 4 cm of clear belt on both sides and lies square on the belt, and
without touching the other four: neither the arms nor the merged parcel may touch any of them at any time (so no
nudging a neighbour to widen the gap). The parcel must be let go before it reaches the scanner: holding it on the belt
or steering it through the scan line doesn't count as merged.

This is what an induction or merge station on a parcel sorter does: parcels from a side table or a feeder belt are
put onto the running line into the empty slots between parcels already on it, and the barcode scanner and the sorter
downstream need each one square and clear of its neighbours. The difficulty is the timing: the gap is only there for
a few seconds in front of the side table, the parcel has to be set down moving with the belt (set down standing
still, it is jerked forward and can turn or slide into a neighbour), and the gap has 3.5 cm of slack either way. The parcels are small: 9 x 6 x 5 cm, the size of small boxed items such as pharmacy or cosmetics cartons on the induction belt of a small-item sorter in e-commerce or pharmacy order fulfilment, whose belts are narrow to match.

Here the gap is already there and the robot fills it with a parcel from off the belt, without moving the others (the
opposite of opening gaps between touching parcels). Tasks that put parts into a moving tote or carton, or feed
parcels onto a standing lane, are related; here the parcel goes onto the running belt itself, into one given slot in a line, and what is scored is the spacing
and squareness a scanner sees downstream.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the conveyor, belt, side table, scanner arch and parcels, the scanner's record and the contact watch (a monitor run every control step), `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds driven objects to yam_arena task files: a shared helper module whose `Belt` (a plate that stays in place while its surface moves, given the belt's velocity before every physics step and put back where it was, so what lies on it is carried by friction) this task reuses; extended with `AwakeBox`, a free box spawned as yam_arena spawns one but with PhysX sleeping disabled, used for the five parcels, with `Belt.carry`, which drives the plate's surface faster by a measured factor so that what rides on it moves at the belt's nominal speed, and with `robot_link_poses`, every arm link's pose for the contact watch |
| `test_task.py` | Offline checks (no Isaac Sim): the layout and the belt, the scanner's record, the contact watch and every scoring level (including pushing a neighbour back to widen the gap, brushing one once, bumping one with the merged parcel, and holding the merged parcel into its pass under the scanner, all 0), and that the demo is reachable, stays clear of the right arm and the arch, and reaches full score with wide margins when replayed with the parcels riding the belt, standing on the side table or held in the fingers and the arms' collision hulls placed along the plan: neither arm comes within 4 cm of the line's parcels, nor the merged parcel within 3 cm, even with the line riding 7 % off speed |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset, control step and simulation step without
modifying yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score`
scorer. The belt's clock is the episode clock (`observation.state_time`): the belt stands still through the reset and
starts with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured
on an L40S:** with the contact watch, the scripted demo ended with success after 396 steps (19.8 s), neither arm
coming within 7.4 cm of the line's parcels (nor within 5 cm of the merged parcel during its pass) and the merged
parcel not within 5.7 cm of them; Dream-1 scored 0 in a 600-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the parcel from the side table onto the running belt, into the empty gap in the line of parcels, square to the belt, so that it has at least 4 cm of clear belt on both sides when it reaches the scanner arch. Let go of it before it reaches the scanner. Don't touch the other parcels. The belt keeps running." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/merge-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the parcel from the side table onto the running belt, into the empty gap in the line of parcels, square to the belt, so that it has at least 4 cm of clear belt on both sides when it reaches the scanner arch. Let go of it before it reaches the scanner. Don't touch the other parcels. The belt keeps running." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/merge-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/merge-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 once `success()` has held for 1 s, which is 1 s after
the last parcel has passed the scanner with everything in order. Otherwise it scores `score()` at the end of the
trial (`SCORE_OVER = "end"`).

## Scoring

A parcel passes the scanner from the moment its leading face reaches the scan line (the plane through the red line
under the scanner's head) until its trailing face has left it. The clear belt between two parcels is the distance from
the line to the next parcel's leading face at the moment the one ahead leaves the line (a photo at that moment, or a
photo-eye's dark time times the line's speed). It is a length along the belt, so it doesn't depend on how fast the
parcels ride. Faces are taken from each parcel's bounding box, so a turned parcel takes up more belt.

The four parcels of the line must not be touched at any time in the trial, by either arm or by the merged parcel.
Every control step the monitor in `task.py` places each arm's collision hulls (the URDF's, where the simulator has
the links) and the parcels' boxes, and a body within 5 mm of one of the four parcels (where PhysX starts making
contacts) touches it. The first touch is latched for the rest of the trial. The only exception: two parcels that
have both passed the scanner may touch, as they gather against the end stop. The merged parcel itself must be let go
before it reaches the scanner: an arm within 5 mm of it at any moment of its pass (holding it on the belt, steering
or pushing it through) also scores 0. As a backstop for a touch between two
control steps, a line that arrives at the scanner disturbed also counts: its spacing at the scan line more than 1.5
cm off how it started (6 cm between the first two and between the last two, 24 cm from the second to the third),
one of the four more than 1.5 cm across the belt from its centre line, or turned or tilted more than 5°. The monitor
records all this every control step, interpolated between steps where it measures lengths.

| Score | Condition | Check on a real table |
|---|---|---|
| 0 | either arm or the merged parcel touched any of the four parcels of the line, at any time (pushing a neighbour to widen the gap, brushing one, bumping one while setting down), or the line arrived at the scanner disturbed; an arm touched the merged parcel during its pass (not let go before the scanner); any parcel, during its pass, not lying on the belt (underside more than 1 cm off the belt's top: lifted, carried over the line, on top of another parcel, fallen off) or reaching over the belt's edge; any of the five not passed by the end of the trial; or the side-table parcel not passing right after the second parcel and right before the third (not in the gap) | video of the whole trial (any touch), a photo-eye or video at the line |
| +1/3 | 4 cm or more of clear belt between the second parcel and the merged one | a photo-eye at the line, or video against a scale on the bed |
| +1/3 | 4 cm or more between the merged parcel and the third | as above |
| +1/3 | the merged parcel square throughout its pass: turned 15° or less from square to the belt, tilted 15° or less from lying on its 9 x 6 cm face | overhead photo with a protractor |

So 1.0 needs all three, with no touch; each missed condition costs a third. Dropping the merged parcel somewhere else
on the belt (behind the line, say, where there is plenty of room) scores 0: the gap is the slot the sorter has
allocated to it. Knocking a parcel off the belt scores 0 as a lost parcel. Only a parcel's first pass counts.

## Runtime and seeds

- The scripted demo, open loop and timed against the belt, uses the left arm only (the right stays in its ready
  pose), gripping the parcel across the belt on its long faces. In Isaac Sim the fingers take about 1 s to close
  from fully open, and once both drives saturate on the parcel the finger pair can slide across as one, so the
  parcel may sit up to 1.7 cm either side of the TCP (still 3.3 cm inside the belt's edges). The demo comes down
  over the parcel (pad tips 1.5 cm above the table), commands the fingers closed at 1.9 to 2.4 s and holds still
  until 3.8 s, then lifts slowly (0.9 s) until the parcel's underside is 1.5 cm above the other parcels' tops. From
  4.7 s, as the gap comes abreast of the side table, it carries the parcel over the belt in 1.6 s while speeding up
  to the belt, ending over the gap's middle and moving with it. It follows the gap down slowly
  (6.3 to 7.5 s) until the parcel's underside is commanded at the belt's top, holds it there until 7.8 s, commands
  the fingers open at 7.8 to 8.2 s and keeps following the gap until 9.2 s while they open, then rises, still
  following it, and parks clear. The parcel passes the scan line from 11.8 to 13.3 s with 7.5 cm of clear belt on
  either side (3.5 cm of slack each way: enough for parcels riding up to 7 % off the belt's speed). The last parcel has passed at 18.6 s, and in Isaac Sim the trial ended with success after
  396 steps (19.8 s). The plan holds its park pose until 21 s so that `yam_scripted` (which ends the rollout 3 s after the
  plan) doesn't stop it before then.
- `MERGE_DEBUG=1` in the environment prints the merged parcel's pose and its offset from the gap's middle every
  0.5 s, and, every control step while the left fingertips are within 10 cm of it, the finger joints (pad gap) and
  the fingertips' midpoint relative to the parcel's centre; every 0.5 s also how far each parcel of the line is from
  where riding at the belt's speed puts it.
- Doing nothing: the four parcels pass, the fifth never does: 0.
- Safety on a real table: On a real table the belt runs at 6 cm/s with side guards along its bed, a guard covers the pinch point between the queued parcels and the end stop, and an emergency stop within the operator's reach cuts both the belt and the arms; parcels are light (150 g) and the end stop keeps them from running off the belt's end.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The belt always starts with the trial.

## Fidelity

The belt is a plate that stays in place while its surface moves: the arms can't slow or stop it. The parcels are
rigid boxes with sharp edges and the station's rubber-like friction (1.2 static, 1.0 dynamic), more than cardboard on
a rubber belt; at the belt's peak acceleration of 0.19 m/s² (only while it starts) they don't slide. Parcels on this
belt model ride a steady ~1.1 cm/s slower than the surface it is given (measured: 4.9 cm/s on a 6 cm/s surface,
6.3 cm/s on a 7.4 cm/s one), most
likely because the plate is put back in place every physics step and PhysX's friction anchors don't follow; the plate
is driven at 7.1 cm/s (6 / 0.845) so that the parcels ride at 6 cm/s, and the scoring measures lengths along the belt, so it
holds whatever their speed. The belt runs at
6 cm/s, much slower than a real parcel line, to suit the arms' pace. The scanner is not simulated: clearances and
orientation are measured from the parcels' poses, as a photo-eye and an overhead camera would.

## Sim runs

The final demo, with the contact watch, ended with success after 396 steps (19.8 s) with 6.8 cm clear ahead of the
merged parcel, 7.8 cm behind it, turned 3 degrees, the line's spacing at the scan line 5.9 / 23.9 / 6.2 cm, and no
touch: the arms' collision hulls came no closer than 7.4 cm to the four parcels of the line and the merged parcel no
closer than 5.7 cm; while the merged parcel passed the scanner, no arm link was within 5 cm of it. Earlier runs of the demo:
lifting 0.5 s after the close command left the parcel behind (fixed by waiting 1.4 s); then the parcels on the belt
rode slower than its surface (4.9 cm/s on 6 cm/s), a steady slip of about 1.1 cm/s, now compensated by driving the
surface at 7.1 cm/s (`CARRY = 0.845` in `task.py`). The scoring measures lengths at the scan line, so it holds
whatever the parcels' speed.
