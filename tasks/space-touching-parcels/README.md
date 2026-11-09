# Space out touching parcels on a running belt before they reach the scanner

A belt conveyor crosses the table 32 cm in front of the arms, running from the robot's right to its left at 5 cm/s.
Three cardboard parcels (9 x 6 x 5 cm) ride on it nose to tail, touching. 58 cm left of the midpoint between the arms
a scanner arch stands over the belt, and the scanner needs at least 5 cm of empty belt in front of each parcel as it
passes. The robot must open both gaps before the parcels get there, on a belt that never stops: lift a parcel and
set it down farther ahead or farther back, slide it, or hold it while the belt runs under it. Every parcel must stay
on the belt.

This is singulation, done by people and robots at the induction end of every parcel sorter: barcode tunnels and
dimensioners read one parcel at a time, and two touching parcels are read as one or not at all. The difficulty here is
the timing: each parcel is moving whenever the robot touches it, a parcel set down must land moving with the belt
without bumping its neighbours, and the front parcel reaches the scanner about 13 s after the start if nothing is done.

The board's nearest task feeds parcels from a tote onto a standing lane with gaps between them; others pick from or
sort on a moving belt. Here the parcels are already on a running belt, nothing is taken off, and what is scored is
the spacing at a fixed point downstream, as a real scanner sees it.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the conveyor, belt, scanner arch and parcels, the scanner's record (a monitor run every control step), `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds driven objects to yam_arena task files. Copied from our unload-paused-shuttle and reach-between-sweeps tasks (with `speed_schedule()` from our sort-surging-belt task), with `Belt` added: a plate that stays in place while its surface moves, given the belt's velocity before every physics step and put back where it was, so what lies on it is carried by friction |
| `test_task.py` | Offline checks (no Isaac Sim): the belt and the layout, the scanner's record and every scoring level, and that the demo is reachable, keeps the arms apart and clear of the arch, and reaches full score when replayed with the parcels riding the belt or held in the fingers |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset, control step and simulation step without
modifying yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score`
scorer. The belt's clock is the episode clock (`observation.state_time`): the belt stands still through the reset and
starts with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured
on an L40S:** the scripted demo ended with success after 531 steps (26.6 s); Dream-1 scored 0 in an 800-step trial.

### Why a belt that stays in place

Parcels have to be set down anywhere along the belt, so carriers (one plate per parcel, as in our other conveyor
tasks) don't fit. A real belt's top run is a surface that moves while the conveyor stays put; `Belt` models exactly
that: a body whose velocity drives its contacts without the body going anywhere (the usual way to model a
conveyor belt in a rigid-body simulator). Each
physics step the 50 kg plate, gravity off, is put back at its place and given the belt's velocity; PhysX moves it by
0.2 mm during the step, and friction carries the parcels at the belt's speed. A long plate translating along the
line would do the same but would have to stick out a metre past the table and run out behind the parcels.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The three parcels on the belt are touching. Space them out on the belt so that each one has at least 5 cm of empty belt in front of it when it reaches the scanner arch. The belt keeps running. Keep all three parcels on the belt." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/parcels-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The three parcels on the belt are touching. Space them out on the belt so that each one has at least 5 cm of empty belt in front of it when it reaches the scanner arch. The belt keeps running. Keep all three parcels on the belt." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/parcels-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/parcels-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 once `success()` has held for 1 s, which is 1 s after
the last parcel has passed the scanner with both gaps 5 cm or more. Otherwise it scores `score()` at the end of the
trial (`SCORE_OVER = "end"`).

## Scoring

A parcel passes the scanner from the moment its leading face reaches the scan line (the plane through the red line
under the scanner's head) until its trailing face has left it. The gap in front of a parcel is the length of belt that
ran past the line between the parcel ahead leaving the line and this parcel reaching it: what a photo-eye at the line
measures (time between the beam clearing and breaking again, times the belt speed). The monitor in `task.py` records
these times every control step, interpolated between steps.

| Score | Condition | Check on a real table |
|---|---|---|
| 0 | any parcel, during its pass, not lying on the belt (underside more than 1 cm off the belt's top: lifted, carried over the line, on top of another parcel, fallen off) or reaching over the belt's edge; or any parcel not passed by the end of the trial | video |
| 0.5 per gap | each of the two gaps (in front of the second and the third parcel to pass) is 5 cm or more | a photo-eye at the line, or video against a scale on the bed |
| 0.25 per gap | a gap of 2.5 to 5 cm | as above |

So 1.0 for both gaps of 5 cm or more; 0.75 for one of 5 cm and one of 2.5 to 5 cm; 0.5, 0.25 or 0 down to parcels
passing touching as they started. Taking a parcel off the belt scores 0, however the others are spaced, as it would
be a lost parcel on a real line. Only a parcel's first pass counts, and what happens to a parcel after it has passed
doesn't change the score. Parcels may change order and may end up rotated: the gaps are measured along the belt
between whichever parcels pass one after the other.

## Runtime and seeds

- The scripted demo, open loop and timed against the belt, uses both arms, one after the other: the right arm comes
  down onto the last parcel at 2.0 s, lifts it 5 cm and sets it down 13 cm farther back (let go by 6.4 s); the left
  arm waits high and comes down onto the front parcel at 4.8 s, once the right arm has carried its parcel away, and
  sets it down 10 cm farther ahead (let go by 8.7 s). Each follows its parcel along the belt while coming down onto it,
  gripping (0.3 s before lifting), lifting, setting it down and letting go, so the parcel is set down with no speed
  relative to the belt; fingers close across the belt, on the parcels' free side faces, 1.1 cm short of the neighbour.
  The parcels then pass the scan line from 10.85 s to 20.9 s with 10.2 cm in front of the second and 13.2 cm in front
  of the third, and in Isaac Sim the trial ended with success after 531 steps (26.6 s). Both arms hold their park pose
  until 23.5 s, so that `yam_scripted` (which ends the rollout 3 s after the plan) doesn't stop it before then.
- Doing nothing: the front parcel reaches the scan line at 12.85 s and all three pass touching (0.2 cm gaps): 0.
- `PARCEL_DEBUG=1` (in the environment of the run) prints each parcel's position every 0.5 s, each parcel's arrival
  at the scan line with the gap in front of it, any fault, and the final score.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The belt always starts with the trial.

## Fidelity

The belt is a plate that stays in place while its surface moves (see above): the arms can't slow or stop it. The
parcels are rigid boxes with sharp edges and the station's rubber-like friction (1.2 static, 1.0 dynamic), more than
cardboard on a rubber belt; at the belt's peak acceleration of 0.16 m/s² (only while it starts) they don't slide. The
belt runs at 5 cm/s, much slower than a real parcel line, to suit the arms' pace. The scanner itself is not simulated:
the scan line and the gaps are measured from the parcels' positions, as a photo-eye would.

## Sim runs so far

- First run (an earlier demo, both arms gripping at the same moment): 0.5. The front parcel was moved ahead, but the
  last parcel's move back didn't take, leaving it touching the middle one. In that demo the two wrists, which lean
  toward each other when both are down at the belt, came within 2.9 cm (collision meshes) while both grippers were
  closing, and the parcel was lifted only 3 cm and gripped above its centre. The demo now works one arm after the
  other (wrists 8 cm apart or more), grips around the parcel's centre of mass and holds 0.3 s before lifting, lifts
  5 cm, and moves the last parcel 13 cm back; the scanner is 6 cm farther downstream to leave time for that.
- Second run (this demo): 1.0, success after 531 steps (26.6 s).
