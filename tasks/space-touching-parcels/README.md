# Space out touching parcels on a running belt before they reach the scanner

A belt conveyor crosses the table 32 cm in front of the arms, running from the robot's right to its left at 5 cm/s.
Three cardboard parcels (9 x 6 x 5 cm) ride on it nose to tail, touching. 58 cm left of the midpoint between the arms
a scanner arch stands over the belt, and the scanner needs at least 5 cm of empty belt in front of each parcel as it
passes. The robot must open both gaps before the parcels get there, on a belt that never stops: lift a parcel and
set it down farther ahead or farther back, slide it, or hold it while the belt runs under it. Every parcel must stay
on the belt until it has passed the scanner: a lift over the belt of up to 5 s is fine, but a parcel set on the table
or on another parcel, or carried out over the table, even briefly and even if put back, scores 0.

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
| `task.py` | The yam_arena task file: the conveyor, belt, scanner arch and parcels, the scanner's record (a monitor run every control step, for the whole trial), `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds driven objects to yam_arena task files. A shared helper module (driven boxes, `speed_schedule()`, `episode_time()`), with `Belt`: a plate that stays in place while its surface moves, given the belt's velocity before every physics step and put back where it was, so what lies on it is carried by friction |
| `test_task.py` | Offline checks (no Isaac Sim): the belt and the layout, the scanner's record and every scoring level (including staging a parcel on the table, or on another parcel, and putting it back before its pass: 0; and gaps measured correctly when the parcels slip on the belt), and that the demo is reachable, keeps the arms apart and clear of the arch, and reaches full score when replayed with the parcels riding the belt or held in the fingers |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset, control step and simulation step without
modifying yam_arena, so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score`
scorer. The belt's clock is the episode clock (`observation.state_time`): the belt stands still through the reset and
starts with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured
on an L40S:** the scripted demo, scored with the rules below, ended with success after 538 steps (26.9 s); Dream-1 scored 0 in an 800-step trial.

### Why a belt that stays in place

Parcels have to be set down anywhere along the belt, so carriers (one moving plate per parcel)
don't fit. A real belt's top run is a surface that moves while the conveyor stays put; `Belt` models exactly
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
  --instruction "The three parcels on the belt are touching. Space them out on the belt so that each one has at least 5 cm of empty belt in front of it when it reaches the scanner arch. The belt keeps running. Keep each parcel on the belt until it has passed the scanner arch: you may lift one briefly, but only over the belt, and never set it down anywhere else." \
  --scorer task_score --max-steps 800 --store-frames --log-dir ~/yam-eval/parcels-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The three parcels on the belt are touching. Space them out on the belt so that each one has at least 5 cm of empty belt in front of it when it reaches the scanner arch. The belt keeps running. Keep each parcel on the belt until it has passed the scanner arch: you may lift one briefly, but only over the belt, and never set it down anywhere else." \
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
under the scanner's head) until its trailing face has left it; only its first pass counts. Until its pass begins, a
parcel must stay on the belt; a brief lift over the belt (up to 5 s at a time; a lift ends only once the parcel has
lain on the belt again for 0.5 s) is allowed. The gap in front of a parcel
is measured between the parcels themselves: the distance along the belt from the trailing face of the parcel ahead
(the one that reached the line before it) to its own leading face, taken when the parcel ahead's trailing face leaves
the line and again when this parcel's leading face reaches it; the smaller counts. (It is not taken from the belt's
travel between those moments, as a photo-eye timing the beam would: the parcels can slip on the belt, and in Isaac
Sim they ride about 1 cm/s slower than its surface, which would overstate every gap by about a quarter.) The monitor
in `task.py` checks every parcel every control step, for the whole trial, and interpolates the crossing moments and
positions between steps. A rule broken once fails the trial, whatever happens afterwards.

| Score | Condition | Check on a real table |
|---|---|---|
| 0 | before its pass, any parcel's centre leaves the area over the belt (between its edges and its ends) even for a moment: set down on the table, knocked off, carried out over the table, even if put back | video, against the belt's edges |
| 0 | before its pass, any parcel off the belt (underside more than 1 cm above the belt's top) for more than 5 s at a time (touching down for less than 0.5 s doesn't end a lift), or resting on another parcel (underside within 1 cm of its top, over it by 1 cm or more both ways) for 0.5 s | video with a timer |
| 0 | any parcel, during its pass, not lying on the belt (underside more than 1 cm off the belt's top: lifted, carried over the line, on top of another parcel, fallen off) or reaching over the belt's edge; or any parcel not passed by the end of the trial | video |
| 0.5 per gap | each of the two gaps (in front of the second and the third parcel to pass) is 5 cm or more | two photos per gap, at the two moments, with a scale on the bed or a tape measure |
| 0.25 per gap | a gap of 2.5 to 5 cm | as above |

So 1.0 for both gaps of 5 cm or more; 0.75 for one of 5 cm and one of 2.5 to 5 cm; 0.5, 0.25 or 0 down to parcels
passing touching as they started. Taking a parcel off the belt before its pass scores 0, however the others are
spaced and even if it is put back, as staging parcels beside a running line would be on a real one; lifting a parcel
and setting it down farther along the belt within 5 s is how the gaps are meant to be opened. What happens to a parcel
after its pass doesn't change the score, except that a parcel put back on the belt in front of the next one shortens
that one's gap. Parcels may change order and may end up rotated: the gaps are measured along the belt between
whichever parcels pass one after the other.

## Runtime and seeds

- The scripted demo, open loop and timed against the belt, uses both arms, one after the other: the right arm comes
  down onto the last parcel at 2.0 s, lifts it 5 cm and sets it down 13 cm farther back (let go by 6.4 s); the left
  arm waits high and comes down onto the front parcel at 4.8 s, once the right arm has carried its parcel away, and
  sets it down 10 cm farther ahead (let go by 8.7 s). Each follows its parcel along the belt while coming down onto it,
  gripping (0.3 s before lifting), lifting, setting it down and letting go, so the parcel is set down with no speed
  relative to the belt; fingers close across the belt, on the parcels' free side faces, 1.1 cm short of the neighbour.
  Each parcel is off the belt for about 2.6 s and its centre stays on the belt's centre line. Riding at the belt's
  speed, the parcels would then pass the scan line from 10.85 s to 20.9 s with 10.2 cm in front of the second and
  13.2 cm in front of the third; in Isaac Sim they ride about 1 cm/s slower than the belt's surface and pass later
  (the gaps between them are the same), and the trial ended with success after 538 steps (26.9 s). Both arms hold
  their park pose until 30 s, so that `yam_scripted` (which ends the rollout 3 s after the plan) doesn't stop it
  before the last parcel has passed.
- Doing nothing: the front parcel reaches the scan line at 12.85 s and all three pass touching (0.2 cm gaps): 0.
- `PARCEL_DEBUG=1` (in the environment of the run) prints each parcel's position every 0.5 s, each parcel's arrival
  at the scan line with the gap in front of it, any fault, and the final score.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The belt always starts with the trial.

## Safety on a real table

The belt runs throughout the trial: keep hands clear of it while it runs, and of the pinch points at its ends, under
the scanner arch and against the end stop. Use a conveyor with an emergency stop within reach of the operator, and
the arms' own e-stop; stop the belt before resetting the parcels. The parcels are 150 g cardboard boxes, so one
dropped or pushed off the belt can't hurt anyone; one that falls off past the table's left edge lands on the floor.

## Fidelity

The belt is a plate that stays in place while its surface moves (see above): the arms can't slow or stop it. The
parcels are rigid boxes with sharp edges and the station's rubber-like friction (1.2 static, 1.0 dynamic), more than
cardboard on a rubber belt; at the belt's peak acceleration of 0.16 m/s² (only while it starts) they don't slide. The
belt runs at 5 cm/s, much slower than a real parcel line, to suit the arms' pace. The scanner itself is not simulated:
the scan line and the gaps are measured from the parcels' positions. In Isaac Sim the parcels ride about 1 cm/s
slower than the belt's surface (they creep on the plate, which is put back every physics step); that only delays
them, since the gaps are measured between the parcels and not from the belt's travel.

## Sim runs so far

- First run (an earlier demo, both arms gripping at the same moment): 0.5. The front parcel was moved ahead, but the
  last parcel's move back didn't take, leaving it touching the middle one. In that demo the two wrists, which lean
  toward each other when both are down at the belt, came within 2.9 cm (collision meshes) while both grippers were
  closing, and the parcel was lifted only 3 cm and gripped above its centre. The demo now works one arm after the
  other (wrists 8 cm apart or more), grips around the parcel's centre of mass and holds 0.3 s before lifting, lifts
  5 cm, and moves the last parcel 13 cm back; the scanner is 6 cm farther downstream to leave time for that.
- Second run (this demo): 1.0, success after 531 steps (26.6 s).
- Third run (this demo, after the review: parcels must stay on the belt before their pass, gaps measured between the
  parcels instead of from the belt's travel, plans held until 30 s): 1.0, success after 538 steps (26.9 s). Rerun
  after a lift was made to end only once the parcel has lain on the belt 0.5 s: the same, 1.0 after 538 steps.
