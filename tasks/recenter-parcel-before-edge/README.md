# Steer a drifting parcel back into the centre lane before the scanner arch

A powered roller conveyor (26 x 74 cm) crosses the table in front of the arms, carrying parcels from the robot's left
to its right. Its rollers are skewed 15 degrees, so they drive a parcel diagonally: 5.8 cm/s along the conveyor and
1.6 cm/s toward its open near side (the side toward the arms; the far side has a low rail). Two white lines mark a
14 cm centre lane, which leads into a scanner arch whose opening (20 cm) fits a 10 cm parcel only if it is within 5 cm
of the lane's centre. A brown parcel (10 x 10 x 6 cm, 150 g) starts on the conveyor in front of the left arm, 4 cm
toward the open side. Untouched, it drifts off the open side about 6 s in, 10 cm before the arch.

The robot must get the parcel between the white lines so that it rides into the arch centred, without letting it fall
off. It can't fix this once: the rollers keep driving the parcel toward the open side after it is let go (1 cm of
drift for every 3.7 cm it travels), so a parcel pushed to the centre early arrives off-centre again. The robot has
to act before the parcel goes over the edge, then either ride along beside it as a moving side rail or leave it far
enough beyond the centre that the drift brings it back by the arch. The parcel is wider than the open gripper, so it
can only be pushed.

Skewed-roller sections drive parcels sideways to line them up against a side rail; where the rail is missing or ends
early, parcels drift off the open side, or reach scanners, label applicators and narrowing guides off-centre and jam
or are misread. Conveyor attendants straighten them by hand while the line keeps running.

How it differs from the board's other conveyor and parcel tasks: those pick parts off a belt (sorting, packing,
sampling, a surging belt), or stop a carton coasting freely on a static low-friction deck and push it onto a square
where it then stays. Here the object never stops and the conveyance keeps acting on it: the robot corrects its
lateral position relative to a moving lane, against a steady sideways drive, and is scored on where it is at the
moment it reaches the arch.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the conveyor, its rollers, lane lines, rail and arch, the parcel, the crossing monitor, `success()`, `score()` and the scripted demo |
| `motionkit.py` | Adds object kinds to yam_arena task files: `ConveyorBed` (a fixed box whose top surface moves, so objects on it are carried by friction), `DrivenBox` (follows a scripted path), `LaunchedBox` (a free box with its own friction, used here at rest for the cardboard parcel) and `Marker` (visual-only, no collider); plus `clock()` and `on_reset()` for the monitor. Copied from our stop-sliding-carton and stop-ball-before-edge tasks, with `ConveyorBed` and `ramp_speed()` added |
| `test_task.py` | Offline checks (no Isaac Sim): the layout (bed clear of the table and fixed parts, lines and arch centred, rollers drawn square to the drive), the untouched ride, every scoring level, and that the demo is reachable, comes down beside the parcel rather than on it, can't tip it, keeps pace with it, leaves it on the centre at the crossing, and keeps the arm clear of the arch and rail |

`motionkit.py` hooks `yam_arena.scene` and the embodiment's reset and simulation step without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. Written
against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured on an L40S:** the
scripted demo ended with success after 189 steps (9.5 s); Dream-1 scored 0 in a 600-step trial.

### How the conveyor is simulated

The roller bed is one heavy rigid box (gravity off) 2 mm above the table. Before every physics step (240 Hz) it is
put back where it stands and given the rollers' surface velocity, so during each step its top slides 0.25 mm under
whatever rests on it and friction drags that along, as the moving surface of a belt or powered rollers does. (It is
the same mechanism as the shuttle and belt carriers in our other tasks, whose pose is also written every physics
step, except that the pose doesn't advance.) A moving plate carrying the parcel couldn't make it drift: an object
riding a plate moves with it, and an angled guide that pushed it outward would have to sweep across the whole lane,
blocking the centred parcel too. The rollers themselves are drawn as visual-only stripes, square to the drive
direction; on a real conveyor the rollers turn in place, so they look still as they do here. The bed starts with the
episode and reaches full speed in 0.5 s.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The parcel on the conveyor is drifting toward the open edge. Get it between the two white lines before it reaches the arch, without letting it fall off." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/recenter-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The parcel on the conveyor is drifting toward the open edge. Get it between the two white lines before it reaches the arch, without letting it fall off." \
  --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/recenter-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/recenter-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end. The trial ends with 1.0 when `success()` has held for 1 s (it holds from the
crossing on); otherwise it scores `score()` at the end of the trial (`SCORE_OVER = "end"`), which keeps the outcome
decided at the crossing, or 0 once the parcel has fallen off.

## Scoring

The parcel "crosses" when its centre passes the arch's upstream face (the faces of its two inner guides).

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | it crosses riding upright on the rollers (top face within 15 degrees of vertical, underside within 1 cm of the rollers' top), its centre within 2 cm of the lane's centre, i.e. wholly between the white lines | video from above as it enters the arch, against the lines |
| 0.5 | it crosses so, its centre 2 to 5 cm off the lane's centre | the same |
| 0.25 | it crosses more than 5 cm off; or it never crosses riding (held back, jammed against a guide, or lifted past the arch) and at the end stands upright on the rollers | video, and a photo at the end |
| 0 | it fell off the conveyor before crossing (its centre past an edge, or its underside 1.5 cm below the rollers' top), or at the end it is tipped over or held off the rollers without having crossed | video |

The outcome is decided at the crossing: what happens after it (the parcel rides on to the arch's end stop) doesn't
change the score. A parcel that falls off before the crossing scores 0 for the rest of the trial.

## Runtime and seeds

- The scripted demo (left arm, closed fingers turned a quarter turn so they face across the conveyor like a paddle)
  is over the parcel's near side at 1.8 s, comes down beside its near face (2 cm clear, riding along with it, pad
  tips 1.5 cm above the rollers) by
  2.6 s, pushes it out to the lane by 4.2 s, then rides beside it as a moving side rail, the rollers pressing it
  against the fingers, holding it 2.4 cm beyond the lane's centre. When its centre is 9 cm from the arch (6.3 s) the
  fingers back off and lift, leaving the arm at least 4 cm (beyond a 4.5 cm link radius) from the arch; the parcel
  drifts the last 2.4 cm on its own and crosses on the centre at about 7.8 s.
  In Isaac Sim the trial ended with success after 189 steps (9.5 s). Even off by 1 cm at the fingers and 30 % in the
  drift, it crosses within 2 cm.
- One fixed layout: `-E object_jitter=0`, so the seed doesn't change the start. The parcel always starts at the same
  place, at rest, and the conveyor always starts with the episode.

## Fidelity

The roller bed is one flat surface moving at the rollers' surface speed in their drive direction, with isotropic
friction (rubber-lagged rollers 0.8 / 0.7, cardboard 0.5 / 0.4, averaged): it leaves out the gaps between rollers and
the small bumps a parcel feels crossing them. Powered rollers that don't slip drive a parcel this way: it moves with
their surface, and pushing it sideways means sliding it along the rollers against friction (about 0.8 N for this
parcel). The parcel is a rigid box. The arch's inner guides and end stop hang 3 mm above the rollers from its roof;
the far rail stands on the table beside the conveyor. The demo's finger contact uses 1.8 cm from the closed fingers'
centre line to where they touch a box, measured in Isaac Sim for an earlier task with the same gripper.
