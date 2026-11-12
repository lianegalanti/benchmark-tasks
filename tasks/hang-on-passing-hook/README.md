# Hang parts on the hooks of a moving overhead conveyor

Two orange hanger parts stand on the table, one in front of each arm: a block with a rectangular eye (a loop) on top
and a small grip tab above the eye. An overhead rail across the far side of the table carries three steel hooks, 30
cm apart, from the robot's left to its right at a steady 5 cm/s; they never stop. Each hook hangs from the rail on a
stem and ends in a bar pointing back toward the robot, 12 cm above the table, with a short upturned tip. The robot
must hang each part on its own hook by its eye before that hook passes a red exit mark at the right of the table,
and let go of it. To do so it has to carry the part to a hook as it comes into reach, move along with the hook, slip
the eye over the tip and onto the bar (6 cm of opening around a 1.2 cm bar), lower it onto the bar and let go, all
while the hook keeps moving.

Overhead hook conveyors carry parts through paint, powder-coating and plating lines, garments through laundries and
finishing, and parts between stations in assembly plants. Loading them is a common manual job: the loader picks a
part and hangs it on the next passing hook, matching the line's pace, since the line doesn't stop for one loader. A
part hung badly falls off further down the line, and an empty hook is lost throughput.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the hooks' motion, the loading monitor, `success()`, `score()` and the scripted demo |
| `geometry.py` | Dimensions of the conveyor, hooks and parts, and the tests the score uses (is a hook's bar through a part's eye, does the eye rest on it, how low is the part) |
| `make_part.py` | Writes `assets/hanger_part.usda`, the part as one rigid body of five boxes (needs `usd-core`) |
| `motionkit.py` | Adds a `DrivenBox` object kind (a box whose pose and velocity follow a scripted path, set before every physics step), a `UsdPart` object kind (a free rigid body from a USD file), `on_step` / `on_reset` hooks with `arm_joints()` for the monitor, and `clock()` |
| `test_task.py` | Offline checks (no Isaac Sim): the hooks' motion and the clearances, every scoring level, the ways of keeping hold that get no hanging credit (gripping the tab, gripping or propping the block, any link within 2 cm of any box of the part, letting go under 1 s before the mark, holding the eye off the bar), and the demo against rigid-grip parts, played through the scorer |

`motionkit.py` is a small helper module, shared with other task folders in this repository (their license applies).
It hooks
`yam_arena.scene` and the embodiment's reset, simulation step and task scoring without modifying yam_arena, so the
task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The hooks' clock
is the episode clock (`observation.state_time`): they stand still at their start through the reset and start with
the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured on an
L40S:** under the revised scorer the scripted demo ended with success after 661 steps (33.1 s); Dream-1 scored 0 in a 900-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the two orange parts on the passing conveyor hooks by the loop on top of each part, one part per hook, before the hooks reach the red mark, and let go of them. The hooks keep moving and won't stop for you." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/hooks-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the two orange parts on the passing conveyor hooks by the loop on top of each part, one part per hook, before the hooks reach the red mark, and let go of them. The hooks keep moving and won't stop for you." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/hooks-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/hooks-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end: 1.0 once both parts have been credited (below) and that has held for 1 s,
otherwise the score at the end of the trial. Run with `-E control_hz=20`: the demo follows the hooks with one
waypoint per control step.

## The scripted demo

Both arms grip the parts by their tabs (fingers closing across the 1.2 cm tab, pad tips stopping 2 mm above the eye's
top bar, so the fingers never reach down to the hook) and keep the parts upright, eye facing the hooks.

- **Left arm, part A onto hook 0.** Picks the part (0 to 4.3 s), carries it to 4 cm in front of hook 0's tip at
  the height where both the tip and the bar pass through the eye (1.9 cm of clearance above and below), arriving
  as the hook comes 42 cm left of the midpoint (5.6 s). Then, with a waypoint every control step at the hook's
  speed: matches it 0.5 s, slides the eye 9 cm forward over the tip and onto the bar (1 s), lowers it 3.1 cm until
  its top bar is 3 mm above the hook's bar (0.8 s), opens the fingers to 5 cm (0.4 s, let go by 8.7 s), rides along
  open 0.3 s, rises 6 cm while following, and parks behind the hooks' path.
- **Right arm, part B onto hook 1.** Picks the part the same way, waits holding it above its start, and runs the
  same sequence on hook 1 from 21.6 s, when hook 1 is 8 cm right of the midpoint (let go by 24.7 s).

Hook 0 passes the exit mark at 26 s and hook 1 at 32 s; in Isaac Sim the trial ended with success after 661 steps (33.1 s).
Both arms stay parked until 35 s, so that `yam_scripted` (which stops 3 s after the plan) doesn't end the trial early.
Offline (`test_task.py`), against parts held rigidly from the moment the fingers close until they open: IK error
under 0.1 mm and 0.02 degree, joints within limits, the carried part never closer than 2 mm to any hook while it is
threaded on (it is let go with 3 mm to spare above the bar; at its tightest the eye passes the tip with 1.9 cm above
and below and the bar with 2.4 cm each side), and every arm link's URDF collision hull at least 1.7 cm from every hook,
the rail and the other part, checked every 0.2 s. Played through the scorer's monitor (every arm link's hull, fingers at their
commanded opening), each part hangs freely from 9.3 s (part A) and 25.3 s (part B), with no arm link within 5 cm of
it in the last 1.5 s before its hook's exit, and the demo scores 1.0.

## Scoring

Per part, summed over the two parts:

| Per part | Condition | Check on a real table |
|---|---|---|
| 0.5 | hung freely on its own hook for the whole last 1 s before that hook passes the red exit mark (the hook's bar centre over the mark's centre line): at every step of that second, the hook's bar through the part's eye with the eye resting on the bar (at most 5 mm between them), the part's lowest point at least 1 cm above the table, and nothing but its hook touching it: every link of both arms (both fingers of each gripper, the gripper bodies and the rest of the arms) and the other part at least 2 cm from every part of it (eye, tab and block); and the robot never held that hook back (its bar never more than 1 cm behind its 5 cm/s schedule) | video, for the second before each hook crosses the mark; a 2 cm gauge between fingers or other part and the part; a stopwatch: each hook over the mark within 0.2 s of 26, 32 and 38 s |
| 0.25 | otherwise, at some point a hook's bar was through the part's eye (held or not) | video |
| 0.1 | otherwise, at some point the part's lowest point was at least 3 cm above the table (lifted) | video, against a 3 cm mark |
| 0 | otherwise | |

A hook credits at most one part: two parts on one hook score 0.5 + 0.25. A credited part stays credited after its
hook has left. Any touch in that last second (a finger on the tab, a gripper holding or propping the block from
below, a wrist resting against it, the other part propping it) restarts the count, so a part still held anywhere, or let go less than 1 s before
the mark, doesn't get the 0.5. In the simulator the monitor places the URDF's collision hulls of every arm link at the
simulated link poses each control step and measures their distance to the part's five boxes. A real slip-clutch
conveyor can be stalled by an arm, while the simulated hooks can't: so a hook the robot stops or slows (its bar ever
more than 1 cm, 0.2 s, behind its schedule; in the simulator, the measured bar position against the schedule) credits
no part. Both parts credited (1.0) is success. "The bar through the eye" means the bar's centre line crosses
the plane of the eye inside its opening; a part laid across the bar, hooked by its tab, or held against a hook is
never through its eye.

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`): there are no seeds to choose, and the hooks always start in the same
place. The last hook passes the exit mark at 38 s, so a 900-step trial (45 s) covers every chance to score. A
900-step trial with Dream-1 takes a few minutes on an L40S, plus Isaac Sim's start.


## Fidelity

Rigid bodies, 240 Hz physics. The hooks are driven exactly along the rail like a stiff chain drive: the arms can't
push or stop them, where a real line's slip-clutch drive could be stalled by an arm (see the reset notes); the scorer gives no hanging credit on a hook more than 1 cm behind schedule. Each hook is three boxes
moving together with 1 mm gaps between them, hanging 5 mm under the rail without touching it; the rail's end
supports, beyond the table, are not modelled. The part is one rigid body of five boxes with its centre of mass set
for a uniform material; its friction is 1.0 static and 0.9 dynamic. Positions handed to Isaac Lab are plain Python
floats.
