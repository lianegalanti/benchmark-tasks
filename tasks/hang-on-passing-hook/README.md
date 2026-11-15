# Hang steel strut plates on the wire hooks of a moving overhead conveyor

Two plain steel strut plates stand on end in wooden loading blocks, one in front of each arm. Each is a standard
3-hole flat plate fitting for 1-5/8 in strut channel (the Unistrut P1066 pattern): 1/4 in (6.35 mm) hot-rolled steel,
5-3/8 x 1-5/8 in (136.5 x 41.3 mm), 0.257 kg, with three 9/16 in (14.3 mm) holes on its centre line, 1-7/8 in
(47.6 mm) apart and 13/16 in (20.6 mm) from each end. An overhead rail across the far side of the table carries three
hooks bent from 4 mm steel wire, 30 cm apart, from the robot's left to its right at a steady 5 cm/s; they never stop.
Each hook hangs from the rail on a stem and ends in a horizontal leg pointing back toward the robot, 15 cm above the
table, with a 4 mm upturn at its end. The robot must hang each plate on its own hook by an end hole, which is 4.8 cm
above the plate's centre of mass so the plate hangs plumb and still below the hook, before that hook passes a red
exit mark at the right of the table, and let go of it. To do so it has to lift the plate out of its block, carry it
to a hook as the hook comes into reach, move along with the hook, slide the 14.3 mm hole over the 4 mm wire and its
4 mm upturn (3 mm to spare above and below), lower the hole's edge onto the leg and let go, all while the hook keeps
moving.

Stamped and cut steel parts such as strut fittings and brackets are painted, powder coated or plated hanging on
wire hooks from overhead conveyors. Loading them is a common manual job: the loader picks a part and threads its
hanging hole onto the next passing hook, matching the line's pace, since the line doesn't stop for one loader. A
part hung by a hole above its centre of mass hangs still; one hung badly swings into its neighbours or falls off
further down the line, and an empty hook is lost throughput.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the hooks' motion, the loading monitor, `success()`, `score()` and the scripted demo |
| `geometry.py` | Dimensions of the conveyor, the wire hooks, the plates and their loading blocks, and the tests the score uses (which hole a hook's leg passes through, is that an end hole above the centre of mass, does the plate rest on the leg, how far from plumb, how low) |
| `make_part.py` | Writes `assets/strut_plate.usda`, the plate as one rigid body of ten boxes around its three holes (needs `usd-core`) |
| `motionkit.py` | Adds a `DrivenBox` object kind (a box whose pose and velocity follow a scripted path, set before every physics step), a `UsdPart` object kind (a free rigid body from a USD file), `on_step` / `on_reset` hooks with `arm_joints()` for the monitor, and `clock()` |
| `test_task.py` | Offline checks (no Isaac Sim): the plate's real dimensions and mass, the hooks' motion and the clearances, every scoring level, the ways of keeping hold or hanging badly that get no hanging credit (gripping the top end or the lower end, propping from below, any link within 2 cm of any part of the plate, letting go under 1 s before the mark, holding the hole off the leg, hanging by the middle hole, more than 10 degrees from plumb, still swinging or sliding), and the demo against rigid-grip plates, played through the scorer |

`motionkit.py` is a small helper module, shared with other task folders in this repository (their license applies).
It hooks `yam_arena.scene` and the embodiment's reset, simulation step and task scoring without modifying yam_arena,
so the task runs with the standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. The hooks'
clock is the episode clock (`observation.state_time`): they stand still at their start through the reset and start
with the first control step. Written against yam-isaac-lab-arena at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured on
an L40S:** the scripted demo ended with success after 661 steps (33 s); Dream-1 scored 0 in a 900-step trial.

## Run a trial

On a machine set up with the official simulator (the Robocurve Brev launchable, or `yam-isaac-lab-arena`'s
`setup.sh`), with a Dream API key in `~/.dreamlabs/credentials`, from this folder:

```bash
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the two steel plates on the passing conveyor hooks, one plate per hook, each by the hole at one of its ends so that it hangs straight down, before the hooks reach the red mark, and let go of them. The hooks keep moving and won't stop for you." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/hooks-dream

# The scripted demo, which should reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Hang the two steel plates on the passing conveyor hooks, one plate per hook, each by the hole at one of its ends so that it hangs straight down, before the hooks reach the red mark, and let go of them. The hooks keep moving and won't stop for you." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/hooks-demo

# One MP4 per camera, next to each run's frames
for log in ~/yam-eval/hooks-dream/*.json; do inspect-robots video "$log"; done

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

The run prints `task_score` at the end: 1.0 once both plates have been credited (below) and that has held for 1 s,
otherwise the score at the end of the trial. Run with `-E control_hz=20`: the demo follows the hooks with one
waypoint per control step.

## The scripted demo

The demo is open loop: it works from the fixed, known layout and the hooks' published schedule (each hook's start
position and its constant 5 cm/s), and never reads the simulator's object poses. Both arms pinch their plate's two long edges at its
top end (fingers closing across the 41.3 mm width, pad tips 4 mm below hole 0's centre and 13 mm to each side of the
wire), lift it straight out of its block and keep it upright, holes facing the hooks. The fingers are commanded 2.5 mm inside the
plate's half width rather than fully closed, so they hold it centred (commanded fully closed, both push at their
effort limit and the plate slid 1.8 cm sideways in the hand while being lifted, and hole 0 missed the wire). While it
follows a hook, the plan aims 2.5 mm ahead of the hook and 3 mm high, the arm's lag and sag measured in Isaac Sim, so
the hole meets the wire centred.

- **Left arm, plate A onto hook 0.** Picks the plate (0 to 4.3 s) and carries it to 4 cm in front of hook 0's upturn,
  hole 0 level with the leg (the wire and upturn pass with 3.1 mm above and below), arriving as the hook comes 42 cm
  left of the midpoint (5.6 s). Then, with a waypoint every control step at the hook's speed: matches it 0.5 s,
  slides hole 0 9 cm forward over the upturn and onto the leg (1 s), lowers the plate 5 mm until the hole's upper
  edge is 2 mm above the leg (0.8 s), opens the fingers to 7.2 cm (0.4 s, let go by 8.7 s), rides along open 0.3 s,
  rises 6 cm while following, and parks behind the hooks' path.
- **Right arm, plate B onto hook 1.** Picks its plate the same way, waits holding it above its block, and runs the
  same sequence on hook 1 from 21.6 s, when hook 1 is 8 cm right of the midpoint (let go by 24.7 s).

Hook 0 passes the exit mark at 26 s and hook 1 at 32 s; in Isaac Sim the trial ended with success after 661
steps (33 s). Both arms stay parked until 35 s, so that `yam_scripted` (which stops 3 s after the plan)
doesn't end the trial early. Offline (`test_task.py`), against plates held rigidly from the moment the fingers close
until they open: IK error under 0.5 mm and 0.1 degree, joints within limits, the carried plate never closer than 2 mm
to any hook while it is threaded on, and every arm link's URDF collision hull (fingers as wide as the plate while
closed on it) at least 1.5 cm from every hook, the rail and the other plate, checked every 0.2 s. Played through the
scorer's monitor (every arm link's hull, fingers at their commanded opening), each plate hangs freely from 9.3 s
(plate A) and 25.3 s (plate B), with no arm link within 5 cm of it in the last 1.5 s before its hook's exit, and the
demo scores 1.0.

## Scoring

Per plate, summed over the two plates:

| Per plate | Condition | Check on a real table |
|---|---|---|
| 0.5 | hung freely and settled on its own hook for the whole last 1 s before that hook passes the red exit mark (the leg's centre over the mark's centre line): at every step of that second, the hook's leg through one of the plate's end holes, that hole above the plate's centre of mass (the middle hole, at the centre of mass, doesn't count), its upper edge resting on the leg (at most 5 mm between them); the plate within 10 degrees of plumb, turning slower than 0.5 rad/s and moving within 3 cm/s of the hook (not swinging or sliding); its lowest point at least 1 cm above the table; and nothing but its hook touching it: every link of both arms (both fingers of each gripper, the gripper bodies and the rest of the arms) and the other plate at least 2 cm from every part of it; and the robot never held that hook back (its leg never more than 1 cm behind its 5 cm/s schedule) | video, for the second before each hook crosses the mark (a plumb line or square on the hung plate; settled: between frames 0.1 s apart it turns under 3 degrees and moves under 3 mm against its hook); a 2 cm gauge between fingers or other plate and the plate; a stopwatch: each hook over the mark within 0.2 s of 26, 32 and 38 s |
| 0.25 | otherwise, at some point a hook's leg went through any of the plate's holes (held or not) | video |
| 0.1 | otherwise, at some point the plate's lowest point was at least 3 cm above the table (out of its 2.5 cm slot) | video, against a 3 cm mark |
| 0 | otherwise | |

A hook credits at most one plate: two plates on one hook score 0.5 + 0.25. A credited plate stays credited after its
hook has left. Any touch in that last second (fingers on the plate's edges, a gripper holding or propping its lower
end, a wrist resting against it, the other plate against it), a swing or a slide restarts the count, so a plate
still held anywhere, let go less than 1 s before the mark, or still swinging then doesn't get the 0.5. In the
simulator the monitor places the URDF's collision hulls of every arm link at the simulated link poses each control
step and measures their distance to the plate's ten boxes. A real slip-clutch conveyor can be stalled by an arm,
while the simulated hooks can't: so a hook the robot stops or slows (its leg ever more than 1 cm, 0.2 s, behind its
schedule; in the simulator, the measured leg position against the schedule) credits no plate. Both plates credited
(1.0) is success. "The leg through a hole" means the leg's centre line crosses the plate's mid-plane inside that
hole; a plate laid across the leg or held against a hook is never through a hole. The plate is symmetric, so it may
be hung by either end hole (turned end for end, hole 2 is on top).

## Runtime and seeds

The layout is fixed (`-E object_jitter=0`): there are no seeds to choose, and the hooks always start in the same
place. The last hook passes the exit mark at 38 s, so a 900-step trial (45 s) covers every chance to score. A
900-step trial with Dream-1 takes a few minutes on an L40S, plus Isaac Sim's start.

## Real-world trials

Drive the hook carriers with a low-force belt and slip clutch at 5 cm/s, so a hook blocked by an arm stops instead of
pushing, and put a padded tray under the hooks' path and the loading blocks to catch a dropped plate (0.26 kg, with
a 2.7 cm fall from a hook). Break the plates' cut edges before use.

## Fidelity

Rigid bodies, 240 Hz physics. The hooks are driven exactly along the rail like a stiff chain drive: the arms can't
push or stop them, where a real line's slip-clutch drive could be stalled by an arm (see the reset notes); the scorer
gives no hanging credit on a hook more than 1 cm behind schedule. Each hook is three boxes of 4 mm square section
(stem, leg and upturn) moving together with 1 mm gaps between them, hanging 5 mm under the rail without touching it;
round wire is modelled square, and the plate's round 14.3 mm holes as 14.3 mm square openings (a hung plate rests
on a flat edge instead of a round one, so it settles a little faster than a real one). The plate is one rigid body
of ten boxes with the real plate's mass (0.257 kg, with round holes) at its centre; steel friction 0.6 static and
0.5 dynamic. The loading blocks are four fixed boxes each; the rail's end supports, beyond the table, are not
modelled. Positions handed to Isaac Lab are plain Python floats.
