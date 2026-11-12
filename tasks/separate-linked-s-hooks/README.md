# Unhook two linked S-hooks and hang each on its own support

A shop display stands on the table in front of the left arm: an upright plywood board with a chrome rail on two
brackets and, lower and further right, a straight chrome peg pointing at the robot. Two large aluminium S-hooks
(15 cm tall, 10 mm rod) hang from the rail hooked into each other: hook A hangs on the rail by its top curl, edge-on
to the robot, and hook B hangs by its top curl from the bottom of hook A's lower curl, facing the robot. The robot
is asked to separate them and hang each one by itself, one on the rail and the other on the peg, without putting
either down.

To do it, the robot has to:
1. take hold of hook B;
2. lift it about 2 cm, until the bottom of hook A's curl, which runs through B's curl, is lower than the tip of
   B's curl (pulling sideways before that drags hook A along, or off the rail);
3. move it sideways, toward B's open side, so A's curl leaves through B's opening; the hooks are now apart;
4. carry hook B to the peg, turned so that its curl can go round the peg, and hang it there;
5. let go and move away, leaving each hook hanging still by itself.

Category: **Impact on jobs.** S-hooks hooked into one another are an everyday nuisance for staff restocking
hardware and kitchenware displays, and in kitchens and butcheries where hooks are hung together after washing.
Separating them is a small topological manipulation: the hooks can't be pulled apart, only lifted and passed through
each other's openings in the right order.

The task is the inverse of assembling a hanging carabiner chain: there, links are clipped through each other's
closed gates; here, two open curls must be unlinked by lifting and sliding, without the gripper or the freed hook
knocking the other one off its support.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions, the layout, and the geometric tests the scorer uses (on a support, hooked together, the gap between the hooks) |
| `scenekit.py` | USD bodies and posed fixed shapes for yam_arena task files (copied unchanged from the `lock-before-loading` task folder in this repository; its license applies) |
| `make_hook.py`, `assets/s_hook.usda` | An S-hook: one rigid body of 17 capsules along the rod's centre line (needs `usd-core` to regenerate) |
| `demo.py` | The scripted demo (left arm), closed loop on the measured pose of hook B |
| `test_task.py` | Offline checks: layout and clearances, every scoring level, and the demo played against a rigid-grip hook B |

`scenekit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task
runs with the standard `yam_arena` embodiment and `task_score` scorer.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The two S-hooks on the display rail are hooked into each other. Separate them and hang each one by itself, one on the rail and the other on the peg, without putting either down." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/s-hooks-dream

# Offline checks (no Isaac Sim)
PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the trial
(`SCORE_OVER = "end"`; the score itself keeps the best milestone reached, cleared at every reset, so that a hook put
down on the table can still zero it).

## Scripted demo

```bash
python demo.py task.py demo_out     # writes demo_*_cam.mp4, setup.png and result.json
```

`yam_scripted` plays only straight-down grips, so `demo.py` drives the embodiment itself. The left arm:
1. passes a waypoint clear of hook A and the rail, then comes in from the front at hook B, pointing forward and
   8 degrees down along B's middle stroke, fingers half open, and closes on the stroke 3 cm below the hook's centre.
   The fingers close in the hook's plane, square to the stroke, so each pad lies along the rod and the grip holds
   the hook like a vise (a top-down grip would put the gripper's palm into hook A, which hangs right above);
2. lifts hook B 2.8 cm and moves it 6 cm to the right (-y): hook A's curl drops out under B's tip and leaves through
   B's opening, with the hooks at least 5 mm apart throughout;
3. backs off 4.5 cm, measures B's pose and lines its top curl up with the peg's axis 3.5 cm in front of the tip
   (B already faces the robot, as the peg needs), measures again and corrects once more, then slides it 6 cm onto
   the peg, lowers it until its curl is 2 mm above the peg, and half-opens the fingers;
4. backs straight out along the gripper's axis and returns to the ready pose by the same waypoint.

The plan is 532 steps (27 s); success needs 1 s more. Offline (`test_task.py`), against a hook B held rigidly in the
grip: IK error under 0.5 mm and 0.1 degree, joints within limits, and the arm's URDF collision hulls stay 6.8 cm above
the table, 2.1 cm from the board, brackets, rail and peg, 1.8 cm from hook A and 5.6 cm from hook B away from the
fingers (half-open fingers 1.4 cm from it); hook B stays 1.25 cm from the display while carried (the closest is the
peg passing its middle stroke inside its curl), and the peg runs through its curl with 1.2 cm to spare.

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | one hook on the rail and the other on the peg (below), not hooked together and at least 3 mm apart, neither held, both grippers' centre lines at least 4 cm from both hooks, both still (under 2 cm/s) for 1 s | photo |
| 0.5 | at some point the hooks were not hooked together (held or not) | video |
| 0.25 | at some point the hooks were at least 1 cm apart everywhere (one lifted off the other) | video, against a 1 cm gauge |
| 0 | otherwise, or a hook touched the table (its lowest point within 5 mm of the table top) or fell off it at any time | video |

**On a support:** the rail or the peg passes through one of the hook's curls, crossing the curl's plane less than
2.5 cm from the curl's centre (the curl is 2.3 cm in radius inside); a hook may hang by either curl. **Hooked
together:** either hook's rod passes through a curl of the other in the same way. Two hooks both on the rail, or a
hook hanging from the other, never succeed. Putting a hook down on the table, even to regrip it, ends the trial's
credit, as the instruction says.

How the simulation judges these, once per control step: each support's axis segment and each segment of the other
hook's rod (the 17 capsule axes) is intersected with the hook's plane and the crossing compared with both curl
centres; gaps come from the rods' centre lines sampled every 2 mm, less the rods' radii; a gripper **holds** a hook
when its grasp point is within 3 cm of the rod's centre line and its fingers are less than 3 cm apart; a gripper's
**centre line** runs from its wrist (the gripper link) to its fingertips.

## Runtime and seeds

A 1,800-step trial with Dream-1 takes a few minutes on an RTX GPU, plus Isaac Sim's start. The layout is fixed
(`-E object_jitter=0`), so there are no seeds to choose. The demo measures hook B's pose before gripping it and again
before sliding it onto the peg, so it does not depend on exact timing.

On an L40S the scripted demo ended with success after 461 steps (23.1 s); Dream-1 scored 0 in a
1,800-step trial (the run limit).

## Fidelity

Rigid bodies, 240 Hz physics. Each S-hook is one rigid body of 17 capsules of 5 mm radius along its centre line:
each 160-degree curl is 8 straight chords of 20 degrees, within 0.4 mm of a true arc, and a chord's middle sits at
the top of each curl, so a hook rests centred on a support. The board, rail, brackets and peg are fixed colliders.
Friction is 0.7 static and 0.6 dynamic on the hooks. Rod flex, plating and a real display's give are not simulated.
Positions handed to Isaac Lab are plain Python floats.
