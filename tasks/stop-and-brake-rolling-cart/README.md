# Stop a runaway cart and set its brake before letting go

A grey track deck (22 x 108 cm) lies across the table in front of the arms, sloping down 1.2 degrees toward the left
and ending 1 cm short of the table's left edge. Two low-friction rails and a central guide bar run along it. A blue
model cart (12 x 16 x 6 cm, 330 g) sits on the rails in front of the right arm with its brake off, and is let go at
rest when the trial starts. It rolls down the slope, passes in front of the left arm about 2.1 s later at about
0.3 m/s, and runs off the end of the track and the table after 3.6 s.

The robot must:
1. stop the cart before it runs off;
2. set its brake: push the orange lever on the cart's near side over toward the cart, past upright, to its stop.
   Its rubber pad then presses on the deck and holds the cart on the slope;
3. let go of the cart.

On the slope a stopped cart is not safe: let go unbraked, it rolls off again. The robot has to catch a moving load,
secure it, and only then release it. Floor-lock brakes on carts work the same way.

This differs from the stop-sliding-carton task (catch a sliding carton and push it to a spot on a level deck) and from
the board's static chocking tasks: here the load is moving, and stopping it is not enough, because the slope keeps
pulling on it. It is only safe once the robot has worked the cart's own brake mechanism, which is part of the moving
object.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the track, the cart, the arms-clear monitor, `success()`, `score()` and the scripted demo |
| `geometry.py` | Dimensions, the track frame, the brake lever's kinematics (pad depth, contact point, handle face), the analytic roll |
| `make_assets.py` | Writes `assets/cart.usda` (needs `usd-core`; the file is committed) |
| `assets/cart.usda` | The cart and its brake lever: a free-floating USD articulation with one hinge, `brake` |
| `scenekit.py` | USD bodies, articulations and posed fixed shapes for yam_arena task files |
| `test_task.py` | Offline checks (no Isaac Sim): geometry, the lever's toggle and hold, the roll, the scoring, the demo |

`scenekit.py` is copied from the close-door-past-hotplate task folder, which copied it from lock-before-loading. This
copy adds two things:
- `UsdBody.free_start`: after every reset, the free cart is put back at its start, at rest, with its lever stowed;
- `Shape.friction`: fixed shapes with their own friction, here the rails' low friction, combined as the minimum.

It hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the
standard `yam_arena` embodiment, `yam_scripted` policy and `task_score` scorer. Written against yam-isaac-lab-arena
at `bd62771` (Isaac Lab-Arena 0.3.1). **Measured in Isaac Sim:** the scripted demo scores 1.0 (`succeeded`, 179
steps). The cart resets to its start at rest, rolls as the analytic roll predicts (0.31 m/s at the left fingers at
2.0 s), stops 1 cm from the left TCP, and ends braked at the lever's 12-degree stop, 4 mm off the centre line, 1
degree of tilt, still; the arms are clear from 7.0 s and the 2 s hold ends the trial at 9.0 s.

## Run a trial

From this folder, in a shell with the yam_arena environment active:

```bash
INSTR="Stop the rolling cart before it runs off the end of its track, then set its brake by pushing the orange lever over toward the cart, and let go of the cart."

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/cart-dream

# The scripted demo, meant to reach full score
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$INSTR" --scorer task_score --max-steps 400 --store-frames --log-dir ~/yam-eval/cart-demo

# Offline checks, in Arena's venv
python -m pytest -q test_task.py
```

`task_score` prints the trial's score: 1.0 on success (held for 2 s). Otherwise it scores `score()` at the end of the
trial (`SCORE_OVER = "end"`), so a cart that was stopped and then let roll off scores 0.

## Scoring

| Score | Condition | Check on a real table |
|---|---|---|
| 1.0 | All of these, held for 2 s, and the cart never left the track:<br>- square on its rails: within 8 mm of the centre line and 4 degrees of the slope<br>- brake lever at least 8 degrees past upright toward the cart (its stop is at 12)<br>- still: under 5 mm/s<br>- every part of both arms at least 2 cm from the cart and the lever | video; a protractor against the lever |
| 0.75 | At the end: still on its rails with the brake set, but an arm within 2 cm (or not yet held for 2 s) | video |
| 0.5 | At the end: still on its rails, brake not set (on the slope, only something holding it keeps it there) | video |
| 0.25 | At the end: on the track and upright, but still moving or pushed off its rails | video |
| 0 | At any time, the cart ran off the end of the track or was pushed or lifted off it. Or, at the end, it is tipped over (more than 30 degrees) or more than 3 cm off the rails | video |

How the simulation judges these, once per control step:
- **The cart's position** is that of its body's origin: on the plane of the rail tops, under its centre. It is taken
  in the track's frame: distance up the slope from the track's end, sideways offset, height above the rails.
- **Left the track:** the centre more than 1 cm past either end of the track, 3 cm below the rails, or 11 cm to the
  side. Once this happens, it is latched for the trial.
- **Arms clear:** each arm is checked as 3 cm capsules between its joints, and the gripper as a 2 cm capsule out to
  the fingertips. They must be 2 cm from a box round the cart's body and a box round the lever where it is now.

## Scripted demo

1. **Stop.** The left arm stands its closed fingers (closing along the track) in the cart's path in front of it,
   3 cm toward the far side of the centre line (away from the right arm), tips 2.5 cm above the rails, by 1.2 s. The cart rolls into their flat side at about 2.1 s and stops, held against
   them by the slope.
2. **Brake.** At 3.3 s the right arm is above the lever's handle. It comes down 1.5 cm outside the leaning handle,
   tips 4.5 cm above the hinge (above the cart's top), and pushes the paddle 9.5 cm toward the cart over 1.6 s, aiming
   2.3 cm past the handle's face at the stop (the arm gives about 1.5 cm against the lever while its pad lifts the
   cart). The lever goes past upright to its stop, and it holds there 0.3 s.
3. **Let go.** The right arm backs off 7 cm and rises. The left arm backs off 4 cm along the track and rises. Both
   are clear (2 cm) at 7.0 s.

The success hold then runs 2 s: in Isaac Sim the trial ends with 1.0 at 9.0 s of simulated time (179 steps).

## Runtime and seeds

- One fixed layout, `-E object_jitter=0`. The cart always starts at the same place, at rest. The demo is open loop,
  timed against the analytic roll, with margins:
  - the left fingers are in place 0.5 s before even a frictionless cart could arrive;
  - the right arm comes in 0.8 s after even a cart with 50 % more rail friction would have stopped.
- A 1,200-step Dream-1 trial takes about 2 minutes on an RTX GPU, plus about 1 minute for Isaac Sim to start.

## Fidelity and design notes

- **Wheels.** The cart slides on runners rather than rolling on wheels, which keeps the contact simple. The rails'
  friction is 0.006, combined as the minimum so it overrides the cart's 0.8. It stands in for the rolling
  resistance of ball-bearing wheels. The cart accelerates down the slope at 0.147 m/s². Static friction is also
  0.006, so a stopped, unbraked cart starts rolling again (tan 1.2 degrees = 0.021).
- **Guide.** Shoes under the cart straddle the guide bar, so an off-centre knock can't turn it off its rails.
- **Brake lever.** It is an over-centre toggle without a spring, with two resting positions:
  - **Stowed** at -40 degrees: its weight (centre of mass 2.5 cm above the hinge) holds it against the stop, and
    the pad is 6 mm above the deck.
  - **Engaged** at +12 degrees: the deck pushes up on the pad 6 mm outboard of the hinge, and the lever's weight
    also turns it into the stop. The cart's slide pulls on the pad along the hinge axis, so it doesn't turn the
    lever.

  Engaged, the pad would reach 2.5 mm below the deck, so it lifts the cart's near runner off its rail. The pad then
  carries about 40 % of the cart's weight. With friction about 0.7, it holds the cart with more than five times the
  slope's pull.
- **Self-collisions.** These are off. The lever never comes within 5 mm of the cart's body or runners anywhere in its
  range (tested).

**Checked in Isaac Sim:**
- `write_root_pose_to_sim_index` puts the free cart back at its start, at rest, with its lever stowed.
- The lever stays stowed while the cart rolls, and stays engaged at its stop once the arms leave.
- The cart stops with its face 1 cm from the left TCP (`FINGER_HALF` is set to this), the lever within the
  paddle's width of the right fingers.
- The first run pushed only 3 mm past the handle's face: the pad meets the deck at -24 degrees, the arm gave about
  1.5 cm against the lift, and the lever stalled at -18 degrees and fell back when the arm left. The push now aims
  2.3 cm past the face (`PUSH_PAST`).
