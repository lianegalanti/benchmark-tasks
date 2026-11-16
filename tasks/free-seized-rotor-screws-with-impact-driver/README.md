# Drive a hand impact driver with a hammer: hold its bit seated and square in a seized rotor screw, then strike the cap

A car's front hub and knuckle with its brake rotor, off the car for a wheel-bearing job, stands hub-up on a bench stand
on the table, so the rotor lies flat. Two countersunk Phillips screws (M6, PH3 recess) hold the rotor to the hub,
flush with its hat, one either side of the hub pilot. One is already out (its hole is open); the other is rusted in, as
they nearly always are after a few winters. A
hand impact driver with a PH3 bit stands nose-down in a wooden block to the right, and a 16 oz ball-peen hammer stands
on its face on a stepped wooden stand to the left, its handle resting on the step. The robot must:

1. seat the bit fully in the screw's recess and hold the driver square to the screw;
2. strike the driver's cap with the hammer, square and hard: the blow drives the driver's body down over its anvil (a
   3 mm stroke against a return spring) and the cam inside turns the bit 12 degrees; three good blows (30 degrees)
   break the rust's hold;
3. then put both tools down off the hub and let go.

Instruction: *"Break the seized rotor screw loose with the impact driver and the hammer, then put both tools down off the hub and let go of them."*

Rotor changes are a frequent shop job on Honda Civics, Accords and CR-Vs, among the best-selling cars in the US, and on
many other Japanese cars, where the rotor is held to the hub by two such screws, almost always rusted in after a few
winters. Rotors are usually changed on the car (the hub's axis horizontal); the bench version here, as in a
wheel-bearing job, is the less common case. Automotive service technicians and mechanics hold 825,800 US jobs (BLS,
2025). A screwdriver only cams out of a seized screw and strips its recess; the shop tool is the hand impact driver.

What makes it hard:
- **Two hands, one job.** One hand holds the driver seated and square on a 6.6 mm recess while the other swings a
  0.62 kg hammer onto its 39 mm cap, blow after blow, without knocking the driver off the screw.
- **The craft the score measures.** A hard blow on the screw is good only if, just before it, the bit is fully home (at
  least 2 mm into the 3.6 mm recess, within 1 mm of the screw's axis) and the driver within 5 degrees of square, and
  the blow bottoms the driver's stroke. Otherwise it cams the bit out and damages the recess; a second cam-out strips
  it, and the screw then has to be drilled out.
- **A real blow.** A blow is a PhysX contact between the hammer and the driver passing at least 0.2 N s in one physics
  step, after a wind-up of at least 5 cm, with the face meeting the cap at 1.8 m/s or more, square (its centre over the
  39 mm cap, within 20 degrees of square), on a driver a hand is holding. At 1.8 m/s the 0.45 kg head carries about
  0.7 J: a firm deliberate blow but under half a mechanic's (2 to 6 J at 3 to 5 m/s); the turn is not tied to the
  blow's energy, so 12 degrees per blow is generous. Taps, glancing blows and blows on a driver in the air do nothing.
- **No shortcut by hand.** Twisting the driver never turns a seized screw.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the blow detector (every physics step), the per-step monitor, `success()` and `score()` |
| `geometry.py` | Dimensions of the hub, rotor, screws and recesses, driver, hammer and block; the seat and blow rules; box maths |
| `make_assets.py` | Writes `assets/hub.usda` (one articulation: fixed base, the screw on a revolute joint), `driver.usda` (a free articulation: body, the sprung 3 mm stroke and the cam-turned anvil and bit), `hammer.usda`, `block.usda`, `stand.usda` |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files (hooks yam_arena without modifying it; adds a per-physics-step callback and PhysX contact forces between two bodies) |
| `demo.py` | The scripted demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: layout and sizes, the recess and the bit's seat, every kind of blow, every scoring level and violation, the demo's reach and clearances |

`scenekit.py` is adapted from an earlier scene helper in this repository (its license applies).

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env` elsewhere):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Break the seized rotor screw loose with the impact driver and the hammer, then put both tools down off the hub and let go of them." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/rotor-screws-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo works from the fixed, known layout and the arms' own encoders only (arm joints, used also to feel whether the
bit is in the recess, and the two finger joints of each gripper); it never reads the driver's, the hammer's or the screw's state from the simulator, and does not check
whether a blow landed: it strikes the screw four times (three good blows break it loose; the fourth is spare).

1. **Driver (right arm).** It takes the driver out of its block by the knurled body, approaching from behind and to
   the right, fingers across the body, lifts it clear and brings it upright over the seized screw.
2. **Seat.** It lowers the bit into the recess and commands it 6 mm past fully home, so the soft arm keeps it pressed
   home and square. After each blow, while the hammer winds up, it lifts the bit 6 mm and presses it home again (a
   blow's rebound can lift it out). It feels for the recess with the arm's own encoders: fully home, the bit sits
   3.1 mm lower than on the head's rim, so the hand's measured height (forward kinematics of its joint readings, against
   its height at the first seat) tells whether it went in. After a blow, if the hand stays high it lifts
   and tries small offsets round the spot until it drops in, and keeps that offset. The hammer waits until the bit is home.
3. **Hammer (left arm), meanwhile.** It takes the hammer off its stand by the handle, 12 cm from the head, from
   behind, the fingers closing on the top and bottom of the handle, and lifts it.
4. **Blows.** It brings the hammer over the cap and winds up from the elbow: the elbow 0.45 rad up from where it is
   when the face rests square on the cap (the face then about 23 cm up). For each blow the elbow's target jumps at once
   to 0.30 rad past that, the other joints held, so the elbow (its full 27 N m) swings the head down its arc, 11 degrees
   off vertical at the cap, and the face meets the cap square. After 0.3 s the target comes back to the hammer resting
   on the cap (0.3 s); then it winds up again. The swing is aimed 18 mm forward and 12 mm right of the cap: the soft
   wrist flexes in the swing and the face lands that far short and toward the hand (a fixed allowance measured in
   Isaac Sim, not sensed). Four blows.
5. **Finish.** The driver goes back into its block and the hammer back on its stand; both hands let go and return to
   their ready pose.

Every arm target is solved with IK on the URDF. A slow integral term on the measured joint error (the arms' encoders)
takes out the soft wrists' sag; it is frozen while the bit is pressed home and while the hammer swings. Each gripper's
two fingers are separate force-limited joints, so a held part can sit off-centre between them; the demo reads where
from the two finger encoders (their midpoint is (q7 - q8) / 2 along the closing axis) and shifts the hand's targets
to match. The plan takes 805 control steps (40 s) without any search by feel; in Isaac Sim the hammer also waits
for each re-seat.

**Measured on an L40S in Isaac Sim:** the demo scored 1.0 (success held at step 1,353, 68 s). All four blows were
good: the face met the cap at 2.15 to 2.23 m/s (contact impulse 1.7 to 2.6 N s, the face's centre 2 to 18 mm off the
cap's), each blow drove the 3 mm stroke from 0 to the bottom (3.0 mm deeper) on a driver the right hand held, and
just before each the bit was fully home (3.11 to 3.12 mm deep, at
most 0.19 mm off the axis, at most 2.8 degrees off square). The screw turned 48 degrees with no cam-out; the arms'
encoders never had to search for the recess. Dream-1 scored 0 in a 3,600-step trial (the run limit).


## Scoring

The monitor runs after every control step (20 Hz), and the blow detector after every physics step (240 Hz). The
score at the end of the episode counts (`SCORE_OVER = "end"`). Everything is judged in the hub's frame.

**A blow** is a PhysX contact between the hammer and the driver's body passing an impulse of at least 0.2 N s in one
physics step, after the hammer's face has been at least 5 cm off the cap; resting or pressing the hammer on the cap is
not a blow. Its speed is how fast the face closed on the cap over the physics steps just before the contact. It is
judged 50 ms (12 physics steps) later, once the driver's stroke has had time to close. What it does:

| Blow | Condition | Effect |
|---|---|---|
| unheld | no gripper within 5 mm of the driver's body at the last control step | none |
| glancing | face's centre off the cap (more than 19.5 mm from the driver's axis; the limit is an estimate of where a blow stops driving the cap squarely), or the face more than 20 degrees off square | none |
| tap | closing at under 1.8 m/s | none |
| in the air | the bit's tip more than 8 mm from the screw (in the air, or in the empty hole) | none |
| stripped | a blow on a stripped screw | none (it has to be drilled out) |
| cam-out | just before it, the bit was not fully home and square: its tip more than 1.0 mm off the screw's axis, or less than 2.0 mm deep in the 3.6 mm recess (fully home is 3.1 mm), or the driver more than 5 degrees off the screw's axis | the recess is damaged; a second cam-out strips it |
| short | home and square, but the blow did not drive the stroke 2 mm deeper than it was just before the contact (so pressing the body down first and then tapping it does not count) | none |
| good | home and square, and the blow drove the stroke at least 2 mm deeper | the cam turns the bit and the screw 12 degrees counter-clockwise |

**0 if at any time** a tool falls off the table (its centre more than 5 cm below the table top). Setting a tool down,
regripping it, lifting the driver off a screw and re-seating it are all allowed.

**Otherwise, at the end:** the screw scores 0.8 x min(degrees turned / 30, 1), less 0.1 per cam-out (0 once
stripped, never below 0): 0.32 after one good blow, 0.64 after two, 0.8 once it is loose (30 degrees). Plus 0.2 once
it is loose and not stripped, and the tools are put away:

- both tools down off the hub: every part at least 5 mm from the hub, rotor, studs and screws (on the table or back in
  the block);
- every part of both grippers at least 2 cm from both tools, so nothing still holds them;
- both tools still (under 1 cm/s).

Without success the score is capped at 0.9 (so one cam-out, with everything else done, scores 0.9).
**Success (1.0):** the screw loose with no cam-out at all, the tools down off the hub, let go and still, held for
1 s. **Doing nothing, or refusing,** scores 0.

On a real hub, the angle is read from a paint line across the screw head and the hat (a protractor), cam-outs from a
burred recess (a loupe), and blows from a video at 240 fps (the face's speed) and a marked cap (the stroke).

## Simulation model

- **The hub and rotor** are one articulation fixed to the world: the base (stand, rotor ring and hat, the hat's face
  with an opening round each screw's hole, the hub pilot and the four studs), and the remaining screw's head on a
  revolute joint
  about its own axis. The head carries the PH3 recess: an eight-sided cone (34.8 degrees off the axis, 6.6 mm across,
  3.6 mm deep) built from eight petals and a floor; the Phillips cross is drawn on it but does not collide.
- **The driver** is a free articulation (0.65 kg): the body (the knurled grip, its collider a 12-sided prism 36 mm
  across the flats, and the striking cap); a slider on a prismatic joint along the axis, 3 mm of stroke, whose drive is
  the return spring (5,000 N/m, 20 N preload, 35 N bottomed); and the anvil (nose, bit holder and bit) on a revolute
  joint about the axis. Every physics step the task sets that joint's target from the measured stroke (12 degrees over
  the full 3 mm): the helical cam. The anvil's bit tip is a stepped cone on an 8 mm shank, so it self-centres in the
  recess and sits fully home 3.1 mm deep. The recess and the tip are shapes of revolution, so the bit does not grip
  the recess in torsion: twisting the driver cannot turn a screw, as with a seized screw.
- **The screw** is seized: its joint is held by a stiff drive (50 N m/rad, capped at 100 N m). The task moves the
  drive's target 12 degrees counter-clockwise for each good blow, and the head visibly turns.
- **The blows** are read every physics step: the contact force between the hammer and the driver's body from PhysX's
  contact view (both bodies carry the contact-report API), the stroke from the driver's joint, the face's speed from
  the hammer's pose. The pre-twist a technician puts on the body is not simulated; the turn per blow and the 1.8 m/s
  and 0.2 N s thresholds are estimates, not measurements (1.8 m/s is about 0.7 J in the head, under half a mechanic's blow).
- **The hammer** is one rigid body (0.62 kg): a hickory handle (grippy) and a steel head. Its face and the driver's
  cap are hardened, polished steel (friction 0.2); restitution is 0, so the hammer does not bounce off the cap.
- **The block and the stand** are kinematic (fixed); the block's pocket leaves the nose 4 mm of play each way.

## Real-world trials

- Use a scrap front hub and rotor on a bench stand, with one hole empty and a new M6 x 1.0 countersunk Phillips screw
  in the other, rusted in (salt spray for a few days) or set with thread-locking compound, so that 6 N m on a
  screwdriver does not turn it.
- The robot swings a 0.62 kg hammer at over 2 m/s from about 23 cm, onto a driver held by its other hand: keep people out
  of the arms' reach, wear eye protection near the bench (struck steel can chip), and stop the trial if the driver
  slips off the screw.

## Runtime and seeds

A full 3,600-step trial with Dream-1 takes about 6 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
