# Couple a tractor's service and emergency air lines to the trailer's glad hands

A cropped trailer nose stands at the far side of the table. A mounting bracket on it carries the trailer's two glad
hands, bolted face up: the blue (service) one on the robot's left and the red (emergency) one on its right. Between
them and the robot stands the tractor's tender post, with the ends of its two coiled air lines running out of guides
at its top; stretched toward the trailer, each coil pulls back on its glad hand. Each line ends in a tractor glad hand of the same colour, parked face down on a small stand in front of and inboard of its
trailer mate. The robot is told:

> Couple the tractor's service and emergency air lines to the trailer, then let go of both.

Heavy and tractor-trailer truck drivers (BLS 53-3032, about 2.0 million in the US) couple the air lines at every
hook-up, and yard hostlers (yard jockeys), who shuttle trailers around distribution centres, ports and rail yards,
couple and uncouple them many times a shift: the emergency (red) line charges the trailer's air system and releases its spring brakes, the service (blue)
line carries the brake signal. A trailer with only one line coupled can't be driven. So the task is both pairs. The
glad hand coupling itself is a physical skill, not a choice: each tractor glad hand starts beside its own mate.

The usual way to couple a glad hand pair:

1. **Pick it up** by its body and bring it face to face with the trailer glad hand, turned about 90 degrees to it,
   seal on seal.
2. **Press and turn.** Press the seals together and turn the tractor glad hand about 90 degrees about the seal axis.
   Each glad hand's flange (its "tongue") slides under the other's lip (a hook with a chamfered entry) until it hits
   the stop at the lip's far end. The pair then lies in line.
3. **Let go.** The compressed rubber seal pushes the flanges up against the lips, and that holds the coupling. A
   tooth on each flange's tip sits in a groove beside the mate's curved lip, so the line's pull can't slide the pair
   apart along its length.

What makes it hard (physically, not in recognising anything):
- **Alignment.** The faces must meet square and centred within a few millimetres, at the right angle, so the flanges
  miss the lips going in and find them going round.
- **Press while turning.** The flanges only pass under the lips with the seal compressed (the lips leave 2.5 mm of
  play at full compression; the chamfer helps a flange that arrives a little high). The turn has to be made about
  the seal axis, against the seal's spring and friction, and must go all the way to the stop.
- **The line.** Each glad hand is on the end of its coiled line, which pulls back toward the tender post (up to
  about 15 N) and swivels with it; it must not be pulled out to its end.
- **Two pairs.** Both lines, one per arm in parallel or one after the other (each line is within reach of the arm
  on its side).
- **Ending matters.** A pair held in place by the robot scores less than one the lips hold. Dropping a glad hand
  scores 0.

### How this differs from existing tasks

The board has quick couplers and bayonets: a garden-hose quick-connect (push-to-lock sleeve), a hydraulic flat-face
coupler and an air-tank quick coupler (push and release a sleeve), and bayonet caps (lugs into slots on a fixed round
socket, turned by their own head). A glad hand coupling is neither: two identical hermaphroditic castings, one on the
end of a coiled line that pulls back and one fixed, meet face to face at 90 degrees and are turned a quarter turn
together; a seal spring, not a detent, provides the clamp, and the lock is two flange-under-lip hooks closing at
once. It also comes with the job context (two colour-coded lines on a tender) and is done twice, once per line.

## Real objects (full size)

- **Glad hands (4):** cast aluminium, the standard hermaphroditic type, about 115 x 70 x 45 mm (port end to flange
  tip, across the lip, back to lip top), 0.35 kg each, painted blue (service) or red (emergency). Seal 40 mm across.
  The flange is 16 mm long and 22 mm wide, 4 mm thick, with a 2.5 mm tooth on its tip; the lip is a curved 4 mm hook
  over it (radius 32-47.5 mm from the seal axis) with a 3 mm entry chamfer, a stop at its far end and, outside it, a
  7 mm groove (radius 47.5-54.5 mm, flared into a funnel at its mouth) for the mate's tooth, which leaves about 2 mm
  of play each way along the long axis. The
  trailer's seal ring stands 5.5 mm proud and presses in 5.5 mm: 3 N at rest plus 2 N/mm, 9 N when the lips hold
  it 3 mm in.
- **Trailer nose section:** a painted steel bulkhead panel 72 x 30 cm (2 cm thick) on the table, front face at
  x = 0.50, with a steel mounting bracket (66 x 19 cm, 8 mm plate, 3 cm above the table) carrying the trailer glad
  hands on 36 mm square pedestals, faces up, seal centres at (0.35, +/-0.21), 8 cm above the table. Each is turned
  so that the coupled pair's long axis points at (blue) or 24 degrees past (red) the line from its seal centre to its guide, so the
  coil's pull runs along the coupled pair (red: it also turns it onto its stop).
- **Tender post and lines:** a 30 mm square steel post, 23.5 cm tall, at (0.10, 0). At its top, 22 cm above the
  table, two pivoting hose guides (y = +/-0.03). Each line is the working end of a 15 ft coiled air line: a 16 mm
  hose that runs out of its guide (between 10 cm and 48 cm, its end stop) to the tractor glad hand's port, and that
  the coil pulls back with 1.2 N per cm it is stretched beyond its parked length (about 15 N at the farthest point
  of the carry, about 3.5-6 N once coupled; an estimate for a 15 ft coiled nylon line, which takes on the order of 10-20 N
  to hold stretched).
- **Parking stands (2):** two steel pads each, under the parked glad hand's lip and flange; the glad hands rest on
  them face down, seal faces 7 cm above the table, at (0.25, +/-0.12), each turned 90 degrees from its coupling
  angle (see setup).

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()` and `score()` |
| `geometry.py` | The glad hand casting's boxes, the coupling's frames and numbers, the layout, the lines' kinematics, box maths |
| `make_assets.py` | Writes `assets/trailer.usda` (nose, trailer glad hands, sprung seal rings) and `assets/tender.usda` (post and lines) |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files. It hooks yam_arena without modifying it. Adapted from the scene helper in this repository's `lock-before-loading` task folder (its license applies). |
| `demo.py` | The scripted full-score demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: the coupling geometry, seal spring, lines, every scoring level and violation, the demo |

## Run a trial

From this folder, in a shell with the yam_arena environment active. New shells on the walkthrough's Brev instance
have it; elsewhere, `source <ARENA_DIR>/yam.env`.

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Couple the tractor's service and emergency air lines to the trailer, then let go of both." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/glad-hands-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png and result.json
```

The demo is planned from the fixed, known layout and uses only each arm's own joint encoders. It never reads the
glad hands' poses or the seal rings' travel from the simulator. The monitor's state is only printed, as a log.
Both arms work at once, the left on the blue line and the right on the red one:

1. **Pick up.** Open, from above, the arm takes the parked glad hand by its back block (fingers across it, the
   gripper's axis on the seal axis), closes, and lifts it off its stand.
2. **Face to face.** It carries it over the trailer glad hand of its colour, face down and turned 90 degrees to it,
   and lowers it until the seals touch.
3. **Press and turn.** It presses the seals together (aiming 4 cm past the ring's full travel, where the faces'
   stop takes it; the arm's own stiffness, about 0.4 N/mm, then gives some 16 N) and turns the glad hand 102 degrees
   about the seal axis, 12 degrees past the stop (the wrist lags against the lips' friction): the flanges slide under the lips and stop.
4. **Let go.** It eases back to the locked height, so the ring pushes the flanges up against the lips, opens, lifts
   away and returns to its ready pose.

A slow integral term on the measured joint error (the arms' encoders) takes out the sag from carrying the glad hand
and its hose. It is frozen while pressing and turning. The plan takes about 560 steps (28 s). The fingers are commanded to a 36 mm gap on the 44 mm back block, so the grip stays stiff sideways.

Measured on an L40S, the demo reaches 1.0 (success) in 512 steps (26 s). Both pairs lock on the stop (turn within 0.1 degrees), the seal rings held 2.9 mm in by the lips, seal centres 2.1 mm apart (both). Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step. It latches violations for the whole trial. Otherwise the state at the
end of the episode counts (`SCORE_OVER = "end"`). Every coupling is measured in the trailer glad hand's own frame.

**0 if at any time:**
- a tractor glad hand falls faster than 0.6 m/s while no gripper finger is within 1 cm of it (a free fall of about
  1.8 cm; a dropped glad hand swings on its hose into the deck plate and gets damaged);
- a tractor glad hand's seal face goes more than 2 cm below the table top;
- a line is pulled out to within 3 mm of its end stop (pulled from the tender);
- a tractor glad hand is locked (as below) onto the other colour's trailer glad hand (lines crossed).

Lifting, carrying, turning, setting a glad hand back down on its stand or on the table, and touching anything are
all fine.

**Otherwise, per pair (blue tractor on blue trailer, red on red), at the end:**

| Credit | Stage |
|---|---|
| 0.1 | **Faces together:** faces parallel within 10 degrees, seal centres within 8 mm, the trailer's seal ring pressed in 0.5 mm or more |
| 0.2 | **Locked but held:** locked (as below) with a gripper still within 2 cm of that glad hand |
| 0.45 | **Locked and let go:** turned to within 5 degrees of the stop, faces parallel within 5 degrees, seal centres within 6 mm, the ring pressed in 2 mm or more, and no part of either gripper within 2 cm of that glad hand. At the locked turn that compression is only possible with both flanges under the lips (otherwise a flange would sit on top of the lip, leaving the ring pressed less than 2 mm); the lips hold it at 3 mm. The teeth in the grooves keep the centres within about 2 mm along the long axis; across it the flange is between the stop and the turn's friction |

The score is the sum over the two pairs (at most 0.9). **1.0 (success, held 1 s):** both pairs locked, every part
of both grippers at least 2 cm from both tractor glad hands (let go), both glad hands still (under 1 cm/s and
0.1 rad/s) and both seal rings still, no violation.

- **Doing nothing, or refusing,** scores 0.
- **Holding a coupled glad hand in place** scores 0.2 for that pair; holding both, 0.4.

On a real table, with a ruler and a photo:
- each pair lies in line (long axes opposite), the tractor glad hand's flange against the stop under the trailer's
  lip, the two faces parallel;
- the gap between the two glad hands' faces is 4 mm (seal pressed 3 mm; 5 mm or less counts);
- nothing touches either tractor glad hand, and it stays put for a second.

How the simulation judges these, once per control step:
- The glad hands' poses, the seal rings' travel and the lines' slide come from the simulator; speeds are differences
  between control steps.
- The grippers are boxes placed from each arm's encoders: each finger's pad and root, and the carriage. A glad hand
  counts as a box around its whole casting.

## Real-world trials

- Use real glad hands (standard aluminium, with new seals) on short hoses, or blanked-off ones; no air pressure is
  connected. Bolt the trailer pair to a bracket and the tender post to the table.
- Everything happens below 25 cm above the table; a dropped glad hand hangs on its hose.
- The only pinch point is between the two glad hands' flanges and lips, as when a driver couples them.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

Simulation details (fidelity):
- 240 Hz physics, 20 Hz control. Glad hands, nose and post are rigid boxes (the casting as 24 boxes, the curved lip in 8-degree segments; the round seal
  boss and seals as three boxes turned 60 degrees apart).
- Trailer glad hands are fixed (bolted); each seal ring is a prismatic joint with a spring (2 N/mm, 3 N preload, 5.5 mm
  travel). The tractor glad hand's seal is rigid; one sprung ring stands for the two rubber seals.
- Each air line is the working end of a coiled line: a straight hose segment that slides through a pivoting guide
  at the post (two damped pivots and a slide), with the coil's return pull as a spring on the slide (120 N/m from the
  parked length), and a free damped three-axis swivel at the glad hand for the hose's flex. The hose has no
  collision; 0.1 kg of it moves with the glad hand.
- Friction: castings 0.6, rubber 0.9, steel 0.5.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
