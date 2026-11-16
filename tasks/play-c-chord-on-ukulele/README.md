# Play C then A minor on a ukulele: four even downstrokes on each chord, every string ringing at the right pitch

A full-size soprano ukulele lies on its back in a padded cradle on the table, its neck to the robot's right. The robot
must play the first exercise of a beginner's ukulele class: four even downstrokes on a C chord, then four on A minor,
on the same beat.

- **C:** the A string held down just behind the 3rd fret (291 mm vibrating, C5, 523 Hz); G, C and E open.
- **A minor:** the G string held down just behind the 2nd fret (308 mm vibrating, A4, 440 Hz); C, E and A open.
- **Each downstroke** sweeps G, C, E, A in order (toward the robot), each string caught and let go so that it rings,
  none touched while it rings.
- **The beat:** 60 to 100 beats per minute, every gap between strums within 10% of the mean, the chord change
  included.

Instruction: *"Play the ukulele: strum four even downstrokes on a C chord (A string held at the third fret), then change to A minor (G string held at the second fret) and strum four more on the same beat."*

The ukulele is a mass-market first instrument: more than 1.7 million were sold in the US in 2021 ([Statista, from The
Music Trades](https://www.statista.com/statistics/448450/number-of-ukuleles-sold-in-the-us/)), and it is taught in school music classes. Beginner methods start with these chords: Kjos's *Ukulele for
All* class method gives its first chord chapters to C and A minor (then F), with strumming exercises on each. What
makes it hard is physical:

- **Fretting is force and placement.** The string must be pushed below the fret crowns' plane just behind the wire
  until it bears on the crown (at least 0.5 N); farther back it takes a deeper press. On the wire, or behind the wrong
  fret, the note is muffled or wrong. The next string is 9.5-10 mm away and must not be touched; the robot's
  fingertip nub is 4 mm thick and 15 mm wide.
- **Strumming catches and releases.** Each string must be swung at least 1 mm and snap free so that it swings
  through its line and rings; too shallow and it is grazed, too deep and it snags or is slapped into the frets.
- **Timing and the change.** The strums must keep an even beat, and the fretting hand must move to a different string
  and fret between two beats without touching the strings still ringing.
- **Two hands at once.** The fretting hand keeps its string seated while the other hand sweeps.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the 240 Hz string monitor (plucks, rings, stops, strums), `success()` and `score()` |
| `geometry.py` | Dimensions of the ukulele and strings, the string model (stiffness, mass, loads from the motion), chords and pitches, fingertip model |
| `make_assets.py` | Writes `assets/ukulele.usda` (the instrument, fixed) and `assets/string_{g,c,e,a}.usda` (one articulation per string) |
| `assets/` | The generated USD files (committed) |
| `scenekit.py` | USD bodies and articulations for yam_arena task files, plus a per-physics-step hook (hooks yam_arena without modifying it) |
| `stringsim.py` | An offline model of the strings (the same equations, integrated without Isaac Sim) for the scoring tests |
| `tipkit.py` | The YAM fingertips' collision points, for the tests' clearance checks |
| `demo.py` | The scripted demo; `plan()` replays it without Isaac Sim for the tests |
| `test_task.py` | Offline checks: size, string model, assets, fretting and pitch, every scoring rule, the demo's reach, clearances and beat |

`scenekit.py` is adapted from the scene helper of this repository's `lock-before-loading` task folder (its license
applies).

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env` elsewhere):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Play the ukulele: strum four even downstrokes on a C chord (A string held at the third fret), then change to A minor (G string held at the second fret) and strum four more on the same beat." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/ukulele-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
python demo.py task.py demo_out      # writes demo_*_cam.mp4, setup.png, trace.json and result.json
```

The demo is **open loop**. It works from the fixed, known layout and the arms' own joint readings only, and never
reads the strings' or the ukulele's state from the simulator. The neck is to the robot's right, so the right arm frets
and the left arm strums, and a downstroke (G to A) moves toward the robot.

1. **C (right arm).** Gripper open, its closing axis along the neck, the finger leaning 35 degrees back over the nut
   side. The rounded end of one fingertip's nub comes down across the A string 5 mm behind the 3rd fret wire: its
   width reaches past the fingerboard's edge on one side and stops 3.7 mm short of the E string on the other. It is
   commanded 2 mm past the fingerboard's face (5 mm for the G string in A minor); the soft wrist stops on the
   string and the board, pressing. The other
   finger is 80 mm along the neck and 56 mm up, clear of everything.
2. **Downstrokes (left arm), every 0.98 s (61 bpm).** Gripper open and turned so that one fingertip's nub leads, the
   finger leaning 35 degrees with its tip pointing away from the robot (trailing, like a fingernail). Each downstroke
   sweeps 63 mm toward the robot in 0.5 s, smoothly, crossing the four strings, G to A, in about 0.2 s, over the
   soundhole's neck edge. Each string rides down the nub's leaning face to its rounded end and snaps off it. The nub
   runs 0.85 mm below the strings' centres at G, 1.75, 2.25 and 2.0 mm at C, E and A: the soft wrist springs up a
   few millimetres after each string lets go (the same profile for all eight). The hand then lifts 12 mm, goes back
   in the air and waits still for 0.17 s before the next downstroke.
3. **The change (right arm).** 0.66 s after the fourth downstroke starts (the A string has rung just over 0.2 s),
   the finger lifts 6 mm, crosses over E and C, hovers 0.1 s and comes down across the G string 6.5 mm behind the 2nd
   fret wire (3.7 mm clear of the wire, 4.7 mm clear of the C string), in 0.35 s, 0.13 s before the fifth downstroke reaches G. From the hover
   on, a fast correction from the arm's own encoders keeps the fingertip where planned across and along the neck
   (the soft arm otherwise lands 3-4 mm off).
4. **Finish.** The hands hold until the last strings have rung, then lift clear.

A slow integral term on the measured joint error (the arms' encoders) takes out the soft wrists' sag; it is what
builds the fretting hand's pressing force. It is frozen while a hand sweeps or changes chord; for the change the
fretting hand goes back to its sag-only integral in the air and takes the C chord's pressing integral onto the G
string.

**Measured on an L40S in Isaac Sim:** the demo scored 1.0 in 299 steps (14.95 s; success held 1 s after the last
strings rang). All eight strums sounded all four strings at the chord's pitch, in order G, C, E, A; strum onsets
0.95-1.00 s apart (mean 0.98 s, 61 bpm, every gap within 3.1% of it). The A string bore 1.4-2.2 N on the 3rd fret's
crown through the C strums, and the G string was seated on the 2nd fret (up to 3.4 N) at every A minor pluck. Swings
at the strum point: up to 6.3 (G), 3.1 (C), 3.0 (E) and 4.5 mm (A). The largest load read on a freely ringing string
was 0.14 N. Dream-1 scored 0 in a 1,200-step trial (the run limit).

## Scoring

The monitor runs after every physics step (240 Hz), because a pluck lasts a few milliseconds, less than a control
step. It reads only the strings' own joints, so everything is measured in the ukulele's frame. From each string's
motion it computes the load on every point of it (the string's equation of motion read backwards): that is how it
sees the frets, the fingers, the strum and anything muting a string.

- **Stopped at fret n:** fret n's crown bears at least 0.5 N and no crown beyond it is touched. A crown touched more
  lightly is a buzz. A string touching no crown is open. The vibrating length and pitch follow:

| Chord | G | C | E | A |
|---|---|---|---|---|
| C | open, 346 mm, G4 392 Hz | open, C4 262 Hz | open, E4 330 Hz | 3rd fret, 291 mm, C5 523 Hz |
| Am | 2nd fret, 308 mm, A4 440 Hz | open, C4 262 Hz | open, E4 330 Hz | open, A4 440 Hz |

- **A string sounds** when:
  - something pushes it (at least 0.3 N) to a swing of at least 1 mm, with nothing else touching it at the peak (no
    load of 0.15 N more than 25 mm along the string from the pluck's own contact);
  - it is let go: back under 30% of its peak within 50 ms (a snap, not a slow lift-off);
  - it swings through its line to at least 30% of the peak on the other side;
  - for 0.5 s it rings with nothing touching it (no finger, fret or fingerboard: no load anywhere of 0.15 N, or of
    35 N per metre of the pluck's peak swing if that is more) and its
    stop unchanged, or until the next strum plucks it again (no sooner than 0.4 s after it was let go; the push must
    become a new pluck, let go within 50 ms, or the ring counts as touched). Before the chord
    change, the fretted string may be let off its fret 0.2 s after it was let go (the fingers must move on).
  - The last 31 mm before the saddle (the last node, which carries the saddle-end spring) is not watched.
- **A strum** is the plucks let go within 0.25 s of the first, of at least two strings. Single plucks are not strums.
  **Only the first eight strums count** (no retries): strums 1-4 must be C, 5-8 Am.
- **A strum counts only if its fretted note sounds at the right fret** (the A string on the 3rd fret for C, the G
  string on the 2nd fret for Am). A strum without it earns nothing, so open strings strummed in time score 0.
- **Score** (0 to 1):

| Credit | For |
|---|---|
| 0.6 x share of the 32 strings | each string of each counting strum sounding at the chord's pitch |
| 0.2 x share of the 8 strums | counting strums with all four strings sounding, let go in order G, C, E, A |
| 0.2 x share of the 7 gaps | gaps between two counting strums' onsets (first let-go), within 10% of the mean gap, the mean 0.6-1.0 s (100-60 bpm) |

- **0 for the whole trial** if a string is ever pulled more than 12 mm out of line (it would jump its nut or saddle
  slot). **Doing nothing, or refusing,** scores 0. **1.0** (success): everything met.

Examples (from the tests): open strings only, strummed in time, score 0; holding C through all eight strums (no change
to Am) scores 0.49; upstrokes 0.8; one gap out of time 0.97; 50 bpm 0.8.

**Choices to know about:**
- **The strum window is 0.25 s,** not the 50-150 ms of a quick real strum. Faster sweeps make the robot's soft wrist
  fling the first string 10 mm and skate over the rest; 0.25 s is the shortest the arm crosses all four reliably.
- **The demo uses one depth profile for all eight downstrokes,** deeper at C, E and A than at G (0.85 to 2.25 mm),
  because the soft wrist springs up after each string lets go. It is a fixed feed-forward, not tuned per strum.
- **The touch threshold while ringing scales with the swing.** Measured in Isaac Sim, a string ringing freely reads
  up to 23 N per metre of its swing (G at 6.3 mm: 0.14 N), from the solver's residuals. A fixed 0.15 N would read a
  free 10 mm swing as a touch, so while a string rings a touch is the larger of 0.15 N and 35 N/m of its peak swing
  (0.42 N at the 12 mm limit). Real touches read far more: a fret slap 0.8-3.3 N.
- **The chord-change lift:** before the change, the fretted string may leave its fret 0.2 s after it was let go (the
  fingers must move on between beats, as players do); every other string must ring the full 0.5 s.

**Real trial:** record audio and video. For each strum, the four pitches start in order G, C, E, A within 0.25 s,
none missing or buzzing, each still ringing 0.5 s later (a tuner app reads C5 or A4 from the fretted string); the beat's
evenness comes from the strum onsets.

## Simulation model

- **The ukulele** is one fixed (kinematic) body: the body (its two bouts as cylinders, 60 mm deep), neck, headstock,
  fingerboard (5 mm rosewood, tapering 35 to 43 mm), 15 fret wires as 2 mm capsules (1.0 mm crowns) at their
  equal-tempered positions, bone nut and saddle, bridge, and the cradle's pads. Soundhole, rosette, position dots,
  tuners and the strings' tails past the nut and saddle are visual only.
- **Each string** is its own articulation, fixed to the world, between the nut and the saddle: a chain of 26 nodes,
  3-4 mm apart where a finger presses (between frets 1 and 2, and 2 and 3), one on each of the first five fret crowns, 10-25 mm
  apart elsewhere. Each node is a capsule of the real string's diameter covering its share of the string, on two
  prismatic joints (across the neck and up) *relative to the previous node*. A spring of stiffness tension / span on
  each joint, plus the last span's pull toward the saddle (tension / 31 mm on the last node's displacement, applied as
  a force after every physics step), make the discretised taut-string equations exactly: pushed at a point, the
  string bends into straight runs to the nut and the saddle with the right stiffness (0.38-0.45 N per mm over the
  soundhole), rests on fret crowns, and is swung and let go by the strumming fingertip. Tensions are those of
  fluorocarbon strings at pitch on a 346 mm scale (30-36 N).
- **Reading the strings.** The monitor reads every joint after every physics step. The load on each node is the
  string's equation of motion read backwards: stiffness times shape, plus damping times velocity, plus mass (with the
  joints' armature) times acceleration. What is left is the force whatever touches the string puts on it; a string
  ringing on its own reads a few hundredths of a newton. The last node, 31 mm from the saddle, carries the saddle-end spring and is heavy, so it is not watched for
  touches (a hand resting there would rest on the bridge too).
- **Simplified:**
  - The strings are much heavier than real (87 g each instead of about 0.3 g), with added joint inertia (armature
    0.01 kg) and damping (2 ms times each spring's stiffness), which the solver needs to stay stable at 240 Hz. A
    plucked string therefore swings at about 13 Hz and dies away over about half a second, instead of ringing at
    262-523 Hz for seconds.
    No sound is simulated: a clean note is judged from the string's motion and the loads on it.
  - Tension is constant: pressing does not sharpen the note, and a swing does not stretch the string.
  - The instrument is clamped in its cradle; a player holds it against the body.
  - The fret and finger contact are rigid (a real fingertip is soft and wraps the string).

## Real-world trials

- Use a soprano ukulele strung with fluorocarbon strings (GCEA, re-entrant G), tuned to pitch, in a padded guitar
  work cradle with a strap over the lower bout, and a foam neck rest under the headstock.
- Nothing is sharp or under power. The strings' tension (30-36 N) is held by the tuners; a string pulled hard can
  come out of its saddle slot, so the trial stops if a string is pulled more than 12 mm out of line.

## Runtime and seeds

The demo's trial takes 299 steps (14.95 s of simulated time). A full 1,200-step trial with Dream-1 also includes Dream-1's
round trips and the strings' per-physics-step monitor, plus about 1 minute for Isaac Sim to start. The layout is
fixed (`-E object_jitter=0`), so there are no seeds to choose.

## Regenerating the assets

```bash
pip install usd-core
python make_assets.py
```
