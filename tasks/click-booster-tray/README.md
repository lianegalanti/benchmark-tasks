# Click a booster seat's tray onto both armrests at its second notch

A toddler's booster seat (40 x 33 x 23 cm, 2.5 kg) stands on the table on its rubber feet, facing the robot, as it
would on a dining chair. Its feeding tray (42 x 25 cm, 0.45 kg) lies flat on the table in front of it. The robot is
asked to put the tray on at its middle setting, locked at the second notch. The setting fits the tray to the child:
at notch 2 the tray's back edge is 12 cm from the backrest (14.5 cm at notch 1, 9.5 cm at notch 3), and the tray
stands 12 cm above the seat.

The tray slides onto both armrests from the front. A sprung catch in each tray end rides over a rack of three notches
on the armrest's outer face (25 mm pitch). Each tooth has a ramp toward the front and a square face toward the back, so
the catches click over the teeth going on and hold against pulling off. A lip along the top of each rack closes the
notches from above, so a locked tray can't be lifted off either. The tray is held only where both catches have dropped
into a notch. Pushed too far, it squeezes the child; not far enough, the child can push it off.

What makes it hard is physical. Knowing which notch is wanted is easy; stopping in it is not:
- The tray has to be lifted to the armrests and kept square, so both rail walls (1.5 mm clearance) enter
  together. A skewed tray binds on one side.
- It has to be pushed so that both catches, 36 cm apart, ride up and over the first tooth and drop into the same
  notch. Clicking over a tooth takes about 12 N, and the push must then stop inside a 25 mm window, before the next
  tooth.
- The booster stands free on its feet, as it does before it is strapped to a chair; the 3 cm move limit stands for the
  straps. Shoving it more than 3 cm, or tipping it, fails the trial. Its feet hold about 20 N, so a careless push
  moves it.

Unlike sliding one carriage along a rail until it clicks, here two catches must reach the same notch, and the stop is
a 25 mm window, not an end stop. The task does not need two arms: one arm pushing the tray's centre could do it.

Parents and childcare workers do this at every meal on boosters and high chairs.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: the two parts, the per-step monitor, `success()`, `score()` and the scripted demo (`demo()`) |
| `geometry.py` | Dimensions of the seat, tray and ratchet, the scoring tests, and the demo's grip poses |
| `partkit.py` | Rigid parts built from boxes (optionally rotated), with spring-loaded sliders along any axis. Also adds per-part friction combining and robot link helpers |
| `test_task.py` | Offline checks: ratchet geometry, fit and clearances, every scoring level and violation, demo reach and timing |

`partkit.py` hooks `yam_arena.scene` and the embodiment's step and reset without modifying yam_arena. The task
therefore runs with the standard `yam_arena` embodiment and the `task_score` scorer.

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env` if needed):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the tray on the booster seat at its middle setting, locked at the second notch." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/booster-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the tray on the booster seat at its middle setting, locked at the second notch." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/booster-demo
```

The demo is open loop on the fixed layout and reads no simulator state. Both arms move together, mirrored:
1. Each arm grips the tray's front rim 19 cm from the centre, fingers closing along x across the 4 mm rim. The pad
   tips stay 7 mm above the plate.
2. The arms lift the tray 4.2 cm above its seated height. Held by its front rim, it pivots about 7 degrees in the
   grips, back edge down. They carry it until its back edge is 4 cm over the armrests' flat tops, then set it down
   back edge first, which levels it. The catches are still 1 cm short of the lip.
3. They slide it straight on at about 2 cm/s. Both catches ride over the first two teeth and drop into notch 2. The
   push ends 6 mm past notch 2's seat, 3.5 mm short of the third tooth's crest. The arms lag a few millimetres while
   the catches click over a tooth.
4. They ease the tray back 7 mm so both catches rest on the second tooth's square face, open, and lift and back away.

The nominal timeline is 24.3 s (486 steps), plus the 1 s success hold. Measured on an L40S, the demo ended with
success after 505 steps (25.3 s); Dream-1 scored 0 in a 1,200-step trial (the run limit).

## Scoring

The scorer runs after every control step and latches violations. The score at the end of the episode counts
(`SCORE_OVER = "end"`).

- **0** if at any time:
  - the booster is moved more than 3 cm, or tipped more than 10 degrees;
  - the tray falls: it drops faster than 0.6 m/s (about a 2 cm fall) while no gripper holds it (the let-go test
    below). Lowering it quickly in the grippers is allowed;
  - anything falls off the table.
- **Locked in a notch:** a catch has dropped in: its tip is within 4 mm of the notch floor, half the 8 mm notch depth
  (on the real seat, look from below). The tray also rests flat on the armrests: plate within 5 mm of their tops,
  tilted under 5 degrees, centred within 6 mm.
  The notch is told by where the catch is. With a ruler, the tray's front edge sticks out past the booster's front
  face by 10.5, 8 or 5.5 cm for notch 1, 2 or 3.
- **Let go:** both grippers' TCPs, pad tips and finger links are at least 2 cm from the tray.
- **Still:** the tray moves slower than 1 cm/s and both grippers slower than 2 cm/s.
- **Otherwise, at the end:**

| Score | Outcome |
|---|---|
| 1.0 | Success, held for 1 s: both catches locked in notch 2, let go, tray and arms still |
| 0.75 | Both catches locked in notch 2, but the tray is still held or moving |
| 0.5 | Both catches locked in notch 1, or both in notch 3 |
| 0.25 | Tray resting flat on both armrests, rails over them by at least 3 cm, let go, not locked (a catch riding on a tooth, say) |
| 0 | Anything else: doing nothing, refusing, a tray held above the armrests |

## Real-world build and safety

- Use a common moulded booster seat with a three-position tray, or 3D-print the racks (acetal or PETG) to the
  dimensions in `geometry.py` and screw them to its armrests. The catches are round-nosed spring slides (0.6 N/mm,
  10 mm travel).
- The booster stands on its non-slip feet with its straps undone, as before it is strapped to a chair, with no child
  and no food. Mark its feet on the table to measure the 3 cm move limit.
- Everything is light (2.5 kg seat, 0.45 kg tray) and below 25 cm. A dropped tray falls at most a few centimetres onto
  the table, and clicking the tray on takes about 12 N.

## Runtime and seeds

A full 1,200-step trial with Dream-1 takes about 2 minutes on an RTX GPU, including its round trips, plus about
1 minute for Isaac Sim to start. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.
