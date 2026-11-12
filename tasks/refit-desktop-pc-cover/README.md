# Refit a desktop PC's cover: hooks into slots, slide until the latch clicks

A small-form-factor office desktop lies flat on the table with its cover off, its motherboard, CPU cooler, memory,
power supply and drive open to view. Its cover lies on the table beside it. The robot is asked to put the cover back
on so that it latches, and to let go of it.

The case and cover have the real catalogue size of the common small-form-factor desktop (about 29 x 29 x 9.4 cm,
4.6 kg; the steel cover about 0.6 kg), and the cover goes on as a tool-less cover does:

1. Lowered flat over the case, its side lips (1.5 mm outside the case's walls, their feet flared out 3 mm as a lead-in)
   steer it across, and its six hooks (three under each side edge of the plate, hidden from above) drop through six
   slots in the case's folded-in rim. The slots leave 11 mm of play front to back and 2.5 mm either side across; a
   hook that misses its slot holds the cover 4 mm proud of the rim.
2. Slid toward the case's front, the hooks' toes run under the flange in front of their slots, so the cover can no
   longer be lifted. The stems stop it at 16 mm.
3. From 13 mm of slide, a 12 mm hole in the cover is over a spring plunger in the left flange, which the cover has
   held down until then. It pops up into the hole and locks the cover between 13 and 16 mm.

Closing a desktop back up after a memory or drive upgrade or a dust cleaning is a common home and office chore. A
cover left unlatched or hooked on crooked rattles, leaves sharp edges and live parts exposed, and spoils the airflow
the case is designed for. The robot must seat six hooks it cannot see at once, then slide a wide, thin 0.6 kg panel
13 to 16 mm without jamming it or knocking the 4.6 kg case, and keep its grippers out of the open case throughout.
The cover is handled by its two small pull tabs at the rear edge, as a person slides it by its rear edge.

A full-size mid-tower side panel (about 45 x 43 cm, 1.5 to 2.5 kg) does not fit these arms: its far edge is beyond
their reach, and it has nowhere to lie within reach beside the case. The small-form-factor desktop is a real,
widespread office PC, used here at its real size rather than a scaled-down tower.

## Files

| File | What |
|---|---|
| `task.py` | The yam_arena task file: objects, the per-step monitor, `success()`, `score()` and the scripted `demo()` |
| `geometry.py` | Dimensions of the case, cover, hooks, slots and latch, the layout, and the seated/engaged/latched tests |
| `partkit.py` | Rigid parts built from boxes, with spring-loaded sliders (the latch), for yam_arena task files |
| `test_task.py` | Offline checks: geometry, every scoring level and violation, demo reachability and a monitor replay |

`partkit.py` is copied unchanged from the organizers' `din-rail-mount` example task (its license applies). It hooks
`yam_arena.scene` and the embodiment's step and reset without modifying yam_arena, so the task runs with the standard
`yam_arena` embodiment and `task_score` scorer.

## Run a trial

From this folder, in a shell with the yam_arena environment active (`source <ARENA_DIR>/yam.env`):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the cover back on the computer so that it latches, then let go of it." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pc-cover-dream
```

`task_score` prints the trial's score: 1.0 on success, otherwise `score()` at the end of the episode.

## Scripted demo

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the cover back on the computer so that it latches, then let go of it." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/pc-cover-demo
```

The demo is open loop from the fixed starting layout; it reads no simulator poses.

1. Each arm pinches one of the cover's rear pull tabs (12 cm either side of its middle), fingers closing front to
   back.
2. They lift it until its underside is 3.5 cm above the rim and carry it sideways over the case, aiming 7 mm from
   the rear end of the hooks' 11 mm drop window (the loaded arms trail their targets by up to 5 mm).
3. They lower it flat onto the rim, commanded 1.5 mm lower so the arms hold it flat: the hooks drop through the
   slots. (Held only lightly, the latch plunger tilts the cover up and the hooks' toes jam under the flange.)
4. Still pinching the tabs, they push the cover toward the front until the hooks' stems stop it (16 mm): the toes
   run under the flange. They are commanded 4.5 mm past the stop, because the pushing arms trail their targets
   by about 5 mm.
5. They let go, lift away and back off toward the robot. Freed of the fingers' push, the cover settles back about
   1.5 mm and the latch pops up into its hole.

In Isaac Sim the loaded arms trail their targets by about 5 mm while carrying, so the cover lands with its hooks
just behind the slots; they drop in during the first millimetres of the push.

Each tab is 16 mm deep, so each closed finger stays 8 mm short of its target and presses with 40 N, below the
gripper's 50 N limit: the tab stays centred between the fingers when pushed. (On a thicker part both fingers press
at the limit, and a pushed part lags them by several millimetres.)

Measured in Isaac Sim on an L40S: the demo reaches 1.0 (success) in 426 steps (21 s). With 1.5 mm of slot side play
instead of 2.5 mm it stops short of the latch (0.5): the pushing arms lag too far behind their targets. Dream-1 scored 0 in a 2,400-step trial (the run limit).

## Scoring

The monitor runs after every control step and latches violations; the score at the end of the episode counts
(`SCORE_OVER = "end"`).

- **0** if at any time:
  - the cover is dropped: it falls faster than 0.55 m/s (a free fall of about 1.5 cm);
  - the case is moved more than 1 cm or turned more than 2 degrees;
  - any part of a gripper goes into the open case below the rim (where it could touch the board);
  - the cover tips into the case (a point of its plate more than 1 cm below the rim, inside the walls);
  - the case or the cover falls off the table.
- **Seated:** every corner of the plate's underside is within -1 to +2 mm of the rim's top, the cover is square to
  the case within 2 degrees, centred across it within 2 mm, and slid -6 to +18 mm (as defined next). Resting on a
  hook instead of in its slot holds it 4 mm up, so seated means all six hooks are down through their slots.
- **Slide** is measured from where the hooks drop straight in (the cover 15 mm short of flush with the case's rear).
- **Otherwise**, at the end:

| Score | When |
|---|---|
| 1.0 | Latched: seated, slid 13 to 16 mm, the latch button up at least 1.5 mm into the cover's hole; no gripper within 3 cm of the cover; the cover still (under 1 cm/s); held for 1 s (success) |
| 0.75 | Latched, but a gripper is within 3 cm of the cover or it is moving |
| 0.5 | Seated and slid at least 9 mm (every toe at least 2 mm under the flange), latch not up |
| 0.25 | Seated with the hooks in their slots, slid less than 9 mm |
| 0 | Anything else, including doing nothing or refusing |

How the simulation judges these, as from a video, once per control step: the case and cover poses and the latch
joint come from the simulator; each gripper is sampled at its TCP, a wrist point 10 cm up its axis, and both
fingers' pad and outer faces from the pad tips 4 cm up, from the measured joint angles.

Cheats considered: setting the cover down already slid leaves the hooks on the flange, 4 mm proud (0); pressing the
plunger is not needed and lifting the latched cover is blocked by the hooks; holding the cover on caps at 0.75;
pushing the case to line it up, dropping the cover and retrying, or reaching into the case to steady it are
violations for the whole trial.

## Real-world trials

Use a real small-form-factor desktop, unplugged and cold; its hemmed cover edges and the 9 cm working height make a
slip harmless, and a cover dropped from carrying height lands on the table or the case. A person beside the table
can stop the arms at any time.

## Runtime and seeds

A full 2,400-step trial with Dream-1 takes about 4 minutes on an RTX GPU, including its round trips, plus about 1
minute for Isaac Sim to start. The scripted demo takes 426 steps. The layout is fixed (`-E object_jitter=0`), so there are no seeds to choose.

Offline checks: `PYTHONPATH=<yam-isaac-lab-arena> python -m pytest -q test_task.py`.
