"""On the station table, ahead of the robot: a soprano ukulele (53 cm long, 346 mm scale) lying on its back in a padded
cradle, its neck to the robot's right (headstock toward the right arm) and its body to the left. Its four strings
(G, C, E, A, re-entrant tuning) run across the table: the G string is farthest from the robot, the A string nearest.
The fingerboard has 15 nickel-silver frets; the soundhole is under the strings at the body's upper bout, in front of
the left arm.

The robot must play four downstrokes on a C chord, then four on an A minor chord, at an even beat (60 to 100 beats
per minute). C: the A string held down just behind the 3rd fret, G, C and E open. A minor: the G string held down
just behind the 2nd fret, C, E and A open. Each downstroke sweeps G, C, E, A in order (toward the robot), each
string caught and let go so that it rings, none of them touched or muted while it rings.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py -E object_jitter=0 --scorer task_score ...
Demo:    python demo.py task.py demo_out
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scenekit import UsdBody, joint_state, on_physics_step, on_reset  # noqa: E402

import geometry as g  # noqa: E402

SCENE = "station"
SCORE_OVER = "end"          # the first eight strums are scored; the score at the end is theirs
SUCCESS_HOLD_S = 1.0

INSTRUCTION = ("Play the ukulele: strum four even downstrokes on a C chord (A string held at the third fret), then "
               "change to A minor (G string held at the second fret) and strum four more on the same beat.")

# -- objects -------------------------------------------------------------------------------------------------------
_U0 = tuple(float(v) for v in g.U0)
UKULELE = UsdBody("ukulele", usd="assets/ukulele.usda", pos=_U0, rot_xyzw=g.U_QUAT_XYZW, size=(0.17, 0.53, 0.10),
                  usd_materials=True,
                  contact_offset=0.002)


def _string_body(name: str) -> UsdBody:
    k = g.joint_stiffness(name)
    stiff, damp = {}, {}
    for i in range(g.N_NODES):
        for axis in ("lat", "vert"):
            stiff[f"{axis}_{i:02d}"] = float(k[i])
            damp[f"{axis}_{i:02d}"] = float(g.DAMP_TAU * k[i])
    return UsdBody(f"string_{name.lower()}", usd=f"assets/string_{name.lower()}.usda", pos=_U0, rot_xyzw=g.U_QUAT_XYZW,
                   size=(0.04, g.SCALE, 0.002), articulation=True, drives=((".*", stiff, damp),),
                   usd_materials=True, self_collisions=False, solver_iters=32, armature=g.ARMATURE,
                   contact_offset=0.002)


STRING_BODIES = {nm: _string_body(nm) for nm in g.NAMES}
OBJECTS = [UKULELE, *STRING_BODIES.values()]

# -- rules (numbers shared by the scorer, the tests and the README) --------------------------------------------------
F_FRET = 0.5            # N: a stopped string bears on its fret's crown with at least this (seated, no buzz)
F_MUTE = 0.15           # N: anything pressing a string (finger, fret, fingerboard) this hard mutes, buzzes or stops it
F_RING_PER_M = 35.0     # N/m: while a string rings, a touch is a load of at least max(F_MUTE, this x the pluck's peak
                        # swing): free ringing reads up to 23 N/m of swing in Isaac Sim (G, 6.3 mm: 0.14 N)
F_PUSH = 0.3            # N: a pluck: something pushes the string at least this hard at its peak
FRET_GRACE_S = 0.010    # s: a stop changes only when it stays changed this long (release transients)
A_START = 0.0005        # m: a string displaced this far from its line is being plucked
A_MIN = 0.001           # m: a pluck swings the string at least this far
RELEASE_FRAC = 0.3      # let go: the displacement falls to this fraction of its peak ...
RELEASE_S = 0.050       # ... within this time of the peak (a snap, not a slow lift-off)
SWING_FRAC = 0.3        # ringing: it then swings through its line to this fraction of the peak on the other side ...
RING_S = 0.5            # ... and rings this long with nothing touching it and its stop unchanged, or until the
RING_MIN_S = 0.4        # next strum catches it again (pushes it within CATCH_S of first touching it), no sooner
CATCH_S = 0.1           # than RING_MIN_S after it was let go, and the push must turn into a new pluck (let go)
PLUCK_ZONE = 0.025      # m along the string either side of the pluck's own contact
WINDOW_S = 0.25         # s: one strum: its plucks let go within this of the first (at least two strings)
LIFT_S = 0.2            # s: before a chord change, a fretted string may be let off its fret this long after it rang
N_STRUMS = len(g.SEQUENCE)
BEAT_RANGE = (0.6, 1.0)  # s between strums (100 to 60 beats per minute) ...
BEAT_TOL = 0.10         # ... each gap within 10% of their mean
OVERPULL = 0.012        # m: a string pulled this far out of line would jump its nut or saddle slot
CREDIT_SOUND, CREDIT_ORDER, CREDIT_BEAT = 0.6, 0.2, 0.2

# The nodes watched for touches: all but the last, 31 mm from the saddle, which carries the saddle-end spring and a
# large mass (anything resting there would rest on the bridge as well).
_WATCH = np.arange(g.N_NODES - 1)
_SPEAK = {n: (_WATCH if n <= 0 else _WATCH[_WATCH > g.I_FRETS[n]]) for n in (-1, 0, 1, 2, 3, 4, 5)}


def rest_line(w, n: int) -> np.ndarray:
    """The string's line when stopped at fret ``n``: straight from where it actually lies on that fret's crown (the
    finger may have pushed it along the crown) to the saddle (N x 2); open: its line from nut to saddle."""
    out = np.zeros((g.N_NODES, 2))
    if n > 0:
        i = g.I_FRETS[n]
        s, s0 = g.S_NODES, g.S_NODES[i]
        out[i:] = np.outer((g.SCALE - s[i:]) / (g.SCALE - s0), np.asarray(w, float)[i])
    return out


def stop_fret(F) -> int:
    """Where a string is stopped, from its loads: 0 open (no fret crown touched), n (fret n's crown bears at least
    F_FRET and no crown beyond it is touched), -1 (it touches a crown without bearing on it: a buzz)."""
    touching = [n for n in range(1, 6) if F[g.I_FRETS[n], 1] >= F_MUTE]
    if not touching:
        return 0
    n = max(touching)
    return n if F[g.I_FRETS[n], 1] >= F_FRET else -1


def muted(F, speak) -> bool:
    """Something presses the string's speaking length (at least F_MUTE) away from the pluck's own contact (the
    largest load, and the nodes within PLUCK_ZONE of it)."""
    mag = np.linalg.norm(F[speak], axis=1)
    if mag.size == 0 or mag.max() < F_MUTE:
        return False
    s = g.S_NODES[speak]
    s0 = s[int(np.argmax(mag))]
    return bool(np.any((mag >= F_MUTE) & (np.abs(s - s0) > PLUCK_ZONE)))


class Monitor:
    """The strings, watched after every physics step (240 Hz) from their joints alone, in the ukulele's frame.

    ``sample(t, w, wdot, wddot, closing)``: each string's node displacements from rest, rates and accelerations
    (N x 2: across the neck, up) and the saddle-end force applied on its last node this step. The loads on the nodes
    come from the string's equation of motion (``geometry.loads``): what touches it."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.t = 0.0
        self.violation = False
        self.why: list[str] = []
        self.stop = {nm: 0 for nm in g.NAMES}            # debounced stop fret of each string
        self._cand = {nm: None for nm in g.NAMES}
        self.events = {nm: None for nm in g.NAMES}       # a pluck being pushed or ringing
        self.plucks: list[dict] = []                     # finished plucks
        self.peaks = {nm: 0.0 for nm in g.NAMES}
        self.crowns = {nm: [0.0] * 5 for nm in g.NAMES}
        self.ring_load = {nm: 0.0 for nm in g.NAMES}      # largest load read on a ringing string (diagnostics)
        self._caught: dict = {}                          # rings cut by a push, waiting for that push to be a pluck

    def _flag(self, why: str) -> None:
        self.violation = True
        if why not in self.why:
            self.why.append(why)

    def sample(self, t: float, w: dict, wdot: dict | None = None, wddot: dict | None = None,
               closing: dict | None = None) -> None:
        self.t = t
        for nm in g.NAMES:
            wn = np.asarray(w[nm], float)
            F = g.loads(nm, wn, None if wdot is None else wdot[nm], None if wddot is None else wddot[nm],
                        None if closing is None else closing[nm])
            if float(np.abs(wn).max()) > OVERPULL:
                self._flag(f"the {nm} string was pulled more than {OVERPULL * 1000:.0f} mm out of line")
            self.crowns[nm] = [round(float(F[g.I_FRETS[n], 1]), 3) for n in range(1, 6)]
            raw = stop_fret(F)
            if raw == self.stop[nm]:
                self._cand[nm] = None
            elif self._cand[nm] is None or self._cand[nm][0] != raw:
                self._cand[nm] = (raw, t)
            elif t - self._cand[nm][1] >= FRET_GRACE_S - 1e-9:
                self.stop[nm], self._cand[nm] = raw, None
            self._track(nm, t, wn, F)

    def _track(self, nm: str, t: float, w, F) -> None:
        stop = self.stop[nm]
        speak = _SPEAK[stop]
        d = (w - rest_line(w, max(stop, 0)))[speak]
        mags = np.linalg.norm(d, axis=1)
        amp = float(mags.max())
        self.peaks[nm] = max(self.peaks[nm], amp)
        load = np.linalg.norm(F[speak], axis=1)
        ev = self.events[nm]
        if ev is None:
            if amp < A_START:
                return
            self.events[nm] = ev = {"string": nm, "t_start": t, "peak": 0.0, "phase": "push", "stop0": stop}
            prev = self._caught.pop(nm, None)
            if prev is not None and t - prev["caught_t"] <= CATCH_S + 1e-9:
                ev["prev"] = prev                         # the ring this pluck cut short (confirmed once let go)
        if ev["phase"] == "push":
            if stop != ev["stop0"]:                       # the fretting hand moved: a press or a lift, not a pluck
                self.events[nm] = None
                return
            ev["max_load"] = max(ev.get("max_load", 0.0), float(load.max()) if load.size else 0.0)
            if amp > ev["peak"]:
                i = int(np.argmax(mags))
                ev.update(peak=amp, t_peak=t, node=int(speak[i]), dir=(d[i] / max(amp, 1e-12)).tolist(),
                          pushed=bool(ev["max_load"] >= F_PUSH), muted=muted(F, speak), stop=stop,
                          loads=[round(float(v), 2) for v in np.linalg.norm(F, axis=1)])
            elif amp <= RELEASE_FRAC * ev["peak"]:
                if t - ev["t_peak"] <= RELEASE_S + 1e-9:
                    ev.update(phase="ring", t_release=t, swing=False, touched=False, restopped=False, ring_max=0.0)
                    prev = ev.pop("prev", None)
                    if prev is not None and ev["pushed"]:  # the next strum plucked it again: the earlier ring stands
                        prev.update(sounded=True, fails=[], pending=False)
                else:
                    self._finish(ev, "lifted off slowly")
            return
        # ringing: through its line to the other side, untouched, its stop unchanged
        proj = float((w - rest_line(w, max(ev["stop"], 0)))[ev["node"]] @ np.asarray(ev["dir"]))
        if proj <= -SWING_FRAC * ev["peak"]:
            ev["swing"] = True
        tau = t - ev["t_release"]
        top = float(load.max()) if load.size else 0.0
        f_touch = max(F_MUTE, F_RING_PER_M * ev["peak"])
        if "cut_t" in ev:                     # touched late: the next strum if it pushes, the fingers if they lift
            if stop != ev["stop"] and ev["stop"] > 0 and tau >= LIFT_S - 1e-9:
                ev["lifted"] = round(tau, 4)
                self._finish(ev, None)
                return
            if top >= F_PUSH and t - ev["t_release"] >= RING_MIN_S - 1e-9:
                ev["caught"] = round(tau, 4)          # pushed again: it stands only if this push is a new pluck
                self._finish(ev, None)
                if ev["sounded"]:
                    ev.update(sounded=False, fails=["cut short, not plucked again"], pending=True, caught_t=t)
                    self._caught[nm] = ev
                return
            if t - ev["cut_t"] > CATCH_S:
                ev["touched"] = True
                ev.setdefault("touch_t", round(ev["cut_t"] - ev["t_release"], 4))
        elif top >= f_touch and not ev["touched"]:
            if tau >= (min(RING_MIN_S, LIFT_S) if ev["stop"] > 0 else RING_MIN_S) - 1e-9:
                ev["cut_t"] = t
            else:
                ev["touched"] = True
                ev["touch_t"] = round(tau, 4)
                ev["touch_load"] = round(top, 3)
        if not ev["touched"] and "cut_t" not in ev:
            self.ring_load[nm] = max(self.ring_load[nm], top)
            ev["ring_max"] = max(ev["ring_max"], round(top, 4))
        if stop != ev["stop"]:
            if tau >= LIFT_S - 1e-9 and ev["stop"] > 0:   # the finger leaves for the next chord: the ring ends
                ev["lifted"] = round(tau, 4)
                self._finish(ev, None)
                return
            ev["restopped"] = True
        if t - ev["t_release"] >= RING_S - 1e-9:
            self._finish(ev, None)

    def _finish(self, ev: dict, why: str | None) -> None:
        if why is None:
            fails = [k for k, bad in (("not pushed", not ev["pushed"]), ("swing under 1 mm", ev["peak"] < A_MIN),
                                       ("muted at the peak", ev["muted"]), ("no swing through its line",
                                                                            not ev["swing"]),
                                       ("touched while ringing", ev["touched"]),
                                       ("stop changed while ringing", ev["restopped"]),
                                       ("buzzing on a fret", ev["stop"] < 0)) if bad]
        else:
            fails = [why]
        ev["sounded"] = not fails
        ev["fails"] = fails
        ev.pop("loads", None) if ev["sounded"] else None
        self.plucks.append(ev)
        self.events[ev["string"]] = None

    # -- strums and the score ---------------------------------------------------------------------------------------
    def strums(self) -> list[list[dict]]:
        """The plucks (finished, or still being pushed or ringing) grouped into strums: each strum's plucks let go
        (or, if never let go, peaked) within WINDOW_S of its first, at least two different strings. Single plucks
        are not strums. The first N_STRUMS strums, in order."""
        items = self.plucks + [e for e in self.events.values() if e is not None and "t_peak" in e]
        items = sorted(items, key=lambda e: e.get("t_release", e["t_peak"]))
        out, i = [], 0
        while i < len(items) and len(out) < N_STRUMS:
            t0 = items[i].get("t_release", items[i]["t_peak"])
            group = []
            while i < len(items) and items[i].get("t_release", items[i]["t_peak"]) <= t0 + WINDOW_S + 1e-9:
                group.append(items[i])
                i += 1
            if len({e["string"] for e in group}) >= 2:
                out.append(group)
        return out

    def report(self) -> dict:
        strums = self.strums()
        rows, onsets = [], []
        for k, group in enumerate(strums):
            chord = g.SEQUENCE[k]
            first = {}
            for e in group:
                first.setdefault(e["string"], e)
            changing = k + 1 < N_STRUMS and g.SEQUENCE[k + 1] != chord
            good = [nm for nm, e in first.items()
                    if e.get("sounded") and e["stop"] == g.CHORDS[chord][nm] and ("lifted" not in e or changing)]
            # the strum counts only if its fretted note sounds at the right fret (A on the 3rd for C, G on the 2nd
            # for Am): an open-string strum is not the chord
            key = next(nm for nm, f in g.CHORDS[chord].items() if f > 0)
            valid = key in good
            order = valid and len(first) == 4 and all(first[nm].get("sounded") for nm in g.NAMES) and all(
                first[a]["t_release"] < first[b]["t_release"] for a, b in zip(g.NAMES, g.NAMES[1:]))
            onset = min(e.get("t_release", e["t_peak"]) for e in group)
            onsets.append(onset)
            rows.append({"chord": chord, "onset": round(onset, 4), "fretted": bool(valid),
                         "sounding": "".join(sorted(good, key=g.NAMES.index)) if valid else "",
                         "order": bool(order)})
        gaps = np.diff(onsets) if len(onsets) > 1 else np.array([])
        beat_ok = 0
        if gaps.size:
            m = float(gaps.mean())
            if BEAT_RANGE[0] <= m <= BEAT_RANGE[1]:
                ok = np.abs(gaps - m) <= BEAT_TOL * m + 1e-9
                both = np.array([rows[k]["fretted"] and rows[k + 1]["fretted"] for k in range(len(gaps))])
                beat_ok = int(np.sum(ok & both))
        sound = sum(len(r["sounding"]) / 4 for r in rows) / N_STRUMS
        order = sum(r["order"] for r in rows) / N_STRUMS
        beat = beat_ok / (N_STRUMS - 1)
        level = CREDIT_SOUND * sound + CREDIT_ORDER * order + CREDIT_BEAT * beat
        return {"strums": rows, "gaps": [round(float(x), 4) for x in gaps], "sound": round(sound, 4),
                "order": round(order, 4), "beat": round(beat, 4), "level": round(level, 6)}

    def score(self) -> float:
        return 0.0 if self.violation else float(self.report()["level"])


MON = Monitor()
_STATE: dict = {"armed": False, "ids": {}}


def reset_state() -> None:
    MON.reset()


def _arrays(environment, nm: str):
    names, q, v = joint_state(environment, STRING_BODIES[nm].name)
    ids = _STATE["ids"].get(nm)
    if ids is None:
        lat = [names.index(f"lat_{i:02d}") for i in range(g.N_NODES)]
        vert = [names.index(f"vert_{i:02d}") for i in range(g.N_NODES)]
        _STATE["ids"][nm] = ids = (np.array(lat), np.array(vert))
    lat, vert = ids
    w = g.absolute(np.column_stack([q[lat], q[vert]]))
    wd = g.absolute(np.column_stack([v[lat], v[vert]]))
    return w, wd


@on_reset
def _reset(environment=None) -> None:
    reset_state()
    _STATE.pop("wd", None)
    _STATE["armed"] = True


def _close_strings(environment, w: dict, wd: dict) -> None:
    """Each string's last span: the pull toward the saddle on its last node, set for the next physics step."""
    import torch

    base = environment.unwrapped
    if "close" not in _STATE:
        _STATE["close"] = {}
        for nm in g.NAMES:
            asset = base.scene[STRING_BODIES[nm].name]
            _STATE["close"][nm] = (asset, [asset.body_names.index(f"node_{g.N_NODES - 1:02d}")])
        _STATE["env_ids"] = torch.arange(base.num_envs, device=base.device)
    for nm in g.NAMES:
        asset, body = _STATE["close"][nm]
        f_u = g.closing_force(nm, w[nm][-1], wd[nm][-1])          # (x, z) in U
        _STATE.setdefault("fc", {})[nm] = f_u
        f = g.ROT_U @ np.array([f_u[0], 0.0, f_u[1]])
        forces = torch.tensor(f, dtype=torch.float32, device=base.device).reshape(1, 1, 3).repeat(base.num_envs, 1, 1)
        asset.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces, torques=torch.zeros_like(forces), body_ids=body, env_ids=_STATE["env_ids"], is_global=True)


@on_physics_step
def _sample(environment, dt: float) -> None:
    w, wd, wdd = {}, {}, {}
    prev = _STATE.get("wd")
    applied = _STATE.get("fc")                        # the saddle-end forces this step ran with
    for nm in g.NAMES:
        w[nm], wd[nm] = _arrays(environment, nm)
        wdd[nm] = np.zeros_like(wd[nm]) if prev is None else (wd[nm] - prev[nm]) / float(dt)
    _STATE["wd"] = wd
    _close_strings(environment, w, wd)
    if _STATE["armed"]:
        MON.sample(MON.t + float(dt), w, wd, wdd, applied)


def violation() -> bool:
    return bool(MON.violation)


def success(objects) -> bool:
    """All eight strums clean: every string sounding (pushed, let go, through its line, ringing 0.5 s untouched) at
    the chord's pitch (C: A at the 3rd fret; Am: G at the 2nd fret; the rest open), in order G, C, E, A within
    0.25 s, at an even beat."""
    return bool(MON.score() >= 1.0 - 1e-9)


def score(objects) -> float:
    """0 if a string was ever pulled more than 12 mm out of line. Otherwise, over the first eight strums (four on C,
    then four on Am): 0.6 x the share of strings sounding at the chord's pitch, 0.2 x the share of strums in order
    G, C, E, A, 0.2 x the share of the seven gaps between strums within 10% of their mean (60 to 100 bpm)."""
    return MON.score()
