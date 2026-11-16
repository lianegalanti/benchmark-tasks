"""Offline checks (no Isaac Sim): the instrument's size and layout, the string model (stiffness, saddle-end spring,
loads read from the motion), the assets, fretting mechanics and pitch, every scoring rule (sounding, ringing, pitch,
order, strum window, beat, first attempts only), and the demo (reach, joint limits, fingertip clearances from the
arm's own collision meshes, the beat, no privileged state).

The scoring tests drive the monitor with stringsim.py: the same string equations the sim builds, integrated offline
at 240 Hz, with fingers, fret crowns and the strum as held nodes.

    python -m pytest -q test_task.py      # in Arena's venv, from this folder
"""

import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from yam_arena import ik
from yam_arena.task_file import load

HERE = Path(__file__).resolve().parent
TASK = load(HERE / "task.py")
M = TASK.module
import demo  # noqa: E402  (task.py put this folder on the path)
import geometry as g  # noqa: E402
import tipkit  # noqa: E402
from stringsim import DT, StringSim  # noqa: E402

PLAN = demo.plan()
STEPS = PLAN[:-1]
N = g.N_NODES
TIPS = tipkit.tip_points(1.0)
I_STRUM = int(np.argmin(np.abs(g.S_NODES - demo.STRUM_S)))


@pytest.fixture(autouse=True)
def fresh():
    M.reset_state()


def finger_z(name: str, depth: float, nodes) -> dict:
    """A finger holding ``nodes`` with the string's underside ``depth`` below the fret crowns' plane."""
    return {i: -(g.gap_over_crown(name, g.S_NODES[i]) + depth) for i in nodes}


def fret_hold(name: str, fret: int, finger_nodes, depth: float = 0.0006) -> dict:
    """Held nodes {i: (x, z)} of a string pressed behind ``fret``: the finger's nodes and the crown it bears on."""
    held = {i: (0.0, z) for i, z in finger_z(name, depth, finger_nodes).items()}
    held[g.I_FRETS[fret]] = (0.0, -g.gap_over_crown(name, g.fret_s(fret)))
    return held


C_SHAPE = {"A": fret_hold("A", 3, (8, 9))}          # finger 5-8 mm behind the 3rd fret wire
AM_SHAPE = {"G": fret_hold("G", 2, (3, 4))}         # finger 4-8 mm behind the 2nd fret wire


class Trial:
    """The four strings (stringsim) and the monitor, stepped at 240 Hz. ``shape``: the fretting hand's held nodes
    per string; ``extra``: more held nodes (a touch)."""

    def __init__(self):
        self.t = 0.0
        self.sims = {nm: StringSim(nm) for nm in g.NAMES}
        self.shape: dict = {}
        self.extra: dict = {}

    def step(self, pluck: dict | None = None):
        for nm, sim in self.sims.items():
            held = dict(self.shape.get(nm, {}))
            held.update(self.extra.get(nm, {}))
            if pluck and nm in pluck:
                held.update(pluck[nm])
            sim.step(held)
        self.t += DT
        M.MON.sample(self.t, {nm: s.w for nm, s in self.sims.items()}, {nm: s.v for nm, s in self.sims.items()},
                     {nm: s.a for nm, s in self.sims.items()})

    def hold(self, seconds: float):
        for _ in range(round(seconds / DT)):
            self.step()

    def pluck(self, nm: str, amp: float = 0.0025, push_s: float = 0.02, node: int = I_STRUM, release_s: float = 0.0):
        """Push string ``nm`` across at ``node`` to ``amp`` over ``push_s``, then let go (at once, or easing back
        over ``release_s``: a slow lift-off)."""
        n = max(1, round(push_s / DT))
        for k in range(1, n + 1):
            self.step({nm: {node: (amp * k / n, self.sims[nm].w[node, 1])}})
        m = round(release_s / DT)
        for k in range(1, m + 1):
            self.step({nm: {node: (amp * (1 - k / m), self.sims[nm].w[node, 1])}})

    def strum(self, order=g.NAMES, spacing: float = 0.04, **kw):
        """One downstroke: each string plucked in turn, ``spacing`` s apart (from let-go to let-go)."""
        for nm in order:
            t0 = self.t
            self.pluck(nm, **kw)
            self.hold(max(0.0, spacing - (self.t - t0)))

    def perform(self, gaps=None, chords=g.SEQUENCE, change=True, order=g.NAMES, shapes=True, **kw):
        """The piece: a downstroke per beat (``gaps``: s between downstroke starts, default 1.0); ``shapes=False``:
        no fretting at all."""
        gaps = gaps or [1.0] * (len(chords) - 1)
        change = change and shapes
        self.shape = dict(C_SHAPE if chords[0] == "C" else AM_SHAPE) if shapes else {}
        self.hold(0.5)
        for k, chord in enumerate(chords):
            t0 = self.t
            self.strum(order=order, **kw)
            if k < len(chords) - 1:
                if change and chords[k + 1] != chord:     # the chord change, 0.3 s before the next downstroke
                    self.hold(gaps[k] - 0.3 - (self.t - t0))
                    self.shape = dict(C_SHAPE if chords[k + 1] == "C" else AM_SHAPE)
                self.hold(gaps[k] - (self.t - t0))
        self.hold(0.7)


def score():
    return M.score(None)


# -- layout, size and assets --------------------------------------------------------------------------------------


def test_it_loads_with_a_fixed_ukulele_and_four_string_articulations():
    assert TASK.names == ("ukulele", "string_g", "string_c", "string_e", "string_a")
    assert TASK.score_over == "end"
    assert M.UKULELE.mass is None and not M.UKULELE.articulation
    for b in M.STRING_BODIES.values():
        assert b.articulation and not b.self_collisions and b.armature == g.ARMATURE
        stiff = b.drives[0][1]
        assert len(stiff) == 2 * N


def test_full_size_soprano():
    assert g.SCALE == pytest.approx(0.346)                                   # 13.6 in scale
    total = g.BODY_END - g.HEAD[0]
    assert 0.52 <= total <= 0.55                                             # 21 in overall
    assert g.BODY_END - g.BODY_JOIN == pytest.approx(0.250, abs=0.002)        # body length
    assert 2 * g.LOWER_BOUT[1] == pytest.approx(0.162) and 2 * g.UPPER_BOUT[1] == pytest.approx(0.135)
    assert g.FB_W[0] == pytest.approx(0.035)                                 # nut width
    assert g.fret_s(12) == pytest.approx(g.SCALE / 2)
    assert g.fret_s(3) == pytest.approx(0.05505, abs=1e-4)
    # string spacing: about 9 mm at the nut, 14 mm at the saddle, 9.8 mm between A and E at the 3rd fret
    s3 = g.fret_s(3)
    assert float(g.string_x("A", s3) - g.string_x("E", s3)) == pytest.approx(0.0098, abs=0.0002)
    # action: 0.5 mm over the 1st fret, about 1 mm at the 3rd, 2.5 mm at the 12th
    assert g.gap_over_crown("A", g.fret_s(1)) == pytest.approx(0.0005, abs=0.00002)
    assert g.gap_over_crown("A", g.fret_s(12)) == pytest.approx(0.0025, abs=0.00002)
    for nm, d, t, *_ in g.STRINGS:
        assert 0.0006 <= d <= 0.00085 and 28 <= t <= 38


def test_layout_in_reach_and_on_the_table():
    # the fretting spot and the strumming spot in front of the arms, the instrument above the table top
    fret = g.u_to_station([g.string_x("A", 0.048), -0.048, 0.0])
    strum = g.u_to_station([0.0, -demo.STRUM_S, 0.0])
    assert fret[1] < 0 < strum[1]
    assert 0.3 <= fret[0] <= 0.4 and 0.3 <= strum[0] <= 0.4
    zt = -(g.U0[2] - g.TABLE)
    for _, c, size in g.cradle_parts():
        assert c[2] - size[2] / 2 == pytest.approx(zt)
    assert g.TOP_Z - g.BODY_DEPTH == pytest.approx(zt + g.CRADLE_Z)


def test_assets_match_the_geometry():
    from pxr import Usd, UsdPhysics

    for nm in g.NAMES:
        stage = Usd.Stage.Open(str(HERE / f"assets/string_{nm.lower()}.usda"))
        root = f"/string_{nm.lower()}"
        k = g.joint_stiffness(nm)
        for i in range(N):
            for axis, ax in (("lat", "X"), ("vert", "Z")):
                prim = stage.GetPrimAtPath(f"{root}/Physics/{axis}_{i:02d}")
                j = UsdPhysics.PrismaticJoint(prim)
                assert j.GetAxisAttr().Get() == ax
                drive = UsdPhysics.DriveAPI.Get(prim, "linear")
                assert drive.GetStiffnessAttr().Get() == pytest.approx(k[i], rel=1e-5)
            node = stage.GetPrimAtPath(f"{root}/Geometry/node_{i:02d}")
            assert node.GetAttribute("physxRigidBody:disableGravity").Get() is True
            cap = stage.GetPrimAtPath(f"{root}/Geometry/node_{i:02d}/cap")
            assert cap.GetAttribute("radius").Get() == pytest.approx(g.string_spec(nm)[1] / 2)
        last = stage.GetPrimAtPath(f"{root}/Geometry/node_{N - 1:02d}")
        assert last.GetAttribute("physics:mass").Get() == pytest.approx(g.LAST_MASS)
    stage = Usd.Stage.Open(str(HERE / "assets/ukulele.usda"))
    assert stage.GetPrimAtPath("/ukulele").GetAttribute("physics:kinematicEnabled").Get() is True
    for n in range(1, g.N_FRETS + 1):
        assert stage.GetPrimAtPath(f"/ukulele/fret{n}").IsValid()


# -- the string model -------------------------------------------------------------------------------------------------


def test_relative_joints_and_closing_spring_make_the_taut_string():
    """Joint springs T/h on each relative joint plus the closing spring T/h_last on their sum (the last node's
    displacement) store exactly the discretised string's energy, so the loads read from the shape are the forces on
    the nodes."""
    rng = np.random.default_rng(0)
    for nm in g.NAMES:
        q = rng.normal(0, 0.001, N)
        w = g.absolute(q)
        e_sim = 0.5 * np.sum(g.joint_stiffness(nm) * q ** 2) + 0.5 * g.tendon_stiffness(nm) * w[-1] ** 2
        e_string = 0.5 * w @ g.stiffness_matrix(nm) @ w
        assert e_sim == pytest.approx(e_string, rel=1e-9)


def test_closing_force_is_the_last_spans_pull_and_stable():
    for nm in g.NAMES:
        f = g.closing_force(nm, np.array([0.001, -0.002]), np.zeros(2))
        assert f == pytest.approx(-g.string_spec(nm)[2] / g.SPANS[-1] * np.array([0.001, -0.002]))
        # applied once per physics step (explicit): omega * dt well under 2 on the heavy last node
        assert math.sqrt(g.tendon_stiffness(nm) / g.LAST_MASS) / 240 < 1.0
        assert g.TENDON_DAMP / 240 / g.LAST_MASS < 0.5


def test_point_stiffness_is_a_real_strings():
    """A string under tension T pushed at a point a from one end deflects F (a (L - a)) / (T L)."""
    for nm in g.NAMES:
        t = g.string_spec(nm)[2]
        a = g.S_NODES[I_STRUM]
        w = g.solve_shape(nm, {I_STRUM: 0.002})
        f = g.loads(nm, w)[I_STRUM]
        assert f == pytest.approx(0.002 * t * g.SCALE / (a * (g.SCALE - a)), rel=1e-6)
        assert np.abs(np.delete(g.loads(nm, w), I_STRUM)).max() < 1e-9   # only the pushed node carries a load
    assert 0.8 < g.loads("A", g.solve_shape("A", {I_STRUM: 0.002}))[I_STRUM] < 1.0   # 0.9 N for 2 mm


def test_a_free_ringing_string_reads_no_touch():
    """The loads are the string's equation of motion read backwards (stiffness, damping, inertia): a string ringing
    on its own reads well under the 0.15 N that counts as a touch."""
    tr = Trial()
    tr.pluck("C", amp=0.004)
    loads = []
    for _ in range(round(0.5 / DT)):
        tr.step()
        s = tr.sims["C"]
        loads.append(np.linalg.norm(g.loads("C", s.w, s.v, s.a)[:-1], axis=1).max())
    tr.hold(0.1)
    assert max(loads[6:]) < 0.3 * M.F_MUTE
    assert M.MON.plucks and M.MON.plucks[-1]["sounded"]


def test_pitch_of_each_stopped_string():
    assert g.speaking_length(3) == pytest.approx(0.29095, abs=1e-4)          # A on the 3rd fret crown: 291 mm
    assert g.pitch_hz("A", 3) == pytest.approx(523.25, abs=0.01)             # C5
    assert g.pitch_hz("G", 2) == pytest.approx(440.0, abs=0.01)              # A4: the Am chord's G string
    assert g.speaking_length(0) == pytest.approx(g.SCALE)
    assert {nm: g.CHORDS["C"][nm] for nm in g.NAMES} == {"G": 0, "C": 0, "E": 0, "A": 3}
    assert {nm: g.CHORDS["Am"][nm] for nm in g.NAMES} == {"G": 2, "C": 0, "E": 0, "A": 0}


def test_the_fret_needs_the_string_pressed_below_the_crown_close_behind_it():
    """Fretting craft: close behind the fret wire a small push seats the string; far behind it it takes a deeper
    press."""
    i3 = g.I_FRETS[3]

    def crown(fixed):
        held = {i3: -g.gap_over_crown("A", g.fret_s(3))}
        held.update(fixed)
        return g.loads("A", g.solve_shape("A", held))[i3]

    assert crown(finger_z("A", 0.0002, (9,))) >= M.F_FRET > crown(finger_z("A", 0.0002, (6,)))
    assert crown(finger_z("A", 0.0006, (6,))) >= M.F_FRET                   # deeper there, it seats too


def _stop(name, held):
    w = np.zeros((N, 2))
    w[:, 1] = g.solve_shape(name, {i: z for i, (_, z) in held.items()})
    return M.stop_fret(g.loads(name, w))


def test_stop_fret_reads_the_fret_the_wire_and_the_wrong_fret():
    assert _stop("A", C_SHAPE["A"]) == 3
    assert _stop("G", AM_SHAPE["G"]) == 2
    assert _stop("E", {}) == 0
    # pressed on the 3rd fret wire itself: the crown carries the finger, not a seated string
    on_wire = {g.I_FRETS[3]: (0.0, -g.gap_over_crown("A", g.fret_s(3)) - 0.0003)}
    assert _stop("A", on_wire) != 3
    # pressed between the 3rd and 4th frets: the 4th fret's crown stops it (C#5, the wrong note)
    past = {11: (0.0, -(g.gap_over_crown("A", g.S_NODES[11]) + 0.0006)),
            g.I_FRETS[3]: (0.0, -g.gap_over_crown("A", g.fret_s(3)) - 0.0002),
            g.I_FRETS[4]: (0.0, -g.gap_over_crown("A", g.fret_s(4)))}
    assert _stop("A", past) == 4


# -- scoring --------------------------------------------------------------------------------------------------------


def test_doing_nothing_scores_0():
    Trial().hold(3.0)
    assert score() == 0.0 and not TASK.success(None)


def test_the_clean_piece_scores_1():
    tr = Trial()
    tr.perform()
    rep = M.MON.report()
    assert [r["chord"] for r in rep["strums"]] == list(g.SEQUENCE)
    assert all(r["sounding"] == "GCEA" and r["order"] for r in rep["strums"]), rep
    assert score() == pytest.approx(1.0) and TASK.success(None)


def test_c_without_the_fret_and_no_chord_change():
    tr = Trial()
    tr.perform(change=False)                              # C shape held throughout: the Am strums have A at 3
    rep = M.MON.report()
    assert [r["sounding"] for r in rep["strums"]] == ["GCEA"] * 4 + [""] * 4   # no G on the 2nd fret: no Am
    assert score() == pytest.approx(0.6 * 4 / 8 + 0.2 * 4 / 8 + 0.2 * 3 / 7)


def test_open_strings_only_score_nothing():
    """Strumming the open strings in time, without fretting, is not the piece: no strum has its fretted note."""
    tr = Trial()
    tr.perform(shapes=False)
    rep = M.MON.report()
    assert len(rep["strums"]) == 8 and not any(r["fretted"] for r in rep["strums"])
    assert score() == 0.0


def test_upstrokes_lose_the_order_credit():
    Trial().perform(order=tuple(reversed(g.NAMES)))
    assert score() == pytest.approx(0.8)


def test_an_uneven_beat_loses_the_gaps_out_of_time():
    Trial().perform(gaps=[0.9, 0.9, 0.9, 1.2, 0.9, 0.9, 0.9])
    assert score() == pytest.approx(1.0 - 0.2 / 7)


def test_too_slow_a_tempo_loses_the_beat_credit():
    Trial().perform(gaps=[1.2] * 7)                       # 50 beats per minute
    assert score() == pytest.approx(0.8)


def test_a_strum_spread_over_more_than_the_window_is_not_one_strum():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum(spacing=0.15)                                # G..A over 0.45 s: G, C one strum; E, A another
    tr.hold(0.7)
    rep = M.MON.report()
    assert [r["sounding"] for r in rep["strums"]] == ["", "EA"]     # the first has no A: not a C chord


def test_a_single_pluck_is_not_a_strum():
    tr = Trial()
    tr.pluck("E")
    tr.hold(0.7)
    assert M.MON.report()["strums"] == []


def test_only_the_first_eight_strums_count():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum(order=("G", "C"))                            # a fumbled first try
    tr.hold(0.9)
    tr.perform()
    rep = M.MON.report()
    assert len(rep["strums"]) == 8 and rep["strums"][0]["sounding"] == ""
    assert score() < 0.95


def test_a_string_touched_while_ringing_does_not_sound():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum()
    tr.extra["E"] = {12: (0.0, 0.0)}                      # a finger lands on the E string 0.1 s later
    tr.hold(0.6)
    rep = M.MON.report()
    assert rep["strums"][0]["sounding"] == "GCA"


def test_the_next_strum_may_cut_the_ring_after_0_4_s():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum()
    tr.hold(0.45)                                         # 100 bpm: the next downstroke 0.6 s after the first
    tr.strum()
    tr.hold(0.7)
    rep = M.MON.report()
    assert [r["sounding"] for r in rep["strums"]] == ["GCEA", "GCEA"]


def test_a_buzz_on_a_fret_while_ringing_does_not_sound():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum()
    # the open C string slaps the 12th fret's crown region: a light contact under it (0.2 N)
    i = 15
    k = g.stiffness_matrix("C")
    tr.extra["C"] = {i: (0.0, -0.0004)}
    tr.hold(0.6)
    assert M.MON.report()["strums"][0]["sounding"] == "GEA"


def test_a_slow_lift_off_is_not_a_pluck():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum(release_s=0.15, spacing=0.0)
    tr.hold(0.7)
    assert all(r["sounding"] == "" for r in M.MON.report()["strums"])


def test_a_swing_nothing_pushes_is_not_a_pluck():
    """A string set swinging by the shape of its own mode, with nothing pushing it, is not plucked."""
    tr = Trial()
    mode = np.sin(np.pi * g.S_NODES / g.SCALE)
    s = tr.sims["C"]
    s.w[:, 0] = 0.002 * mode
    tr.hold(0.7)
    assert all(not p["sounded"] for p in M.MON.plucks)


def test_the_finger_may_leave_for_the_chord_change_after_0_2_s():
    """Before a chord change the fretted string may be let off its fret 0.2 s after it rang (and only then)."""
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    for k in range(4):
        tr.strum()
        tr.hold(0.36)
        if k == 3:
            tr.shape = {}                                 # the A string rang 0.36 s: the finger leaves it
            tr.hold(0.3)
            tr.shape = dict(AM_SHAPE)
        tr.hold(0.6 - (0.3 if k == 3 else 0.0))
    for k in range(4):
        tr.strum()
        tr.hold(0.96)
    rep = M.MON.report()
    assert [r["sounding"] for r in rep["strums"]] == ["GCEA"] * 8


def test_a_stopped_string_pushed_along_the_crown_is_still_at_rest():
    """The finger may drag the stopped string a little along the fret crown: its line then runs from where it lies
    on the crown, and the next strum still plucks it cleanly."""
    tr = Trial()
    shape = {i: (0.002, z) for i, (_, z) in AM_SHAPE["G"].items()}   # 2 mm along the crown, toward the C string
    tr.shape = {"G": shape}
    tr.hold(1.0)                                          # the press itself settles
    assert M.MON.events["G"] is None
    tr.strum()
    tr.hold(0.7)
    g_pluck = [p for p in M.MON.plucks if p["string"] == "G"][-1]
    assert g_pluck["sounded"] and g_pluck["stop"] == 2


def test_lifting_the_finger_while_the_a_string_rings_fails_it():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum()
    tr.shape = {}                                         # the finger leaves the fret 0.1 s later
    tr.hold(0.7)
    assert M.MON.report()["strums"][0]["sounding"] == ""             # the C chord's A did not ring: no credit


def test_a_firm_touch_at_0_45_s_is_not_the_next_strum():
    """Late in its ring a string may be cut only by the next strum plucking it again, not by a hand resting on it."""
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum()
    tr.hold(0.33)                                         # 0.45 s after G was let go
    tr.extra["G"] = {I_STRUM: (0.0, -0.0015)}             # a palm presses G down (0.5 N) and stays
    tr.hold(0.6)
    rep = M.MON.report()
    assert rep["strums"][0]["sounding"] == "CEA"
    g_pluck = [p for p in M.MON.plucks if p["string"] == "G"][0]
    assert not g_pluck["sounded"]


def test_the_touch_threshold_while_ringing_scales_with_the_swing():
    """Measured in Isaac Sim: a freely ringing string reads up to 23 N/m of its swing (G at 6.3 mm: 0.14 N). The
    touch threshold while ringing is the larger of 0.15 N and 35 N/m of the pluck's peak: a 12 mm swing, the largest
    allowed, may read 0.28 N free, under its 0.42 N threshold; a fret slap reads 0.8-3.3 N."""
    for peak in (0.002, 0.0064, 0.012):
        assert 0.023 * peak * 1000 < max(M.F_MUTE, M.F_RING_PER_M * peak) * 0.7
    assert M.F_RING_PER_M * M.OVERPULL < 0.8


def test_a_big_swing_that_touches_nothing_still_sounds():
    tr = Trial()
    tr.shape = dict(C_SHAPE)
    tr.hold(0.3)
    tr.strum(amp=0.006)
    tr.hold(0.7)
    assert M.MON.report()["strums"][0]["sounding"] == "GCEA"


def test_pulling_a_string_out_of_line_scores_0_for_good():
    tr = Trial()
    tr.perform()
    assert score() == pytest.approx(1.0)
    tr.pluck("C", amp=0.013)
    assert score() == 0.0 and not TASK.success(None)


def test_the_saddle_end_node_is_not_watched_for_touches():
    F = np.zeros((N, 2))
    F[I_STRUM, 0] = 1.0                                   # the strum's contact
    F[N - 1, 0] = 0.8                                     # the last node's reading
    assert not M.muted(F, M._WATCH)
    F[12, 1] = 0.3                                        # a real touch elsewhere on the string
    assert M.muted(F, M._WATCH)


# -- the demo -------------------------------------------------------------------------------------------------------


def test_demo_reach_and_limits():
    worst = PLAN[-1]["worst"]
    for side in ("left", "right"):
        assert worst[side][0] < 0.5 and worst[side][1] < 0.1
        q = np.array([st[side][0] for st in STEPS])
        assert np.all(q >= np.array(ik.LOWER[:6]) - 1e-6) and np.all(q <= np.array(ik.UPPER[:6]) + 1e-6)


def test_demo_fits_the_run_limit():
    assert len(STEPS) + 60 < 1200
    assert len(demo.PROFILES) == 8 and all(p == demo.PROFILE for p in demo.PROFILES)   # one strum for all eight


def test_demo_does_not_read_simulator_objects():
    src = inspect.getsource(demo.Script) + inspect.getsource(demo.plan) + inspect.getsource(demo.fret_pose) + \
        inspect.getsource(demo.strum_pose)
    for word in ("object_states", "joint_state", "_arrays", "MON", "scene["):
        assert word not in src


def _tip_world(step, arm, tip):
    _, _, tcp, rot = step[arm]
    return tipkit.to_world(TIPS[tip], tcp, rot)


def _string_gap(points, nm, w=None):
    out = np.inf
    for i in range(N):
        a, b = g.node_segment(nm, i, (0.0, 0.0) if w is None else w[i])
        out = min(out, float(tipkit.point_seg_dist(points, a, b).min()) - g.string_spec(nm)[1] / 2)
    return out


def _in_u(points):
    return (g.ROT_U.T @ (points - g.U0).T).T


@pytest.mark.parametrize("chord,name,fret,neighbour", [("C", "A", 3, "E"), ("Am", "G", 2, "C")])
def test_the_fretting_tip_presses_across_its_string_behind_the_fret_clear_of_the_next(chord, name, fret, neighbour):
    st = next(s for s in STEPS if s["phase"]["fret"] == f"hold {chord}")
    pts = {t: _tip_world(st, demo.FRET_ARM, t) for t in TIPS}
    low = min(pts, key=lambda t: pts[t][:, 2].min())
    u = _in_u(pts[low])
    zmin = u[:, 2].min()
    edge = u[u[:, 2] < zmin + 0.0005]                     # the nub's rounded end: a line across the strings
    s0, s1 = -edge[:, 1].max(), -edge[:, 1].min()
    assert g.fret_s(fret - 1) + 0.005 < s0 and s1 < g.fret_s(fret) - 0.002
    s = 0.5 * (s0 + s1)
    xs = float(g.string_x(name, s))
    assert edge[:, 0].min() < xs - 0.003 and edge[:, 0].max() > xs + 0.003   # across the string, both sides
    assert zmin < g.string_z(name, s) + g.string_spec(name)[1] / 2 - 0.0015  # it presses the string down
    assert _string_gap(pts[low], neighbour) > 0.003
    for _, c, h, r in g.fret_capsules():
        a, b = g.u_to_station(np.add(c, (-h, 0, 0))), g.u_to_station(np.add(c, (h, 0, 0)))
        assert tipkit.point_seg_dist(pts[low], a, b).min() - r > 0.0012
    other = [t for t in pts if t != low][0]
    for nm in g.NAMES:
        assert _string_gap(pts[other], nm) > 0.04


def test_the_chord_change_clears_the_strings_and_waits_out_the_ring():
    change = [s for s in STEPS if s["phase"]["fret"] == "change"]
    assert 0 < len(change) * demo.DT <= 0.41
    # the G string is pressed at least 0.1 s before the fifth downstroke reaches it
    x0, x1 = demo.strum_span()
    fg = (float(g.string_x("G", demo.STRUM_S)) - x0) / (x1 - x0)
    tt = np.linspace(0, 1, 20001)
    t_g = demo.SWEEP_S * float(tt[np.argmin(np.abs(3 * tt ** 2 - 2 * tt ** 3 - fg))])
    sc0 = demo.Script()
    k5 = g.SEQUENCE.index("Am")
    end = (max(i for i, s in enumerate(STEPS) if s["phase"]["fret"] == "change") + 1) * demo.DT
    assert end + 0.1 <= sc0.starts[k5] + t_g
    sc = demo.Script()
    k4 = g.SEQUENCE.index("Am") - 1
    # the A string (crossed about 0.35 s into the downstroke, let go 0.05 s later) has rung at least LIFT_S before
    # the finger leaves it
    x0, x1 = demo.strum_span()
    f = (float(g.string_x("A", demo.STRUM_S)) - x0) / (x1 - x0)
    ts = np.linspace(0, 1, 20001)
    t_a = demo.SWEEP_S * float(ts[np.argmin(np.abs(3 * ts ** 2 - 2 * ts ** 3 - f))])
    assert demo.CHANGE_AT >= t_a + 0.05 + M.LIFT_S - 0.02
    for st in change[1:-1]:                               # in transit the nub passes over E and C, clear of them
        pts = {t: _tip_world(st, demo.FRET_ARM, t) for t in TIPS}
        low = min(pts, key=lambda t: pts[t][:, 2].min())
        for nm in ("C", "E"):
            assert _string_gap(pts[low], nm) > 0.0015


def test_the_strumming_tip_crosses_every_string_and_clears_the_soundboard():
    sweeps = [s for s in STEPS if s["phase"]["strum"].startswith("strum")]
    returns = [s for s in STEPS if s["phase"]["strum"] == "return"]
    poised = next(s for s in STEPS if s["phase"]["strum"] == "poised")
    pts = {t: _tip_world(poised, demo.STRUM_ARM, t) for t in TIPS}
    low = min(pts, key=lambda t: pts[t][:, 2].min())
    other = [t for t in TIPS if t != low][0]
    for nm in g.NAMES:
        assert _string_gap(pts[low], nm) > 0.003          # poised beyond the G string, clear of all four
    for st in sweeps:
        assert _in_u(_tip_world(st, demo.STRUM_ARM, low))[:, 2].min() > g.TOP_Z + 0.003   # over the soundboard
        for nm in g.NAMES:
            assert _string_gap(_tip_world(st, demo.STRUM_ARM, other), nm) > 0.02
    for st in returns[1:]:                                # the way back is over the strings
        for nm in g.NAMES:
            assert _string_gap(_tip_world(st, demo.STRUM_ARM, low), nm) > 0.004
    x0, x1 = demo.strum_span()
    for nm in g.NAMES:
        assert x0 + 0.005 < g.string_x(nm, demo.STRUM_S) < x1 - 0.005
    assert demo.STRUM_S - g.NUB_HALF - g.NUB_R > g.FB_END + 0.003


def test_the_planned_beat_and_strum():
    sc = demo.Script()
    gaps = np.diff(sc.starts)
    assert len(sc.starts) == len(g.SEQUENCE) == 8
    assert M.BEAT_RANGE[0] <= gaps.mean() <= M.BEAT_RANGE[1]
    assert np.all(np.abs(gaps - gaps.mean()) <= 0.7 * M.BEAT_TOL * gaps.mean())
    x0, x1 = demo.strum_span()

    def t_at(x):                                          # the smoothstep sweep reaches x at this time
        f = (x - x0) / (x1 - x0)
        ts = np.linspace(0, 1, 20001)
        return demo.SWEEP_S * float(ts[np.argmin(np.abs(3 * ts ** 2 - 2 * ts ** 3 - f))])

    crossing = t_at(float(g.string_x("A", demo.STRUM_S))) - t_at(float(g.string_x("G", demo.STRUM_S)))
    assert crossing < 0.85 * M.WINDOW_S
    # a downstroke and the way back fit in a beat; the downstroke runs toward the robot (G is the farthest string)
    assert demo.SWEEP_S + 0.07 + 0.18 + 0.07 + 0.15 < demo.BEAT     # with a still moment before each downstroke
    assert g.u_to_station([g.string_x("G", 0.2), -0.2, 0])[0] > g.u_to_station([g.string_x("A", 0.2), -0.2, 0])[0]


def test_the_hands_stay_apart():
    for st in STEPS:
        assert np.linalg.norm(st["left"][2] - st["right"][2]) > 0.10
