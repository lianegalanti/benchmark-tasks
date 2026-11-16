"""An offline model of the strings for the tests: the same discretised taut string the sim builds (stiffness,
damping, masses and joint armature from geometry.py), integrated with implicit Euler at 240 Hz, with nodes that a
finger, a fret or a strum hold at a prescribed displacement. Pure Python (NumPy)."""

from __future__ import annotations

import numpy as np

import geometry as g

DT = 1.0 / 240
SUB = 10            # implicit Euler substeps per 240 Hz sample (keeps its numerical damping small)


class StringSim:
    def __init__(self, name: str):
        self.name = name
        n = g.N_NODES
        self.w = np.zeros((n, 2))
        self.v = np.zeros((n, 2))
        self.a = np.zeros((n, 2))
        K = g.stiffness_matrix(name)
        kt = g.tendon_stiffness(name)
        Kj = K.copy()
        Kj[-1, -1] -= kt
        C = g.DAMP_TAU * Kj
        C[-1, -1] += g.TENDON_DAMP
        self.K, self.C = K, C
        self.M = [g.mass_matrix(name, 0), g.mass_matrix(name, 1)]

    def step(self, held: dict | None = None) -> None:
        """One 240 Hz step; ``held`` {node: (x, z) displacement} pins those nodes (moved linearly through the step;
        their loads are the holding forces). ``a`` is the step's mean acceleration, as the monitor estimates it."""
        held = held or {}
        start = {i: self.w[i].copy() for i in held}
        v0 = self.v.copy()
        for k in range(1, SUB + 1):
            self._sub({i: start[i] + (np.asarray(held[i]) - start[i]) * k / SUB for i in held}, DT / SUB)
        self.a = (self.v - v0) / DT

    def _sub(self, held: dict, dt: float) -> None:
        n = g.N_NODES
        free = [i for i in range(n) if i not in held]
        hi = sorted(held)
        for ax in range(2):
            M = self.M[ax]
            A = M + dt * self.C + dt * dt * self.K
            rhs = M @ self.v[:, ax] - dt * self.K @ self.w[:, ax]
            v_new = np.zeros(n)
            for i in hi:
                v_new[i] = (held[i][ax] - self.w[i, ax]) / dt
            if free:
                v_new[free] = np.linalg.solve(A[np.ix_(free, free)], rhs[free] - A[np.ix_(free, hi)] @ v_new[hi])
            self.v[:, ax] = v_new
            self.w[:, ax] = self.w[:, ax] + dt * v_new
