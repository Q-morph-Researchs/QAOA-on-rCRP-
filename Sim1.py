#!/usr/bin/env python3
"""
crp_qaoa.py -- QAOA for the restricted Container Relocation Problem (rCRP).
Single file.  Everything: instances, exact solver, QUBO, Ising, QAOA, validation,
benchmark-style results table.

    pip install numpy scipy                # minimum
    pip install qiskit qiskit-aer          # only for --backend aer / --noise

    python crp_qaoa.py --stacks "1,6,3,8; 5,2,7; 4; 9" --h 4
    python crp_qaoa.py --dir instances/group4 --glob "data3-3-*" --csv out.csv

Run `python crp_qaoa.py --help` for all options, or see the USAGE section at the
bottom of this docstring.

BACKENDS
--------
--backend numpy  (default)  QAOA evaluated from its defining equation on the
                            state vector.  Exact, noiseless.
--backend aer               The same circuit built as Qiskit gates and run on
                            AerSimulator.  Mathematically identical: measured
                            agreement 3e-17, and 8.5x slower (27x with shots).
                            Use it for a cross-check, for shot noise, and for
                            --noise.  Not for sweeps.

USAGE
-----
  --stacks "1,6,3,8; 5,2,7; 4; 9" --h 4     one bay typed in (bottom -> top)
  --file data3-3-11 --published 3           one benchmark instance
  --dir DIR --glob "data3-3-*"              a whole class
  --random 40 --w 3 --h 5 --n 9             random instances
  --sweep-depth 1 2 3 4 5 6 --repeats 10    the main experiment
  --backend aer --shots 4096                gate-based, shot-based
  --backend aer --noise 0.001               depolarising noise per 2-qubit gate
  --crosscheck                              report |P_aer - P_numpy|
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import glob as globmod
import itertools
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

Bay = Tuple[Tuple[int, ...], ...]          # stacks, each listed bottom -> top


# ###########################################################################
#  SECTION 1 -- INSTANCES
# ###########################################################################

@dataclass
class Instance:
    name: str
    w: int                                  # stacks
    h: int                                  # tier limit
    n: int                                  # containers
    bay: Bay
    published_opt: Optional[int] = None

    def __post_init__(self):
        labels = sorted(c for s in self.bay for c in s)
        if labels != list(range(1, self.n + 1)):
            raise ValueError(f"{self.name}: labels must be exactly 1..{self.n}, got {labels}")
        if len(self.bay) != self.w:
            raise ValueError(f"{self.name}: expected {self.w} stacks, got {len(self.bay)}")
        for s in self.bay:
            if len(s) > self.h:
                raise ValueError(f"{self.name}: stack {s} exceeds tier limit {self.h}")

    def render(self) -> str:
        out = []
        for tier in range(self.h - 1, -1, -1):
            row = "".join(f"{s[tier]:>4}" if len(s) > tier else "   ." for s in self.bay)
            out.append(f"  t{tier + 1} |{row}")
        out.append("      " + "".join(f"{'-':>4}" for _ in self.bay))
        out.append("      " + "".join(f"{f'S{i+1}':>4}" for i in range(self.w)))
        return "\n".join(out)


def parse_instance_file(path: str, name: Optional[str] = None) -> Instance:
    """Read a benchmark instance file.

    Header `w h [n]`, then one line per stack bottom->top, with or without a
    leading per-stack count.  Both layouts are auto-detected.  '#' comments and
    blank lines ignored.
    """
    toks: List[List[int]] = []
    with open(path) as fh:
        for line in fh:
            line = line.split("#")[0].strip()
            if line:
                toks.append([int(x) for x in re.split(r"[\s,]+", line) if x.strip()])
    if not toks:
        raise ValueError(f"{path}: empty file")
    head = toks[0]
    w, h = head[0], head[1]
    rows = toks[1:]
    if len(rows) < w:
        raise ValueError(f"{path}: header says {w} stacks, found {len(rows)} stack lines")
    rows = rows[:w]
    leading = all(len(r) >= 1 and r[0] == len(r) - 1 for r in rows)
    stacks = tuple(tuple(r[1:]) if leading else tuple(r) for r in rows)
    n = head[2] if len(head) > 2 else sum(len(s) for s in stacks)
    return Instance(name or os.path.basename(path), w, h, n, stacks)


def random_instance(w: int, h: int, n: int, seed: int, name: str = "random") -> Instance:
    rng = random.Random(seed)
    labels = list(range(1, n + 1))
    rng.shuffle(labels)
    stacks: List[List[int]] = [[] for _ in range(w)]
    for lab in labels:
        opts = [i for i in range(w) if len(stacks[i]) < h]
        if not opts:
            raise ValueError("bay too small for n containers")
        stacks[rng.choice(opts)].append(lab)
    return Instance(name, w, h, n, tuple(tuple(s) for s in stacks))


# ###########################################################################
#  SECTION 2 -- EXACT SOLVER (ground truth)
# ###########################################################################

_TIER = 4


def _set_tier(h: int) -> None:
    global _TIER
    if h != _TIER:
        _TIER = h
        _solve.cache_clear()


def blocking_lower_bound(bay: Bay) -> int:
    """Each container above a smaller label must move at least once."""
    return sum(1 for s in bay for i, c in enumerate(s) if any(c > o for o in s[:i]))


@lru_cache(maxsize=None)
def _solve(bay: Bay) -> int:
    labels = [c for s in bay for c in s]
    if not labels:
        return 0
    target = min(labels)
    si = next(i for i, s in enumerate(bay) if target in s)
    stack = bay[si]
    if stack[-1] == target:                       # clear -> retrieve (free)
        nb = list(bay); nb[si] = stack[:-1]
        return _solve(tuple(nb))
    best = math.inf
    top = stack[-1]
    for j in range(len(bay)):                     # restricted: only this top moves
        if j == si or len(bay[j]) >= _TIER:
            continue
        nb = list(bay); nb[si] = stack[:-1]; nb[j] = bay[j] + (top,)
        best = min(best, 1 + _solve(tuple(nb)))
    return best


def exact_relocations(bay: Bay, h: int) -> int:
    _set_tier(h)
    return _solve(tuple(tuple(s) for s in bay))


# ###########################################################################
#  SECTION 3 -- CLASSICAL HEURISTIC BASELINES
# ###########################################################################

def heuristic_relocations(bay: Bay, h: int, rule: str) -> Optional[int]:
    stacks = [list(s) for s in bay]
    total = 0
    while any(stacks):
        target = min(c for s in stacks for c in s)
        si = next(i for i, s in enumerate(stacks) if target in s)
        if stacks[si][-1] == target:
            stacks[si].pop(); continue
        cand = [j for j in range(len(stacks)) if j != si and len(stacks[j]) < h]
        if not cand:
            return None
        c = stacks[si].pop()
        if rule == "level":          # keep stack heights even
            j = min(cand, key=lambda j: (len(stacks[j]), j))
        elif rule == "rindex":       # prefer the stack whose lowest label is largest
            j = max(cand, key=lambda j: (min(stacks[j]) if stacks[j] else math.inf, -j))
        else:
            raise ValueError(rule)
        stacks[j].append(c); total += 1
    return total


# ###########################################################################
#  SECTION 4 -- THE BATCH SUBPROBLEM
# ###########################################################################

@dataclass
class Batch:
    target: int
    target_stack: int
    blockers: List[int]              # relocation order, topmost first
    dests: List[int]                 # candidate destination stack indices
    free: Dict[int, int]


def extract_batch(bay: Bay, h: int) -> Optional[Batch]:
    labels = [c for s in bay for c in s]
    if not labels:
        return None
    target = min(labels)
    si = next(i for i, s in enumerate(bay) if target in s)
    above = list(bay[si][bay[si].index(target) + 1:])
    if not above:
        return None                                  # already clear
    blockers = above[::-1]
    dests = [j for j in range(len(bay)) if j != si and len(bay[j]) < h]
    free = {j: h - len(bay[j]) for j in dests}
    if sum(free.values()) < len(blockers):
        return None
    return Batch(target, si, blockers, dests, free)


# ###########################################################################
#  SECTION 5 -- QUBO
# ###########################################################################

@dataclass
class QUBO:
    Q: np.ndarray
    offset: float
    n: int
    n_decision: int
    nB: int
    nS: int
    blockers: List[int]
    dests: List[int]
    free: Dict[int, int]
    A: float
    B: float
    slack: Dict[int, List[int]] = field(default_factory=dict)

    def q(self, i: int, j: int) -> int:
        return i * self.nS + j


def _slack_weights(f: int) -> List[int]:
    """Binary weights representing every integer in [0, f] and nothing above."""
    weights, total, k = [], 0, 0
    while total < f:
        wt = min(2 ** k, f - total)
        weights.append(wt); total += wt; k += 1
    return weights


def build_qubo(bay: Bay, batch: Batch, A: float = 2.0, B: float = 2.0,
               capacity: str = "slack", corrected: bool = True) -> Optional[QUBO]:
    """Cost = burial + mis-stack + one-hot penalty + capacity penalty.

    1. burial     w[i][j] x_ij                     placement buries an earlier-due container
    2. mis-stack  v[i][k](1-w[k][j]) x_ij x_kj     two blockers stacked wrong way round
    3. one-hot    A (1 - sum_j x_ij)^2             each blocker goes exactly one place
    4. capacity   at most free[j] blockers on j

    The (1 - w[k][j]) factor removes a double-count: a blocker that already
    buries a pre-existing smaller container must be relocated anyway, so also
    charging it for burying blocker i counts the same crane move twice.
    Without it the objective is NOT a valid lower bound.
    """
    blockers, dests, free = batch.blockers, batch.dests, batch.free
    nB, nS = len(blockers), len(dests)
    n_dec = nB * nS

    min_lab = {j: (min(bay[j]) if bay[j] else math.inf) for j in dests}
    w = np.array([[1.0 if min_lab[j] < c else 0.0 for j in dests] for c in blockers])
    v = np.zeros((nB, nB))
    for i in range(nB):
        for k in range(i + 1, nB):
            v[i, k] = 1.0 if blockers[i] < blockers[k] else 0.0

    slack: Dict[int, List[int]] = {}
    n_tot = n_dec
    binding = [j for j in range(nS) if free[dests[j]] < nB]
    if capacity == "strict":
        if any(free[dests[j]] != 1 for j in binding):
            return None
    else:
        for j in binding:
            f = free[dests[j]]
            if f == 1:
                continue                               # exact as a pairwise penalty
            wts = _slack_weights(f)
            slack[j] = list(range(n_tot, n_tot + len(wts)))
            n_tot += len(wts)

    Q = np.zeros((n_tot, n_tot)); offset = 0.0
    q = lambda i, j: i * nS + j

    for i in range(nB):                                          # 1
        for j in range(nS):
            Q[q(i, j), q(i, j)] += w[i, j]
    for i in range(nB):                                          # 2
        for k in range(i + 1, nB):
            if not v[i, k]:
                continue
            for j in range(nS):
                coef = v[i, k] * (1.0 - w[k, j]) if corrected else v[i, k]
                if coef:
                    Q[q(i, j), q(k, j)] += coef
    for i in range(nB):                                          # 3
        offset += A
        for j in range(nS):
            Q[q(i, j), q(i, j)] -= A
            for jp in range(j + 1, nS):
                Q[q(i, j), q(i, jp)] += 2 * A
    for j in binding:                                            # 4
        f = free[dests[j]]
        if f == 1:
            for i in range(nB):
                for k in range(i + 1, nB):
                    Q[q(i, j), q(k, j)] += B
        else:                     # B (sum_i x_ij + sum_t wt_t y_t - f)^2, x^2 = x
            terms = [(q(i, j), 1) for i in range(nB)] + list(zip(slack[j], _slack_weights(f)))
            offset += B * f * f
            for (a, ca) in terms:
                Q[a, a] += B * (ca * ca - 2 * f * ca)
                for (b, cb) in terms:
                    if a < b:
                        Q[a, b] += 2 * B * ca * cb

    return QUBO(Q, offset, n_tot, n_dec, nB, nS, blockers, dests, free, A, B, slack)


def all_energies(qubo: QUBO) -> Tuple[np.ndarray, np.ndarray]:
    """Cost of every one of the 2^n bitstrings.  Exhaustive; n <= ~22."""
    bits = np.array(list(itertools.product([0, 1], repeat=qubo.n)), dtype=np.int8)
    x = bits.astype(np.float64)
    return bits, np.einsum("bi,ij,bj->b", x, qubo.Q, x) + qubo.offset


def decode(bitrow: Sequence[int], qubo: QUBO) -> Tuple[Dict[int, int], bool]:
    plan: Dict[int, int] = {}
    for i in range(qubo.nB):
        chosen = [j for j in range(qubo.nS) if bitrow[qubo.q(i, j)] == 1]
        if len(chosen) != 1:
            return {}, False                              # one-hot violated
        plan[qubo.blockers[i]] = qubo.dests[chosen[0]]
    for j in range(qubo.nS):
        d = qubo.dests[j]
        if sum(1 for s in plan.values() if s == d) > qubo.free[d]:
            return plan, False                            # capacity violated
    return plan, True


def apply_plan(bay: Bay, batch: Batch, plan: Dict[int, int]) -> Tuple[Bay, int]:
    stacks = [list(s) for s in bay]
    si, moves = batch.target_stack, 0
    while stacks[si][-1] != batch.target:
        c = stacks[si].pop(); stacks[plan[c]].append(c); moves += 1
    stacks[si].pop()                                      # retrieve the target
    return tuple(tuple(s) for s in stacks), moves


# ###########################################################################
#  SECTION 6 -- QUBO <-> ISING
# ###########################################################################
# Only needed to build a GATE circuit.  The numpy backend works directly with
# the QUBO cost, because shifting every energy by a constant is a global phase
# and changes no probability.

def qubo_to_ising(qubo: QUBO):
    """Standard substitution x = (1 - z)/2.  Returns (h, J, offset)."""
    Q = (qubo.Q + qubo.Q.T) / 2.0
    n = qubo.n
    h = np.zeros(n); J: Dict[Tuple[int, int], float] = {}; off = qubo.offset
    for i in range(n):
        off += Q[i, i] / 2.0; h[i] -= Q[i, i] / 2.0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            off += Q[i, j] / 4.0
            h[i] -= Q[i, j] / 4.0
            h[j] -= Q[i, j] / 4.0
            if i < j:
                J[(i, j)] = J.get((i, j), 0.0) + Q[i, j] / 2.0
    return h, J, off


def ising_energies(h, J, off, bits: np.ndarray) -> np.ndarray:
    z = 1.0 - 2.0 * bits.astype(np.float64)
    E = z @ h + off
    for (i, j), c in J.items():
        E = E + c * z[:, i] * z[:, j]
    return E


# ###########################################################################
#  SECTION 7 -- QAOA
# ###########################################################################
# |psi_p> = e^{-i b_p B} e^{-i g_p C} ... e^{-i b_1 B} e^{-i g_1 C} |+>^n
#
# C is diagonal (entry E(z) for bitstring z), so the cost layer multiplies
# amplitude z by exp(-i gamma E(z)).  B = sum_i X_i factorises into a 2x2
# rotation per qubit.  That is the whole algorithm.

def qaoa_probs_numpy(E: np.ndarray, theta: np.ndarray, p: int, n: int) -> np.ndarray:
    gammas, betas = theta[:p], theta[p:]
    psi = np.full(2 ** n, 2.0 ** (-n / 2), dtype=complex)
    for l in range(p):
        psi = psi * np.exp(-1j * gammas[l] * E)             # cost layer
        t = psi.reshape([2] * n)                            # mixer layer
        c, s = np.cos(betas[l]), -1j * np.sin(betas[l])
        for qb in range(n):
            t = np.moveaxis(t, qb, 0)
            a0, a1 = t[0].copy(), t[1].copy()
            t[0] = c * a0 + s * a1
            t[1] = s * a0 + c * a1
            t = np.moveaxis(t, 0, qb)
        psi = t.reshape(-1)
    return np.abs(psi) ** 2


# --------------------------------------------------------------------------
#  Qiskit / Aer backend.  Same algorithm, expressed as gates.
# --------------------------------------------------------------------------

def build_circuit(h, J, n: int, theta: np.ndarray, p: int, measure: bool = False):
    """One QAOA layer = RZ(2*g*h_i) per field, CX-RZ(2*g*J_ij)-CX per coupling,
    RX(2*b) per qubit.  Note the factor of 2: Qiskit's rz(t) is exp(-i t Z/2)."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for l in range(p):
        g, b = theta[l], theta[p + l]
        for i in range(n):
            if abs(h[i]) > 1e-12:
                qc.rz(2 * g * h[i], i)
        for (i, j), c in J.items():
            if abs(c) > 1e-12:
                qc.cx(i, j); qc.rz(2 * g * c, j); qc.cx(i, j)
        for i in range(n):
            qc.rx(2 * b, i)
    if measure:
        qc.measure_all()
    return qc


def _aer_index_map(bits: np.ndarray) -> np.ndarray:
    """Qiskit is little-endian: its basis index is the bitstring reversed."""
    return np.array([int("".join(str(int(x)) for x in row[::-1]), 2) for row in bits])


class AerBackend:
    """Gate-based evaluation on AerSimulator.  Identical maths, ~8.5x slower."""

    def __init__(self, qubo: QUBO, bits: np.ndarray, shots: int = 0,
                 noise: float = 0.0, seed: int = 0):
        from qiskit_aer import AerSimulator
        self.h, self.J, _ = qubo_to_ising(qubo)
        self.n = qubo.n
        self.shots = shots
        self.order = _aer_index_map(bits)
        nm = None
        if noise > 0:
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(noise, 2), ["cx"])
            nm.add_all_qubit_quantum_error(depolarizing_error(noise / 10, 1),
                                           ["rz", "rx", "h"])
        method = "density_matrix" if noise > 0 else "statevector"
        self.sim = AerSimulator(method=method, noise_model=nm, seed_simulator=seed)
        self.noise = noise

    def probs(self, theta: np.ndarray, p: int) -> np.ndarray:
        from qiskit import transpile
        if self.shots > 0 or self.noise > 0:
            qc = build_circuit(self.h, self.J, self.n, theta, p, measure=True)
            res = self.sim.run(transpile(qc, self.sim),
                               shots=self.shots or 4096).result().get_counts()
            pr = np.zeros(2 ** self.n)
            tot = sum(res.values())
            for key, cnt in res.items():
                pr[int(key.replace(" ", ""), 2)] = cnt / tot
        else:
            qc = build_circuit(self.h, self.J, self.n, theta, p)
            qc.save_statevector()
            sv = self.sim.run(transpile(qc, self.sim)).result().get_statevector()
            pr = np.abs(np.asarray(sv)) ** 2
        return pr[self.order]                       # back to our bit ordering


# --------------------------------------------------------------------------

def make_objective(probs_fn: Callable, E: np.ndarray, p: int,
                   alpha: float = 1.0) -> Callable:
    """Mean energy (alpha=1) or CVaR over the cheapest alpha fraction."""
    order = np.argsort(E); E_sorted = E[order]

    def F(theta) -> float:
        pr = probs_fn(np.asarray(theta), p)
        if alpha >= 1.0:
            return float(pr @ E)
        ps = pr[order]
        cum = np.cumsum(ps)
        k = min(int(np.searchsorted(cum, alpha) + 1), len(ps))
        wts = ps[:k].copy()
        wts[-1] -= cum[k - 1] - alpha
        return float(wts @ E_sorted[:k] / alpha)
    return F


@dataclass
class QAOAResult:
    probs: np.ndarray
    theta: np.ndarray
    F: float
    n_evals: int


def run_qaoa(E: np.ndarray, n: int, p: int = 3, alpha: float = 0.15,
             restarts: int = 16, maxiter: int = 400, seed: int = 0,
             probs_fn: Optional[Callable] = None) -> QAOAResult:
    """COBYLA with random restarts.  The landscape is non-convex, so a single
    run from a random start is a coin flip -- restarting is standard practice."""
    if probs_fn is None:
        probs_fn = lambda th, pp: qaoa_probs_numpy(E, th, pp, n)
    rng = np.random.default_rng(seed)
    F = make_objective(probs_fn, E, p, alpha)
    evals = [0]

    def counted(th):
        evals[0] += 1
        return F(th)

    best = None
    for _ in range(restarts):
        x0 = np.concatenate([rng.uniform(0, 0.8, p),         # gammas start small
                             rng.uniform(0, np.pi / 2, p)])  # betas
        r = minimize(counted, x0, method="COBYLA",
                     options={"maxiter": maxiter, "rhobeg": 0.3})
        if best is None or r.fun < best.fun:
            best = r
    return QAOAResult(probs_fn(best.x, p), best.x, float(best.fun), evals[0])


# ###########################################################################
#  SECTION 8 -- METRICS AND CLASSICAL SOLVER BASELINES
# ###########################################################################

def metrics(probs: np.ndarray, E: np.ndarray, feasible: np.ndarray) -> Dict[str, float]:
    opt = (E == E.min())
    p_opt = float(probs[opt].sum())
    n_feas = int(feasible.sum())
    k95 = (math.inf if p_opt <= 1e-15
           else math.ceil(math.log(0.05) / math.log(max(1e-15, 1 - min(p_opt, 1 - 1e-12)))))
    mean, rand = float(probs @ E), float(E.mean())
    return dict(
        P_opt=p_opt,
        P_feasible=float(probs[feasible].sum()),
        mean_cost=mean,
        normalised_cost=(mean - E.min()) / (rand - E.min()) if rand > E.min() else 0.0,
        baseline_uniform=float(opt.sum()) / len(E),          # weak baseline
        baseline_feasible=float((opt & feasible).sum()) / max(n_feas, 1),   # strict
        n_feasible=n_feas,
        shots95=k95,
    )


def sample_counts(probs: np.ndarray, shots: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pr = np.clip(probs, 0, None)
    idx = rng.choice(len(pr), size=shots, p=pr / pr.sum())
    return np.bincount(idx, minlength=len(pr))


def simulated_annealing(E: np.ndarray, n: int, budget: int, seed: int = 0) -> float:
    """Matched-evaluation-budget classical baseline.  This column exists so that
    a quantum-speedup claim cannot be made by accident."""
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(0, 2 ** n)); cur = E[idx]; best = cur
    for t in range(budget):
        T = max(1e-3, 1.0 - t / budget)
        idx2 = idx ^ (1 << int(rng.integers(0, n)))
        new = E[idx2]
        if new <= cur or rng.random() < math.exp(-(new - cur) / T):
            idx, cur = idx2, new
            best = min(best, cur)
    return float(best)


# ###########################################################################
#  SECTION 9 -- VALIDATION SUITE
# ###########################################################################

@dataclass
class Validation:
    checks: Dict[str, bool] = field(default_factory=dict)

    def record(self, name: str, ok: bool) -> None:
        self.checks[name] = self.checks.get(name, True) and bool(ok)

    @property
    def all_passed(self) -> bool:
        return all(self.checks.values())

    def report(self) -> str:
        return "\n".join(f"   {'PASS' if v else 'FAIL'}  {k}" for k, v in self.checks.items())


def validate_batch(qubo: QUBO, bits: np.ndarray, E: np.ndarray,
                   probs: Optional[np.ndarray], val: Validation) -> None:
    feas = np.array([decode(b, qubo)[1] for b in bits])
    if (~feas).any():
        val.record("V1 infeasible states penalised >= A", E[~feas].min() >= qubo.A - 1e-9)
    val.record("V2 optimum is feasible", bool(feas[int(np.argmin(E))]))
    h, J, off = qubo_to_ising(qubo)
    val.record("V3 Ising == QUBO on all 2^n states",
               bool(np.allclose(ising_energies(h, J, off, bits), E, atol=1e-9)))
    pr0 = qaoa_probs_numpy(E, np.array([0.0, 0.31]), 1, qubo.n)
    val.record("V4 gamma=0 gives uniform", bool(np.allclose(pr0, 2.0 ** -qubo.n)))
    if probs is not None:
        val.record("V5 probabilities sum to 1", abs(float(probs.sum()) - 1.0) < 1e-6)


def falsification_control(qubo: QUBO, bits: np.ndarray, E: np.ndarray) -> bool:
    """Corrupt one Ising field; V3 must then FAIL.  Proves the test can fail --
    the difference between a test and a decoration."""
    h, J, off = qubo_to_ising(qubo)
    h = h.copy(); h[0] += 1.0
    return not np.allclose(ising_energies(h, J, off, bits), E, atol=1e-9)


# ###########################################################################
#  SECTION 10 -- ROLLING-HORIZON PIPELINE
# ###########################################################################

@dataclass
class Config:
    p: int = 3
    alpha: float = 0.15                # CVaR tail; 1.0 = standard mean-energy QAOA
    restarts: int = 16
    maxiter: int = 400
    seed: int = 0
    shots: int = 4096
    A: float = 2.0
    B: float = 2.0
    capacity: str = "slack"            # "slack" (general) | "strict" (scope-limited)
    corrected: bool = True
    max_qubits: int = 20
    verify: bool = True
    sa_baseline: bool = True
    backend: str = "numpy"             # "numpy" | "aer"
    aer_shots: int = 0                 # 0 = exact statevector on Aer
    noise: float = 0.0                 # depolarising error per 2-qubit gate
    crosscheck: bool = False


@dataclass
class InstanceResult:
    name: str; w: int; h: int; n: int
    opt: Optional[int] = None
    published: Optional[int] = None
    qubo: Optional[int] = None          # rolling horizon, EXACT QUBO minimiser
    qaoa: Optional[int] = None          # rolling horizon, QAOA + shots
    lvl: Optional[int] = None
    ridx: Optional[int] = None
    sa: Optional[int] = None
    time: float = 0.0
    qubits: int = 0
    batches: int = 0
    evals: int = 0
    p_opt: float = float("nan")
    p_feas: float = float("nan")
    shots95: float = float("nan")
    base_feas: float = float("nan")
    base_unif: float = float("nan")
    lb: int = 0
    xcheck: float = float("nan")
    status: str = "ok"
    validation: Validation = field(default_factory=Validation)

    @property
    def gap(self):
        """Total: QAOA pipeline vs exact optimum."""
        return None if (self.qaoa is None or self.opt is None) else self.qaoa - self.opt

    @property
    def myopia_gap(self):
        """Encoding's fault: perfect QUBO solving vs exact optimum."""
        return None if (self.qubo is None or self.opt is None) else self.qubo - self.opt

    @property
    def solver_gap(self):
        """QAOA's fault: QAOA vs its own QUBO's optimum."""
        return None if (self.qaoa is None or self.qubo is None) else self.qaoa - self.qubo

    @property
    def matches_published(self):
        if self.opt is None or self.published is None:
            return None
        return self.opt == self.published


def _rolling(inst: Instance, cfg: Config, policy: str, res: InstanceResult):
    """Retrieve-and-relocate to completion under one policy.

    policy "exact_qubo" : QUBO solved exactly at every batch
    policy "qaoa"       : QUBO -> QAOA -> shots -> best legal plan

    Running both isolates the two error sources (myopia vs solver).
    """
    bay = inst.bay
    total = 0
    popts, pfeas, s95, bfeas, bunif, sas, xs = [], [], [], [], [], [], []
    rng = np.random.default_rng(cfg.seed if policy == "qaoa" else cfg.seed + 7919)

    while any(bay):
        batch = extract_batch(bay, inst.h)
        if batch is None:                                  # target already clear
            labels = [c for s in bay for c in s]
            if not labels:
                break
            target = min(labels)
            si = next(i for i, s in enumerate(bay) if target in s)
            if bay[si][-1] != target:
                return None, "stuck: no destination"
            nb = list(bay); nb[si] = bay[si][:-1]; bay = tuple(nb)
            continue

        qubo = build_qubo(bay, batch, A=cfg.A, B=cfg.B,
                          capacity=cfg.capacity, corrected=cfg.corrected)
        if qubo is None:
            return None, "out of scope (capacity=strict)"
        if qubo.n > cfg.max_qubits:
            return None, f"batch needs {qubo.n} qubits > max_qubits={cfg.max_qubits}"

        bits, E = all_energies(qubo)
        feas = np.array([decode(b, qubo)[1] for b in bits])

        if policy == "exact_qubo":
            cand = np.flatnonzero(feas)
            plan, _ = decode(bits[cand[np.argmin(E[cand])]], qubo)
        else:
            probs_fn = None
            if cfg.backend == "aer":
                be = AerBackend(qubo, bits, shots=cfg.aer_shots, noise=cfg.noise,
                                seed=int(rng.integers(0, 2 ** 31)))
                probs_fn = lambda th, pp: be.probs(th, pp)

            r = run_qaoa(E, qubo.n, p=cfg.p, alpha=cfg.alpha, restarts=cfg.restarts,
                         maxiter=cfg.maxiter, seed=int(rng.integers(0, 2 ** 31)),
                         probs_fn=probs_fn)
            res.evals += r.n_evals

            if cfg.crosscheck:
                pn = qaoa_probs_numpy(E, r.theta, cfg.p, qubo.n)
                pa = AerBackend(qubo, bits).probs(r.theta, cfg.p)
                xs.append(float(np.max(np.abs(pa - pn))))

            if cfg.verify:
                validate_batch(qubo, bits, E, r.probs, res.validation)

            m = metrics(r.probs, E, feas)
            popts.append(m["P_opt"]); pfeas.append(m["P_feasible"])
            s95.append(m["shots95"]); bfeas.append(m["baseline_feasible"])
            bunif.append(m["baseline_uniform"])
            if cfg.sa_baseline:
                sas.append(simulated_annealing(E, qubo.n, r.n_evals,
                           seed=int(rng.integers(0, 2 ** 31))) == E.min())

            # QAOA's real output: take shots, keep the best LEGAL plan.
            # Any sampled bitstring that decodes is a legal plan -- the quantum
            # part cannot fool you, because you check classically.
            counts = sample_counts(r.probs, cfg.shots, seed=int(rng.integers(0, 2 ** 31)))
            plan = None
            for idx in np.argsort(-counts):
                if counts[idx] == 0:
                    break
                pl, ok = decode(bits[idx], qubo)
                if ok:
                    plan = pl; break
            if plan is None:
                return None, "no feasible plan in shot sample"
            res.qubits = max(res.qubits, qubo.n)
            res.batches += 1

        bay, moves = apply_plan(bay, batch, plan)
        total += moves

    if policy == "qaoa" and popts:
        res.p_opt = float(np.mean(popts)); res.p_feas = float(np.mean(pfeas))
        finite = [x for x in s95 if math.isfinite(x)]
        res.shots95 = float(max(finite)) if finite else math.inf
        res.base_feas = float(np.mean(bfeas)); res.base_unif = float(np.mean(bunif))
        if sas:
            res.sa = int(round(100 * float(np.mean(sas))))
        if xs:
            res.xcheck = float(max(xs))
    return total, "ok"


def solve_instance(inst: Instance, cfg: Config, compute_exact: bool = True) -> InstanceResult:
    res = InstanceResult(inst.name, inst.w, inst.h, inst.n, published=inst.published_opt)
    res.lb = blocking_lower_bound(inst.bay)
    t0 = time.time()
    if compute_exact:
        try:
            res.opt = exact_relocations(inst.bay, inst.h)
        except RecursionError:
            res.status = "exact solver overflow"
    res.lvl = heuristic_relocations(inst.bay, inst.h, "level")
    res.ridx = heuristic_relocations(inst.bay, inst.h, "rindex")
    res.qubo, st1 = _rolling(inst, cfg, "exact_qubo", res)
    res.qaoa, st2 = _rolling(inst, cfg, "qaoa", res)
    res.status = st2 if st2 != "ok" else st1
    res.time = time.time() - t0
    return res


# ###########################################################################
#  SECTION 11 -- RESULTS TABLE
# ###########################################################################

COLUMNS = [
    ("Name", "{:<14}"), ("w", "{:>3}"), ("h", "{:>4}"), ("n", "{:>4}"),
    ("Opt", "{:>5}"), ("Pub", "{:>5}"), ("QUBO", "{:>6}"), ("QAOA", "{:>6}"),
    ("Myop", "{:>6}"), ("Solv", "{:>6}"), ("Gap", "{:>5}"),
    ("Lvl", "{:>5}"), ("RIdx", "{:>6}"), ("Lb", "{:>4}"),
    ("Time", "{:>8}"), ("Qb", "{:>4}"), ("Bat", "{:>5}"), ("Evals", "{:>8}"),
    ("Popt", "{:>7}"), ("Pfeas", "{:>8}"), ("Sh95", "{:>6}"), ("RndF", "{:>7}"),
    ("SA%", "{:>5}"), ("Val", "{:>5}"),
]


def _cell(r: InstanceResult, key: str) -> str:
    num = lambda x: "-" if x is None else f"{x:d}"
    f4 = lambda x: "-" if (x is None or math.isnan(x)) else f"{x:.4f}"
    return {
        "Name": r.name, "w": str(r.w), "h": str(r.h), "n": str(r.n),
        "Opt": num(r.opt), "Pub": num(r.published), "QUBO": num(r.qubo),
        "QAOA": num(r.qaoa), "Myop": num(r.myopia_gap), "Solv": num(r.solver_gap),
        "Gap": num(r.gap), "Lvl": num(r.lvl), "RIdx": num(r.ridx), "Lb": str(r.lb),
        "Time": f"{r.time:.2f}", "Qb": str(r.qubits), "Bat": str(r.batches),
        "Evals": str(r.evals), "Popt": f4(r.p_opt), "Pfeas": f4(r.p_feas),
        "Sh95": "-" if not math.isfinite(r.shots95) else str(int(r.shots95)),
        "RndF": f4(r.base_feas), "SA%": num(r.sa),
        "Val": ("OK" if r.validation.all_passed else "FAIL") if r.validation.checks else "-",
    }[key]


def format_table(results: List[InstanceResult], title: str = "") -> str:
    head = "".join(f.format(name) for name, f in COLUMNS)
    lines = ([title, "=" * len(head)] if title else []) + [head, "-" * len(head)]
    for r in results:
        lines.append("".join(f.format(_cell(r, name)) for name, f in COLUMNS))
    lines += ["-" * len(head), summary(results)]
    return "\n".join(lines)


def summary(results: List[InstanceResult]) -> str:
    ok = [r for r in results if r.status == "ok" and r.opt is not None and r.qaoa is not None]
    if not ok:
        return "no completed instances -- " + "; ".join(
            f"{r.name}: {r.status}" for r in results[:5])
    gaps = np.array([r.gap for r in ok], float)
    out = [f"instances             {len(results)} total, {len(ok)} completed",
           f"QAOA == exact optimum  {np.mean(gaps == 0):.0%}   mean total gap "
           f"{gaps.mean():+.3f} relocations"]
    myo = np.array([r.myopia_gap for r in ok if r.myopia_gap is not None], float)
    sol = np.array([r.solver_gap for r in ok if r.solver_gap is not None], float)
    if len(myo):
        out.append(f"  ... myopia (one-batch horizon)   {myo.mean():+.3f}   "
                   f"[{np.mean(myo == 0):.0%} of instances unaffected]")
    if len(sol):
        out.append(f"  ... solver (QAOA vs its QUBO)    {sol.mean():+.3f}   "
                   f"[{np.mean(sol == 0):.0%} of instances unaffected]")
    for label, attr in (("levelling", "lvl"), ("reshuffle-index", "ridx")):
        v = np.array([getattr(r, attr) - r.opt for r in ok
                      if getattr(r, attr) is not None], float)
        if len(v):
            out.append(f"{label:22s} {np.mean(v == 0):.0%} optimal, mean gap {v.mean():+.3f}")
    pub = [r for r in results if r.matches_published is not None]
    if pub:
        out.append(f"exact solver vs published optimum: "
                   f"{np.mean([r.matches_published for r in pub]):.0%} agreement "
                   f"over {len(pub)} instances")
    po = np.array([r.p_opt for r in ok if not math.isnan(r.p_opt)])
    bf = np.array([r.base_feas for r in ok if not math.isnan(r.base_feas)])
    if len(po):
        out.append(f"median P_opt {np.median(po):.4f}   median random-feasible baseline "
                   f"{np.median(bf):.4f}   ratio {np.median(po)/max(np.median(bf),1e-12):.2f}x")
    sa = [r.sa for r in ok if r.sa is not None]
    if sa:
        out.append(f"simulated annealing at matched budget: optimum in {np.mean(sa):.0f}% of batches")
    xc = [r.xcheck for r in ok if not math.isnan(r.xcheck)]
    if xc:
        out.append(f"backend cross-check  max |P_aer - P_numpy| = {max(xc):.2e}")
    bad = [r for r in results if r.status != "ok"]
    if bad:
        out.append("incomplete: " + "; ".join(f"{r.name} ({r.status})" for r in bad[:5]))
    return "\n".join(out)


def write_csv(results: List[InstanceResult], path: str) -> None:
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow([name for name, _ in COLUMNS] + ["status"])
        for r in results:
            wr.writerow([_cell(r, name) for name, _ in COLUMNS] + [r.status])


# ###########################################################################
#  SECTION 12 -- COMMAND LINE
# ###########################################################################

def _parse_stacks(spec: str) -> Bay:
    return tuple(tuple(int(x) for x in part.replace(",", " ").split())
                 for part in spec.split(";"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="crp_qaoa.py",
        description="QAOA pipeline for the restricted Container Relocation Problem",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    s = ap.add_argument_group("instance source (choose one)")
    s.add_argument("--stacks", help="'1,6,3,8; 5,2,7; 4; 9'  (bottom -> top)")
    s.add_argument("--file"); s.add_argument("--dir")
    s.add_argument("--glob", default="*")
    s.add_argument("--random", type=int, metavar="K")
    p_ = ap.add_argument_group("instance parameters")
    p_.add_argument("--w", type=int); p_.add_argument("--h", type=int)
    p_.add_argument("--n", type=int); p_.add_argument("--name", default="instance")
    p_.add_argument("--published", type=int)
    p_.add_argument("--show", action="store_true", help="draw each bay")
    a_ = ap.add_argument_group("algorithm")
    a_.add_argument("-p", "--depth", type=int, default=3)
    a_.add_argument("--alpha", type=float, default=0.15,
                    help="CVaR tail; 1.0 = standard mean-energy QAOA")
    a_.add_argument("--restarts", type=int, default=16)
    a_.add_argument("--maxiter", type=int, default=400)
    a_.add_argument("--shots", type=int, default=4096)
    a_.add_argument("--penalty-a", type=float, default=2.0)
    a_.add_argument("--penalty-b", type=float, default=2.0)
    a_.add_argument("--capacity", choices=["slack", "strict"], default="slack")
    a_.add_argument("--naive-pair", action="store_true",
                    help="reinstate the double-counting mis-stack term")
    a_.add_argument("--max-qubits", type=int, default=20)
    b_ = ap.add_argument_group("backend")
    b_.add_argument("--backend", choices=["numpy", "aer"], default="numpy")
    b_.add_argument("--aer-shots", type=int, default=0,
                    help="Aer shot count per evaluation (0 = exact statevector)")
    b_.add_argument("--noise", type=float, default=0.0,
                    help="depolarising error per 2-qubit gate (implies aer)")
    b_.add_argument("--crosscheck", action="store_true",
                    help="report max |P_aer - P_numpy|")
    e_ = ap.add_argument_group("experiment")
    e_.add_argument("--seed", type=int, default=0)
    e_.add_argument("--repeats", type=int, default=1)
    e_.add_argument("--sweep-depth", type=int, nargs="+", metavar="P")
    e_.add_argument("--no-verify", action="store_true")
    e_.add_argument("--no-sa", action="store_true")
    e_.add_argument("--no-exact", action="store_true")
    e_.add_argument("--csv"); e_.add_argument("--quiet", action="store_true")
    return ap


def collect_instances(a) -> List[Instance]:
    if a.stacks:
        if a.h is None:
            sys.exit("--stacks requires --h")
        st = _parse_stacks(a.stacks)
        return [Instance(a.name, len(st), a.h, sum(len(x) for x in st), st,
                         published_opt=a.published)]
    if a.file:
        i = parse_instance_file(a.file); i.published_opt = a.published; return [i]
    if a.dir:
        paths = sorted(p for p in globmod.glob(os.path.join(a.dir, "*"))
                       if os.path.isfile(p) and fnmatch.fnmatch(os.path.basename(p), a.glob))
        if not paths:
            sys.exit(f"no files matching {a.glob!r} in {a.dir}")
        return [parse_instance_file(p) for p in paths]
    if a.random:
        if not (a.w and a.h and a.n):
            sys.exit("--random requires --w, --h and --n")
        return [random_instance(a.w, a.h, a.n, a.seed + i, f"rand-{a.w}-{a.h}-{i+1}")
                for i in range(a.random)]
    sys.exit("no instance source given (see --help)")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    instances = collect_instances(a)
    if a.show:
        for i in instances[:10]:
            print(f"\n{i.name}  (w={i.w}, h={i.h}, n={i.n})\n{i.render()}")
        print()

    backend = "aer" if (a.noise > 0 or a.backend == "aer") else "numpy"
    base = dict(alpha=a.alpha, restarts=a.restarts, maxiter=a.maxiter, shots=a.shots,
                A=a.penalty_a, B=a.penalty_b, capacity=a.capacity,
                corrected=not a.naive_pair, max_qubits=a.max_qubits,
                verify=not a.no_verify, sa_baseline=not a.no_sa,
                backend=backend, aer_shots=a.aer_shots, noise=a.noise,
                crosscheck=a.crosscheck)

    depths = a.sweep_depth or [a.depth]
    allr: List[InstanceResult] = []
    for p in depths:
        rows = []
        for inst in instances:
            for rep in range(a.repeats):
                cfg = Config(p=p, seed=a.seed + 1000 * rep, **base)
                r = solve_instance(inst, cfg, compute_exact=not a.no_exact)
                if a.repeats > 1:
                    r.name = f"{inst.name}#{rep+1}"
                rows.append(r)
        allr += rows
        if not a.quiet:
            print(format_table(rows,
                  f"backend={backend}  p={p}  alpha={a.alpha}  restarts={a.restarts}  "
                  f"shots={a.shots}  A={a.penalty_a} B={a.penalty_b}"
                  + (f"  noise={a.noise}" if a.noise else "")))
            print()

    if len(depths) > 1 and not a.quiet:
        print("DEPTH SWEEP SUMMARY")
        print(f"  {'p':>3} {'P_opt':>9} {'RndFeas':>9} {'ratio':>7} {'QAOA=Opt':>9}")
        k = len(allr) // len(depths)
        for i, p in enumerate(depths):
            c = allr[i * k:(i + 1) * k]
            po = np.array([r.p_opt for r in c if not math.isnan(r.p_opt)])
            bf = np.array([r.base_feas for r in c if not math.isnan(r.base_feas)])
            hit = [r.gap == 0 for r in c if r.gap is not None]
            if len(po):
                print(f"  {p:>3} {np.median(po):9.4f} {np.median(bf):9.4f} "
                      f"{np.median(po)/max(np.median(bf),1e-12):6.2f}x {np.mean(hit):8.0%}")
        print()

    if a.csv:
        write_csv(allr, a.csv)
        print(f"wrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
