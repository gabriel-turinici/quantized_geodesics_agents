# Offline per-step value table: V(cell) by Bellman value iteration over the 5 primitive moves
# (wait, up, down, left, right), car layouts sampled uniformly (500 layouts). Self-contained: uses
# only leaf3125_common.py and simutils. Writes value_table_step.npy and value_table_step.png.
# Usage: python3 leaf3125_step_table.py
import os, time
import numpy as np
from leaf3125_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
W = 500
MOVES = PRIM                      # the 5 primitive moves
rng = np.random.default_rng(12345)
R = rng.integers(0, N, size=(W, NUM_CARS))   # car rows (stored positions) at the decision time
C = rng.integers(0, N, size=(W, NUM_CARS))   # car columns
cells = [(r, c) for r in range(N) for c in range(N)]
idx = {cell: i for i, cell in enumerate(cells)}

def interval(num, den):
    if den < 0:
        num, den = -num, -den
    return (num >= 0) & (num < den)

def swept_vec(a_old, a_new, br, bc):
    # vectorised _swept_positions_collide; car stored position (br, bc), unwrapped next (br - 1, bc)
    r0 = a_old[0] - br
    c0 = a_old[1] - bc
    vr = (a_new[0] - a_old[0]) + 1
    vc = (a_new[1] - a_old[1])
    if vr == 0 and vc == 0:
        return (r0 == 0) & (c0 == 0)
    if vr == 0:
        return (r0 == 0) & interval(-c0, vc)
    if vc == 0:
        return (c0 == 0) & interval(-r0, vr)
    return ((r0 * vc) == (c0 * vr)) & interval(-r0, vr)

def one_step(cell, move):
    # status per sampled layout after one move: 0 collision, 1 goal, 2 survived; plus the next cell
    cand = (cell[0] + move[0], cell[1] + move[1])
    a_new = cand if (0 <= cand[0] < N and 0 <= cand[1] < N) else cell
    nr = (R - 1) % N
    hit = ((nr == a_new[0]) & (C == a_new[1])) | swept_vec(cell, a_new, R, C)
    coll = hit.any(axis=1)
    goal = (~coll) & (a_new == GOAL)
    status = np.full(W, 2)
    status[coll] = 0
    status[goal] = 1
    return status, a_new

def build():
    t0 = time.time()
    G = np.zeros((len(cells), len(MOVES))); S = np.zeros_like(G); E = np.zeros((len(cells), len(MOVES)), dtype=int)
    for ci, cell in enumerate(cells):
        for mi, move in enumerate(MOVES):
            status, end = one_step(cell, move)
            G[ci, mi] = np.mean(status == 1)
            S[ci, mi] = np.mean(status == 2)
            E[ci, mi] = idx[end]
    print(f"one-step tables built in {time.time() - t0:.2f}s")
    return G, S, E

def check(G, S, E, trials=200):
    rs = np.random.default_rng(7)
    bad = 0
    for _ in range(trials):
        cell = cells[rs.integers(len(cells))]
        mi = rs.integers(len(MOVES))
        ci = idx[cell]
        g = s = 0
        for w in range(W):
            cars = [(int(R[w, j]), int(C[w, j])) for j in range(NUM_CARS)]
            agent, cars, coll, goal = step(cell, cars, MOVES[mi])
            g += (goal and not coll); s += (not coll and not goal)
        if abs(g / W - G[ci, mi]) > 1e-12 or abs(s / W - S[ci, mi]) > 1e-12:
            bad += 1
    print(f"check vs scalar step on {trials} random (cell, move) pairs: mismatches {bad}")

def value_iteration(G, S, E, tol=1e-9, max_sweeps=5000):
    V = np.zeros(len(cells))
    V[idx[GOAL]] = 1.0
    for sweep in range(1, max_sweeps + 1):
        Vn = (G + S * V[E]).max(axis=1)
        Vn[idx[GOAL]] = 1.0
        diff = np.abs(Vn - V).max()
        V = Vn
        if diff < tol:
            break
    return V, sweep, diff

if __name__ == '__main__':
    G, S, E = build()
    check(G, S, E)
    V, sweeps, diff = value_iteration(G, S, E)
    Vm = V.reshape(N, N)
    np.save(os.path.join(HERE, 'value_table_step.npy'), Vm)
    print(f"value iteration: {sweeps} sweeps, last change {diff:.2e}")
    print("V at start", START, "=", round(float(Vm[START]), 4), " V at goal", GOAL, "=", Vm[GOAL])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Vm, origin='upper', cmap='viridis', vmin=0, vmax=1)
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('estimated probability of reaching the goal')
        ax.scatter([START[1]], [START[0]], marker='s', s=90, facecolors='none', edgecolors='white', linewidths=2, label='start')
        ax.scatter([GOAL[1]], [GOAL[0]], marker='*', s=220, color='red', edgecolors='white', label='goal')
        ax.text(START[1] + 0.8, START[0] + 0.8, f"{Vm[START]:.3f}", color='white', fontsize=9)
        ax.set_xlabel('column')
        ax.set_ylabel('row')
        ax.set_title(f'Offline Bellman value table, per-step moves: {sweeps} sweeps, {W} layouts')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, 'value_table_step.png'), dpi=150)
        plt.close(fig)
        print('plot written: value_table_step.png')
    except ImportError:
        print('matplotlib not available: plot skipped')
