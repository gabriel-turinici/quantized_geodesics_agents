# Offline unconditional value table V(cell) = probability of reaching the goal from a cell,
# averaged over car layouts (sampled uniformly), by Bellman value iteration over the 25x25 cells
# with the 10-tool library. Writes value_table.npy (25x25) and checks the vectorised outcome
# computation against the scalar step on the same sampled layouts.
import sys, time
import numpy as np
from leaf_common import *

W = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
rng = np.random.default_rng(12345)
R = rng.integers(0, N, size=(W, NUM_CARS))   # car rows at time 0 (stored positions)
C = rng.integers(0, N, size=(W, NUM_CARS))   # car columns at time 0
cells = [(r, c) for r in range(N) for c in range(N)]
idx = {cell: i for i, cell in enumerate(cells)}

def interval(num, den):
    # 0 <= num/den < 1 for an integer array num and an integer scalar den (den != 0)
    if den < 0:
        num, den = -num, -den
    return (num >= 0) & (num < den)

def swept_vec(a_old, a_new, br, bc):
    # vectorised _swept_positions_collide: the car's stored position is (br, bc); its unwrapped
    # next position is (br - 1, bc), since cars move one cell up per step.
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

def outcomes(cell, tool):
    """status per sampled world: 0 collision, 1 goal, 2 survived the tool; plus the agent end cell."""
    status = np.full(W, 2)
    alive = np.ones(W, dtype=bool)
    a = cell
    for k in range(5):
        d = tool[k]
        cand = (a[0] + d[0], a[1] + d[1])
        a_new = cand if (0 <= cand[0] < N and 0 <= cand[1] < N) else a
        br = (R - k) % N
        bc = C
        nr = (br - 1) % N
        hit = ((nr == a_new[0]) & (bc == a_new[1])) | swept_vec(a, a_new, br, bc)
        coll = hit.any(axis=1)
        goal_hit = (~coll) & (a_new == GOAL)
        newly = alive & (coll | goal_hit)
        status[newly & coll] = 0
        status[newly & goal_hit] = 1
        alive = alive & ~newly
        a = a_new
    return status, a

def build():
    t0 = time.time()
    T = len(TOOL_NAMES)
    G = np.zeros((len(cells), T)); S = np.zeros((len(cells), T)); E = np.zeros((len(cells), T), dtype=int)
    for ci, cell in enumerate(cells):
        for ti, name in enumerate(TOOL_NAMES):
            status, end = outcomes(cell, TOOLS[name])
            G[ci, ti] = np.mean(status == 1)
            S[ci, ti] = np.mean(status == 2)
            E[ci, ti] = idx[end]
    print(f"outcome tables built in {time.time() - t0:.1f}s for {W} layouts")
    return G, S, E

def check(G, S, E, trials=40):
    # the vectorised outcomes must equal the scalar step on the same layouts (exact agreement)
    rs = np.random.default_rng(7)
    bad = 0
    for _ in range(trials):
        cell = cells[rs.integers(len(cells))]
        name = TOOL_NAMES[rs.integers(len(TOOL_NAMES))]
        ci = idx[cell]; ti = TOOL_NAMES.index(name)
        g = s = 0
        for w in range(W):
            cars = [(int(R[w, j]), int(C[w, j])) for j in range(NUM_CARS)]
            agent = cell
            out = 2
            for a in TOOLS[name]:
                agent, cars, coll, goal = step(agent, cars, a)
                if coll or goal:
                    out = 0 if coll else 1
                    break
            g += (out == 1); s += (out == 2)
        if abs(g / W - G[ci, ti]) > 1e-12 or abs(s / W - S[ci, ti]) > 1e-12:
            bad += 1
    print(f"check vs scalar step on {trials} random (cell, tool) pairs: mismatches {bad}")

def value_iteration(G, S, E, tol=1e-9, max_sweeps=1000):
    V = np.zeros(len(cells))
    V[idx[GOAL]] = 1.0
    for sweep in range(1, max_sweeps + 1):
        Q = G + S * V[E]
        Vn = Q.max(axis=1)
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
    np.save(os.path.join(HERE, 'value_table.npy'), Vm)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(Vm, origin='upper', cmap='viridis', vmin=0, vmax=1)   # row 0 at the top, as in the grid
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('estimated probability of reaching the goal')
        ax.scatter([START[1]], [START[0]], marker='s', s=90, facecolors='none', edgecolors='white', linewidths=2, label='start')
        ax.scatter([GOAL[1]], [GOAL[0]], marker='*', s=220, color='red', edgecolors='white', label='goal')
        ax.text(START[1] + 0.8, START[0] + 0.8, f"{Vm[START]:.3f}", color='white', fontsize=9)
        ax.set_xlabel('column')
        ax.set_ylabel('row')
        ax.set_title(f'Offline Bellman value table: {sweeps} sweeps, {W} sampled car layouts')
        ax.legend(loc='upper left', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, 'value_table.png'), dpi=150)
        plt.close(fig)
        print('plot written: value_table.png')
    except ImportError:
        print('matplotlib not available: plot skipped')
    print(f"value iteration: {sweeps} sweeps, last change {diff:.2e}")
    print("V at start", START, "=", round(float(Vm[START]), 4), " V at goal", GOAL, "=", Vm[GOAL])
    print("V along the top row:", [round(float(x), 3) for x in Vm[0, ::4]])
    print("V along the main diagonal:", [round(float(Vm[i, i]), 3) for i in range(0, N, 4)])
