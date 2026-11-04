# Self-contained flat Monte Carlo planner over the 3125 five-step sequences, with a leaf value at the
# end of each rollout (leaf = V[agent cell] from value_table_step.npy, built by leaf3125_step_table.py).
# Uses only leaf3125_common.py and simutils. Rollouts: redraw with REDRAW_TRIES=500 and an exact
# fallback over all sequences when the redraws fail.
# Usage: python3 mc_leaf3125.py run <seed_from> <seed_to> <n_particles> <horizon> <leaf|noleaf>   (seed_to exclusive)
import sys, os, time, random, datetime
from collections import Counter
import numpy as np
from simutils.env import MovingWorldEnv
from simutils.config import NUM_CARS
from leaf3125_common import *

HERE = os.path.dirname(os.path.abspath(__file__))
VT = np.load(os.path.join(HERE, 'value_table_step.npy'))
USE_LEAF = True
REDRAW_TRIES = 500
MAX_DECISIONS = 30

def leaf_value(agent):
    return float(VT[agent[0], agent[1]]) if USE_LEAF else 0.0

def new_stats():
    return {'rollouts': 0, 'checks': 0, 'rejections': 0, 'exact_fallbacks': 0}

def visible(agent, cars):
    b = bounds_of(agent)
    return [p for p in cars if in_window(p, b)]

def rollout(agent, cars, decisions, rng, stats):
    stats['rollouts'] += 1
    a, cs = agent, list(cars)
    for _ in range(decisions):
        vis = visible(a, cs)
        tool = None
        for _ in range(REDRAW_TRIES):
            cand = rng.choice(ALL_NAMES)
            stats['checks'] += 1
            if not replay_collides(a, vis, ALL[cand]):
                tool = cand
                break
            stats['rejections'] += 1
        if tool is None:
            stats['exact_fallbacks'] += 1
            safe = [c for c in ALL_NAMES if not replay_collides(a, vis, ALL[c])]
            stats['checks'] += len(ALL_NAMES)
            tool = rng.choice(safe) if safe else 'WWWWW'
        a, cs, goal, coll = run_tool(a, cs, ALL[tool])
        if goal:
            return 1.0
        if coll:
            return 0.0
    return leaf_value(a)

def choose(env, rng, n_part, horizon, nprng, stats, info):
    agent = env.agent_pos
    vis = [tuple(c.position) for c in env.cars if in_window(c.position, bounds_of(agent))]   # observation only
    safe = [t for t in ALL_NAMES if not replay_collides(agent, vis, ALL[t])]
    if not safe:
        return 'WWWWW', 'forced'
    if len(safe) == 1:
        return safe[0], 'single'
    b = bounds_of(agent)
    grey = [(r, c) for r in range(N) for c in range(N) if not in_window((r, c), b)]
    window_cells = (b.row_max - b.row_min + 1) * (b.col_max - b.col_min + 1)
    d = len(vis) / window_cells
    values = {}
    for t in safe:
        total = 0.0
        for _ in range(n_part):
            k = nprng.binomial(len(grey), d)
            idx_ = nprng.choice(len(grey), size=k, replace=False) if k > 0 else []
            cars = vis + [grey[i] for i in idx_]
            a2, c2, goal, coll = run_tool(agent, cars, ALL[t])
            if goal:
                total += 1
            elif not coll:
                total += rollout(a2, c2, horizon, rng, stats)
        values[t] = total / n_part
    best = max(values.values())
    cands = [t for t, v in values.items() if v == best]
    info.update(values=values, n_safe=len(safe), n_tied=len(cands))
    return rng.choice(cands), 'search'

def header(s0, s1, n_part, horizon):
    print(f"# self-contained 3125 leaf-value planner log, {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"# seeds {s0}..{s1 - 1} (seed_to exclusive)")
    print(f"# N_PARTICLES {n_part}   HORIZON {horizon}   LEAF_VALUE {'on' if USE_LEAF else 'off'}   REDRAW_TRIES {REDRAW_TRIES}")
    print(f"# MAX_DECISIONS {MAX_DECISIONS}   tool set size {len(ALL_NAMES)} (3125 sequences)")
    print(f"# value table value_table_step.npy: V(start {START}) = {VT[START]:.4f}, V(goal {GOAL}) = {VT[GOAL]:.1f}")
    print(f"# START_POS {START}   GOAL_POS {GOAL}", flush=True)

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def episode(seed, n_part, horizon, rng, nprng, stats):
    env = MovingWorldEnv(size=N, num_cars=NUM_CARS, start_pos=START_POS, goal_pos=GOAL_POS, seed=seed)
    print(f"seed {seed} start agent {env.agent_pos} dist {manhattan(env.agent_pos, GOAL)} cars {len(env.cars)}", flush=True)
    t_ep = time.time()
    for d in range(1, MAX_DECISIONS + 1):
        if env.is_terminal():
            break
        agent = env.agent_pos
        before = dict(stats)
        info = {}
        t0 = time.time()
        tool, kind = choose(env, rng, n_part, horizon, nprng, stats, info)
        dt = time.time() - t0
        vals = info.get('values', {})
        if vals:
            best = max(vals.values())
            zeros = sum(1 for v in vals.values() if v == 0)
            mean = sum(vals.values()) / len(vals)
            top = Counter(round(v, 3) for v in vals.values()).most_common(3)
            detail = (f"best {best:.3f} ties {info['n_tied']} zeros {zeros}/{info['n_safe']} "
                      f"mean {mean:.3f} chosen_value {vals[tool]:.3f} top_values {top}")
        else:
            detail = f"kind {kind}"
        print(f"seed {seed} dec {d:2d} agent {agent} dist {manhattan(agent, GOAL)} V {VT[agent]:.3f} "
              f"tool {tool} {detail} checks {stats['checks'] - before['checks']} "
              f"rejections {stats['rejections'] - before['rejections']} "
              f"exact_fallbacks {stats['exact_fallbacks'] - before['exact_fallbacks']} time {dt:.1f}s", flush=True)
        env.execute_turn([(a[0], a[1], 1) for a in ALL[tool]])
    if env.outcome is None:
        env.mark_failed()
    print(f"seed {seed} END outcome {env.outcome} turns {env.turn_count} final agent {env.agent_pos} "
          f"dist {manhattan(env.agent_pos, GOAL)} time {time.time() - t_ep:.1f}s", flush=True)
    return env.outcome

if __name__ == '__main__':
    if len(sys.argv) < 7 or sys.argv[1] != 'run':
        sys.exit('usage: python3 mc_leaf3125.py run <seed_from> <seed_to> <n_particles> <horizon> <leaf|noleaf>')
    s0, s1, n_part, horizon = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    USE_LEAF = sys.argv[6] == 'leaf'
    header(s0, s1, n_part, horizon)
    rng = random.Random(7); nprng = np.random.default_rng(7); stats = new_stats()
    out = []
    for s in range(s0, s1):
        out.append(episode(s, n_part, horizon, rng, nprng, stats))
    print('SUMMARY leaf3125', 'on' if USE_LEAF else 'off', 'goal', out.count('goal'),
          'collision', out.count('collision'), 'failed', out.count('failed'), 'of', len(out), flush=True)
