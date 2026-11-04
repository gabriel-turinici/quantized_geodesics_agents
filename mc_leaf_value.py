# Flat Monte Carlo planner over the 10-tool library, with a leaf value at the end of each rollout:
# if a rollout ends without reaching the goal and without a collision, its score is V[agent cell],
# the offline unconditional value table (value_table.npy, built by leaf_value_table.py).
# Usage:
#   python3 mc_leaf_value.py run <seed_from> <seed_to> <n_particles> <horizon> <leaf|noleaf>
#   (seed_to exclusive)
import sys, time, random, datetime
from collections import Counter
import numpy as np
from leaf_common import *

LEAF_VALUE = np.load(os.path.join(HERE, 'value_table.npy'))
USE_LEAF = True
MAX_DECISIONS = 30

def leaf_value(agent):
    return float(LEAF_VALUE[agent[0], agent[1]]) if USE_LEAF else 0.0

def new_stats():
    return {'rollouts': 0}

def rollout(agent, cars, decisions, rng, stats):
    stats['rollouts'] += 1
    a, cs = agent, list(cars)
    for _ in range(decisions):
        b = bounds_of(a)
        vis = [p for p in cs if in_window(p, b)]
        safe = [t for t in TOOL_NAMES if not replay_collides(a, vis, TOOLS[t])]
        tool = rng.choice(safe) if safe else 'WWWWW'
        a, cs, goal, coll = run_tool(a, cs, TOOLS[tool])
        if goal:
            return 1.0
        if coll:
            return 0.0
    return leaf_value(a)

def choose(env, rng, n_part, horizon, nprng, stats, info):
    agent = env.agent_pos
    b = bounds_of(agent)
    vis = [tuple(c.position) for c in env.cars if in_window(c.position, b)]   # observation only
    safe = [t for t in TOOL_NAMES if not replay_collides(agent, vis, TOOLS[t])]
    if not safe:
        return 'WWWWW', 'forced'
    if len(safe) == 1:
        return safe[0], 'single'
    grey = [(r, c) for r in range(N) for c in range(N) if not in_window((r, c), b)]
    window_cells = (b.row_max - b.row_min + 1) * (b.col_max - b.col_min + 1)
    d = len(vis) / window_cells
    values = {}
    for t in safe:
        total = 0.0
        for _ in range(n_part):
            k = nprng.binomial(len(grey), d)
            idx = nprng.choice(len(grey), size=k, replace=False) if k > 0 else []
            cars = vis + [grey[i] for i in idx]
            a2, c2, goal, coll = run_tool(agent, cars, TOOLS[t])
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
    print(f"# leaf-value planner log, {datetime.datetime.now().isoformat(timespec='seconds')}")
    print(f"# seeds {s0}..{s1 - 1} (seed_to exclusive)")
    print(f"# N_PARTICLES {n_part}   HORIZON {horizon}   LEAF_VALUE {'on' if USE_LEAF else 'off'}")
    print(f"# MAX_DECISIONS {MAX_DECISIONS}   tool set size {len(TOOL_NAMES)} (10-tool library)")
    print(f"# value table: V(start {START}) = {LEAF_VALUE[START]:.4f}, V(goal {GOAL}) = {LEAF_VALUE[GOAL]:.1f}")
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
        print(f"seed {seed} dec {d:2d} agent {agent} dist {manhattan(agent, GOAL)} V {LEAF_VALUE[agent]:.3f} "
              f"tool {tool} {detail} time {dt:.1f}s", flush=True)
        env.execute_turn([(a[0], a[1], 1) for a in TOOLS[tool]])
    if env.outcome is None:
        env.mark_failed()
    print(f"seed {seed} END outcome {env.outcome} turns {env.turn_count} final agent {env.agent_pos} "
          f"dist {manhattan(env.agent_pos, GOAL)} time {time.time() - t_ep:.1f}s", flush=True)
    return env.outcome

if __name__ == '__main__':
    if len(sys.argv) < 7 or sys.argv[1] != 'run':
        sys.exit('usage: python3 mc_leaf_value.py run <seed_from> <seed_to> <n_particles> <horizon> <leaf|noleaf>')
    s0, s1, n_part, horizon = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    USE_LEAF = sys.argv[6] == 'leaf'
    header(s0, s1, n_part, horizon)
    rng = random.Random(7); nprng = np.random.default_rng(7); stats = new_stats()
    out = []
    for s in range(s0, s1):
        out.append(episode(s, n_part, horizon, rng, nprng, stats))
    print('SUMMARY leaf-value', 'on' if USE_LEAF else 'off', 'goal', out.count('goal'),
          'collision', out.count('collision'), 'failed', out.count('failed'), 'of', len(out), flush=True)
