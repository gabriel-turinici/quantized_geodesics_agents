"""Flat Monte Carlo planner for the grid environment (reference policy, no LLM).

The planner chooses one of the tools of ``quantized_actions.npz`` (the quantized
library plus the one-step primitives, 10 tools in total) at each decision, using
the same observations, the same tool library and the same collision filter as the
LLM pipeline. It is a reference point for the LLM results, not a method of the paper.

Decision procedure
------------------
1. Safe tools. A tool is removed if it would hit a car that is visible in the
   observation window at decision time (``replay_collides``). If no tool survives,
   the agent waits (``WWWWW``); if exactly one survives, it is taken.
2. Sampled worlds. For every safe tool, ``n_part`` hypothetical configurations of
   the unobserved cars are drawn: each unobserved cell receives a car with
   probability equal to the density of cars seen in the window (visible cars divided
   by the number of window cells). The visible cars keep their observed positions.
3. Rollouts. In each sampled world the candidate tool is executed for five steps.
   If the goal is not reached and no collision occurs, a random policy continues for
   up to ``horizon`` further decisions. At each of these decisions it chooses uniformly
   among the tools that pass the visible-car filter inside that sampled world.
   The score of a candidate is the fraction of sampled worlds in which the goal is
   reached.
4. The candidate with the highest score is chosen (random tie-break) and executed
   in the real environment. The next decision is taken from the new observation.

Information used
----------------
- Observation: the agent position and the cars inside the observation window. The
  decision code receives a view object with ``agent_pos`` and ``cars`` (``ObsView``),
  so the decision code never reads cars outside the window.
- Dynamics: the rules stated in the LLM prompt. Cars move up one row per step and wrap
  around the grid; the agent is blocked by the grid edges. ``step`` reproduces
  ``MovingWorldEnv.advance_one_timestep``; ``validate`` checks the agreement on random
  action sequences (6,565 steps in the reported check).
- Goal: the exact goal cell (``GOAL_POS``). The LLM has to find the goal from visual
  cues, so this knowledge favours the planner.
- Not used: positions of unobserved cars, learning across episodes, and any tuned
  parameter beyond the ones listed below.

Parameters
----------
``n_part``: sampled worlds per candidate tool (200 in the reported runs).
``horizon``: rollout decisions after the first one (8 in the reported runs).
``MAX_DECISIONS``: decisions per episode (30, as for the LLM runs).

Command line
------------
``python3 mc_pilot.py validate``
    Checks ``step`` against ``MovingWorldEnv`` on random action sequences.
``python3 mc_pilot.py run FIRST END N_PART HORIZON``
    Runs the episodes with seeds FIRST..END-1 and prints one line per episode.
    It uses one random stream shared by all episodes, so results depend on the
    episode order. The reported results were produced by ``run_pilot_logged.py``,
    which seeds one stream per episode.
"""
import sys
import time
import random
import numpy as np
from simutils.env import MovingWorldEnv
from simutils.env_utils import compute_window_bounds, _swept_positions_collide
from simutils.config import GRID_SIZE, NUM_CARS, START_POS, GOAL_POS, WINDOW_HALF_SIZE

N = GRID_SIZE
MAX_DECISIONS = 30
TOOLS = {k: [tuple(int(x) for x in a) for a in v] for k, v in np.load('quantized_actions.npz').items()}
TOOL_NAMES = list(TOOLS)


class ObsView:
    """The planner's only view of the world: the agent position and the cars inside
    the observation window at the current time."""

    def __init__(self, env):
        b = bounds_of(env.agent_pos)
        self.agent_pos = env.agent_pos
        self.cars = [_Pos(tuple(c.position)) for c in env.cars if in_window(c.position, b)]


class _Pos:
    def __init__(self, position):
        self.position = position


def bounds_of(agent):
    """Observation window around ``agent`` (clipped to the grid by the environment helper)."""
    return compute_window_bounds(agent, WINDOW_HALF_SIZE, N)


def in_window(p, b):
    """True if cell ``p`` lies inside the window ``b``."""
    return b.row_min <= p[0] <= b.row_max and b.col_min <= p[1] <= b.col_max


def step(agent, cars, a):
    """One timestep, with the rules of ``MovingWorldEnv.advance_one_timestep``.

    The agent moves by ``a`` unless the move leaves the grid. Every car moves up one
    row and wraps around. Returns ``(agent, cars, collision, goal)``.
    """
    ar, ac = agent
    cand = (ar + a[0], ac + a[1])
    agent_new = cand if (0 <= cand[0] < N and 0 <= cand[1] < N) else agent
    new_cars, coll = [], False
    for (r, c) in cars:
        unr = (r - 1, c)
        newp = ((r - 1) % N, c)
        new_cars.append(newp)
        if newp == agent_new or _swept_positions_collide(agent, agent_new, (r, c), unr):
            coll = True
    goal = (not coll) and agent_new == GOAL_POS
    return agent_new, new_cars, coll, goal


def run_tool(agent, cars, tool):
    """Executes the five steps of ``tool``, stopping at a collision or at the goal.
    Returns ``(agent, cars, goal, collision)``."""
    coll = goal = False
    for a in tool:
        agent, cars, coll, goal = step(agent, cars, a)
        if coll or goal:
            break
    return agent, cars, goal, coll


def replay_collides(agent, vis, tool):
    """True if ``tool`` would collide with one of the cars in ``vis`` (the visible cars)."""
    _, _, goal, coll = run_tool(agent, list(vis), tool)
    return coll


def rollout(agent, cars, decisions, rng):
    """Random rollout: returns 1 if the goal is reached within ``decisions`` decisions,
    0 on a collision or when the decisions run out. At each decision a tool is drawn
    uniformly among those that pass the visible-car filter (``WWWWW`` if none does)."""
    for _ in range(decisions):
        b = bounds_of(agent)
        vis = [p for p in cars if in_window(p, b)]
        safe = [t for t in TOOL_NAMES if not replay_collides(agent, vis, TOOLS[t])]
        tool = rng.choice(safe) if safe else 'WWWWW'
        agent, cars, goal, coll = run_tool(agent, cars, TOOLS[tool])
        if goal:
            return 1
        if coll:
            return 0
    return 0


def choose(view, rng, n_part, horizon, nprng):
    """Chooses a tool from the observation ``view`` (an object with ``agent_pos`` and
    ``cars``; only the cars inside the window are used).

    Returns ``(tool_name, kind)`` where ``kind`` is ``'forced'`` (no safe tool, the agent
    waits), ``'single'`` (one safe tool) or ``'search'`` (Monte Carlo choice).
    """
    agent = view.agent_pos
    b = bounds_of(agent)
    vis = [tuple(c.position) for c in view.cars if in_window(c.position, b)]
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
        total = 0
        for _ in range(n_part):
            k = nprng.binomial(len(grey), d)
            idx = nprng.choice(len(grey), size=k, replace=False) if k > 0 else []
            cars = vis + [grey[i] for i in idx]
            a2, c2, goal, coll = run_tool(agent, cars, TOOLS[t])
            if goal:
                total += 1
            elif not coll:
                total += rollout(a2, c2, horizon, rng)
        values[t] = total / n_part
    best = max(values.values())
    cands = [t for t, v in values.items() if v == best]
    return rng.choice(cands), 'search'


def run_episode(seed, n_part, horizon, rng, nprng):
    """Runs one episode in ``MovingWorldEnv`` with the given seed. The planner receives
    only ``ObsView(env)``. Returns ``(outcome, turns, counts_of_decision_kinds)``."""
    env = MovingWorldEnv(size=N, num_cars=NUM_CARS, start_pos=START_POS, goal_pos=GOAL_POS, seed=seed)
    kinds = {}
    for _ in range(MAX_DECISIONS):
        if env.is_terminal():
            break
        tool, kind = choose(ObsView(env), rng, n_part, horizon, nprng)
        kinds[kind] = kinds.get(kind, 0) + 1
        env.execute_turn([(a[0], a[1], 1) for a in TOOLS[tool]])
    if env.outcome is None:
        env.mark_failed()
    return env.outcome, env.turn_count, kinds


def validate(trials=300):
    """Checks that ``step`` gives the same positions, collisions and goal flags as
    ``MovingWorldEnv.advance_one_timestep`` on random action sequences. Returns the number
    of steps compared; raises AssertionError on the first mismatch."""
    rng = random.Random(0)
    checked = 0
    for s in range(trials):
        env = MovingWorldEnv(size=N, num_cars=NUM_CARS, start_pos=START_POS, goal_pos=GOAL_POS, seed=1000 + s)
        agent = env.agent_pos
        cars = [tuple(c.position) for c in env.cars]
        for _ in range(60):
            a = rng.choice([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)])
            res = env.advance_one_timestep(a)
            agent, cars, coll, goal = step(agent, cars, a)
            assert res.agent_pos == agent, (s, res.agent_pos, agent)
            assert [tuple(p) for p in res.car_positions] == cars, s
            assert res.collision == coll and res.goal_reached == goal, s
            checked += 1
            if res.terminal:
                break
    return checked


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'validate':
        print('validated steps:', validate())
    else:
        seeds = range(int(sys.argv[2]), int(sys.argv[3]))
        n_part, horizon = int(sys.argv[4]), int(sys.argv[5])
        rng = random.Random(7)
        nprng = np.random.default_rng(7)
        out = []
        for s in seeds:
            t0 = time.time()
            outcome, turns, kinds = run_episode(s, n_part, horizon, rng, nprng)
            print(f"seed {s}: {outcome} turns={turns} kinds={kinds} time={time.time()-t0:.1f}s", flush=True)
            out.append(outcome)
        print('SUMMARY goal', out.count('goal'), 'of', len(out), flush=True)
