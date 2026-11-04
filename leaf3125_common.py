# Self-contained helpers for the 3125-sequence leaf-value planner. Uses only simutils (environment
# rules, read-only). It does not use the 10-tool library or any other planner file.
import itertools
import numpy as np
from simutils.config import GRID_SIZE as N, NUM_CARS, START_POS, GOAL_POS, WINDOW_HALF_SIZE
from simutils.env_utils import compute_window_bounds, _swept_positions_collide

START = tuple(START_POS)
GOAL = tuple(GOAL_POS)
PRIM = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]          # W (wait), U, D, L, R
ALL = {"".join("WUDLR"[PRIM.index(a)] for a in seq): list(seq)
       for seq in itertools.product(PRIM, repeat=5)}          # all 3125 five-step sequences
ALL_NAMES = list(ALL)

def bounds_of(agent):
    return compute_window_bounds(agent, WINDOW_HALF_SIZE, N)

def in_window(p, b):
    return b.row_min <= p[0] <= b.row_max and b.col_min <= p[1] <= b.col_max

def step(agent, cars, a):
    # same rules as MovingWorldEnv.advance_one_timestep: cars move up and wrap, agent holds at edges
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
    goal = (not coll) and agent_new == GOAL
    return agent_new, new_cars, coll, goal

def run_tool(agent, cars, tool):
    coll = goal = False
    for a in tool:
        agent, cars, coll, goal = step(agent, cars, a)
        if coll or goal:
            break
    return agent, cars, goal, coll

def replay_collides(agent, vis, tool):
    _, _, goal, coll = run_tool(agent, list(vis), tool)
    return coll
