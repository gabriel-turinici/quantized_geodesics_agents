"""Audit: planner decisions do not depend on cars outside the observation window.

For the first decisions of one episode (seed 1), the planner is run twice with the same
random streams: once with the full environment and once with the observation view
(``ObsView``, which holds only the cars inside the window). The chosen tools must be
identical. Run from the repository root:
    python3 audit_obs_only.py
"""
import random
import numpy as np
import mc_pilot as mp
from mc_pilot import ObsView
from simutils.env import MovingWorldEnv
from simutils.config import NUM_CARS, START_POS, GOAL_POS


seed = 1
env = MovingWorldEnv(size=mp.N, num_cars=NUM_CARS, start_pos=START_POS, goal_pos=GOAL_POS, seed=seed)
same = 0
for k in range(5):
    if env.is_terminal():
        break
    r1, n1 = random.Random(100 + k), np.random.default_rng(100 + k)
    r2, n2 = random.Random(100 + k), np.random.default_rng(100 + k)
    t_full, _ = mp.choose(env, r1, 200, 8, n1)
    t_obs, _ = mp.choose(ObsView(env), r2, 200, 8, n2)
    print(f"decision {k+1}: full-env tool {t_full}  observation-only tool {t_obs}  identical={t_full == t_obs}")
    same += int(t_full == t_obs)
    env.execute_turn([(a[0], a[1], 1) for a in mp.TOOLS[t_full]])
print("identical decisions:", same)
