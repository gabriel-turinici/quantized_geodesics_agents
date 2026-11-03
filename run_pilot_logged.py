"""Logged runs of the Monte Carlo planner (``mc_pilot.py``), with one figure and one text log per episode.

Usage (from the repository root, where ``quantized_actions.npz`` and ``simutils/`` are):
    python3 run_pilot_logged.py FIRST END N_PART HORIZON
    e.g. python3 run_pilot_logged.py 1 101 200 8   (the 100 episodes reported in the paper)

Seeds: episode ``seed`` is the environment seed and the seed of its random streams
(``seed * 7 + 1``), so each episode can be re-run on its own and runs can be split
across calls. The planner settings are those of the paper: 200 sampled worlds per
candidate tool, a rollout horizon of 8 decisions, and at most 30 decisions per episode.

Outputs, written in the current folder:
    episode_SS.png   one panel per decision: the full grid with the true car positions
                     (red = visible to the planner, orange outline = hidden), the agent
                     path (blue), the goal (green) and the unobserved area (grey).
    episode_SS.txt   one line per decision: agent position, chosen tool, how it was chosen,
                     and the number of visible and total cars.
    pilot_logged_summary.txt   one line per episode: seed, outcome, number of decisions
                     (lines are appended, so run the same range only once).

The planner receives only ``ObsView(env)`` (own position and cars in the window).
"""
import sys
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
sys.path.insert(0, ".")
import mc_pilot as mp
from simutils.env import MovingWorldEnv
from simutils.config import NUM_CARS, START_POS, GOAL_POS

N = mp.N

from mc_pilot import ObsView

def run_logged(seed, n_part, horizon):
    rng = random.Random(seed * 7 + 1)
    nprng = np.random.default_rng(seed * 7 + 1)
    env = MovingWorldEnv(size=N, num_cars=NUM_CARS, start_pos=START_POS, goal_pos=GOAL_POS, seed=seed)
    frames = []
    for k in range(mp.MAX_DECISIONS):
        if env.is_terminal():
            break
        agent = env.agent_pos
        b = mp.bounds_of(agent)
        cars = [tuple(c.position) for c in env.cars]
        tool, kind = mp.choose(ObsView(env), rng, n_part, horizon, nprng)
        frames.append(dict(k=k, agent=agent, cars=cars,
                           bounds=(b.row_min, b.row_max, b.col_min, b.col_max), tool=tool, kind=kind))
        env.execute_turn([(a[0], a[1], 1) for a in mp.TOOLS[tool]])
    if env.outcome is None:
        env.mark_failed()
    frames.append(dict(k=len(frames), agent=env.agent_pos, cars=[tuple(c.position) for c in env.cars],
                       bounds=None, tool=None, kind="end"))
    return env.outcome, frames

def draw_episode(seed, outcome, frames, path):
    n = len(frames)
    ncols = 6
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.3 * ncols, 2.4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    trail = []
    for i, ax in enumerate(axes):
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-0.5, N - 0.5); ax.set_ylim(N - 0.5, -0.5)
        ax.set_facecolor("white")
        if i >= n:
            ax.axis("off")
            continue
        f = frames[i]
        trail.append(f["agent"])
        ax.add_patch(Rectangle((-0.5, -0.5), N, N, color="0.55", zorder=0))
        if f["bounds"] is not None:
            r0, r1, c0, c1 = f["bounds"]
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1, color="white", zorder=1))
        for (r, c) in f["cars"]:
            visible = f["bounds"] is not None and f["bounds"][0] <= r <= f["bounds"][1] and f["bounds"][2] <= c <= f["bounds"][3]
            if visible:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="red", zorder=3))
            else:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="orange", lw=1.2, zorder=3))
        if len(trail) > 1:
            ys = [p[0] for p in trail]; xs = [p[1] for p in trail]
            ax.plot(xs, ys, color="blue", lw=0.8, zorder=4)
        ax.add_patch(Rectangle((GOAL_POS[1] - 0.5, GOAL_POS[0] - 0.5), 1, 1, color="green", zorder=5))
        ax.add_patch(Rectangle((f["agent"][1] - 0.5, f["agent"][0] - 0.5), 1, 1, color="blue", zorder=6))
        if f["kind"] == "end":
            title = f"end: {outcome}"
        else:
            title = f"t{f['k']+1}: {f['tool']} ({f['kind']})"
        ax.set_title(title, fontsize=8)
    fig.suptitle(f"seed {seed}: {outcome}  |  red = visible car, orange = hidden car (true state), blue = agent, green = goal, grey = unobserved", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=80)
    plt.close(fig)

def write_log(seed, outcome, frames, path):
    with open(path, "w") as fh:
        fh.write(f"seed {seed}  outcome {outcome}  decisions {len(frames)-1}\n")
        for f in frames:
            if f["kind"] == "end":
                fh.write(f"end: agent {f['agent']}\n")
                continue
            r0, r1, c0, c1 = f["bounds"]
            vis = sum(1 for (r, c) in f["cars"] if r0 <= r <= r1 and c0 <= c <= c1)
            fh.write(f"t{f['k']+1}: agent {f['agent']} tool {f['tool']} kind {f['kind']} "
                     f"visible_cars {vis} total_cars {len(f['cars'])} window rows {r0}-{r1} cols {c0}-{c1}\n")

if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    n_part, horizon = int(sys.argv[3]), int(sys.argv[4])
    summary = []
    for seed in range(start, end):
        outcome, frames = run_logged(seed, n_part, horizon)
        draw_episode(seed, outcome, frames, f"episode_{seed:02d}.png")
        write_log(seed, outcome, frames, f"episode_{seed:02d}.txt")
        line = f"seed {seed}: {outcome} decisions {len(frames)-1}"
        print(line, flush=True)
        with open("pilot_logged_summary.txt", "a") as fh:
            fh.write(line + "\n")
