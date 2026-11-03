# -*- coding: utf-8 -*-
"""Random-choice reference policy (no LLM), for comparison with the LLM runs.

Each turn uses the same observations, library and collision filter as the LLM
pipeline, but the tool is chosen at random:

  1. the collision check (``simutils.collision_check``) replays each tool against
     only the cars currently inside the observation window and drops the tools that
     would hit one of them;
  2. if no tool remains, the agent waits (``WWWWW``), with no random draw;
  3. if exactly one tool remains, it is taken, with no random draw;
  4. otherwise one of the remaining tools is drawn uniformly (``random.choice``).

Each episode starts from a new random environment (its seed is drawn from the
global random stream), and the episode ends at the goal, on a collision, or after
``MAX_TURNS`` = 30 turns. Each run writes ``chat.txt`` with an ``outcome:`` line per
turn, and the last such line is the outcome of the episode (goal, collision, or None
when the turn limit is reached). The reported results (``baseline_results/``) were
obtained with 10 chunks of 10 episodes, each chunk seeded with ``random.seed(2026 + k)``.

Run from the repository root (``quantized_actions.npz`` and ``simutils/`` are needed):
    python3 run_random_baseline.py [output_dir]
    NR_RUNS episodes are run; output_dir defaults to ``random_baseline``.
"""

# ============================================================
# Global parameters
# ============================================================
TOOL_ACTIONS_NPZ     = "quantized_actions.npz"     # name-keyed .npz: key = tool name ("RRWDD")



import sys
MAX_TURNS       = 30
NUM_CARS_SIM    = None     # None -> config.NUM_CARS
SIM_PLOTS_DIR   = sys.argv[1] if len(sys.argv) > 1 else "random_baseline"
SIM_LOG_PKL     = "tool_execution_log.pkl"
TXT_LOG_FILE    = "chat.txt"
MAKE_ZIP        = False
NR_RUNS = 100   # number of times to run main()

import simutils.log_stderr_stdout   # tee stdout/stderr to output.txt

import json
import pickle
import random
random.seed(2026)  # random baseline
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

from simutils.config import (
    GRID_SIZE, START_POS, GOAL_POS, WINDOW_HALF_SIZE, FIXED_TURN_LENGTH,
    USE_HAZARD_CLOSEUP, NUM_CARS,
)
from simutils.env import MovingWorldEnv
from simutils.env_utils import compute_window_bounds
from simutils.visualization import plot_single_trajectory
from simutils.collision_check import collision_with_visible_cars



def main():
    num_cars  = NUM_CARS_SIM if NUM_CARS_SIM is not None else NUM_CARS

    # ---- load tools + their short descriptions ----
    _npz = np.load(TOOL_ACTIONS_NPZ)
    tools = {name: _npz[name] for name in _npz.files}   # {name -> (T, 2) (drow, dcol)}
    tool_names = list(tools)
    print(f"Loaded {len(tools)} tools: {tool_names}")

    # ---- plots directory (archive an existing one) ----
    simulation_plots_dir = Path(SIM_PLOTS_DIR)
    if simulation_plots_dir.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(f"{SIM_PLOTS_DIR}_{timestamp}")
        shutil.move(str(simulation_plots_dir), str(backup_dir))
        print(f"Previous simulation plots in {simulation_plots_dir} moved to {backup_dir}")
    simulation_plots_dir.mkdir(exist_ok=True)
    print(f"Simulation plots will be saved to: {simulation_plots_dir}")

    simulation_log = []

    # ---- sample a new environment ----
    env_seed = random.randint(0, 100000)
    env = MovingWorldEnv(
        size=GRID_SIZE, num_cars=num_cars,
        start_pos=START_POS, goal_pos=GOAL_POS, seed=env_seed,
    )
    print(f"\n--- Starting new simulation for env_seed={env_seed} ---")
    print(f"Initial environment: agent_pos={env.agent_pos}, goal_pos={env.goal_pos}, "
          f"hazard_closeup={'on' if USE_HAZARD_CLOSEUP else 'off'}")

    simulation_agent_path = [env.agent_pos]
    simulation_car_positions_history = [[car.position for car in env.cars]]

    # turn 0 renders
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE, title="Turn 0: Initial State (Full View)",
        save_only=str(simulation_plots_dir / "turn_0_full_view.png"),
    )
    initial_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE, title="Turn 0: Observation Window (for LLM)",
        window_bounds=initial_window_bounds,
        save_only=str(simulation_plots_dir / "turn_0_obs_view_for_llm.png"),
    )

    simulation_log.append({
        'turn': 0, 'chosen_tool': None, 'safe_tools': None, 'filtered_out': None,
        'agent_path_steps': [env.agent_pos],
        'car_positions_pre_turn': [car.position for car in env.cars],
        'car_positions_post_turn': [car.position for car in env.cars],
        'outcome': None,
    })

    for turn_idx in range(MAX_TURNS):
        if env.is_terminal():
            print(f"Simulation terminated early at turn {turn_idx} with outcome: {env.outcome}")
            break
        print(f"\n--- Turn {turn_idx + 1} ---")

        car_positions_pre_turn = [car.position for car in env.cars]

        # 1. full view (log only)
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: Current State (Full View, Pre-LLM)",
            save_only=str(simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_pre_llm.png"),
        )

        # 2. images for the LLM: wide observation window (+ optional general close-up)
        current_window_bounds = compute_window_bounds(env.agent_pos, WINDOW_HALF_SIZE, GRID_SIZE)
        current_obs_view_path = simulation_plots_dir / f"turn_{turn_idx + 1}_obs_view_for_llm.png"
        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: Observation Window (for LLM)",
            window_bounds=current_window_bounds,
            save_only=str(current_obs_view_path),
        )
        current_zoom_view_path = None
        if USE_HAZARD_CLOSEUP:
            _ar, _ac = env.agent_pos
            current_zoom_view_path = simulation_plots_dir / f"turn_{turn_idx + 1}_zoom_view_for_llm.png"
            plot_single_trajectory(
                agent_path=simulation_agent_path,
                final_car_positions=simulation_car_positions_history[-1],
                grid_size=GRID_SIZE,
                title=f"Turn {turn_idx + 1}: Close-up",
                window_bounds=current_window_bounds,
                crop_rows=(_ar - 1, current_window_bounds.row_max),
                crop_cols=(_ac - FIXED_TURN_LENGTH, _ac + FIXED_TURN_LENGTH),
                show_relative_ticks=True,
                save_only=str(current_zoom_view_path),
            )

        # 3. code collision filter: keep tools that do NOT hit a visible car
        safe_tools = [n for n in tool_names
                      if not collision_with_visible_cars(env, tools[n])]
        filtered_out = [n for n in tool_names if n not in safe_tools]
        print(f"visible-car collision filter: {len(safe_tools)}/{len(tool_names)} tools pass -> {safe_tools}")

        # 4. choose a tool
        if len(safe_tools) == 0:
            chosen_tool = "WWWWW" if "WWWWW" in tools else tool_names[0]
            llm_final_response = "(no LLM call: no tool avoids a visible car -> forced WWWWW)"
            llm_thinking_response = ""
            print(f"No safe tool -> forced {chosen_tool}, no LLM call.")
        elif len(safe_tools) == 1:
            chosen_tool = safe_tools[0]
            llm_final_response = f"(no LLM call: only one safe tool -> {chosen_tool})"
            llm_thinking_response = ""
            print(f"Only one safe tool -> {chosen_tool}, no LLM call.")
        else:
            chosen_tool = random.choice(safe_tools)
            llm_final_response = f"(random baseline: uniform choice -> {chosen_tool})"
            llm_thinking_response = ""
            print(f"Random baseline chose tool: {chosen_tool}")

        # 5. execute
        selected_actions = tools.get(chosen_tool, np.zeros((FIXED_TURN_LENGTH, 2), dtype=int))
        print(f"Executing tool {chosen_tool}: {selected_actions.tolist()}")
        turn_result = env.execute_turn(
            action_tuples=[(int(a[0]), int(a[1]), 1) for a in selected_actions]
        )
        simulation_agent_path.append(turn_result.final_position)
        simulation_car_positions_history.append([car.position for car in env.cars])
        print(f"Agent moved to: {env.agent_pos}   turn outcome: {turn_result.outcome}")

        plot_single_trajectory(
            agent_path=simulation_agent_path,
            final_car_positions=simulation_car_positions_history[-1],
            grid_size=GRID_SIZE,
            title=f"Turn {turn_idx + 1}: State After Tool {chosen_tool} Execution (Full View)",
            save_only=str(simulation_plots_dir / f"turn_{turn_idx + 1}_full_view_post_llm.png"),
        )

        simulation_log.append({
            'turn': turn_idx + 1,
            'chosen_tool': chosen_tool,
            'safe_tools': safe_tools,
            'filtered_out': filtered_out,
            'llm_final_response': llm_final_response,
            'llm_thinking_response': llm_thinking_response,
            'agent_path_steps': [step.agent_pos for step in turn_result.step_results],
            'car_positions_pre_turn': car_positions_pre_turn,
            'car_positions_post_turn': [car.position for car in env.cars],
            'outcome': turn_result.outcome,
        })

    print("\n--- Simulation Finished ---")
    logfile = str(simulation_plots_dir / SIM_LOG_PKL)
    with open(logfile, "wb") as f:
        pickle.dump(simulation_log, f)
    print(f"Simulation log saved to {logfile}")

    lines = []
    for entry in simulation_log:
        lines.append(f"=== Turn {entry.get('turn')} ===")
        for k, v in entry.items():
            lines.append(f"{k}: {v}")
        lines.append("")
    (simulation_plots_dir / TXT_LOG_FILE).write_text("\n".join(lines), encoding="utf-8")

    if env.outcome is None:          # MAX_TURNS exhausted without goal/collision
        env.mark_failed()
    print(f"Final agent position: {env.agent_pos}")
    print(f"Overall simulation outcome: {env.outcome}")

    # marker file named after the outcome
    with open(str(simulation_plots_dir / env.outcome), "w") as f:
        pass

    plot_single_trajectory(
        agent_path=simulation_agent_path,
        final_car_positions=simulation_car_positions_history[-1],
        grid_size=GRID_SIZE, title=f"Final State (Outcome: {env.outcome})",
    )

    if MAKE_ZIP:
        zip_name = f"backup_results_{simulation_plots_dir}.zip"
        exts = {".txt", ".png", ".npz", ".pkl"}
        with zipfile.ZipFile(zip_name, "w", zipfile.ZIP_DEFLATED) as z:
            for f in simulation_plots_dir.rglob("*"):
                if f.is_file():
                    z.write(f, arcname=f.relative_to(simulation_plots_dir.parent))
            for f in Path(".").glob("*"):
                if f.is_file() and f.suffix.lower() in exts:
                    z.write(f, arcname=f.name)
        print("Created:", zip_name)


if __name__ == "__main__":
    for jj in range(NR_RUNS):
        print(f"start overall run number {jj}")
        main()
    print('end of all runs')
