# -*- coding: utf-8 -*-
"""simutils — package split out of
``geodesic_dataset_generator_gt_v2_3_main_file.py`` (originally a Colab notebook).

# Geodesic dataset generator — self-sufficient bulk dataset builder (no LLM, no rendering)

This package generates the bulk training dataset of geodesics and pickles it.
Run ``geodesic_dataset_generation.py`` then ``geodesic_quantization.py`` (see the
pipeline list at the bottom of this docstring).

It shares its environment/search code with `movement_geodesics.ipynb` (copied here
verbatim, not imported — this project's convention is self-contained notebooks, not
shared modules) but is a deliberately **separate, lighter** pipeline: no
per-environment folders, no per-timestep state files, no rendering, no mp4 encoding.
`movement_geodesics.ipynb` keeps that folder+mp4 pipeline for generating a handful of
*inspectable sample* geodesics; this notebook is for generating the real dataset at
scale (10k+), where rendering every one of them would be absurd.

Each sample is the minimum information needed to regenerate that geodesic's first (up
to) 5 steps from scratch: `grid_size`, `car_spawn_positions`, `car_direction`,
`car_speed`, `start_pos`, `goal_pos`, `actions_truncated`. The whole dataset (a flat
list of these dicts) is pickled in one file, then wrapped in a PyTorch
`Dataset`/`DataLoader` pair (`geodesic_dataset`/`geodesic_dataloader`).

See `moving_geodesic_description.md` for the full design writeup (shared with
`movement_geodesics.ipynb`).

# change log

v2_2 to v2_3 : added other clustering procedures in order to also have up-going geodesics
v2_3+        : action encoding switched to pure (row, column). See below.

## Coordinate convention

We work in (row, column) coordinates. Row and column both start at 0, so (0, 0)
is the top-left cell and, on a GRID_SIZE x GRID_SIZE grid,
(GRID_SIZE-1, GRID_SIZE-1) is the bottom-right cell. Row increases downward,
column increases rightward. An action is a (drow, dcol) delta added directly to
(row, column):  up=(-1,0)  down=(1,0)  left=(0,-1)  right=(0,1)  wait=(0,0).
There is no cartesian x/y; plotting code that needs an (x, y) frame does the
``x = column, y = row`` swap locally and says so.

## Module layout

- ``config``         : all parameters / world constants + the ``Car`` dataclass
- ``env_utils``      : window bounds, swept-collision math, result dataclasses
- ``env``            : ``MovingWorldEnv``
- ``geodesic``       : ``car_position_at_time`` + ``find_geodesic`` (BFS search)
- ``dataset``        : env sampling, geodesic sample generation, pickling, torch ``Dataset``
- ``clustering``     : the trajectory-clustering routines
- ``tool_names``     : name a tool by its action sequence (U/D/L/R/W), e.g. "RRWDD"
- ``visualization``  : ``plot_centers`` (name-keyed tools dict), ``run_env_walk``, ``plot_single_trajectory``
- ``prompts``        : the environment / tool-description prompt strings
- ``vision``         : legacy single-image ``vision_query`` (delegates to ``vision_qwen``)
- ``vision_qwen``    : Ollama ``/api/generate`` vision query over ``requests`` (no ``openai``)

Pipeline scripts next to this package (run in order):
  geodesic_dataset_generation.py -> geodesic_quantization.py -> generate_tool_description.py
  run_agent_simulation_v2.py     (tool-selection agent)
  run_agent_simulation_notools_v1.py  (agent plans raw actions, no tools)
"""
