"""Replay reversal scenario: seed=80354, same=25, inter=65, bcas=0.04."""
from __future__ import annotations

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from src.helpers.helpers_multi import replay_run

RESULTS_CSV = "data_cluster/resource_grid/results.csv"
SCENARIO = "data/single_run/input/scenarios/scenario_2btg_pooling_limit_baseline_stove.yaml"

if __name__ == "__main__":
    # Ambulance pooling, 1 amb
    replay_run(results_csv=RESULTS_CSV, scenario_path=SCENARIO, row_index=346696)
    # Ambulance pooling, 2 amb
    replay_run(results_csv=RESULTS_CSV, scenario_path=SCENARIO, row_index=346697)
