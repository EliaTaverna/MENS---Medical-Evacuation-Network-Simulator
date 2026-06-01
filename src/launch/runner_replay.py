from __future__ import annotations

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from src.helpers.helpers_multi import replay_run

if __name__ == "__main__":
    replay_run(
        results_csv="data/multi_run/output/20260527_000623_scenario_elia_midterm/results.csv",
        scenario_path="data/single_run/input/scenarios/scenario_second_test.yaml",
        seed=3109,
    )
