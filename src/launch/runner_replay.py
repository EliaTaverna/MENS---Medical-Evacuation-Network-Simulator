from __future__ import annotations

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from src.helpers.helpers_multi import replay_run

if __name__ == "__main__":
    # Case A — low overlap (waves=1, dur=7): pooling helps
    replay_run(
        results_csv="data_cluster/1rq_rerun_sobol/results.csv",
        scenario_path="data/single_run/input/scenarios/scenario_2btg_pooling_limit_baseline_stove.yaml",
        row_index=49820,  # stove scenario 13820: bcas=0.039, wpd=1, dur=7, same=15, inter=26 → mort 44%
    )
    replay_run(
        results_csv="data_cluster/1rq_rerun_sobol/results.csv",
        scenario_path="data/single_run/input/scenarios/scenario_2btg_pooling_limit_baseline_stove.yaml",
        row_index=51820,  # fac scenario 13820: same params → mort 22%
    )

    # Case B — high overlap (waves=6, dur=41): pooling kills
    replay_run(
        results_csv="data_cluster/1rq_rerun_sobol/results.csv",
        scenario_path="data/single_run/input/scenarios/scenario_2btg_pooling_limit_baseline_stove.yaml",
        row_index=40903,  # stove scenario 10903: bcas=0.045, wpd=6, dur=41, same=17, inter=27 → mort 0%
    )
    replay_run(
        results_csv="data_cluster/1rq_rerun_sobol/results.csv",
        scenario_path="data/single_run/input/scenarios/scenario_2btg_pooling_limit_baseline_stove.yaml",
        row_index=42903,  # fac scenario 10903: same params → mort 27%
    )
