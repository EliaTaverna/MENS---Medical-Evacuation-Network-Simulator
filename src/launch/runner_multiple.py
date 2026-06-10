from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# ── Silence the cosmetic "ipyparallel not installed" UserWarning from
#    ema_workbench.em_framework in BOTH this process and every MP worker:
#    - filterwarnings handles the main process
#    - PYTHONWARNINGS is inherited by spawned workers on Windows and read
#      by the Python interpreter at startup, before ema_workbench is imported
warnings.filterwarnings("ignore", message=".*ipyparallel.*")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning:ema_workbench")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

import logging
from datetime import datetime
from ema_workbench import ema_logging
from src.helpers.helpers_multi import (
    load_experiment, build_ema_model, run_ema, save_experiment_results,
)

# ── Point this at your experiment YAML ───────────────────────────────────────
EXPERIMENT_PATH = "data/multi_run/input/experiment_2btg_grid_full.yaml"
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ema_logging.log_to_stderr(logging.WARNING)

    experiment = load_experiment(EXPERIMENT_PATH)
    model = build_ema_model(experiment)

    output_folder = experiment.get("output_folder", "data/multi_run/output")
    scenario_name = Path(experiment["scenario"]).stem
    run_folder = Path(output_folder) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{scenario_name}"
    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {run_folder}")

    experiments_df, outcomes = run_ema(model, experiment, output_dir=run_folder)
    save_experiment_results(experiments_df, outcomes, experiment, run_folder=run_folder)
