from __future__ import annotations

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from ema_workbench.util import ema_logging
from src.helpers.helpers_multi import (
    load_experiment, build_ema_model, run_ema, save_experiment_results,
)


if __name__ == "__main__":
    ema_logging.log_to_stderr(ema_logging.INFO)
    experiment = load_experiment("data/multi_run/input/experiment.yaml")
    model = build_ema_model(experiment)
    experiments_df, outcomes = run_ema(model, experiment)
    save_experiment_results(experiments_df, outcomes, experiment)
