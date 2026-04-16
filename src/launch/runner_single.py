from __future__ import annotations

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

from src.helpers.runner import run_simulation
from src.helpers.helpers_run import prepare_run, save_logs, save_network_image, save_gantt_chart, make_video, print_summary


if __name__ == "__main__":
    data, settings, run_folder = prepare_run(
        scenario_path="data/single_run/input/scenarios/scenario_second_test.yaml",
    )
    results = run_simulation(data, settings, run_folder)

    if settings["save_logs"]:
        save_logs(results)
        save_network_image(data, results)
        save_gantt_chart(data, results)
    if settings["make_video"]:
        make_video(data, results, settings)

    print_summary(results, data)
