from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# ── Silence the cosmetic "ipyparallel not installed" UserWarning from
#    ema_workbench.em_framework in BOTH this process and every MP worker.
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

# ── Configure the sequence ───────────────────────────────────────────────────
# Name of this sequence. Output goes to
#   {output_folder}/sequence_{SEQUENCE_NAME}/{timestamp}_{scenario_name}/
# per experiment. Re-running with the same name does not wipe the folder —
# new timestamped subfolders just coexist with the old ones.
SEQUENCE_NAME = "sa_oat_3brigades"

# Experiment YAMLs to run, in order. Each runs with its own model build and
# EMA call, just like runner_multiple.py would — only the output path is
# redirected under the shared sequence folder.
#
# This set is a one-at-a-time sensitivity analysis on
# scenario_3_brigades_3_battalions_big_r2e_no_r2b: each YAML sweeps exactly
# one axis via full factorial with 30 seeds per level, all other knobs held
# at scenario default.
EXPERIMENT_PATHS: list[str] = [
    # Casualty generation
    "data/multi_run/input/experiment_bcas_rate_sa_oat.yaml",
    "data/multi_run/input/experiment_waves_per_day_sa_oat.yaml",
    "data/multi_run/input/experiment_wave_duration_min_sa_oat.yaml",
    "data/multi_run/input/experiment_n_days_sa_oat.yaml",
    "data/multi_run/input/experiment_fraction_t1_bcas_sa_oat.yaml",
    "data/multi_run/input/experiment_fraction_t2_bcas_sa_oat.yaml",
    # Force / doctrine
    "data/multi_run/input/experiment_n_battalions_in_combat_sa_oat.yaml",
    # Capacities
    "data/multi_run/input/experiment_dcr_capacity_sa_oat.yaml",
    "data/multi_run/input/experiment_pec_capacity_sa_oat.yaml",
    # Ambulances — CCP->R1 leg
    "data/multi_run/input/experiment_n_amb_r1_to_ccp_sa_oat.yaml",
    "data/multi_run/input/experiment_amb_speed_ccp_r1_sa_oat.yaml",
    # Ambulances — R1->R2 leg (silent with default prescriptions; active
    # once prescriptions include DCS)
    "data/multi_run/input/experiment_n_amb_r1_r2_sa_oat.yaml",
    "data/multi_run/input/experiment_amb_speed_r1_r2_sa_oat.yaml",
    # R2 capacity (silent with default prescriptions)
    "data/multi_run/input/experiment_dcs_capacity_sa_oat.yaml",
    # Distances — CCP<->R1 tiers
    "data/multi_run/input/experiment_r1_ccp_same_km_sa_oat.yaml",
    "data/multi_run/input/experiment_r1_ccp_neighbor_km_sa_oat.yaml",
    # Distances — R1<->R2-enhanced tiers (silent with default prescriptions)
    "data/multi_run/input/experiment_r2_enhanced_r1_same_km_sa_oat.yaml",
    "data/multi_run/input/experiment_r2_enhanced_r1_neighbor_km_sa_oat.yaml",
    # Pooling levers
    "data/multi_run/input/experiment_facility_eligibility_sa_oat.yaml",
    "data/multi_run/input/experiment_ambulance_eligibility_sa_oat.yaml",
]
# ─────────────────────────────────────────────────────────────────────────────


def _run_one(experiment_path: str, sequence_folder: Path) -> tuple[Path, bool, str | None]:
    """Run a single experiment into its own subfolder of the sequence folder.

    Returns ``(run_folder, ok, error_message)``. On failure, writes a
    ``FAILED.txt`` sentinel inside the run folder and returns ``ok=False``
    so the sequence can continue with the next experiment.
    """
    experiment    = load_experiment(experiment_path)
    scenario_name = Path(experiment["scenario"]).stem
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder    = sequence_folder / f"{timestamp}_{scenario_name}"
    run_folder.mkdir(parents=True, exist_ok=True)
    print(f"  output → {run_folder}")

    try:
        model = build_ema_model(experiment)
        experiments_df, outcomes = run_ema(model, experiment, output_dir=run_folder)
        save_experiment_results(experiments_df, outcomes, experiment,
                                run_folder=run_folder)
        return run_folder, True, None
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"  [FAIL] {err}")
        (run_folder / "FAILED.txt").write_text(err, encoding="utf-8")
        return run_folder, False, err


if __name__ == "__main__":
    ema_logging.log_to_stderr(logging.WARNING)

    if not EXPERIMENT_PATHS:
        raise SystemExit("EXPERIMENT_PATHS is empty — nothing to run.")

    # The sequence folder sits under the first experiment's configured
    # output_folder. Individual experiments' output_folder settings are
    # ignored in sequence mode — the whole batch is grouped under one root.
    first = load_experiment(EXPERIMENT_PATHS[0])
    base  = Path(first.get("output_folder", "data/multi_run/output"))
    sequence_folder = base / f"sequence_{SEQUENCE_NAME}"
    sequence_folder.mkdir(parents=True, exist_ok=True)

    print(f"Sequence folder: {sequence_folder}")
    print(f"Running {len(EXPERIMENT_PATHS)} experiment(s) in sequence.\n")

    results: list[tuple[Path, bool, str | None]] = []
    for i, path in enumerate(EXPERIMENT_PATHS, 1):
        print(f"=== [{i}/{len(EXPERIMENT_PATHS)}] {path} ===")
        results.append(_run_one(path, sequence_folder))
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Sequence '{SEQUENCE_NAME}' complete.")
    n_ok   = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"  {n_ok} succeeded, {n_fail} failed.")
    for run_folder, ok, err in results:
        marker = "OK  " if ok else "FAIL"
        tail   = f"  ({err})" if err else ""
        print(f"  [{marker}] {run_folder.relative_to(_REPO_ROOT)}{tail}")
