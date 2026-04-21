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
# Name of a subfolder under INPUT_BASE that contains the experiment YAMLs to
# run. All *.yaml files in that subfolder are picked up (sorted by name) and
# run in order. Output goes to
#   {output_folder}/sequence_{INPUT_SUBFOLDER}/{timestamp}_{scenario_name}/
# per experiment. Re-running with the same subfolder does not wipe the output
# folder — new timestamped subfolders just coexist with the old ones.
INPUT_BASE = Path("data/multi_run/input")
INPUT_SUBFOLDER = "sa_oat_3brigades"
# ─────────────────────────────────────────────────────────────────────────────


def _discover_experiments(input_dir: Path) -> list[str]:
    """Return all ``*.yaml`` files in ``input_dir`` as repo-relative strings,
    sorted by filename for a deterministic run order."""
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder does not exist: {input_dir}")
    yamls = sorted(p for p in input_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in (".yaml", ".yml"))
    return [str(p.as_posix()) for p in yamls]


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

    input_dir = INPUT_BASE / INPUT_SUBFOLDER
    experiment_paths = _discover_experiments(input_dir)
    if not experiment_paths:
        raise SystemExit(f"No *.yaml files found in {input_dir}.")

    # The sequence folder sits under the first experiment's configured
    # output_folder. Individual experiments' output_folder settings are
    # ignored in sequence mode — the whole batch is grouped under one root.
    # The sequence name is derived from the input subfolder name.
    first = load_experiment(experiment_paths[0])
    base  = Path(first.get("output_folder", "data/multi_run/output"))
    sequence_folder = base / f"sequence_{INPUT_SUBFOLDER}"
    sequence_folder.mkdir(parents=True, exist_ok=True)

    print(f"Input folder:    {input_dir}")
    print(f"Sequence folder: {sequence_folder}")
    print(f"Running {len(experiment_paths)} experiment(s) in sequence.\n")

    results: list[tuple[Path, bool, str | None]] = []
    for i, path in enumerate(experiment_paths, 1):
        print(f"=== [{i}/{len(experiment_paths)}] {path} ===")
        results.append(_run_one(path, sequence_folder))
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"Sequence '{INPUT_SUBFOLDER}' complete.")
    n_ok   = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_ok
    print(f"  {n_ok} succeeded, {n_fail} failed.")
    for run_folder, ok, err in results:
        marker = "OK  " if ok else "FAIL"
        tail   = f"  ({err})" if err else ""
        print(f"  [{marker}] {run_folder.relative_to(_REPO_ROOT)}{tail}")
