"""Multi-run helpers: EMA Workbench integration driven entirely by a scenario YAML."""

from __future__ import annotations

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

import numpy as np
from ema_workbench import (
    Model, RealParameter, IntegerParameter, CategoricalParameter,
    ScalarOutcome, ArrayOutcome, SequentialEvaluator, perform_experiments,
)

from src import (
    load_scenario, load_data_from_scenario, load_triage_weights_from_scenario,
)
from src.helpers.scenario import _lifetime_df_to_factories
from src.helpers.casualty_generator import generate_casualties_df_from_waves
from src.helpers.runner import run_simulation


# ============================================================
# EXPERIMENT LOADING
# ============================================================

def load_experiment(experiment_path: str) -> dict:
    """Load an experiment configuration from a YAML file.

    Parameters
    ----------
    experiment_path : str
        Absolute or relative path to the experiment YAML file.

    Returns
    -------
    dict
        Parsed experiment configuration containing keys such as
        ``scenario``, ``uncertainties``, ``levers``, ``outcomes``,
        ``array_outcomes``, ``run_settings``, and ``output_folder``.
    """
    with open(experiment_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# CASUALTY CONFIG
# ============================================================

def _parse_spawning_points(value) -> list:
    """Parse a spawning_points cell into a list of lowercase CCP IDs.

    Accepts either a Python list (from YAML) or a raw string such as
    ``"[ccp1, ccp2]"`` (from an Excel cell).

    Parameters
    ----------
    value : list or str
        The raw spawning_points value from the force-structure DataFrame.

    Returns
    -------
    list[str]
        Lowercase, stripped CCP facility IDs, e.g. ``["ccp1", "ccp2"]``.
    """
    if isinstance(value, list):
        return [c.strip().lower() for c in value]
    return [c.strip().lower() for c in str(value).strip("[]").split(",")]


def _load_casualty_config(scenario: dict) -> dict:
    """Load casualty config from the scenario's sampling config file.

    Reads the force structure and sampling-settings YAML referenced by
    the scenario, builds a ``spawn_map`` for all battalions marked as
    involved, and returns every field that ``RunSetter`` needs to
    generate casualties for each run.

    Parameters
    ----------
    scenario : dict
        Parsed scenario dict as returned by ``load_scenario``.  Must
        contain a ``casualty_generation`` block with a ``config`` key
        pointing to a sampling-settings YAML relative to
        ``scenario["_base_path"]``.

    Returns
    -------
    dict
        Keys:

        - ``spawn_map`` (dict): maps CCP facility ID → dict with keys
          ``labels``, ``rate``, and ``par``.
        - ``par_total`` (int): total personnel at risk across all units.
        - ``n_days`` (int): simulation duration in days.
        - ``waves_per_day`` (int): number of casualty waves per day.
        - ``wave_duration_min`` (int): duration of each wave in minutes.
        - ``diagnosis_mix_df`` (pd.DataFrame): triage-label probabilities.
        - ``rx_df`` (pd.DataFrame): prescription lookup table.
        - ``lifetime_factory_by_triage`` (dict): callable factories for
          patient lifetime distributions keyed by triage label.
        - ``force_structure_df`` (pd.DataFrame): full force-structure table.
        - ``bcas_labels`` (list[str]): BCAS / KIA / MIA triage labels.
        - ``bcas_rate`` (float): baseline battle-casualty rate.
        - ``dnbi_labels`` (list[str]): DNBI triage labels.
        - ``dnbi_rate`` (float): baseline non-battle-injury rate.
    """
    cas_gen   = scenario["casualty_generation"]
    base_path = Path(scenario["_base_path"])

    config_path = base_path / cas_gen["config"]
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    diagnosis_mix_df = pd.read_excel(base_path / raw["diagnosis_mix"])
    rx_df = pd.read_excel(base_path / raw["prescriptions"])
    rx_df["prescription"] = rx_df["prescription"].apply(eval)

    force_df  = pd.read_excel(base_path / raw["force_structure"])
    bcas_rate = float(raw.get("bcas_rate", 0.023))
    dnbi_rate = float(raw.get("dnbi_rate", 0.0))

    all_labels  = diagnosis_mix_df["triage"].astype(str).tolist()
    bcas_labels = [l for l in all_labels if "BCAS" in l or l.startswith(("KIA", "MIA"))]
    dnbi_labels = [l for l in all_labels if "DNBI" in l]

    par_total = int(force_df["par"].sum())

    # Base spawn_map: only involved battalions
    involved  = force_df[force_df["involved"] == True]
    spawn_map = {}
    for _, row in involved.iterrows():
        for fid in _parse_spawning_points(row["spawning_points"]):
            spawn_map[fid] = {"labels": bcas_labels, "rate": bcas_rate, "par": par_total}

    return {
        "spawn_map":                  spawn_map,
        "par_total":                  par_total,
        "n_days":                     int(raw["n_days"]),
        "waves_per_day":              int(raw["waves_per_day"]),
        "wave_duration_min":          int(raw["wave_duration_min"]),
        "diagnosis_mix_df":           diagnosis_mix_df,
        "rx_df":                      rx_df,
        "lifetime_factory_by_triage": _lifetime_df_to_factories(diagnosis_mix_df),
        "force_structure_df":         force_df,
        "bcas_labels":                bcas_labels,
        "bcas_rate":                  bcas_rate,
        "dnbi_labels":                dnbi_labels,
        "dnbi_rate":                  dnbi_rate,
    }


# ============================================================
# OUTCOMES
# ============================================================

def compute_outcomes(results: dict) -> dict:
    """Compute all scalar and time-series outcomes from a single simulation run.

    Called once per run inside ``RunSetter.__call__`` and returns a flat
    dict that EMA Workbench maps to its ``ScalarOutcome`` / ``ArrayOutcome``
    objects.

    Parameters
    ----------
    results : dict
        Raw results dict returned by ``run_simulation``.  Expected keys:

        - ``all_patients`` (dict[id, Patient]): every patient simulated.
        - ``wait_times_df`` (pd.DataFrame, optional): per-patient wait records
          with columns ``triage``, ``wait_type``, ``duration``.
        - ``transport_df`` (pd.DataFrame, optional): ambulance departure /
          arrival events with columns ``amb_id``, ``event``, ``time``,
          ``from_loc``, ``to_loc``.
        - ``patient_log_df`` (pd.DataFrame, optional): advanced phase log used
          to compute per-phase durations.
        - ``data_collector_records`` (list[dict], optional): interval snapshots
          written by ``DataCollector`` with keys ``time``, ``facility_id``,
          ``metric``, ``T1``, ``T2``, ``T3``.

    Returns
    -------
    dict
        Flat dict of outcome values.  Scalar keys include:

        - ``death_rate``, ``safe_rate``: overall fractions.
        - ``death_rate_{triage}``, ``n_patients_{triage}``,
          ``n_dead_{triage}``: per-triage-label metrics.
        - ``mean_wait_care_{triage}``, ``mean_wait_transport_{triage}``:
          average wait durations.
        - ``total_transport_time``: summed ambulance travel time.
        - Phase-duration keys from ``_compute_phase_durations`` if a
          patient log is present.

        Array keys (1-D numpy arrays, one value per DataCollector snapshot):

        - ``ts_{metric}_{facility_id}_{triage}``: time-series queue length,
          e.g. ``ts_transport_CCP1_T1``.
    """
    from src.helpers.helpers_run import _compute_phase_durations

    all_patients     = results["all_patients"]
    wait_times       = results.get("wait_times_df", pd.DataFrame())
    transport_events = results.get("transport_df",  pd.DataFrame())
    patient_log_df   = results.get("patient_log_df")

    outcomes = {}
    n_total  = len(all_patients)

    if n_total == 0:
        outcomes["death_rate"]          = 0.0
        outcomes["safe_rate"]           = 0.0
        outcomes["total_transport_time"] = 0.0
        return outcomes

    triages = sorted({p.triage for p in all_patients.values()})

    n_dead = sum(1 for p in all_patients.values() if p.is_dead)
    n_safe = sum(1 for p in all_patients.values() if p.is_safe)
    outcomes["death_rate"] = n_dead / n_total
    outcomes["safe_rate"]  = n_safe / n_total

    for triage in triages:
        pts = [p for p in all_patients.values() if p.triage == triage]
        n_t = len(pts)
        n_dead_t = sum(1 for p in pts if p.is_dead)
        outcomes[f"death_rate_{triage}"] = n_dead_t / n_t if n_t > 0 else 0.0
        outcomes[f"n_patients_{triage}"] = n_t
        outcomes[f"n_dead_{triage}"]     = n_dead_t

    wait_df = wait_times if not wait_times.empty else pd.DataFrame()
    for triage in triages:
        for wait_type, prefix in [("care", "mean_wait_care"), ("transport", "mean_wait_transport")]:
            col = f"{prefix}_{triage}"
            if not wait_df.empty:
                subset = wait_df[
                    (wait_df["triage"] == triage) & (wait_df["wait_type"] == wait_type)
                ]
                outcomes[col] = float(subset["duration"].mean()) if len(subset) > 0 else 0.0
            else:
                outcomes[col] = 0.0

    total_time = 0.0
    if not transport_events.empty:
        deps = transport_events[transport_events["event"] == "departure"]
        arrs = transport_events[transport_events["event"] == "arrival"]
        for amb_id in transport_events["amb_id"].unique():
            amb_deps = deps[deps["amb_id"] == amb_id].sort_values("time")
            for _, arr in arrs[arrs["amb_id"] == amb_id].sort_values("time").iterrows():
                matching = amb_deps[
                    (amb_deps["time"] < arr["time"]) &
                    (amb_deps["from_loc"] == arr["from_loc"]) &
                    (amb_deps["to_loc"] == arr["to_loc"])
                ]
                if not matching.empty:
                    total_time += arr["time"] - matching.iloc[-1]["time"]
    outcomes["total_transport_time"] = total_time

    # Phase durations from advanced patient log
    if patient_log_df is not None and not patient_log_df.empty:
        phase_durations = _compute_phase_durations(patient_log_df)
        outcomes.update(phase_durations)

    # Time series from DataCollector
    records = results.get("data_collector_records", [])
    if records:
        rec_df = pd.DataFrame(records)
        for (fac_id, metric), grp in rec_df.groupby(["facility_id", "metric"]):
            grp_sorted = grp.sort_values("time")
            for triage in ["T1", "T2", "T3"]:
                key = f"ts_{metric}_{fac_id}_{triage}"
                outcomes[key] = grp_sorted[triage].to_numpy(dtype=float)

    return outcomes


# ============================================================
# EMA MODEL
# ============================================================

def _make_ema_param(p: dict):
    """Build an EMA parameter object from a YAML parameter dict.

    Parameters
    ----------
    p : dict
        Parameter specification from the experiment YAML.  Required key:
        ``name``.  Optional key: ``type`` (``"real"`` | ``"integer"`` |
        ``"categorical"``; defaults to ``"real"``).  Type-specific keys:

        - ``real`` / ``integer``: ``lower`` (numeric), ``upper`` (numeric).
        - ``categorical``: ``values`` (list).

    Returns
    -------
    RealParameter | IntegerParameter | CategoricalParameter
        The corresponding EMA Workbench parameter object.
    """
    ptype = str(p.get("type", "real")).lower()
    if ptype == "integer":
        return IntegerParameter(p["name"], p["lower"], p["upper"])
    if ptype == "categorical":
        return CategoricalParameter(p["name"], p["values"])
    return RealParameter(p["name"], p["lower"], p["upper"])


def build_ema_model(experiment: dict) -> Model:
    """Pre-load scenario data once and build an EMA ``Model`` backed by ``RunSetter``.

    All heavy I/O (Excel files, YAML configs) happens here so that worker
    processes only receive the already-loaded data structures.
    ``RunSetter`` is a picklable callable, which is required for EMA
    multiprocessing.

    Parameters
    ----------
    experiment : dict
        Experiment configuration as returned by ``load_experiment``.
        Relevant keys:

        - ``scenario`` (str): path to the single-run scenario YAML.
        - ``uncertainties`` (list[dict]): sampled parameter specs.
        - ``levers`` (list[dict]): policy parameter specs.
        - ``outcomes`` (list[str]): names of scalar outcomes.
        - ``array_outcomes`` (list[str]): names of array (time-series) outcomes.

    Returns
    -------
    ema_workbench.Model
        Configured EMA model with ``uncertainties``, ``levers``, and
        ``outcomes`` populated.  The model function is a ``RunSetter``
        instance that receives one scenario dict per call and returns a
        flat outcomes dict.
    """
    from src.components.run_setter import RunSetter

    scenario    = load_scenario(experiment["scenario"])
    base_data   = load_data_from_scenario(scenario)
    base_config = _load_casualty_config(scenario)
    till        = scenario.get("simulation", {}).get("till", 1000.0)

    # Attach r1 facility IDs so RunSetter can build DNBI spawn entries
    fac_df = base_data["facility_df"]
    base_config["r1_ids"] = (
        fac_df.loc[fac_df["facility_type"] == "Role1", "facility_id"]
        .str.lower().tolist()
    )

    scalar_outcomes = [o for o in experiment.get("outcomes", [])]
    array_outcomes  = [o for o in experiment.get("array_outcomes", [])]
    expected_outcomes = scalar_outcomes + array_outcomes

    runner = RunSetter(base_data, base_config, scenario, till,
                       expected_outcomes=expected_outcomes)

    model = Model("medevacmodel", function=runner)
    model.uncertainties = [_make_ema_param(p) for p in experiment.get("uncertainties", [])]
    model.levers        = [_make_ema_param(p) for p in experiment.get("levers", [])]
    model.outcomes      = (
        [ScalarOutcome(o) for o in scalar_outcomes] +
        [ArrayOutcome(o)  for o in array_outcomes]
    )

    return model


# ============================================================
# RUN EMA EXPERIMENTS
# ============================================================

def run_ema(model: Model, experiment: dict, output_dir: Path = None) -> tuple:
    """Execute all EMA experiments, optionally in memory-bounded batches.

    Reads ``run_settings`` from the experiment config to decide whether to
    run sequentially or in parallel, and whether to split work into batches
    to cap peak RAM usage.  When batching is active, each batch is saved to
    ``output_dir`` as a numbered CSV/NPZ; these files are later cleaned up
    by ``save_experiment_results``.

    Parameters
    ----------
    model : ema_workbench.Model
        EMA model built by ``build_ema_model``.
    experiment : dict
        Experiment configuration as returned by ``load_experiment``.
        Relevant ``run_settings`` keys:

        - ``n_scenarios`` (int): number of sampled uncertainty scenarios
          (default 10).
        - ``n_policies`` (int): number of lever policies per scenario
          (default 1).
        - ``max_parallel`` (int): worker processes; ``1`` uses
          ``SequentialEvaluator`` (default 1).
        - ``batch_size`` (int): max scenarios per batch; ``0`` or
          ``>= n_scenarios`` disables batching (default 0).
    output_dir : Path, optional
        Directory where intermediate batch files are written when batching
        is active.  Ignored when batching is disabled.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        - ``experiments_df``: one row per run with all uncertainty /
          lever / policy columns.
        - ``outcomes``: dict mapping outcome name → 1-D numpy array
          (scalars) or 2-D numpy array (time series), length = total runs.
    """
    run_settings = experiment.get("run_settings", {})
    n_scenarios  = run_settings.get("n_scenarios", 10)
    n_policies   = run_settings.get("n_policies",  1)
    max_parallel = run_settings.get("max_parallel", 1)
    batch_size   = run_settings.get("batch_size", 0)

    if max_parallel > 1:
        from ema_workbench import MultiprocessingEvaluator
        evaluator_cls    = MultiprocessingEvaluator
        evaluator_kwargs = {"n_processes": max_parallel}
    else:
        evaluator_cls    = SequentialEvaluator
        evaluator_kwargs = {}

    # ── No batching ───────────────────────────────────────────────────────────
    if not batch_size:
        print(f"Running {n_scenarios} scenarios × {n_policies} policies "
              f"(max_parallel={max_parallel})...")
        with evaluator_cls(model, **evaluator_kwargs) as evaluator:
            experiments_df, outcomes = perform_experiments(
                model, scenarios=n_scenarios, policies=n_policies,
                evaluator=evaluator,
            )
        return experiments_df, outcomes

    # ── Batched run ───────────────────────────────────────────────────────────
    # The evaluator is opened and closed PER BATCH so that worker processes
    # are fully killed and restarted between batches.  This ensures that any
    # memory accumulated inside a worker (Python heap fragmentation, salabim
    # residuals, etc.) is released by the OS when the process exits, giving
    # each batch a clean slate.
    n_total   = n_scenarios * n_policies
    n_batches = (n_scenarios + batch_size - 1) // batch_size
    print(f"Running {n_scenarios} scenarios × {n_policies} policies "
          f"in {n_batches} batches of {batch_size} "
          f"(max_parallel={max_parallel}, total={n_total} runs)...")

    all_experiments = []
    all_outcomes    = {}

    for batch_id in range(n_batches):
        batch_n = min(batch_size, n_scenarios - batch_id * batch_size)
        print(f"\n--- Batch {batch_id + 1}/{n_batches} "
              f"({batch_n} scenarios × {n_policies} policies) ---")

        with evaluator_cls(model, **evaluator_kwargs) as evaluator:
            exp_df, outcomes = perform_experiments(
                model, scenarios=batch_n, policies=n_policies,
                evaluator=evaluator,
            )

        all_experiments.append(exp_df)
        for k, v in outcomes.items():
            if k not in all_outcomes:
                all_outcomes[k] = v
            else:
                all_outcomes[k] = np.concatenate([all_outcomes[k], v], axis=0)

        if output_dir is not None:
            save_batch_results(exp_df, outcomes, output_dir, batch_id)

    merged_experiments = pd.concat(all_experiments, ignore_index=True)
    return merged_experiments, all_outcomes


# ============================================================
# SAVE RESULTS
# ============================================================

def _split_outcomes(outcomes: dict) -> tuple[dict, dict]:
    """Split an outcomes dict into scalar and array sub-dicts.

    EMA Workbench collects all outcomes in a single flat dict.  This helper
    separates them so scalars go to ``results.csv`` and 2-D arrays go to
    ``timeseries.npz``.

    Parameters
    ----------
    outcomes : dict
        Mixed outcomes dict where values are either 1-D numpy arrays
        (one value per run) or 2-D numpy arrays (time-series; shape
        ``n_runs × n_timesteps``).

    Returns
    -------
    tuple[dict, dict]
        ``(scalar_outcomes, array_outcomes)`` — two dicts with the same
        keys partitioned by dimensionality.  Non-ndarray values (e.g.
        empty lists) are placed in ``scalar_outcomes``.
    """
    scalar, array = {}, {}
    for k, v in outcomes.items():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            array[k] = v
        else:
            scalar[k] = v
    return scalar, array


def merge_results(experiments_df: pd.DataFrame, outcomes: dict) -> pd.DataFrame:
    """Merge the EMA experiments table with scalar outcomes into one DataFrame.

    Array (time-series) outcomes are ignored here; they are stored separately
    in ``timeseries.npz`` by ``save_experiment_results``.

    Parameters
    ----------
    experiments_df : pd.DataFrame
        The experiments table returned by ``perform_experiments``, with one
        row per run and columns for all uncertainties, levers, and metadata
        (e.g. ``scenario``, ``policy``, ``seed``).
    outcomes : dict
        Raw outcomes dict as returned by ``run_ema``.  May contain both
        scalar and array values; array values are dropped automatically via
        ``_split_outcomes``.

    Returns
    -------
    pd.DataFrame
        Single wide DataFrame: ``experiments_df`` columns followed by one
        column per scalar outcome.  Row count equals the number of runs.
    """
    scalar_outcomes, _ = _split_outcomes(outcomes)
    outcomes_df = pd.DataFrame(scalar_outcomes)
    return pd.concat(
        [experiments_df.reset_index(drop=True), outcomes_df.reset_index(drop=True)],
        axis=1,
    )


def save_experiment_results(experiments_df: pd.DataFrame, outcomes: dict,
                             experiment: dict, run_folder: Path = None) -> Path:
    """Persist final experiment results to disk and clean up batch files.

    Writes ``results.csv`` (scalar outcomes merged with the experiments
    table) and, when present, ``timeseries.npz`` (array outcomes).
    After writing, any intermediate ``batch_*`` files left by ``run_ema``
    are deleted.

    Parameters
    ----------
    experiments_df : pd.DataFrame
        Experiments table as returned by ``run_ema``.
    outcomes : dict
        Outcomes dict as returned by ``run_ema``.  Scalar and array values
        are separated automatically.
    experiment : dict
        Experiment configuration used to derive the output folder and
        scenario name when ``run_folder`` is not provided.  Relevant keys:

        - ``scenario`` (str): used to name the run folder.
        - ``output_folder`` (str): base directory for output
          (default ``"data/multi_run/output"``).
    run_folder : Path, optional
        Pre-created output directory.  When ``None``, a timestamped folder
        is created under ``experiment["output_folder"]``.

    Returns
    -------
    Path
        Path to the run folder where all outputs were saved.
    """
    if run_folder is None:
        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_name = Path(experiment["scenario"]).stem
        output_folder = experiment.get("output_folder", "data/multi_run/output")
        run_folder    = Path(output_folder) / f"{timestamp}_{scenario_name}"
        run_folder.mkdir(parents=True, exist_ok=True)

    scalar_outcomes, array_outcomes = _split_outcomes(outcomes)

    merged_df = merge_results(experiments_df, scalar_outcomes)
    merged_df.to_csv(run_folder / "results.csv", index=False)
    print(f"  results.csv: {len(merged_df)} runs × {len(merged_df.columns)} columns")

    if array_outcomes:
        np.savez_compressed(run_folder / "timeseries.npz", **array_outcomes)
        shapes = {k: v.shape for k, v in array_outcomes.items()}
        print(f"  timeseries.npz: {len(array_outcomes)} arrays — {shapes}")

    # Remove intermediate batch files (temporary only)
    for f in run_folder.glob("batch_*"):
        f.unlink()

    print(f"Done. Results saved to {run_folder}/")
    return run_folder


def save_batch_results(experiments_df: pd.DataFrame, outcomes: dict,
                        output_dir: Path, batch_id: int) -> None:
    """Save one batch's results to numbered files for later merging.

    Writes ``batch_{id:04d}.csv`` for scalar outcomes and, when array
    outcomes are present, ``batch_{id:04d}_timeseries.npz``.  These files
    are temporary — ``save_experiment_results`` deletes them after the
    final merge.

    Parameters
    ----------
    experiments_df : pd.DataFrame
        Experiments table for this batch only.
    outcomes : dict
        Outcomes dict for this batch only.
    output_dir : Path
        Directory in which to write the batch files.
    batch_id : int
        Zero-based batch index, used to generate unique file names.

    Returns
    -------
    None
    """
    scalar_outcomes, array_outcomes = _split_outcomes(outcomes)
    merged_df = merge_results(experiments_df, scalar_outcomes)
    out_path  = output_dir / f"batch_{batch_id:04d}.csv"
    merged_df.to_csv(out_path, index=False)
    if array_outcomes:
        np.savez_compressed(output_dir / f"batch_{batch_id:04d}_timeseries.npz", **array_outcomes)
    print(f"Batch {batch_id}: {len(merged_df)} runs → {out_path}")


# ============================================================
# REPLAY
# ============================================================

def replay_run(results_csv: str, scenario_path: str,
               seed: int = None, row_index: int = None) -> None:
    """Re-run a single row from a multi-run results CSV with full single-run logging.

    Reconstructs the exact run (seed, uncertainties, levers) from the CSV row,
    executes the simulation, and writes all outputs — including per-C2 dispatch
    logs — to a timestamped replay folder under data/single_run/output/.

    Identify the run by seed (preferred) or by row index.
    """
    from src.components.run_setter import RunSetter
    from src.helpers.runner import run_simulation
    from src.helpers.helpers_run import (
        save_logs, save_network_image, save_gantt_chart, print_summary,
    )

    results_df = pd.read_csv(results_csv)

    if seed is not None:
        matches = results_df.index[results_df["seed"] == seed].tolist()
        if not matches:
            raise ValueError(f"No run with seed={seed} found in {results_csv}")
        if len(matches) > 1:
            raise ValueError(f"Multiple rows with seed={seed} — use row_index instead")
        row_index = matches[0]
    elif row_index is not None:
        if row_index < 0 or row_index >= len(results_df):
            raise IndexError(f"row_index {row_index} out of range (0–{len(results_df) - 1})")
    else:
        raise ValueError("Provide either seed or row_index")

    row = results_df.iloc[row_index].to_dict()
    print(f"Replaying row {row_index}: seed={row.get('seed')}, "
          f"bcas_rate={row.get('bcas_rate')}, "
          f"wave_duration_min={row.get('wave_duration_min')}, "
          f"n_battalions_in_combat={row.get('n_battalions_in_combat')}")

    scenario    = load_scenario(scenario_path)
    base_data   = load_data_from_scenario(scenario)
    base_config = _load_casualty_config(scenario)
    till        = scenario.get("simulation", {}).get("till", 1000.0)

    fac_df = base_data["facility_df"]
    base_config["r1_ids"] = (
        fac_df.loc[fac_df["facility_type"] == "Role1", "facility_id"]
        .str.lower().tolist()
    )

    # Strip EMA metadata columns and NaN outcome values — only pass
    # actual uncertainty / lever / seed parameters to prepare()
    EMA_META = {"scenario", "policy", "model", "experiment id"}
    run_kwargs = {
        k: v for k, v in row.items()
        if k not in EMA_META and not (isinstance(v, float) and pd.isna(v))
    }

    runner = RunSetter(base_data, base_config, scenario, till)
    data, settings = runner.prepare(**run_kwargs)

    # Enable all logs and restore video settings from the scenario
    vid = scenario.get("video", {})
    settings["outputs"]          = {
        "patient_log": True, "safe_patients": True,
        "transport_events": True, "wait_times": True,
        "facility_states": vid.get("enabled", False),
    }
    settings["make_video"]       = vid.get("enabled", False)
    settings["video_fps"]        = vid.get("fps", 24)
    settings["video_time_scale"] = vid.get("time_scale", 5.0)
    settings["video_dpi"]        = vid.get("dpi", 100)
    settings["video_duration"]   = vid.get("duration")
    settings["video_start_time"] = float(vid.get("start_time", 0.0))
    settings["state_log_interval"] = vid.get("state_log_interval", 5.0)

    run_folder = Path("data/single_run/output") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_replay_row{row_index}"
    run_folder.mkdir(parents=True, exist_ok=True)

    results = run_simulation(data, settings, run_folder)

    save_logs(results)
    save_network_image(data, results)
    save_gantt_chart(data, results)
    if settings["make_video"]:
        from src.helpers.helpers_run import make_video
        make_video(data, results, settings)
    print_summary(results, data)
    print(f"\nReplay outputs written to {run_folder}/")
