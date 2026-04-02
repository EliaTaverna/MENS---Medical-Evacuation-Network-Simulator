"""Multi-run helpers: EMA Workbench integration driven entirely by a scenario YAML."""

from __future__ import annotations

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

from ema_workbench import (
    Model, RealParameter, IntegerParameter,
    ScalarOutcome, SequentialEvaluator, perform_experiments,
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
    with open(experiment_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# CASUALTY CONFIG (from scenario's config file or inline)
# ============================================================

def _load_casualty_config(scenario: dict) -> dict:
    """Load casualty config from the scenario, resolving the config file if needed."""
    cas_gen = scenario["casualty_generation"]
    base_path = Path(scenario["_base_path"])

    if "spawn_map" in cas_gen:
        raw = dict(cas_gen)
    else:
        config_path = base_path / cas_gen["config"]
        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

    diagnosis_mix_df = pd.read_excel(base_path / cas_gen["diagnosis_mix"])
    rx_df = pd.read_excel(base_path / cas_gen["prescriptions"])
    rx_df["prescription"] = rx_df["prescription"].apply(eval)

    return {
        "spawn_map":             raw["spawn_map"],
        "par_total":             raw["par_total"],
        "n_days":                raw["n_days"],
        "waves_per_day":         raw["waves_per_day"],
        "wave_duration_min":     raw["wave_duration_min"],
        "diagnosis_mix_df":      diagnosis_mix_df,
        "rx_df":                 rx_df,
        "lifetime_factory_by_triage": _lifetime_df_to_factories(diagnosis_mix_df),
    }


# ============================================================
# PREPARE A SINGLE MULTI-RUN
# ============================================================

def prepare_multi_run(base_data: dict, base_config: dict, base_scenario: dict,
                      till: float, **kwargs) -> tuple:
    """Apply parameter overrides from EMA and assemble (data, settings).

    Supported parameters:
      seed              – master random seed
      par_total         – total personnel at risk
      n_days, waves_per_day, wave_duration_min – casualty schedule
      <lever_key>       – any key present in base_scenario["levers"] overrides that lever
      till              – simulation duration (minutes)
    """
    seed = int(kwargs.get("seed", base_scenario.get("seed", 42)))

    spawn_map = {fid: dict(cfg) for fid, cfg in base_config["spawn_map"].items()}

    full_df, _ = generate_casualties_df_from_waves(
        spawn_map=spawn_map,
        par_total=int(kwargs.get("par_total",         base_config["par_total"])),
        n_days=int(kwargs.get("n_days",               base_config["n_days"])),
        waves_per_day=int(kwargs.get("waves_per_day", base_config["waves_per_day"])),
        wave_duration_min=int(kwargs.get("wave_duration_min", base_config["wave_duration_min"])),
        diagnosis_mix_df=base_config["diagnosis_mix_df"],
        rx_df=base_config["rx_df"],
        lifetime_factory_by_triage=base_config["lifetime_factory_by_triage"],
        seed=seed,
    )
    casualties_df = full_df[
        ~full_df["triage"].str.startswith(("KIA", "MIA"))
    ].copy().reset_index(drop=True)

    # Build levers: start from scenario base, apply any matching overrides
    levers = dict(base_scenario.get("levers", {}))
    for key, val in kwargs.items():
        if key in levers:
            levers[key] = val
    levers["random_seed"] = seed

    data = {
        **base_data,
        "casualties_df": casualties_df,
        "triage_weights": load_triage_weights_from_scenario(base_scenario),
    }
    settings = {
        "seed":               seed,
        "till":               float(kwargs.get("till", till)),
        "trace":              False,
        "make_video":         False,
        "save_logs":          False,
        "state_log_interval": None,
        "levers":             levers,
        "outputs":            {},
    }

    return data, settings


# ============================================================
# OUTCOMES
# ============================================================

def compute_outcomes(results: dict) -> dict:
    all_patients = results["all_patients"]
    wait_times = results.get("wait_times_df", pd.DataFrame())
    transport_events = results.get("transport_df", pd.DataFrame())

    n_total = len(all_patients)
    if n_total == 0:
        return {"death_rate": 0.0, "safe_rate": 0.0, "total_transport_time": 0.0}

    triages = sorted({p.triage for p in all_patients.values()})

    outcomes = {}
    n_dead = sum(1 for p in all_patients.values() if p.is_dead)
    n_safe = sum(1 for p in all_patients.values() if p.is_safe)
    outcomes["death_rate"] = n_dead / n_total
    outcomes["safe_rate"] = n_safe / n_total

    for triage in triages:
        pts = [p for p in all_patients.values() if p.triage == triage]
        n_t = len(pts)
        outcomes[f"death_rate_{triage}"] = (
            sum(1 for p in pts if p.is_dead) / n_t if n_t > 0 else 0.0
        )

    wait_df = wait_times if not wait_times.empty else pd.DataFrame()
    for triage in triages:
        for wait_type, prefix in [("care", "mean_wait_care"), ("transport", "mean_wait_transport")]:
            col = f"{prefix}_{triage}"
            if not wait_df.empty:
                subset = wait_df[(wait_df["triage"] == triage) & (wait_df["wait_type"] == wait_type)]
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

    return outcomes


# ============================================================
# EMA MODEL
# ============================================================

def build_ema_model(experiment: dict) -> Model:
    """Pre-load scenario data once, then build the EMA Model with a run_model closure."""
    scenario   = load_scenario(experiment["scenario"])
    base_data  = load_data_from_scenario(scenario)
    base_config = _load_casualty_config(scenario)
    till = scenario.get("simulation", {}).get("till", 1000.0)

    def run_model(**kwargs):
        data, settings = prepare_multi_run(base_data, base_config, scenario, till, **kwargs)
        results = run_simulation(data, settings, run_folder=None)
        return compute_outcomes(results)

    model = Model("medevacmodel", function=run_model)

    def _make_param(p):
        cls = IntegerParameter if p["type"] == "integer" else RealParameter
        return cls(p["name"], p["lower"], p["upper"])

    model.uncertainties = [_make_param(p) for p in experiment.get("uncertainties", [])]
    model.levers        = [_make_param(p) for p in experiment.get("levers", [])]
    model.outcomes      = [ScalarOutcome(o) for o in experiment.get("outcomes", [])]

    return model


# ============================================================
# RUN EMA EXPERIMENTS
# ============================================================

def run_ema(model: Model, experiment: dict) -> tuple:
    run_settings = experiment.get("run_settings", {})
    n_scenarios = run_settings.get("n_scenarios", 10)
    n_policies  = run_settings.get("n_policies", 1)
    print(f"Running {n_scenarios} scenarios x {n_policies} policies...")

    with SequentialEvaluator(model) as evaluator:
        experiments_df, outcomes = perform_experiments(
            model,
            scenarios=n_scenarios,
            policies=n_policies,
            evaluator=evaluator,
        )
    return experiments_df, outcomes


# ============================================================
# SAVE RESULTS
# ============================================================

def merge_results(experiments_df: pd.DataFrame, outcomes: dict) -> pd.DataFrame:
    outcomes_df = pd.DataFrame(outcomes)
    return pd.concat([experiments_df.reset_index(drop=True), outcomes_df.reset_index(drop=True)], axis=1)


def save_experiment_results(experiments_df: pd.DataFrame, outcomes: dict, experiment: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_name = Path(experiment["scenario"]).stem
    output_folder = experiment.get("output_folder", "data/multi_run/output")
    run_folder = Path(output_folder) / f"{timestamp}_{scenario_name}"
    run_folder.mkdir(parents=True, exist_ok=True)

    merged_df = merge_results(experiments_df, outcomes)
    merged_df.to_csv(run_folder / "results.csv", index=False)

    print(f"Done. Results saved to {run_folder}/")
    print(f"  {len(merged_df)} runs x {len(merged_df.columns)} columns")
    return run_folder
