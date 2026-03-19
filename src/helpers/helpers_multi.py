from __future__ import annotations

import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

from ema_workbench import (
    Model, RealParameter, IntegerParameter,
    ScalarOutcome, SequentialEvaluator, perform_experiments,
)

import salabim as sim
from src.helpers.casualty_generator import generate_casualties_df_from_waves
from src.helpers.scenario import _lifetime_df_to_factories
from src.helpers.runner import run_simulation


# ============================================================
# EXPERIMENT LOADING
# ============================================================

def load_experiment(experiment_path: str) -> dict:
    with open(experiment_path, "r") as f:
        return yaml.safe_load(f)


def load_data_from_experiment(experiment: dict) -> dict:
    """Load structural simulation data (facilities, ambulances, etc.) from experiment.yaml paths."""
    base = Path(experiment["data_root"])
    components = experiment["components"]
    policy = experiment["policy_matrixes"]

    distance_matrix = pd.read_excel(base / components["distance_matrix"], index_col=0)

    axp_df = pd.read_excel(base / components["axp"])
    axp_df["assigned_facility_ids"] = axp_df["assigned_facility_ids"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) else []
    )

    facility_df = pd.read_excel(base / components["facilities"])
    facility_df["c2_up_id"] = facility_df["c2_up_id"].replace({pd.NA: None, "": None}).where(
        facility_df["c2_up_id"].notna(), None
    )

    units_df = pd.read_excel(base / components["units"], dtype={"unit_id": str})
    units_df["service_time_dist"] = units_df.apply(
        lambda r: sim.Uniform(r["service_time_min"], r["service_time_max"]), axis=1
    )

    ambulance_df = pd.read_excel(base / components["ambulances"])

    # Facility eligibility datasets — one DataFrame per lever value
    fac_elig_datasets = {
        int(k): pd.read_excel(base / v, index_col=0)
        for k, v in policy["facility_eligibility"].items()
    }

    # Ambulance eligibility datasets — one DataFrame per lever value
    amb_elig_datasets = {
        int(k): pd.read_excel(base / v, index_col=0)
        for k, v in policy["ambulance_eligibility"].items()
    }

    c2_df = pd.read_excel(base / components["c2"])
    c2_df["facility_ids"] = c2_df["facility_ids"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) else []
    )
    c2_df["ambulance_ids"] = c2_df["ambulance_ids"].apply(
        lambda x: [s.strip() for s in str(x).split(",")] if pd.notna(x) else []
    )

    return {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
        "fac_eligibility_datasets": fac_elig_datasets,
        "amb_eligibility_datasets": amb_elig_datasets,
    }


def load_casualty_config_from_experiment(experiment: dict) -> dict:
    """Load casualty generation config (diagnosis mix, prescriptions) from experiment.yaml paths."""
    base = Path(experiment["data_root"])
    cas = experiment["casualty_config"]

    diagnosis_mix_df = pd.read_excel(base / cas["diagnosis_mix"])
    rx_df = pd.read_excel(base / cas["prescriptions"])
    rx_df["prescription"] = rx_df["prescription"].apply(eval)

    return {
        "spawn_map": cas["spawn_map"],
        "rate_groups": cas.get("rate_groups", {}),
        "diagnosis_mix_df": diagnosis_mix_df,
        "rx_df": rx_df,
        "lifetime_factory_by_triage": _lifetime_df_to_factories(diagnosis_mix_df),
        "triage_weights": dict(experiment.get("triage_weights", {})),
    }


# ============================================================
# PREPARE A SINGLE MULTI-RUN
# ============================================================

def prepare_multi_run(base_data: dict, base_config: dict, till: float, **kwargs) -> tuple:
    """Generate casualties from EMA parameters and assemble (data, settings).

    base_data and base_config are pre-loaded once outside the EMA loop.
    Returns (data, settings) ready to pass to run_simulation.
    No run_folder — multi-run does not write individual output folders.
    """
    # Build spawn_map, overriding rates for any named rate_groups that appear in kwargs
    spawn_map = {fid: dict(cfg) for fid, cfg in base_config["spawn_map"].items()}
    for rate_name, labels in base_config.get("rate_groups", {}).items():
        if rate_name in kwargs:
            label_set = set(labels)
            for cfg in spawn_map.values():
                if set(cfg["labels"]) == label_set:
                    cfg["rate"] = float(kwargs[rate_name])

    full_df, _ = generate_casualties_df_from_waves(
        spawn_map=spawn_map,
        par_total=int(kwargs["par_total"]),
        n_days=int(kwargs["n_days"]),
        waves_per_day=int(kwargs["waves_per_day"]),
        wave_duration_min=int(kwargs["wave_duration_min"]),
        diagnosis_mix_df=base_config["diagnosis_mix_df"],
        rx_df=base_config["rx_df"],
        lifetime_factory_by_triage=base_config.get("lifetime_factory_by_triage"),
        seed=int(kwargs["casualty_seed"]),
    )
    casualties_df = full_df[
        ~full_df["triage"].str.startswith(("KIA", "MIA"))
    ].copy().reset_index(drop=True)

    data = {**base_data, "casualties_df": casualties_df, "triage_weights": base_config.get("triage_weights", {})}

    settings = {
        "till": till,
        "trace": False,
        "make_video": False,
        "save_logs": False,
        "state_log_interval": None,
        "levers": {
            "facility_eligibility":  int(kwargs.get("facility_eligibility", 0)),
            "ambulance_eligibility": int(kwargs.get("ambulance_eligibility", 0)),
            "joint_assignment":      int(kwargs.get("joint_assignment", 0)),
        },
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
    """Pre-load structural data once, then build the EMA Model with a run_model closure."""
    base_data = load_data_from_experiment(experiment)
    base_config = load_casualty_config_from_experiment(experiment)
    till = experiment.get("simulation", {}).get("till", 1000.0)

    def run_model(**kwargs):
        data, settings = prepare_multi_run(base_data, base_config, till, **kwargs)
        results = run_simulation(data, settings, run_folder=None)
        return compute_outcomes(results)

    model = Model("medevacmodel", function=run_model)

    model.uncertainties = [
        IntegerParameter(u["name"], u["lower"], u["upper"]) if u["type"] == "integer"
        else RealParameter(u["name"], u["lower"], u["upper"])
        for u in experiment["uncertainties"]
    ]

    model.levers = [
        IntegerParameter(lv["name"], lv["lower"], lv["upper"]) if lv["type"] == "integer"
        else RealParameter(lv["name"], lv["lower"], lv["upper"])
        for lv in experiment["levers"]
    ]

    model.outcomes = [ScalarOutcome(o["name"]) for o in experiment["outcomes"]]

    return model


# ============================================================
# RUN EMA EXPERIMENTS
# ============================================================

def run_ema(model: Model, experiment: dict) -> tuple:
    run_settings = experiment.get("run_settings", {})
    n_scenarios = run_settings.get("n_scenarios", 10)
    n_policies = run_settings.get("n_policies", 1)
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
    """Merge EMA experiments and outcomes into a single flat DataFrame."""
    outcomes_df = pd.DataFrame(outcomes)
    return pd.concat([experiments_df.reset_index(drop=True), outcomes_df.reset_index(drop=True)], axis=1)


def save_experiment_results(experiments_df: pd.DataFrame, outcomes: dict, experiment: dict) -> Path:
    """
    Create a timestamped output folder under data/multi_run/output/
    and save a single merged CSV of experiments + outcomes.

    Folder name: {experiment_name}_{timestamp}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = experiment.get("name", "experiment").replace(" ", "_")
    run_folder = Path("data/multi_run/output") / f"{timestamp}_{experiment_name}"
    run_folder.mkdir(parents=True, exist_ok=True)

    merged_df = merge_results(experiments_df, outcomes)
    merged_df.to_csv(run_folder / "results.csv", index=False)

    print(f"Done. Results saved to {run_folder}/")
    print(f"  {len(merged_df)} runs x {len(merged_df.columns)} columns")
    return run_folder
