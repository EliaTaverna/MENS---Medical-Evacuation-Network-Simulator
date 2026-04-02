"""Scenario loading utilities."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Union

import pandas as pd
import salabim as sim
import yaml


def load_scenario(scenario_path: Union[str, Path]) -> dict:
    """
    Load a scenario YAML file.

    Parameters:
        scenario_path: path to the scenario YAML file

    Returns:
        dict with scenario configuration
    """
    scenario_path = Path(scenario_path)

    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    # Store the base path for resolving relative paths
    scenario["_base_path"] = scenario_path.parent.parent  # input folder

    return scenario


def _build_service_time_dist(row):
    """Build a service time distribution from a units DataFrame row.

    Supported ``service_time_type`` values (column optional, defaults to 'uniform'):
    - 'uniform'          : sim.Uniform(service_time_min, service_time_max)
    - 'normal'           : sim.Normal(service_time_mean, service_time_std)
    - 'normal_by_triage' : dict {'mild': Normal(mild_mean, mild_std),
                                 'severe': Normal(severe_mean, severe_std)}
                          T1_BCAS / T2_BCAS → severe; all others → mild.
    """
    dist_type = "uniform"
    if "service_time_type" in row.index:
        v = row["service_time_type"]
        if pd.notna(v):
            dist_type = str(v).lower()

    if dist_type == "normal_by_triage":
        return {
            "mild":   sim.Normal(float(row["service_time_mild_mean"]),   float(row["service_time_mild_std"])),
            "severe": sim.Normal(float(row["service_time_severe_mean"]), float(row["service_time_severe_std"])),
        }
    if dist_type == "normal":
        return sim.Normal(float(row["service_time_mean"]), float(row["service_time_std"]))
    # default: uniform
    return sim.Uniform(row["service_time_min"], row["service_time_max"])


def load_data_from_scenario(scenario: dict) -> dict:
    """
    Load all simulation data based on a scenario configuration.

    Parameters:
        scenario: dict from load_scenario()

    Returns:
        dict with all DataFrames needed for simulation
    """
    base_path = Path(scenario["_base_path"])

    def resolve(relative_path: str) -> str:
        return str(base_path / relative_path)

    components = scenario["components"]
    policy = scenario["policy_matrixes"]

    # Distance matrix — normalise IDs to lowercase
    distance_matrix = pd.read_excel(resolve(components["distance_matrix"]), index_col=0)
    distance_matrix.index   = distance_matrix.index.str.lower()
    distance_matrix.columns = distance_matrix.columns.str.lower()

    # AXP data — normalise IDs to lowercase
    axp_df = pd.read_excel(resolve(components["axp"]))
    axp_df["axp_id"] = axp_df["axp_id"].str.lower()
    axp_df["assigned_facility_ids"] = axp_df["assigned_facility_ids"].apply(
        lambda x: [s.strip().lower() for s in str(x).split(",")] if pd.notna(x) else []
    )

    # Facilities — normalise IDs to lowercase
    facility_df = pd.read_excel(resolve(components["facilities"]))
    facility_df["facility_id"] = facility_df["facility_id"].str.lower()
    facility_df["c2_up_id"] = facility_df["c2_up_id"].replace({pd.NA: None, "": None}).where(
        facility_df["c2_up_id"].notna(), None
    )
    facility_df["c2_up_id"] = facility_df["c2_up_id"].apply(
        lambda x: x.lower() if isinstance(x, str) else x
    )

    # Units
    units_df = pd.read_excel(resolve(components["units"]), dtype={"unit_id": str})
    units_df["service_time_dist"] = units_df.apply(_build_service_time_dist, axis=1)

    # Ambulances — normalise all IDs to lowercase
    ambulance_df = pd.read_excel(resolve(components["ambulances"]))
    ambulance_df["amb_id"]       = ambulance_df["amb_id"].str.lower()
    ambulance_df["start_loc_id"] = ambulance_df["start_loc_id"].str.lower()
    ambulance_df["c2_id"]        = ambulance_df["c2_id"].str.lower()

    # Facility eligibility datasets — normalise IDs to lowercase
    fac_elig_datasets = {}
    for k, v in policy["facility_eligibility"].items():
        df = pd.read_excel(resolve(v), index_col=0)
        df.index   = df.index.str.lower()
        df.columns = df.columns.str.lower()
        fac_elig_datasets[int(k)] = df

    # Ambulance eligibility datasets — normalise IDs to lowercase
    amb_elig_datasets = {}
    for k, v in policy["ambulance_eligibility"].items():
        df = pd.read_excel(resolve(v), index_col=0)
        df.index   = df.index.str.lower()
        df.columns = df.columns.str.lower()
        amb_elig_datasets[int(k)] = df

    # C2 data — normalise all IDs to lowercase
    c2_df = pd.read_excel(resolve(components["c2"]))
    c2_df["c2_id"] = c2_df["c2_id"].str.lower()
    c2_df["facility_ids"] = c2_df["facility_ids"].apply(
        lambda x: [s.strip().lower() for s in str(x).split(",")] if pd.notna(x) else []
    )
    c2_df["ambulance_ids"] = c2_df["ambulance_ids"].apply(
        lambda x: [s.strip().lower() for s in str(x).split(",")] if pd.notna(x) else []
    )

    # Coordinates (optional) — covers all nodes including AXPs, used for visualization
    coord_df = None
    if "coordinates" in components:
        coord_df = pd.read_excel(resolve(components["coordinates"]))
        coord_df["facility_id"] = coord_df["facility_id"].str.lower()

    return {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
        "coord_df": coord_df,
        "fac_eligibility_datasets": fac_elig_datasets,
        "amb_eligibility_datasets": amb_elig_datasets,
    }


def load_casualty_config_from_scenario(scenario: dict) -> dict:
    """
    Load casualty generation config based on scenario.

    Parameters:
        scenario: dict from load_scenario()

    Returns:
        dict with casualty generation configuration, or None if from_dataset is True
    """
    cas_gen = scenario["casualty_generation"]

    # Check if loading from dataset
    if cas_gen.get("from_dataset", False):
        return None

    base_path = Path(scenario["_base_path"])

    def resolve(relative_path: str) -> Path:
        return base_path / relative_path

    # Load single combined diagnosis mix
    diagnosis_mix_df = pd.read_excel(resolve(cas_gen["diagnosis_mix"]))

    # Load prescriptions
    rx_df = pd.read_excel(resolve(cas_gen["prescriptions"]))
    rx_df["prescription"] = rx_df["prescription"].apply(eval)

    # Build lifetime factories
    lifetime_factory_by_triage = _lifetime_df_to_factories(diagnosis_mix_df)

    return {
        "spawn_map":        cas_gen["spawn_map"],
        "par_total":        cas_gen["par_total"],
        "n_days":           cas_gen["n_days"],
        "waves_per_day":    cas_gen["waves_per_day"],
        "wave_duration_min": cas_gen["wave_duration_min"],
        "seed":             cas_gen.get("seed", 42),
        "diagnosis_mix_df": diagnosis_mix_df,
        "rx_df": rx_df,
        "lifetime_factory_by_triage": lifetime_factory_by_triage,
    }


def load_casualties_from_scenario(scenario: dict) -> pd.DataFrame:
    """
    Load pre-generated casualties DataFrame from scenario.

    Expects columns:
        - t_create: float (minutes from sim start)
        - spawn_facility_id: str
        - triage: str
        - prescription: str (will be eval'd to list, e.g. "['SUR', 'ICU']")
        - lifetime_low: float
        - lifetime_mode: float
        - lifetime_high: float

    Parameters:
        scenario: dict from load_scenario()

    Returns:
        DataFrame with casualty data including reconstructed lifetime_dist
    """
    base_path = Path(scenario["_base_path"])
    cas_gen = scenario["casualty_generation"]

    casualties_path = base_path / cas_gen["casualties_dataframe"]
    casualties_df = pd.read_excel(casualties_path)

    # Parse prescription column if it's a string
    if "prescription" in casualties_df.columns:
        casualties_df["prescription"] = casualties_df["prescription"].apply(
            lambda x: eval(x) if isinstance(x, str) else x
        )

    # Reconstruct lifetime_dist as Triangular from low/mode/high
    casualties_df["lifetime_dist"] = casualties_df.apply(
        lambda row: sim.Triangular(
            float(row["lifetime_low"]),
            float(row["lifetime_high"]),
            float(row["lifetime_mode"])
        ),
        axis=1
    )

    return casualties_df


def load_triage_weights_from_scenario(scenario: dict) -> dict:
    """Extract {triage: weight} from the levers section of the scenario YAML."""
    return dict(scenario.get("levers", {}).get("triage_weights", {}))


def _lifetime_df_to_factories(diagnosis_mix_df: pd.DataFrame) -> dict:
    """Convert diagnosis_mix dataframe to lifetime factory functions."""
    factories = {}

    for _, row in diagnosis_mix_df.iterrows():
        triage = row["triage"]
        dist_type = row["lifetime_distribution"].lower()

        if dist_type == "uniform":
            low, high = row["lifetime_low"], row["lifetime_high"]
            factories[triage] = lambda l=low, h=high: sim.Uniform(l, h)
        elif dist_type == "triangular":
            low = row["lifetime_low"]
            high = row["lifetime_high"]
            mode = row.get("lifetime_mode", (low + high) / 2)
            if pd.isna(mode):
                mode = (low + high) / 2
            factories[triage] = lambda l=low, m=mode, h=high: sim.Triangular(l, h, m)
        elif dist_type == "exponential":
            mean = row["lifetime_mean"]
            factories[triage] = lambda m=mean: sim.Exponential(m)
        elif dist_type == "normal":
            mean, std = row["lifetime_mean"], row["lifetime_std"]
            factories[triage] = lambda m=mean, s=std: sim.Normal(m, s)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    return factories
