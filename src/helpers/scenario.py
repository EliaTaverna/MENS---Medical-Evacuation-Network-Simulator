"""Scenario loading utilities."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Union

import pandas as pd
import yaml


class _LifetimeFactory:
    """Picklable callable that returns a lifetime param dict.

    Replaces lambdas in _lifetime_df_to_factories so the factory dict
    can be pickled by multiprocessing workers.
    """
    __slots__ = ("_params",)

    def __init__(self, **params):
        self._params = params

    def __call__(self):
        return dict(self._params)


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


def _build_service_time_params(row) -> dict:
    """Return a plain-dict description of the service-time distribution for a unit row.

    No salabim objects are created here.  The actual distribution is built
    inside Unit.setup(), which runs after the seeded sim.Environment is active.

    Supported ``service_time_type`` values (column optional, defaults to 'uniform'):
    - 'uniform'          : {"type": "uniform", "low": min, "high": max}
    - 'normal'           : {"type": "normal",  "mean": mean, "std": std}
    - 'normal_by_triage' : {"type": "normal_by_triage", "mild_mean": ..., ...}
    """
    dist_type = "uniform"
    if "service_time_type" in row.index:
        v = row["service_time_type"]
        if pd.notna(v):
            dist_type = str(v).lower()

    if dist_type == "normal_by_triage":
        return {
            "type":        "normal_by_triage",
            "mild_mean":   float(row["service_time_mild_mean"]),
            "mild_std":    float(row["service_time_mild_std"]),
            "severe_mean": float(row["service_time_severe_mean"]),
            "severe_std":  float(row["service_time_severe_std"]),
        }
    if dist_type == "normal":
        return {"type": "normal", "mean": float(row["service_time_mean"]), "std": float(row["service_time_std"])}
    # default: uniform
    return {"type": "uniform", "low": float(row["service_time_min"]), "high": float(row["service_time_max"])}


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
    units_df["service_time_params"] = units_df.apply(_build_service_time_params, axis=1)

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


def _parse_spawning_points(value) -> list:
    """Parse a spawning_points cell into a list of lowercase CCP IDs.

    Accepts either a Python list (from YAML) or a raw string such as
    ``"[ccp1, ccp2]"`` (from an Excel cell).
    """
    if isinstance(value, list):
        return [c.strip().lower() for c in value]
    return [c.strip().lower() for c in str(value).strip("[]").split(",")]


def load_casualty_config_from_scenario(scenario: dict) -> dict:
    """Load casualty generation config from a scenario dict.

    Supports two formats for the ``casualty_generation`` block in the scenario YAML:

    **Config-pointer format** (preferred)::

        casualty_generation:
          config: casualty_generation_settings/sampling_settings/config_400.yaml

    The referenced YAML is the single source of truth for all casualty parameters
    (PAR, rates, force structure, schedule).  The function reads it, loads the
    force structure, and builds the ``spawn_map`` from battalions marked as
    ``involved``.  BCAS entries come from ``bcas_spawning_points``; DNBI
    entries come from ``dnbi_spawning_points`` when ``dnbi_rate > 0``.

    **Inline format** (legacy, still supported)::

        casualty_generation:
          par_total: 3500
          bcas_rate: 0.023
          spawn_map: {...}
          ...

    Parameters:
        scenario: dict from :func:`load_scenario`.  Must contain a
            ``casualty_generation`` block and ``_base_path``.

    Returns:
        dict with keys ``spawn_map``, ``par_total``, ``n_days``,
        ``waves_per_day``, ``wave_duration_min``, ``seed``,
        ``diagnosis_mix_df``, ``rx_df``, ``lifetime_factory_by_triage``.
        Returns ``None`` if ``from_dataset`` is ``True``.
    """
    cas_gen = scenario["casualty_generation"]

    # ── Pre-generated dataset ────────────────────────────────────────────────
    if cas_gen.get("from_dataset", False):
        return None

    base_path = Path(scenario["_base_path"])

    def resolve(relative_path: str) -> Path:
        return base_path / relative_path

    # ── Config-pointer format ────────────────────────────────────────────────
    if "config" in cas_gen:
        with open(resolve(cas_gen["config"]), "r") as f:
            raw = yaml.safe_load(f)

        diagnosis_mix_df = pd.read_excel(resolve(raw["diagnosis_mix"]))
        rx_df            = pd.read_excel(resolve(raw["prescriptions"]))
        rx_df["prescription"] = rx_df["prescription"].apply(eval)

        force_df  = pd.read_excel(resolve(raw["force_structure"]))
        bcas_rate = float(raw.get("bcas_rate", 0.023))
        dnbi_rate = float(raw.get("dnbi_rate", 0.0))

        all_labels  = diagnosis_mix_df["triage"].astype(str).tolist()
        bcas_labels = [lb for lb in all_labels if "BCAS" in lb or lb.startswith(("KIA", "MIA"))]
        dnbi_labels = [lb for lb in all_labels if "DNBI" in lb]

        # Only battalions in active combat generate casualties
        involved  = force_df[force_df["involved"] == True]
        par_total = int(involved["par"].sum())
        spawn_map: dict = {}
        for _, row in involved.iterrows():
            for fid in _parse_spawning_points(row["bcas_spawning_points"]):
                spawn_map[fid] = {"labels": bcas_labels, "rate": bcas_rate, "par": par_total}
            if dnbi_rate > 0:
                for fid in _parse_spawning_points(row["dnbi_spawning_points"]):
                    spawn_map[fid] = {"labels": dnbi_labels, "rate": dnbi_rate, "par": par_total}

        return {
            "spawn_map":                  spawn_map,
            "par_total":                  par_total,
            "n_days":                     int(raw["n_days"]),
            "waves_per_day":              int(raw["waves_per_day"]),
            "wave_duration_min":          int(raw["wave_duration_min"]),
            "seed":                       int(raw.get("seed", 42)),
            "diagnosis_mix_df":           diagnosis_mix_df,
            "rx_df":                      rx_df,
            "lifetime_factory_by_triage": _lifetime_df_to_factories(diagnosis_mix_df),
        }

    # ── Inline format (legacy) ───────────────────────────────────────────────
    diagnosis_mix_df = pd.read_excel(resolve(cas_gen["diagnosis_mix"]))
    rx_df            = pd.read_excel(resolve(cas_gen["prescriptions"]))
    rx_df["prescription"] = rx_df["prescription"].apply(eval)

    return {
        "spawn_map":                  cas_gen["spawn_map"],
        "par_total":                  cas_gen["par_total"],
        "n_days":                     cas_gen["n_days"],
        "waves_per_day":              cas_gen["waves_per_day"],
        "wave_duration_min":          cas_gen["wave_duration_min"],
        "seed":                       cas_gen.get("seed", 42),
        "diagnosis_mix_df":           diagnosis_mix_df,
        "rx_df":                      rx_df,
        "lifetime_factory_by_triage": _lifetime_df_to_factories(diagnosis_mix_df),
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
        DataFrame with casualty data including raw lifetime parameter columns
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

    # Raw lifetime parameter columns (lifetime_dist_type, lifetime_low,
    # lifetime_high, lifetime_mode) are used directly by CasualtySpawner →
    # Patient.setup(), which creates the salabim distribution inside the
    # seeded simulation environment.

    return casualties_df


def load_triage_weights_from_scenario(scenario: dict) -> dict:
    """Extract {triage: weight} from the levers section of the scenario YAML."""
    return dict(scenario.get("levers", {}).get("triage_weights", {}))


def _lifetime_df_to_factories(diagnosis_mix_df: pd.DataFrame) -> dict:
    """Convert diagnosis_mix dataframe to lifetime param-dict factory functions.

    Each factory returns a plain dict — no salabim objects are created here.
    """
    factories = {}

    for _, row in diagnosis_mix_df.iterrows():
        triage = row["triage"]
        dist_type = row["lifetime_distribution"].lower()

        if dist_type == "uniform":
            low, high = float(row["lifetime_low"]), float(row["lifetime_high"])
            factories[triage] = _LifetimeFactory(type="uniform", low=low, high=high)
        elif dist_type == "triangular":
            low  = float(row["lifetime_low"])
            high = float(row["lifetime_high"])
            mode = row.get("lifetime_mode", (low + high) / 2)
            if pd.isna(mode):
                mode = (low + high) / 2
            factories[triage] = _LifetimeFactory(type="triangular", low=low, high=high, mode=float(mode))
        elif dist_type == "exponential":
            mean = float(row["lifetime_mean"])
            factories[triage] = _LifetimeFactory(type="exponential", mean=mean)
        elif dist_type == "normal":
            mean = float(row["lifetime_mean"])
            std  = float(row["lifetime_std"])
            factories[triage] = _LifetimeFactory(type="normal", mean=mean, std=std)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    return factories
