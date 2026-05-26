"""Multi-run helpers: EMA Workbench integration driven entirely by a scenario YAML."""

from __future__ import annotations

import gc
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime

import numpy as np
from ema_workbench import (
    Model, RealParameter, IntegerParameter, CategoricalParameter,
    ScalarOutcome, ArrayOutcome, SequentialEvaluator, perform_experiments,
)
from ema_workbench.em_framework.evaluators import Samplers
from ema_workbench.em_framework.samplers import (
    sample_uncertainties, sample_levers, FullFactorialSampler,
)
from ema_workbench.em_framework.salib_samplers import SobolSampler
from ema_workbench.util.utilities import save_results

_SAMPLER_MAP = {
    "lhs":          Samplers.LHS,
    "fullfactorial": Samplers.FF,
    "ff":           Samplers.FF,
    "mc":           Samplers.MC,
    "uniform_lhs":  Samplers.UNIFORM_LHS,
    "sobol":        Samplers.SOBOL,
}

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
    with open(experiment_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# CASUALTY CONFIG
# ============================================================

def _parse_facility_list(value) -> list:
    """Parse a comma-joined facility-ID cell into a list of lowercase IDs.

    Handles the three shapes that end up in a force_structure row:
    - Python list (from YAML): returned as-is after strip/lower.
    - Non-empty string (from Excel): split on commas.
    - NaN / empty / missing: returned as ``[]``.
    """
    if value is None:
        return []
    if isinstance(value, float) and value != value:  # NaN
        return []
    if isinstance(value, list):
        return [c.strip().lower() for c in value if str(c).strip()]
    s = str(value).strip("[]").strip()
    if not s:
        return []
    return [c.strip().lower() for c in s.split(",") if c.strip()]


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
    with open(config_path, "r", encoding="utf-8") as f:
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

    # Only battalions in active combat generate casualties
    involved  = force_df[force_df["involved"] == True]
    par_total = int(involved["par"].sum())
    spawn_map = {}
    for _, row in involved.iterrows():
        for fid in _parse_facility_list(row["ccp_ids"]):
            spawn_map[fid] = {"labels": bcas_labels, "rate": bcas_rate, "par": par_total}
        if dnbi_rate > 0:
            for fid in _parse_facility_list(row["r1_ids"]):
                spawn_map[fid] = {"labels": dnbi_labels, "rate": dnbi_rate, "par": par_total}

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

    # ── Pooling counts ──────────────────────────────────────────────────
    # Count trips that are same-battalion vs cross-battalion (within brigade)
    # vs cross-brigade, separately for facility (destination) and transport
    # (which ambulance picked up).
    _pool_keys = [
        "n_facility_same_bn", "n_facility_cross_bn",
        "n_facility_same_bde", "n_facility_cross_bde",
        "n_transport_same_bn", "n_transport_cross_bn",
        "n_transport_same_bde", "n_transport_cross_bde",
    ]
    for k in _pool_keys:
        outcomes[k] = 0

    def _parse_bde_bn(loc_id: str):
        """Extract (brigade, battalion) from a facility ID like ccp_br1_bt2_3."""
        bde, bn = None, None
        for part in str(loc_id).lower().split("_"):
            if part.startswith("br"):
                bde = part
            elif part.startswith("bt"):
                bn = part
        return bde, bn

    if not transport_events.empty:
        departures = transport_events[transport_events["event"] == "departure"].copy()

        # Build ambulance home lookup: amb_id -> (brigade, battalion)
        amb_home = {}
        ambulance_df = results.get("ambulance_df")
        if ambulance_df is not None:
            for _, row in ambulance_df.iterrows():
                amb_home[str(row["amb_id"])] = _parse_bde_bn(row["start_loc_id"])

        for _, dep in departures.iterrows():
            from_bde, from_bn = _parse_bde_bn(dep["from_loc"])
            to_bde, to_bn = _parse_bde_bn(dep["to_loc"])

            # Facility pooling: compare origin vs destination
            # Skip trips where either end lacks a battalion (e.g. R1→R2)
            if from_bde and to_bde:
                if from_bde != to_bde:
                    outcomes["n_facility_cross_bde"] += 1
                elif from_bn and to_bn and from_bn != to_bn:
                    outcomes["n_facility_cross_bn"] += 1
                    outcomes["n_facility_same_bde"] += 1
                elif from_bn and to_bn:
                    outcomes["n_facility_same_bn"] += 1
                    outcomes["n_facility_same_bde"] += 1
                # else: one or both lack a battalion → skip bn-level count

            # Transport pooling: compare ambulance home vs pickup location
            amb_id = str(dep["amb_id"])
            home_bde, home_bn = amb_home.get(amb_id, (None, None))
            if home_bde and from_bde:
                if home_bde != from_bde:
                    outcomes["n_transport_cross_bde"] += 1
                elif home_bn and from_bn and home_bn != from_bn:
                    outcomes["n_transport_cross_bn"] += 1
                    outcomes["n_transport_same_bde"] += 1
                else:
                    outcomes["n_transport_same_bn"] += 1
                    outcomes["n_transport_same_bde"] += 1

    # Phase durations from advanced patient log
    if patient_log_df is not None and not patient_log_df.empty:
        phase_durations = _compute_phase_durations(patient_log_df)
        outcomes.update(phase_durations)

    # Deaths by phase × triage
    if patient_log_df is not None and not patient_log_df.empty:
        _PHASE_TO_LOC = {
            "WAITING_TRANSPORT:ccp":   "ccp_wait_transport",
            "WAITING_CARE:dcr":        "r1_wait_care",
            "WAITING_CARE:pec":        "r1_wait_care",
            "WAITING_TRANSPORT:role1": "r1_wait_transport",
            "WAITING_CARE:dcs":        "r2_wait_care",
        }
        _LOCATIONS = ["ccp_wait_transport", "r1_wait_care",
                       "r1_wait_transport", "r2_wait_care"]
        death_counts: dict[str, int] = {}
        for loc in _LOCATIONS:
            for t in ["T1", "T2", "T3"]:
                death_counts[f"n_died_{loc}_{t}"] = 0

        dead_ids = patient_log_df.loc[
            patient_log_df["action"] == "DIED", "patient_id"
        ].unique()
        for pid in dead_ids:
            prows = patient_log_df[patient_log_df["patient_id"] == pid].sort_values("time")
            died_idx = prows.index[prows["action"] == "DIED"][0]
            pos = prows.index.get_loc(died_idx)
            for i in range(pos - 1, -1, -1):
                act = prows.iloc[i]["action"]
                if act in _PHASE_TO_LOC:
                    triage_raw = str(prows.iloc[i]["triage"])
                    t_short = triage_raw.split("_")[0] if "_" in triage_raw else triage_raw
                    loc = _PHASE_TO_LOC[act]
                    key = f"n_died_{loc}_{t_short}"
                    if key in death_counts:
                        death_counts[key] += 1
                    break
        outcomes.update(death_counts)

    # ── Median phase wait times & completion counts ────────────────────
    # For each phase × triage (T1, T2):
    #   - Patients who entered and completed the phase: duration = end - start
    #   - Patients who entered but died during: duration = inf (pushes median up)
    #   - Patients who never entered (died before or not generated): excluded
    # Phases:
    #   ccp_transport:  WAITING_TRANSPORT:ccp  → CLOCK_PAUSED:transport
    #   r1_care:        WAITING_CARE:(dcr|pec) → CLOCK_PAUSED:care
    #   r1_transport:   WAITING_TRANSPORT:role1 → CLOCK_PAUSED:transport
    #   r2_care:        WAITING_CARE:dcs        → CLOCK_PAUSED:care
    _PHASES = {
        "ccp_transport": {
            "start": lambda a: a == "WAITING_TRANSPORT:ccp",
            "end":   lambda a: a == "CLOCK_PAUSED:transport",
        },
        "r1_care": {
            "start": lambda a: a in ("WAITING_CARE:dcr", "WAITING_CARE:pec"),
            "end":   lambda a: a == "CLOCK_PAUSED:care",
        },
        "r1_transport": {
            "start": lambda a: a == "WAITING_TRANSPORT:role1",
            "end":   lambda a: a == "CLOCK_PAUSED:transport",
        },
        "r2_care": {
            "start": lambda a: a == "WAITING_CARE:dcs",
            "end":   lambda a: a == "CLOCK_PAUSED:care",
        },
    }

    if patient_log_df is not None and not patient_log_df.empty:
        # Pre-group by patient for efficiency
        _phase_durations: dict[str, list] = {}     # phase_triage -> list of durations (inf for died)
        _phase_completed: dict[str, int]  = {}     # phase_triage -> count completed
        _phase_started:   dict[str, int]  = {}     # phase_triage -> count entered

        for phase_name in _PHASES:
            for t in ("T1", "T2"):
                _phase_durations[f"{phase_name}_{t}"] = []
                _phase_completed[f"{phase_name}_{t}"] = 0
                _phase_started[f"{phase_name}_{t}"]   = 0

        # Timeline tracking for T2: fraction meeting evacuation timelines
        # R1 care within 1h of CCP arrival, R2 care within 2h of CCP arrival
        _T2_TIMELINE_R1_MAX = 60.0    # minutes
        _T2_TIMELINE_R2_MAX = 120.0   # minutes
        _t2_total = 0
        _t2_within = 0

        for pid, grp in patient_log_df.groupby("patient_id"):
            triage_raw = str(grp["triage"].iloc[0])
            t_short = triage_raw.split("_")[0] if "_" in triage_raw else triage_raw
            if t_short not in ("T1", "T2"):
                continue

            rows = grp.sort_values("time")
            actions = rows["action"].tolist()
            times = rows["time"].tolist()
            died = "DIED" in actions

            # Track key timestamps for timeline KPI (T2 only)
            t_ccp = None       # time entering CCP (WAITING_TRANSPORT:ccp)
            t_r1_care = None   # time R1 care starts (CLOCK_PAUSED:care after dcr/pec)
            t_r2_care = None   # time R2 care starts (CLOCK_PAUSED:care after dcs)
            _saw_r1_wait = False
            _saw_r2_wait = False

            for phase_name, checks in _PHASES.items():
                key = f"{phase_name}_{t_short}"
                # Find first start action
                start_idx = None
                for i, a in enumerate(actions):
                    if checks["start"](a):
                        start_idx = i
                        break
                if start_idx is None:
                    continue  # never entered this phase
                _phase_started[key] += 1

                # Find first end action after start
                end_idx = None
                for j in range(start_idx + 1, len(actions)):
                    if checks["end"](actions[j]):
                        end_idx = j
                        break

                if end_idx is not None:
                    _phase_durations[key].append(times[end_idx] - times[start_idx])
                    _phase_completed[key] += 1
                elif died:
                    pass  # died during phase — excluded from median, counted in mortality
                # else: simulation ended mid-phase, exclude

                # Collect timestamps for timeline KPI
                if t_short == "T2":
                    if phase_name == "ccp_transport" and start_idx is not None:
                        t_ccp = times[start_idx]
                    elif phase_name == "r1_care":
                        _saw_r1_wait = True
                        if end_idx is not None:
                            t_r1_care = times[end_idx]
                    elif phase_name == "r2_care":
                        _saw_r2_wait = True
                        if end_idx is not None:
                            t_r2_care = times[end_idx]

            # Evaluate timeline for this T2 patient
            if t_short == "T2" and t_ccp is not None:
                _t2_total += 1
                # R1 timeline: must reach R1 care
                r1_ok = (t_r1_care is not None and
                         (t_r1_care - t_ccp) <= _T2_TIMELINE_R1_MAX)
                # R2 timeline: only checked if patient entered R2 care phase
                if _saw_r2_wait:
                    r2_ok = (t_r2_care is not None and
                             (t_r2_care - t_ccp) <= _T2_TIMELINE_R2_MAX)
                else:
                    r2_ok = True  # no R2 needed
                if r1_ok and r2_ok:
                    _t2_within += 1

        for phase_name in _PHASES:
            for t in ("T1", "T2"):
                key = f"{phase_name}_{t}"
                durations = _phase_durations[key]
                if durations:
                    outcomes[f"median_wait_{key}"] = float(np.median(durations))
                else:
                    outcomes[f"median_wait_{key}"] = 0.0
                outcomes[f"n_completed_{key}"] = _phase_completed[key]
                outcomes[f"n_started_{key}"]   = _phase_started[key]
                started = _phase_started[key]
                outcomes[f"mortality_{key}"] = (
                    (started - _phase_completed[key]) / started
                    if started > 0 else 0.0
                )

        outcomes["frac_t2_within_timeline"] = (
            _t2_within / _t2_total if _t2_total > 0 else 0.0
        )
    else:
        for phase_name in _PHASES:
            for t in ("T1", "T2"):
                key = f"{phase_name}_{t}"
                outcomes[f"median_wait_{key}"] = 0.0
                outcomes[f"n_completed_{key}"] = 0
                outcomes[f"n_started_{key}"]   = 0
                outcomes[f"mortality_{key}"]   = 0.0
        outcomes["frac_t2_within_timeline"] = 0.0

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
    run_settings  = experiment.get("run_settings", {})
    n_scenarios   = run_settings.get("n_scenarios", 10)
    n_policies    = run_settings.get("n_policies",  1)
    max_parallel  = run_settings.get("max_parallel", 1)
    batch_size    = run_settings.get("batch_size", 0)
    unc_sampler   = _SAMPLER_MAP.get(str(run_settings.get("uncertainty_sampler", "lhs")).lower(), Samplers.LHS)
    lever_sampler = _SAMPLER_MAP.get(str(run_settings.get("lever_sampler",       "lhs")).lower(), Samplers.LHS)

    # Sobol + batch_size > 0: route to the chunked path that preserves the
    # single Saltelli design but evaluates it in ordered, pool-restarting
    # chunks.  Plain batching would re-sample and break S_i / S_T estimators.
    if unc_sampler is Samplers.SOBOL and batch_size > 0:
        return run_ema_sobol_chunked(model, experiment, output_dir)

    # EMA treats policies=0 as "no levers → 1 dummy policy"; reflect that in display
    n_policies_display = max(n_policies, 1)

    if max_parallel > 1:
        from ema_workbench import MultiprocessingEvaluator
        evaluator_cls    = MultiprocessingEvaluator
        evaluator_kwargs = {"n_processes": max_parallel}
    else:
        evaluator_cls    = SequentialEvaluator
        evaluator_kwargs = {}

    # ── No batching ───────────────────────────────────────────────────────────
    if not batch_size:
        print(f"Running {n_scenarios} scenarios × {n_policies_display} policies "
              f"(max_parallel={max_parallel}, unc={unc_sampler.name}, levers={lever_sampler.name})...")
        with evaluator_cls(model, **evaluator_kwargs) as evaluator:
            experiments_df, outcomes = perform_experiments(
                model, scenarios=n_scenarios, policies=n_policies,
                evaluator=evaluator,
                uncertainty_sampling=unc_sampler,
                lever_sampling=lever_sampler,
            )
        return experiments_df, outcomes

    # ── Batched run ───────────────────────────────────────────────────────────
    # The evaluator is opened and closed PER BATCH so that worker processes
    # are fully killed and restarted between batches.  This ensures that any
    # memory accumulated inside a worker (Python heap fragmentation, salabim
    # residuals, etc.) is released by the OS when the process exits, giving
    # each batch a clean slate.
    n_total   = n_scenarios * n_policies_display
    n_batches = (n_scenarios + batch_size - 1) // batch_size
    print(f"Running {n_scenarios} scenarios × {n_policies_display} policies "
          f"in {n_batches} batches of {batch_size} "
          f"(max_parallel={max_parallel}, unc={unc_sampler.name}, levers={lever_sampler.name}, "
          f"total={n_total} runs)...")

    all_experiments = []
    all_outcomes    = {}

    for batch_id in range(n_batches):
        batch_n = min(batch_size, n_scenarios - batch_id * batch_size)
        print(f"\n--- Batch {batch_id + 1}/{n_batches} "
              f"({batch_n} scenarios × {n_policies_display} policies) ---")

        with evaluator_cls(model, **evaluator_kwargs) as evaluator:
            exp_df, outcomes = perform_experiments(
                model, scenarios=batch_n, policies=n_policies,
                evaluator=evaluator,
                uncertainty_sampling=unc_sampler,
                lever_sampling=lever_sampler,
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
# SOBOL — chunked evaluation of a single Saltelli design
# ============================================================

def run_ema_sobol_chunked(model: Model, experiment: dict, output_dir: Path) -> tuple:
    """Evaluate one Saltelli design in ordered chunks with pool restarts.

    The full ``N*(2k+2)`` Sobol design is sampled ONCE up front; chunks slice
    the resulting scenario list by row index.  A fresh evaluator (and fresh
    worker pool) is opened per chunk, which lets the OS reclaim worker RSS
    between chunks.  Each chunk's outputs are flushed to disk and cleared from
    RAM, bounding main-process memory to ~one chunk during the sweep.

    After all chunks complete, chunk files are read back in order and merged
    into a single ``(experiments_df, outcomes)`` tuple — the same shape
    ``save_experiment_results`` expects — so downstream saving and offline
    analysis are unchanged.

    Critically, the Sobol sampler is invoked once, so row ``j`` in block **A**,
    block **B**, and each ``A_B^(i)`` refers to the same underlying Sobol
    point.  Per-chunk resampling would break that pairing (see the README
    "Samplers and batching" section).

    Parameters
    ----------
    model : ema_workbench.Model
        EMA model built by ``build_ema_model``.
    experiment : dict
        Experiment configuration.  Honoured ``run_settings`` keys:

        - ``n_scenarios`` (int): Saltelli ``N``; total rows per policy are
          ``N*(2k+2)`` if ``sobol_second_order`` is true else ``N*(k+2)``.
        - ``n_policies`` (int): passed to the lever sampler (full-factorial
          resolution per lever).  0 → single dummy policy.
        - ``max_parallel`` (int): workers per chunk; 1 uses ``SequentialEvaluator``.
        - ``batch_size`` (int): chunk size **in scenarios** (not total runs).
          Each chunk evaluates ``batch_size × n_policies_realised`` runs.
        - ``sobol_second_order`` (bool, default True): passed to SALib's
          Saltelli sampler.  Must match what the analysis script uses.
        - ``lever_sampler`` (str, default ``"ff"``): only ``ff`` / ``lhs`` are
          wired up here; anything else falls back to full-factorial.
    output_dir : Path
        Directory where intermediate ``batch_*`` chunk files are written.
        Cleaned up after the final reassembled save.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Same shape as ``run_ema``: ``(experiments_df, outcomes)`` with row
        order preserved across chunks.  Per-policy Saltelli ordering is
        intact — ``experiments_df[experiments_df["policy"] == p]`` yields
        rows in the exact Sobol row order SALib expects.
    """
    rs             = experiment.get("run_settings", {})
    N              = int(rs.get("n_scenarios", 10))
    n_policies     = int(rs.get("n_policies", 0))
    max_parallel   = int(rs.get("max_parallel", 1))
    chunk_size     = int(rs.get("batch_size", 512))
    second_order   = bool(rs.get("sobol_second_order", True))
    lever_sampler_name = str(rs.get("lever_sampler", "ff")).lower()

    # ── 1. Sample the full Saltelli design ONCE. ─────────────────────────────
    scenarios_full = list(sample_uncertainties(
        model, N, sampler=SobolSampler(second_order=second_order)
    ))
    total_scenarios = len(scenarios_full)
    k = len(model.uncertainties)
    expected = N * (2 * k + 2) if second_order else N * (k + 2)
    if total_scenarios != expected:
        print(f"  ! Saltelli design size {total_scenarios} != expected {expected} "
              f"(k={k}, second_order={second_order}) — proceeding with sampled size.")

    # ── 2. Materialise policies ONCE. ────────────────────────────────────────
    # FF on integer levers is deterministic, so sampling it once vs. per-chunk
    # is semantically equivalent; doing it once keeps the policy list
    # identical across chunks and avoids any surprise from future sampler
    # changes.
    if model.levers and n_policies > 0:
        lever_sampler_cls = FullFactorialSampler() if lever_sampler_name == "ff" else FullFactorialSampler()
        policies_full = list(sample_levers(model, n_policies, sampler=lever_sampler_cls))
        n_policies_realised = len(policies_full)
    else:
        policies_full = 0  # EMA substitutes a single dummy null-policy
        n_policies_realised = 1

    n_chunks      = (total_scenarios + chunk_size - 1) // chunk_size
    total_runs    = total_scenarios * n_policies_realised

    if max_parallel > 1:
        from ema_workbench import MultiprocessingEvaluator
        evaluator_cls    = MultiprocessingEvaluator
        evaluator_kwargs = {"n_processes": max_parallel}
    else:
        evaluator_cls    = SequentialEvaluator
        evaluator_kwargs = {}

    print(f"Sobol chunked: {total_scenarios} Saltelli rows × {n_policies_realised} policies "
          f"= {total_runs} runs, split into {n_chunks} chunks of ≤{chunk_size} scenarios "
          f"(max_parallel={max_parallel}, second_order={second_order}).")

    # ── 3. Evaluate chunks in order, flushing each to disk. ──────────────────
    for chunk_id in range(n_chunks):
        start = chunk_id * chunk_size
        end   = min(start + chunk_size, total_scenarios)
        chunk = scenarios_full[start:end]
        print(f"\n--- Chunk {chunk_id + 1}/{n_chunks} "
              f"(scenarios [{start}:{end}) × {n_policies_realised} policies "
              f"= {(end - start) * n_policies_realised} runs) ---")

        with evaluator_cls(model, **evaluator_kwargs) as evaluator:
            exp_df, outcomes = perform_experiments(
                model, scenarios=chunk, policies=policies_full, evaluator=evaluator,
            )

        if output_dir is not None:
            save_batch_results(exp_df, outcomes, output_dir, chunk_id)

        # Drop every chunk-scoped ref we hold in the main process so the Python
        # allocator can reuse the arena for the next chunk.  `evaluator` is
        # explicitly del'd because `with ... as evaluator` leaves the name
        # bound in this scope after the block exits — the dead Pool / queue /
        # log-reader thread would otherwise linger until the next rebind.
        del exp_df, outcomes, chunk, evaluator
        gc.collect()

    # ── 4. Reassemble chunks in order for the final save. ────────────────────
    scalar_outcome_names = list(experiment.get("outcomes", []))
    return _reassemble_sobol_chunks(output_dir, n_chunks, scalar_outcome_names)


def _reassemble_sobol_chunks(output_dir: Path, n_chunks: int,
                              scalar_outcome_names: list) -> tuple:
    """Read ordered chunk files back and reconstruct (experiments_df, outcomes).

    Chunk files were written by ``save_batch_results``:
      - ``batch_{id:04d}.csv``: experiments + scalar outcomes merged.
      - ``batch_{id:04d}_timeseries.npz``: array outcomes (optional).

    To return the raw ``(experiments_df, outcomes)`` format, we split the
    scalar outcome columns back out of the merged CSV using the outcome
    names from the experiment YAML.  Row order across chunks is preserved
    by iterating ``chunk_id`` in ascending order.
    """
    all_merged_dfs = []
    array_parts    = {}  # name -> list[np.ndarray]

    for chunk_id in range(n_chunks):
        csv_path = output_dir / f"batch_{chunk_id:04d}.csv"
        all_merged_dfs.append(pd.read_csv(csv_path))

        npz_path = output_dir / f"batch_{chunk_id:04d}_timeseries.npz"
        if npz_path.exists():
            with np.load(npz_path) as arc:
                for k in arc.files:
                    array_parts.setdefault(k, []).append(arc[k])

    merged_df = pd.concat(all_merged_dfs, ignore_index=True)

    # Split scalar outcomes back out of the merged CSV.
    present_scalars = [name for name in scalar_outcome_names if name in merged_df.columns]
    experiments_df  = merged_df.drop(columns=present_scalars)
    outcomes        = {name: merged_df[name].to_numpy() for name in present_scalars}
    for name, parts in array_parts.items():
        outcomes[name] = np.concatenate(parts, axis=0)

    return experiments_df, outcomes


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

    # Persist the raw (experiments, outcomes) tuple exactly as returned by
    # perform_experiments, in EMA's native tarball format.  Reload later with:
    #     from ema_workbench import load_results
    #     experiments, outcomes = load_results("ema_results.tar.gz")
    ema_results_path = run_folder / "ema_results.tar.gz"
    save_results((experiments_df, outcomes), str(ema_results_path))
    print(f"  ema_results.tar.gz: raw bench output ({len(experiments_df)} runs, "
          f"{len(outcomes)} outcomes)")

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
