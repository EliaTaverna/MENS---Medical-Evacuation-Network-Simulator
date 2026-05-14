"""
Tests for the triage-based C2 dispatch pipeline.

Organised in four levels:
  1. fac_eligibility  — pure function unit tests
  2. amb_eligibility  — pure function unit tests
  3. solve_triage     — unit tests with mock objects
  4. craft_plans      — unit tests with mock state
  5. Integration      — full simulation end-to-end

Run with:  pytest tests/test_c2.py -v
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import salabim as sim

from src.policies.policies import (
    fac_eligibility,
    amb_eligibility,
    craft_plans,
    unload_planned,
    load_planned,
)
from src.policies.solvers import solve_triage
from src.components.builder import ModelBuilder, ModelConnector
from src.components.ambulance import Ambulance
from src.components.patient import Patient
from src.components.spawner import CasualtySpawner


# =============================================================================
# SHARED MOCK INFRASTRUCTURE
# =============================================================================

class MockLocation:
    def __init__(self, loc_id: str):
        self.loc_id = loc_id


class MockAmb:
    """Minimal ambulance stub satisfying the solve_triage interface."""
    def __init__(self, amb_id: str, t1_t2_capacity: int, t3_capacity: int, loc_id: str):
        self.amb_id = amb_id
        self.t1_t2_capacity = t1_t2_capacity
        self.t3_capacity = t3_capacity
        self.location = MockLocation(loc_id)

    def _loc_id(self, x):
        return getattr(x, "loc_id", x)


class MockUnit:
    """Minimal unit stub for expected wait queries."""
    def __init__(self, ewt: float = 0.0):
        self._ewt = ewt

    def expected_wait_if_join_now(self) -> float:
        return self._ewt


class MockFacility:
    """Minimal facility stub for solve_triage / craft_plans."""
    def __init__(self, loc_id: str, units: dict | None = None):
        self.loc_id = loc_id
        self.units = units or {}
        self.exit_queue = []


class MockEnv:
    """Minimal salabim environment stub for unit tests."""
    def __init__(
        self,
        amb_by_id: dict | None = None,
        facility_by_id: dict | None = None,
        current_time: float = 0.0,
        fac_eligibility_datasets: dict | None = None,
        amb_eligibility_datasets: dict | None = None,
    ):
        self.amb_by_id = amb_by_id or {}
        self.facility_by_id = facility_by_id or {}
        self._t = current_time
        self.fac_eligibility_datasets = fac_eligibility_datasets or {}
        self.amb_eligibility_datasets = amb_eligibility_datasets or {}

    def now(self) -> float:
        return self._t


def _requests_df(*rows: dict) -> pd.DataFrame:
    DEFAULTS = {
        "req_id": None,
        "casualty_id": None,
        "triage": "T1",
        "issuing_facility_id": "F1",
        "eligible_facilities": ["F2"],
        "eligible_ambulances": ["AMB1"],
        "t_issued": 0.0,
        "needed_unit": None,
        "status": "pending",
    }
    records = []
    for i, row in enumerate(rows):
        r = dict(DEFAULTS)
        r["req_id"] = f"R{i+1}"
        r["casualty_id"] = f"P{i+1}"
        r.update(row)
        records.append(r)
    return pd.DataFrame(records)


def _dist(locs: list[str], pairs: dict) -> pd.DataFrame:
    """Build a symmetric distance DataFrame from {(a, b): minutes} dict."""
    df = pd.DataFrame(0.0, index=locs, columns=locs)
    for (a, b), d in pairs.items():
        df.loc[a, b] = d
        df.loc[b, a] = d
    return df


# =============================================================================
# 1. FAC_ELIGIBILITY
# =============================================================================

def test_fac_elig_basic():
    matrix = pd.DataFrame(
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        index=["F1", "F2", "F3"], columns=["F1", "F2", "F3"],
    )
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == ["F2"]


def test_fac_elig_origin_absent():
    matrix = pd.DataFrame([[0, 1]], index=["F1"], columns=["F1", "F2"])
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F_UNKNOWN"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == []


def test_fac_elig_lever_missing():
    env = MockEnv(fac_eligibility_datasets={})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == []


def test_fac_elig_all_zeros_row():
    matrix = pd.DataFrame([[0, 0]], index=["F1"], columns=["F1", "F2"])
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == []


def test_fac_elig_empty_requests():
    matrix = pd.DataFrame([[0, 1]], index=["F1"], columns=["F1", "F2"])
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = pd.DataFrame(columns=["issuing_facility_id"])
    out = fac_eligibility(env, df, lever_value=0)
    assert out.empty


# =============================================================================
# 2. AMB_ELIGIBILITY
# =============================================================================

def test_amb_elig_basic():
    matrix = pd.DataFrame([[1]], index=["AMB1"], columns=["F1"])
    amb = MockAmb("AMB1", 1, 1, "AXP1")
    env = MockEnv(amb_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = amb_eligibility(env, df, idle_ambulances=[amb], lever_value=0)
    assert "AMB1" in out.loc[0, "eligible_ambulances"]


def test_amb_elig_origin_not_in_columns():
    matrix = pd.DataFrame([[1]], index=["AMB1"], columns=["F1"])
    amb = MockAmb("AMB1", 1, 1, "AXP1")
    env = MockEnv(amb_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F_OTHER"})
    out = amb_eligibility(env, df, idle_ambulances=[amb], lever_value=0)
    assert out.loc[0, "eligible_ambulances"] == []


def test_amb_elig_no_matrix_returns_all_idle():
    amb1 = MockAmb("AMB1", 1, 1, "AXP1")
    amb2 = MockAmb("AMB2", 1, 1, "AXP2")
    env = MockEnv(amb_eligibility_datasets={})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = amb_eligibility(env, df, idle_ambulances=[amb1, amb2], lever_value=0)
    result = out.loc[0, "eligible_ambulances"]
    assert "AMB1" in result and "AMB2" in result


def test_amb_elig_non_idle_excluded():
    matrix = pd.DataFrame([[1, 1]], index=["AMB1"], columns=["F1", "F2"])
    amb1 = MockAmb("AMB1", 1, 1, "AXP1")
    amb2 = MockAmb("AMB2", 1, 1, "AXP2")
    env = MockEnv(amb_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = amb_eligibility(env, df, idle_ambulances=[amb2], lever_value=0)
    assert "AMB1" not in out.loc[0, "eligible_ambulances"]


# =============================================================================
# 3. SOLVE_TRIAGE
# =============================================================================

def test_triage_empty_df():
    env = MockEnv()
    df = pd.DataFrame(columns=[
        "req_id", "casualty_id", "triage", "issuing_facility_id",
        "eligible_facilities", "eligible_ambulances", "t_issued", "needed_unit",
    ])
    out = solve_triage(env, df, idle_ambulances=[], travel_time_matrix=None)
    assert "assigned_facility_id" in out.columns
    assert "assigned_amb_id" in out.columns
    assert out.empty


def test_triage_single_t1_assigned():
    """One T1 request with one eligible ambulance → gets assigned."""
    amb = MockAmb("AMB1", 2, 1, "AXP1")
    fac = MockFacility("F2")
    env = MockEnv(amb_by_id={"AMB1": amb}, facility_by_id={"F2": fac})
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df({})
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assert out.loc[0, "assigned_amb_id"] == "AMB1"
    assert out.loc[0, "assigned_facility_id"] == "F2"


def test_triage_no_eligible_facilities():
    amb = MockAmb("AMB1", 2, 1, "AXP1")
    env = MockEnv(amb_by_id={"AMB1": amb})
    df = _requests_df({"eligible_facilities": []})
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=None)
    assert out.loc[0, "assigned_facility_id"] is None


def test_triage_no_eligible_ambulances():
    amb = MockAmb("AMB1", 2, 1, "AXP1")
    env = MockEnv(amb_by_id={"AMB1": amb})
    df = _requests_df({"eligible_ambulances": []})
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=None)
    assert out.loc[0, "assigned_amb_id"] is None


def test_triage_t1_loaded_before_t2():
    """T1 and T2 at same facility, t1_t2_capacity=1 → T1 gets the slot."""
    amb = MockAmb("AMB1", 1, 0, "AXP1")
    fac = MockFacility("F2")
    env = MockEnv(amb_by_id={"AMB1": amb}, facility_by_id={"F2": fac})
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"triage": "T1", "t_issued": 5.0},
        {"triage": "T2", "t_issued": 0.0},
    )
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assigned = out[out["assigned_amb_id"].notna()]
    assert len(assigned) == 1
    assert assigned.iloc[0]["triage"] == "T1"


def test_triage_t3_fills_t3_capacity():
    """T1 + T3 at facility. t1_t2_cap=1, t3_cap=2. T1 takes T1/T2 slot, T3 fills T3 slot."""
    amb = MockAmb("AMB1", 1, 2, "AXP1")
    fac = MockFacility("F2")
    env = MockEnv(amb_by_id={"AMB1": amb}, facility_by_id={"F2": fac})
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"triage": "T1"},
        {"triage": "T3"},
        {"triage": "T3"},
        {"triage": "T3"},
    )
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assigned = out[out["assigned_amb_id"].notna()]
    assert len(assigned) == 3  # 1 T1 + 2 T3
    assert assigned[assigned["triage"] == "T1"].shape[0] == 1
    assert assigned[assigned["triage"] == "T3"].shape[0] == 2


def test_triage_t2_pair_both_loaded():
    """Two T2s at facility, t1_t2_cap=2 → both loaded."""
    amb = MockAmb("AMB1", 2, 1, "AXP1")
    fac = MockFacility("F2")
    env = MockEnv(amb_by_id={"AMB1": amb}, facility_by_id={"F2": fac})
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"triage": "T2"},
        {"triage": "T2"},
    )
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assigned = out[out["assigned_amb_id"].notna()]
    assert len(assigned) == 2
    assert (assigned["triage"] == "T2").all()


def test_triage_nearest_ambulance_selected():
    """Two ambulances at different distances → nearest is selected."""
    amb1 = MockAmb("AMB1", 2, 1, "AXP1")
    amb2 = MockAmb("AMB2", 2, 1, "AXP2")
    fac = MockFacility("F2")
    env = MockEnv(amb_by_id={"AMB1": amb1, "AMB2": amb2}, facility_by_id={"F2": fac})
    dm = _dist(
        ["F1", "F2", "AXP1", "AXP2"],
        {("F1", "F2"): 10, ("AXP1", "F1"): 1, ("AXP2", "F1"): 20,
         ("AXP1", "F2"): 11, ("AXP2", "F2"): 30},
    )
    df = _requests_df({"eligible_ambulances": ["AMB1", "AMB2"]})
    out = solve_triage(env, df, idle_ambulances=[amb1, amb2], travel_time_matrix=dm)
    assert out.loc[0, "assigned_amb_id"] == "AMB1"


def test_triage_only_one_ambulance_dispatched():
    """Two idle ambulances, one request → only one is dispatched."""
    amb1 = MockAmb("AMB1", 2, 1, "AXP1")
    amb2 = MockAmb("AMB2", 2, 1, "AXP2")
    fac = MockFacility("F2")
    env = MockEnv(amb_by_id={"AMB1": amb1, "AMB2": amb2}, facility_by_id={"F2": fac})
    dm = _dist(
        ["F1", "F2", "AXP1", "AXP2"],
        {("F1", "F2"): 10, ("AXP1", "F1"): 5, ("AXP2", "F1"): 5},
    )
    df = _requests_df({"eligible_ambulances": ["AMB1", "AMB2"]})
    out = solve_triage(env, df, idle_ambulances=[amb1, amb2], travel_time_matrix=dm)
    assigned_ambs = out["assigned_amb_id"].dropna().unique()
    assert len(assigned_ambs) == 1


def test_triage_best_destination_by_transport_plus_ewt():
    """Two eligible facilities: F2 (EWT=0, dist=10) and F3 (EWT=0, dist=5) → F3 wins."""
    amb = MockAmb("AMB1", 2, 1, "AXP1")
    fac_f2 = MockFacility("F2", units={"surgery": MockUnit(ewt=0.0)})
    fac_f3 = MockFacility("F3", units={"surgery": MockUnit(ewt=0.0)})
    env = MockEnv(
        amb_by_id={"AMB1": amb},
        facility_by_id={"F2": fac_f2, "F3": fac_f3},
    )
    dm = _dist(
        ["F1", "F2", "F3", "AXP1"],
        {("F1", "F2"): 10, ("F1", "F3"): 5, ("AXP1", "F1"): 0},
    )
    df = _requests_df({
        "eligible_facilities": ["F2", "F3"],
        "needed_unit": "surgery",
    })
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assert out.loc[0, "assigned_facility_id"] == "F3"


def test_triage_ewt_tiebreaker():
    """Two facilities equidistant: F2 (EWT=100) and F3 (EWT=0) → F3 wins."""
    amb = MockAmb("AMB1", 2, 1, "AXP1")
    fac_f2 = MockFacility("F2", units={"surgery": MockUnit(ewt=100.0)})
    fac_f3 = MockFacility("F3", units={"surgery": MockUnit(ewt=0.0)})
    env = MockEnv(
        amb_by_id={"AMB1": amb},
        facility_by_id={"F2": fac_f2, "F3": fac_f3},
    )
    dm = _dist(
        ["F1", "F2", "F3", "AXP1"],
        {("F1", "F2"): 10, ("F1", "F3"): 10, ("AXP1", "F1"): 0},
    )
    df = _requests_df({
        "eligible_facilities": ["F2", "F3"],
        "needed_unit": "surgery",
    })
    out = solve_triage(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assert out.loc[0, "assigned_facility_id"] == "F3"


# =============================================================================
# 4. CRAFT_PLANS
# =============================================================================

def _make_craft_setup(n_patients: int = 1, t1_t2_capacity: int = 1, t3_capacity: int = 1):
    patients = [
        SimpleNamespace(patient_id=f"P{i+1}", triage="T1")
        for i in range(n_patients)
    ]
    requests = [
        SimpleNamespace(req_id=f"R{i+1}", patient=patients[i], status="pending")
        for i in range(n_patients)
    ]

    fac_pickup = MockFacility("F1")
    fac_pickup.exit_queue = list(requests)
    fac_dropoff = MockFacility("F2")

    amb = SimpleNamespace(
        amb_id="AMB1",
        t1_t2_capacity=t1_t2_capacity,
        t3_capacity=t3_capacity,
    )

    mock_env = SimpleNamespace(
        facility_by_id={"F1": fac_pickup, "F2": fac_dropoff},
        amb_by_id={"AMB1": amb},
    )
    c2 = SimpleNamespace(env=mock_env)

    rows = [
        {
            "req_id": f"R{i+1}",
            "casualty_id": f"P{i+1}",
            "triage": "T1",
            "issuing_facility_id": "F1",
            "assigned_facility_id": "F2",
            "assigned_amb_id": "AMB1",
            "status": "pending",
        }
        for i in range(n_patients)
    ]
    df = pd.DataFrame(rows)
    return c2, df, patients, requests


def test_craft_plans_basic_structure():
    c2, df, patients, _ = _make_craft_setup(n_patients=1)
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    assert "AMB1" in plans
    plan = plans["AMB1"]
    assert len(plan) == 2
    pickup_step, dropoff_step = plan
    assert pickup_step.facility.loc_id == "F1"
    assert dropoff_step.facility.loc_id == "F2"


def test_craft_plans_patients_in_load_and_unload():
    c2, df, patients, _ = _make_craft_setup(n_patients=1)
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    pickup_step, dropoff_step = plans["AMB1"]
    assert patients[0] in pickup_step.load_patients
    assert patients[0] in dropoff_step.unload_patients


def test_craft_plans_missing_columns_raises():
    c2 = SimpleNamespace(env=SimpleNamespace(facility_by_id={}, amb_by_id={}))
    df = pd.DataFrame([{"req_id": "R1"}])
    try:
        craft_plans(c2=c2, assigned_ambulances_df=df)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_craft_plans_empty_df():
    c2 = SimpleNamespace(env=SimpleNamespace(facility_by_id={}, amb_by_id={}))
    df = pd.DataFrame()
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    assert plans == {}


def test_craft_plans_respects_ambulance_capacity():
    c2, df, patients, _ = _make_craft_setup(n_patients=2, t1_t2_capacity=1, t3_capacity=0)
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    pickup_step = plans["AMB1"][0]
    assert len(pickup_step.load_patients) == 1


# =============================================================================
# 5. INTEGRATION TESTS
# =============================================================================

def _make_integration_scenario(n_ambulances: int = 1, t1_t2_capacity: int = 2, t3_capacity: int = 1):
    axp_ids = [f"AXP{i+1}" for i in range(n_ambulances)]
    locs = ["F1", "F2"] + axp_ids

    dist_data = {("F1", "F2"): 10.0}
    for axp in axp_ids:
        dist_data[(axp, "F1")] = 0.0
        dist_data[(axp, "F2")] = 10.0
    travel_time_matrix = _dist(locs, dist_data)

    axp_df = pd.DataFrame([
        {"axp_id": axp, "assigned_facility_ids": ["F1"]}
        for axp in axp_ids
    ])

    facility_df = pd.DataFrame([
        {"facility_id": "F1", "facility_type": "CCP", "c2_up_id": "C2",
         "unload_policy": "unload_planned", "load_policy": "load_planned"},
        {"facility_id": "F2", "facility_type": "Role1", "c2_up_id": None,
         "unload_policy": "unload_planned", "load_policy": "load_planned"},
    ])

    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": 10,
        "service_time_params": {"type": "uniform", "low": 1, "high": 1},
    }])

    ambulance_df = pd.DataFrame([
        {"amb_id": f"AMB{i+1}", "t1_t2_capacity": t1_t2_capacity,
         "t3_capacity": t3_capacity,
         "start_loc_id": axp_ids[i], "c2_id": "C2"}
        for i in range(n_ambulances)
    ])

    fac_elig = pd.DataFrame([[0, 1], [0, 0]], index=["F1", "F2"], columns=["F1", "F2"])
    amb_elig = pd.DataFrame(
        [[1, 1]] * n_ambulances,
        index=[f"AMB{i+1}" for i in range(n_ambulances)],
        columns=["F1", "F2"],
    )

    c2_df = pd.DataFrame([{
        "c2_id": "C2",
        "facility_ids": ["F1", "F2"],
        "ambulance_ids": [f"AMB{i+1}" for i in range(n_ambulances)],
    }])

    return {
        "travel_time_matrix": travel_time_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
        "fac_eligibility_datasets": {0: fac_elig},
        "amb_eligibility_datasets": {0: amb_elig},
    }


def _casualties(rows: list[dict]) -> pd.DataFrame:
    records = []
    for i, r in enumerate(rows):
        lifetime = r.get("lifetime", 9999)
        records.append({
            "patient_id": f"P{i+1}",
            "t_create": r.get("t_create", 0),
            "triage": r["triage"],
            "spawn_facility_id": r.get("spawn_facility_id", "F1"),
            "prescription": r.get("prescription", ["surgery"]),
            "lifetime_low": lifetime,
            "lifetime_high": lifetime,
            "lifetime_dist_type": "uniform",
            "lifetime_mode": float("nan"),
            "injury_time": r.get("t_create", 0),
        })
    return pd.DataFrame(records)


def _run_integration(data, casualties_df, levers=None, till=200):
    env = sim.Environment(trace=False)
    env.levers = {
        "facility_eligibility": 0,
        "ambulance_eligibility": 0,
        "random_seed": 42,
        **(levers or {}),
    }
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    env.transport_events = []
    env.wait_times = []
    env.unit_busy_periods = {}
    env.all_patients = {}
    env.safe = []
    env.facility_states = []
    env.patient_log_rows = []
    env.c2_dispatch_logs = {}
    env.data_collector_records = []
    env.patient_log_columns = [
        "time", "patient_id", "triage", "diagnosis", "action", "life_budget",
        "needed_unit", "current_location", "is_dead", "is_safe",
    ]

    ModelBuilder(env).build_all(
        data["travel_time_matrix"], data["axp_df"], data["facility_df"],
        data["units_df"], data["ambulance_df"], data["c2_df"],
        fac_eligibility_datasets=data.get("fac_eligibility_datasets"),
        amb_eligibility_datasets=data.get("amb_eligibility_datasets"),
    )
    ModelConnector(env).connect_all()
    CasualtySpawner(casualties_df=casualties_df)
    env.run(till=till)

    transport_df = pd.DataFrame(env.transport_events)
    return transport_df, env


def test_integration_t1_dispatched_immediately():
    """T1 patient triggers immediate dispatch — arrives at F2 within ~10 min travel."""
    data = _make_integration_scenario(n_ambulances=1)
    casualties = _casualties([
        {"triage": "T1", "t_create": 0},
    ])
    transport_df, env = _run_integration(data, casualties)

    arrivals_at_f2 = transport_df[
        (transport_df["event"] == "arrival") & (transport_df["to_loc"] == "F2")
    ]
    assert len(arrivals_at_f2) >= 1
    assert arrivals_at_f2.iloc[0]["time"] <= 15  # ~0 pickup + 10 travel


def test_integration_t1_dispatched_before_t2():
    """T1 and T2 spawn simultaneously, one ambulance (t1_t2_cap=1).
    T1 is dispatched first (immediate). T2 stays until ambulance returns
    and the T2 threshold or a second T2 triggers dispatch."""
    data = _make_integration_scenario(n_ambulances=1, t1_t2_capacity=1, t3_capacity=1)
    casualties = _casualties([
        {"triage": "T1", "t_create": 0},
        {"triage": "T2", "t_create": 0},
        {"triage": "T2", "t_create": 1},  # second T2 triggers pair dispatch on ambulance return
    ])
    transport_df, env = _run_integration(data, casualties, till=300)

    arrivals_at_f2 = transport_df[
        (transport_df["event"] == "arrival") & (transport_df["to_loc"] == "F2")
    ].sort_values("time")

    assert len(arrivals_at_f2) >= 2
    first_arrival = arrivals_at_f2.iloc[0]
    assert "T1" in first_arrival["patient_triages"]


def test_integration_second_t1_dispatched_after_return():
    """Two T1 patients, one ambulance. Second T1 dispatched after ambulance returns."""
    data = _make_integration_scenario(n_ambulances=1, t1_t2_capacity=1, t3_capacity=1)
    casualties = _casualties([
        {"triage": "T1", "t_create": 0},
        {"triage": "T1", "t_create": 0},
    ])
    transport_df, env = _run_integration(data, casualties, till=300)

    arrivals_at_f2 = transport_df[
        (transport_df["event"] == "arrival") & (transport_df["to_loc"] == "F2")
    ]
    assert len(arrivals_at_f2) == 2

    times = sorted(arrivals_at_f2["time"].tolist())
    assert times[1] > times[0] + 5


def test_integration_t1_bundles_t3():
    """T1 + T3 at same facility, t1_t2_cap=1 t3_cap=1 → both loaded together."""
    data = _make_integration_scenario(n_ambulances=1, t1_t2_capacity=1, t3_capacity=1)
    casualties = _casualties([
        {"triage": "T1", "t_create": 0},
        {"triage": "T3", "t_create": 0},
    ])
    transport_df, env = _run_integration(data, casualties)

    arrivals_at_f2 = transport_df[
        (transport_df["event"] == "arrival") & (transport_df["to_loc"] == "F2")
    ]
    # Both should arrive in a single trip
    assert len(arrivals_at_f2) == 1
    triages = arrivals_at_f2.iloc[0]["patient_triages"]
    assert "T1" in triages
    assert "T3" in triages


# =============================================================================
# RUNNER
# =============================================================================

def run_all_tests():
    tests = [
        # fac_eligibility
        ("fac_elig: basic",                        test_fac_elig_basic),
        ("fac_elig: origin absent",                test_fac_elig_origin_absent),
        ("fac_elig: lever missing",                test_fac_elig_lever_missing),
        ("fac_elig: all zeros row",                test_fac_elig_all_zeros_row),
        ("fac_elig: empty requests",               test_fac_elig_empty_requests),
        # amb_eligibility
        ("amb_elig: basic",                        test_amb_elig_basic),
        ("amb_elig: origin not in columns",        test_amb_elig_origin_not_in_columns),
        ("amb_elig: no matrix returns all idle",   test_amb_elig_no_matrix_returns_all_idle),
        ("amb_elig: non-idle excluded",            test_amb_elig_non_idle_excluded),
        # solve_triage
        ("triage: empty df",                       test_triage_empty_df),
        ("triage: single T1 assigned",             test_triage_single_t1_assigned),
        ("triage: no eligible facilities",         test_triage_no_eligible_facilities),
        ("triage: no eligible ambulances",         test_triage_no_eligible_ambulances),
        ("triage: T1 loaded before T2",            test_triage_t1_loaded_before_t2),
        ("triage: T3 fills T3 capacity",           test_triage_t3_fills_t3_capacity),
        ("triage: T2 pair both loaded",            test_triage_t2_pair_both_loaded),
        ("triage: nearest ambulance selected",     test_triage_nearest_ambulance_selected),
        ("triage: only one amb dispatched",        test_triage_only_one_ambulance_dispatched),
        ("triage: best dest by transport+ewt",     test_triage_best_destination_by_transport_plus_ewt),
        ("triage: ewt tiebreaker",                 test_triage_ewt_tiebreaker),
        # craft_plans
        ("craft_plans: basic structure",           test_craft_plans_basic_structure),
        ("craft_plans: patients in load/unload",   test_craft_plans_patients_in_load_and_unload),
        ("craft_plans: missing columns raises",    test_craft_plans_missing_columns_raises),
        ("craft_plans: empty df",                  test_craft_plans_empty_df),
        ("craft_plans: respects capacity",         test_craft_plans_respects_ambulance_capacity),
        # integration
        ("integration: T1 dispatched immediately", test_integration_t1_dispatched_immediately),
        ("integration: T1 before T2",             test_integration_t1_dispatched_before_t2),
        ("integration: second T1 after return",   test_integration_second_t1_dispatched_after_return),
        ("integration: T1 bundles T3",            test_integration_t1_bundles_t3),
    ]

    passed, failed, errored = 0, 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  [ERROR] {name}: {e}")
            traceback.print_exc()
            errored += 1

    total = len(tests)
    print(f"\n{passed}/{total} passed  |  {failed} failed  |  {errored} errors")
    return failed == 0 and errored == 0


if __name__ == "__main__":
    print("=" * 70)
    print("C2 DISPATCH PIPELINE TESTS")
    print("=" * 70)
    run_all_tests()
