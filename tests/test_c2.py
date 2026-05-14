"""
Comprehensive tests for the C2 dispatch pipeline.

Organised in four levels:
  1. fac_eligibility  — pure function unit tests          (5 tests)
  2. amb_eligibility  — pure function unit tests          (4 tests)
  3. solve_ilp        — unit tests with mock objects      (16 tests)
  4. craft_plans      — unit tests with mock state        (5 tests)
  5. Integration      — full simulation end-to-end        (3 tests)

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
from src.helpers.ilp import solve_ilp
from src.components.builder import ModelBuilder, ModelConnector
from src.components.ambulance import Ambulance
from src.components.patient import Patient
from src.components.spawner import CasualtySpawner
from src.helpers.observer import SimulationObserver


# =============================================================================
# SHARED MOCK INFRASTRUCTURE
# =============================================================================

class MockLocation:
    def __init__(self, loc_id: str):
        self.loc_id = loc_id


class MockAmb:
    """Minimal ambulance stub satisfying the solve_ilp interface."""
    def __init__(self, amb_id: str, capacity: int, loc_id: str):
        self.amb_id = amb_id
        self.capacity = capacity
        self.location = MockLocation(loc_id)

    def _loc_id(self, x):
        return getattr(x, "loc_id", x)


class MockUnit:
    """Minimal unit stub for snapshot_facilities EWT queries."""
    def __init__(self, ewt: float = 0.0):
        self._ewt = ewt

    def expected_wait_if_join_now(self) -> float:
        return self._ewt


class MockFacility:
    """Minimal facility stub for solve_ilp / craft_plans."""
    def __init__(self, loc_id: str, units: dict | None = None):
        self.loc_id = loc_id
        self.units = units or {}
        self.exit_queue = []


class MockEnv:
    """Minimal salabim environment stub for unit tests."""
    def __init__(
        self,
        triage_weights: dict | None = None,
        amb_by_id: dict | None = None,
        facility_by_id: dict | None = None,
        current_time: float = 0.0,
        fac_eligibility_datasets: dict | None = None,
        amb_eligibility_datasets: dict | None = None,
    ):
        self.triage_weights = triage_weights or {}
        self.amb_by_id = amb_by_id or {}
        self.facility_by_id = facility_by_id or {}
        self._t = current_time
        self.fac_eligibility_datasets = fac_eligibility_datasets or {}
        self.amb_eligibility_datasets = amb_eligibility_datasets or {}

    def now(self) -> float:
        return self._t


def _requests_df(*rows: dict) -> pd.DataFrame:
    """
    Build a requests DataFrame.  Each row dict may override any field.
    Defaults represent a single T1 patient at F1 eligible for F2 via AMB1.
    """
    DEFAULTS = {
        "req_id": None,           # filled in below
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
    """Build a symmetric distance DataFrame from {(a, b): km} dict."""
    df = pd.DataFrame(0.0, index=locs, columns=locs)
    for (a, b), d in pairs.items():
        df.loc[a, b] = d
        df.loc[b, a] = d
    return df


# =============================================================================
# 1. FAC_ELIGIBILITY
# =============================================================================

def test_fac_elig_basic():
    """F1 is eligible to send to F2 according to the matrix."""
    matrix = pd.DataFrame(
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        index=["F1", "F2", "F3"], columns=["F1", "F2", "F3"],
    )
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == ["F2"]


def test_fac_elig_origin_absent():
    """Origin not in matrix index → empty list."""
    matrix = pd.DataFrame([[0, 1]], index=["F1"], columns=["F1", "F2"])
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F_UNKNOWN"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == []


def test_fac_elig_lever_missing():
    """No dataset for the requested lever → all rows get empty lists."""
    env = MockEnv(fac_eligibility_datasets={})          # no lever 0
    df = _requests_df({"issuing_facility_id": "F1"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == []


def test_fac_elig_all_zeros_row():
    """All zeros in origin row → no eligible destinations."""
    matrix = pd.DataFrame([[0, 0]], index=["F1"], columns=["F1", "F2"])
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = fac_eligibility(env, df, lever_value=0)
    assert out.loc[0, "eligible_facilities"] == []


def test_fac_elig_empty_requests():
    """Empty input DataFrame passes through unchanged."""
    matrix = pd.DataFrame([[0, 1]], index=["F1"], columns=["F1", "F2"])
    env = MockEnv(fac_eligibility_datasets={0: matrix})
    df = pd.DataFrame(columns=["issuing_facility_id"])
    out = fac_eligibility(env, df, lever_value=0)
    assert out.empty


# =============================================================================
# 2. AMB_ELIGIBILITY
# =============================================================================

def test_amb_elig_basic():
    """AMB1 is eligible for origin F1 → returned in eligible_ambulances."""
    matrix = pd.DataFrame([[1]], index=["AMB1"], columns=["F1"])
    amb = MockAmb("AMB1", 1, "AXP1")
    env = MockEnv(amb_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = amb_eligibility(env, df, idle_ambulances=[amb], lever_value=0)
    assert "AMB1" in out.loc[0, "eligible_ambulances"]


def test_amb_elig_origin_not_in_columns():
    """Origin not in matrix columns → empty eligible list."""
    matrix = pd.DataFrame([[1]], index=["AMB1"], columns=["F1"])
    amb = MockAmb("AMB1", 1, "AXP1")
    env = MockEnv(amb_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F_OTHER"})
    out = amb_eligibility(env, df, idle_ambulances=[amb], lever_value=0)
    assert out.loc[0, "eligible_ambulances"] == []


def test_amb_elig_no_matrix_returns_all_idle():
    """No dataset for lever → all idle ambulances are eligible."""
    amb1 = MockAmb("AMB1", 1, "AXP1")
    amb2 = MockAmb("AMB2", 1, "AXP2")
    env = MockEnv(amb_eligibility_datasets={})
    df = _requests_df({"issuing_facility_id": "F1"})
    out = amb_eligibility(env, df, idle_ambulances=[amb1, amb2], lever_value=0)
    result = out.loc[0, "eligible_ambulances"]
    assert "AMB1" in result and "AMB2" in result


def test_amb_elig_non_idle_excluded():
    """Ambulance not in idle_ambulances list is excluded even if in matrix."""
    matrix = pd.DataFrame([[1, 1]], index=["AMB1"], columns=["F1", "F2"])
    amb1 = MockAmb("AMB1", 1, "AXP1")
    amb2 = MockAmb("AMB2", 1, "AXP2")   # not in matrix but doesn't matter
    env = MockEnv(amb_eligibility_datasets={0: matrix})
    df = _requests_df({"issuing_facility_id": "F1"})
    # Only amb2 is idle → AMB1 should NOT appear despite being in matrix
    out = amb_eligibility(env, df, idle_ambulances=[amb2], lever_value=0)
    assert "AMB1" not in out.loc[0, "eligible_ambulances"]


# =============================================================================
# 3. SOLVE_ILP
# =============================================================================

def _make_ilp_env(
    triage_weights: dict | None = None,
    amb_map: dict | None = None,
    fac_map: dict | None = None,
    current_time: float = 0.0,
) -> MockEnv:
    """Convenience constructor for ILP-focused env mocks."""
    return MockEnv(
        triage_weights=triage_weights or {"T1": 3.0, "T2": 2.0, "T3": 1.0},
        amb_by_id=amb_map or {},
        facility_by_id=fac_map or {},
        current_time=current_time,
    )


def test_ilp_empty_df():
    """Empty requests DataFrame → output has assignment columns set to None."""
    env = _make_ilp_env()
    df = pd.DataFrame(columns=[
        "req_id", "casualty_id", "triage", "issuing_facility_id",
        "eligible_facilities", "eligible_ambulances", "t_issued", "needed_unit",
    ])
    out = solve_ilp(env, df, idle_ambulances=[], travel_time_matrix=None)
    assert "assigned_facility_id" in out.columns
    assert "assigned_amb_id" in out.columns
    assert out.empty


def test_ilp_single_request_assigned():
    """One request with one eligible ambulance and facility → gets assigned."""
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(amb_map={"AMB1": amb}, fac_map={"F2": fac})
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df({})   # one row with all defaults
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assert out.loc[0, "assigned_amb_id"] == "AMB1"
    assert out.loc[0, "assigned_facility_id"] == "F2"


def test_ilp_no_eligible_facilities():
    """Empty eligible_facilities list → no assignment."""
    amb = MockAmb("AMB1", 1, "AXP1")
    env = _make_ilp_env(amb_map={"AMB1": amb})
    df = _requests_df({"eligible_facilities": []})
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=None)
    assert out.loc[0, "assigned_facility_id"] is None


def test_ilp_no_eligible_ambulances():
    """Empty eligible_ambulances list → no assignment."""
    amb = MockAmb("AMB1", 1, "AXP1")
    env = _make_ilp_env(amb_map={"AMB1": amb})
    df = _requests_df({"eligible_ambulances": []})
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=None)
    assert out.loc[0, "assigned_amb_id"] is None


def test_ilp_triage_priority_t1_beats_t3():
    """
    T1 and T3 both waiting.  One ambulance, capacity 1.
    With w_T1=100 >> w_T3=1, T1 must be the one assigned.
    """
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        triage_weights={"T1": 100.0, "T2": 10.0, "T3": 1.0},
        amb_map={"AMB1": amb},
        fac_map={"F2": fac},
    )
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1", "triage": "T1"},
        {"req_id": "R2", "casualty_id": "P2", "triage": "T3"},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assigned = out[out["assigned_amb_id"].notna()]
    assert len(assigned) == 1
    assert assigned.iloc[0]["triage"] == "T1"


def test_ilp_triage_priority_t2_beats_t3():
    """w_T2=50 >> w_T3=1 → T2 assigned over T3 when only one slot available."""
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        triage_weights={"T1": 100.0, "T2": 50.0, "T3": 1.0},
        amb_map={"AMB1": amb},
        fac_map={"F2": fac},
    )
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1", "triage": "T2"},
        {"req_id": "R2", "casualty_id": "P2", "triage": "T3"},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assigned = out[out["assigned_amb_id"].notna()]
    assert len(assigned) == 1
    assert assigned.iloc[0]["triage"] == "T2"


def test_ilp_sojourn_tiebreaker():
    """
    Two T2 patients, same triage weight.
    One has sojourn=50 min, the other 0.  With beta=1 the long-waiting
    patient accumulates more score and must be assigned.
    """
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        triage_weights={"T1": 3.0, "T2": 2.0, "T3": 1.0},
        amb_map={"AMB1": amb},
        fac_map={"F2": fac},
        current_time=50.0,   # sim is at t=50
    )
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        # t_issued=0 → sojourn = 50 - 0 = 50
        {"req_id": "R1", "casualty_id": "P1", "triage": "T2", "t_issued": 0.0},
        # t_issued=50 → sojourn = 0
        {"req_id": "R2", "casualty_id": "P2", "triage": "T2", "t_issued": 50.0},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm, beta=1.0)
    assigned = out[out["assigned_amb_id"].notna()]
    assert len(assigned) == 1
    # sojourn=50 patient has score 2 + 1*50 = 52; sojourn=0 has score 2 + 0 = 2
    assert assigned.iloc[0]["req_id"] == "R1"


def test_ilp_triage_dominates_sojourn_when_gap_large():
    """
    T3 has been waiting 9 000 min.  T1 just arrived.
    With w_T1=100, w_T3=1, beta=0.01:
      T1 score = 100 + 0.01*0 = 100
      T3 score =   1 + 0.01*9000 = 91
    T1 must still win.
    """
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        triage_weights={"T1": 100.0, "T2": 10.0, "T3": 1.0},
        amb_map={"AMB1": amb},
        fac_map={"F2": fac},
        current_time=9000.0,
    )
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 5, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1", "triage": "T1", "t_issued": 9000.0},  # sojourn=0
        {"req_id": "R2", "casualty_id": "P2", "triage": "T3", "t_issued": 0.0},     # sojourn=9000
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm, beta=0.01)
    assigned = out[out["assigned_amb_id"].notna()]
    assert assigned.iloc[0]["triage"] == "T1"


def test_ilp_sojourn_crosses_breakeven_t3_wins():
    """
    Same setup but T3 sojourn = 10 001 min:
      T1 score = 100
      T3 score = 1 + 0.01*10001 = 101.01
    T3 must now win.
    """
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        triage_weights={"T1": 100.0, "T2": 10.0, "T3": 1.0},
        amb_map={"AMB1": amb},
        fac_map={"F2": fac},
        current_time=10001.0,
    )
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 5, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1", "triage": "T1", "t_issued": 10001.0},  # sojourn=0
        {"req_id": "R2", "casualty_id": "P2", "triage": "T3", "t_issued": 0.0},      # sojourn=10001
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm, beta=0.01)
    assigned = out[out["assigned_amb_id"].notna()]
    assert assigned.iloc[0]["triage"] == "T3"


def test_ilp_gamma_selects_closer_ambulance():
    """
    Two ambulances, AMB1 is 1 km from F1, AMB2 is 20 km from F1.
    With gamma=1 the ILP minimises transport time → AMB1 assigned.
    """
    amb1 = MockAmb("AMB1", 1, "AXP1")
    amb2 = MockAmb("AMB2", 1, "AXP2")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        amb_map={"AMB1": amb1, "AMB2": amb2},
        fac_map={"F2": fac},
    )
    dm = _dist(
        ["F1", "F2", "AXP1", "AXP2"],
        {("F1", "F2"): 10, ("AXP1", "F1"): 1, ("AXP2", "F1"): 20,
         ("AXP1", "F2"): 11, ("AXP2", "F2"): 30},
    )
    df = _requests_df({"eligible_ambulances": ["AMB1", "AMB2"]})
    out = solve_ilp(env, df, idle_ambulances=[amb1, amb2], travel_time_matrix=dm, gamma=1.0)
    assert out.loc[0, "assigned_amb_id"] == "AMB1"


def test_ilp_capacity_one_limits_assignment():
    """
    Ambulance capacity = 1 with two pending requests.
    Only one of the two requests must be assigned.
    """
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(amb_map={"AMB1": amb}, fac_map={"F2": fac})
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1"},
        {"req_id": "R2", "casualty_id": "P2"},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    n_assigned = out["assigned_amb_id"].notna().sum()
    assert n_assigned == 1


def test_ilp_capacity_two_assigns_both():
    """
    Ambulance capacity = 2 with two pending requests, both eligible for same facility.
    Both requests must be assigned to the same ambulance and destination.
    """
    amb = MockAmb("AMB1", 2, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(amb_map={"AMB1": amb}, fac_map={"F2": fac})
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1"},
        {"req_id": "R2", "casualty_id": "P2"},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assert out["assigned_amb_id"].notna().sum() == 2
    assert (out["assigned_amb_id"] == "AMB1").all()


def test_ilp_same_destination_constraint():
    """
    One ambulance (cap=2), two patients, both eligible for F2 and F3.
    F2 is closer (5 km), F3 is farther (20 km).
    With gamma=1 and same-destination constraint, both patients must go to F2.
    """
    amb = MockAmb("AMB1", 2, "AXP1")
    fac_f2 = MockFacility("F2")
    fac_f3 = MockFacility("F3")
    env = _make_ilp_env(
        amb_map={"AMB1": amb},
        fac_map={"F2": fac_f2, "F3": fac_f3},
    )
    dm = _dist(
        ["F1", "F2", "F3", "AXP1"],
        {("F1", "F2"): 5, ("F1", "F3"): 20, ("AXP1", "F1"): 0},
    )
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1", "eligible_facilities": ["F2", "F3"]},
        {"req_id": "R2", "casualty_id": "P2", "eligible_facilities": ["F2", "F3"]},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm, gamma=1.0)
    assigned = out[out["assigned_facility_id"].notna()]
    assert len(assigned) == 2
    # same-destination: both go to the same (cheaper) facility
    assert assigned["assigned_facility_id"].nunique() == 1
    assert assigned["assigned_facility_id"].iloc[0] == "F2"


def test_ilp_two_ambulances_two_requests_both_assigned():
    """
    Two ambulances (cap=1 each), two requests → each ambulance gets one request.
    """
    amb1 = MockAmb("AMB1", 1, "AXP1")
    amb2 = MockAmb("AMB2", 1, "AXP2")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        amb_map={"AMB1": amb1, "AMB2": amb2},
        fac_map={"F2": fac},
    )
    dm = _dist(["F1", "F2", "AXP1", "AXP2"], {
        ("F1", "F2"): 10, ("AXP1", "F1"): 0, ("AXP2", "F1"): 0,
    })
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1", "eligible_ambulances": ["AMB1", "AMB2"]},
        {"req_id": "R2", "casualty_id": "P2", "eligible_ambulances": ["AMB1", "AMB2"]},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb1, amb2], travel_time_matrix=dm)
    assert out["assigned_amb_id"].notna().sum() == 2
    # each ambulance assigned at most once (capacity 1)
    assert out["assigned_amb_id"].nunique() == 2


def test_ilp_delta_selects_lower_ewt_facility():
    """
    Two eligible facilities: F2 (EWT=0) and F3 (EWT=100).
    With delta=1, F2 must be preferred.
    """
    amb = MockAmb("AMB1", 1, "AXP1")
    fac_f2 = MockFacility("F2", units={"surgery": MockUnit(ewt=0.0)})
    fac_f3 = MockFacility("F3", units={"surgery": MockUnit(ewt=100.0)})
    env = _make_ilp_env(
        amb_map={"AMB1": amb},
        fac_map={"F2": fac_f2, "F3": fac_f3},
    )
    dm = _dist(["F1", "F2", "F3", "AXP1"], {
        ("F1", "F2"): 10, ("F1", "F3"): 10, ("AXP1", "F1"): 0,
    })
    df = _requests_df({
        "eligible_facilities": ["F2", "F3"],
        "needed_unit": "surgery",
    })
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm, delta=1.0)
    assert out.loc[0, "assigned_facility_id"] == "F2"


def test_ilp_unknown_triage_defaults_to_weight_one():
    """
    Patient with unknown triage category "T4" falls back to weight 1.0,
    the same as T3.  Competing against a T1 (weight 3) patient with only one
    ambulance slot, the T1 must win.
    """
    amb = MockAmb("AMB1", 1, "AXP1")
    fac = MockFacility("F2")
    env = _make_ilp_env(
        triage_weights={"T1": 3.0, "T2": 2.0, "T3": 1.0},   # no T4 key
        amb_map={"AMB1": amb},
        fac_map={"F2": fac},
    )
    dm = _dist(["F1", "F2", "AXP1"], {("F1", "F2"): 10, ("AXP1", "F1"): 0})
    df = _requests_df(
        {"req_id": "R1", "casualty_id": "P1", "triage": "T1"},
        {"req_id": "R2", "casualty_id": "P2", "triage": "T4"},
    )
    out = solve_ilp(env, df, idle_ambulances=[amb], travel_time_matrix=dm)
    assigned = out[out["assigned_amb_id"].notna()]
    assert assigned.iloc[0]["triage"] == "T1"


# =============================================================================
# 4. CRAFT_PLANS
# =============================================================================

def _make_craft_setup(n_patients: int = 1, amb_capacity: int = 1):
    """
    Build a minimal mock state for craft_plans:
      - AMB1 with given capacity
      - F1 (pickup) with n_patients requests in exit_queue
      - F2 (dropoff)
    Returns (c2, df) where df is a valid assigned_ambulances_df.
    """
    # mock patients and requests in pickup facility's exit_queue
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

    amb = SimpleNamespace(amb_id="AMB1", capacity=amb_capacity)

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
    """One patient → plan has exactly two steps: pickup and dropoff."""
    c2, df, patients, _ = _make_craft_setup(n_patients=1)
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    assert "AMB1" in plans
    plan = plans["AMB1"]
    assert len(plan) == 2
    pickup_step, dropoff_step = plan
    assert pickup_step.facility.loc_id == "F1"
    assert dropoff_step.facility.loc_id == "F2"


def test_craft_plans_patients_in_load_and_unload():
    """
    The pickup step loads the patient; the dropoff step unloads them.
    """
    c2, df, patients, _ = _make_craft_setup(n_patients=1)
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    pickup_step, dropoff_step = plans["AMB1"]
    assert patients[0] in pickup_step.load_patients
    assert patients[0] in dropoff_step.unload_patients


def test_craft_plans_missing_columns_raises():
    """Missing required column raises ValueError."""
    c2 = SimpleNamespace(env=SimpleNamespace(facility_by_id={}, amb_by_id={}))
    df = pd.DataFrame([{"req_id": "R1"}])   # missing issuing_facility_id etc.
    try:
        craft_plans(c2=c2, assigned_ambulances_df=df)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_craft_plans_empty_df():
    """Empty assignment DataFrame → empty plans dict."""
    c2 = SimpleNamespace(env=SimpleNamespace(facility_by_id={}, amb_by_id={}))
    df = pd.DataFrame()
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    assert plans == {}


def test_craft_plans_respects_ambulance_capacity():
    """
    Ambulance capacity = 1 but two patients are assigned in the DataFrame.
    craft_plans must load at most 1 patient.
    """
    c2, df, patients, _ = _make_craft_setup(n_patients=2, amb_capacity=1)
    plans = craft_plans(c2=c2, assigned_ambulances_df=df)
    pickup_step = plans["AMB1"][0]
    assert len(pickup_step.load_patients) == 1


# =============================================================================
# 5. INTEGRATION TESTS
# =============================================================================
#
# These tests run the full salabim simulation (with SimulationObserver) and
# verify C2 dispatch behaviour end-to-end.
#
# Topology:
#   AXP1 ──0km── F1 (CCP, no unit)  ──10km──  F2 (Role1, surgery unit)
#   C2 is attached to F1.

def _make_integration_scenario(n_ambulances: int = 1, amb_capacity: int = 1):
    """
    Build DataFrames for a two-facility integration test.
    All ambulances start at AXP(s) co-located with F1.
    """
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
        {"facility_id": "F1", "facility_type": "CCP",   "c2_up_id": "C2",
         "unload_policy": "unload_planned", "load_policy": "load_planned"},
        {"facility_id": "F2", "facility_type": "Role1", "c2_up_id": None,
         "unload_policy": "unload_planned", "load_policy": "load_planned"},
    ])

    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": 10,                       # plenty of capacity
        "service_time_min": 1,
        "service_time_max": 1,
        "service_time_dist": sim.Uniform(1, 1),
    }])

    ambulance_df = pd.DataFrame([
        {"amb_id": f"AMB{i+1}", "capacity": amb_capacity,
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
    """Build a casualties DataFrame for integration tests."""
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
            "lifetime_dist": sim.Uniform(lifetime, lifetime),
            "injury_time": r.get("t_create", 0),
        })
    return pd.DataFrame(records)


def _run_integration(data, casualties_df, triage_weights, levers=None, till=200):
    """
    Run a full simulation using SimulationObserver and return
    (transport_df, env).
    """
    env = sim.Environment(trace=False)
    env.levers = {
        "facility_eligibility": 0,
        "ambulance_eligibility": 0,
        "beta": 0.0,
        "gamma": 1.0,
        "delta": 0.0,
        "random_seed": 42,
        **(levers or {}),
    }
    env.triage_weights = triage_weights
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    with SimulationObserver(env) as observer:
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


def test_integration_t1_dispatched_before_t3():
    """
    T1 and T3 patients spawn simultaneously at F1.
    One ambulance (cap=1).  With w_T1=100 >> w_T3=1, the ILP must dispatch
    T1 first; it arrives at F2 before T3 does.
    """
    data = _make_integration_scenario(n_ambulances=1, amb_capacity=1)
    casualties = _casualties([
        {"triage": "T1", "t_create": 0},
        {"triage": "T3", "t_create": 0},
    ])
    transport_df, env = _run_integration(
        data, casualties,
        triage_weights={"T1": 100.0, "T2": 10.0, "T3": 1.0},
    )

    arrivals_at_f2 = transport_df[
        (transport_df["event"] == "arrival") & (transport_df["to_loc"] == "F2")
    ].sort_values("time")

    assert len(arrivals_at_f2) >= 2, "Expected at least 2 arrivals at F2"

    first_arrival = arrivals_at_f2.iloc[0]
    second_arrival = arrivals_at_f2.iloc[1]

    # First arrival carries T1 onboard
    assert "T1" in first_arrival["patient_triages"], (
        f"Expected T1 in first arrival, got {first_arrival['patient_triages']}"
    )
    # Second arrival carries T3
    assert "T3" in second_arrival["patient_triages"]
    # T1 arrived strictly before T3
    assert first_arrival["time"] < second_arrival["time"]


def test_integration_second_patient_dispatched_after_return():
    """
    Two patients, one ambulance (cap=1).
    After delivering the first patient the ambulance returns idle and
    picks up the second patient.  Both must eventually arrive at F2.
    """
    data = _make_integration_scenario(n_ambulances=1, amb_capacity=1)
    casualties = _casualties([
        {"triage": "T1", "t_create": 0},
        {"triage": "T1", "t_create": 0},
    ])
    transport_df, env = _run_integration(
        data, casualties,
        triage_weights={"T1": 3.0, "T2": 2.0, "T3": 1.0},
        till=300,
    )

    arrivals_at_f2 = transport_df[
        (transport_df["event"] == "arrival") & (transport_df["to_loc"] == "F2")
    ]
    # Both patients must reach F2
    assert len(arrivals_at_f2) == 2

    times = sorted(arrivals_at_f2["time"].tolist())
    # Second trip takes longer (amb goes F2→F1→F2), so arrivals must differ
    assert times[1] > times[0] + 5   # at least one extra leg (~10 min)


def test_integration_two_ambulances_dispatch_in_parallel():
    """
    Two ambulances, two patients spawned at the same time.
    Both should be dispatched in the same ILP batch and arrive at F2
    at approximately the same time (within a few simulated minutes).
    """
    data = _make_integration_scenario(n_ambulances=2, amb_capacity=1)
    casualties = _casualties([
        {"triage": "T1", "t_create": 0},
        {"triage": "T2", "t_create": 0},
    ])
    transport_df, env = _run_integration(
        data, casualties,
        triage_weights={"T1": 3.0, "T2": 2.0, "T3": 1.0},
    )

    arrivals_at_f2 = transport_df[
        (transport_df["event"] == "arrival") & (transport_df["to_loc"] == "F2")
    ].sort_values("time")

    # Both patients delivered
    assert len(arrivals_at_f2) == 2, (
        f"Expected 2 arrivals at F2, got {len(arrivals_at_f2)}.\n"
        f"All transport events:\n{transport_df.to_string()}"
    )

    amb_ids_used = set(arrivals_at_f2["amb_id"].tolist())
    # Each patient was carried by a different ambulance (parallel dispatch)
    assert len(amb_ids_used) == 2, (
        f"Expected 2 different ambulances, got {amb_ids_used}.\n"
        f"Arrivals at F2:\n{arrivals_at_f2.to_string()}"
    )

    times = sorted(arrivals_at_f2["time"].tolist())
    # Arrivals are close in time (both dispatched in the same 2-min window)
    assert abs(times[1] - times[0]) < 5, (
        f"Expected arrivals within 5 min of each other, got times={times}"
    )


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
        # solve_ilp
        ("ilp: empty df",                          test_ilp_empty_df),
        ("ilp: single request assigned",           test_ilp_single_request_assigned),
        ("ilp: no eligible facilities",            test_ilp_no_eligible_facilities),
        ("ilp: no eligible ambulances",            test_ilp_no_eligible_ambulances),
        ("ilp: T1 beats T3 (triage weights)",      test_ilp_triage_priority_t1_beats_t3),
        ("ilp: T2 beats T3 (triage weights)",      test_ilp_triage_priority_t2_beats_t3),
        ("ilp: sojourn tiebreaker",                test_ilp_sojourn_tiebreaker),
        ("ilp: triage dominates large sojourn",    test_ilp_triage_dominates_sojourn_when_gap_large),
        ("ilp: sojourn crosses breakeven",         test_ilp_sojourn_crosses_breakeven_t3_wins),
        ("ilp: gamma selects closer ambulance",    test_ilp_gamma_selects_closer_ambulance),
        ("ilp: capacity=1 limits assignment",      test_ilp_capacity_one_limits_assignment),
        ("ilp: capacity=2 assigns both",           test_ilp_capacity_two_assigns_both),
        ("ilp: same-destination constraint",       test_ilp_same_destination_constraint),
        ("ilp: two ambs two requests",             test_ilp_two_ambulances_two_requests_both_assigned),
        ("ilp: delta selects lower-EWT facility",  test_ilp_delta_selects_lower_ewt_facility),
        ("ilp: unknown triage defaults to 1.0",    test_ilp_unknown_triage_defaults_to_weight_one),
        # craft_plans
        ("craft_plans: basic structure",           test_craft_plans_basic_structure),
        ("craft_plans: patients in load/unload",   test_craft_plans_patients_in_load_and_unload),
        ("craft_plans: missing columns raises",    test_craft_plans_missing_columns_raises),
        ("craft_plans: empty df",                  test_craft_plans_empty_df),
        ("craft_plans: respects capacity",         test_craft_plans_respects_ambulance_capacity),
        # integration
        ("integration: T1 before T3",             test_integration_t1_dispatched_before_t3),
        ("integration: second patient after return", test_integration_second_patient_dispatched_after_return),
        ("integration: two ambs parallel",        test_integration_two_ambulances_dispatch_in_parallel),
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
