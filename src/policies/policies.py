"""Policy functions and registries for C2 decision making."""

from __future__ import annotations
import random
import pandas as pd
import salabim as sim
from typing import Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.components.ambulance import Ambulance
    from src.components.facility import Facility
    from src.components.c2 import TransportC2

from src.components.requests import PlanStep


# ============================================================
# LOAD/UNLOAD POLICIES
# ============================================================

def unload_planned(amb, facility):
    """Unload patients according to the current plan step."""
    step = amb.current_step
    return step.unload_patients if (step and step.facility is facility) else []


def load_planned(amb, facility):
    """Load patients according to the current plan step."""
    step = amb.current_step
    return step.load_patients if (step and step.facility is facility) else []


# ============================================================
# TYPE ALIASES
# ============================================================

# All policy functions receive env as first argument
FacilityEligibilityFn = Callable[["sim.Environment", pd.DataFrame, pd.DataFrame], pd.DataFrame]
FacilityAssignmentFn = Callable[["sim.Environment", pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame]
AmbEligibilityFn = Callable[["sim.Environment", pd.DataFrame, pd.DataFrame, List["Ambulance"]], pd.DataFrame]
AmbAssignmentFn = Callable[["sim.Environment", pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame]


# ============================================================
# FACILITY ELIGIBILITY POLICIES
# ============================================================

def fac_eligibility_base(env: sim.Environment, requests_df: pd.DataFrame, corridors_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Add column 'eligible_facilities' (list[str]) to requests_df.

    corridors: DataFrame with facilities as both index and columns.
    Value 1 if row facility can send to column facility, 0 otherwise.
    """
    df = requests_df.copy()
    df["eligible_facilities"] = [[] for _ in range(len(df))]

    if df.empty or corridors_matrix is None or corridors_matrix.empty:
        return df

    def get_eligible(issuing_fac_id):
        if issuing_fac_id not in corridors_matrix.index:
            return []
        row = corridors_matrix.loc[issuing_fac_id]
        return [col for col in row.index if row[col] == 1]

    df["eligible_facilities"] = df["issuing_facility_id"].apply(get_eligible)
    return df


# ============================================================
# FACILITY ASSIGNMENT POLICIES
# ============================================================

def fac_assign_template(
    env: sim.Environment,
    eligible_facilities_df: pd.DataFrame,
    facilities_df: pd.DataFrame,
    distance_matrix: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Choose randomly one facility among eligible_facilities.

    Adds column 'assigned_facility_id' and returns the enhanced df.
    """
    df = eligible_facilities_df.copy()

    if "eligible_facilities" not in df.columns:
        df["eligible_facilities"] = [[] for _ in range(len(df))]

    def pick_one(xs):
        return random.choice(xs) if isinstance(xs, list) and len(xs) > 0 else None

    df["assigned_facility_id"] = df["eligible_facilities"].apply(pick_one)
    return df


# ============================================================
# AMBULANCE ELIGIBILITY POLICIES
# ============================================================

def amb_eligibility_base(
    env: sim.Environment,
    assigned_facilities_df: pd.DataFrame,
    ambulances_eligibility: pd.DataFrame,
    idle_ambulances: List["Ambulance"],
) -> pd.DataFrame:
    """
    Add column 'eligible_ambulances' (list[str]) to assigned_facilities_df.

    ambulances_eligibility: DataFrame with ambulances as index, facilities as columns.
    Value 1 if ambulance serves that facility, 0 otherwise.

    Eligible ambulance = idle now AND serves both issuing_facility_id and assigned_facility_id.
    """
    df = assigned_facilities_df.copy()
    df["eligible_ambulances"] = [[] for _ in range(len(df))]

    if df.empty:
        return df

    required_req = {"issuing_facility_id", "assigned_facility_id"}
    if not required_req.issubset(df.columns):
        return df

    idle_ids = [a.amb_id for a in (idle_ambulances or [])]

    # If no constraint table, every idle ambulance is eligible
    if ambulances_eligibility is None or ambulances_eligibility.empty:
        df["eligible_ambulances"] = [idle_ids[:] for _ in range(len(df))]
        return df

    ae = ambulances_eligibility

    def get_eligible_ambs(row):
        issuing = row["issuing_facility_id"]
        assigned = row["assigned_facility_id"]
        eligible = []
        for amb_id in idle_ids:
            if amb_id not in ae.index:
                continue
            # Ambulance must serve both pickup and dropoff facilities
            serves_issuing = issuing in ae.columns and ae.loc[amb_id, issuing] == 1
            serves_assigned = assigned in ae.columns and ae.loc[amb_id, assigned] == 1
            if serves_issuing and serves_assigned:
                eligible.append(amb_id)
        return eligible

    df["eligible_ambulances"] = df.apply(get_eligible_ambs, axis=1)
    return df


# ============================================================
# AMBULANCE ASSIGNMENT POLICIES
# ============================================================

def amb_assign_template(
    env: sim.Environment,
    eligible_pairs_df: pd.DataFrame,
    distance_matrix: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Choose randomly one ambulance among eligible_ambulances.

    Adds column 'assigned_amb_id' and returns the enhanced df.
    """
    df = eligible_pairs_df.copy()

    if "eligible_ambulances" not in df.columns:
        df["eligible_ambulances"] = [[] for _ in range(len(df))]

    def pick_one(xs):
        return random.choice(xs) if isinstance(xs, list) and len(xs) > 0 else None

    df["assigned_amb_id"] = df["eligible_ambulances"].apply(pick_one)
    return df


# ============================================================
# PLAN CRAFTING AND DISPATCH
# ============================================================

def craft_plans(
    c2: "TransportC2",
    assigned_ambulances_df: pd.DataFrame,
) -> dict[str, list["PlanStep"]]:
    """
    PURE planner: returns plans, does NOT dispatch/activate.

    Input must contain:
      - req_id
      - issuing_facility_id
      - assigned_facility_id
      - assigned_amb_id

    Output:
      { amb_id: [PlanStep(pickup_facility), PlanStep(dropoff_facility)] }
    """
    plans: dict[str, list["PlanStep"]] = {}

    if assigned_ambulances_df is None or assigned_ambulances_df.empty:
        return plans

    df = assigned_ambulances_df.copy()

    required = {"req_id", "issuing_facility_id", "assigned_facility_id", "assigned_amb_id"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"send_plans missing columns: {missing}")

    env = c2.env

    def req_map_for_facility(fac):
        return {getattr(r, "req_id", None): r for r in getattr(fac, "exit_queue", [])}

    for amb_id, g in df.groupby("assigned_amb_id", sort=False):
        if amb_id is None or (isinstance(amb_id, float) and pd.isna(amb_id)):
            continue
        amb_id = str(amb_id)

        amb = env.amb_by_id.get(amb_id)
        if amb is None:
            continue

        # pickup facility = the facility with most cargo for this ambulance
        pickup_facility_id = g["issuing_facility_id"].value_counts().index[0]
        gg = g[g["issuing_facility_id"] == pickup_facility_id].copy()
        if gg.empty:
            continue

        # choose up to capacity reqs from this pickup facility
        gg = gg.sample(frac=1).reset_index(drop=True)
        gg = gg.iloc[: min(int(amb.capacity), len(gg))]

        # dropoff facility = most common among the selected ones
        dropoff_facility_id = gg["assigned_facility_id"].value_counts().index[0]

        pickup_fac = env.facility_by_id.get(pickup_facility_id)
        dropoff_fac = env.facility_by_id.get(dropoff_facility_id)
        if pickup_fac is None or dropoff_fac is None:
            continue

        # resolve Patient objects from pickup_fac.exit_queue using req_ids
        req_ids = gg["req_id"].astype(str).tolist()
        rmap = req_map_for_facility(pickup_fac)
        req_objs = [rmap.get(rid) for rid in req_ids]
        req_objs = [r for r in req_objs if r is not None and getattr(r, "patient", None) is not None]
        patients = [r.patient for r in req_objs]

        if not patients:
            continue

        # Mark actual request objects as assigned
        for req in req_objs:
            req.status = "assigned"

        pickup_step = PlanStep(facility=pickup_fac, unload_patients=[], load_patients=patients)
        dropoff_step = PlanStep(facility=dropoff_fac, unload_patients=patients, load_patients=[])

        plans[amb_id] = [pickup_step, dropoff_step]

    return plans


def dispatch_ambulances(c2: "TransportC2", plans_craft: dict[str, list["PlanStep"]]):
    """
    Equip ambulances with a plan and dispatch them (activate).
    """
    env = c2.env

    for amb_id, plan in (plans_craft or {}).items():
        amb = env.amb_by_id.get(amb_id)
        if amb is None:
            continue

        # avoid double dispatch
        if amb not in c2.idle_ambulances:
            continue

        # attach plan
        amb.plan = plan

        # mark ambulance busy
        c2.idle_ambulances.remove(amb)

        # mark requests assigned
        loaded_patient_ids = {p.patient_id for step in plan for p in step.load_patients}
        if loaded_patient_ids:
            mask = (
                (c2.pending_transport_requests_df["status"] == "pending")
                & (c2.pending_transport_requests_df["casualty_id"].astype(str).isin(
                    {str(x) for x in loaded_patient_ids}))
            )
            c2.pending_transport_requests_df.loc[mask, "status"] = "assigned"

        # dispatch
        amb.activate()


# ============================================================
# FACILITY SNAPSHOT
# ============================================================

def snapshot_facilities(facilities: List["Facility"]) -> pd.DataFrame:
    """
    Snapshot current facility state into a DataFrame.

    Spec:
      - One row per facility
      - One column per service type (global set across all facilities)
      - Cell = expected waiting time *if someone joins that service queue now*
      - Cell = False if that service is not present in the facility
    """
    if not facilities:
        return pd.DataFrame()

    # 1) Collect the global set of service types across all facilities
    all_service_types = set()
    for fac in facilities:
        all_service_types.update(fac.units.keys())
    all_service_types = sorted(all_service_types)

    # 2) Build rows
    rows = []
    for fac in facilities:
        row = {"facility_id": fac.loc_id}

        for service_type in all_service_types:
            u = fac.units.get(service_type, None)

            if u is None:
                row[service_type] = False
            else:
                row[service_type] = float(u.expected_wait_if_join_now())

        rows.append(row)

    df = pd.DataFrame(rows)

    return df


# ============================================================
# POLICY REGISTRIES
# ============================================================

FACILITY_ELIGIBILITY_POLICIES: Dict[int, FacilityEligibilityFn] = {
    0: fac_eligibility_base,
}

FACILITY_ASSIGNMENT_POLICIES: Dict[int, FacilityAssignmentFn] = {
    0: fac_assign_template,
}

AMBULANCE_ELIGIBILITY_POLICIES: Dict[int, AmbEligibilityFn] = {
    0: amb_eligibility_base,
}

AMBULANCE_ASSIGNMENT_POLICIES: Dict[int, AmbAssignmentFn] = {
    0: amb_assign_template,
}
