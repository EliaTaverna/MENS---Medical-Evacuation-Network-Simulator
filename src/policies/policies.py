"""Policy functions and registries for C2 decision making."""

from __future__ import annotations
import pandas as pd
import salabim as sim
from typing import List, Optional, TYPE_CHECKING

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
# FACILITY ELIGIBILITY
# ============================================================

def fac_eligibility(env: sim.Environment, requests_df: pd.DataFrame, lever_value: int) -> pd.DataFrame:
    """
    Add column 'eligible_facilities' (list[str]) to requests_df.

    Reads from env.fac_eligibility_datasets[lever_value].
    Dataset format: DataFrame with issuing facilities as index, destination
    facilities as columns; value 1 if eligible, 0 otherwise.
    """
    df = requests_df.copy()
    df["eligible_facilities"] = [[] for _ in range(len(df))]

    if df.empty:
        return df

    matrix = (getattr(env, "fac_eligibility_datasets", {}) or {}).get(lever_value)
    if matrix is None or matrix.empty:
        return df

    def get_eligible(issuing_fac_id):
        if issuing_fac_id not in matrix.index:
            return []
        row = matrix.loc[issuing_fac_id]
        return [col for col in row.index if row[col] == 1]

    df["eligible_facilities"] = df["issuing_facility_id"].apply(get_eligible)
    return df


# ============================================================
# AMBULANCE ELIGIBILITY
# ============================================================

def amb_eligibility(
    env: sim.Environment,
    requests_df: pd.DataFrame,
    idle_ambulances: List["Ambulance"],
    lever_value: int,
) -> pd.DataFrame:
    """
    Add column 'eligible_ambulances' (list[str]) to requests_df.

    Reads from env.amb_eligibility_datasets[lever_value].
    Dataset format: DataFrame with ambulance IDs as index, facility IDs as columns;
    value 1 if that ambulance may serve requests issued from that facility.

    Called after fac_eligibility, before the dispatch solver.
    """
    df       = requests_df.copy()
    idle_ids = [a.amb_id for a in (idle_ambulances or [])]
    ae       = (getattr(env, "amb_eligibility_datasets", {}) or {}).get(lever_value)

    if ae is None or ae.empty:
        df["eligible_ambulances"] = [idle_ids[:] for _ in range(len(df))]
        return df

    def get_eligible(origin: str) -> list:
        return [
            amb_id for amb_id in idle_ids
            if amb_id in ae.index
            and origin in ae.columns
            and ae.loc[amb_id, origin] == 1
        ]

    df["eligible_ambulances"] = df["issuing_facility_id"].apply(get_eligible)
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


