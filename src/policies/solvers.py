"""Dispatch solvers for the C2 assignment step.

Each solver must have the signature:

    def solve_<name>(
        env,
        requests_df: pd.DataFrame,
        idle_ambulances: List[Ambulance],
        travel_time_matrix: Optional[pd.DataFrame],
        **kwargs,
    ) -> pd.DataFrame:

The returned DataFrame must contain 'assigned_facility_id' and
'assigned_amb_id' columns (None where unassigned).
"""

from __future__ import annotations

import random
import pandas as pd
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import salabim as sim
    from src.components.ambulance import Ambulance


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _triage_priority(triage: str) -> int:
    """Lower value = higher urgency."""
    t = str(triage)
    if "T1" in t: return 0
    if "T2" in t: return 1
    if "T3" in t: return 2
    return 99


def _travel_min(loc_a: str, loc_b: str, travel_time_matrix) -> float:
    if travel_time_matrix is None:
        return 0.0
    if loc_a not in travel_time_matrix.index or loc_b not in travel_time_matrix.columns:
        return 0.0
    return float(travel_time_matrix.loc[loc_a, loc_b])


def _expected_wait(dest: str, needed_unit: Optional[str], env) -> float:
    if env is None or needed_unit is None:
        return 0.0
    fac = getattr(env, "facility_by_id", {}).get(dest)
    if fac is None:
        return 0.0
    unit = getattr(fac, "units", {}).get(needed_unit)
    if unit is None:
        return 0.0
    return unit.expected_wait_if_join_now()


def _pick_nearest_ambulance(candidates: list, origin: str, travel_time_matrix):
    """Pick the nearest ambulance to origin; random tiebreak among equidistant."""
    if not candidates:
        return None

    def _dist(amb):
        return _travel_min(amb._loc_id(amb.location), origin, travel_time_matrix)

    min_d = min(_dist(a) for a in candidates)
    nearest = [a for a in candidates if _dist(a) == min_d]
    return random.choice(nearest)


def _pick_best_destination(origin: str, eligible_facs: list, needed_unit, env, travel_time_matrix):
    """Pick destination minimizing transport_time + expected_wait."""
    if not eligible_facs:
        return None
    return min(
        eligible_facs,
        key=lambda dest: (
            _travel_min(origin, dest, travel_time_matrix)
            + _expected_wait(dest, needed_unit, env)
        ),
    )


# ---------------------------------------------------------------------------
# Triage-based solver
# ---------------------------------------------------------------------------

def solve_triage(
    env: "sim.Environment",
    requests_df: pd.DataFrame,
    idle_ambulances: List["Ambulance"],
    travel_time_matrix: Optional[pd.DataFrame],
    trigger_type: str = "t1",
    **_kwargs,
) -> pd.DataFrame:
    """
    Triage-aware dispatcher.

    The C2 has already selected a single facility's worth of requests.
    This solver:
      1. Picks the best destination (min transport_time + expected_wait).
      2. Picks the nearest idle ambulance (random tiebreak).
      3. Loads the ambulance respecting triage priority and capacity:
         - T1/T2 slots filled first (T1 > T2 by t_issued)
         - T3 slots filled with T3s
      4. Only one ambulance is dispatched per call.

    Returns requests_df with assigned_facility_id / assigned_amb_id filled.
    """
    df = requests_df.copy()
    df["assigned_facility_id"] = None
    df["assigned_amb_id"] = None

    if df.empty or not idle_ambulances:
        return df

    origin = df["issuing_facility_id"].iloc[0]
    needed_unit = df["needed_unit"].iloc[0] if "needed_unit" in df.columns else None

    # --- pick destination ---
    eligible_facs = df["eligible_facilities"].iloc[0] if "eligible_facilities" in df.columns else []
    if not eligible_facs:
        eligible_facs_all = df["eligible_facilities"].dropna()
        for facs in eligible_facs_all:
            if facs:
                eligible_facs = facs
                break
    if not eligible_facs:
        return df

    best_dest = _pick_best_destination(origin, eligible_facs, needed_unit, env, travel_time_matrix)
    if best_dest is None:
        return df

    # --- pick ambulance ---
    eligible_amb_ids = set()
    for _, row in df.iterrows():
        ea = row.get("eligible_ambulances") or []
        eligible_amb_ids.update(ea)

    candidates = [a for a in idle_ambulances if a.amb_id in eligible_amb_ids]
    if not candidates:
        return df

    best_amb = _pick_nearest_ambulance(candidates, origin, travel_time_matrix)
    if best_amb is None:
        return df

    # --- load ambulance by triage priority ---
    t1t2_cap = int(getattr(best_amb, "t1_t2_capacity", 1))
    t3_cap = int(getattr(best_amb, "t3_capacity", 1))

    df["_triage_prio"] = df["triage"].apply(_triage_priority)
    df["_t_issued"] = pd.to_numeric(df["t_issued"], errors="coerce").fillna(0)
    df = df.sort_values(["_triage_prio", "_t_issued"], ascending=[True, True])

    t1t2_rows = df[df["triage"].str.startswith(("T1", "T2"))].head(t1t2_cap)
    t3_rows = df[df["triage"].str.startswith("T3")].head(t3_cap)
    loaded = pd.concat([t1t2_rows, t3_rows])

    for idx in loaded.index:
        df.loc[idx, "assigned_facility_id"] = best_dest
        df.loc[idx, "assigned_amb_id"] = best_amb.amb_id

    df.drop(columns=["_triage_prio", "_t_issued"], inplace=True, errors="ignore")
    return df
