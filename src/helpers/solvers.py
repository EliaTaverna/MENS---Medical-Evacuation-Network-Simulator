"""Dispatch solvers for the C2 assignment step.

Registry pattern: add a new solver function here and register it in
c2.py::_DISPATCH_SOLVERS to make it available as a dispatch_policy value.

Each solver must have the signature:

    def solve_<name>(
        env,
        requests_df: pd.DataFrame,
        idle_ambulances: List[Ambulance],
        distance_matrix: Optional[pd.DataFrame],
        **kwargs,
    ) -> pd.DataFrame:

The returned DataFrame must contain 'assigned_facility_id' and
'assigned_amb_id' columns (None where unassigned).
"""

from __future__ import annotations

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


def _nearest_id(origin: str, candidates: list, distance_matrix,
                env=None, needed_unit: Optional[str] = None) -> Optional[str]:
    """Return the candidate id nearest to origin; break ties by queue length."""
    if not candidates:
        return None
    if distance_matrix is None or origin not in distance_matrix.index:
        return candidates[0]

    dist_map = {}
    for cid in candidates:
        if cid in distance_matrix.columns:
            dist_map[cid] = float(distance_matrix.loc[origin, cid])
        else:
            dist_map[cid] = float("inf")

    min_d = min(dist_map.values())
    nearest = [cid for cid, d in dist_map.items() if d == min_d]

    if len(nearest) == 1:
        return nearest[0]

    if env is not None and needed_unit is not None:
        def _queue_len(cid):
            fac = getattr(env, "facility_by_id", {}).get(cid)
            if fac is None:
                return 0
            return len(fac.entry_lines.get(needed_unit, []))
        nearest.sort(key=_queue_len)

    return nearest[0]


def _least_congested_id(origin: str, candidates: list, distance_matrix,
                        env=None, needed_unit: Optional[str] = None) -> Optional[str]:
    """Return the eligible facility with the shortest expected wait for needed_unit.

    Primary key : expected_wait_if_join_now() on the relevant Unit.
    Tiebreaker  : distance from origin.
    Falls back to _nearest_id when env / needed_unit are unavailable.
    """
    if not candidates:
        return None

    if env is None or needed_unit is None:
        return _nearest_id(origin, candidates, distance_matrix, env=env, needed_unit=needed_unit)

    facility_by_id = getattr(env, "facility_by_id", {})

    def _expected_wait(cid):
        fac = facility_by_id.get(cid)
        if fac is None:
            return float("inf")
        unit = getattr(fac, "units", {}).get(needed_unit)
        if unit is None:
            return float("inf")
        return unit.expected_wait_if_join_now()

    def _dist(cid):
        if distance_matrix is None or origin not in distance_matrix.index:
            return 0.0
        return float(distance_matrix.loc[origin, cid]) if cid in distance_matrix.columns else float("inf")

    return min(candidates, key=lambda cid: (_expected_wait(cid), _dist(cid)))


# ---------------------------------------------------------------------------
# Greedy solver
# ---------------------------------------------------------------------------

def solve_greedy(
    env: "sim.Environment",
    requests_df: pd.DataFrame,
    idle_ambulances: List["Ambulance"],
    distance_matrix: Optional[pd.DataFrame],
    **_kwargs,
) -> pd.DataFrame:
    """
    Greedy dispatcher: FIFO + triage priority, joint (ambulance, destination) selection.

    For each pending request batch:
      1. Pick the anchor: longest-waiting request (tie-break: T1 > T2 > T3).
      2. Jointly select (amb, dest) minimising:
             pickup_time(amb → origin)
           + transport_time(origin → dest)
           + expected_wait_if_join_now(dest)
      3. Bundle all co-located requests eligible for the chosen ambulance (up to capacity),
         all sent to the same destination.
      4. Repeat until no idle ambulances or no pending requests remain.

    Returns requests_df with assigned_facility_id / assigned_amb_id filled.
    """
    df = requests_df.copy()
    df["assigned_facility_id"] = None
    df["assigned_amb_id"]      = None

    if df.empty or not idle_ambulances:
        return df

    caps      = {a.amb_id: int(getattr(a, "capacity", 1)) for a in idle_ambulances}
    idle_pool = list(idle_ambulances)
    assigned  = pd.Series(False, index=df.index)

    df["_t_issued"]    = pd.to_numeric(df["t_issued"], errors="coerce").fillna(0)
    df["_triage_prio"] = df["triage"].apply(_triage_priority)
    df = df.sort_values(["_t_issued", "_triage_prio"], ascending=[True, True])

    _speed_kmh = getattr(idle_ambulances[0], "speed_kmh", 30)
    _km_to_min = 60.0 / _speed_kmh

    def _travel_min(loc_a: str, loc_b: str) -> float:
        if distance_matrix is None:
            return 0.0
        if loc_a not in distance_matrix.index or loc_b not in distance_matrix.columns:
            return 0.0
        return float(distance_matrix.loc[loc_a, loc_b]) * _km_to_min

    def _expected_wait(dest: str, needed_unit: Optional[str]) -> float:
        if env is None or needed_unit is None:
            return 0.0
        fac = getattr(env, "facility_by_id", {}).get(dest)
        if fac is None:
            return 0.0
        unit = getattr(fac, "units", {}).get(needed_unit)
        if unit is None:
            return 0.0
        return unit.expected_wait_if_join_now()

    while idle_pool:
        pending = df[~assigned]
        if pending.empty:
            break

        anchor        = pending.iloc[0]
        origin        = anchor["issuing_facility_id"]
        eligible_facs = anchor.get("eligible_facilities") or []
        eligible_ambs = set(anchor.get("eligible_ambulances") or [])
        needed_unit   = anchor.get("needed_unit")

        if not eligible_facs or not eligible_ambs:
            assigned[anchor.name] = True
            continue

        candidates = [a for a in idle_pool if a.amb_id in eligible_ambs]
        if not candidates:
            assigned[anchor.name] = True
            continue

        best_dest = min(
            eligible_facs,
            key=lambda dest: _travel_min(origin, dest) + _expected_wait(dest, needed_unit),
        )
        best_amb = min(
            candidates,
            key=lambda amb: _travel_min(amb._loc_id(amb.location), origin),
        )

        if best_amb is None or best_dest is None:
            assigned[anchor.name] = True
            continue

        cap         = caps.get(best_amb.amb_id, 1)
        same_origin = pending[
            (pending["issuing_facility_id"] == origin)
            & pending["eligible_ambulances"].apply(lambda ea: best_amb.amb_id in (ea or []))
        ].head(cap)

        for idx in same_origin.index:
            df.loc[idx, "assigned_facility_id"] = best_dest
            df.loc[idx, "assigned_amb_id"]      = best_amb.amb_id
            assigned[idx] = True

        idle_pool.remove(best_amb)

    df.drop(columns=["_t_issued", "_triage_prio"], inplace=True, errors="ignore")
    return df
