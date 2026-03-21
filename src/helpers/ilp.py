"""Joint facility + ambulance ILP solver and greedy heuristic dispatcher."""

from __future__ import annotations
import pandas as pd
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import salabim as sim
    from src.components.ambulance import Ambulance

from src.helpers.snapshot import snapshot_facilities


# ---------------------------------------------------------------------------
# Helpers shared by both solvers
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

    # Tiebreaker: facility with shortest queue for the needed service type
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
# Greedy heuristic dispatcher
# ---------------------------------------------------------------------------

def solve_greedy(
    env: "sim.Environment",
    requests_df: pd.DataFrame,
    idle_ambulances: List["Ambulance"],
    distance_matrix: Optional[pd.DataFrame],
    **_kwargs,          # absorbs ILP-only params (beta, gamma, delta, M, …)
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

    # sort once: longest-waiting first, triage as tie-breaker
    df["_t_issued"]    = pd.to_numeric(df["t_issued"], errors="coerce").fillna(0)
    df["_triage_prio"] = df["triage"].apply(_triage_priority)
    df = df.sort_values(["_t_issued", "_triage_prio"], ascending=[True, True])

    # km → minutes conversion using first ambulance's speed (all share the same speed)
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

        # --- step 1: anchor = longest-waiting unassigned request ---
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

        # --- step 2: pick dest = min(transport_time + ewt), then nearest ambulance ---
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

        # --- step 3: bundle co-located requests for this ambulance ---
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


def solve_ilp(
    env: "sim.Environment",
    requests_df: pd.DataFrame,
    idle_ambulances: List["Ambulance"],
    distance_matrix: Optional[pd.DataFrame],
    beta:  float = 0.0,
    gamma: float = 0.0,
    delta: float = 0.0,
    M:     float = 10_000.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    4-term ILP with same-destination constraint.

    Reads eligible_facilities and eligible_ambulances from requests_df
    (populated by fac_eligibility and amb_eligibility respectively).

    Objective: maximise Σ x[r,a,f] · score(r,a,f)

        score(r, a, f) = M
                       + triage_weight(r)                 from env.triage_weights {T1, T2, T3}
                       + beta  * sojourn(r)               now - t_issued
                       - gamma * transport_time(r, a, f)  d(amb→origin) + d(origin→facility)
                       - delta * operation_time(r, f)     expected wait if joined now

    y[a,f] ∈ {0,1} enforces that all patients on the same ambulance share one destination.
    """
    try:
        from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, PULP_CBC_CMD
    except ImportError:
        raise ImportError("pulp is required: pip install pulp")

    df = requests_df.copy()
    if df.empty:
        df["assigned_facility_id"] = None
        df["assigned_amb_id"] = None
        return df

    fac_snap  = snapshot_facilities(list(env.facility_by_id.values()))
    fac_index = fac_snap.set_index("facility_id") if not fac_snap.empty else pd.DataFrame()
    caps      = {a.amb_id: int(getattr(a, "capacity", 1)) for a in idle_ambulances}
    req_idx        = df.index.tolist()
    _triage_weights = getattr(env, "triage_weights", {})

    prob        = LpProblem("ilp_4term", LpMaximize)
    x, y, costs = {}, {}, {}

    for idx, row in df.iterrows():
        eligible_facs = row.get("eligible_facilities") or []
        eligible_ambs = row.get("eligible_ambulances") or []
        origin        = row["issuing_facility_id"]
        unit_type     = row.get("needed_unit")
        triage_val    = _triage_weights.get(str(row.get("triage", "")), 1.0)
        sojourn       = max(float(env.now()) - float(row.get("t_issued", env.now())), 0.0)

        for amb_id in eligible_ambs:
            amb = env.amb_by_id.get(str(amb_id))
            if amb is None:
                continue

            amb_loc  = amb._loc_id(amb.location)
            d_pickup = (
                float(distance_matrix.loc[amb_loc, origin])
                if distance_matrix is not None
                   and amb_loc in distance_matrix.index
                   and origin  in distance_matrix.columns
                else 0.0
            )

            for fac_id in eligible_facs:
                d_transport = (
                    float(distance_matrix.loc[origin, fac_id])
                    if distance_matrix is not None
                       and origin in distance_matrix.index
                       and fac_id in distance_matrix.columns
                    else 0.0
                )

                ewt = 0.0
                if (
                    not fac_index.empty
                    and fac_id    in fac_index.index
                    and unit_type
                    and unit_type in fac_index.columns
                ):
                    val = fac_index.at[fac_id, unit_type]
                    if val is not False and not (isinstance(val, float) and pd.isna(val)):
                        ewt = float(val)

                key        = (idx, amb_id, fac_id)
                x[key]     = LpVariable(f"x_{idx}_{amb_id}_{fac_id}", cat="Binary")
                costs[key] = (
                    M
                    + triage_val
                    + beta  * sojourn
                    - gamma * (d_pickup + d_transport)
                    - delta * ewt
                )

                if (amb_id, fac_id) not in y:
                    y[(amb_id, fac_id)] = LpVariable(f"y_{amb_id}_{fac_id}", cat="Binary")

    if not x:
        df["assigned_facility_id"] = None
        df["assigned_amb_id"] = None
        return df

    # objective
    prob += lpSum(costs[k] * x[k] for k in x)

    # each request assigned at most once
    for idx in req_idx:
        vars_r = [x[k] for k in x if k[0] == idx]
        if vars_r:
            prob += lpSum(vars_r) <= 1

    # same-destination: x[r,a,f] <= cap(a) * y[a,f]  and  Σ_f y[a,f] <= 1
    for (amb_id, fac_id), yvar in y.items():
        vars_xaf = [x[k] for k in x if k[1] == amb_id and k[2] == fac_id]
        if vars_xaf:
            prob += lpSum(vars_xaf) <= caps.get(amb_id, 1) * yvar
    for amb_id in caps:
        vars_y = [y[k] for k in y if k[0] == amb_id]
        if vars_y:
            prob += lpSum(vars_y) <= 1

    # capacity: assignments per ambulance <= capacity
    for amb_id in caps:
        vars_a = [x[k] for k in x if k[1] == amb_id]
        if vars_a:
            prob += lpSum(vars_a) <= caps[amb_id]

    prob.solve(PULP_CBC_CMD(msg=0, timeLimit=5, options=[f"RandomS {random_seed}"]))

    af = {i: None for i in req_idx}
    aa = {i: None for i in req_idx}
    for (idx, amb_id, fac_id), var in x.items():
        if (value(var) or 0) > 0.5:
            af[idx] = fac_id
            aa[idx] = amb_id

    df["assigned_facility_id"] = [af[i] for i in req_idx]
    df["assigned_amb_id"]      = [aa[i] for i in req_idx]
    return df
