"""Transport Command and Control (C2) component."""

from __future__ import annotations
import json
import pandas as pd
import salabim as sim
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.components.ambulance import Ambulance
    from src.components.facility import Facility

from src.components.requests import TransportRequest
from src.policies.policies import (
    fac_eligibility,
    amb_eligibility,
    craft_plans,
    dispatch_ambulances,
)
from src.policies.solvers import solve_triage

_T2_WAIT_THRESHOLD = 240  # minutes (4 hours) before a lone T2 triggers dispatch
_T3_WAIT_THRESHOLD = 720  # minutes (12 hours) before T3s trigger dispatch

_DISPATCH_SOLVERS = {
    "greedy": solve_triage,
}


class TransportC2(sim.Component):
    """
    C2 dispatcher for one evacuation leg.

    Wakes on two events:
      - A facility issues a T1 or qualifying T2 request  (receive_request)
      - An ambulance reports idle                          (ambulance_idle)

    Immediate dispatch rules (on request arrival):
      - T1: dispatch nearest idle ambulance immediately.
      - T2: dispatch only if another T2 is already waiting at the same facility.
      - T3: ignored (picked up by idle-ambulance polling).

    On ambulance idle (priority order):
      1. Any T1 waiting         -> dispatch to the facility with the longest-waiting T1
      2. Any facility with 2+ T2s -> dispatch there
      3. Hold until next threshold crossing:
         - T2 waiting > _T2_WAIT_THRESHOLD  -> dispatch, bundle T3s
         - T3 waiting > _T3_WAIT_THRESHOLD  -> dispatch, bundle other T3s

    No order queue: if no ambulance is available the request is ignored
    and reconsidered when an ambulance becomes idle.
    """

    def setup(
        self,
        c2_id: str,
        facility_ids: list[str],
        ambulance_ids: list[str],
        lever_value_facility_eligibility: int = 0,
        lever_value_ambulance_eligibility: int = 0,
        dispatch_policy: str = "greedy",
    ):
        self.c2_id = c2_id
        self.facility_ids = list(facility_ids)
        self.ambulance_ids = list(ambulance_ids)

        self.facilities: List["Facility"] = []
        self.ambulances: List["Ambulance"] = []
        self.idle_ambulances: List["Ambulance"] = []

        self._pending_rows: list[dict] = []
        self.pending_transport_requests_df = self._init_pending_requests_df()

        self.lever_value_facility_eligibility = int(lever_value_facility_eligibility)
        self.lever_value_ambulance_eligibility = int(lever_value_ambulance_eligibility)

        self.dispatch_policy = dispatch_policy
        self._wake_reason = "init"

    # --------------------------------------------------------
    # inbound events
    # --------------------------------------------------------

    def receive_request(self, req: "TransportRequest"):
        """Append request. Only wake C2 for T1 or a qualifying T2 pair."""
        issuing_facility_id = getattr(getattr(req, "issuing_facility", None), "loc_id", None) \
                              or getattr(req, "issuing_facility_id", None)
        triage = getattr(req, "triage", None) or ""

        row = {
            "req_id":              getattr(req, "req_id", None),
            "casualty_id":         getattr(req, "casualty_id", None) or getattr(req.patient, "patient_id", None),
            "triage":              triage,
            "issuing_facility_id": issuing_facility_id,
            "injury_time":         getattr(req, "injury_time", None),
            "t_issued":            getattr(req, "t_issued", None),
            "needed_unit":         getattr(req, "needed_unit", None),
            "status":              getattr(req, "status", "pending"),
        }

        self._pending_rows.append(row)

        should_wake = False
        if str(triage).startswith("T1"):
            should_wake = True
        elif str(triage).startswith("T2"):
            should_wake = self._has_pending_t2_at(issuing_facility_id)

        if should_wake:
            self._wake_reason = "request"
            self.activate()

    def _has_pending_t2_at(self, facility_id: str) -> bool:
        """Check if there's already a pending T2 at the given facility."""
        for r in self._pending_rows[:-1]:
            if (r["issuing_facility_id"] == facility_id
                    and str(r.get("triage", "")).startswith("T2")
                    and r.get("status", "pending") == "pending"):
                return True
        df = self.pending_transport_requests_df
        if df.empty:
            return False
        match = df[
            (df["issuing_facility_id"] == facility_id)
            & (df["triage"].str.startswith("T2"))
            & (df["status"] == "pending")
        ]
        return not match.empty

    def ambulance_idle(self, amb: "Ambulance"):
        """Mark ambulance idle and wake C2."""
        if amb in self.ambulances and amb not in self.idle_ambulances:
            self.idle_ambulances.append(amb)
            self._wake_reason = "ambulance_idle"
            self.activate()

    # --------------------------------------------------------
    # main loop
    # --------------------------------------------------------

    def process(self):
        while True:
            self._expire_stale_requests()

            pending = self._get_pending_requests()

            if pending.empty or not self.idle_ambulances:
                self.passivate()
                continue

            action = self._choose_action(pending)

            if action is None:
                wait = self._time_to_next_threshold(pending)
                if wait is None:
                    self.passivate()
                else:
                    self.hold(wait)
                    self._wake_reason = "threshold"
                continue

            trigger_type, anchor_facility_id, anchor_indices = action
            dispatch_df = pending.loc[anchor_indices].copy()

            elig_df = fac_eligibility(
                env=self.env,
                requests_df=dispatch_df,
                lever_value=self.lever_value_facility_eligibility,
            )
            elig_df = amb_eligibility(
                env=self.env,
                requests_df=elig_df,
                idle_ambulances=self.idle_ambulances,
                lever_value=self.lever_value_ambulance_eligibility,
            )

            snapshot_ambs = [a.amb_id for a in self.idle_ambulances]
            solver = _DISPATCH_SOLVERS.get(self.dispatch_policy, solve_triage)
            assigned_df = solver(
                env=self.env,
                requests_df=elig_df,
                idle_ambulances=self.idle_ambulances,
                travel_time_matrix=self.env.travel_time_matrix,
                trigger_type=trigger_type,
            )

            plans_craft = craft_plans(c2=self, assigned_ambulances_df=assigned_df)
            dispatch_ambulances(c2=self, plans_craft=plans_craft)
            self._log_dispatch(dispatch_df, snapshot_ambs, plans_craft)

            if not plans_craft:
                self._mark_indices_expired(anchor_indices)
                self.hold(0)

    # --------------------------------------------------------
    # triage-based decision logic
    # --------------------------------------------------------

    def _choose_action(self, pending: pd.DataFrame):
        """
        Evaluate pending requests and return the dispatch action to take,
        or None if nothing meets dispatch criteria.

        Returns: (trigger_type, facility_id, indices) or None
          trigger_type: "t1", "t2_pair", "t2_threshold", "t3_threshold"
        """
        now = self.env.now()

        t1s = pending[pending["triage"].str.startswith("T1")]
        if not t1s.empty:
            t1s_sorted = t1s.sort_values("t_issued", ascending=True)
            anchor = t1s_sorted.iloc[0]
            fac_id = anchor["issuing_facility_id"]
            at_fac = pending[pending["issuing_facility_id"] == fac_id].index.tolist()
            return ("t1", fac_id, at_fac)

        t2s = pending[pending["triage"].str.startswith("T2")]
        if not t2s.empty:
            fac_counts = t2s.groupby("issuing_facility_id").size()
            paired = fac_counts[fac_counts >= 2]
            if not paired.empty:
                fac_id = paired.index[0]
                at_fac = pending[pending["issuing_facility_id"] == fac_id].index.tolist()
                return ("t2_pair", fac_id, at_fac)

        if not t2s.empty:
            t2s_wait = t2s.copy()
            t2s_wait["_wait"] = now - pd.to_numeric(t2s_wait["t_issued"], errors="coerce").fillna(now)
            long_t2 = t2s_wait[t2s_wait["_wait"] >= _T2_WAIT_THRESHOLD]
            if not long_t2.empty:
                anchor = long_t2.sort_values("t_issued", ascending=True).iloc[0]
                fac_id = anchor["issuing_facility_id"]
                at_fac = pending[pending["issuing_facility_id"] == fac_id].index.tolist()
                return ("t2_threshold", fac_id, at_fac)

        t3s = pending[pending["triage"].str.startswith("T3")]
        if not t3s.empty:
            t3s_wait = t3s.copy()
            t3s_wait["_wait"] = now - pd.to_numeric(t3s_wait["t_issued"], errors="coerce").fillna(now)
            long_t3 = t3s_wait[t3s_wait["_wait"] >= _T3_WAIT_THRESHOLD]
            if not long_t3.empty:
                anchor = long_t3.sort_values("t_issued", ascending=True).iloc[0]
                fac_id = anchor["issuing_facility_id"]
                at_fac = pending[pending["issuing_facility_id"] == fac_id].index.tolist()
                return ("t3_threshold", fac_id, at_fac)

        return None

    # --------------------------------------------------------
    # internal helpers
    # --------------------------------------------------------

    def _expire_stale_requests(self):
        """Mark pending requests whose patients left the exit_queue."""
        df = self.pending_transport_requests_df
        if df.empty:
            return
        pending_mask = df["status"] == "pending"
        if not pending_mask.any():
            return

        exit_req_cache: dict[str, set] = {}
        for idx in df.index[pending_mask]:
            fac_id = df.at[idx, "issuing_facility_id"]
            if fac_id not in exit_req_cache:
                fac = self.env.facility_by_id.get(fac_id)
                if fac is None:
                    exit_req_cache[fac_id] = set()
                else:
                    exit_req_cache[fac_id] = {
                        getattr(r, "req_id", None)
                        for r in getattr(fac, "exit_queue", [])
                    }
            if df.at[idx, "req_id"] not in exit_req_cache[fac_id]:
                df.at[idx, "status"] = "expired"

    def _mark_indices_expired(self, indices):
        """Unconditionally expire these indices (dispatch failed)."""
        df = self.pending_transport_requests_df
        for idx in indices:
            if idx in df.index:
                df.at[idx, "status"] = "expired"

    def _time_to_next_threshold(self, pending: pd.DataFrame):
        """Return sim-minutes until the earliest T2/T3 threshold, or None."""
        now = self.env.now()
        soonest = None

        t2s = pending[pending["triage"].str.startswith("T2")]
        if not t2s.empty:
            earliest_t2 = pd.to_numeric(t2s["t_issued"], errors="coerce").min()
            if pd.notna(earliest_t2):
                due = earliest_t2 + _T2_WAIT_THRESHOLD - now
                if due > 0:
                    soonest = due
                else:
                    return 0

        t3s = pending[pending["triage"].str.startswith("T3")]
        if not t3s.empty:
            earliest_t3 = pd.to_numeric(t3s["t_issued"], errors="coerce").min()
            if pd.notna(earliest_t3):
                due = earliest_t3 + _T3_WAIT_THRESHOLD - now
                if due > 0:
                    soonest = min(soonest, due) if soonest is not None else due
                else:
                    return 0

        return soonest

    def _init_pending_requests_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "req_id", "casualty_id", "triage", "issuing_facility_id",
            "injury_time", "t_issued", "needed_unit", "status",
        ])

    def _get_pending_requests(self) -> pd.DataFrame:
        if self._pending_rows:
            new = pd.DataFrame(self._pending_rows)
            self._pending_rows.clear()
            if self.pending_transport_requests_df.empty:
                self.pending_transport_requests_df = new
            else:
                self.pending_transport_requests_df = pd.concat(
                    [self.pending_transport_requests_df, new], ignore_index=True,
                )
        df = self.pending_transport_requests_df
        if df.empty:
            return df
        return df[df["status"] == "pending"].copy()

    def _log_dispatch(self, requests_df: pd.DataFrame, snapshot_ambs: list, plans_craft: dict):
        """Append one decision row to env.c2_dispatch_logs[self.c2_id]."""
        logs = getattr(self.env, "c2_dispatch_logs", None)
        if logs is None:
            return

        transport_requests = [
            {
                "req_id":              str(row.get("req_id",              "")),
                "patient_id":          str(row.get("casualty_id",         "")),
                "triage":              str(row.get("triage",              "")),
                "issuing_facility_id": str(row.get("issuing_facility_id", "")),
                "t_issued":            row.get("t_issued"),
                "needed_unit":         str(row.get("needed_unit",         "")),
            }
            for _, row in requests_df.iterrows()
        ]

        assignments = {
            amb_id: [p.patient_id for step in plan for p in step.load_patients]
            for amb_id, plan in plans_craft.items()
        }

        logs.setdefault(self.c2_id, []).append({
            "event":                json.dumps(self._wake_reason),
            "transport_requests":   json.dumps(transport_requests),
            "available_ambulances": json.dumps(snapshot_ambs),
            "assignments":          json.dumps(assignments),
        })
