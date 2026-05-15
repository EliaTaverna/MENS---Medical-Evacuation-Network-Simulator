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

_BATCH_INTERVAL = 5       # minutes between dispatch episodes
_T2_WAIT_THRESHOLD = 240  # minutes (4 hours) before a lone T2 triggers dispatch
_T3_WAIT_THRESHOLD = 720  # minutes (12 hours) before T3s trigger dispatch

_DISPATCH_SOLVERS = {
    "greedy": solve_triage,
}

_DF_COLUMNS = [
    "req_id", "casualty_id", "triage", "issuing_facility_id",
    "injury_time", "t_issued", "needed_unit", "status",
]


class TransportC2(sim.Component):
    """
    C2 dispatcher for one evacuation leg.

    Dispatches in two modes:
      - Immediate: T1 request or qualifying T2 pair wakes C2 instantly.
      - Batched:   every _BATCH_INTERVAL minutes, evaluate all pending
        requests against thresholds (T2 > 4 h, T3 > 12 h) and dispatch.

    Triage priority within each dispatch episode:
      1. T1 (oldest first)
      2. T2 pairs (two or more at same facility)
      3. T2 over threshold
      4. T3 over threshold

    One ambulance dispatched per actionable facility; loop until no more
    idle ambulances or no more actionable requests.
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

        self._pending: list[dict] = []

        self.lever_value_facility_eligibility = int(lever_value_facility_eligibility)
        self.lever_value_ambulance_eligibility = int(lever_value_ambulance_eligibility)

        self.dispatch_policy = dispatch_policy
        self._wake_reason = "init"

    @property
    def pending_transport_requests_df(self):
        if self._pending:
            return pd.DataFrame(self._pending, columns=_DF_COLUMNS)
        return pd.DataFrame(columns=_DF_COLUMNS)

    # --------------------------------------------------------
    # inbound events
    # --------------------------------------------------------

    def receive_request(self, req: "TransportRequest"):
        """Append request. Only wake C2 immediately for T1 or qualifying T2 pair."""
        issuing_facility_id = getattr(getattr(req, "issuing_facility", None), "loc_id", None) \
                              or getattr(req, "issuing_facility_id", None)
        triage = getattr(req, "triage", None) or ""

        self._pending.append({
            "req_id":              getattr(req, "req_id", None),
            "casualty_id":         getattr(req, "casualty_id", None) or getattr(req.patient, "patient_id", None),
            "triage":              triage,
            "issuing_facility_id": issuing_facility_id,
            "injury_time":         getattr(req, "injury_time", None),
            "t_issued":            getattr(req, "t_issued", None),
            "needed_unit":         getattr(req, "needed_unit", None),
            "status":              "pending",
        })

        should_wake = False
        if triage[:2] == "T1":
            should_wake = True
        elif triage[:2] == "T2":
            should_wake = self._has_pending_t2_at(issuing_facility_id)

        if should_wake:
            self._wake_reason = "request"
            self.activate()

    def _has_pending_t2_at(self, facility_id: str) -> bool:
        for r in self._pending[:-1]:
            if (r["issuing_facility_id"] == facility_id
                    and r["triage"][:2] == "T2"
                    and r["status"] == "pending"):
                return True
        return False

    def ambulance_idle(self, amb: "Ambulance"):
        """Register ambulance as idle (dispatch happens at next batch episode)."""
        if amb in self.ambulances and amb not in self.idle_ambulances:
            self.idle_ambulances.append(amb)

    # --------------------------------------------------------
    # main loop
    # --------------------------------------------------------

    def process(self):
        while True:
            self.hold(_BATCH_INTERVAL)

            self._dispatch_episode()

    def _dispatch_episode(self):
        """Run one full dispatch episode: expire stale, then dispatch all actionable."""
        self._expire_stale()
        self._pending = [r for r in self._pending if r["status"] == "pending"]

        while self._pending and self.idle_ambulances:
            action = self._choose_action(self._pending)
            if action is None:
                break

            trigger_type, anchor_facility_id, anchor_rows = action
            dispatch_df = pd.DataFrame(anchor_rows)

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
                for r in anchor_rows:
                    r["status"] = "expired"

            self._pending = [r for r in self._pending if r["status"] == "pending"]

    # --------------------------------------------------------
    # triage-based decision logic
    # --------------------------------------------------------

    def _choose_action(self, pending: list[dict]):
        """
        Returns: (trigger_type, facility_id, rows_at_facility) or None
        """
        now = self.env.now()

        t1s = [r for r in pending if r["triage"][:2] == "T1"]
        if t1s:
            t1s.sort(key=lambda r: r["t_issued"] or 0)
            fac_id = t1s[0]["issuing_facility_id"]
            at_fac = [r for r in pending if r["issuing_facility_id"] == fac_id]
            return ("t1", fac_id, at_fac)

        t2s = [r for r in pending if r["triage"][:2] == "T2"]
        if t2s:
            fac_counts: dict[str, int] = {}
            for r in t2s:
                fid = r["issuing_facility_id"]
                fac_counts[fid] = fac_counts.get(fid, 0) + 1
            for fac_id, cnt in fac_counts.items():
                if cnt >= 2:
                    at_fac = [r for r in pending if r["issuing_facility_id"] == fac_id]
                    return ("t2_pair", fac_id, at_fac)

        if t2s:
            for r in sorted(t2s, key=lambda r: r["t_issued"] or 0):
                if r["t_issued"] is not None and (now - r["t_issued"]) >= _T2_WAIT_THRESHOLD:
                    fac_id = r["issuing_facility_id"]
                    at_fac = [r for r in pending if r["issuing_facility_id"] == fac_id]
                    return ("t2_threshold", fac_id, at_fac)

        t3s = [r for r in pending if r["triage"][:2] == "T3"]
        if t3s:
            for r in sorted(t3s, key=lambda r: r["t_issued"] or 0):
                if r["t_issued"] is not None and (now - r["t_issued"]) >= _T3_WAIT_THRESHOLD:
                    fac_id = r["issuing_facility_id"]
                    at_fac = [r for r in pending if r["issuing_facility_id"] == fac_id]
                    return ("t3_threshold", fac_id, at_fac)

        return None

    # --------------------------------------------------------
    # internal helpers
    # --------------------------------------------------------

    def _expire_stale(self):
        """Mark pending requests whose patients left the exit_queue."""
        exit_req_cache: dict[str, set] = {}
        for r in self._pending:
            if r["status"] != "pending":
                continue
            fac_id = r["issuing_facility_id"]
            if fac_id not in exit_req_cache:
                fac = self.env.facility_by_id.get(fac_id)
                if fac is None:
                    exit_req_cache[fac_id] = set()
                else:
                    exit_req_cache[fac_id] = {
                        getattr(rq, "req_id", None)
                        for rq in getattr(fac, "exit_queue", [])
                    }
            if r["req_id"] not in exit_req_cache[fac_id]:
                r["status"] = "expired"

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
