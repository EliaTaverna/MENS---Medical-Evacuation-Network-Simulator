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

_BATCH_INTERVAL = 5  # minutes between dispatch episodes

# Default dispatch rules — evaluated in list order; first match wins.
# Each rule: triage prefix, min count at one facility, min wait (minutes).
# A rule with min_wait_min == 0 is a pure count trigger (fires immediately).
# A rule with min_wait_min > 0 requires BOTH count AND wait to be met.
# Typically two rules per triage: a count-trigger + a wait-trigger.
DEFAULT_DISPATCH_RULES = [
    {"triage": "T1", "min_count": 1, "min_wait_min": 0},     # T1 count-trigger
    {"triage": "T1", "min_count": 1, "min_wait_min": 60},    # T1 wait-trigger
    {"triage": "T2", "min_count": 2, "min_wait_min": 0},     # T2 count-trigger
    {"triage": "T2", "min_count": 1, "min_wait_min": 240},   # T2 wait-trigger
    {"triage": "T3", "min_count": 4, "min_wait_min": 0},     # T3 count-trigger
    {"triage": "T3", "min_count": 1, "min_wait_min": 720},   # T3 wait-trigger
]

_DISPATCH_SOLVERS = {
    "greedy": solve_triage,
}

_DF_COLUMNS = [
    "req_id", "casualty_id", "triage", "issuing_facility_id",
    "injury_time", "t_issued", "needed_unit", "status",
]


def _rule_label(rule: dict) -> str:
    """Build a human-readable trigger label from a dispatch rule.

    Rules with ``min_wait_min == 0`` are count triggers; rules with
    ``min_wait_min > 0`` are wait triggers.
    """
    t = rule["triage"].lower()
    if rule["min_wait_min"] == 0:
        return f"{t}_count_{rule['min_count']}"     # e.g. "t1_count_1", "t2_count_2"
    return f"{t}_wait_{rule['min_wait_min']}"       # e.g. "t1_wait_60", "t2_wait_240"


class TransportC2(sim.Component):
    """
    C2 dispatcher for one evacuation leg.

    Dispatch rules are an ordered list of conditions. Each rule specifies:
      - triage:       triage prefix to match (e.g. "T1", "T2", "T3")
      - min_count:    minimum patients of that triage at one facility
      - min_wait_min: minimum waiting time (minutes) of the oldest patient
                      (0 = no wait required, fires on count alone)

    Rules are evaluated in list order during each dispatch episode.
    The first rule whose condition is met selects the facility and triggers
    a dispatch. One ambulance is sent per match; the loop repeats until
    no more rules match or no idle ambulances remain.
    """

    def setup(
        self,
        c2_id: str,
        facility_ids: list[str],
        ambulance_ids: list[str],
        lever_value_facility_eligibility: int = 0,
        lever_value_ambulance_eligibility: int = 0,
        dispatch_policy: str = "greedy",
        dispatch_rules: list[dict] | None = None,
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
        self.dispatch_rules = dispatch_rules or list(DEFAULT_DISPATCH_RULES)
        self._wake_reason = "init"

        # Pre-compute which triage prefixes can trigger an immediate wake
        # (rules with min_wait_min == 0).  For each prefix, store the
        # min_count that would satisfy at least one rule.
        self._immediate_wake: dict[str, int] = {}
        for rule in self.dispatch_rules:
            if rule["min_wait_min"] == 0:
                prefix = rule["triage"][:2]
                prev = self._immediate_wake.get(prefix)
                if prev is None or rule["min_count"] < prev:
                    self._immediate_wake[prefix] = rule["min_count"]

    @property
    def pending_transport_requests_df(self):
        if self._pending:
            return pd.DataFrame(self._pending, columns=_DF_COLUMNS)
        return pd.DataFrame(columns=_DF_COLUMNS)

    # --------------------------------------------------------
    # inbound events
    # --------------------------------------------------------

    def receive_request(self, req: "TransportRequest"):
        """Append request. Wake C2 immediately if a zero-wait rule is met."""
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

        prefix = triage[:2]
        needed = self._immediate_wake.get(prefix)
        if needed is not None:
            # Count how many pending of this triage at this facility
            count = sum(
                1 for r in self._pending
                if r["issuing_facility_id"] == issuing_facility_id
                and r["triage"][:2] == prefix
                and r["status"] == "pending"
            )
            if count >= needed:
                self._wake_reason = "request"
                self.activate()

    def ambulance_idle(self, amb: "Ambulance"):
        """Register ambulance as idle; wake C2 immediately if any zero-wait rule is satisfiable."""
        if amb not in self.ambulances or amb in self.idle_ambulances:
            return
        self.idle_ambulances.append(amb)

        # Check if any immediate-wake rule is currently satisfied
        for prefix, needed in self._immediate_wake.items():
            fac_counts: dict[str, int] = {}
            for r in self._pending:
                if r["triage"][:2] == prefix and r["status"] == "pending":
                    fid = r["issuing_facility_id"]
                    fac_counts[fid] = fac_counts.get(fid, 0) + 1
            if any(c >= needed for c in fac_counts.values()):
                self._wake_reason = "ambulance_idle"
                self.activate()
                return

    # --------------------------------------------------------
    # main loop
    # --------------------------------------------------------

    def process(self):
        while True:
            self.hold(_BATCH_INTERVAL)
            self._wake_reason = "batch"
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
            dispatch_ambulances(c2=self, plans_craft=plans_craft, trigger_type=trigger_type)
            self._log_dispatch(dispatch_df, snapshot_ambs, plans_craft, trigger_type)

            if not plans_craft:
                for r in anchor_rows:
                    r["status"] = "expired"

            self._pending = [r for r in self._pending if r["status"] == "pending"]

    # --------------------------------------------------------
    # rule-based decision logic
    # --------------------------------------------------------

    def _choose_action(self, pending: list[dict]):
        """
        Evaluate dispatch rules in priority order.
        Returns: (trigger_label, facility_id, all_rows_at_facility) or None
        """
        now = self.env.now()

        for rule in self.dispatch_rules:
            prefix = rule["triage"][:2]
            min_count = rule["min_count"]
            min_wait = rule["min_wait_min"]

            # Collect matching requests grouped by facility
            by_fac: dict[str, list[dict]] = {}
            for r in pending:
                if r["triage"][:2] == prefix:
                    fid = r["issuing_facility_id"]
                    by_fac.setdefault(fid, []).append(r)

            # Check each facility against this rule's conditions
            # Sort facilities by oldest request to be deterministic
            for fac_id, fac_rows in sorted(
                by_fac.items(),
                key=lambda item: min((r["t_issued"] or 0) for r in item[1]),
            ):
                if len(fac_rows) < min_count:
                    continue
                if min_wait > 0:
                    oldest_issued = min((r["t_issued"] or 0) for r in fac_rows)
                    if oldest_issued is None or (now - oldest_issued) < min_wait:
                        continue

                # Rule matched — return ALL pending at this facility
                at_fac = [r for r in pending if r["issuing_facility_id"] == fac_id]
                return (_rule_label(rule), fac_id, at_fac)

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

    def _log_dispatch(self, requests_df: pd.DataFrame, snapshot_ambs: list,
                       plans_craft: dict, trigger_type: str = ""):
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

        assigned_ids = {pid for pids in assignments.values() for pid in pids}
        triage_of = {
            str(row.get("casualty_id", "")): str(row.get("triage", ""))
            for _, row in requests_df.iterrows()
        }
        cargo_triages = [triage_of.get(pid, "") for pid in assigned_ids]
        n_t1 = sum(1 for t in cargo_triages if t.startswith("T1"))
        n_t2 = sum(1 for t in cargo_triages if t.startswith("T2"))
        n_t3 = sum(1 for t in cargo_triages if t.startswith("T3"))

        logs.setdefault(self.c2_id, []).append({
            "time":                 self.env.now(),
            "event":                self._wake_reason,
            "trigger":              trigger_type,
            "n_requests":           len(requests_df),
            "n_idle_ambulances":    len(snapshot_ambs),
            "n_assigned":           len(assigned_ids),
            "n_t1_cargo":           n_t1,
            "n_t2_cargo":           n_t2,
            "n_t3_cargo":           n_t3,
            "transport_requests":   json.dumps(transport_requests),
            "available_ambulances": json.dumps(snapshot_ambs),
            "assignments":          json.dumps(assignments),
        })
