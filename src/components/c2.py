"""Transport Command and Control (C2) component."""

from __future__ import annotations
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
from src.helpers.ilp import solve_ilp, solve_greedy

_BATCH_WINDOW = 2  # simulated minutes to wait before running dispatch

_DISPATCH_SOLVERS = {
    "ilp":    solve_ilp,
    "greedy": solve_greedy,
}


class TransportC2(sim.Component):
    """
    C2 dispatcher for one evacuation leg.

    Dataflow:
      pending requests
      -> fac_eligibility   (adds eligible_facilities per request)
      -> amb_eligibility   (adds eligible_ambulances per request)
      -> solve_ilp         (4-term weighted ILP: α·triage + β·sojourn − γ·transport − δ·EWT)
      -> craft_plans -> dispatch

    Scheduling logic:
      - While idle ambulances are available: run ILP every _BATCH_WINDOW minutes,
        batching all requests that accumulated in the window.
      - When all ambulances are busy: sleep until one returns idle.
      - New requests do NOT wake C2 — they accumulate silently until the next cycle.
    """

    def setup(
        self,
        c2_id: str,
        facility_ids: list[str],
        ambulance_ids: list[str],
        lever_value_facility_eligibility: int = 0,
        lever_value_ambulance_eligibility: int = 0,
        beta:  float = 0.0,
        gamma: float = 0.0,
        delta: float = 0.0,
        random_seed: int = 42,
        dispatch_policy: str = "ilp",
    ):
        self.c2_id = c2_id
        self.facility_ids = list(facility_ids)
        self.ambulance_ids = list(ambulance_ids)

        # connected later
        self.facilities: List["Facility"] = []
        self.ambulances: List["Ambulance"] = []
        self.idle_ambulances: List["Ambulance"] = []

        # pending requests table
        self.pending_transport_requests_df = self._init_pending_requests_df()

        # eligibility levers
        self.lever_value_facility_eligibility = int(lever_value_facility_eligibility)
        self.lever_value_ambulance_eligibility = int(lever_value_ambulance_eligibility)

        # ILP objective weights
        self.beta        = float(beta)
        self.gamma       = float(gamma)
        self.delta       = float(delta)
        self.random_seed = int(random_seed)
        self.dispatch_policy = dispatch_policy

    # --------------------------------------------------------
    # inbound events
    # --------------------------------------------------------

    def receive_request(self, req: "TransportRequest"):
        """
        Append request to pending table.
        Does NOT wake C2 — requests accumulate until the next dispatch cycle.
        """
        issuing_facility_id = getattr(getattr(req, "issuing_facility", None), "loc_id", None) \
                              or getattr(req, "issuing_facility_id", None)

        row = {
            "req_id":              getattr(req, "req_id", None),
            "casualty_id":         getattr(req, "casualty_id", None) or getattr(req.patient, "patient_id", None),
            "triage":              getattr(req, "triage", None),
            "issuing_facility_id": issuing_facility_id,
            "injury_time":         getattr(req, "injury_time", None),
            "t_issued":            getattr(req, "t_issued", None),
            "needed_unit":         getattr(req, "needed_unit", None),
            "status":              getattr(req, "status", "pending"),
        }

        self.pending_transport_requests_df = pd.concat(
            [self.pending_transport_requests_df, pd.DataFrame([row])],
            ignore_index=True
        )

    def ambulance_idle(self, amb: "Ambulance"):
        """Mark ambulance idle. Wake C2 only on 0→1 transition."""
        if amb in self.ambulances and amb not in self.idle_ambulances:
            was_empty = len(self.idle_ambulances) == 0
            self.idle_ambulances.append(amb)
            if was_empty:
                self.activate()

    # --------------------------------------------------------
    # main loop
    # --------------------------------------------------------

    def process(self):
        while True:
            if len(self.idle_ambulances) == 0:
                self.passivate()

            requests_df = self._get_pending_requests()
            if requests_df.empty or len(self.idle_ambulances) == 0:
                if len(self.idle_ambulances) > 0:
                    self.hold(_BATCH_WINDOW)
                continue

            # step 1: facility eligibility
            elig_df = fac_eligibility(
                env=self.env,
                requests_df=requests_df,
                lever_value=self.lever_value_facility_eligibility,
            )

            # step 2: ambulance eligibility
            elig_df = amb_eligibility(
                env=self.env,
                requests_df=elig_df,
                idle_ambulances=self.idle_ambulances,
                lever_value=self.lever_value_ambulance_eligibility,
            )

            # step 3: solve (ILP or greedy)
            solver = _DISPATCH_SOLVERS.get(self.dispatch_policy, solve_ilp)
            assigned_df = solver(
                env=self.env,
                requests_df=elig_df,
                idle_ambulances=self.idle_ambulances,
                distance_matrix=self.env.distance_matrix,
                beta=self.beta,
                gamma=self.gamma,
                delta=self.delta,
                random_seed=self.random_seed,
            )

            plans_craft = craft_plans(c2=self, assigned_ambulances_df=assigned_df)
            dispatch_ambulances(c2=self, plans_craft=plans_craft)

            if len(self.idle_ambulances) > 0:
                self.hold(_BATCH_WINDOW)

    # --------------------------------------------------------
    # internal helpers
    # --------------------------------------------------------

    def _init_pending_requests_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[
            "req_id", "casualty_id", "triage", "issuing_facility_id",
            "injury_time", "t_issued", "needed_unit", "status",
        ])

    def _get_pending_requests(self) -> pd.DataFrame:
        df = self.pending_transport_requests_df
        if df.empty:
            return df
        return df[df["status"] == "pending"].copy()
