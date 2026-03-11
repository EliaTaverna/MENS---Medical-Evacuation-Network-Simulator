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
    FACILITY_ELIGIBILITY_POLICIES,
    FACILITY_ASSIGNMENT_POLICIES,
    AMBULANCE_ELIGIBILITY_POLICIES,
    AMBULANCE_ASSIGNMENT_POLICIES,
    craft_plans,
    dispatch_ambulances,
    snapshot_facilities,
)


class TransportC2(sim.Component):
    """
    C2 dispatcher for one evacuation leg.

    Dataflow (fixed in code):
      pending requests -> facility snapshot -> request->facility eligibility
      -> request->facility assignment -> request/facility->ambulance eligibility
      -> ambulance assignment -> craft plans -> dispatch

    Levers are implemented by dict registries:
      FACILITY_ELIGIBILITY_POLICIES
      FACILITY_ASSIGNMENT_POLICIES
      AMBULANCE_ELIGIBILITY_POLICIES
      AMBULANCE_ASSIGNMENT_POLICIES

    Add new lever:
      1) implement new function with the correct signature
      2) register it in the appropriate dict with a new integer key
      No changes required in process().
    """

    def setup(
        self,
        c2_id: str,
        facility_ids: list[str],
        ambulance_ids: list[str],
        corridors_matrix: pd.DataFrame,
        ambulances_eligibility: pd.DataFrame,
        lever_value_facility_eligibility: int = 0,
        lever_value_facility_assignment: int = 0,
        lever_value_ambulance_eligibility: int = 0,
        lever_value_ambulance_assignment: int = 0,
    ):
        self.c2_id = c2_id
        self.facility_ids = list(facility_ids)
        self.ambulance_ids = list(ambulance_ids)

        # connected later
        self.facilities: List["Facility"] = []
        self.ambulances: List["Ambulance"] = []
        self.idle_ambulances: List["Ambulance"] = []

        # core tables
        self.pending_transport_requests_df = self._init_pending_requests_df()
        self.corridors_matrix = corridors_matrix.copy()
        self.ambulances_eligibility = ambulances_eligibility.copy()

        # lever values
        self.lever_value_facility_eligibility = int(lever_value_facility_eligibility)
        self.lever_value_facility_assignment = int(lever_value_facility_assignment)
        self.lever_value_ambulance_eligibility = int(lever_value_ambulance_eligibility)
        self.lever_value_ambulance_assignment = int(lever_value_ambulance_assignment)

    # --------------------------------------------------------
    # inbound events
    # --------------------------------------------------------

    def receive_request(self, req: "TransportRequest"):
        """
        Convert received TransportRequest -> row in pending_transport_requests_df and activate C2.
        """
        issuing_facility_id = getattr(getattr(req, "issuing_facility", None), "loc_id", None) \
                              or getattr(req, "issuing_facility_id", None)

        row = {
            "req_id": getattr(req, "req_id", None),
            "casualty_id": getattr(req, "casualty_id", None) or getattr(req.patient, "patient_id", None),
            "triage": getattr(req, "triage", None),
            "issuing_facility_id": issuing_facility_id,
            "injury_time": getattr(req, "injury_time", None),
            "t_issued": getattr(req, "t_issued", None),
            "needed_unit": getattr(req, "needed_unit", None),
            "status": getattr(req, "status", "pending"),
        }

        self.pending_transport_requests_df = pd.concat(
            [self.pending_transport_requests_df, pd.DataFrame([row])],
            ignore_index=True
        )

        self.activate()

    def ambulance_idle(self, amb: "Ambulance"):
        """Mark ambulance idle, then activate C2 to try dispatching."""
        if amb in self.ambulances and amb not in self.idle_ambulances:
            self.idle_ambulances.append(amb)
            self.activate()

    # --------------------------------------------------------
    # main loop
    # --------------------------------------------------------

    def process(self):
        while True:
            self.passivate()

            requests_df = self._get_pending_requests()
            if requests_df.empty:
                continue

            if len(self.idle_ambulances) == 0:
                continue

            facilities_df = snapshot_facilities(self.facilities)

            # ---- requests -> eligible facilities (policy) ----
            fac_elig_fn = FACILITY_ELIGIBILITY_POLICIES.get(self.lever_value_facility_eligibility)

            eligible_facilities_df = fac_elig_fn(
                env=self.env,
                requests_df=requests_df,
                corridors_matrix=self.corridors_matrix,
            )

            # ---- eligible facilities -> assigned facility (policy) ----
            fac_assign_fn = FACILITY_ASSIGNMENT_POLICIES.get(self.lever_value_facility_assignment)
            assigned_facilities_df = fac_assign_fn(
                env=self.env,
                eligible_facilities_df=eligible_facilities_df,
                facilities_df=facilities_df,
                distance_matrix=self.env.distance_matrix,
            )

            # ---- assigned facility -> eligible ambulances (policy) ----
            amb_elig_fn = AMBULANCE_ELIGIBILITY_POLICIES.get(self.lever_value_ambulance_eligibility)
            eligible_pairs_df = amb_elig_fn(
                env=self.env,
                assigned_facilities_df=assigned_facilities_df,
                ambulances_eligibility=self.ambulances_eligibility,
                idle_ambulances=self.idle_ambulances,
            )

            # ---- eligible pairs -> assigned ambulances (policy) ----
            amb_assign_fn = AMBULANCE_ASSIGNMENT_POLICIES.get(self.lever_value_ambulance_assignment)
            assigned_ambulances_df = amb_assign_fn(
                env=self.env,
                eligible_pairs_df=eligible_pairs_df,
                distance_matrix=self.env.distance_matrix,
            )

            plans_craft = craft_plans(c2=self, assigned_ambulances_df=assigned_ambulances_df)
            dispatch_ambulances(c2=self, plans_craft=plans_craft)

    # --------------------------------------------------------
    # internal df helpers
    # --------------------------------------------------------

    def _init_pending_requests_df(self) -> pd.DataFrame:
        """Canonical schema for pending transport requests."""
        return pd.DataFrame(columns=[
            "req_id",
            "casualty_id",
            "triage",
            "issuing_facility_id",
            "injury_time",
            "t_issued",
            "needed_unit",
            "status",
        ])

    def _get_pending_requests(self) -> pd.DataFrame:
        """Return only rows still pending."""
        df = self.pending_transport_requests_df
        if df.empty:
            return df
        return df[df["status"] == "pending"].copy()
