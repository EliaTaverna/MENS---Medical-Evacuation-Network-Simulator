"""Ambulance component for patient transport."""

from __future__ import annotations
import random
import salabim as sim
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.components.patient import Patient
    from src.components.requests import PlanStep

from src.components.axp import AXP


class Ambulance(sim.Component):
    """An ambulance that transports patients between facilities."""

    def setup(
        self,
        amb_id: str,
        t1_t2_capacity: int,
        t3_capacity: int,
        start_loc_id: str,
        c2_id: str,
        load_unload_time_min: float = 0.0,
        load_unload_time_max: float = 0.0,
    ):
        self.amb_id = amb_id
        self.t1_t2_capacity = int(t1_t2_capacity)
        self.t3_capacity    = int(t3_capacity)
        self.start_loc_id = start_loc_id
        self.c2_id = c2_id
        self.load_unload_time_min = float(load_unload_time_min)
        self.load_unload_time_max = float(load_unload_time_max)
        self.c2 = None
        self.location = None  # AXP or Facility object, set by connector
        self.onboard: List[Patient] = []
        self.plan: List[PlanStep] | None = None
        self.current_step: PlanStep | None = None

    def _loc_id(self, x):
        return getattr(x, "loc_id", x)

    def _announce_idle_here(self):
        """
        Optional bookkeeping: if at an AXP, register in its list.
        Tell C2 you're idle.
        """
        if isinstance(self.location, AXP):
            self.location.arrive(self)
        self.c2.ambulance_idle(self)

    def _leave_current_spot(self):
        if isinstance(self.location, AXP):
            self.location.depart(self)

    def process(self):
        while True:
            # 1) I'm idle wherever I am right now
            self._announce_idle_here()
            self.passivate()  # C2 wakes me when it assigns plan

            # 2) I've been dispatched
            self._leave_current_spot()
            self._log_mission_event("mission_start")

            # 3) Execute plan to completion
            for step in self.plan or []:
                self.current_step = step

                self._travel_to(step.facility)

                self.location = step.facility

                # Hold for unloading (patients still on board, clocks paused)
                if self.load_unload_time_max > 0 and self.onboard:
                    lo, hi = self.load_unload_time_min, self.load_unload_time_max
                    self.hold(random.uniform(lo, hi))

                step.facility.ambulance_arrives(self)
                self.passivate()  # facility wakes me after load/unload

                # Hold for loading (new patients already on board)
                if self.load_unload_time_max > 0 and self.onboard:
                    lo, hi = self.load_unload_time_min, self.load_unload_time_max
                    self.hold(random.uniform(lo, hi))

            # 4) Plan complete: stay where you are (last location)
            self._log_mission_event("mission_end")
            self.current_step = None
            self.plan = None
            # loop back: will announce idle at this same location next iteration

    def _log_mission_event(self, event_type: str):
        """Log a mission_start or mission_end event to transport_events."""
        if not hasattr(self.env, "transport_events"):
            return
        plan_patients = []
        if self.plan:
            for s in self.plan:
                for p in s.load_patients:
                    plan_patients.append({
                        "patient_id": p.patient_id,
                        "triage": p.triage,
                    })
        self.env.transport_events.append({
            "time": float(self.env.now()),
            "event": event_type,
            "amb_id": self.amb_id,
            "from_loc": self._loc_id(self.location) if self.location else "",
            "to_loc": "",
            "patient_ids": [pp["patient_id"] for pp in plan_patients],
            "patient_triages": [pp["triage"] for pp in plan_patients],
        })

    def _travel_to(self, destination):
        from_id = self._loc_id(self.location)
        to_id = self._loc_id(destination)
        if from_id == to_id:
            return
        if hasattr(self.env, "transport_events"):
            self.env.transport_events.append({
                "time": float(self.env.now()),
                "event": "departure",
                "amb_id": self.amb_id,
                "from_loc": from_id,
                "to_loc": to_id,
                "patient_ids": [p.patient_id for p in self.onboard],
                "patient_triages": [p.triage for p in self.onboard],
            })
        minutes = self.env.travel_time_matrix.loc[from_id, to_id]
        self.hold(float(minutes))
        if hasattr(self.env, "transport_events"):
            self.env.transport_events.append({
                "time": float(self.env.now()),
                "event": "arrival",
                "amb_id": self.amb_id,
                "from_loc": from_id,
                "to_loc": to_id,
                "patient_ids": [p.patient_id for p in self.onboard],
                "patient_triages": [p.triage for p in self.onboard],
            })
