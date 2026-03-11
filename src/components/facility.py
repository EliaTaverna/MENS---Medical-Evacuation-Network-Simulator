"""Facility component for medical treatment locations."""

from __future__ import annotations
import uuid
import salabim as sim
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from src.components.patient import Patient

from src.components.requests import TransportRequest


class Facility(sim.Component):
    """A medical facility that provides care and manages patient flow."""

    def setup(self, facility_id: str, facility_type: str, c2_up_id: str | None, unload_key: str, load_key: str,
              service_types: list[str]):
        self.loc_id = facility_id
        self.facility_type = facility_type
        self.c2_up_id = c2_up_id
        self.c2_up = None
        self.service_types = list(service_types)
        self.unload_key = unload_key
        self.load_key = load_key
        self.units = {st: None for st in self.service_types}
        self.entry_lines = {st: [] for st in self.service_types}
        self.exit_queue: List[TransportRequest] = []
        self._amb = None

    def ambulance_arrives(self, amb):
        """Called when an ambulance arrives at this facility."""
        self._amb = amb
        self.activate()

    def patient_handed_over(self, p: Patient):
        """Handle a patient being handed over to this facility."""
        p.current_facility = self
        if hasattr(p, "_log_action"):
            p._log_action(life_budget=getattr(p, "_last_life_budget", None))

        needed = p.unit_needed

        if needed in self.entry_lines:
            # stamp join time (for realized wait)
            p._unit_queue_entry_time = self.env.now()
            unit = self.units[needed]
            p._expected_wait_at_join = unit.expected_wait_if_join_now()
            unit.ewait_monitor.tally(p._expected_wait_at_join)

            self.entry_lines[needed].append(p)
            # Start death clock - patient is waiting in queue
            p.start_death_clock()
            unit.activate()
            return

        req = TransportRequest(
            req_id=uuid.uuid4().hex,
            patient=p,
            issuing_facility=self,
            t_issued=self.env.now(),
            injury_time=p.injury_time,
            triage=p.triage,
            needed_unit=p.unit_needed,
        )

        p.patient_state = "waiting_transport"
        self.exit_queue.append(req)
        # Start death clock - patient is waiting for transport
        p.start_death_clock()
        self.c2_up.receive_request(req)

    def process(self):
        while True:
            self.passivate()
            amb = self._amb
            self._amb = None

            unload_fn = self.env.unload_policy[self.unload_key]
            for p in unload_fn(amb, self):
                amb.onboard.remove(p)
                self.patient_handed_over(p)

            load_fn = self.env.load_policy[self.load_key]
            for p in load_fn(amb, self):
                # remove the assigned request from exit_queue
                for req in list(self.exit_queue):
                    if req.patient is p and req.status == "assigned":
                        self.exit_queue.remove(req)
                        break
                # Pause death clock - patient is being transported
                p.pause_death_clock()
                amb.onboard.append(p)
                p.patient_state = "in_transport"

            amb.activate()
