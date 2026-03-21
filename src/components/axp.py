"""Ambulance Exchange Point (AXP) component."""

from __future__ import annotations
import salabim as sim


class AXP(sim.Component):
    """An ambulance exchange point where ambulances can wait."""

    def setup(self, axp_id: str, assigned_facility_ids: list[str]):
        self.loc_id = axp_id
        self.assigned_facility_ids = list(assigned_facility_ids)
        self.assigned_facilities = []
        self.ambulances_here = []

    def arrive(self, amb):

        """Register an ambulance's arrival at this AXP."""
        if amb not in self.ambulances_here:
            self.ambulances_here.append(amb)

    def depart(self, amb):
        """Register an ambulance's departure from this AXP."""
        if amb in self.ambulances_here:
            self.ambulances_here.remove(amb)
