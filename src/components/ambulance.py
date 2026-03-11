"""Ambulance component for patient transport."""

from __future__ import annotations
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
        capacity: int,
        speed_kmh: float,
        start_loc_id: str,
        c2_id: str,
    ):
        self.amb_id = amb_id
        self.capacity = int(capacity)
        self.speed_kmh = float(speed_kmh)
        self.start_loc_id = start_loc_id
        self.c2_id = c2_id
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

            # 3) Execute plan to completion
            for step in self.plan or []:
                self.current_step = step

                self._travel_to(step.facility)

                self.location = step.facility
                step.facility.ambulance_arrives(self)
                self.passivate()  # facility wakes me after load/unload

            # 4) Plan complete: stay where you are (last location)
            self.current_step = None
            self.plan = None
            # loop back: will announce idle at this same location next iteration

    def _travel_to(self, destination):
        from_id = self._loc_id(self.location)
        to_id = self._loc_id(destination)
        if from_id == to_id:
            return
        dist_km = self.env.distance_matrix.loc[from_id, to_id]
        minutes = (dist_km / self.speed_kmh) * 60.0
        self.hold(minutes)
