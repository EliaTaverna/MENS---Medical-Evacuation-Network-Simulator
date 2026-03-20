"""Patient component for the simulation."""

from __future__ import annotations
import salabim as sim


class Patient(sim.Component):
    """A casualty requiring medical care and evacuation.

    Death clock logic (following Meisner et al. 2025):
    - Clock TICKS only while waiting (in queue for treatment or transport)
    - Clock PAUSED during transport and during treatment
    - Clock EXTENDED after each treatment step
    """

    def setup(self, patient_id: str, triage: str, prescription: list, life_time_dist, spawn_facility, injury_time):
        self.patient_id = patient_id
        self.triage = triage
        self.prescription = list(prescription)
        self.life_time_dist = life_time_dist
        self.spawn_facility = spawn_facility
        self.current_facility = spawn_facility
        self.rx_index = 0
        self.unit_needed = self.prescription[0] if self.prescription else None
        self.is_safe = False
        self.is_dead = False
        self.t_safe = None
        self.patient_state = "in_care"
        self.injury_time = float(injury_time)

        # Death clock state
        self._remaining_survival_time = float(self.life_time_dist.sample())
        self._clock_running = False
        self._wait_start_time = None

        self._log(action="SPAWNED", life_budget=self._remaining_survival_time)

    @property
    def life_budget(self) -> float:
        """Current remaining survival time, accounting for elapsed clock time."""
        if self._clock_running and self._wait_start_time is not None:
            return self._remaining_survival_time - (self.env.now() - self._wait_start_time)
        return self._remaining_survival_time

    def _loc_id(self):
        loc = getattr(self, "current_facility", None)
        return getattr(loc, "loc_id", None) or str(loc)

    def _log(self, action: str, life_budget):
        if not hasattr(self.env, "patient_log_rows"):
            self.env.patient_log_rows = []

        self.env.patient_log_rows.append({
            "time": float(self.env.now()),
            "patient_id": self.patient_id,
            "action": str(action),
            "life_budget": None if life_budget is None else float(life_budget),
            "needed_unit": self.unit_needed,
            "current_location": self._loc_id(),
            "is_dead": bool(self.is_dead),
            "is_safe": bool(self.is_safe),
        })

    def start_death_clock(self):
        """Start or resume the death clock. Called when patient starts waiting."""
        if self.is_dead or self.is_safe:
            return

        self._clock_running = True
        self._wait_start_time = self.env.now()
        self._log(action="CLOCK_STARTED", life_budget=self._remaining_survival_time)
        self.activate()

    def pause_death_clock(self):
        """Pause the death clock. Called when transport or treatment starts."""
        if not self._clock_running:
            return

        # Deduct elapsed waiting time
        if self._wait_start_time is not None:
            elapsed = self.env.now() - self._wait_start_time
            self._remaining_survival_time -= elapsed
            self._wait_start_time = None

        self._clock_running = False
        self._log(action="CLOCK_PAUSED", life_budget=self._remaining_survival_time)

        # Check if patient died while waiting
        if self._remaining_survival_time <= 0:
            self._die()
        else:
            # Interrupt the hold so process can passivate
            self.activate()

    def extend_survival_time(self):
        """Extend survival time after treatment. Called after care_completed."""
        extension = float(self.life_time_dist.sample())
        self._remaining_survival_time += extension
        self._log(action="SURVIVAL_EXTENDED", life_budget=self._remaining_survival_time)

    def _die(self):
        """Patient dies."""
        self.is_dead = True
        self.patient_state = "dead"
        self._clock_running = False
        self._log(action="DIED", life_budget=self._remaining_survival_time)

    def process(self):
        """Main process - handles death by timeout when clock is running."""
        while True:
            if self.is_dead or self.is_safe:
                return

            if not self._clock_running:
                self.passivate()
                continue

            # Hold for remaining survival time
            self.hold(self._remaining_survival_time)

            # If we reach here without being interrupted, patient died
            if self._clock_running and not self.is_dead and not self.is_safe:
                self._remaining_survival_time = 0
                self._die()
                return

    def care_completed(self, unit):
        """Called when a unit finishes providing care to this patient."""
        if self.is_dead:
            return

        unit_id = unit.name() if hasattr(unit, "name") else str(unit)
        self._log(action=f"CARE_COMPLETED:{unit_id}", life_budget=self._remaining_survival_time)

        # Extend survival time after treatment
        self.extend_survival_time()

        self.rx_index += 1
        if self.rx_index >= len(self.prescription):
            self.is_safe = True
            self.unit_needed = None
            self._log(action="BECAME_SAFE", life_budget=self._remaining_survival_time)
            self.activate(process="out_of_danger")
            return

        self.unit_needed = self.prescription[self.rx_index]
        self._log(action="NEXT_UNIT_SET", life_budget=self._remaining_survival_time)

    def out_of_danger(self):
        """Process for patient who has completed all required care."""
        self.is_safe = True
        self.t_safe = self.env.now()
        self.patient_state = "safe"
        self._clock_running = False
        self.env.safe.append(self)

        self._log(action="LOGGED_SAFE_AND_EXIT", life_budget=self._remaining_survival_time)
        self.cancel()
