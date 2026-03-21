"""Unit and UnitService components for medical care."""

from __future__ import annotations
import salabim as sim


class Unit(sim.Component):
    """A medical unit that provides a specific type of care."""

    def setup(self, facility, capacity: int, service_time_dist, service_type: str):
        self.facility = facility
        self.capacity = int(capacity)
        self.service_time_dist = service_time_dist
        self.service_type = service_type
        self.in_service = 0
        self._active_services: list[UnitService] = []

        # realized queue waiting times (when service starts)
        self.wait_monitor = sim.Monitor(name=f"{self.name()}_wait_realized")

        # expected waiting times at queue-join moments
        self.ewait_monitor = sim.Monitor(name=f"{self.name()}_wait_expected_at_join")

    def _dist_mean(self, d) -> float:
        """Get the mean service time from a distribution or dict of distributions."""
        if isinstance(d, dict):
            means = [self._dist_mean(v) for v in d.values()]
            return sum(means) / len(means)
        if hasattr(d, "_mean"):
            return float(d._mean)
        for a_name, b_name in [("_lowerbound", "_upperbound"), ("a", "b"), ("low", "high")]:
            if hasattr(d, a_name) and hasattr(d, b_name):
                return (float(getattr(d, a_name)) + float(getattr(d, b_name))) / 2.0
        return 1.0

    def expected_wait_if_join_now(self) -> float:
        """Calculate expected waiting time if a patient joins the queue now."""
        q = self.facility.entry_lines[self.service_type]
        n = len(q)
        M = self.capacity
        if M == 0:
            return float("inf")
        ES = self._dist_mean(self.service_time_dist)
        return (n * ES) / M

    def _start_service(self, p) -> None:
        """Dequeue patient p, begin their UnitService, and register it."""
        p.pause_death_clock()
        if p.is_dead:
            return

        t_enter = getattr(p, "_unit_queue_entry_time", None)
        if t_enter is not None:
            wait_duration = self.env.now() - t_enter
            self.wait_monitor.tally(wait_duration)
            # Observer-compatible wait time tracking
            if hasattr(self.env, "wait_times"):
                self.env.wait_times.append({
                    "triage": p.triage,
                    "facility_type": self.facility.facility_type,
                    "wait_type": "care",
                    "duration": wait_duration,
                    "unit_id": self.name(),
                })

        # Track service start for busy-period logging
        p._unit_service_start = self.env.now()

        self.in_service += 1
        svc = UnitService(name=f"{self.name()}_svc", unit=self, patient=p)
        self._active_services.append(svc)

    def process(self):
        while True:
            q = self.facility.entry_lines[self.service_type]

            # Fill empty slots with the most urgent queued patients
            while self.in_service < self.capacity and len(q) > 0:
                idx = min(range(len(q)), key=lambda i: q[i].life_budget)
                p = q.pop(idx)
                self._start_service(p)

            self.passivate()

    def service_done(self, p):
        """Called when a patient's service completes normally."""
        self.in_service -= 1
        self._active_services = [s for s in self._active_services if s.patient is not p]

        # Log busy period
        if hasattr(self.env, "unit_busy_periods") and getattr(p, "_unit_service_start", None) is not None:
            unit_id = self.name()
            self.env.unit_busy_periods.setdefault(unit_id, []).append({
                "start": p._unit_service_start,
                "end": self.env.now(),
            })
            p._unit_service_start = None

        if p.is_dead:
            self.activate()
            return

        p.care_completed(self)

        if not p.is_safe and not p.is_dead:
            self.facility.patient_handed_over(p)

        self.activate()


class UnitService(sim.Component):
    """A service session for a single patient at a unit."""

    def setup(self, unit: Unit, patient):
        self.unit = unit
        self.patient = patient

    def process(self):
        dist = self.unit.service_time_dist

        if isinstance(dist, dict):
            triage = getattr(self.patient, "triage", "")
            if "T1_BCAS" in triage or "T2_BCAS" in triage:
                d = dist.get("severe", next(iter(dist.values())))
            else:
                d = dist.get("mild", next(iter(dist.values())))
            duration = max(1.0, d.sample())
        else:
            duration = max(1.0, dist.sample())

        self.hold(duration)
        self.unit.service_done(self.patient)
        self.cancel()
