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
        # Fallback for Uniform via bounds
        for a_name, b_name in [("_lowerbound", "_upperbound"), ("a", "b"), ("low", "high")]:
            if hasattr(d, a_name) and hasattr(d, b_name):
                return (float(getattr(d, a_name)) + float(getattr(d, b_name))) / 2.0
        return 1.0

    def expected_wait_if_join_now(self) -> float:
        """Calculate expected waiting time if a patient joins the queue now."""
        q = self.facility.entry_lines[self.service_type]
        n = len(q)  # people ahead
        M = self.capacity

        if M == 0:
            return float("inf")

        ES = self._dist_mean(self.service_time_dist)
        return (n * ES) / M

    def process(self):
        while True:
            q = self.facility.entry_lines[self.service_type]

            while self.in_service < self.capacity and len(q) > 0:
                idx = min(
                    range(len(q)),
                    key=lambda i: q[i]._unit_queue_entry_time,
                )
                p = q.pop(idx)

                # Pause death clock - treatment is starting
                p.pause_death_clock()

                # Check if patient died while waiting
                if p.is_dead:
                    continue

                # realized waiting time
                t_enter = getattr(p, "_unit_queue_entry_time", None)
                if t_enter is not None:
                    self.wait_monitor.tally(self.env.now() - t_enter)

                self.in_service += 1
                UnitService(name=f"{self.name()}_svc", unit=self, patient=p)

            self.passivate()

    def service_done(self, p):
        """Called when a patient's service is complete."""
        self.in_service -= 1

        # Patient might have died while waiting (clock was paused during treatment,
        # but death could have been triggered when pause_death_clock was called)
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
            duration = dist.sample()
        self.hold(duration)
        self.unit.service_done(self.patient)
        self.cancel()
