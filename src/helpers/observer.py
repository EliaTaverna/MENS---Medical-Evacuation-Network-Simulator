from __future__ import annotations

from src.components.ambulance import Ambulance
from src.components.facility import Facility
from src.components.unit import Unit
from src.components.patient import Patient


class SimulationObserver:
    """
    Context manager that attaches observability to a salabim simulation environment.

    Monkey-patches Ambulance, Facility, Unit and Patient to collect transport
    events, wait times, unit utilization and patient registry — then restores
    all original methods on exit.

    Usage:
        env = sim.Environment(...)
        with SimulationObserver(env) as observer:
            # build model, run simulation
            env.run(till=till)
        data = observer.results()
    """

    def __init__(self, env):
        self.env = env
        self._originals: dict = {}

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "SimulationObserver":
        self._init_collections()
        self._patch_all()
        return self

    def __exit__(self, *args) -> None:
        self._restore_all()

    # ------------------------------------------------------------------ #
    # Collection initialisation                                            #
    # ------------------------------------------------------------------ #

    def _init_collections(self) -> None:
        env = self.env
        env.transport_events = []
        env.wait_times = []
        env.unit_busy_periods = {}
        env.all_patients = {}
        env.safe = []
        env.facility_states = []
        env.patient_log_rows = []
        env.patient_log_columns = [
            "time", "patient_id", "action", "life_budget",
            "needed_unit", "current_location", "is_dead", "is_safe",
        ]

    # ------------------------------------------------------------------ #
    # Patching                                                             #
    # ------------------------------------------------------------------ #

    def _patch_all(self) -> None:
        self._patch_ambulance()
        self._patch_facility_handed_over()
        self._patch_facility_process()
        self._patch_unit_service_done()
        self._patch_patient_setup()

    def _patch_ambulance(self) -> None:
        original = Ambulance._travel_to
        self._originals["ambulance_travel_to"] = original

        def patched_travel_to(self, destination):
            from_id = self._loc_id(self.location)
            to_id = self._loc_id(destination)
            if from_id != to_id:
                self.env.transport_events.append({
                    "time": float(self.env.now()),
                    "event": "departure",
                    "amb_id": self.amb_id,
                    "from_loc": from_id,
                    "to_loc": to_id,
                    "patient_triages": [p.triage for p in self.onboard].copy(),
                })
            original(self, destination)
            if from_id != to_id:
                self.env.transport_events.append({
                    "time": float(self.env.now()),
                    "event": "arrival",
                    "amb_id": self.amb_id,
                    "at_loc": to_id,
                    "from_loc": from_id,
                    "to_loc": to_id,
                    "patient_triages": [p.triage for p in self.onboard].copy(),
                })

        Ambulance._travel_to = patched_travel_to

    def _patch_facility_handed_over(self) -> None:
        original = Facility.patient_handed_over
        self._originals["facility_handed_over"] = original

        def patched_patient_handed_over(self, p):
            if p.unit_needed not in self.entry_lines:
                p._transport_wait_start = self.env.now()
                p._transport_wait_facility_type = self.facility_type
            original(self, p)

        Facility.patient_handed_over = patched_patient_handed_over

    def _patch_facility_process(self) -> None:
        original = Facility.process
        self._originals["facility_process"] = original

        def patched_facility_process(self):
            while True:
                while not self._amb_queue:
                    self.passivate()
                amb = self._amb_queue.pop(0)

                unload_fn = self.env.unload_policy[self.unload_key]
                for p in unload_fn(amb, self):
                    amb.onboard.remove(p)
                    self.patient_handed_over(p)

                load_fn = self.env.load_policy[self.load_key]
                for p in load_fn(amb, self):
                    if hasattr(p, "_transport_wait_start") and p._transport_wait_start is not None:
                        wait_duration = self.env.now() - p._transport_wait_start
                        self.env.wait_times.append({
                            "triage": p.triage,
                            "facility_type": p._transport_wait_facility_type,
                            "wait_type": "transport",
                            "duration": wait_duration,
                        })
                        p._transport_wait_start = None
                    for req in list(self.exit_queue):
                        if req.patient is p and req.status == "assigned":
                            self.exit_queue.remove(req)
                            break
                    p.pause_death_clock()
                    amb.onboard.append(p)
                    p.patient_state = "in_transport"

                amb.activate()

        Facility.process = patched_facility_process

    def _patch_unit_service_done(self) -> None:
        original = Unit.service_done
        self._originals["unit_service_done"] = original

        def patched_service_done(self, p):
            if hasattr(p, "_unit_service_start") and p._unit_service_start is not None:
                unit_id = p._unit_service_unit_id
                self.env.unit_busy_periods[unit_id].append({
                    "start": p._unit_service_start,
                    "end": self.env.now(),
                })
                p._unit_service_start = None
            original(self, p)

        Unit.service_done = patched_service_done

    def _patch_patient_setup(self) -> None:
        original = Patient.setup
        self._originals["patient_setup"] = original

        def patched_patient_setup(self, patient_id, triage, prescription, life_time_dist, spawn_facility, injury_time):
            original(self, patient_id, triage, prescription, life_time_dist, spawn_facility, injury_time)
            self.env.all_patients[patient_id] = self

        Patient.setup = patched_patient_setup

    # ------------------------------------------------------------------ #
    # Restore                                                              #
    # ------------------------------------------------------------------ #

    def _restore_all(self) -> None:
        Ambulance._travel_to = self._originals["ambulance_travel_to"]
        Facility.patient_handed_over = self._originals["facility_handed_over"]
        Facility.process = self._originals["facility_process"]
        Unit.service_done = self._originals["unit_service_done"]
        Patient.setup = self._originals["patient_setup"]

    # ------------------------------------------------------------------ #
    # Results                                                              #
    # ------------------------------------------------------------------ #

    def results(self) -> dict:
        """Return collected simulation data as a dict."""
        return {
            "transport_events": self.env.transport_events,
            "wait_times": self.env.wait_times,
            "unit_busy_periods": self.env.unit_busy_periods,
            "all_patients": self.env.all_patients,
            "facility_states": self.env.facility_states,
            "patient_log_rows": self.env.patient_log_rows,
            "patient_log_columns": self.env.patient_log_columns,
        }
