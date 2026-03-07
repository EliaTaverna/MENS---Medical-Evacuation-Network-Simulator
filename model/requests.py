"""Transport request and plan step data structures."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model.patient import Patient
    from model.facility import Facility


class TransportRequest:
    """A request for patient transport between facilities."""

    def __init__(
        self,
        req_id: str,
        patient: Patient,
        issuing_facility: Facility,
        t_issued: float,
        injury_time: float,
        triage: str,
        needed_unit: str | None,
    ):
        self.req_id = req_id
        self.patient = patient
        self.casualty_id = patient.patient_id
        self.triage = triage
        self.issuing_facility = issuing_facility
        self.injury_time = float(injury_time)
        self.t_issued = float(t_issued)
        self.needed_unit = needed_unit
        self.status = "pending"


class PlanStep:
    """A single step in an ambulance's transport plan."""

    def __init__(self, facility, unload_patients, load_patients):
        self.facility = facility
        self.unload_patients = list(unload_patients)
        self.load_patients = list(load_patients)
