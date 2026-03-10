"""
MEDEVAC Simulation Model Package

Modular discrete-event simulation for medical evacuation operations.
"""

from model.requests import TransportRequest, PlanStep
from model.patient import Patient
from model.unit import Unit, UnitService
from model.facility import Facility
from model.axp import AXP
from model.ambulance import Ambulance
from model.c2 import TransportC2
from model.spawner import CasualtySpawner
from model.builder import ModelBuilder, ModelConnector
from model.runner import run_model, load_data_from_xlsx
from model.policies import (
    unload_planned,
    load_planned,
    fac_eligibility_base,
    fac_assign_template,
    amb_eligibility_base,
    amb_assign_template,
    craft_plans,
    dispatch_ambulances,
    snapshot_facilities,
    FACILITY_ELIGIBILITY_POLICIES,
    FACILITY_ASSIGNMENT_POLICIES,
    AMBULANCE_ELIGIBILITY_POLICIES,
    AMBULANCE_ASSIGNMENT_POLICIES,
)

__all__ = [
    # Core components
    "Patient",
    "Unit",
    "UnitService",
    "Facility",
    "AXP",
    "Ambulance",
    "TransportC2",
    "CasualtySpawner",
    # Data structures
    "TransportRequest",
    "PlanStep",
    # Builder/Connector
    "ModelBuilder",
    "ModelConnector",
    # Runner
    "run_model",
    "load_data_from_xlsx",
    # Policies
    "unload_planned",
    "load_planned",
    "fac_eligibility_base",
    "fac_assign_template",
    "amb_eligibility_base",
    "amb_assign_template",
    "craft_plans",
    "dispatch_ambulances",
    "snapshot_facilities",
    "FACILITY_ELIGIBILITY_POLICIES",
    "FACILITY_ASSIGNMENT_POLICIES",
    "AMBULANCE_ELIGIBILITY_POLICIES",
    "AMBULANCE_ASSIGNMENT_POLICIES",
]
