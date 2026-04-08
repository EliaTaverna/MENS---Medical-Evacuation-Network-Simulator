from __future__ import annotations

import gc
import salabim as sim
from pathlib import Path

from src import ModelBuilder, ModelConnector, unload_planned, load_planned
from src.components.spawner import CasualtySpawner
from src.components.data_collector import DataCollector
from src.helpers.helpers_run import collect_results


def run_simulation(data: dict, settings: dict, run_folder: Path) -> dict:
    env = sim.Environment(trace=settings["trace"], random_seed=settings.get("sim_seed", settings.get("seed", 42)))
    env.levers = settings["levers"]
    env.triage_weights = data.get("triage_weights", {})
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    # Initialise data-collection containers
    env.transport_events = []
    env.wait_times = []
    env.unit_busy_periods = {}
    env.all_patients = {}
    env.safe = []
    env.facility_states = []
    env.patient_log_rows = []
    env.c2_dispatch_logs = {}
    env.data_collector_records = []          # c2_id -> list[dispatch-decision rows]
    env.patient_log_columns = [
        "time", "patient_id", "triage", "diagnosis", "action", "life_budget",
        "needed_unit", "current_location", "is_dead", "is_safe",
    ]

    ModelBuilder(env).build_all(
        data["distance_matrix"], data["axp_df"], data["facility_df"],
        data["units_df"], data["ambulance_df"], data["c2_df"],
        fac_eligibility_datasets=data.get("fac_eligibility_datasets"),
        amb_eligibility_datasets=data.get("amb_eligibility_datasets"),
    )
    ModelConnector(env).connect_all()

    if settings["make_video"] or settings.get("outputs", {}).get("facility_states", False):
        from src.helpers.visualization import FacilityStateLogger
        FacilityStateLogger(interval=settings["state_log_interval"])

    dc_interval = settings.get("data_collector_interval", 0.0)
    if dc_interval and dc_interval > 0:
        DataCollector(interval=float(dc_interval))

    CasualtySpawner(casualties_df=data["casualties_df"])
    print(f"Running simulation for {settings['till']} minutes...")
    env.run(till=settings["till"])

    results = collect_results(env, data, settings, run_folder)

    # Explicitly drop the salabim environment so all greenlet stacks are
    # freed immediately rather than waiting for Python's cyclic GC.
    del env
    gc.collect()

    return results
