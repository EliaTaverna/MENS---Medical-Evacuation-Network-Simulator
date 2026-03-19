from __future__ import annotations

import salabim as sim
from pathlib import Path

from src import ModelBuilder, ModelConnector, unload_planned, load_planned
from src.components.spawner import CasualtySpawner
from src.helpers.observer import SimulationObserver
from src.helpers.helpers_run import collect_results


def run_simulation(data: dict, settings: dict, run_folder: Path) -> dict:
    env = sim.Environment(trace=settings["trace"])
    env.levers = settings["levers"]
    env.triage_weights = data.get("triage_weights", {})
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    with SimulationObserver(env) as observer:
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

        CasualtySpawner(casualties_df=data["casualties_df"])
        print(f"Running simulation for {settings['till']} minutes...")
        env.run(till=settings["till"])

    return collect_results(env, observer, data, settings, run_folder)
