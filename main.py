from __future__ import annotations

import pandas as pd
import salabim as sim
from pathlib import Path
from casualty_generator import (
    load_casualty_generation_config,
    generate_casualties_df_from_waves,
)
from model import load_data_from_xlsx
from model.builder import ModelBuilder, ModelConnector
from model.spawner import CasualtySpawner
from model.policies import unload_planned, load_planned
from model.ambulance import Ambulance


def run_simulation(
    scenario: str = "scenario_1",
    data_folder: str = "data",
    levers: dict = None,
    till: float = 1000.0,
    trace: bool = False,
    casualty_seed: int = None,
    make_video: bool = False,
    save_logs: bool = False,
    state_log_interval: float = 5.0,
    video_fps: int = 24,
    video_time_scale: float = 5.0,
    video_dpi: int = 100,
    video_path: str = "transport_simulation.mp4",
    logs_path: str = "simulation_logs",
):
    """
    Run full simulation with optional video and log export.

    Parameters:
        scenario: scenario folder name (e.g., "scenario_1", "scenario_2")
        data_folder: root data folder
        levers: dict of lever values for C2 policies
        till: simulation end time (minutes)
        trace: enable salabim trace
        casualty_seed: optional seed override for casualty generation

        make_video: if True, generate MP4 visualization
        save_logs: if True, save logs to xlsx files

        state_log_interval: how often to snapshot facility states (minutes)
        video_fps: frames per second for video
        video_time_scale: simulation minutes per video frame
        video_dpi: video resolution
        video_path: output path for video file
        logs_path: folder for log files

    Returns:
        results: dict with env, safe_df, transport_df, facility_states_df
    """
    if levers is None:
        levers = {
            "facility_eligibility": 0,
            "facility_assignment": 0,
            "ambulance_eligibility": 0,
            "ambulance_assignment": 0,
        }

    # Build scenario paths
    scenario_path = Path(data_folder) / scenario
    casualty_gen_path = scenario_path / "casualty_generation"
    sim_init_path = scenario_path / "simulation_initiation"

    # Load casualty generation config
    config = load_casualty_generation_config(casualty_gen_path)

    if casualty_seed is not None:
        config["seed"] = casualty_seed

    # Generate casualties
    casualties_df, wave_dict = generate_casualties_df_from_waves(
        facility_ids=config["facility_ids"],
        par_total=config["par_total"],
        n_days=config["n_days"],
        waves_per_day=config["waves_per_day"],
        wave_duration_min=config["wave_duration_min"],
        rate_per_day=config["rate_per_day"],
        diagnosis_mix_df=config["diagnosis_mix_df"],
        rx_df=config["rx_df"],
        facility_weights=config["facility_weights"],
        lifetime_factory_by_triage=config["lifetime_factory_by_triage"],
        seed=config["seed"],
    )

    print(f"Generated {len(casualties_df)} casualties")

    # Load simulation data
    data = load_data_from_xlsx(str(scenario_path), load_casualties=False)
    data["casualties_df"] = casualties_df

    # Create environment
    env = sim.Environment(trace=trace)
    env.patient_log_rows = []
    env.patient_log_columns = [
        "time", "patient_id", "action", "life_budget",
        "needed_unit", "current_location", "is_dead", "is_safe"
    ]
    env.safe = []
    env.transport_events = []
    env.facility_states = []

    env.levers = levers
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    # Patch ambulance to log transport events
    original_travel_to = Ambulance._travel_to

    def patched_travel_to(self, destination):
        from_id = self._loc_id(self.location)
        to_id = self._loc_id(destination)

        if from_id != to_id:
            patient_triages = [p.triage for p in self.onboard]
            self.env.transport_events.append({
                'time': float(self.env.now()),
                'event': 'departure',
                'amb_id': self.amb_id,
                'from_loc': from_id,
                'to_loc': to_id,
                'patient_triages': patient_triages.copy(),
            })

        original_travel_to(self, destination)

        if from_id != to_id:
            patient_triages = [p.triage for p in self.onboard]
            self.env.transport_events.append({
                'time': float(self.env.now()),
                'event': 'arrival',
                'amb_id': self.amb_id,
                'at_loc': to_id,
                'from_loc': from_id,
                'to_loc': to_id,
                'patient_triages': patient_triages.copy(),
            })

    Ambulance._travel_to = patched_travel_to

    try:
        # Build model
        builder = ModelBuilder(env)
        builder.build_all(
            data["distance_matrix"],
            data["axp_df"],
            data["facility_df"],
            data["units_df"],
            data["ambulance_df"],
            data["c2_df"]
        )

        connector = ModelConnector(env)
        connector.connect_all()

        # Start facility state logger if making video or saving logs
        if make_video or save_logs:
            from visualization import FacilityStateLogger
            FacilityStateLogger(interval=state_log_interval)

        # Spawn casualties and run
        spawner = CasualtySpawner(casualties_df=data["casualties_df"])

        print(f"Running simulation for {till} minutes...")
        env.run(till=till)

        env.patient_log_df = pd.DataFrame(
            env.patient_log_rows,
            columns=env.patient_log_columns
        )

    finally:
        Ambulance._travel_to = original_travel_to

    # Collect results
    transport_df = pd.DataFrame(env.transport_events)
    facility_states_df = pd.DataFrame(env.facility_states)

    safe_data = [
        (p.patient_id, p.triage, p.t_safe, p.spawn_facility.loc_id)
        for p in env.safe
    ]
    safe_df = pd.DataFrame(
        safe_data,
        columns=["patient_id", "triage", "t_safe", "spawn_facility_id"]
    )

    print(f"Simulation complete. {len(env.safe)} patients safe.")
    print(f"Logged {len(transport_df)} transport events")
    print(f"Logged {len(facility_states_df)} facility state snapshots")

    # Save logs if requested
    if save_logs:
        import os
        os.makedirs(logs_path, exist_ok=True)

        env.patient_log_df.to_excel(f"{logs_path}/patient_log.xlsx", index=False)
        transport_df.to_excel(f"{logs_path}/transport_events.xlsx", index=False)
        facility_states_df.to_excel(f"{logs_path}/facility_states.xlsx", index=False)
        safe_df.to_excel(f"{logs_path}/safe_patients.xlsx", index=False)

        print(f"Logs saved to {logs_path}/")

    # Generate video if requested
    if make_video and len(transport_df) > 0:
        import matplotlib
        matplotlib.use('Agg')

        from visualization import create_transport_video

        print(f"\nCreating video...")
        create_transport_video(
            distance_matrix=data["distance_matrix"],
            facility_df=data["facility_df"],
            transport_events=transport_df,
            ambulance_df=data["ambulance_df"],
            facility_states_df=facility_states_df,
            fps=video_fps,
            time_scale=video_time_scale,
            dpi=video_dpi,
            save_path=video_path,
        )

    return {
        "env": env,
        "safe_df": safe_df,
        "transport_df": transport_df,
        "facility_states_df": facility_states_df,
        "patient_log_df": env.patient_log_df,
    }


def print_summary(results: dict):
    """Print simulation summary statistics."""
    log_df = results["patient_log_df"]

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    # Count unique patients by final state
    final_states = log_df.groupby("patient_id").last()
    n_dead = (final_states["is_dead"] == True).sum()
    n_safe = (final_states["is_safe"] == True).sum()
    n_total = len(final_states)

    print(f"Total patients: {n_total}")
    print(f"Died: {n_dead} ({100*n_dead/n_total:.1f}%)")
    print(f"Safe: {n_safe} ({100*n_safe/n_total:.1f}%)")
    print(f"In progress: {n_total - n_dead - n_safe}")


if __name__ == "__main__":
    # ============================================================
    # CONFIGURATION - Edit these parameters
    # ============================================================

    results = run_simulation(
        # Scenario and data settings
        scenario="scenario_1",       # Scenario folder name
        data_folder="data",          # Root data folder
        till=5000,                    # Simulation duration (minutes)
        trace=False,
        casualty_seed=None,          # Set for reproducibility

        # Policy levers
        levers={
            "facility_eligibility": 0,
            "facility_assignment": 0,
            "ambulance_eligibility": 0,
            "ambulance_assignment": 0,
        },

        # Output options
        make_video=True,             # Generate MP4 video
        save_logs=True,              # Save CSV logs

        # Video settings
        state_log_interval=5.0,      # Facility state snapshot interval (minutes)
        video_fps=24,                # Frames per second
        video_time_scale=5.0,        # Sim minutes per frame (higher = faster)
        video_dpi=100,               # Resolution
        video_path="transport_simulation.mp4",

        # Log settings
        logs_path="simulation_logs",
    )

    print_summary(results)
