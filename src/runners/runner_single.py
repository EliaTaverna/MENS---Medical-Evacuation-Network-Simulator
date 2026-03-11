from __future__ import annotations

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import salabim as sim
import shutil
from datetime import datetime
from src.helpers.casualty_generator import generate_casualties_df_from_waves
from src import (
    load_scenario, load_data_from_scenario, load_casualty_config_from_scenario,
    load_casualties_from_scenario,
    ModelBuilder, ModelConnector, Ambulance, unload_planned, load_planned
)
from src.components.spawner import CasualtySpawner


def run_simulation(
    scenario_path: str = "data/single_run/input/scenarios/scenario_1.yaml",
    # Override scenario settings (optional)
    levers: dict = None,
    till: float = None,
    trace: bool = None,
    casualty_seed: int = None,
    make_video: bool = None,
    save_logs: bool = None,
    state_log_interval: float = None,
    video_fps: int = None,
    video_time_scale: float = None,
    video_dpi: int = None,
    output_folder: str = "data/single_run/output",
):
    """
    Run full simulation from a scenario file.

    Parameters:
        scenario_path: path to scenario YAML file

        # All below override scenario settings if provided:
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
        output_folder: base folder for output (run folder created inside)

    Returns:
        results: dict with env, safe_df, transport_df, facility_states_df
    """
    # Load scenario
    scenario_file = Path(scenario_path)
    scenario = load_scenario(scenario_path)
    scenario_name = scenario.get('name', scenario_file.stem)
    print(f"Loaded scenario: {scenario_name}")

    # Create output run folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = Path(output_folder) / f"{timestamp}_{scenario_name}"
    run_folder.mkdir(parents=True, exist_ok=True)

    # Copy scenario YAML to output folder
    shutil.copy2(scenario_file, run_folder / scenario_file.name)
    print(f"Output folder: {run_folder}")

    # Get settings from scenario, with parameter overrides
    sim_settings = scenario.get("simulation", {})
    video_settings = scenario.get("video", {})
    log_settings = scenario.get("logs", {})

    # Apply defaults from scenario, allow overrides
    levers = levers if levers is not None else scenario.get("levers", {
        "facility_eligibility": 0,
        "facility_assignment": 0,
        "ambulance_eligibility": 0,
        "ambulance_assignment": 0,
    })
    till = till if till is not None else sim_settings.get("till", 1000.0)
    trace = trace if trace is not None else sim_settings.get("trace", False)
    casualty_seed = casualty_seed if casualty_seed is not None else sim_settings.get("casualty_seed")

    make_video = make_video if make_video is not None else video_settings.get("enabled", False)
    video_fps = video_fps if video_fps is not None else video_settings.get("fps", 24)
    video_time_scale = video_time_scale if video_time_scale is not None else video_settings.get("time_scale", 5.0)
    video_dpi = video_dpi if video_dpi is not None else video_settings.get("dpi", 100)
    state_log_interval = state_log_interval if state_log_interval is not None else video_settings.get("state_log_interval", 5.0)
    video_duration = video_settings.get("duration")  # None = until last transport event

    save_logs = save_logs if save_logs is not None else log_settings.get("enabled", True)

    # Load or generate casualties
    cas_gen = scenario.get("casualty_generation", {})
    if cas_gen.get("from_dataset", False):
        # Load pre-generated casualties from file
        casualties_df = load_casualties_from_scenario(scenario)
        print(f"Loaded {len(casualties_df)} casualties from dataset")
    else:
        # Generate casualties dynamically
        config = load_casualty_config_from_scenario(scenario)

        if casualty_seed is not None:
            config["seed"] = casualty_seed

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

    # Load simulation data from scenario
    data = load_data_from_scenario(scenario)
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
        # Build src
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
            from src.helpers.visualization import FacilityStateLogger
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
        env.patient_log_df.to_excel(run_folder / "patient_log.xlsx", index=False)
        transport_df.to_excel(run_folder / "transport_events.xlsx", index=False)
        facility_states_df.to_excel(run_folder / "facility_states.xlsx", index=False)
        safe_df.to_excel(run_folder / "safe_patients.xlsx", index=False)

        print(f"Logs saved to {run_folder}/")

    # Generate video if requested
    if make_video and len(transport_df) > 0:
        import matplotlib
        matplotlib.use('Agg')

        from src.helpers.visualization import create_transport_video

        video_path = run_folder / "simulation.mp4"
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
            save_path=str(video_path),
            duration=video_duration,
        )

    return {
        "env": env,
        "safe_df": safe_df,
        "transport_df": transport_df,
        "facility_states_df": facility_states_df,
        "patient_log_df": env.patient_log_df,
        "run_folder": run_folder,
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
    # Run with default scenario
    results = run_simulation(
        scenario_path="data/single_run/input/scenarios/scenario_2.yaml",
    )

    print_summary(results)
