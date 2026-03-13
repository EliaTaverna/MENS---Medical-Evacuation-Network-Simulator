from __future__ import annotations

import os
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
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

    # Track waiting times: list of (triage, facility_type, wait_type, duration)
    # wait_type: "care" or "transport"
    env.wait_times = []

    # Track unit utilization: dict of unit_id -> list of (start_time, end_time)
    env.unit_busy_periods = {}

    # Track all patients with their triages (for statistics)
    env.all_patients = {}  # patient_id -> Patient object

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

    # Patch Facility to track transport wait times
    from src.components.facility import Facility
    original_patient_handed_over = Facility.patient_handed_over

    def patched_patient_handed_over(self, p):
        # Record when patient starts waiting for transport
        if p.unit_needed not in self.entry_lines:
            p._transport_wait_start = self.env.now()
            p._transport_wait_facility_type = self.facility_type
        original_patient_handed_over(self, p)

    Facility.patient_handed_over = patched_patient_handed_over

    original_facility_process = Facility.process

    def patched_facility_process(self):
        while True:
            self.passivate()
            amb = self._amb
            self._amb = None

            unload_fn = self.env.unload_policy[self.unload_key]
            for p in unload_fn(amb, self):
                amb.onboard.remove(p)
                self.patient_handed_over(p)

            load_fn = self.env.load_policy[self.load_key]
            for p in load_fn(amb, self):
                # Record transport wait time
                if hasattr(p, "_transport_wait_start") and p._transport_wait_start is not None:
                    wait_duration = self.env.now() - p._transport_wait_start
                    self.env.wait_times.append({
                        "triage": p.triage,
                        "facility_type": p._transport_wait_facility_type,
                        "wait_type": "transport",
                        "duration": wait_duration,
                    })
                    p._transport_wait_start = None

                # remove the assigned request from exit_queue
                for req in list(self.exit_queue):
                    if req.patient is p and req.status == "assigned":
                        self.exit_queue.remove(req)
                        break
                # Pause death clock - patient is being transported
                p.pause_death_clock()
                amb.onboard.append(p)
                p.patient_state = "in_transport"

            amb.activate()

    Facility.process = patched_facility_process

    # Patch Unit to track care wait times
    from src.components.unit import Unit, UnitService
    original_unit_process = Unit.process

    def patched_unit_process(self):
        while True:
            q = self.facility.entry_lines[self.service_type]

            while self.in_service < self.capacity and len(q) > 0:
                priority = {"T1": 0, "T2": 1, "T3": 2}

                idx = min(
                    range(len(q)),
                    key=lambda i: (priority.get(q[i].triage, 99), q[i]._unit_queue_entry_time)
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
                    wait_duration = self.env.now() - t_enter
                    self.wait_monitor.tally(wait_duration)
                    # Record care wait time
                    self.env.wait_times.append({
                        "triage": p.triage,
                        "facility_type": self.facility.facility_type,
                        "wait_type": "care",
                        "duration": wait_duration,
                        "unit_id": self.name(),
                    })

                self.in_service += 1
                # Track unit busy period start
                unit_id = self.name()
                if unit_id not in self.env.unit_busy_periods:
                    self.env.unit_busy_periods[unit_id] = []
                p._unit_service_start = self.env.now()
                p._unit_service_unit_id = unit_id

                UnitService(name=f"{self.name()}_svc", unit=self, patient=p)

            self.passivate()

    Unit.process = patched_unit_process

    original_service_done = Unit.service_done

    def patched_service_done(self, p):
        # Track unit busy period end
        if hasattr(p, "_unit_service_start") and p._unit_service_start is not None:
            unit_id = p._unit_service_unit_id
            self.env.unit_busy_periods[unit_id].append({
                "start": p._unit_service_start,
                "end": self.env.now(),
            })
            p._unit_service_start = None
        original_service_done(self, p)

    Unit.service_done = patched_service_done

    # Patch Patient to track all patients
    from src.components.patient import Patient
    original_patient_setup = Patient.setup

    def patched_patient_setup(self, patient_id, triage, prescription, life_time_dist, spawn_facility, injury_time):
        original_patient_setup(self, patient_id, triage, prescription, life_time_dist, spawn_facility, injury_time)
        # Register patient for statistics
        self.env.all_patients[patient_id] = self

    Patient.setup = patched_patient_setup

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
        Facility.patient_handed_over = original_patient_handed_over
        Facility.process = original_facility_process
        Unit.process = original_unit_process
        Unit.service_done = original_service_done
        Patient.setup = original_patient_setup

    # Collect results
    transport_df = pd.DataFrame(env.transport_events)
    facility_states_df = pd.DataFrame(env.facility_states)
    wait_times_df = pd.DataFrame(env.wait_times) if env.wait_times else pd.DataFrame()

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
        "wait_times_df": wait_times_df,
        "unit_busy_periods": env.unit_busy_periods,
        "all_patients": env.all_patients,
        "sim_duration": till,
        "run_folder": run_folder,
        "distance_matrix": data["distance_matrix"],
    }


def print_summary(results: dict):
    """Print simulation summary statistics."""
    log_df = results["patient_log_df"]
    wait_times_df = results.get("wait_times_df", pd.DataFrame())
    transport_df = results.get("transport_df", pd.DataFrame())
    unit_busy_periods = results.get("unit_busy_periods", {})
    sim_duration = results.get("sim_duration", 0)
    distance_matrix = results.get("distance_matrix")

    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    # -----------------------------------------------------------------
    # 1. Patient outcomes by triage
    # -----------------------------------------------------------------
    env = results.get("env")

    # Build patient -> triage mapping from env.all_patients (most reliable source)
    triage_map = {}
    if env and hasattr(env, "all_patients"):
        for pid, patient in env.all_patients.items():
            triage_map[pid] = patient.triage

    # Get final states from log
    final_states = log_df.groupby("patient_id").last().reset_index()
    final_states["triage"] = final_states["patient_id"].map(triage_map)

    print("\n--- PATIENT OUTCOMES ---")
    n_total = len(final_states)
    print(f"Total patients: {n_total}")

    for triage in ["T1", "T2", "T3"]:
        subset = final_states[final_states["triage"] == triage]
        n_triage = len(subset)
        if n_triage == 0:
            continue
        n_dead = (subset["is_dead"] == True).sum()
        n_safe = (subset["is_safe"] == True).sum()
        n_in_progress = n_triage - n_dead - n_safe

        print(f"\n  {triage}: {n_triage} patients")
        print(f"    Deaths:      {n_dead:4d} ({100*n_dead/n_triage:5.1f}%)")
        print(f"    Safe:        {n_safe:4d} ({100*n_safe/n_triage:5.1f}%)")
        print(f"    In progress: {n_in_progress:4d} ({100*n_in_progress/n_triage:5.1f}%)")

    # Overall totals
    n_dead_total = (final_states["is_dead"] == True).sum()
    n_safe_total = (final_states["is_safe"] == True).sum()
    n_in_progress_total = n_total - n_dead_total - n_safe_total
    print(f"\n  TOTAL:")
    print(f"    Deaths:      {n_dead_total:4d} ({100*n_dead_total/n_total:5.1f}%)")
    print(f"    Safe:        {n_safe_total:4d} ({100*n_safe_total/n_total:5.1f}%)")
    print(f"    In progress: {n_in_progress_total:4d} ({100*n_in_progress_total/n_total:5.1f}%)")

    # -----------------------------------------------------------------
    # 2. Waiting time statistics
    # -----------------------------------------------------------------
    if not wait_times_df.empty:
        print("\n--- WAITING TIME STATISTICS (minutes) ---")
        print("Format: 25th percentile / Median / 75th percentile")

        # Get unique facility types and wait types
        facility_types = sorted(wait_times_df["facility_type"].unique())
        triages = ["T1", "T2", "T3"]

        for wait_type in ["care", "transport"]:
            print(f"\n  Waiting for {wait_type.upper()}:")
            wt_subset = wait_times_df[wait_times_df["wait_type"] == wait_type]

            if wt_subset.empty:
                print("    No data")
                continue

            for facility_type in facility_types:
                ft_subset = wt_subset[wt_subset["facility_type"] == facility_type]
                if ft_subset.empty:
                    continue

                print(f"\n    At {facility_type}:")
                for triage in triages:
                    t_subset = ft_subset[ft_subset["triage"] == triage]
                    if t_subset.empty or len(t_subset) == 0:
                        continue

                    durations = t_subset["duration"].values
                    p25 = np.percentile(durations, 25)
                    p50 = np.percentile(durations, 50)
                    p75 = np.percentile(durations, 75)

                    print(f"      {triage}: {p25:6.1f} / {p50:6.1f} / {p75:6.1f}  (n={len(durations)})")

    # -----------------------------------------------------------------
    # 3. Cumulative transport time
    # -----------------------------------------------------------------
    if not transport_df.empty and distance_matrix is not None:
        print("\n--- CUMULATIVE TRANSPORT TIME ---")

        # Calculate travel times from transport events
        departures = transport_df[transport_df["event"] == "departure"].copy()
        arrivals = transport_df[transport_df["event"] == "arrival"].copy()

        # Match departures to arrivals by amb_id and sequence
        amb_travel_times = {}
        for amb_id in transport_df["amb_id"].unique():
            amb_deps = departures[departures["amb_id"] == amb_id].sort_values("time")
            amb_arrs = arrivals[arrivals["amb_id"] == amb_id].sort_values("time")

            total_time = 0
            for _, arr in amb_arrs.iterrows():
                # Find matching departure (latest departure before this arrival with same from/to)
                matching_deps = amb_deps[
                    (amb_deps["time"] < arr["time"]) &
                    (amb_deps["from_loc"] == arr["from_loc"]) &
                    (amb_deps["to_loc"] == arr["to_loc"])
                ]
                if not matching_deps.empty:
                    dep = matching_deps.iloc[-1]
                    travel_time = arr["time"] - dep["time"]
                    total_time += travel_time

            amb_travel_times[amb_id] = total_time

        total_transport_time = sum(amb_travel_times.values())
        print(f"  Total transport time (all ambulances): {total_transport_time:.1f} minutes")

        print("\n  By ambulance:")
        for amb_id, travel_time in sorted(amb_travel_times.items()):
            print(f"    {amb_id}: {travel_time:.1f} minutes")

    # -----------------------------------------------------------------
    # 4. Unit utilization
    # -----------------------------------------------------------------
    if unit_busy_periods and sim_duration > 0:
        print("\n--- UNIT UTILIZATION ---")

        for unit_id, periods in sorted(unit_busy_periods.items()):
            if not periods:
                print(f"  {unit_id}: 0.0%")
                continue

            # Calculate total busy time
            total_busy = sum(p["end"] - p["start"] for p in periods)
            utilization = (total_busy / sim_duration) * 100

            print(f"  {unit_id}: {utilization:.1f}% ({total_busy:.1f} min busy / {sim_duration:.1f} min total)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    # Run with default scenario
    results = run_simulation(
        scenario_path="data/single_run/input/scenarios/scenario_2.yaml",
    )

    print_summary(results)
