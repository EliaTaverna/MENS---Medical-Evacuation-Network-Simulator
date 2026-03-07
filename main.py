from __future__ import annotations

import salabim as sim
from casualty_generator import (
    load_casualty_generation_config,
    generate_casualties_df_from_waves,
    DEFAULT_CONFIG_DIR,
)
from model import load_data_from_csv, run_model


def run_simulation(
    data_folder: str = "data",
    levers: dict = None,
    till: float = 30000.0,
    trace: bool = False,
    casualty_seed: int = None,
):
    """
    Run full simulation: generate casualties and run the model.

    Parameters:
        data_folder: path to data folder
        levers: dict of lever values for C2 policies
        till: simulation end time (minutes)
        trace: enable salabim trace
        casualty_seed: optional seed override for casualty generation

    Returns:
        env: simulation environment with patient_log_df
        safe_df: DataFrame of patients who became safe
    """
    if levers is None:
        levers = {
            "facility_eligibility": 0,
            "facility_assignment": 0,
            "ambulance_eligibility": 0,
            "ambulance_assignment": 0,
        }

    # Load casualty generation config
    config = load_casualty_generation_config(DEFAULT_CONFIG_DIR)

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

    # Load simulation data (facilities, units, ambulances, etc.) - skip casualties
    data = load_data_from_csv(data_folder, load_casualties=False)

    # Use freshly generated casualties
    data["casualties_df"] = casualties_df

    # Run simulation
    env, safe_df = run_model(
        distance_matrix=data["distance_matrix"],
        axp_df=data["axp_df"],
        facility_df=data["facility_df"],
        units_df=data["units_df"],
        ambulance_df=data["ambulance_df"],
        c2_df=data["c2_df"],
        casualties_df=data["casualties_df"],
        levers=levers,
        till=till,
        trace=trace,
    )

    return env, safe_df


if __name__ == "__main__":
    env, safe_df = run_simulation(
        data_folder="data",
        levers={
            "facility_eligibility": 0,
            "facility_assignment": 0,
            "ambulance_eligibility": 0,
            "ambulance_assignment": 0,
        },
        till=30000,
        trace=False,
    )

    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)

    log_df = env.patient_log_df

    # Count unique patients by final state
    final_states = log_df.groupby("patient_id").last()
    n_dead = (final_states["is_dead"] == True).sum()
    n_safe = (final_states["is_safe"] == True).sum()
    n_total = len(final_states)

    print(f"Total patients: {n_total}")
    print(f"Died: {n_dead} ({100*n_dead/n_total:.1f}%)")
    print(f"Safe: {n_safe} ({100*n_safe/n_total:.1f}%)")
    print(f"In progress: {n_total - n_dead - n_safe}")
