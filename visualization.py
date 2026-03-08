"""Visualization module for MEDEVAC simulation with smooth ambulance movement and patient tracking."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import PathCollection
from sklearn.manifold import MDS


def compute_positions_from_distance_matrix(distance_matrix: pd.DataFrame) -> dict:
    """
    Use MDS to infer 2D positions from the distance matrix.

    Returns:
        dict mapping location_id -> (x, y)
    """
    locations = list(distance_matrix.index)
    dist_array = distance_matrix.values.astype(float)

    # Replace any inf/nan with large values
    dist_array = np.nan_to_num(dist_array, nan=1000.0, posinf=1000.0, neginf=0.0)

    # Make symmetric (take average of matrix and its transpose)
    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)

    # MDS to get 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto', n_init=4)
    coords = mds.fit_transform(dist_array)

    return {loc: (coords[i, 0], coords[i, 1]) for i, loc in enumerate(locations)}


def get_facility_type(facility_id: str, facility_df: pd.DataFrame) -> str:
    """Get the facility type for a given facility ID."""
    row = facility_df[facility_df['facility_id'] == facility_id]
    if len(row) > 0:
        return row.iloc[0]['facility_type']
    return 'Unknown'


FACILITY_COLORS = {
    'CCP': '#e74c3c',      # Red
    'Role1': '#3498db',    # Blue
    'Role2': '#2ecc71',    # Green
    'Role3': '#9b59b6',    # Purple
    'AXP': '#f39c12',      # Orange
    'Unknown': '#95a5a6',  # Gray
}

FACILITY_MARKERS = {
    'CCP': 's',      # Square
    'Role1': 'o',    # Circle
    'Role2': '^',    # Triangle up
    'Role3': 'D',    # Diamond
    'AXP': 'P',      # Plus (filled)
    'Unknown': 'x',
}

TRIAGE_COLORS = {
    'T1': '#ff0000',  # Red - most urgent
    'T2': '#ffcc00',  # Yellow - urgent
    'T3': '#00cc00',  # Green - less urgent
}


def plot_network_static(
    distance_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    ambulance_df: pd.DataFrame = None,
    title: str = "MEDEVAC Network",
    figsize: tuple = (14, 10),
):
    """Create a static plot of the MEDEVAC network."""
    positions = compute_positions_from_distance_matrix(distance_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges (connections) - only for nearby facilities
    locations = list(distance_matrix.index)
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i < j:
                dist = distance_matrix.loc[loc1, loc2]
                if dist < 25:
                    x1, y1 = positions[loc1]
                    x2, y2 = positions[loc2]
                    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.1, linewidth=0.5)

    # Draw facilities
    for loc_id, (x, y) in positions.items():
        fac_type = get_facility_type(loc_id, facility_df)
        color = FACILITY_COLORS.get(fac_type, FACILITY_COLORS['Unknown'])
        marker = FACILITY_MARKERS.get(fac_type, 'o')
        size = 200 if fac_type in ['Role2', 'Role3'] else 150

        ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(loc_id, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Draw ambulance starting positions if provided
    if ambulance_df is not None:
        amb_counts = ambulance_df['start_loc_id'].value_counts()
        for loc_id, count in amb_counts.items():
            if loc_id in positions:
                x, y = positions[loc_id]
                ax.annotate(f'[{count} amb]', (x, y), xytext=(5, -15),
                           textcoords='offset points', fontsize=7, color='darkblue')

    # Create legend
    legend_handles = []
    for fac_type, color in FACILITY_COLORS.items():
        if fac_type != 'Unknown':
            marker = FACILITY_MARKERS[fac_type]
            handle = plt.scatter([], [], c=color, marker=marker, s=100,
                                edgecolors='black', linewidths=1, label=fac_type)
            legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc='upper left', title='Facility Type')
    ax.set_title(title)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax, positions


def run_simulation_with_logging(
    data_folder: str = "data",
    levers: dict = None,
    till: float = 1000.0,
):
    """
    Run simulation and collect transport events with patient triage info.

    Returns:
        env: simulation environment
        transport_df: DataFrame of transport events with patient details
        distance_matrix: the distance matrix
        facility_df: facility information
        ambulance_df: ambulance information
    """
    import salabim as sim
    from model import load_data_from_csv
    from model.builder import ModelBuilder, ModelConnector
    from model.spawner import CasualtySpawner
    from model.policies import unload_planned, load_planned
    from model.ambulance import Ambulance
    from casualty_generator import load_casualty_generation_config, generate_casualties_df_from_waves, DEFAULT_CONFIG_DIR

    if levers is None:
        levers = {
            "facility_eligibility": 0,
            "facility_assignment": 0,
            "ambulance_eligibility": 0,
            "ambulance_assignment": 0,
        }

    # Load data
    data = load_data_from_csv(data_folder, load_casualties=False)

    # Generate casualties
    config = load_casualty_generation_config(DEFAULT_CONFIG_DIR)
    casualties_df, _ = generate_casualties_df_from_waves(
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
    data["casualties_df"] = casualties_df

    # Create environment with transport logging
    env = sim.Environment(trace=False)
    env.patient_log_rows = []
    env.patient_log_columns = ["time", "patient_id", "action", "life_budget", "needed_unit", "current_location", "is_dead", "is_safe"]
    env.safe = []
    env.transport_events = []

    env.levers = levers
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    # Monkey-patch Ambulance._travel_to to log events with patient triage info
    original_travel_to = Ambulance._travel_to

    def patched_travel_to(self, destination):
        from_id = self._loc_id(self.location)
        to_id = self._loc_id(destination)

        if from_id != to_id:
            # Collect patient triage info
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

        spawner = CasualtySpawner(casualties_df=data["casualties_df"])
        env.run(till=till)
        env.patient_log_df = pd.DataFrame(env.patient_log_rows, columns=env.patient_log_columns)

    finally:
        Ambulance._travel_to = original_travel_to

    transport_df = pd.DataFrame(env.transport_events)

    return env, transport_df, data["distance_matrix"], data["facility_df"], data["ambulance_df"]


def create_transport_video(
    distance_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    transport_events: pd.DataFrame,
    ambulance_df: pd.DataFrame,
    fps: int = 30,
    time_scale: float = 5.0,
    figsize: tuple = (16, 10),
    save_path: str = "transport_simulation.mp4",
    dpi: int = 150,
):
    """
    Create an MP4 video showing ambulances moving smoothly with patient dots.

    Parameters:
        distance_matrix: DataFrame with distances between locations
        facility_df: DataFrame with facility information
        transport_events: DataFrame with transport events including patient_triages
        ambulance_df: DataFrame with ambulance info (for initial positions)
        fps: Frames per second for the video
        time_scale: Simulation minutes per frame
        figsize: Figure size
        save_path: Path to save the MP4 file
        dpi: Resolution of the video
    """
    positions = compute_positions_from_distance_matrix(distance_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Set up the static background
    ax.set_facecolor('#f5f5f5')

    # Draw edges (connections)
    locations = list(distance_matrix.index)
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i < j:
                dist = distance_matrix.loc[loc1, loc2]
                if dist < 30:
                    x1, y1 = positions[loc1]
                    x2, y2 = positions[loc2]
                    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.15, linewidth=0.5)

    # Draw facilities
    for loc_id, (x, y) in positions.items():
        fac_type = get_facility_type(loc_id, facility_df)
        color = FACILITY_COLORS.get(fac_type, FACILITY_COLORS['Unknown'])
        marker = FACILITY_MARKERS.get(fac_type, 'o')
        size = 250 if fac_type in ['Role2', 'Role3'] else 180
        ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(loc_id, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=7, fontweight='bold')

    # Create legend for facilities
    legend_handles = []
    for fac_type, color in FACILITY_COLORS.items():
        if fac_type != 'Unknown':
            marker = FACILITY_MARKERS[fac_type]
            handle = plt.scatter([], [], c=color, marker=marker, s=100,
                                edgecolors='black', linewidths=1, label=fac_type)
            legend_handles.append(handle)

    # Add patient triage legend
    for triage, color in TRIAGE_COLORS.items():
        handle = plt.scatter([], [], c=color, marker='o', s=60, edgecolors='black', linewidths=0.5, label=f'Patient {triage}')
        legend_handles.append(handle)

    # Add ambulance legend
    handle = plt.scatter([], [], c='white', marker='s', s=100, edgecolors='#333333', linewidths=2, label='Ambulance')
    legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc='upper left', title='Legend', fontsize=8)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Time display
    time_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Stats display
    stats_text = ax.text(0.98, 0.88, '', transform=ax.transAxes, fontsize=10,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Build transport timeline: for each ambulance, track its state over time
    # State: {amb_id: {'location': loc_id, 'in_transit': bool, 'from': loc, 'to': loc, 'start_t': t, 'end_t': t, 'patients': [triages]}}

    # Initialize ambulance states from their starting positions
    amb_initial_state = {}
    for _, row in ambulance_df.iterrows():
        amb_initial_state[row['amb_id']] = {
            'location': row['start_loc_id'],
            'in_transit': False,
            'patients': [],
        }

    # Process events into a timeline
    events_sorted = transport_events.sort_values('time').to_dict('records') if len(transport_events) > 0 else []

    max_time = transport_events['time'].max() if len(transport_events) > 0 else 100
    n_frames = int(max_time / time_scale) + 1

    def get_state_at_time(t):
        """Get the state of all ambulances at time t."""
        # Start from initial state
        amb_state = {amb_id: state.copy() for amb_id, state in amb_initial_state.items()}

        for event in events_sorted:
            if event['time'] > t:
                break

            amb_id = event['amb_id']

            if event['event'] == 'departure':
                from_loc = event['from_loc']
                to_loc = event['to_loc']
                patient_triages = event.get('patient_triages', [])

                # Calculate travel time based on distance and speed (30 km/h)
                dist = distance_matrix.loc[from_loc, to_loc]
                travel_time = (dist / 30.0) * 60.0  # minutes

                amb_state[amb_id] = {
                    'in_transit': True,
                    'from_loc': from_loc,
                    'to_loc': to_loc,
                    'start_time': event['time'],
                    'end_time': event['time'] + travel_time,
                    'patients': patient_triages,
                }

            elif event['event'] == 'arrival':
                at_loc = event['at_loc']
                patient_triages = event.get('patient_triages', [])

                amb_state[amb_id] = {
                    'location': at_loc,
                    'in_transit': False,
                    'patients': patient_triages,
                }

        return amb_state

    def get_ambulance_visual_data(t):
        """Get positions and patient data for all ambulances at time t."""
        state = get_state_at_time(t)

        amb_data = []  # List of (x, y, [patient_triages])

        for amb_id, amb_state in state.items():
            if amb_state.get('in_transit', False):
                # Interpolate position
                start_t = amb_state['start_time']
                end_t = amb_state['end_time']

                if t >= end_t:
                    # Already arrived
                    x, y = positions.get(amb_state['to_loc'], (0, 0))
                elif t >= start_t:
                    progress = (t - start_t) / (end_t - start_t)
                    progress = min(1.0, max(0.0, progress))
                    x1, y1 = positions.get(amb_state['from_loc'], (0, 0))
                    x2, y2 = positions.get(amb_state['to_loc'], (0, 0))
                    x = x1 + progress * (x2 - x1)
                    y = y1 + progress * (y2 - y1)
                else:
                    x, y = positions.get(amb_state.get('location', amb_state.get('from_loc')), (0, 0))

                patients = amb_state.get('patients', [])
                amb_data.append((x, y, patients, True))  # True = in transit
            else:
                loc = amb_state.get('location')
                if loc and loc in positions:
                    x, y = positions[loc]
                    patients = amb_state.get('patients', [])
                    amb_data.append((x, y, patients, False))  # False = stationary

        return amb_data

    # Dynamic elements that will be updated each frame
    ambulance_artists = []
    patient_artists = []

    def init():
        return []

    def animate(frame):
        nonlocal ambulance_artists, patient_artists

        # Remove previous dynamic elements
        for artist in ambulance_artists + patient_artists:
            artist.remove()
        ambulance_artists = []
        patient_artists = []

        t = frame * time_scale
        amb_data = get_ambulance_visual_data(t)

        n_in_transit = 0
        n_patients_moving = 0

        for x, y, patients, in_transit in amb_data:
            if in_transit:
                n_in_transit += 1
                n_patients_moving += len(patients)

            # Draw ambulance as a white square with black border
            amb_color = '#4a90d9' if in_transit else '#cccccc'
            amb_artist = ax.scatter(x, y, c=amb_color, marker='s', s=120,
                                   edgecolors='#333333', linewidths=2, zorder=10)
            ambulance_artists.append(amb_artist)

            # Draw patients as colored dots around/on the ambulance
            if patients:
                n_patients = len(patients)
                # Arrange patients in a small cluster near the ambulance
                for i, triage in enumerate(patients):
                    # Offset patients slightly from ambulance center
                    angle = (2 * np.pi * i / max(n_patients, 1)) + np.pi/4
                    offset = 1.5  # km offset
                    px = x + offset * np.cos(angle)
                    py = y + offset * np.sin(angle)

                    color = TRIAGE_COLORS.get(triage, '#888888')
                    patient_artist = ax.scatter(px, py, c=color, marker='o', s=50,
                                               edgecolors='black', linewidths=0.5, zorder=11)
                    patient_artists.append(patient_artist)

        # Update time display
        hours = int(t // 60)
        mins = int(t % 60)
        time_text.set_text(f'Time: {hours:02d}:{mins:02d} ({t:.0f} min)')

        # Update stats
        stats_text.set_text(f'Ambulances moving: {n_in_transit}\nPatients in transit: {n_patients_moving}')

        ax.set_title(f'MEDEVAC Simulation', fontsize=14, fontweight='bold')

        return ambulance_artists + patient_artists + [time_text, stats_text]

    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000//fps, blit=False)

    # Save as MP4
    print(f"Saving to {save_path}...")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'MEDEVAC Simulation'}, bitrate=5000)
    anim.save(save_path, writer=writer, dpi=dpi)
    print(f"Video saved to {save_path}")

    plt.close(fig)
    return save_path


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')

    from model import load_data_from_csv

    print("Loading data...")
    data = load_data_from_csv("data", load_casualties=False)

    print("Creating static network visualization...")
    fig, ax, positions = plot_network_static(
        data["distance_matrix"],
        data["facility_df"],
        data["ambulance_df"],
        title="MEDEVAC Network Layout"
    )
    plt.savefig("network_static.png", dpi=150)
    print("Saved: network_static.png")
    plt.close()

    print("\nRunning simulation with transport logging...")
    env, transport_df, dist_matrix, facility_df, ambulance_df = run_simulation_with_logging(
        data_folder="data",
        till=500.0
    )

    print(f"Logged {len(transport_df)} transport events")

    if len(transport_df) > 0:
        print("\nCreating transport video...")
        create_transport_video(
            dist_matrix,
            facility_df,
            transport_df,
            ambulance_df,
            fps=30,
            time_scale=2.0,  # 2 simulation minutes per frame
            save_path="transport_simulation.mp4"
        )
