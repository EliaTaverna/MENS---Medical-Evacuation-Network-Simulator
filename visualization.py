"""Visualization module for MEDEVAC simulation."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
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


def plot_network_static(
    distance_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    ambulance_df: pd.DataFrame = None,
    title: str = "MEDEVAC Network",
    show_distances: bool = False,
    figsize: tuple = (14, 10),
):
    """
    Create a static plot of the MEDEVAC network.

    Parameters:
        distance_matrix: DataFrame with distances between locations
        facility_df: DataFrame with facility information
        ambulance_df: Optional DataFrame with ambulance starting positions
        title: Plot title
        show_distances: If True, show distance labels on edges
        figsize: Figure size
    """
    positions = compute_positions_from_distance_matrix(distance_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges (connections) - only for nearby facilities
    locations = list(distance_matrix.index)
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i < j:
                dist = distance_matrix.loc[loc1, loc2]
                if dist < 25:  # Only show nearby connections
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
    ax.set_xlabel('X (km, inferred from distances)')
    ax.set_ylabel('Y (km, inferred from distances)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax, positions


class TransportLogger:
    """Logs transport events during simulation for visualization."""

    def __init__(self):
        self.events = []

    def log_departure(self, time: float, amb_id: str, from_loc: str, to_loc: str, patients_onboard: int):
        self.events.append({
            'time': time,
            'event': 'departure',
            'amb_id': amb_id,
            'from_loc': from_loc,
            'to_loc': to_loc,
            'patients': patients_onboard,
        })

    def log_arrival(self, time: float, amb_id: str, at_loc: str, patients_onboard: int):
        self.events.append({
            'time': time,
            'event': 'arrival',
            'amb_id': amb_id,
            'at_loc': at_loc,
            'patients': patients_onboard,
        })

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.events)


def create_transport_animation(
    distance_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    transport_events: pd.DataFrame,
    interval_ms: int = 100,
    time_scale: float = 10.0,
    figsize: tuple = (14, 10),
    save_path: str = None,
):
    """
    Create an animation of ambulance transports.

    Parameters:
        distance_matrix: DataFrame with distances between locations
        facility_df: DataFrame with facility information
        transport_events: DataFrame with columns [time, event, amb_id, from_loc, to_loc, at_loc, patients]
        interval_ms: Milliseconds between animation frames
        time_scale: Simulation minutes per animation frame
        figsize: Figure size
        save_path: If provided, save animation to this file
    """
    positions = compute_positions_from_distance_matrix(distance_matrix)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw static network
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

    # Create legend
    legend_handles = []
    for fac_type, color in FACILITY_COLORS.items():
        if fac_type != 'Unknown':
            marker = FACILITY_MARKERS[fac_type]
            handle = plt.scatter([], [], c=color, marker=marker, s=100,
                                edgecolors='black', linewidths=1, label=fac_type)
            legend_handles.append(handle)
    ax.legend(handles=legend_handles, loc='upper left', title='Facility Type')

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Ambulance markers (will be updated each frame)
    amb_scatter = ax.scatter([], [], c='red', marker='>', s=80, zorder=10, label='Ambulances')
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

    # Track active transports
    active_transports = {}  # amb_id -> {from_loc, to_loc, start_time, end_time}

    max_time = transport_events['time'].max() if len(transport_events) > 0 else 100
    n_frames = int(max_time / time_scale) + 1

    def get_ambulance_positions_at_time(t):
        """Get all ambulance positions at simulation time t."""
        amb_positions = {}

        for _, event in transport_events.iterrows():
            amb_id = event['amb_id']
            event_time = event['time']

            if event_time > t:
                continue

            if event['event'] == 'departure':
                # Calculate travel time
                from_loc = event['from_loc']
                to_loc = event['to_loc']
                if from_loc in positions and to_loc in positions:
                    dist = distance_matrix.loc[from_loc, to_loc]
                    travel_time = (dist / 30.0) * 60.0  # Assume 30 km/h
                    active_transports[amb_id] = {
                        'from_loc': from_loc,
                        'to_loc': to_loc,
                        'start_time': event_time,
                        'end_time': event_time + travel_time,
                    }
            elif event['event'] == 'arrival':
                if amb_id in active_transports:
                    del active_transports[amb_id]
                amb_positions[amb_id] = positions.get(event['at_loc'], (0, 0))

        # Calculate positions for ambulances in transit
        for amb_id, transport in list(active_transports.items()):
            if t >= transport['end_time']:
                amb_positions[amb_id] = positions.get(transport['to_loc'], (0, 0))
            elif t >= transport['start_time']:
                # Interpolate position
                progress = (t - transport['start_time']) / (transport['end_time'] - transport['start_time'])
                progress = min(1.0, max(0.0, progress))
                x1, y1 = positions.get(transport['from_loc'], (0, 0))
                x2, y2 = positions.get(transport['to_loc'], (0, 0))
                amb_positions[amb_id] = (x1 + progress * (x2 - x1), y1 + progress * (y2 - y1))

        return amb_positions

    def init():
        amb_scatter.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return amb_scatter, time_text

    def animate(frame):
        t = frame * time_scale
        amb_pos = get_ambulance_positions_at_time(t)

        if amb_pos:
            coords = np.array(list(amb_pos.values()))
            amb_scatter.set_offsets(coords)
        else:
            amb_scatter.set_offsets(np.empty((0, 2)))

        time_text.set_text(f'Time: {t:.0f} min ({t/60:.1f} hrs)')
        ax.set_title(f'MEDEVAC Simulation - {len(amb_pos)} ambulances active')

        return amb_scatter, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=interval_ms, blit=False)

    if save_path:
        anim.save(save_path, writer='pillow', fps=1000//interval_ms)
        print(f"Animation saved to {save_path}")

    plt.tight_layout()
    return fig, anim


def run_simulation_with_logging(
    data_folder: str = "data",
    levers: dict = None,
    till: float = 1000.0,
):
    """
    Run simulation and collect transport events for visualization.

    Returns:
        env: simulation environment
        transport_df: DataFrame of transport events
        distance_matrix: the distance matrix
        facility_df: facility information
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
    env.transport_events = []  # Add transport logging

    env.levers = levers
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    # Monkey-patch Ambulance._travel_to to log events
    original_travel_to = Ambulance._travel_to

    def patched_travel_to(self, destination):
        from_id = self._loc_id(self.location)
        to_id = self._loc_id(destination)
        if from_id != to_id:
            self.env.transport_events.append({
                'time': float(self.env.now()),
                'event': 'departure',
                'amb_id': self.amb_id,
                'from_loc': from_id,
                'to_loc': to_id,
                'patients': len(self.onboard),
            })

        original_travel_to(self, destination)

        if from_id != to_id:
            self.env.transport_events.append({
                'time': float(self.env.now()),
                'event': 'arrival',
                'amb_id': self.amb_id,
                'at_loc': to_id,
                'from_loc': from_id,
                'to_loc': to_id,
                'patients': len(self.onboard),
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
        # Restore original method
        Ambulance._travel_to = original_travel_to

    transport_df = pd.DataFrame(env.transport_events)

    return env, transport_df, data["distance_matrix"], data["facility_df"], data["ambulance_df"]


if __name__ == "__main__":
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

    print("\nRunning simulation with transport logging...")
    env, transport_df, dist_matrix, facility_df, ambulance_df = run_simulation_with_logging(
        data_folder="data",
        till=500.0
    )

    print(f"Logged {len(transport_df)} transport events")

    if len(transport_df) > 0:
        print("\nCreating transport animation...")
        fig, anim = create_transport_animation(
            dist_matrix,
            facility_df,
            transport_df,
            interval_ms=100,
            time_scale=5.0,
            save_path="transport_animation.gif"
        )
        print("Saved: transport_animation.gif")

    plt.show()
