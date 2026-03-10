"""Visualization module for MEDEVAC simulation."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.manifold import MDS
import salabim as sim


# ============================================================
# CONSTANTS
# ============================================================

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


# ============================================================
# HELPER FUNCTIONS
# ============================================================

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


def generate_ambulance_colors(ambulance_ids: list) -> dict:
    """Generate unique colors for each ambulance using a colormap."""
    import matplotlib.cm as cm

    n = len(ambulance_ids)
    cmap = cm.get_cmap('tab20', n) if n <= 20 else cm.get_cmap('hsv', n)

    colors = {}
    for i, amb_id in enumerate(sorted(ambulance_ids)):
        colors[amb_id] = cmap(i / max(n - 1, 1))[:3]

    return colors


# ============================================================
# FACILITY STATE LOGGER (salabim Component)
# ============================================================

class FacilityStateLogger(sim.Component):
    """Logs facility states at regular intervals during simulation."""

    def setup(self, interval: float = 1.0):
        self.interval = interval
        self.env.facility_states = []

    def process(self):
        while True:
            self._log_state()
            self.hold(self.interval)

    def _log_state(self):
        t = float(self.env.now())

        for fac_id, fac in self.env.facility_by_id.items():
            # Count patients waiting in queue by triage
            waiting = {'T1': 0, 'T2': 0, 'T3': 0}
            for service_type, queue in fac.entry_lines.items():
                for p in queue:
                    if p.triage in waiting:
                        waiting[p.triage] += 1

            # Count patients waiting for transport by triage
            for req in fac.exit_queue:
                if req.patient.triage in waiting:
                    waiting[req.patient.triage] += 1

            # Count total patients in care
            total_in_care = sum(unit.in_service for unit in fac.units.values() if unit is not None)

            self.env.facility_states.append({
                'time': t,
                'facility_id': fac_id,
                'T1_waiting': waiting['T1'],
                'T2_waiting': waiting['T2'],
                'T3_waiting': waiting['T3'],
                'total_in_care': total_in_care,
            })


# ============================================================
# STATIC NETWORK PLOT
# ============================================================

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


# ============================================================
# VIDEO CREATION
# ============================================================

def create_transport_video(
    distance_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    transport_events: pd.DataFrame,
    ambulance_df: pd.DataFrame,
    facility_states_df: pd.DataFrame = None,
    fps: int = 30,
    time_scale: float = 5.0,
    figsize: tuple = (16, 10),
    save_path: str = "transport_simulation.mp4",
    dpi: int = 150,
):
    """
    Create an MP4 video showing ambulances moving with patient dots.

    Parameters:
        distance_matrix: DataFrame with distances between locations
        facility_df: DataFrame with facility information
        transport_events: DataFrame with transport events
        ambulance_df: DataFrame with ambulance info
        facility_states_df: DataFrame with facility patient counts over time
        fps: Frames per second for the video
        time_scale: Simulation minutes per frame
        figsize: Figure size
        save_path: Path to save the MP4 file
        dpi: Resolution of the video
    """
    positions = compute_positions_from_distance_matrix(distance_matrix)
    amb_colors = generate_ambulance_colors(ambulance_df['amb_id'].tolist())

    fig, ax = plt.subplots(figsize=figsize)
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

    # Create legend
    legend_handles = []
    for fac_type, color in FACILITY_COLORS.items():
        if fac_type != 'Unknown':
            marker = FACILITY_MARKERS[fac_type]
            handle = plt.scatter([], [], c=color, marker=marker, s=100,
                                edgecolors='black', linewidths=1, label=fac_type)
            legend_handles.append(handle)

    for triage, color in TRIAGE_COLORS.items():
        handle = plt.scatter([], [], c=color, marker='o', s=60, edgecolors='black', linewidths=0.5, label=f'Patient {triage}')
        legend_handles.append(handle)

    handle = plt.scatter([], [], c='white', marker='s', s=100, edgecolors='#333333', linewidths=2, label='Ambulance')
    legend_handles.append(handle)

    ax.legend(handles=legend_handles, loc='upper left', title='Legend', fontsize=8)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Time and stats display
    time_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, fontsize=14,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    stats_text = ax.text(0.98, 0.88, '', transform=ax.transAxes, fontsize=10,
                         verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    # Initialize ambulance states
    amb_initial_state = {}
    for _, row in ambulance_df.iterrows():
        amb_initial_state[row['amb_id']] = {
            'location': row['start_loc_id'],
            'in_transit': False,
            'patients': [],
        }

    events_sorted = transport_events.sort_values('time').to_dict('records') if len(transport_events) > 0 else []
    max_time = transport_events['time'].max() if len(transport_events) > 0 else 100
    n_frames = int(max_time / time_scale) + 1

    # Build facility states lookup
    facility_states_by_time = {}
    if facility_states_df is not None and len(facility_states_df) > 0:
        for _, row in facility_states_df.iterrows():
            t = row['time']
            fac_id = row['facility_id']
            if t not in facility_states_by_time:
                facility_states_by_time[t] = {}
            facility_states_by_time[t][fac_id] = {
                'T1_waiting': row['T1_waiting'],
                'T2_waiting': row['T2_waiting'],
                'T3_waiting': row['T3_waiting'],
                'total_in_care': row['total_in_care'],
            }

    def get_facility_state_at_time(t, fac_id):
        if not facility_states_by_time:
            return None
        times = sorted(facility_states_by_time.keys())
        selected_t = None
        for time_key in times:
            if time_key <= t:
                selected_t = time_key
            else:
                break
        if selected_t is None:
            return None
        return facility_states_by_time.get(selected_t, {}).get(fac_id)

    def get_state_at_time(t):
        amb_state = {amb_id: state.copy() for amb_id, state in amb_initial_state.items()}

        for event in events_sorted:
            if event['time'] > t:
                break

            amb_id = event['amb_id']

            if event['event'] == 'departure':
                from_loc = event['from_loc']
                to_loc = event['to_loc']
                patient_triages = event.get('patient_triages', [])

                dist = distance_matrix.loc[from_loc, to_loc]
                travel_time = (dist / 30.0) * 60.0

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
        state = get_state_at_time(t)
        amb_data = []

        for amb_id, amb_state in state.items():
            if amb_state.get('in_transit', False):
                start_t = amb_state['start_time']
                end_t = amb_state['end_time']

                if t >= end_t:
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
                amb_data.append((amb_id, x, y, patients, True))
            else:
                loc = amb_state.get('location')
                if loc and loc in positions:
                    x, y = positions[loc]
                    patients = amb_state.get('patients', [])
                    amb_data.append((amb_id, x, y, patients, False))

        return amb_data

    ambulance_artists = []
    patient_artists = []
    facility_text_artists = []

    def init():
        return []

    def animate(frame):
        nonlocal ambulance_artists, patient_artists, facility_text_artists

        for artist in ambulance_artists + patient_artists + facility_text_artists:
            artist.remove()
        ambulance_artists = []
        patient_artists = []
        facility_text_artists = []

        t = frame * time_scale
        amb_data = get_ambulance_visual_data(t)

        n_in_transit = 0
        n_patients_moving = 0

        for amb_id, x, y, patients, in_transit in amb_data:
            if in_transit:
                n_in_transit += 1
                n_patients_moving += len(patients)

            base_color = amb_colors.get(amb_id, (0.5, 0.5, 0.5))
            if in_transit:
                color = base_color
                edge_width = 2
            else:
                color = tuple(c * 0.5 + 0.3 for c in base_color)
                edge_width = 1

            amb_artist = ax.scatter(x, y, c=[color], marker='s', s=120,
                                   edgecolors='black', linewidths=edge_width, zorder=10)
            ambulance_artists.append(amb_artist)

            if patients:
                n_patients = len(patients)
                for i, triage in enumerate(patients):
                    angle = (2 * np.pi * i / max(n_patients, 1)) + np.pi/4
                    offset = 1.5
                    px = x + offset * np.cos(angle)
                    py = y + offset * np.sin(angle)

                    color = TRIAGE_COLORS.get(triage, '#888888')
                    patient_artist = ax.scatter(px, py, c=color, marker='o', s=50,
                                               edgecolors='black', linewidths=0.5, zorder=11)
                    patient_artists.append(patient_artist)

        # Draw facility patient counts
        for fac_id in facility_df['facility_id'].tolist():
            if fac_id not in positions:
                continue
            x, y = positions[fac_id]
            state = get_facility_state_at_time(t, fac_id)
            if state:
                t1w = state['T1_waiting']
                t2w = state['T2_waiting']
                t3w = state['T3_waiting']
                total_c = state['total_in_care']
                total_w = t1w + t2w + t3w
                if total_w > 0 or total_c > 0:
                    text = f"W:{t1w}/{t2w}/{t3w}\nC:{total_c}"
                    txt = ax.text(x, y + 2.5, text, fontsize=6, ha='center', va='bottom',
                                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                          edgecolor='gray', alpha=0.85),
                                 zorder=15)
                    facility_text_artists.append(txt)

        hours = int(t // 60)
        mins = int(t % 60)
        time_text.set_text(f'Time: {hours:02d}:{mins:02d} ({t:.0f} min)')
        stats_text.set_text(f'Ambulances moving: {n_in_transit}\nPatients in transit: {n_patients_moving}')
        ax.set_title(f'MEDEVAC Simulation', fontsize=14, fontweight='bold')

        return ambulance_artists + patient_artists + facility_text_artists + [time_text, stats_text]

    print(f"Creating animation with {n_frames} frames...")
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000//fps, blit=False)

    print(f"Saving to {save_path}...")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'MEDEVAC Simulation'}, bitrate=5000)
    anim.save(save_path, writer=writer, dpi=dpi)
    print(f"Video saved to {save_path}")

    plt.close(fig)
    return save_path
