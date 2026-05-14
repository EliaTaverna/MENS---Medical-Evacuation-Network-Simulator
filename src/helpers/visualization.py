"""Visualization module for MEDEVAC simulation."""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import imageio_ffmpeg
import matplotlib
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
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
    'T1_BCAS': '#ff0000',
    'T2_BCAS': '#ffcc00',
    'T3_BCAS': '#00cc00',
    'T1_DNBI': '#ff6666',
    'T2_DNBI': '#ffdd66',
    'T3_DNBI': '#66dd66',
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def compute_positions_from_travel_time_matrix(travel_time_matrix: pd.DataFrame) -> dict:
    """
    Use MDS to infer 2D positions from the distance matrix.

    Returns:
        dict mapping location_id -> (x, y)
    """
    locations = list(travel_time_matrix.index)
    dist_array = travel_time_matrix.values.astype(float)

    # Replace any inf/nan with large values
    dist_array = np.nan_to_num(dist_array, nan=1000.0, posinf=1000.0, neginf=0.0)

    # Make symmetric (take average of matrix and its transpose)
    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)

    # MDS to get 2D coordinates
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto', n_init=4)
    coords = mds.fit_transform(dist_array)

    return {loc: (coords[i, 0], coords[i, 1]) for i, loc in enumerate(locations)}


def compute_positions_hierarchical(
    travel_time_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    coord_df: pd.DataFrame = None,
) -> dict:
    """
    Return 2D positions for each facility.

    If coord_df is provided (columns: facility_id, x/x_km, y/y_km — covers all nodes
    including AXPs), use it directly.  Otherwise fall back to facility_df x_km/y_km,
    and finally to MDS on the distance matrix.
    """
    locations = list(travel_time_matrix.index)

    # --- fast path: full coordinate reference available (covers AXPs too) ---
    if coord_df is not None and "facility_id" in coord_df.columns:
        x_col = "x_km" if "x_km" in coord_df.columns else "x"
        y_col = "y_km" if "y_km" in coord_df.columns else "y"
        coord_map = {str(r["facility_id"]): (float(r[x_col]), float(r[y_col]))
                     for _, r in coord_df.iterrows()}
        return {loc: coord_map.get(loc, (0.0, 0.0)) for loc in locations}

    # --- fallback: facility_df real coordinates (no AXPs) ---
    if facility_df is not None and "x_km" in facility_df.columns and "y_km" in facility_df.columns:
        coord_map = {str(r["facility_id"]): (float(r["x_km"]), float(r["y_km"]))
                     for _, r in facility_df.iterrows()}
        return {loc: coord_map.get(loc, (0.0, 0.0)) for loc in locations}

    # --- fallback: MDS ---
    dist_array = travel_time_matrix.values.astype(float)
    dist_array = np.nan_to_num(dist_array, nan=1000.0, posinf=1000.0, neginf=0.0)
    dist_array = (dist_array + dist_array.T) / 2
    np.fill_diagonal(dist_array, 0)

    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42,
              normalized_stress='auto', n_init=4)
    coords = mds.fit_transform(dist_array)
    raw = {loc: coords[i] for i, loc in enumerate(locations)}

    fac_type_map = {}
    if facility_df is not None:
        for _, row in facility_df.iterrows():
            fac_type_map[str(row['facility_id'])] = str(row['facility_type'])
    for loc in locations:
        if loc not in fac_type_map:
            fac_type_map[loc] = 'AXP' if str(loc).upper().startswith('AXP') else 'Unknown'

    ccp_pts = np.array([raw[l] for l in locations if fac_type_map.get(l) == 'CCP'])
    r2_pts  = np.array([raw[l] for l in locations if fac_type_map.get(l) in ('Role2', 'Role3')])

    if len(ccp_pts) > 0 and len(r2_pts) > 0:
        axis = r2_pts.mean(axis=0) - ccp_pts.mean(axis=0)
        angle = np.arctan2(axis[1], axis[0]) + np.pi
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    else:
        R = np.eye(2)

    return {loc: tuple(R @ raw[loc]) for loc in locations}


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

        def triage_key(tr):
            for k in ('T1', 'T2', 'T3'):
                if k in tr:
                    return k
            return None

        for fac_id, fac in self.env.facility_by_id.items():
            # Transport requests are facility-level (shared across all units)
            transport_requests = {'T1': 0, 'T2': 0, 'T3': 0}
            for req in fac.exit_queue:
                k = triage_key(req.patient.triage)
                if k:
                    transport_requests[k] += 1

            # One record per unit (keyed by service_type)
            for svc_type, unit in fac.units.items():
                if unit is None:
                    continue

                # Waiting: only patients queued for this service type
                waiting = {'T1': 0, 'T2': 0, 'T3': 0}
                for p in fac.entry_lines.get(svc_type, []):
                    k = triage_key(p.triage)
                    if k:
                        waiting[k] += 1

                # In care: patients actively being served by this unit
                in_care = {'T1': 0, 'T2': 0, 'T3': 0}
                for svc in unit._active_services:
                    k = triage_key(svc.patient.triage)
                    if k:
                        in_care[k] += 1

                self.env.facility_states.append({
                    'time': t,
                    'facility_id': fac_id,
                    'service_type': svc_type,
                    'T1_waiting':   waiting['T1'],
                    'T2_waiting':   waiting['T2'],
                    'T3_waiting':   waiting['T3'],
                    'T1_in_care':   in_care['T1'],
                    'T2_in_care':   in_care['T2'],
                    'T3_in_care':   in_care['T3'],
                    'T1_transport': transport_requests['T1'],
                    'T2_transport': transport_requests['T2'],
                    'T3_transport': transport_requests['T3'],
                })


# ============================================================
# STATIC NETWORK PLOT
# ============================================================

def plot_network_static(
    travel_time_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    ambulance_df: pd.DataFrame = None,
    coord_df: pd.DataFrame = None,
    title: str = "MEDEVAC Network",
    figsize: tuple = (22, 12),
):
    """Create a static plot of the MEDEVAC network."""
    positions = compute_positions_hierarchical(travel_time_matrix, facility_df, coord_df=coord_df)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges (connections). Label one representative per unique time value
    # so the plot shows every travel-time class exactly once, instead of piling
    # overlapping labels on top of each other at edge midpoints.
    locations = list(travel_time_matrix.index)
    labeled_vals = set()
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i >= j:
                continue
            tt = travel_time_matrix.loc[loc1, loc2]
            try:
                t_val = float(tt)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(t_val) or t_val <= 0:
                continue
            x1, y1 = positions[loc1]
            x2, y2 = positions[loc2]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.35, linewidth=0.6, zorder=1)
            key = round(t_val, 6)
            if key not in labeled_vals:
                labeled_vals.add(key)
                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                ax.annotate(
                    f"{t_val:g} min",
                    (mx, my),
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#555555", lw=0.6, alpha=0.95),
                    zorder=2,
                )

    # Draw facilities
    for loc_id, (x, y) in positions.items():
        fac_type = get_facility_type(loc_id, facility_df)
        color = FACILITY_COLORS.get(fac_type, FACILITY_COLORS['Unknown'])
        marker = FACILITY_MARKERS.get(fac_type, 'o')
        size = 250 if fac_type in ['Role2', 'Role3'] else 180

        ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='black', linewidths=1, zorder=5)
        ax.annotate(loc_id, (x, y), xytext=(10, 6), textcoords='offset points', fontsize=8, fontweight='bold')

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
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax, positions


# ============================================================
# VIDEO CREATION
# ============================================================

def create_transport_video(
    travel_time_matrix: pd.DataFrame,
    facility_df: pd.DataFrame,
    transport_events: pd.DataFrame,
    ambulance_df: pd.DataFrame,
    facility_states_df: pd.DataFrame = None,
    coord_df: pd.DataFrame = None,
    fps: int = 30,
    time_scale: float = 5.0,
    figsize: tuple = (32, 18),
    save_path: str = "transport_simulation.mp4",
    dpi: int = 120,
    duration: float = None,
    start_time: float = 0.0,
):
    """
    Create an MP4 video showing ambulances moving with patient dots.

    Parameters:
        travel_time_matrix: DataFrame with distances between locations
        facility_df: DataFrame with facility information
        transport_events: DataFrame with transport events
        ambulance_df: DataFrame with ambulance info
        facility_states_df: DataFrame with facility patient counts over time
        fps: Frames per second for the video
        time_scale: Simulation minutes per frame
        figsize: Figure size
        save_path: Path to save the MP4 file
        dpi: Resolution of the video
        duration: Video duration in minutes from start_time (None = until last event)
        start_time: Simulation time (minutes) at which the video begins (default 0)
    """
    positions = compute_positions_hierarchical(travel_time_matrix, facility_df, coord_df=coord_df)
    amb_colors = generate_ambulance_colors(ambulance_df['amb_id'].tolist())

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#f5f5f5')

    # Draw edges (connections)
    locations = list(travel_time_matrix.index)
    for i, loc1 in enumerate(locations):
        for j, loc2 in enumerate(locations):
            if i < j:
                dist = travel_time_matrix.loc[loc1, loc2]
                if dist < 30:
                    x1, y1 = positions[loc1]
                    x2, y2 = positions[loc2]
                    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.15, linewidth=0.5)

    # Draw facilities
    for loc_id, (x, y) in positions.items():
        fac_type = get_facility_type(loc_id, facility_df)
        color = FACILITY_COLORS.get(fac_type, FACILITY_COLORS['Unknown'])
        marker = FACILITY_MARKERS.get(fac_type, 'o')
        size = 280 if fac_type in ['Role2', 'Role3'] else 200
        ax.scatter(x, y, c=color, marker=marker, s=size, edgecolors='black', linewidths=1.5, zorder=5)
        ax.annotate(loc_id, (x, y), xytext=(10, 6), textcoords='offset points', fontsize=11, fontweight='bold')

    ax.set_xlabel('X (km)', fontsize=13)
    ax.set_ylabel('Y (km)', fontsize=13)
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)

    # Time and stats display
    time_text = ax.text(0.01, 0.99, '', transform=ax.transAxes, fontsize=16,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='#888888', alpha=0.95))

    # Initialize ambulance states
    amb_initial_state = {}
    for _, row in ambulance_df.iterrows():
        amb_initial_state[row['amb_id']] = {
            'location': row['start_loc_id'],
            'in_transit': False,
            'patients': [],
        }

    events_sorted = transport_events.sort_values('time').to_dict('records') if len(transport_events) > 0 else []

    # Determine end time
    if duration is not None:
        max_time = start_time + duration
    elif facility_states_df is not None and len(facility_states_df) > 0:
        max_time = facility_states_df['time'].max()
    elif len(transport_events) > 0:
        max_time = transport_events['time'].max()
    else:
        max_time = start_time + 100

    # Frames run from start_time to max_time
    n_frames = int((max_time - start_time) / time_scale) + 1

    # Build facility states lookup: {time: {fac_id: {svc_type: state_dict}}}
    facility_states_by_time = {}
    if facility_states_df is not None and len(facility_states_df) > 0:
        for _, row in facility_states_df.iterrows():
            t      = row['time']
            fac_id = row['facility_id']
            svc    = row.get('service_type', 'UNK')
            facility_states_by_time.setdefault(t, {}).setdefault(fac_id, {})[svc] = {
                'T1_waiting':   row.get('T1_waiting',   0),
                'T2_waiting':   row.get('T2_waiting',   0),
                'T3_waiting':   row.get('T3_waiting',   0),
                'T1_in_care':   row.get('T1_in_care',   0),
                'T2_in_care':   row.get('T2_in_care',   0),
                'T3_in_care':   row.get('T3_in_care',   0),
                'T1_transport': row.get('T1_transport', 0),
                'T2_transport': row.get('T2_transport', 0),
                'T3_transport': row.get('T3_transport', 0),
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

                travel_time = float(travel_time_matrix.loc[from_loc, to_loc])

                amb_state[amb_id] = {
                    'in_transit': True,
                    'from_loc': from_loc,
                    'to_loc': to_loc,
                    'start_time': event['time'],
                    'end_time': event['time'] + travel_time,
                    'patients': patient_triages,
                }

            elif event['event'] == 'arrival':
                at_loc = event.get('at_loc') or event['to_loc']
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
                    # Don't show patients when idle - they're shown at the facility instead
                    amb_data.append((amb_id, x, y, [], False))

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

        t = start_time + frame * time_scale
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

            # Moving triage label — always shown, shows 0s when empty
            t1_c = sum(1 for tr in patients if 'T1' in tr)
            t2_c = sum(1 for tr in patients if 'T2' in tr)
            t3_c = sum(1 for tr in patients if 'T3' in tr)
            amb_lbl = ax.annotate(
                f"{t1_c}/{t2_c}/{t3_c}",
                (x, y),
                xytext=(0, -10), textcoords='offset points',
                fontsize=7, ha='center', va='top',
                color='black', zorder=12,
            )
            ambulance_artists.append(amb_lbl)


        # Draw facility patient counts and patient dots
        for fac_id in facility_df['facility_id'].tolist():
            if fac_id not in positions:
                continue
            x, y = positions[fac_id]
            unit_states = get_facility_state_at_time(t, fac_id)  # {svc_type: state_dict} or None

            # One label per unit, spread horizontally around facility position
            if unit_states:
                svc_types = list(unit_states.keys())
                n_units   = len(svc_types)
                dx        = 2.2   # km between unit label centres
                for i, svc_type in enumerate(svc_types):
                    s    = unit_states[svc_type]
                    x_lbl = x + (i - (n_units - 1) / 2) * dx
                    text = (f"{svc_type}\n"
                            f"W:{s['T1_waiting']}/{s['T2_waiting']}/{s['T3_waiting']}\n"
                            f"C:{s['T1_in_care']}/{s['T2_in_care']}/{s['T3_in_care']}\n"
                            f"TR:{s['T1_transport']}/{s['T2_transport']}/{s['T3_transport']}")
                    txt = ax.annotate(
                        text, (x_lbl, y),
                        xytext=(0, 22), textcoords='offset points',
                        fontsize=8, ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                                  edgecolor='#aaaaaa', alpha=0.92),
                        zorder=15,
                    )
                    facility_text_artists.append(txt)


        hours = int(t // 60)
        mins = int(t % 60)
        time_text.set_text(f'Time: {hours:02d}:{mins:02d} ({t:.0f} min)')
        ax.set_title(f'MEDEVAC Simulation', fontsize=18, fontweight='bold')

        return ambulance_artists + patient_artists + facility_text_artists + [time_text]

    print(f"Creating animation with {n_frames} frames "
          f"(sim {start_time:.0f}→{max_time:.0f} min, time_scale={time_scale}, fps={fps}, "
          f"→ {n_frames/fps:.1f} s of video)...")
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames, interval=1000//fps, blit=False)

    print(f"Saving to {save_path}...")
    writer = FFMpegWriter(fps=fps, metadata={'title': 'MEDEVAC Simulation'}, bitrate=5000)

    # Per-frame progress — rendering 1000+ frames at dpi=100 with a large
    # figure takes many minutes, and without output the run looks hung.
    # Report every ~1% (or every 10 frames, whichever is more often) and show
    # elapsed + ETA so the user can decide whether to wait.
    import time as _time
    _report_every = max(1, min(10, n_frames // 100 or 1))
    _t0 = _time.time()
    def _progress(current_frame, total_frames):
        done = current_frame + 1
        if done == 1 or done == total_frames or done % _report_every == 0:
            elapsed = _time.time() - _t0
            pct     = 100.0 * done / total_frames
            rate    = done / elapsed if elapsed > 0 else 0.0
            eta     = (total_frames - done) / rate if rate > 0 else float("inf")
            print(f"  frame {done:>5d}/{total_frames}  ({pct:5.1f}%)  "
                  f"elapsed {elapsed:6.1f}s  "
                  f"rate {rate:4.1f} fps  "
                  f"ETA {eta:6.0f}s", flush=True)

    anim.save(save_path, writer=writer, dpi=dpi, progress_callback=_progress)
    print(f"Video saved to {save_path}")

    plt.close(fig)
    return save_path
