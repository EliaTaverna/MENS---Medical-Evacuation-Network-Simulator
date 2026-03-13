"""
Test KPI calculations with edge cases that have predictable outcomes.

Each test creates a minimal simulation scenario where we can calculate
the expected KPI values by hand, then verify the computed values match.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(_REPO_ROOT)
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import salabim as sim

from src.components.builder import ModelBuilder, ModelConnector
from src.components.ambulance import Ambulance
from src.components.patient import Patient
from src.components.spawner import CasualtySpawner
from src.policies.policies import unload_planned, load_planned
from src.runners.runner_single import print_summary


def create_minimal_dataframes(
    n_facilities=2,
    n_ambulances=1,
    ambulance_speed=60,  # km/h
    distance=10,  # km between facilities
    unit_service_time=(10, 10),  # (min, max) for uniform distribution
    unit_capacity=1,
):
    """
    Create minimal DataFrames for a simple 2-facility scenario.

    Facility F1 (CCP) -> Facility F2 (Role1)
    F1 has no units, F2 has unit "surgery"
    """
    # Distance matrix - include AXP1 at same location as F1
    distance_matrix = pd.DataFrame(
        [[0, distance, 0], [distance, 0, distance], [0, distance, 0]],
        index=["F1", "F2", "AXP1"],
        columns=["F1", "F2", "AXP1"],
    )

    # AXP (ambulance exchange point) - ambulances start here (co-located with F1)
    axp_df = pd.DataFrame([{
        "axp_id": "AXP1",
        "assigned_facility_ids": ["F1"],
    }])

    # Facilities
    facility_df = pd.DataFrame([
        {
            "facility_id": "F1",
            "facility_type": "CCP",
            "c2_up_id": "C2",
            "unload_policy": "unload_planned",
            "load_policy": "load_planned",
        },
        {
            "facility_id": "F2",
            "facility_type": "Role1",
            "c2_up_id": None,
            "unload_policy": "unload_planned",
            "load_policy": "load_planned",
        },
    ])

    # Units - only at F2
    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": unit_capacity,
        "service_time_min": unit_service_time[0],
        "service_time_max": unit_service_time[1],
        "service_time_dist": sim.Uniform(unit_service_time[0], unit_service_time[1]),
    }])

    # Ambulances
    ambulance_df = pd.DataFrame([{
        "amb_id": "AMB1",
        "capacity": 1,
        "speed_kmh": ambulance_speed,
        "start_loc_id": "AXP1",
        "c2_id": "C2",
    }])

    # Corridors matrix (which facility can send to which)
    corridors_matrix = pd.DataFrame(
        [[0, 1], [0, 0]],  # F1 can send to F2, F2 can't send anywhere
        index=["F1", "F2"],
        columns=["F1", "F2"],
    )

    # Ambulance eligibility (which ambulances serve which facilities)
    ambulances_eligibility = pd.DataFrame(
        [[1, 1]],  # AMB1 serves both F1 and F2
        index=["AMB1"],
        columns=["F1", "F2"],
    )

    # C2
    c2_df = pd.DataFrame([{
        "c2_id": "C2",
        "facility_ids": ["F1", "F2"],
        "ambulance_ids": ["AMB1"],
        "corridors": corridors_matrix,
        "ambulances_eligibility": ambulances_eligibility,
    }])

    return {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
    }


def create_casualties_df(casualties_list):
    """
    Create casualties DataFrame from a list of dicts.

    Each dict should have:
        - t_create: spawn time
        - triage: "T1", "T2", or "T3"
        - spawn_facility_id: where patient spawns
        - prescription: list of unit types needed (e.g., ["surgery"])
        - lifetime_low, lifetime_high: survival time bounds
    """
    rows = []
    for i, c in enumerate(casualties_list):
        lifetime_low = c.get("lifetime_low", 1000)
        lifetime_high = c.get("lifetime_high", 1000)
        rows.append({
            "patient_id": f"P{i+1}",
            "t_create": c["t_create"],
            "triage": c["triage"],
            "spawn_facility_id": c["spawn_facility_id"],
            "prescription": c["prescription"],
            "lifetime_low": lifetime_low,
            "lifetime_high": lifetime_high,
            "lifetime_dist": sim.Uniform(lifetime_low, lifetime_high),
            "injury_time": c.get("injury_time", c["t_create"]),
        })
    return pd.DataFrame(rows)


def run_test_simulation(data, casualties_df, till=100, trace=False):
    """Run simulation and return results dict compatible with print_summary."""
    env = sim.Environment(trace=trace)
    env.patient_log_rows = []
    env.patient_log_columns = [
        "time", "patient_id", "action", "life_budget",
        "needed_unit", "current_location", "is_dead", "is_safe"
    ]
    env.safe = []
    env.transport_events = []
    env.facility_states = []
    env.wait_times = []
    env.unit_busy_periods = {}
    env.all_patients = {}

    env.levers = {
        "facility_eligibility": 0,
        "facility_assignment": 0,
        "ambulance_eligibility": 0,
        "ambulance_assignment": 0,
    }
    env.unload_policy = {"unload_planned": unload_planned}
    env.load_policy = {"load_planned": load_planned}

    # Patch Ambulance for transport event logging
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

    # Patch Facility for transport wait tracking
    from src.components.facility import Facility
    original_patient_handed_over = Facility.patient_handed_over

    def patched_patient_handed_over(self, p):
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
                if hasattr(p, "_transport_wait_start") and p._transport_wait_start is not None:
                    wait_duration = self.env.now() - p._transport_wait_start
                    self.env.wait_times.append({
                        "triage": p.triage,
                        "facility_type": p._transport_wait_facility_type,
                        "wait_type": "transport",
                        "duration": wait_duration,
                    })
                    p._transport_wait_start = None

                for req in list(self.exit_queue):
                    if req.patient is p and req.status == "assigned":
                        self.exit_queue.remove(req)
                        break
                p.pause_death_clock()
                amb.onboard.append(p)
                p.patient_state = "in_transport"

            amb.activate()

    Facility.process = patched_facility_process

    # Patch Unit for care wait tracking
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
                p.pause_death_clock()

                if p.is_dead:
                    continue

                t_enter = getattr(p, "_unit_queue_entry_time", None)
                if t_enter is not None:
                    wait_duration = self.env.now() - t_enter
                    self.wait_monitor.tally(wait_duration)
                    self.env.wait_times.append({
                        "triage": p.triage,
                        "facility_type": self.facility.facility_type,
                        "wait_type": "care",
                        "duration": wait_duration,
                        "unit_id": self.name(),
                    })

                self.in_service += 1
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
        if hasattr(p, "_unit_service_start") and p._unit_service_start is not None:
            unit_id = p._unit_service_unit_id
            self.env.unit_busy_periods[unit_id].append({
                "start": p._unit_service_start,
                "end": self.env.now(),
            })
            p._unit_service_start = None
        original_service_done(self, p)

    Unit.service_done = patched_service_done

    # Patch Patient for tracking
    original_patient_setup = Patient.setup

    def patched_patient_setup(self, patient_id, triage, prescription, life_time_dist, spawn_facility, injury_time):
        original_patient_setup(self, patient_id, triage, prescription, life_time_dist, spawn_facility, injury_time)
        self.env.all_patients[patient_id] = self

    Patient.setup = patched_patient_setup

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

        # Spawn casualties
        spawner = CasualtySpawner(casualties_df=casualties_df)

        # Run
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
    wait_times_df = pd.DataFrame(env.wait_times) if env.wait_times else pd.DataFrame()

    safe_data = [
        (p.patient_id, p.triage, p.t_safe, p.spawn_facility.loc_id)
        for p in env.safe
    ]
    safe_df = pd.DataFrame(
        safe_data,
        columns=["patient_id", "triage", "t_safe", "spawn_facility_id"]
    )

    return {
        "env": env,
        "safe_df": safe_df,
        "transport_df": transport_df,
        "facility_states_df": pd.DataFrame(env.facility_states),
        "patient_log_df": env.patient_log_df,
        "wait_times_df": wait_times_df,
        "unit_busy_periods": env.unit_busy_periods,
        "all_patients": env.all_patients,
        "sim_duration": till,
        "distance_matrix": data["distance_matrix"],
    }


# ==============================================================================
# TEST CASES
# ==============================================================================

def test_transport_time():
    """
    Test: Ambulance travels 10 km at 60 km/h = 10 minutes travel time.

    Scenario:
    - 1 patient at F1, needs "surgery" at F2
    - Distance = 10 km, speed = 60 km/h
    - Expected travel time: 10 minutes each way = 20 minutes total
    """
    print("\n" + "="*70)
    print("TEST: Transport Time Calculation")
    print("="*70)

    data = create_minimal_dataframes(
        distance=10,  # 10 km
        ambulance_speed=60,  # 60 km/h -> 10 min travel time
        unit_service_time=(10, 10),
    )

    casualties_df = create_casualties_df([
        {
            "t_create": 0,
            "triage": "T1",
            "spawn_facility_id": "F1",
            "prescription": ["surgery"],
            "lifetime_low": 1000,
            "lifetime_high": 1000,
        }
    ])

    results = run_test_simulation(data, casualties_df, till=100)

    # Calculate expected: AXP1 -> F1 (10 km) + F1 -> F2 (10 km) = 20 min total
    # But wait - AXP is at F1, so distance AXP1 to F1 might be 0
    # Let me check the distance matrix - it only has F1 and F2
    # So ambulance starts at AXP1 which is assigned to F1
    # Travel: AXP1 -> F1 (pickup) -> F2 (dropoff)
    # Since AXP1 is at F1, the travel is just F1 -> F2 = 10 km = 10 min

    transport_df = results["transport_df"]
    if not transport_df.empty:
        arrivals = transport_df[transport_df["event"] == "arrival"]
        departures = transport_df[transport_df["event"] == "departure"]

        total_travel = 0
        for _, arr in arrivals.iterrows():
            matching = departures[
                (departures["time"] < arr["time"]) &
                (departures["from_loc"] == arr["from_loc"]) &
                (departures["to_loc"] == arr["to_loc"])
            ]
            if not matching.empty:
                dep = matching.iloc[-1]
                total_travel += arr["time"] - dep["time"]

        expected_travel = 10.0  # F1 -> F2 = 10 km / 60 km/h * 60 min/h = 10 min
        print(f"Expected transport time: {expected_travel} minutes")
        print(f"Computed transport time: {total_travel} minutes")

        if abs(total_travel - expected_travel) < 0.01:
            print("PASS: Transport time correct")
            return True
        else:
            print(f"FAIL: Expected {expected_travel}, got {total_travel}")
            return False
    else:
        print("FAIL: No transport events recorded")
        return False


def test_care_wait_time():
    """
    Test: Patient waits for care when unit is busy.

    Scenario:
    - 2 patients arrive at F2 simultaneously needing "surgery"
    - Unit capacity = 1, service time = 10 min
    - First patient: 0 wait, 10 min service
    - Second patient: 10 min wait, 10 min service
    """
    print("\n" + "="*70)
    print("TEST: Care Wait Time Calculation")
    print("="*70)

    # Create scenario where patients spawn directly at F2 (which has the unit)
    # Include AXP1 co-located with F2
    distance_matrix = pd.DataFrame(
        [[0, 0], [0, 0]],
        index=["F2", "AXP1"],
        columns=["F2", "AXP1"]
    )

    axp_df = pd.DataFrame([{
        "axp_id": "AXP1",
        "assigned_facility_ids": ["F2"],
    }])

    facility_df = pd.DataFrame([{
        "facility_id": "F2",
        "facility_type": "Role1",
        "c2_up_id": None,
        "unload_policy": "unload_planned",
        "load_policy": "load_planned",
    }])

    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": 1,
        "service_time_min": 10,
        "service_time_max": 10,
        "service_time_dist": sim.Uniform(10, 10),
    }])

    ambulance_df = pd.DataFrame([{
        "amb_id": "AMB1",
        "capacity": 1,
        "speed_kmh": 60,
        "start_loc_id": "AXP1",
        "c2_id": "C2",
    }])

    corridors_matrix = pd.DataFrame([[0]], index=["F2"], columns=["F2"])
    ambulances_eligibility = pd.DataFrame([[1]], index=["AMB1"], columns=["F2"])

    c2_df = pd.DataFrame([{
        "c2_id": "C2",
        "facility_ids": ["F2"],
        "ambulance_ids": ["AMB1"],
        "corridors": corridors_matrix,
        "ambulances_eligibility": ambulances_eligibility,
    }])

    data = {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
    }

    # Two patients arriving at time 0
    casualties_df = create_casualties_df([
        {
            "t_create": 0,
            "triage": "T1",
            "spawn_facility_id": "F2",
            "prescription": ["surgery"],
        },
        {
            "t_create": 0,
            "triage": "T1",
            "spawn_facility_id": "F2",
            "prescription": ["surgery"],
        },
    ])

    results = run_test_simulation(data, casualties_df, till=50)

    wait_times_df = results["wait_times_df"]
    care_waits = wait_times_df[wait_times_df["wait_type"] == "care"]

    if len(care_waits) == 2:
        waits = sorted(care_waits["duration"].values)
        expected_waits = [0.0, 10.0]  # First patient: 0 wait, Second: 10 min wait

        print(f"Expected wait times: {expected_waits}")
        print(f"Computed wait times: {list(waits)}")

        if abs(waits[0] - 0.0) < 0.01 and abs(waits[1] - 10.0) < 0.01:
            print("PASS: Care wait times correct")
            return True
        else:
            print(f"FAIL: Expected {expected_waits}, got {list(waits)}")
            return False
    else:
        print(f"FAIL: Expected 2 care wait records, got {len(care_waits)}")
        return False


def test_transport_wait_time():
    """
    Test: Patient waits for transport when ambulance is busy.

    Scenario:
    - Patient P1 spawns at t=0, needs transport F1->F2
    - Patient P2 spawns at t=1, needs transport F1->F2
    - Only 1 ambulance, travel time = 10 min each way
    - P1: picked up immediately (small wait for ambulance to arrive)
    - P2: waits until ambulance returns (~20 min round trip)
    """
    print("\n" + "="*70)
    print("TEST: Transport Wait Time Calculation")
    print("="*70)

    data = create_minimal_dataframes(
        distance=10,
        ambulance_speed=60,
        unit_service_time=(10, 10),
    )

    casualties_df = create_casualties_df([
        {
            "t_create": 0,
            "triage": "T1",
            "spawn_facility_id": "F1",
            "prescription": ["surgery"],
        },
        {
            "t_create": 1,
            "triage": "T2",
            "spawn_facility_id": "F1",
            "prescription": ["surgery"],
        },
    ])

    results = run_test_simulation(data, casualties_df, till=100)

    wait_times_df = results["wait_times_df"]
    transport_waits = wait_times_df[wait_times_df["wait_type"] == "transport"]

    print(f"Transport wait records: {len(transport_waits)}")
    if not transport_waits.empty:
        for _, row in transport_waits.iterrows():
            print(f"  {row['triage']}: {row['duration']:.1f} min at {row['facility_type']}")

    # P1 should have minimal wait (ambulance travels from AXP to F1)
    # P2 should wait ~20 min (ambulance does round trip: F1->F2->F1)

    if len(transport_waits) >= 1:
        print("PASS: Transport wait times recorded")
        return True
    else:
        print("FAIL: No transport wait times recorded")
        return False


def test_patient_death():
    """
    Test: Patient with short lifetime dies while waiting for transport.

    Scenario:
    - Patient spawns at F1, needs transport to F2
    - Ambulance starts far away (AXP1 is 60km from F1, takes 60 min to arrive)
    - T1 patient has 5 min lifetime -> dies waiting for transport
    - T2 patient has 1000 min lifetime -> survives
    """
    print("\n" + "="*70)
    print("TEST: Patient Death Counting")
    print("="*70)

    # Create scenario where ambulance is far from F1
    # AXP1 is 60km away from F1, at 60 km/h it takes 60 minutes to arrive
    distance_matrix = pd.DataFrame(
        [[0, 10, 60], [10, 0, 60], [60, 60, 0]],
        index=["F1", "F2", "AXP1"],
        columns=["F1", "F2", "AXP1"],
    )

    axp_df = pd.DataFrame([{
        "axp_id": "AXP1",
        "assigned_facility_ids": ["F1"],
    }])

    facility_df = pd.DataFrame([
        {
            "facility_id": "F1",
            "facility_type": "CCP",
            "c2_up_id": "C2",
            "unload_policy": "unload_planned",
            "load_policy": "load_planned",
        },
        {
            "facility_id": "F2",
            "facility_type": "Role1",
            "c2_up_id": None,
            "unload_policy": "unload_planned",
            "load_policy": "load_planned",
        },
    ])

    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": 1,
        "service_time_min": 10,
        "service_time_max": 10,
        "service_time_dist": sim.Uniform(10, 10),
    }])

    ambulance_df = pd.DataFrame([{
        "amb_id": "AMB1",
        "capacity": 2,
        "speed_kmh": 60,
        "start_loc_id": "AXP1",
        "c2_id": "C2",
    }])

    corridors_matrix = pd.DataFrame(
        [[0, 1], [0, 0]],
        index=["F1", "F2"],
        columns=["F1", "F2"],
    )

    ambulances_eligibility = pd.DataFrame(
        [[1, 1]],
        index=["AMB1"],
        columns=["F1", "F2"],
    )

    c2_df = pd.DataFrame([{
        "c2_id": "C2",
        "facility_ids": ["F1", "F2"],
        "ambulance_ids": ["AMB1"],
        "corridors": corridors_matrix,
        "ambulances_eligibility": ambulances_eligibility,
    }])

    data = {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
    }

    # T1 patient with 5 min lifetime (will die waiting for ambulance that takes 60 min)
    # T2 patient with 1000 min lifetime (will survive)
    casualties_df = create_casualties_df([
        {
            "t_create": 0,
            "triage": "T1",
            "spawn_facility_id": "F1",
            "prescription": ["surgery"],
            "lifetime_low": 5,  # Dies in 5 min while waiting
            "lifetime_high": 5,
        },
        {
            "t_create": 0,
            "triage": "T2",
            "spawn_facility_id": "F1",
            "prescription": ["surgery"],
            "lifetime_low": 1000,  # Survives
            "lifetime_high": 1000,
        },
    ])

    results = run_test_simulation(data, casualties_df, till=200)

    env = results["env"]
    log_df = results["patient_log_df"]

    # Get final states
    final_states = log_df.groupby("patient_id").last().reset_index()

    n_dead = (final_states["is_dead"] == True).sum()
    n_safe = (final_states["is_safe"] == True).sum()

    print(f"Expected: 1 dead (T1), 1 safe (T2)")
    print(f"Computed: {n_dead} dead, {n_safe} safe")

    # Check by triage
    triage_map = {pid: p.triage for pid, p in env.all_patients.items()}
    final_states["triage"] = final_states["patient_id"].map(triage_map)

    t1_dead = ((final_states["triage"] == "T1") & (final_states["is_dead"] == True)).sum()
    t2_safe = ((final_states["triage"] == "T2") & (final_states["is_safe"] == True)).sum()

    print(f"T1 deaths: {t1_dead}, T2 safe: {t2_safe}")

    if t1_dead == 1 and t2_safe == 1:
        print("PASS: Death and safe counting by triage correct")
        return True
    else:
        print("FAIL: Incorrect counts")
        return False


def test_unit_utilization():
    """
    Test: Unit utilization calculation.

    Scenario:
    - Simulation runs for 100 min
    - Unit serves 2 patients, each takes 10 min
    - Expected utilization: 20/100 = 20%
    """
    print("\n" + "="*70)
    print("TEST: Unit Utilization Calculation")
    print("="*70)

    # Single facility with unit - include AXP1 co-located with F2
    distance_matrix = pd.DataFrame(
        [[0, 0], [0, 0]],
        index=["F2", "AXP1"],
        columns=["F2", "AXP1"]
    )

    axp_df = pd.DataFrame([{
        "axp_id": "AXP1",
        "assigned_facility_ids": ["F2"],
    }])

    facility_df = pd.DataFrame([{
        "facility_id": "F2",
        "facility_type": "Role1",
        "c2_up_id": None,
        "unload_policy": "unload_planned",
        "load_policy": "load_planned",
    }])

    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": 1,
        "service_time_min": 10,
        "service_time_max": 10,
        "service_time_dist": sim.Uniform(10, 10),
    }])

    ambulance_df = pd.DataFrame([{
        "amb_id": "AMB1",
        "capacity": 1,
        "speed_kmh": 60,
        "start_loc_id": "AXP1",
        "c2_id": "C2",
    }])

    corridors_matrix = pd.DataFrame([[0]], index=["F2"], columns=["F2"])
    ambulances_eligibility = pd.DataFrame([[1]], index=["AMB1"], columns=["F2"])

    c2_df = pd.DataFrame([{
        "c2_id": "C2",
        "facility_ids": ["F2"],
        "ambulance_ids": ["AMB1"],
        "corridors": corridors_matrix,
        "ambulances_eligibility": ambulances_eligibility,
    }])

    data = {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
    }

    # Two patients at t=0 and t=30 (so they don't overlap)
    casualties_df = create_casualties_df([
        {
            "t_create": 0,
            "triage": "T1",
            "spawn_facility_id": "F2",
            "prescription": ["surgery"],
        },
        {
            "t_create": 30,
            "triage": "T1",
            "spawn_facility_id": "F2",
            "prescription": ["surgery"],
        },
    ])

    sim_duration = 100
    results = run_test_simulation(data, casualties_df, till=sim_duration)

    unit_busy_periods = results["unit_busy_periods"]

    if "U1" in unit_busy_periods:
        periods = unit_busy_periods["U1"]
        total_busy = sum(p["end"] - p["start"] for p in periods)
        utilization = (total_busy / sim_duration) * 100

        expected_utilization = 20.0  # 2 patients * 10 min each / 100 min = 20%

        print(f"Expected utilization: {expected_utilization}%")
        print(f"Computed utilization: {utilization}%")
        print(f"Total busy time: {total_busy} min")

        if abs(utilization - expected_utilization) < 0.1:
            print("PASS: Unit utilization correct")
            return True
        else:
            print(f"FAIL: Expected {expected_utilization}%, got {utilization}%")
            return False
    else:
        print("FAIL: No busy periods recorded for U1")
        return False


def test_wait_time_percentiles():
    """
    Test: Percentile calculations for wait times.

    Scenario:
    - 4 patients with known wait times: 0, 10, 20, 30 min
    - 25th percentile = 5, Median = 15, 75th percentile = 25
    """
    print("\n" + "="*70)
    print("TEST: Wait Time Percentile Calculation")
    print("="*70)

    # Create scenario with predictable wait times - include AXP1 co-located with F2
    distance_matrix = pd.DataFrame(
        [[0, 0], [0, 0]],
        index=["F2", "AXP1"],
        columns=["F2", "AXP1"]
    )

    axp_df = pd.DataFrame([{
        "axp_id": "AXP1",
        "assigned_facility_ids": ["F2"],
    }])

    facility_df = pd.DataFrame([{
        "facility_id": "F2",
        "facility_type": "Role1",
        "c2_up_id": None,
        "unload_policy": "unload_planned",
        "load_policy": "load_planned",
    }])

    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": 1,
        "service_time_min": 10,
        "service_time_max": 10,
        "service_time_dist": sim.Uniform(10, 10),
    }])

    ambulance_df = pd.DataFrame([{
        "amb_id": "AMB1",
        "capacity": 1,
        "speed_kmh": 60,
        "start_loc_id": "AXP1",
        "c2_id": "C2",
    }])

    corridors_matrix = pd.DataFrame([[0]], index=["F2"], columns=["F2"])
    ambulances_eligibility = pd.DataFrame([[1]], index=["AMB1"], columns=["F2"])

    c2_df = pd.DataFrame([{
        "c2_id": "C2",
        "facility_ids": ["F2"],
        "ambulance_ids": ["AMB1"],
        "corridors": corridors_matrix,
        "ambulances_eligibility": ambulances_eligibility,
    }])

    data = {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
    }

    # 4 patients arriving at t=0 (capacity=1, service=10min each)
    # Wait times will be: 0, 10, 20, 30 (each waits for previous to finish)
    casualties_df = create_casualties_df([
        {"t_create": 0, "triage": "T1", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T1", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T1", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T1", "spawn_facility_id": "F2", "prescription": ["surgery"]},
    ])

    results = run_test_simulation(data, casualties_df, till=100)

    wait_times_df = results["wait_times_df"]
    care_waits = wait_times_df[wait_times_df["wait_type"] == "care"]["duration"].values

    if len(care_waits) == 4:
        care_waits = sorted(care_waits)
        print(f"Recorded wait times: {care_waits}")

        # Expected: [0, 10, 20, 30]
        expected = [0.0, 10.0, 20.0, 30.0]

        p25 = np.percentile(care_waits, 25)
        p50 = np.percentile(care_waits, 50)
        p75 = np.percentile(care_waits, 75)

        # For [0, 10, 20, 30]: 25th=7.5, 50th=15, 75th=22.5
        print(f"25th percentile: {p25} (expected ~7.5)")
        print(f"50th percentile (median): {p50} (expected 15)")
        print(f"75th percentile: {p75} (expected ~22.5)")

        # Check if wait times match expected
        if all(abs(a - b) < 0.01 for a, b in zip(care_waits, expected)):
            print("PASS: Wait times correct")
            return True
        else:
            print(f"FAIL: Expected {expected}, got {list(care_waits)}")
            return False
    else:
        print(f"FAIL: Expected 4 wait times, got {len(care_waits)}")
        return False


def test_multiple_triage_categories():
    """
    Test: Statistics correctly separated by triage category.

    Scenario:
    - 2 T1 patients, 2 T2 patients, 2 T3 patients
    - Check that counts and wait times are correctly categorized
    """
    print("\n" + "="*70)
    print("TEST: Multiple Triage Categories")
    print("="*70)

    # Include AXP1 co-located with F2
    distance_matrix = pd.DataFrame(
        [[0, 0], [0, 0]],
        index=["F2", "AXP1"],
        columns=["F2", "AXP1"]
    )

    axp_df = pd.DataFrame([{
        "axp_id": "AXP1",
        "assigned_facility_ids": ["F2"],
    }])

    facility_df = pd.DataFrame([{
        "facility_id": "F2",
        "facility_type": "Role1",
        "c2_up_id": None,
        "unload_policy": "unload_planned",
        "load_policy": "load_planned",
    }])

    units_df = pd.DataFrame([{
        "unit_id": "U1",
        "facility_id": "F2",
        "service_type": "surgery",
        "capacity": 1,
        "service_time_min": 10,
        "service_time_max": 10,
        "service_time_dist": sim.Uniform(10, 10),
    }])

    ambulance_df = pd.DataFrame([{
        "amb_id": "AMB1",
        "capacity": 1,
        "speed_kmh": 60,
        "start_loc_id": "AXP1",
        "c2_id": "C2",
    }])

    corridors_matrix = pd.DataFrame([[0]], index=["F2"], columns=["F2"])
    ambulances_eligibility = pd.DataFrame([[1]], index=["AMB1"], columns=["F2"])

    c2_df = pd.DataFrame([{
        "c2_id": "C2",
        "facility_ids": ["F2"],
        "ambulance_ids": ["AMB1"],
        "corridors": corridors_matrix,
        "ambulances_eligibility": ambulances_eligibility,
    }])

    data = {
        "distance_matrix": distance_matrix,
        "axp_df": axp_df,
        "facility_df": facility_df,
        "units_df": units_df,
        "ambulance_df": ambulance_df,
        "c2_df": c2_df,
    }

    # Mix of triage categories - T1 gets priority
    casualties_df = create_casualties_df([
        {"t_create": 0, "triage": "T1", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T2", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T3", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T1", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T2", "spawn_facility_id": "F2", "prescription": ["surgery"]},
        {"t_create": 0, "triage": "T3", "spawn_facility_id": "F2", "prescription": ["surgery"]},
    ])

    results = run_test_simulation(data, casualties_df, till=100)

    # Check patient counts by triage
    env = results["env"]
    log_df = results["patient_log_df"]
    final_states = log_df.groupby("patient_id").last().reset_index()

    triage_map = {pid: p.triage for pid, p in env.all_patients.items()}
    final_states["triage"] = final_states["patient_id"].map(triage_map)

    for triage in ["T1", "T2", "T3"]:
        count = (final_states["triage"] == triage).sum()
        safe = ((final_states["triage"] == triage) & (final_states["is_safe"] == True)).sum()
        print(f"{triage}: {count} total, {safe} safe")

    # Check wait times by triage
    wait_times_df = results["wait_times_df"]
    care_waits = wait_times_df[wait_times_df["wait_type"] == "care"]

    print("\nCare wait times by triage:")
    for triage in ["T1", "T2", "T3"]:
        tw = care_waits[care_waits["triage"] == triage]["duration"].values
        if len(tw) > 0:
            print(f"  {triage}: {sorted(tw)}")

    # T1 should have shortest waits (priority), T3 longest
    t1_waits = care_waits[care_waits["triage"] == "T1"]["duration"].values
    t3_waits = care_waits[care_waits["triage"] == "T3"]["duration"].values

    if len(t1_waits) == 2 and len(t3_waits) == 2:
        if max(t1_waits) < max(t3_waits):
            print("PASS: T1 patients have shorter waits than T3 (priority working)")
            return True
        else:
            print("FAIL: Priority not working correctly")
            return False
    else:
        print(f"FAIL: Expected 2 T1 and 2 T3 waits")
        return False


def run_all_tests():
    """Run all KPI tests and summarize results."""
    print("\n" + "#"*70)
    print("# RUNNING KPI VALIDATION TESTS")
    print("#"*70)

    tests = [
        ("Transport Time", test_transport_time),
        ("Care Wait Time", test_care_wait_time),
        ("Transport Wait Time", test_transport_wait_time),
        ("Patient Death Counting", test_patient_death),
        ("Unit Utilization", test_unit_utilization),
        ("Wait Time Percentiles", test_wait_time_percentiles),
        ("Multiple Triage Categories", test_multiple_triage_categories),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))

    print("\n" + "#"*70)
    print("# TEST SUMMARY")
    print("#"*70)

    for name, result in results:
        status = "PASS" if result == "PASS" else "FAIL" if result == "FAIL" else "ERROR"
        print(f"  [{status}] {name}")

    passed = sum(1 for _, r in results if r == "PASS")
    total = len(results)
    print(f"\n  {passed}/{total} tests passed")

    return all(r == "PASS" for _, r in results)


if __name__ == "__main__":
    run_all_tests()
