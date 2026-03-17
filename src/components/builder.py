"""Model builder and connector classes."""

from __future__ import annotations
import pandas as pd

from src.components.axp import AXP
from src.components.facility import Facility
from src.components.unit import Unit
from src.components.ambulance import Ambulance
from src.components.c2 import TransportC2


class ModelBuilder:
    """Builds simulation components from DataFrames."""

    def __init__(self, env):
        self.env = env

    def build_roads(self, distance_matrix: pd.DataFrame):
        """
        distance_matrix: DataFrame with locations (facilities/AXPs) as both index and columns.
        Values are distances in km. Use inf or NaN for no direct connection.
        """
        self.env.distance_matrix = distance_matrix.copy()

    def build_axps(self, axp_df: pd.DataFrame):
        self.env.axp_by_id = {}
        for r in axp_df.itertuples(index=False):
            self.env.axp_by_id[r.axp_id] = AXP(
                name=r.axp_id,
                axp_id=r.axp_id,
                assigned_facility_ids=r.assigned_facility_ids
            )

    def build_eligibility_datasets(self, fac_eligibility_datasets: dict, amb_eligibility_datasets: dict):
        """Attach pre-loaded eligibility lookup tables to the environment."""
        self.env.fac_eligibility_datasets = fac_eligibility_datasets
        self.env.amb_eligibility_datasets = amb_eligibility_datasets

    def build_c2s(self, c2_df: pd.DataFrame):
        self.env.c2_by_id = {}
        levers = getattr(self.env, "levers", {})
        for r in c2_df.itertuples(index=False):
            self.env.c2_by_id[r.c2_id] = TransportC2(
                name=r.c2_id,
                c2_id=r.c2_id,
                facility_ids=r.facility_ids,
                ambulance_ids=r.ambulance_ids,
                lever_value_facility_eligibility=levers.get("facility_eligibility", 0),
                lever_value_ambulance_eligibility=levers.get("ambulance_eligibility", 0),
                alpha=levers.get("alpha", 1.0),
                beta=levers.get("beta",  0.0),
                gamma=levers.get("gamma", 0.0),
                delta=levers.get("delta", 0.0),
                random_seed=levers.get("random_seed", 42),
            )

    def build_facilities(self, facility_df: pd.DataFrame, units_df: pd.DataFrame):
        fac_to_service_types = {}
        for r in units_df.itertuples(index=False):
            fac_to_service_types.setdefault(r.facility_id, []).append(r.service_type)

        self.env.facility_by_id = {}
        for r in facility_df.itertuples(index=False):
            service_types = fac_to_service_types.get(r.facility_id, [])
            self.env.facility_by_id[r.facility_id] = Facility(
                name=r.facility_id,
                facility_id=r.facility_id,
                facility_type=r.facility_type,
                c2_up_id=r.c2_up_id,
                unload_key=r.unload_policy,
                load_key=r.load_policy,
                service_types=service_types,
            )

    def build_units(self, units_df: pd.DataFrame):
        self.env.unit_by_id = {}
        for r in units_df.itertuples(index=False):
            unit_id = str(r.unit_id)
            u = Unit(
                name=unit_id,
                facility=None,
                capacity=int(r.capacity),
                service_time_dist=r.service_time_dist,
                service_type=r.service_type,
            )
            u._facility_id = r.facility_id
            self.env.unit_by_id[unit_id] = u

    def build_ambulances(self, ambulance_df: pd.DataFrame):
        self.env.amb_by_id = {}
        for r in ambulance_df.itertuples(index=False):
            amb = Ambulance(
                name=r.amb_id,
                amb_id=r.amb_id,
                capacity=int(r.capacity),
                speed_kmh=float(r.speed_kmh),
                start_loc_id=r.start_loc_id,
                c2_id=r.c2_id,
            )
            self.env.amb_by_id[r.amb_id] = amb

    def build_all(self, distance_matrix, axp_df, facility_df, units_df, ambulance_df, c2_df,
                  fac_eligibility_datasets=None, amb_eligibility_datasets=None):
        self.build_roads(distance_matrix)
        self.build_axps(axp_df)
        self.build_eligibility_datasets(fac_eligibility_datasets or {}, amb_eligibility_datasets or {})
        self.build_c2s(c2_df)
        self.build_facilities(facility_df, units_df)
        self.build_units(units_df)
        self.build_ambulances(ambulance_df)


class ModelConnector:
    """Connects simulation components after building."""

    def __init__(self, env):
        self.env = env

    def connect_units(self):
        for unit_id, u in self.env.unit_by_id.items():
            fac = self.env.facility_by_id[u._facility_id]
            u.facility = fac

    def connect_facilities(self):
        for fac in self.env.facility_by_id.values():
            fac.c2_up = self.env.c2_by_id.get(fac.c2_up_id) if fac.c2_up_id else None
            # Connect units by service_type: find units belonging to this facility
            for unit in self.env.unit_by_id.values():
                if unit._facility_id == fac.loc_id:
                    fac.units[unit.service_type] = unit

    def connect_axps(self):
        self.env.axp_of_facility = {}
        for axp in self.env.axp_by_id.values():
            axp.assigned_facilities = [
                self.env.facility_by_id[fid]
                for fid in axp.assigned_facility_ids
                if fid in self.env.facility_by_id
            ]
            for fid in axp.assigned_facility_ids:
                self.env.axp_of_facility[fid] = axp

    def connect_ambulances(self):
        for amb in self.env.amb_by_id.values():
            amb.c2 = self.env.c2_by_id[amb.c2_id]

            # resolve start location
            if amb.start_loc_id in self.env.axp_by_id:
                amb.location = self.env.axp_by_id[amb.start_loc_id]
            else:
                amb.location = self.env.facility_by_id[amb.start_loc_id]

    def connect_c2s(self):
        for c2 in self.env.c2_by_id.values():
            c2.facilities = [self.env.facility_by_id[fid] for fid in c2.facility_ids]
            c2.ambulances = [self.env.amb_by_id[aid] for aid in c2.ambulance_ids]

    def connect_all(self):
        self.connect_units()
        self.connect_facilities()
        self.connect_axps()
        self.connect_ambulances()
        self.connect_c2s()
