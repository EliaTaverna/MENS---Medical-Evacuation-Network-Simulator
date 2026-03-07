"""Casualty spawner component."""

from __future__ import annotations
import uuid
import salabim as sim

from model.patient import Patient


class CasualtySpawner(sim.Component):
    """Spawns casualties at scheduled times according to a DataFrame."""

    def setup(self, casualties_df):
        self.df = casualties_df.sort_values("t_create").reset_index(drop=True)

    def process(self):
        for _, row in self.df.iterrows():
            dt = float(row.t_create) - self.env.now()
            if dt > 0:
                self.hold(dt)
            spawn = self.env.facility_by_id[row.spawn_facility_id]
            injury_time = float(getattr(row, "injury_time", row.t_create))
            p = Patient(
                patient_id=uuid.uuid4().hex,
                triage=row.triage,
                prescription=row.prescription,
                life_time_dist=row.lifetime_dist,
                spawn_facility=spawn,
                injury_time=injury_time,
            )
            spawn.patient_handed_over(p)
