"""Periodic data collector — snapshots queue lengths at every facility."""

from __future__ import annotations
import salabim as sim


class DataCollector(sim.Component):
    """Wakes every *interval* minutes and records how many T1/T2/T3 patients are:
    - waiting for transport at each Role 1 facility (exit_queue)
    - waiting for DCS surgery at each Role 2 facility (entry_lines["DCS"])

    Records are appended to env.data_collector_records as plain dicts:
        {time, facility_id, metric, T1, T2, T3}
    where metric is "transport" (Role1) or "dcs" (Role2).
    """

    def setup(self, interval: float = 10.0):
        self.interval = interval

    def process(self):
        while True:
            self.hold(self.interval)
            t = self.env.now()

            for fac in self.env.facility_by_id.values():

                if fac.facility_type == "Role1":
                    counts = {"T1": 0, "T2": 0, "T3": 0}
                    for req in fac.exit_queue:
                        prefix = req.triage[:2]
                        if prefix in counts:
                            counts[prefix] += 1
                    self.env.data_collector_records.append(
                        {"time": t, "facility_id": fac.loc_id, "metric": "transport",
                         **counts}
                    )

                    dcr_queue = fac.entry_lines.get("DCR", [])
                    counts_dcr = {"T1": 0, "T2": 0, "T3": 0}
                    for p in dcr_queue:
                        prefix = p.triage[:2]
                        if prefix in counts_dcr:
                            counts_dcr[prefix] += 1
                    self.env.data_collector_records.append(
                        {"time": t, "facility_id": fac.loc_id, "metric": "dcr",
                         **counts_dcr}
                    )

                elif fac.facility_type == "Role2":
                    dcs_queue = fac.entry_lines.get("DCS", [])
                    counts = {"T1": 0, "T2": 0, "T3": 0}
                    for p in dcs_queue:
                        prefix = p.triage[:2]
                        if prefix in counts:
                            counts[prefix] += 1
                    self.env.data_collector_records.append(
                        {"time": t, "facility_id": fac.loc_id, "metric": "dcs",
                         **counts}
                    )
