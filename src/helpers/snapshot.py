"""Simulation state snapshot helpers."""

from __future__ import annotations
import pandas as pd
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.components.facility import Facility


def snapshot_facilities(facilities: List["Facility"]) -> pd.DataFrame:
    """
    Snapshot current facility state into a DataFrame.

    Returns one row per facility, one column per service type (global set
    across all facilities). Cell value = expected waiting time if someone
    joins that service queue now; False if the service is not present.
    """
    if not facilities:
        return pd.DataFrame()

    all_service_types = sorted(
        {st for fac in facilities for st in fac.units.keys()}
    )

    rows = []
    for fac in facilities:
        row = {"facility_id": fac.loc_id}
        for service_type in all_service_types:
            u = fac.units.get(service_type)
            row[service_type] = (
                float(u.expected_wait_if_join_now()) if u is not None else False
            )
        rows.append(row)

    return pd.DataFrame(rows)
