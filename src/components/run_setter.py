"""RunSetter — assembles (data, settings) for a single simulation run."""

from __future__ import annotations

import gc
import random
import numpy as np
import pandas as pd

from src import load_triage_weights_from_scenario
from src.helpers.casualty_generator import generate_casualties_df_from_waves
from src.helpers.runner import run_simulation


class RunSetter:
    """Prepares and optionally executes a single simulation run.

    Instantiated once per EMA experiment — holds shared base objects loaded
    from the scenario (expensive Excel reads happen once, not per run).

    - Call prepare(**kwargs) to get (data, settings) for manual execution.
    - Call the instance directly (**kwargs) for EMA integration — prepares,
      runs the simulation, and returns outcomes. Picklable for multiprocessing.

    To add a new uncertainty or lever type:
        1. Add a private _apply_<category> method with the modification logic.
        2. Call it from prepare() in the appropriate position.
    """

    def __init__(self, base_data: dict, base_config: dict, scenario: dict, till: float,
                 expected_outcomes: list = None, constants: dict = None):
        self.base_data        = base_data
        self.base_config      = base_config
        self.scenario         = scenario
        self.till             = till
        self.expected_outcomes = expected_outcomes or []
        self.constants        = constants or {}

    def __call__(self, **kwargs) -> dict:
        """EMA entry point: prepare → run → compute outcomes → clean up."""
        from src.helpers.helpers_multi import compute_outcomes

        data, settings = self.prepare(**kwargs)
        results  = run_simulation(data, settings, run_folder=None)
        outcomes = compute_outcomes(results)
        # Ensure every declared outcome is present — fill with 0.0 (scalars)
        # or an empty array (time series) if missing.
        for name in self.expected_outcomes:
            if name not in outcomes:
                outcomes[name] = np.array([]) if name.startswith("ts_") else 0.0
        del results, data
        gc.collect()
        return outcomes

    def prepare(self, **kwargs) -> tuple:
        master_seed = self._resolve_seed(kwargs)
        seeds       = self._derive_seeds(master_seed)
        data     = dict(self.base_data)
        # Travel-time overrides run in layers, each applied only if its kwargs
        # are present.  Ring-tier methods set same/neighbor/non_contig cells;
        # the legacy R2-split method below is kept so old YAMLs (r2_basic_*,
        # r2_enhanced_*) still work.  The generic per-pair `tt_a__b`
        # override runs last, so it always wins over topology-level values.
        data     = self._apply_ccp_r1_times(data, kwargs)
        data     = self._apply_r1_r1_times(data, kwargs)
        data     = self._apply_r1_axp_times(data, kwargs)
        data     = self._apply_r2_r1_times(data, kwargs)   # legacy R2-split
        data     = self._apply_r1_r2_times(data, kwargs)   # unified R2 tiers
        data     = self._apply_axp_r2_times(data, kwargs)
        data     = self._apply_tt_overrides(data, kwargs)
        data     = self._apply_dcs_capacity(data, kwargs)
        data     = self._apply_pec_capacity(data, kwargs)
        data     = self._apply_dcr_capacity(data, kwargs)
        data     = self._apply_set_ambulances_r1_to_ccp(data, kwargs, seeds["ambulances"])
        data     = self._apply_set_ambulances_r1_r2(data, kwargs, seeds["ambulances"])
        data     = self._apply_casualties(data, kwargs, seeds["casualties"],
                                          battalion_seed=seeds["battalions"])
        levers   = self._apply_levers(kwargs, master_seed)
        settings = self._build_settings(levers, master_seed, seeds["simulation"], kwargs)
        return data, settings

    # ── Seed ──────────────────────────────────────────────────────────────────

    def _resolve_seed(self, kwargs: dict) -> int:
        return int(kwargs.get("seed", self.scenario.get("seed", 42)))

    def _derive_seeds(self, master_seed: int) -> dict:
        """Derive independent child seeds from one master seed.

        Uses numpy SeedSequence so children are statistically decorrelated
        yet fully reproducible from the master.  The master seed itself is
        stored in results.csv — these derived values are internal.

        Streams
        -------
        simulation  : salabim environment RNG (service times, patient draws)
        casualties  : wave placement, casualty counts, diagnoses, lifetimes
        ambulances  : ambulance / battalion removal sampling
        battalions  : battalion selection (kept separate from ambulances)
        """
        ss       = np.random.SeedSequence(master_seed)
        children = ss.spawn(4)
        # SeedSequence children → positive 32-bit int (salabim / stdlib safe)
        to_int   = lambda s: int(s.generate_state(1)[0]) % (2 ** 31 - 1) + 1
        return {
            "simulation": to_int(children[0]),
            "casualties": to_int(children[1]),
            "ambulances": to_int(children[2]),
            "battalions": to_int(children[3]),
        }

    # ── Network modifications ─────────────────────────────────────────────────

    # ── Tier classification helpers (shared by all ring-tier apply methods) ──

    @staticmethod
    def _set_pairs(dm, a_ids, b_ids, value):
        """Symmetrically write ``value`` into ``dm[a, b] = dm[b, a]`` for every
        (a, b) ∈ a_ids × b_ids that both exist in ``dm.index``.  A no-op when
        ``value`` is None or either list is empty.
        """
        if value is None or not a_ids or not b_ids:
            return
        v = float(value)
        for a in a_ids:
            if a not in dm.index:
                continue
            for b in b_ids:
                if b not in dm.index:
                    continue
                dm.loc[a, b] = v
                dm.loc[b, a] = v

    @classmethod
    def _group_by_brigade(cls, force_df, *cols: str) -> dict:
        """Return ``{brigade_lowercase: [ids from the listed columns, deduped]}``.

        Used to derive non-contig sets: e.g. "same-brigade R1s not in own and
        not in neighbor" needs the full per-brigade R1 roster.
        """
        result: dict = {}
        for _, row in force_df.iterrows():
            bde = str(row.get("brigade", "")).lower()
            if not bde:
                continue
            for col in cols:
                for fid in cls._parse_facility_list(row.get(col)):
                    result.setdefault(bde, []).append(fid)
        return {b: list(dict.fromkeys(ids)) for b, ids in result.items()}

    def _apply_ccp_r1_times(self, data: dict, kwargs: dict) -> dict:
        """Override CCP↔R1 travel times via three levers (same/neighbor/non_contig battalion).

        Levers (all optional, minutes):
        - ``r1_ccp_same_min``:       CCP ↔ R1 of the same battalion
        - ``r1_ccp_neighbor_min``:   CCP ↔ R1 of a neighbor battalion (same brigade)
        - ``r1_ccp_non_contig_min``: CCP ↔ R1 of a non-contig battalion (same
                                    brigade, not own, not neighbor)
        """
        same_min       = kwargs.get("r1_ccp_same_min")
        nbr_min        = kwargs.get("r1_ccp_neighbor_min")
        non_contig_min = kwargs.get("r1_ccp_non_contig_min")
        if same_min is None and nbr_min is None and non_contig_min is None:
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r1s_by_bde = self._group_by_brigade(force_df, "r1_ids")

        for _, row in force_df.iterrows():
            bde         = str(row.get("brigade", "")).lower()
            ccp_ids     = self._parse_facility_list(row.get("ccp_ids"))
            r1_ids      = self._parse_facility_list(row.get("r1_ids"))
            nbr_r1_ids  = self._parse_facility_list(row.get("neighbor_r1_ids"))
            same_bde_r1s = r1s_by_bde.get(bde, [])
            non_contig_r1_ids = [r for r in same_bde_r1s
                                 if r not in r1_ids and r not in nbr_r1_ids]
            self._set_pairs(dm, r1_ids,             ccp_ids, same_min)
            self._set_pairs(dm, nbr_r1_ids,         ccp_ids, nbr_min)
            self._set_pairs(dm, non_contig_r1_ids,  ccp_ids, non_contig_min)

        data["travel_time_matrix"] = dm
        return data

    def _apply_r1_r1_times(self, data: dict, kwargs: dict) -> dict:
        """Override R1↔R1 travel times for within-brigade tiers and cross-brigade (flat).

        Levers (all optional, minutes):
        - ``r1_r1_neighbor_min``:   R1 ↔ R1 of a neighbor battalion (same brigade)
        - ``r1_r1_non_contig_min``: R1 ↔ R1 of a non-contig battalion (same brigade)
        - ``r1_r1_inter_bde_min``:  R1 ↔ R1 across brigades (single flat value)
        """
        neighbor_min   = kwargs.get("r1_r1_neighbor_min")
        non_contig_min = kwargs.get("r1_r1_non_contig_min")
        inter_bde_min  = kwargs.get("r1_r1_inter_bde_min")
        if all(v is None for v in (neighbor_min, non_contig_min, inter_bde_min)):
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r1s_by_bde = self._group_by_brigade(force_df, "r1_ids")

        for _, row in force_df.iterrows():
            bde        = str(row.get("brigade", "")).lower()
            own_r1s    = self._parse_facility_list(row.get("r1_ids"))
            nbr_r1s    = self._parse_facility_list(row.get("neighbor_r1_ids"))
            same_bde_r1s = r1s_by_bde.get(bde, [])
            non_contig_r1s = [r for r in same_bde_r1s
                              if r not in own_r1s and r not in nbr_r1s]
            inter_bde_r1s  = [r for b, rs in r1s_by_bde.items()
                              if b != bde for r in rs]
            self._set_pairs(dm, own_r1s, nbr_r1s,        neighbor_min)
            self._set_pairs(dm, own_r1s, non_contig_r1s, non_contig_min)
            self._set_pairs(dm, own_r1s, inter_bde_r1s,  inter_bde_min)

        data["travel_time_matrix"] = dm
        return data

    def _apply_r1_axp_times(self, data: dict, kwargs: dict) -> dict:
        """Override R1↔AXP travel times by brigade-ring tier.

        Levers (all optional, minutes):
        - ``r1_axp_same_min``:       R1 ↔ AXP of the same brigade
        - ``r1_axp_neighbor_min``:   R1 ↔ AXP of a neighbor brigade
        - ``r1_axp_non_contig_min``: R1 ↔ AXP of a non-contig brigade
        """
        same_min       = kwargs.get("r1_axp_same_min")
        neighbor_min   = kwargs.get("r1_axp_neighbor_min")
        non_contig_min = kwargs.get("r1_axp_non_contig_min")
        if all(v is None for v in (same_min, neighbor_min, non_contig_min)):
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        axps_by_bde = self._group_by_brigade(force_df, "axp_ids")

        for _, row in force_df.iterrows():
            bde           = str(row.get("brigade", "")).lower()
            own_r1s       = self._parse_facility_list(row.get("r1_ids"))
            own_axps      = self._parse_facility_list(row.get("axp_ids"))
            nbr_axps      = self._parse_facility_list(row.get("neighbor_axp_ids"))
            other_bde_axps = [a for b, axs in axps_by_bde.items()
                              if b != bde for a in axs]
            non_contig_axps = [a for a in other_bde_axps if a not in nbr_axps]
            self._set_pairs(dm, own_r1s, own_axps,        same_min)
            self._set_pairs(dm, own_r1s, nbr_axps,        neighbor_min)
            self._set_pairs(dm, own_r1s, non_contig_axps, non_contig_min)

        data["travel_time_matrix"] = dm
        return data

    def _apply_r1_r2_times(self, data: dict, kwargs: dict) -> dict:
        """Override R1↔R2 travel times by brigade-ring tier (R2b + R2e treated uniformly).

        Levers (all optional, minutes):
        - ``r1_r2_same_min``:       R1 ↔ R2 of the same brigade
        - ``r1_r2_neighbor_min``:   R1 ↔ R2 of a neighbor brigade
        - ``r1_r2_non_contig_min``: R1 ↔ R2 of a non-contig brigade
        """
        same_min       = kwargs.get("r1_r2_same_min")
        neighbor_min   = kwargs.get("r1_r2_neighbor_min")
        non_contig_min = kwargs.get("r1_r2_non_contig_min")
        if all(v is None for v in (same_min, neighbor_min, non_contig_min)):
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r2s_by_bde = self._group_by_brigade(force_df, "r2b_ids", "r2e_ids")

        for _, row in force_df.iterrows():
            bde      = str(row.get("brigade", "")).lower()
            own_r1s  = self._parse_facility_list(row.get("r1_ids"))
            own_r2s  = (self._parse_facility_list(row.get("r2b_ids"))
                        + self._parse_facility_list(row.get("r2e_ids")))
            nbr_r2s  = (self._parse_facility_list(row.get("neighbor_r2b_ids"))
                        + self._parse_facility_list(row.get("neighbor_r2e_ids")))
            other_bde_r2s = [r for b, rs in r2s_by_bde.items()
                             if b != bde for r in rs]
            non_contig_r2s = [r for r in other_bde_r2s if r not in nbr_r2s]
            self._set_pairs(dm, own_r1s, own_r2s,        same_min)
            self._set_pairs(dm, own_r1s, nbr_r2s,        neighbor_min)
            self._set_pairs(dm, own_r1s, non_contig_r2s, non_contig_min)

        data["travel_time_matrix"] = dm
        return data

    def _apply_axp_r2_times(self, data: dict, kwargs: dict) -> dict:
        """Override AXP↔R2 travel times by brigade-ring tier (R2b + R2e uniformly).

        Levers (all optional, minutes):
        - ``axp_r2_same_min``:       AXP ↔ R2 of the same brigade
        - ``axp_r2_neighbor_min``:   AXP ↔ R2 of a neighbor brigade
        - ``axp_r2_non_contig_min``: AXP ↔ R2 of a non-contig brigade
        """
        same_min       = kwargs.get("axp_r2_same_min")
        neighbor_min   = kwargs.get("axp_r2_neighbor_min")
        non_contig_min = kwargs.get("axp_r2_non_contig_min")
        if all(v is None for v in (same_min, neighbor_min, non_contig_min)):
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r2s_by_bde = self._group_by_brigade(force_df, "r2b_ids", "r2e_ids")

        for _, row in force_df.iterrows():
            bde      = str(row.get("brigade", "")).lower()
            own_axps = self._parse_facility_list(row.get("axp_ids"))
            own_r2s  = (self._parse_facility_list(row.get("r2b_ids"))
                        + self._parse_facility_list(row.get("r2e_ids")))
            nbr_r2s  = (self._parse_facility_list(row.get("neighbor_r2b_ids"))
                        + self._parse_facility_list(row.get("neighbor_r2e_ids")))
            other_bde_r2s = [r for b, rs in r2s_by_bde.items()
                             if b != bde for r in rs]
            non_contig_r2s = [r for r in other_bde_r2s if r not in nbr_r2s]
            self._set_pairs(dm, own_axps, own_r2s,        same_min)
            self._set_pairs(dm, own_axps, nbr_r2s,        neighbor_min)
            self._set_pairs(dm, own_axps, non_contig_r2s, non_contig_min)

        data["travel_time_matrix"] = dm
        return data

    def _apply_r2_r1_times(self, data: dict, kwargs: dict) -> dict:
        """Override R1↔R2 travel times via four levers (basic/enhanced × same/neighbor).

        Levers (all optional, minutes):
        - ``r2_basic_r1_same_min``:     R2-basic ↔ same-brigade R1
        - ``r2_basic_r1_neighbor_min``: R2-basic ↔ neighbor-brigade R1
        - ``r2_enhanced_r1_same_min``:  R2-enhanced ↔ same-brigade R1
        - ``r2_enhanced_r1_neighbor_min``: R2-enhanced ↔ neighbor-brigade R1
        """
        same_basic    = kwargs.get("r2_basic_r1_same_min")
        nbr_basic     = kwargs.get("r2_basic_r1_neighbor_min")
        same_enhanced = kwargs.get("r2_enhanced_r1_same_min")
        nbr_enhanced  = kwargs.get("r2_enhanced_r1_neighbor_min")
        if all(v is None for v in (same_basic, nbr_basic, same_enhanced, nbr_enhanced)):
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()

        def _set_pairs(r1s, r2s, value):
            if value is None or not r1s or not r2s:
                return
            v = float(value)
            for r1 in r1s:
                if r1 not in dm.index:
                    continue
                for r2 in r2s:
                    if r2 not in dm.index:
                        continue
                    dm.loc[r1, r2] = v
                    dm.loc[r2, r1] = v

        for _, row in force_df.iterrows():
            r1_ids       = self._parse_facility_list(row.get("r1_ids"))
            r2b_ids      = self._parse_facility_list(row.get("r2b_ids"))
            r2e_ids      = self._parse_facility_list(row.get("r2e_ids"))
            nbr_r2b_ids  = self._parse_facility_list(row.get("neighbor_r2b_ids"))
            nbr_r2e_ids  = self._parse_facility_list(row.get("neighbor_r2e_ids"))
            _set_pairs(r1_ids, r2b_ids,     same_basic)
            _set_pairs(r1_ids, nbr_r2b_ids, nbr_basic)
            _set_pairs(r1_ids, r2e_ids,     same_enhanced)
            _set_pairs(r1_ids, nbr_r2e_ids, nbr_enhanced)

        data["travel_time_matrix"] = dm
        return data

    def _apply_tt_overrides(self, data: dict, kwargs: dict) -> dict:
        """Override arbitrary pairwise travel times from `tt_<id1>__<id2>` levers.

        Any kwarg whose key starts with ``tt_`` and whose tail splits
        on ``__`` into exactly two non-empty tokens is treated as an explicit
        override for that facility pair. Both ``dm[id1, id2]`` and
        ``dm[id2, id1]`` are set to the lever value (minutes).

        Separator is a double underscore because facility IDs themselves
        contain single underscores (e.g. ``r1_br1_bt1_1``). Unknown IDs raise
        — a typo in the lever name should fail loudly, not be ignored.
        Applied last so explicit overrides always win over the programmatic shifts.
        """
        overrides = []
        for key, val in kwargs.items():
            if not key.startswith("tt_") or val is None:
                continue
            tail = key[len("tt_"):]
            parts = tail.split("__")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            overrides.append((key, parts[0], parts[1], float(val)))

        if not overrides:
            return data

        dm = data["travel_time_matrix"].copy()
        for key, id1, id2, val in overrides:
            if id1 not in dm.index or id2 not in dm.index:
                raise ValueError(
                    f"Lever {key!r}: facility id {id1!r} or {id2!r} not in "
                    f"travel time matrix."
                )
            dm.loc[id1, id2] = val
            dm.loc[id2, id1] = val
        data["travel_time_matrix"] = dm
        return data

    def _apply_dcs_capacity(self, data: dict, kwargs: dict) -> dict:
        """Set the capacity of every DCS unit across all facilities."""
        cap = kwargs.get("dcs_capacity")
        if cap is None:
            return data

        units_df = data["units_df"].copy()
        units_df.loc[units_df["service_type"] == "DCS", "capacity"] = int(cap)
        data["units_df"] = units_df
        return data

    def _apply_pec_capacity(self, data: dict, kwargs: dict) -> dict:
        """Set the capacity of every PEC unit across all facilities."""
        cap = kwargs.get("pec_capacity")
        if cap is None:
            return data

        units_df = data["units_df"].copy()
        units_df.loc[units_df["service_type"] == "PEC", "capacity"] = int(cap)
        data["units_df"] = units_df
        return data

    def _apply_dcr_capacity(self, data: dict, kwargs: dict) -> dict:
        """Set the capacity of every DCR unit across all facilities."""
        cap = kwargs.get("dcr_capacity")
        if cap is None:
            return data

        units_df = data["units_df"].copy()
        units_df.loc[units_df["service_type"] == "DCR", "capacity"] = int(cap)
        data["units_df"] = units_df
        return data

    def _apply_set_ambulances_r1_to_ccp(self, data: dict, kwargs: dict, seed: int) -> dict:
        """Set the number of CCP→R1 ambulances (dispatched by C2_CCP_R1) PER R1."""
        n = kwargs.get("n_amb_r1_to_ccp")
        if n is None:
            return data

        amb_df = data["ambulance_df"]
        mask   = amb_df["c2_id"].str.lower() == "c2_ccp_r1"
        if not mask.any():
            return data

        homes = sorted(amb_df.loc[mask, "start_loc_id"].unique().tolist())
        rng   = random.Random(seed)
        for home in homes:
            sub_seed = rng.randrange(1 << 30)
            data = self._set_ambulances(
                data, int(n), c2_id="c2_ccp_r1", seed=sub_seed, start_loc_id=home
            )
        return data

    def _apply_set_ambulances_r1_r2(self, data: dict, kwargs: dict, seed: int) -> dict:
        """Set the total number of R1→R2 ambulances (dispatched by C2_R1_R2)."""
        n = kwargs.get("n_amb_r1_r2")
        if n is None:
            return data
        return self._set_ambulances(data, int(n), c2_id="c2_r1_r2", seed=seed)

    def _set_ambulances(self, data: dict, n: int, c2_id: str, seed: int,
                        start_loc_id: str = None) -> dict:
        """Set the number of ambulances on a leg (optionally at a single home) to exactly n."""
        rng      = random.Random(seed)
        amb_df   = data["ambulance_df"]
        c2_df    = data["c2_df"]
        amb_elig = data.get("amb_eligibility_datasets", {})

        mask = amb_df["c2_id"].str.lower() == c2_id.lower()
        if start_loc_id is not None:
            mask &= amb_df["start_loc_id"] == start_loc_id
        candidates = amb_df.loc[mask, "amb_id"].tolist()
        current_n  = len(candidates)

        if n == current_n or not candidates:
            return data

        if n < current_n:
            to_remove = set(rng.sample(candidates, current_n - n))

            data["ambulance_df"] = (
                amb_df[~amb_df["amb_id"].isin(to_remove)].reset_index(drop=True)
            )
            new_c2_df = c2_df.copy()
            new_c2_df["ambulance_ids"] = new_c2_df["ambulance_ids"].apply(
                lambda ids: [aid for aid in ids if aid not in to_remove]
            )
            data["c2_df"] = new_c2_df
            new_elig = {}
            for k, df in amb_elig.items():
                new_elig[k] = df.loc[[i for i in df.index if i not in to_remove]]
            data["amb_eligibility_datasets"] = new_elig

        else:
            amb_df   = amb_df.copy()
            new_c2_df = c2_df.copy()
            new_elig  = {k: df.copy() for k, df in amb_elig.items()}
            existing_ids = set(amb_df["amb_id"].tolist())

            for _ in range(n - current_n):
                source_id  = rng.choice(candidates)
                source_row = amb_df[amb_df["amb_id"] == source_id].iloc[0].copy()

                counter = 1
                while True:
                    new_id = f"{source_id}_x{counter}"
                    if new_id not in existing_ids:
                        break
                    counter += 1
                existing_ids.add(new_id)

                source_row["amb_id"] = new_id
                amb_df = pd.concat([amb_df, pd.DataFrame([source_row])], ignore_index=True)

                new_c2_df["ambulance_ids"] = new_c2_df["ambulance_ids"].apply(
                    lambda ids, sid=source_id, nid=new_id: ids + [nid] if sid in ids else ids
                )

                for k, df in new_elig.items():
                    if source_id in df.index:
                        new_row = df.loc[[source_id]].copy()
                        new_row.index = [new_id]
                        new_elig[k] = pd.concat([df, new_row])

            data["ambulance_df"]           = amb_df
            data["c2_df"]                  = new_c2_df
            data["amb_eligibility_datasets"] = new_elig

        return data

    # ── Casualty generation ───────────────────────────────────────────────────

    def _build_diagnosis_mix_df(self, kwargs: dict):
        """Return the diagnosis_mix_df, possibly with overridden BCAS shares."""
        f_t1 = kwargs.get("fraction_t1_bcas")
        f_t2 = kwargs.get("fraction_t2_bcas")
        base = self.base_config["diagnosis_mix_df"]
        if f_t1 is None and f_t2 is None:
            return base

        pinned: dict = {}
        if f_t1 is not None:
            pinned["T1_BCAS"] = float(f_t1)
        if f_t2 is not None:
            pinned["T2_BCAS"] = float(f_t2)
        pinned_sum = sum(pinned.values())
        if pinned_sum > 1.0 + 1e-9:
            raise ValueError(
                f"fraction_t1_bcas + fraction_t2_bcas = {pinned_sum:.4f} > 1.0"
            )

        bcas_mask = base["triage"].astype(str).isin(
            ["T1_BCAS", "T2_BCAS", "T3_BCAS", "KIA_BCAS", "MIA_BCAS"]
        )
        df = base.copy()
        bcas_rows = df.loc[bcas_mask].copy()
        new_freq = {}
        pinned_set = set(pinned.keys())
        residual_rows = bcas_rows[~bcas_rows["triage"].isin(pinned_set)]
        residual_base_sum = float(residual_rows["frequency"].sum())
        residual_budget = 1.0 - pinned_sum

        if residual_budget > 1e-9 and residual_base_sum == 0.0:
            raise ValueError(
                "BCAS residual labels have zero total baseline frequency; cannot "
                "allocate the residual budget proportionally."
            )

        for _, row in bcas_rows.iterrows():
            label = str(row["triage"])
            if label in pinned:
                new_freq[label] = pinned[label]
            elif residual_base_sum > 0:
                new_freq[label] = float(row["frequency"]) / residual_base_sum * residual_budget
            else:
                new_freq[label] = 0.0

        df.loc[bcas_mask, "frequency"] = df.loc[bcas_mask, "triage"].astype(str).map(new_freq)
        return df

    def _apply_casualties(self, data: dict, kwargs: dict, seed: int,
                          battalion_seed: int = None) -> dict:
        spawn_map = self._resolve_spawn_map(kwargs, battalion_seed or seed)

        bcas_rate = kwargs.get("bcas_rate")
        if bcas_rate is not None:
            for cfg in spawn_map.values():
                if any("BCAS" in lbl for lbl in cfg.get("labels", [])):
                    cfg["rate"] = float(bcas_rate)

        par_total = next(iter(spawn_map.values()), {}).get(
            "par", self.base_config.get("par_total", 1000)
        )
        diagnosis_mix_df = self._build_diagnosis_mix_df(kwargs)
        full_df, wave_dicts = generate_casualties_df_from_waves(
            spawn_map=spawn_map,
            par_total=par_total,
            n_days=int(kwargs.get("n_days",                       self.base_config["n_days"])),
            waves_per_day=int(kwargs.get("waves_per_day",         self.base_config["waves_per_day"])),
            wave_duration_min=int(kwargs.get("wave_duration_min", self.base_config["wave_duration_min"])),
            diagnosis_mix_df=diagnosis_mix_df,
            rx_df=self.base_config["rx_df"],
            seed=seed,
        )
        data["casualties_df"] = full_df[
            ~full_df["triage"].str.startswith(("KIA", "MIA"))
        ].copy().reset_index(drop=True)
        data["wave_dicts"]     = wave_dicts
        data["triage_weights"] = load_triage_weights_from_scenario(self.scenario)
        return data

    def _resolve_spawn_map(self, kwargs: dict, seed: int) -> dict:
        """Return spawn_map for this run."""
        n_bat = kwargs.get("n_battalions_in_combat")
        force_df = self.base_config.get("force_structure_df")
        if n_bat is not None and force_df is not None:
            return self._select_battalions(force_df, int(n_bat), seed)

        return {fid: dict(cfg) for fid, cfg in self.base_config["spawn_map"].items()}

    def _select_battalions(self, force_df, n: int, seed: int) -> dict:
        """Randomly select n battalions from the full pool and build their spawn_map."""
        indices = random.Random(seed).sample(range(len(force_df)), min(n, len(force_df)))
        selected = force_df.iloc[indices]
        par = int(selected["par"].sum())

        bcas_labels = self.base_config["bcas_labels"]
        bcas_rate   = self.base_config["bcas_rate"]
        dnbi_labels = self.base_config["dnbi_labels"]
        dnbi_rate   = self.base_config["dnbi_rate"]

        spawn_map = {}
        for _, row in selected.iterrows():
            for fid in self._parse_facility_list(row["ccp_ids"]):
                spawn_map[fid] = {"labels": bcas_labels, "rate": bcas_rate, "par": par}
            if dnbi_rate > 0:
                for fid in self._parse_facility_list(row["r1_ids"]):
                    spawn_map[fid] = {"labels": dnbi_labels, "rate": dnbi_rate, "par": par}

        return spawn_map

    @staticmethod
    def _parse_facility_list(value) -> list:
        """Parse a comma-joined facility-ID cell into a list of lowercase IDs."""
        if value is None:
            return []
        if isinstance(value, float) and value != value:  # NaN
            return []
        if isinstance(value, list):
            return [c.strip().lower() for c in value if str(c).strip()]
        s = str(value).strip("[]").strip()
        if not s:
            return []
        return [c.strip().lower() for c in s.split(",") if c.strip()]

    # ── Levers ────────────────────────────────────────────────────────────────

    def _apply_levers(self, kwargs: dict, seed: int) -> dict:
        """Build the levers dict, overriding any scenario defaults with kwargs values."""
        levers = dict(self.scenario.get("levers", {}))
        for key, val in kwargs.items():
            if key in levers:
                levers[key] = val
        levers["random_seed"] = seed
        return levers

    # ── Settings ──────────────────────────────────────────────────────────────

    def _build_settings(self, levers: dict, master_seed: int, sim_seed: int,
                        kwargs: dict) -> dict:
        dc_interval = self.scenario.get("simulation", {}).get("data_collector_interval", 0.0)
        return {
            "seed":                     master_seed,
            "sim_seed":                 sim_seed,
            "till":                     float(kwargs.get("till", self.till)),
            "trace":                    False,
            "make_video":               False,
            "save_logs":                False,
            "state_log_interval":       None,
            "levers":                   levers,
            "outputs":                  {},
            "data_collector_interval":  float(dc_interval),
        }
