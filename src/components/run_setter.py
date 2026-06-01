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
        # Travel-time overrides: 2-level schema (same/inter at battalion
        # level, same/other at brigade level).  The per-pair `tt_a__b`
        # override runs last, so it always wins over topology-level values.
        data     = self._apply_ccp_r1_times(data, kwargs)
        data     = self._apply_r1_r1_times(data, kwargs)
        data     = self._apply_axp_r1_times(data, kwargs)
        data     = self._apply_r1_r2_times(data, kwargs)
        data     = self._apply_r2_r2_times(data, kwargs)
        data     = self._apply_tt_overrides(data, kwargs)
        data     = self._apply_dcs_capacity(data, kwargs)
        data     = self._apply_pec_capacity(data, kwargs)
        data     = self._apply_dcr_capacity(data, kwargs)
        data     = self._apply_set_ambulances_r1_to_ccp(data, kwargs, seeds["ambulances"])
        data     = self._apply_set_ambulances_r1_r2(data, kwargs, seeds["ambulances"])
        data     = self._apply_casualties(data, kwargs, seeds["casualties"],
                                          battalion_seed=seeds["battalions"])
        data     = self._apply_dispatch_rules(data, kwargs)
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
        """Override CCP↔R1 travel times (same-BN vs. inter-BN within brigade).

        Kwargs (all optional, minutes):
        - ``ccp_r1_same_bn_min``:  CCP ↔ R1 of the same battalion
        - ``ccp_r1_inter_bn_min``: CCP ↔ R1 of another battalion (same brigade)
        """
        same_bn  = kwargs.get("ccp_r1_same_bn_min")
        inter_bn = kwargs.get("ccp_r1_inter_bn_min")
        if same_bn is None and inter_bn is None:
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r1s_by_bde = self._group_by_brigade(force_df, "r1_ids")

        for _, row in force_df.iterrows():
            bde      = str(row.get("brigade", "")).lower()
            ccp_ids  = self._parse_facility_list(row.get("ccp_ids"))
            own_r1s  = self._parse_facility_list(row.get("r1_ids"))
            other_bn_r1s = [r for r in r1s_by_bde.get(bde, []) if r not in own_r1s]
            self._set_pairs(dm, ccp_ids, own_r1s,      same_bn)
            self._set_pairs(dm, ccp_ids, other_bn_r1s, inter_bn)

        data["travel_time_matrix"] = dm
        return data

    def _apply_r1_r1_times(self, data: dict, kwargs: dict) -> dict:
        """Override R1↔R1 travel times (inter-battalion within same brigade).

        Kwargs (all optional, minutes):
        - ``r1_r1_inter_bn_min``: R1 ↔ R1 across battalions (same brigade)
        """
        inter_bn = kwargs.get("r1_r1_inter_bn_min")
        if inter_bn is None:
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r1s_by_bde = self._group_by_brigade(force_df, "r1_ids")

        for _, row in force_df.iterrows():
            bde     = str(row.get("brigade", "")).lower()
            own_r1s = self._parse_facility_list(row.get("r1_ids"))
            other_bn_r1s = [r for r in r1s_by_bde.get(bde, []) if r not in own_r1s]
            self._set_pairs(dm, own_r1s, other_bn_r1s, inter_bn)

        data["travel_time_matrix"] = dm
        return data

    def _apply_axp_r1_times(self, data: dict, kwargs: dict) -> dict:
        """Override AXP↔R1 travel times (same-BDE vs. other-BDE, flat).

        Kwargs (all optional, minutes):
        - ``axp_r1_same_bde_min``:  AXP ↔ R1 within the same brigade
        - ``axp_r1_other_bde_min``: AXP ↔ R1 in any other brigade
        """
        same_bde  = kwargs.get("axp_r1_same_bde_min")
        other_bde = kwargs.get("axp_r1_other_bde_min")
        if same_bde is None and other_bde is None:
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r1s_by_bde  = self._group_by_brigade(force_df, "r1_ids")
        axps_by_bde = self._group_by_brigade(force_df, "axp_ids")

        for bde, axps in axps_by_bde.items():
            own_r1s   = r1s_by_bde.get(bde, [])
            other_r1s = [r for b, rs in r1s_by_bde.items() if b != bde for r in rs]
            self._set_pairs(dm, axps, own_r1s,   same_bde)
            self._set_pairs(dm, axps, other_r1s, other_bde)

        data["travel_time_matrix"] = dm
        return data

    def _apply_r1_r2_times(self, data: dict, kwargs: dict) -> dict:
        """Override R1↔R2 travel times (same-BDE vs. other-BDE, flat; R2b+R2e uniform).

        Kwargs (all optional, minutes):
        - ``r1_r2_same_bde_min``:  R1 ↔ R2 within the same brigade
        - ``r1_r2_other_bde_min``: R1 ↔ R2 in any other brigade
        """
        same_bde  = kwargs.get("r1_r2_same_bde_min")
        other_bde = kwargs.get("r1_r2_other_bde_min")
        if same_bde is None and other_bde is None:
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r1s_by_bde = self._group_by_brigade(force_df, "r1_ids")
        r2s_by_bde = self._group_by_brigade(force_df, "r2b_ids", "r2e_ids")

        for bde, r1s in r1s_by_bde.items():
            own_r2s   = r2s_by_bde.get(bde, [])
            other_r2s = [r for b, rs in r2s_by_bde.items() if b != bde for r in rs]
            self._set_pairs(dm, r1s, own_r2s,   same_bde)
            self._set_pairs(dm, r1s, other_r2s, other_bde)

        data["travel_time_matrix"] = dm
        return data

    def _apply_r2_r2_times(self, data: dict, kwargs: dict) -> dict:
        """Override R2↔R2 travel times (inter-brigade only, flat).

        Kwargs (all optional, minutes):
        - ``r2_r2_other_bde_min``: R2 ↔ R2 across brigades
        """
        other_bde = kwargs.get("r2_r2_other_bde_min")
        if other_bde is None:
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["travel_time_matrix"].copy()
        r2s_by_bde = self._group_by_brigade(force_df, "r2b_ids", "r2e_ids")

        bdes = list(r2s_by_bde.keys())
        for i, bde_a in enumerate(bdes):
            for bde_b in bdes[i + 1:]:
                self._set_pairs(dm, r2s_by_bde[bde_a], r2s_by_bde[bde_b], other_bde)

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

    # ── Dispatch rules ───────────────────────────────────────────────────────

    def _apply_dispatch_rules(self, data: dict, kwargs: dict) -> dict:
        """Override individual dispatch-rule parameters while keeping rule order fixed.

        The default dispatch_rules (from the scenario YAML) are an ordered list
        of dicts, each with {triage, min_count, min_wait_min}.  This method
        lets experiment levers tweak numeric fields without changing the
        priority ordering.

        For each triage class T the two tuneable aspects are:

        - **count threshold** (``t<x>_min_count``): how many patients of that
          triage must accumulate before the rule fires.  Applied to the
          count-triggered rule (the one with ``min_wait_min == 0``) when
          one exists, otherwise to the single rule for that triage.
        - **wait threshold** (``t<x>_min_wait_min``): minutes a patient waits
          before becoming eligible regardless of count.  Applied to the
          time-triggered rule (``min_wait_min > 0``) when one exists,
          otherwise to the single rule for that triage.

        Supported kwargs (all optional):
        - ``t1_min_count``      (default 1)
        - ``t1_min_wait_min``   (default 60)
        - ``t2_min_count``      (default 2)
        - ``t2_min_wait_min``   (default 240)
        - ``t3_min_count``      (default 4)
        - ``t3_min_wait_min``   (default 720)
        """
        _TRIAGE_KEYS = {
            "T1": ("t1_min_count", "t1_min_wait_min"),
            "T2": ("t2_min_count", "t2_min_wait_min"),
            "T3": ("t3_min_count", "t3_min_wait_min"),
        }

        # Collect only the kwargs that are actually provided
        overrides: dict[str, dict] = {}  # triage → {"count": int, "wait": int}
        for tri, (count_key, wait_key) in _TRIAGE_KEYS.items():
            c = kwargs.get(count_key)
            w = kwargs.get(wait_key)
            if c is not None or w is not None:
                overrides[tri] = {}
                if c is not None:
                    overrides[tri]["count"] = int(c)
                if w is not None:
                    overrides[tri]["wait"] = int(w)

        if not overrides:
            return data

        rules = [dict(r) for r in (data.get("dispatch_rules") or [])]
        if not rules:
            return data

        # Group rule indices by triage
        by_triage: dict[str, list[int]] = {}
        for i, rule in enumerate(rules):
            by_triage.setdefault(rule["triage"], []).append(i)

        for tri, ov in overrides.items():
            indices = by_triage.get(tri, [])
            if not indices:
                continue

            if len(indices) == 1:
                # Single rule for this triage — apply both overrides to it
                idx = indices[0]
                if "count" in ov:
                    rules[idx]["min_count"] = ov["count"]
                if "wait" in ov:
                    rules[idx]["min_wait_min"] = ov["wait"]
            else:
                # Multiple rules (e.g. T2 has a count-trigger and a wait-trigger).
                # count override → rule with min_wait_min == 0
                # wait override  → rule with min_wait_min > 0
                for idx in indices:
                    if rules[idx]["min_wait_min"] == 0 and "count" in ov:
                        rules[idx]["min_count"] = ov["count"]
                    elif rules[idx]["min_wait_min"] > 0 and "wait" in ov:
                        rules[idx]["min_wait_min"] = ov["wait"]

        data["dispatch_rules"] = rules
        return data

    # ── Levers ────────────────────────────────────────────────────────────────

    def _apply_levers(self, kwargs: dict, seed: int) -> dict:
        """Build the levers dict, overriding any scenario defaults with kwargs values.

        Supports decomposed binary pooling levers: if any of
        ``{transport,capacity}_{bn,bde}_pool`` are present in kwargs,
        they are mapped to the integer ``ambulance_eligibility`` and
        ``facility_eligibility`` values the simulation expects:

            bn=0, bde=0 → 0 (strict)
            bn=1, bde=0 → 1 (bn_pool)
            bn=1, bde=1 → 2 (bde_pool)
        """
        levers = dict(self.scenario.get("levers", {}))

        # Map 4 binary levers → 2 integer eligibility values
        _BINARY_LEVER_GROUPS = {
            "ambulance_eligibility":  ("transport_bn_pool", "transport_bde_pool"),
            "facility_eligibility":   ("capacity_bn_pool",  "capacity_bde_pool"),
        }
        _binary_keys = set()
        for elig_key, (bn_key, bde_key) in _BINARY_LEVER_GROUPS.items():
            if bn_key in kwargs or bde_key in kwargs:
                bn  = int(kwargs.get(bn_key, 0))
                bde = int(kwargs.get(bde_key, 0))
                if bde:
                    levers[elig_key] = 2   # bde_pool (implies bn)
                elif bn:
                    levers[elig_key] = 1   # bn_pool
                else:
                    levers[elig_key] = 0   # strict
                _binary_keys.update((bn_key, bde_key))

        for key, val in kwargs.items():
            if key in levers and key not in _binary_keys:
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
