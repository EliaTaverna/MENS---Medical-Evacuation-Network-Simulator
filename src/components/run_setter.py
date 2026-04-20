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
        data     = self._apply_ccp_r1_distances(data, kwargs)
        data     = self._apply_r2_r1_distances(data, kwargs)
        data     = self._apply_distance_overrides(data, kwargs)
        data     = self._apply_dcs_capacity(data, kwargs)
        data     = self._apply_pec_capacity(data, kwargs)
        data     = self._apply_dcr_capacity(data, kwargs)
        data     = self._apply_set_ambulances_r1_to_ccp(data, kwargs, seeds["ambulances"])
        data     = self._apply_set_ambulances_r1_r2(data, kwargs, seeds["ambulances"])
        data     = self._apply_amb_speed_ccp_r1(data, kwargs)
        data     = self._apply_amb_speed_r1_r2(data, kwargs)
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

    def _apply_ccp_r1_distances(self, data: dict, kwargs: dict) -> dict:
        """Override CCP↔R1 distances via two levers (same-bn / neighbor-bn).

        Levers (all optional, km):
        - ``r1_ccp_same_km``:     CCP ↔ R1 of the same battalion
        - ``r1_ccp_neighbor_km``: CCP ↔ R1 of a neighbor battalion (same brigade)

        Each lever, when provided, rewrites every matching distance-matrix entry
        in both directions (``dm[r1, ccp] = dm[ccp, r1] = value``). Pairs are
        enumerated from ``force_structure_df``: for each battalion row, own
        ``ccp_ids`` are paired with own ``r1_ids`` (the "same" tier) and with
        ``neighbor_r1_ids`` (the "neighbor" tier, i.e. R1s of neighbor
        battalions in the same brigade).

        Applied before ``_apply_r2_r1_distances`` and
        ``_apply_distance_overrides`` in ``prepare()``; explicit per-pair
        overrides still win over these topology-level values.
        """
        same_km = kwargs.get("r1_ccp_same_km")
        nbr_km  = kwargs.get("r1_ccp_neighbor_km")
        if same_km is None and nbr_km is None:
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["distance_matrix"].copy()

        def _set_pairs(r1s, ccps, value):
            if value is None or not r1s or not ccps:
                return
            v = float(value)
            for r1 in r1s:
                if r1 not in dm.index:
                    continue
                for ccp in ccps:
                    if ccp not in dm.index:
                        continue
                    dm.loc[r1, ccp] = v
                    dm.loc[ccp, r1] = v

        for _, row in force_df.iterrows():
            ccp_ids     = self._parse_facility_list(row.get("ccp_ids"))
            r1_ids      = self._parse_facility_list(row.get("r1_ids"))
            nbr_r1_ids  = self._parse_facility_list(row.get("neighbor_r1_ids"))
            _set_pairs(r1_ids,     ccp_ids, same_km)
            _set_pairs(nbr_r1_ids, ccp_ids, nbr_km)

        data["distance_matrix"] = dm
        return data

    def _apply_r2_r1_distances(self, data: dict, kwargs: dict) -> dict:
        """Override R1↔R2 distances via four levers (basic/enhanced × same/neighbor).

        Levers (all optional, km):
        - ``r2_basic_r1_same_km``:     R2-basic ↔ same-brigade R1
        - ``r2_basic_r1_neighbor_km``: R2-basic ↔ neighbor-brigade R1
        - ``r2_enhanced_r1_same_km``:  R2-enhanced ↔ same-brigade R1
        - ``r2_enhanced_r1_neighbor_km``: R2-enhanced ↔ neighbor-brigade R1

        Each lever, when provided, rewrites every matching distance-matrix entry
        in both directions (``dm[r1, r2] = dm[r2, r1] = value``). Pairs are
        enumerated from ``force_structure_df``: for each battalion row, own
        ``r1_ids`` are paired with own-brigade ``r2b_ids`` / ``r2e_ids`` (the
        "same" tier) and with ``neighbor_r2b_ids`` / ``neighbor_r2e_ids`` (the
        "neighbor" tier). AXP↔R2 entries are not touched — use
        ``distance_<a>__<b>`` overrides if you need those too.

        Applied after ``_apply_ccp_r1_distances`` and before
        ``_apply_distance_overrides`` so explicit per-pair overrides always win.
        """
        same_basic    = kwargs.get("r2_basic_r1_same_km")
        nbr_basic     = kwargs.get("r2_basic_r1_neighbor_km")
        same_enhanced = kwargs.get("r2_enhanced_r1_same_km")
        nbr_enhanced  = kwargs.get("r2_enhanced_r1_neighbor_km")
        if all(v is None for v in (same_basic, nbr_basic, same_enhanced, nbr_enhanced)):
            return data

        force_df = self.base_config.get("force_structure_df")
        if force_df is None:
            return data

        dm = data["distance_matrix"].copy()

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

        data["distance_matrix"] = dm
        return data

    def _apply_distance_overrides(self, data: dict, kwargs: dict) -> dict:
        """Override arbitrary pairwise distances from `distance_<id1>__<id2>` levers.

        Any kwarg whose key starts with ``distance_`` and whose tail splits
        on ``__`` into exactly two non-empty tokens is treated as an explicit
        override for that facility pair. Both ``dm[id1, id2]`` and
        ``dm[id2, id1]`` are set to the lever value (km).

        Separator is a double underscore because facility IDs themselves
        contain single underscores (e.g. ``r1_br1_bt1_1``). Unknown IDs raise
        — a typo in the lever name should fail loudly, not be ignored.
        Applied after ``_apply_ccp_r1_distances`` and ``_apply_r2_r1_distances``
        so explicit overrides always win over the programmatic shifts.
        """
        overrides = []
        for key, val in kwargs.items():
            if not key.startswith("distance_") or val is None:
                continue
            tail = key[len("distance_"):]
            parts = tail.split("__")
            if len(parts) != 2 or not parts[0] or not parts[1]:
                continue
            overrides.append((key, parts[0], parts[1], float(val)))

        if not overrides:
            return data

        dm = data["distance_matrix"].copy()
        for key, id1, id2, val in overrides:
            if id1 not in dm.index or id2 not in dm.index:
                raise ValueError(
                    f"Lever {key!r}: facility id {id1!r} or {id2!r} not in "
                    f"distance matrix."
                )
            dm.loc[id1, id2] = val
            dm.loc[id2, id1] = val
        data["distance_matrix"] = dm
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

    def _apply_amb_speed_ccp_r1(self, data: dict, kwargs: dict) -> dict:
        """Set the speed of all CCP→R1 ambulances (dispatched by C2_CCP_R1).

        Partitioning key is ``c2_id``, not ``start_loc_id`` prefix: the prefix
        scheme was ambiguous once the 3-fleet layout introduced R1-homed
        ambulances on both legs, and was reversed even in legacy scenarios.
        """
        speed = kwargs.get("amb_speed_ccp_r1")
        if speed is None:
            return data

        amb_df = data["ambulance_df"].copy()
        mask = amb_df["c2_id"].str.lower() == "c2_ccp_r1"
        amb_df.loc[mask, "speed_kmh"] = float(speed)
        data["ambulance_df"] = amb_df
        return data

    def _apply_amb_speed_r1_r2(self, data: dict, kwargs: dict) -> dict:
        """Set the speed of all R1→R2 ambulances (dispatched by C2_R1_R2)."""
        speed = kwargs.get("amb_speed_r1_r2")
        if speed is None:
            return data

        amb_df = data["ambulance_df"].copy()
        mask = amb_df["c2_id"].str.lower() == "c2_r1_r2"
        amb_df.loc[mask, "speed_kmh"] = float(speed)
        data["ambulance_df"] = amb_df
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
        """Set the number of CCP→R1 ambulances (dispatched by C2_CCP_R1) PER R1.

        Semantics: ``n`` is interpreted as the count at each Role-1 home base.
        The resulting total fleet is ``n * number_of_R1s`` that host this leg.
        Each home base is resized independently so counts are deterministic
        (e.g. n=4 with 2 R1s → 4 at r1_a, 4 at r1_b; not a random 3/5 split).
        """
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
        """Set the number of ambulances on a leg (optionally at a single home) to exactly n.

        The leg is identified by the dispatching C2 (``c2_id`` column in
        ``ambulance_df``), not by ``start_loc_id`` prefix. This works for both
        the legacy 2-fleet layout (CCP→R1 at R1, R1→R2 at AXP) and the new
        3-fleet layout (Fleet 1 CCP→R1 at R1, Fleet 2 R1→R2 at R1,
        Fleet 3 R1→R2 at AXP).

        If ``start_loc_id`` is given, the target ``n`` is applied only within
        that home base (useful for per-R1 sizing); otherwise it's applied to
        the whole C2 group.

        If n > current count: duplicates random ambulances from the leg, assigning
        new unique IDs, and registers them with the owning C2 and all ambulance
        eligibility datasets (inheriting the source ambulance's eligibility row).
        If n < current count: removes randomly selected ambulances from the leg,
        the C2 roster, and the eligibility datasets.
        If n == current count: no-op.
        """
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

                # register with the same C2 as the source ambulance
                new_c2_df["ambulance_ids"] = new_c2_df["ambulance_ids"].apply(
                    lambda ids, sid=source_id, nid=new_id: ids + [nid] if sid in ids else ids
                )

                # inherit eligibility rows from source
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
        full_df, wave_dicts = generate_casualties_df_from_waves(
            spawn_map=spawn_map,
            par_total=par_total,
            n_days=int(kwargs.get("n_days",                       self.base_config["n_days"])),
            waves_per_day=int(kwargs.get("waves_per_day",         self.base_config["waves_per_day"])),
            wave_duration_min=int(kwargs.get("wave_duration_min", self.base_config["wave_duration_min"])),
            diagnosis_mix_df=self.base_config["diagnosis_mix_df"],
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
        """Return spawn_map for this run.

        If n_battalions_in_combat is specified and force_structure_df is available,
        randomly selects that many battalions and builds the spawn_map from their CCPs.
        Otherwise falls back to the base spawn_map (already filtered to involved battalions).
        """
        n_bat = kwargs.get("n_battalions_in_combat")
        force_df = self.base_config.get("force_structure_df")
        if n_bat is not None and force_df is not None:
            return self._select_battalions(force_df, int(n_bat), seed)

        return {fid: dict(cfg) for fid, cfg in self.base_config["spawn_map"].items()}

    def _select_battalions(self, force_df, n: int, seed: int) -> dict:
        """Randomly select n battalions from the full pool and build their spawn_map.

        The 'involved' flag in the force structure is only for the default baseline run.
        EMA picks battalions freely — any battalion in force_df is eligible.
        """
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
        """Parse a comma-joined facility-ID cell into a list of lowercase IDs.

        Handles list (from YAML), non-empty string (from Excel), and
        NaN / None / empty (treated as no IDs).
        """
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
        """Build the levers dict, overriding any scenario defaults with kwargs values.

        Extend here for new policy levers — any key present in the scenario's
        levers section is automatically picked up.
        """
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
            "seed":                     master_seed,  # logged to results.csv for reproducibility
            "sim_seed":                 sim_seed,     # actual salabim RNG seed (derived)
            "till":                     float(kwargs.get("till", self.till)),
            "trace":                    False,
            "make_video":               False,
            "save_logs":                False,
            "state_log_interval":       None,
            "levers":                   levers,
            "outputs":                  {},
            "data_collector_interval":  float(dc_interval),
        }
