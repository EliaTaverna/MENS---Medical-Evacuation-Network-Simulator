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
        data     = self._apply_r2_retreat(data, kwargs)
        data     = self._apply_r1_access(data, kwargs)
        data     = self._apply_dcs_capacity(data, kwargs)
        data     = self._apply_pec_capacity(data, kwargs)
        data     = self._apply_dcr_capacity(data, kwargs)
        data     = self._apply_set_ambulances_ccp_r1(data, kwargs, seeds["ambulances"])
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

    def _apply_r2_retreat(self, data: dict, kwargs: dict) -> dict:
        """Shift all Role 2 facilities away from the front by retreat_km kilometres.

        Adds retreat_km to every distance_matrix entry between a Role 2 node and
        any non-Role-2 node (row and column).  Inter-Role-2 distances are unchanged
        because all r2 nodes retreat together as a formation.
        """
        retreat_km = float(kwargs.get("r2_retreat_km", 0.0))
        if retreat_km == 0.0:
            return data

        facility_df = data["facility_df"]
        r2_ids = set(
            facility_df.loc[facility_df["facility_type"] == "Role2", "facility_id"].tolist()
        )
        if not r2_ids:
            return data

        dm = data["distance_matrix"].copy()
        non_r2 = [idx for idx in dm.index if idx not in r2_ids]

        for r2_id in r2_ids:
            dm.loc[r2_id, non_r2] += retreat_km
            dm.loc[non_r2, r2_id] += retreat_km

        data["distance_matrix"] = dm
        return data

    def _apply_r1_access(self, data: dict, kwargs: dict) -> dict:
        """Set CCP↔R1 distances for the closest and second-closest R1 per CCP.

        Rankings are determined once from the original (unmodified) distance
        matrix in self.base_data, so they are stable regardless of lever values
        or other modifications applied earlier in prepare().

        Only CCP↔R1 entries are touched; all other distances are unchanged.
        Either lever can be omitted independently.
        """
        closest_km = kwargs.get("r1_closest_km")
        second_km  = kwargs.get("r1_second_km")

        if closest_km is None and second_km is None:
            return data

        facility_df = data["facility_df"]
        r1_ids  = facility_df.loc[
            facility_df["facility_type"] == "Role1", "facility_id"
        ].tolist()
        ccp_ids = facility_df.loc[
            facility_df["facility_type"] == "CCP", "facility_id"
        ].tolist()

        ref_dm = self.base_data["distance_matrix"]
        dm     = data["distance_matrix"].copy()

        r1_ids  = [r for r in r1_ids  if r in ref_dm.index]
        ccp_ids = [c for c in ccp_ids if c in ref_dm.index]

        if not r1_ids or not ccp_ids:
            return data

        for ccp in ccp_ids:
            ranked = ref_dm.loc[ccp, r1_ids].sort_values()
            distinct = ranked.unique()  # sorted ascending, ties collapsed

            if closest_km is not None and len(distinct) >= 1:
                for r1 in ranked.index[ranked == distinct[0]]:
                    dm.loc[ccp, r1] = float(closest_km)
                    dm.loc[r1, ccp] = float(closest_km)

            if second_km is not None and len(distinct) >= 2:
                for r1 in ranked.index[ranked == distinct[1]]:
                    dm.loc[ccp, r1] = float(second_km)
                    dm.loc[r1, ccp] = float(second_km)

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
        """Set the speed of all CCP→R1 ambulances (starters at AXP locations)."""
        speed = kwargs.get("amb_speed_ccp_r1")
        if speed is None:
            return data

        amb_df = data["ambulance_df"].copy()
        mask = amb_df["start_loc_id"].str.startswith("axp_")
        amb_df.loc[mask, "speed_kmh"] = float(speed)
        data["ambulance_df"] = amb_df
        return data

    def _apply_amb_speed_r1_r2(self, data: dict, kwargs: dict) -> dict:
        """Set the speed of all R1→R2 ambulances (starters at R1 locations)."""
        speed = kwargs.get("amb_speed_r1_r2")
        if speed is None:
            return data

        amb_df = data["ambulance_df"].copy()
        mask = amb_df["start_loc_id"].str.startswith("r1_")
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

    def _apply_set_ambulances_ccp_r1(self, data: dict, kwargs: dict, seed: int) -> dict:
        """Set the total number of CCP→R1 ambulances (starters at AXP locations)."""
        n = kwargs.get("n_amb_ccp_r1")
        if n is None:
            return data
        return self._set_ambulances(data, int(n), start_prefix="axp_", seed=seed)

    def _apply_set_ambulances_r1_r2(self, data: dict, kwargs: dict, seed: int) -> dict:
        """Set the total number of R1→R2 ambulances (starters at R1 locations)."""
        n = kwargs.get("n_amb_r1_r2")
        if n is None:
            return data
        return self._set_ambulances(data, int(n), start_prefix="r1_", seed=seed)

    def _set_ambulances(self, data: dict, n: int, start_prefix: str, seed: int) -> dict:
        """Set the number of ambulances on a leg to exactly n.

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

        candidates = amb_df[amb_df["start_loc_id"].str.startswith(start_prefix)]["amb_id"].tolist()
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
            for fid in self._parse_spawning_points(row["bcas_spawning_points"]):
                spawn_map[fid] = {"labels": bcas_labels, "rate": bcas_rate, "par": par}
            if dnbi_rate > 0:
                for fid in self._parse_spawning_points(row["dnbi_spawning_points"]):
                    spawn_map[fid] = {"labels": dnbi_labels, "rate": dnbi_rate, "par": par}

        return spawn_map

    @staticmethod
    def _parse_spawning_points(value) -> list:
        """Parse a spawning_points cell into a list of lowercase CCP IDs."""
        if isinstance(value, list):
            return [c.strip().lower() for c in value]
        return [c.strip().lower() for c in str(value).strip("[]").split(",")]

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
