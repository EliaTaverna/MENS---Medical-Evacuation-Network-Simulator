from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Union
import random

import numpy as np
import pandas as pd
import yaml


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class Wave:
    wave_id: str
    t_start: float          # minutes from sim start
    t_end: float            # minutes from sim start
    facility_id: str
    n_casualties: int = 0
    df: Optional[pd.DataFrame] = None  # at least: t_create, spawn_facility_id


WaveDict = Dict[str, List[Wave]]  # facility_id -> list[Wave]


# ============================================================
# HELPERS
# ============================================================

def _normalize_probs(p: np.ndarray) -> np.ndarray:
    s = float(p.sum())
    if s <= 0:
        raise ValueError("Probabilities must sum to > 0")
    return p / s


# ============================================================
# CONFIG LOADING
# ============================================================

_THIS_DIR = Path(__file__).parent
DEFAULT_CONFIG_DIR = _THIS_DIR / "data" / "single_run" / "input" / "casualty_generation_settings"
DEFAULT_OUTPUT_DIR = _THIS_DIR / "data" / "single_run" / "input" / "casualty_dataframes"


def load_casualty_generation_config(
    config_dir: Union[str, Path] = DEFAULT_CONFIG_DIR,
) -> dict:
    """
    Load casualty generation configuration from YAML + XLSX files.

    Expected files in config_dir:
        - sampling_settings/config.yaml: parameters including spawn_map
        - diagnosis/diagnosis_mix.xlsx: all triage labels with frequencies
        - prescriptions/prescriptions_meisner.xlsx: label, frequency, prescription
    """
    config_dir = Path(config_dir)

    with open(config_dir / "sampling_settings" / "config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    diagnosis_mix_df = pd.read_excel(config_dir / "diagnosis" / "diagnosis_mix.xlsx")

    rx_df = pd.read_excel(config_dir / "prescriptions" / "prescriptions_meisner.xlsx")
    rx_df["prescription"] = rx_df["prescription"].apply(eval)

    lifetime_factory_by_triage = _lifetime_df_to_factories(diagnosis_mix_df)

    return {
        "spawn_map": params["spawn_map"],
        "par_total": params["par_total"],
        "n_days": params["n_days"],
        "waves_per_day": params["waves_per_day"],
        "wave_duration_min": params["wave_duration_min"],
        "seed": params.get("seed", 42),
        "diagnosis_mix_df": diagnosis_mix_df,
        "rx_df": rx_df,
        "lifetime_factory_by_triage": lifetime_factory_by_triage,
    }


def _lifetime_df_to_factories(diagnosis_mix_df: pd.DataFrame) -> Dict[str, Callable[[], dict]]:
    """Convert diagnosis_mix dataframe to lifetime param-dict factory functions.

    Each factory returns a plain dict describing the distribution parameters,
    e.g. {"type": "triangular", "low": 60, "high": 240, "mode": 120}.
    No salabim objects are created here — distributions are built inside the
    simulation (after the seeded environment is active).
    """
    factories = {}

    for _, row in diagnosis_mix_df.iterrows():
        triage = row["triage"]
        dist_type = row["lifetime_distribution"].lower()

        if dist_type == "uniform":
            low, high = float(row["lifetime_low"]), float(row["lifetime_high"])
            factories[triage] = lambda l=low, h=high: {"type": "uniform", "low": l, "high": h}
        elif dist_type == "triangular":
            low  = float(row["lifetime_low"])
            high = float(row["lifetime_high"])
            mode = row.get("lifetime_mode", (low + high) / 2)
            if pd.isna(mode):
                mode = (low + high) / 2
            mode = float(mode)
            factories[triage] = lambda l=low, h=high, m=mode: {"type": "triangular", "low": l, "high": h, "mode": m}
        elif dist_type == "exponential":
            mean = float(row["lifetime_mean"])
            factories[triage] = lambda m=mean: {"type": "exponential", "mean": m}
        elif dist_type == "normal":
            mean = float(row["lifetime_mean"])
            std  = float(row["lifetime_std"])
            factories[triage] = lambda m=mean, s=std: {"type": "normal", "mean": m, "std": s}
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    return factories


def save_casualties_df(
    casualties_df: pd.DataFrame,
    output_path: Union[str, Path] = None,
) -> Path:
    """Save casualties dataframe to XLSX.

    The dataframe is expected to carry raw lifetime parameter columns
    (lifetime_dist_type, lifetime_low, lifetime_high, lifetime_mode) rather
    than salabim distribution objects — these are written directly.
    """
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / "casualties.xlsx"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    casualties_df.to_excel(output_path, index=False)
    print(f"Saved casualties dataframe to: {output_path}")
    return output_path


# ============================================================
# STEP A) WAVE PLACER
# ============================================================

def place_waves_randomly_per_day(
    facility_ids: List[str],
    n_days: int,
    waves_per_day: int,
    wave_duration_min: int,
    seed: int = 123,
) -> WaveDict:
    """
    Places 'waves_per_day' non-overlapping waves of duration 'wave_duration_min'
    randomly within each day for each facility.

    Slot method:
      day split into S = floor(1440 / wave_duration_min) slots.
      For each (facility, day) sample 'waves_per_day' distinct slots.
    """
    rng = random.Random(seed)
    minutes_per_day = 24 * 60

    if wave_duration_min <= 0:
        raise ValueError("wave_duration_min must be > 0")
    if n_days <= 0:
        raise ValueError("n_days must be > 0")
    if waves_per_day <= 0:
        raise ValueError("waves_per_day must be > 0")
    if not facility_ids:
        raise ValueError("facility_ids must be a non-empty list")

    slots_per_day = minutes_per_day // wave_duration_min
    if waves_per_day > slots_per_day:
        raise ValueError(
            f"waves_per_day={waves_per_day} too large for wave_duration_min={wave_duration_min}. "
            f"Max is {slots_per_day}."
        )

    wave_dict: WaveDict = {fid: [] for fid in facility_ids}
    wave_counter = 0
    slot_indices = list(range(slots_per_day))

    for day in range(n_days):
        day_start = day * minutes_per_day
        for fid in facility_ids:
            chosen_slots = rng.sample(slot_indices, k=waves_per_day)
            chosen_slots.sort()

            for s in chosen_slots:
                t_start = day_start + s * wave_duration_min
                t_end = t_start + wave_duration_min
                wave_id = f"w_{wave_counter:06d}"
                wave_counter += 1

                wave_dict[fid].append(
                    Wave(
                        wave_id=wave_id,
                        t_start=float(t_start),
                        t_end=float(t_end),
                        facility_id=fid,
                    )
                )

    return wave_dict


# ============================================================
# STEP B) WAVE CASUALTY ALLOCATOR
# ============================================================

def allocate_casualties_to_waves(
    wave_dict: WaveDict,
    par_total: int,
    rate_per_day: float,
    facility_weights: Optional[Dict[str, float]] = None,
    seed: int = 123,
) -> None:
    """
    Mutates wave_dict in-place by setting wave.n_casualties.

    - par_total: global PAR (e.g., 18800)
    - rate_per_day: casualties per soldier per day
    - facility_weights: optional weights per facility_id. If None, equal weights.

    Daily total ~ Poisson(par_total * rate_per_day), split across facilities
    (weighted multinomial), then across that facility's waves (random multinomial).
    """
    if par_total < 0:
        raise ValueError("par_total must be >= 0")
    if rate_per_day < 0:
        raise ValueError("rate_per_day must be >= 0")

    rng = np.random.default_rng(seed)
    minutes_per_day = 24 * 60

    facility_ids = list(wave_dict.keys())
    if not facility_ids:
        return

    if facility_weights is None:
        weights = np.ones(len(facility_ids), dtype=float)
    else:
        weights = np.array([float(facility_weights.get(fid, 0.0)) for fid in facility_ids], dtype=float)
        if (weights < 0).any():
            raise ValueError("facility_weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("facility_weights sum must be > 0 (or pass None for equal split)")

    weights = _normalize_probs(weights)

    all_days = set()
    for fid in facility_ids:
        for w in wave_dict[fid]:
            all_days.add(int(w.t_start // minutes_per_day))
    if not all_days:
        return
    n_days = max(all_days) + 1

    for day_idx in range(n_days):
        lam_day = par_total * rate_per_day
        n_day_total = int(rng.poisson(lam=lam_day))

        if n_day_total <= 0:
            for fid in facility_ids:
                for w in wave_dict[fid]:
                    if int(w.t_start // minutes_per_day) == day_idx:
                        w.n_casualties = 0
            continue

        n_by_facility = rng.multinomial(n_day_total, weights)

        for fid, n_fac_day in zip(facility_ids, n_by_facility):
            day_waves = [w for w in wave_dict[fid] if int(w.t_start // minutes_per_day) == day_idx]
            if not day_waves:
                continue

            day_waves.sort(key=lambda ww: ww.t_start)
            k = len(day_waves)

            wave_weights = np.ones(k, dtype=float) / k
            counts = rng.multinomial(int(n_fac_day), wave_weights)
            for w, c in zip(day_waves, counts):
                w.n_casualties = int(c)


# ============================================================
# STEP C) WAVE FILLER
# ============================================================

def fill_wave_dataframes(
    wave_dict: WaveDict,
    seed: int = 123,
) -> None:
    rng = np.random.default_rng(seed)

    for fid, waves in wave_dict.items():
        for w in waves:
            n = int(w.n_casualties)
            if n <= 0:
                w.df = pd.DataFrame(columns=["t_create", "spawn_facility_id"])
                continue

            t = rng.uniform(low=w.t_start, high=w.t_end, size=n)
            w.df = (
                pd.DataFrame({"t_create": t.astype(float), "spawn_facility_id": [fid] * n})
                .sort_values("t_create")
                .reset_index(drop=True)
            )


# ============================================================
# STEP D) CASUALTY POPULATOR
# ============================================================

def populate_patient_attributes(
    wave_dict: WaveDict,
    diagnosis_mix_df: pd.DataFrame,
    rx_df: pd.DataFrame,
    lifetime_factory_by_triage: Optional[Dict[str, Callable[[], object]]] = None,
    seed: int = 123,
) -> None:
    rng = np.random.default_rng(seed)

    diag_labels = diagnosis_mix_df["triage"].astype(str).to_numpy()
    diag_p = _normalize_probs(diagnosis_mix_df["frequency"].astype(float).to_numpy())

    if lifetime_factory_by_triage is None:
        lifetime_factory_by_triage = {}
    default_lifetime_factory = lambda: {"type": "uniform", "low": 60.0, "high": 120.0}

    rx_df = rx_df.copy()
    rx_df["label"] = rx_df["label"].astype(str)

    rx_by_label: Dict[str, Tuple[np.ndarray, List[List[str]]]] = {}
    for label, g in rx_df.groupby("label"):
        presc_list = [list(x) for x in g["prescription"].tolist()]
        p = _normalize_probs(g["frequency"].astype(float).to_numpy())
        rx_by_label[label] = (p, presc_list)

    global_presc_list = [list(x) for x in rx_df["prescription"].tolist()]
    global_p = _normalize_probs(rx_df["frequency"].astype(float).to_numpy())

    for _, waves in wave_dict.items():
        for w in waves:
            if w.df is None or len(w.df) == 0:
                if w.df is None:
                    w.df = pd.DataFrame(columns=["t_create", "spawn_facility_id"])
                w.df["triage"]             = pd.Series(dtype=str)
                w.df["prescription"]       = pd.Series(dtype=object)
                w.df["lifetime_dist_type"] = pd.Series(dtype=str)
                w.df["lifetime_low"]       = pd.Series(dtype=float)
                w.df["lifetime_high"]      = pd.Series(dtype=float)
                w.df["lifetime_mode"]      = pd.Series(dtype=float)
                continue

            n = len(w.df)
            triage = rng.choice(diag_labels, size=n, p=diag_p).astype(str)

            prescriptions:      List[List[str]] = []
            lf_types:           List[str]       = []
            lf_lows:            List[float]     = []
            lf_highs:           List[float]     = []
            lf_modes:           List[object]    = []

            for tr in triage:
                if tr in rx_by_label:
                    p, presc_list = rx_by_label[tr]
                    idx = int(rng.choice(np.arange(len(presc_list)), p=p))
                    prescriptions.append(list(presc_list[idx]))
                else:
                    idx = int(rng.choice(np.arange(len(global_presc_list)), p=global_p))
                    prescriptions.append(list(global_presc_list[idx]))

                lf_factory = lifetime_factory_by_triage.get(tr, default_lifetime_factory)
                params = lf_factory()
                lf_types.append(params["type"])
                lf_lows.append(params.get("low", params.get("mean")))
                lf_highs.append(params.get("high", params.get("std")))
                lf_modes.append(params.get("mode"))

            w.df = w.df.copy()
            w.df["triage"]            = triage
            w.df["prescription"]      = prescriptions
            w.df["lifetime_dist_type"] = lf_types
            w.df["lifetime_low"]       = lf_lows
            w.df["lifetime_high"]      = lf_highs
            w.df["lifetime_mode"]      = lf_modes


# ============================================================
# STEP E) MERGE WAVES -> FINAL casualties_df
# ============================================================

def merge_waves_to_casualties_df(*wave_dicts: WaveDict) -> pd.DataFrame:
    frames = []
    for wave_dict in wave_dicts:
        for waves in wave_dict.values():
            for w in waves:
                if w.df is not None and len(w.df) > 0:
                    frames.append(w.df)

    if not frames:
        return pd.DataFrame(columns=[
            "t_create", "spawn_facility_id", "triage", "prescription",
            "lifetime_dist_type", "lifetime_low", "lifetime_high", "lifetime_mode",
        ])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("t_create").reset_index(drop=True)


# ============================================================
# STREAMS-BASED ORCHESTRATOR  (primary interface)
#
# STREAMS is a list of dicts, each with:
#   - facility_ids : List[str]   — spawn locations for this stream
#   - rate         : float       — casualties per soldier per day
#   - mix_df       : DataFrame   — triage labels, frequencies, lifetime params
#
# par_total    : total PAR across all involved units
# n_days, waves_per_day, wave_duration_min : wave schedule
# rx_df        : prescriptions DataFrame
# seed         : base RNG seed
# ============================================================

def generate_casualties_df_from_streams(
    streams: List[Dict],
    par_total: int,
    n_days: int,
    waves_per_day: int,
    wave_duration_min: int,
    rx_df: pd.DataFrame,
    seed: int = 123,
) -> Tuple[pd.DataFrame, List[WaveDict]]:
    """
    Primary orchestrator matching the notebook's STREAMS approach.

    Each stream dict must have keys: facility_ids, rate, mix_df.
    Returns (casualties_df, list_of_wave_dicts).
    """
    all_factories: Dict[str, Callable] = {}
    for stream in streams:
        all_factories.update(_lifetime_df_to_factories(stream["mix_df"]))

    all_wave_dicts: List[WaveDict] = []

    for i, stream in enumerate(streams):
        base_seed = seed + i * 10

        wdict = place_waves_randomly_per_day(
            facility_ids=stream["facility_ids"],
            n_days=n_days,
            waves_per_day=waves_per_day,
            wave_duration_min=wave_duration_min,
            seed=base_seed,
        )
        allocate_casualties_to_waves(
            wave_dict=wdict,
            par_total=par_total,
            rate_per_day=stream["rate"],
            seed=base_seed + 1,
        )
        fill_wave_dataframes(wdict, seed=base_seed + 2)
        populate_patient_attributes(
            wave_dict=wdict,
            diagnosis_mix_df=stream["mix_df"],
            rx_df=rx_df,
            lifetime_factory_by_triage=all_factories,
            seed=base_seed + 3,
        )
        all_wave_dicts.append(wdict)

    casualties_df = merge_waves_to_casualties_df(*all_wave_dicts)
    return casualties_df, all_wave_dicts


# ============================================================
# SPAWN-MAP ORCHESTRATOR  (legacy — wraps generate_casualties_df_from_streams)
#
# spawn_map: {facility_id: {"labels": [...], "rate": float}}
#
# Facilities sharing the same (labels, rate) are automatically
# grouped into one stream.
# ============================================================

def generate_casualties_df_from_waves(
    spawn_map: Dict[str, Dict],
    par_total: int,
    n_days: int,
    waves_per_day: int,
    wave_duration_min: int,
    diagnosis_mix_df: pd.DataFrame,
    rx_df: pd.DataFrame,
    seed: int = 123,
) -> Tuple[pd.DataFrame, List[WaveDict]]:
    """
    Returns (casualties_df, list_of_wave_dicts), one WaveDict per stream.

    spawn_map example:
        {
            "ccp1": {"labels": ["T1_BCAS", "T2_BCAS", "T3_BCAS", "KIA", "MIA"], "rate": 0.023},
            "ccp2": {"labels": ["T1_BCAS", "T2_BCAS", "T3_BCAS", "KIA", "MIA"], "rate": 0.023},
            "r1_bde1": {"labels": ["T1_DNBI", "T2_DNBI", "T3_DNBI"], "rate": 0.0013},
        }
    """
    # Group facilities that share the same (labels, rate) into one stream,
    # then delegate to generate_casualties_df_from_streams.
    grouped: Dict[Tuple, Dict] = {}
    for fid, cfg in spawn_map.items():
        key = (tuple(sorted(cfg["labels"])), float(cfg["rate"]))
        if key not in grouped:
            grouped[key] = {
                "facility_ids": [],
                "labels": list(cfg["labels"]),
                "rate": float(cfg["rate"]),
            }
        grouped[key]["facility_ids"].append(fid)

    streams_list = []
    for group in grouped.values():
        sub_mix = diagnosis_mix_df[
            diagnosis_mix_df["triage"].isin(group["labels"])
        ].copy()
        sub_mix = sub_mix.copy()
        sub_mix["frequency"] = sub_mix["frequency"] / sub_mix["frequency"].sum()
        streams_list.append({
            "facility_ids": group["facility_ids"],
            "rate":         group["rate"],
            "mix_df":       sub_mix,
        })

    return generate_casualties_df_from_streams(
        streams=streams_list,
        par_total=par_total,
        n_days=n_days,
        waves_per_day=waves_per_day,
        wave_duration_min=wave_duration_min,
        rx_df=rx_df,
        seed=seed,
    )


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    SAVE_OUTPUT = True
    OUTPUT_PATH = "../../data/single_run/input/casualty_dataframes/casualties.xlsx"

    print(f"Loading config from: {DEFAULT_CONFIG_DIR}")
    config = load_casualty_generation_config(DEFAULT_CONFIG_DIR)

    casualties_df, wave_dicts = generate_casualties_df_from_waves(
        spawn_map=config["spawn_map"],
        par_total=config["par_total"],
        n_days=config["n_days"],
        waves_per_day=config["waves_per_day"],
        wave_duration_min=config["wave_duration_min"],
        diagnosis_mix_df=config["diagnosis_mix_df"],
        rx_df=config["rx_df"],
        seed=config["seed"],
    )

    simulation_input_df = casualties_df[
        ~casualties_df["triage"].str.startswith(("KIA", "MIA"))
    ].copy().reset_index(drop=True)

    if SAVE_OUTPUT:
        save_casualties_df(simulation_input_df, OUTPUT_PATH)

    print("\n=== Triage breakdown (all generated) ===")
    print(casualties_df["triage"].value_counts().sort_index().to_string())
    print(f"\nTotal generated:       {len(casualties_df)}")
    print(f"  of which KIA:        {(casualties_df['triage'] == 'KIA').sum()}")
    print(f"  of which MIA:        {(casualties_df['triage'] == 'MIA').sum()}")
    print(f"\nSimulation input:      {len(simulation_input_df)}")
    print(f"\nSpawn locations:")
    print(simulation_input_df["spawn_facility_id"].value_counts().sort_index().to_string())
    print(f"\nFirst 10 rows:")
    print(simulation_input_df.head(10))
