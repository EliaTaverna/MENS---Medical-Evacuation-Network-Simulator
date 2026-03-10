from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple, Union
import random

import numpy as np
import pandas as pd
import salabim as sim
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
DEFAULT_CONFIG_DIR = _THIS_DIR / "data" / "casualty_generation"
DEFAULT_OUTPUT_DIR = _THIS_DIR / "data" / "simulation_initiation"


def load_casualty_generation_config(
    config_dir: Union[str, Path] = DEFAULT_CONFIG_DIR,
) -> dict:
    """
    Load casualty generation configuration from YAML + XLSX files.

    Expected files in config_dir:
        - config.yaml: parameters (par_total, n_days, facility_ids, etc.)
        - diagnosis_mix.xlsx: triage, frequency, lifetime_distribution, lifetime_low, lifetime_high
        - prescriptions.xlsx: label, frequency, prescription
    """
    config_dir = Path(config_dir)

    # Load parameters from YAML
    with open(config_dir / "config.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    # Facility IDs from YAML
    facility_ids = params["facility_ids"]
    facility_weights = None  # equal distribution by default

    # Load diagnosis mix (includes lifetime info)
    diagnosis_mix_df = pd.read_excel(config_dir / "diagnosis_mix.xlsx")

    # Load prescriptions
    rx_df = pd.read_excel(config_dir / "prescriptions.xlsx")
    rx_df["prescription"] = rx_df["prescription"].apply(eval)

    # Extract lifetime distributions from diagnosis_mix_df
    lifetime_factory_by_triage = _lifetime_df_to_factories(diagnosis_mix_df)

    return {
        "facility_ids": facility_ids,
        "facility_weights": facility_weights,
        "par_total": params["par_total"],
        "n_days": params["n_days"],
        "waves_per_day": params["waves_per_day"],
        "wave_duration_min": params["wave_duration_min"],
        "rate_per_day": params["rate_per_day"],
        "seed": params.get("seed", 42),
        "diagnosis_mix_df": diagnosis_mix_df,
        "rx_df": rx_df,
        "lifetime_factory_by_triage": lifetime_factory_by_triage,
    }


def _lifetime_df_to_factories(diagnosis_mix_df: pd.DataFrame) -> Dict[str, Callable[[], object]]:
    """Convert diagnosis_mix dataframe to lifetime factory functions."""
    factories = {}

    for _, row in diagnosis_mix_df.iterrows():
        triage = row["triage"]
        dist_type = row["lifetime_distribution"].lower()

        if dist_type == "uniform":
            low, high = row["lifetime_low"], row["lifetime_high"]
            factories[triage] = lambda l=low, h=high: sim.Uniform(l, h)
        elif dist_type == "triangular":
            low = row["lifetime_low"]
            high = row["lifetime_high"]
            mode = row.get("lifetime_mode", (low + high) / 2)
            if pd.isna(mode):
                mode = (low + high) / 2
            factories[triage] = lambda l=low, m=mode, h=high: sim.Triangular(l, h, m)
        elif dist_type == "exponential":
            mean = row["lifetime_mean"]
            factories[triage] = lambda m=mean: sim.Exponential(m)
        elif dist_type == "normal":
            mean, std = row["lifetime_mean"], row["lifetime_std"]
            factories[triage] = lambda m=mean, s=std: sim.Normal(m, s)
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")

    return factories


def save_casualties_df(
    casualties_df: pd.DataFrame,
    output_path: Union[str, Path] = None,
) -> Path:
    """Save casualties dataframe to XLSX, extracting lifetime bounds."""
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / "casualties.xlsx"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_df = casualties_df.copy()

    def get_dist_type(d):
        cls_name = type(d).__name__.lower()
        if "triangular" in cls_name:
            return "triangular"
        elif "uniform" in cls_name:
            return "uniform"
        elif "exponential" in cls_name:
            return "exponential"
        elif "normal" in cls_name:
            return "normal"
        return "uniform"

    def get_low(d):
        for attr in ["_lowerbound", "low", "a", "min"]:
            if hasattr(d, attr):
                return getattr(d, attr)
        return None

    def get_high(d):
        for attr in ["_upperbound", "high", "b", "max"]:
            if hasattr(d, attr):
                return getattr(d, attr)
        return None

    def get_mode(d):
        for attr in ["_mode", "mode", "c"]:
            if hasattr(d, attr):
                return getattr(d, attr)
        return None

    save_df["lifetime_dist_type"] = save_df["lifetime_dist"].apply(get_dist_type)
    save_df["lifetime_low"] = save_df["lifetime_dist"].apply(get_low)
    save_df["lifetime_high"] = save_df["lifetime_dist"].apply(get_high)
    save_df["lifetime_mode"] = save_df["lifetime_dist"].apply(get_mode)

    save_cols = [c for c in save_df.columns if c != "lifetime_dist"]
    save_df[save_cols].to_excel(output_path, index=False)
    print(f"Saved casualties dataframe to: {output_path}")
    return output_path


# ============================================================
# STEP A) WAVE PLACER
#   - creates wave_dict with Wave objects (no casualties yet)
#   - random placement per day per facility
#   - NO overlap by construction using random slot selection
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
# STEP B) WAVE CASUALTY CALCULATOR (FIXED PAR LOGIC)
#
# You pass:
#   - par_total (e.g., 18800)
#   - rate_per_day (e.g., 0.0243/day)
#
# For each day:
#   N_day_total ~ Poisson(par_total * rate_per_day)
# Then distribute N_day_total across facilities (weighted), then across waves of that facility that day.
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
    - rate_per_day: casualties per soldier per day (e.g., 0.023 + 0.0013 = 0.0243)
    - facility_weights: optional weights per facility_id. If None, equal weights.
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

    # weights over facilities
    if facility_weights is None:
        weights = np.ones(len(facility_ids), dtype=float)
    else:
        weights = np.array([float(facility_weights.get(fid, 0.0)) for fid in facility_ids], dtype=float)
        if (weights < 0).any():
            raise ValueError("facility_weights must be non-negative")
        if float(weights.sum()) <= 0:
            raise ValueError("facility_weights sum must be > 0 (or pass None for equal split)")

    weights = _normalize_probs(weights)

    # Determine how many days exist from the waves (robust even if called with mismatch)
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
            # set all waves this day to 0
            for fid in facility_ids:
                for w in wave_dict[fid]:
                    if int(w.t_start // minutes_per_day) == day_idx:
                        w.n_casualties = 0
            continue

        # split across facilities via multinomial
        n_by_facility = rng.multinomial(n_day_total, weights)

        # now split each facility's daily total across that facility's waves in that day
        for fid, n_fac_day in zip(facility_ids, n_by_facility):
            day_waves = [w for w in wave_dict[fid] if int(w.t_start // minutes_per_day) == day_idx]
            if not day_waves:
                continue

            day_waves.sort(key=lambda ww: ww.t_start)
            k = len(day_waves)

            base = int(n_fac_day) // k
            rem = int(n_fac_day) % k

            for i, w in enumerate(day_waves):
                w.n_casualties = base + (1 if i < rem else 0)


# ============================================================
# STEP C) WAVE FILLER
#   - creates wave.df with ONLY: t_create, spawn_facility_id
#   - inside each wave: conditional on n, times ~ Uniform(t_start, t_end)
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
#   Adds: triage, prescription, lifetime_dist
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
    default_lifetime_factory = lambda: sim.Uniform(60, 120)

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
                w.df["triage"] = pd.Series(dtype=str)
                w.df["prescription"] = pd.Series(dtype=object)
                w.df["lifetime_dist"] = pd.Series(dtype=object)
                continue

            n = len(w.df)
            triage = rng.choice(diag_labels, size=n, p=diag_p).astype(str)

            prescriptions: List[List[str]] = []
            lifetime_dists: List[object] = []

            for tr in triage:
                if tr in rx_by_label:
                    p, presc_list = rx_by_label[tr]
                    idx = int(rng.choice(np.arange(len(presc_list)), p=p))
                    prescriptions.append(list(presc_list[idx]))
                else:
                    idx = int(rng.choice(np.arange(len(global_presc_list)), p=global_p))
                    prescriptions.append(list(global_presc_list[idx]))

                lf_factory = lifetime_factory_by_triage.get(tr, default_lifetime_factory)
                lifetime_dists.append(lf_factory())

            w.df = w.df.copy()
            w.df["triage"] = triage
            w.df["prescription"] = prescriptions
            w.df["lifetime_dist"] = lifetime_dists


# ============================================================
# STEP E) MERGE WAVES -> FINAL casualties_df
# ============================================================

def merge_waves_to_casualties_df(wave_dict: WaveDict) -> pd.DataFrame:
    frames = []
    for waves in wave_dict.values():
        for w in waves:
            if w.df is not None and len(w.df) > 0:
                frames.append(w.df)

    if not frames:
        return pd.DataFrame(columns=["t_create", "spawn_facility_id", "triage", "prescription", "lifetime_dist"])

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("t_create").reset_index(drop=True)


# ============================================================
# ORCHESTRATOR
# ============================================================

def generate_casualties_df_from_waves(
    facility_ids: List[str],
    par_total: int,
    n_days: int,
    waves_per_day: int,
    wave_duration_min: int,
    rate_per_day: float,
    diagnosis_mix_df: pd.DataFrame,
    rx_df: pd.DataFrame,
    facility_weights: Optional[Dict[str, float]] = None,
    lifetime_factory_by_triage: Optional[Dict[str, Callable[[], object]]] = None,
    seed: int = 123,
) -> Tuple[pd.DataFrame, WaveDict]:
    wave_dict = place_waves_randomly_per_day(
        facility_ids=facility_ids,
        n_days=n_days,
        waves_per_day=waves_per_day,
        wave_duration_min=wave_duration_min,
        seed=seed,
    )

    allocate_casualties_to_waves(
        wave_dict=wave_dict,
        par_total=par_total,
        rate_per_day=rate_per_day,
        facility_weights=facility_weights,
        seed=seed + 1,
    )

    fill_wave_dataframes(wave_dict=wave_dict, seed=seed + 2)

    populate_patient_attributes(
        wave_dict=wave_dict,
        diagnosis_mix_df=diagnosis_mix_df,
        rx_df=rx_df,
        lifetime_factory_by_triage=lifetime_factory_by_triage,
        seed=seed + 3,
    )

    casualties_df = merge_waves_to_casualties_df(wave_dict)
    return casualties_df, wave_dict


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":

    SAVE_OUTPUT = True
    OUTPUT_PATH = "data/simulation_initiation/casualties.xlsx"

    # Load configuration from files
    print(f"Loading config from: {DEFAULT_CONFIG_DIR}")
    config = load_casualty_generation_config(DEFAULT_CONFIG_DIR)

    # Generate casualties
    casualties_df, wave_dict = generate_casualties_df_from_waves(
        facility_ids=config["facility_ids"],
        par_total=config["par_total"],
        n_days=config["n_days"],
        waves_per_day=config["waves_per_day"],
        wave_duration_min=config["wave_duration_min"],
        rate_per_day=config["rate_per_day"],
        diagnosis_mix_df=config["diagnosis_mix_df"],
        rx_df=config["rx_df"],
        facility_weights=config["facility_weights"],
        lifetime_factory_by_triage=config["lifetime_factory_by_triage"],
        seed=config["seed"],
    )

    # Save if enabled
    if SAVE_OUTPUT:
        save_casualties_df(casualties_df, OUTPUT_PATH)

    # Print summary
    print("\ncasualties_df head:")
    print(casualties_df.head(10))
    print(f"\nTotal casualties: {len(casualties_df)}")

    print("\nPer-facility counts:")
    print(casualties_df["spawn_facility_id"].value_counts())