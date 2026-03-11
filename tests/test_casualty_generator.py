"""Tests for casualty_generator.py"""

import pytest
import numpy as np
import pandas as pd
import salabim as sim

from src.helpers.casualty_generator import (
    _normalize_probs,
    _lifetime_df_to_factories,
    place_waves_randomly_per_day,
    allocate_casualties_to_waves,
    fill_wave_dataframes,
    populate_patient_attributes,
    merge_waves_to_casualties_df,
    generate_casualties_df_from_waves,
    Wave,
)


class TestNormalizeProbs:
    """Tests for the _normalize_probs helper function."""

    def test_basic_normalization(self):
        """Verify that an array of integers is correctly normalized to sum to 1.0."""
        p = np.array([1, 2, 3])
        result = _normalize_probs(p)
        assert np.isclose(result.sum(), 1.0)
        assert np.allclose(result, [1/6, 2/6, 3/6])

    def test_already_normalized(self):
        """Verify that an already-normalized array remains unchanged."""
        p = np.array([0.5, 0.3, 0.2])
        result = _normalize_probs(p)
        assert np.isclose(result.sum(), 1.0)
        assert np.allclose(result, [0.5, 0.3, 0.2])

    def test_zero_sum_raises(self):
        """Verify that an all-zeros array raises ValueError."""
        p = np.array([0, 0, 0])
        with pytest.raises(ValueError, match="must sum to > 0"):
            _normalize_probs(p)

    def test_negative_sum_raises(self):
        """Verify that an array with negative sum raises ValueError."""
        p = np.array([-1, -2, -3])
        with pytest.raises(ValueError, match="must sum to > 0"):
            _normalize_probs(p)


class TestLifetimeDfToFactories:
    """Tests for the _lifetime_df_to_factories function that creates distribution factories."""

    def test_uniform_distribution(self):
        """Verify that uniform distributions are correctly parsed and factory functions work."""
        df = pd.DataFrame({
            "triage": ["T1", "T2"],
            "lifetime_distribution": ["uniform", "uniform"],
            "lifetime_low": [10, 20],
            "lifetime_high": [30, 40],
        })
        factories = _lifetime_df_to_factories(df)

        assert "T1" in factories
        assert "T2" in factories

        # Check that factories create valid distributions
        dist_t1 = factories["T1"]()
        assert isinstance(dist_t1, sim.Uniform)

    def test_unknown_distribution_raises(self):
        """Verify that an unknown distribution type raises ValueError."""
        df = pd.DataFrame({
            "triage": ["T1"],
            "lifetime_distribution": ["unknown_dist"],
            "lifetime_low": [10],
            "lifetime_high": [30],
        })
        with pytest.raises(ValueError, match="Unknown distribution type"):
            _lifetime_df_to_factories(df)


class TestPlaceWavesRandomlyPerDay:
    """Tests for the place_waves_randomly_per_day function that schedules casualty waves."""

    def test_basic_wave_placement(self):
        """Verify correct number of waves are created for each facility and day."""
        wave_dict = place_waves_randomly_per_day(
            facility_ids=["F1", "F2"],
            n_days=2,
            waves_per_day=3,
            wave_duration_min=60,
            seed=42,
        )

        assert "F1" in wave_dict
        assert "F2" in wave_dict
        # 2 days * 3 waves = 6 waves per facility
        assert len(wave_dict["F1"]) == 6
        assert len(wave_dict["F2"]) == 6

    def test_wave_no_overlap(self):
        """Verify that waves within the same facility do not overlap in time."""
        wave_dict = place_waves_randomly_per_day(
            facility_ids=["F1"],
            n_days=1,
            waves_per_day=5,
            wave_duration_min=60,
            seed=42,
        )

        waves = wave_dict["F1"]
        # Sort by start time
        waves_sorted = sorted(waves, key=lambda w: w.t_start)

        # Check no overlap
        for i in range(len(waves_sorted) - 1):
            assert waves_sorted[i].t_end <= waves_sorted[i + 1].t_start

    def test_wave_times_within_day(self):
        """Verify that each wave's start and end times fall within its respective day."""
        wave_dict = place_waves_randomly_per_day(
            facility_ids=["F1"],
            n_days=2,
            waves_per_day=2,
            wave_duration_min=30,
            seed=42,
        )

        for wave in wave_dict["F1"]:
            day = int(wave.t_start // (24 * 60))
            day_start = day * 24 * 60
            day_end = (day + 1) * 24 * 60
            assert wave.t_start >= day_start
            assert wave.t_end <= day_end

    def test_invalid_wave_duration_raises(self):
        """Verify that zero or negative wave duration raises ValueError."""
        with pytest.raises(ValueError):
            place_waves_randomly_per_day(
                facility_ids=["F1"],
                n_days=1,
                waves_per_day=1,
                wave_duration_min=0,
            )

    def test_too_many_waves_raises(self):
        """Verify that requesting more waves than can fit in a day raises ValueError."""
        # 24 hours = 1440 minutes, with 60 min waves = max 24 waves
        with pytest.raises(ValueError):
            place_waves_randomly_per_day(
                facility_ids=["F1"],
                n_days=1,
                waves_per_day=25,
                wave_duration_min=60,
            )


class TestAllocateCasualtiestoWaves:
    """Tests for the allocate_casualties_to_waves function that distributes casualties."""

    def test_casualties_distributed(self):
        """Verify that casualties are allocated to waves based on PAR and rate."""
        wave_dict = place_waves_randomly_per_day(
            facility_ids=["F1"],
            n_days=1,
            waves_per_day=3,
            wave_duration_min=60,
            seed=42,
        )

        allocate_casualties_to_waves(
            wave_dict=wave_dict,
            par_total=1000,
            rate_per_day=0.1,  # ~100 casualties per day
            seed=42,
        )

        total_casualties = sum(w.n_casualties for w in wave_dict["F1"])
        # Should be roughly 100, but stochastic
        assert total_casualties > 0

    def test_facility_weights(self):
        """Verify that facility weights correctly bias casualty distribution."""
        wave_dict = place_waves_randomly_per_day(
            facility_ids=["F1", "F2"],
            n_days=5,
            waves_per_day=3,
            wave_duration_min=60,
            seed=42,
        )

        # F1 gets 90%, F2 gets 10%
        allocate_casualties_to_waves(
            wave_dict=wave_dict,
            par_total=10000,
            rate_per_day=0.1,
            facility_weights={"F1": 0.9, "F2": 0.1},
            seed=42,
        )

        f1_total = sum(w.n_casualties for w in wave_dict["F1"])
        f2_total = sum(w.n_casualties for w in wave_dict["F2"])

        # F1 should have significantly more
        assert f1_total > f2_total * 2


class TestFillWaveDataframes:
    """Tests for the fill_wave_dataframes function that creates casualty records."""

    def test_dataframes_created(self):
        """Verify that DataFrames are created with correct columns and row count."""
        wave_dict = {"F1": [Wave(wave_id="w1", t_start=0, t_end=60, facility_id="F1", n_casualties=5)]}

        fill_wave_dataframes(wave_dict, seed=42)

        df = wave_dict["F1"][0].df
        assert df is not None
        assert len(df) == 5
        assert "t_create" in df.columns
        assert "spawn_facility_id" in df.columns

    def test_times_within_wave(self):
        """Verify that all casualty creation times fall within the wave's time window."""
        wave_dict = {"F1": [Wave(wave_id="w1", t_start=100, t_end=200, facility_id="F1", n_casualties=10)]}

        fill_wave_dataframes(wave_dict, seed=42)

        df = wave_dict["F1"][0].df
        assert (df["t_create"] >= 100).all()
        assert (df["t_create"] <= 200).all()

    def test_empty_wave(self):
        """Verify that a wave with zero casualties produces an empty DataFrame."""
        wave_dict = {"F1": [Wave(wave_id="w1", t_start=0, t_end=60, facility_id="F1", n_casualties=0)]}

        fill_wave_dataframes(wave_dict, seed=42)

        df = wave_dict["F1"][0].df
        assert len(df) == 0


class TestPopulatePatientAttributes:
    """Tests for the populate_patient_attributes function that assigns triage and prescriptions."""

    @pytest.fixture
    def sample_wave_dict(self):
        """Create a sample wave dict with 10 casualties for testing."""
        wave_dict = {"F1": [Wave(wave_id="w1", t_start=0, t_end=60, facility_id="F1", n_casualties=10)]}
        fill_wave_dataframes(wave_dict, seed=42)
        return wave_dict

    @pytest.fixture
    def sample_diagnosis_mix_df(self):
        """Create a sample diagnosis mix DataFrame with two triage categories."""
        return pd.DataFrame({
            "triage": ["T1", "T2"],
            "frequency": [0.5, 0.5],
            "lifetime_distribution": ["uniform", "uniform"],
            "lifetime_low": [10, 20],
            "lifetime_high": [30, 40],
        })

    @pytest.fixture
    def sample_rx_df(self):
        """Create a sample prescriptions DataFrame with one prescription per triage."""
        return pd.DataFrame({
            "label": ["T1", "T2"],
            "frequency": [1.0, 1.0],
            "prescription": [["DCR"], ["PEC"]],
        })

    def test_attributes_added(self, sample_wave_dict, sample_diagnosis_mix_df, sample_rx_df):
        """Verify that triage, prescription, and lifetime_dist columns are added."""
        populate_patient_attributes(
            wave_dict=sample_wave_dict,
            diagnosis_mix_df=sample_diagnosis_mix_df,
            rx_df=sample_rx_df,
            seed=42,
        )

        df = sample_wave_dict["F1"][0].df
        assert "triage" in df.columns
        assert "prescription" in df.columns
        assert "lifetime_dist" in df.columns

    def test_triage_values_valid(self, sample_wave_dict, sample_diagnosis_mix_df, sample_rx_df):
        """Verify that all assigned triage values come from the diagnosis mix."""
        populate_patient_attributes(
            wave_dict=sample_wave_dict,
            diagnosis_mix_df=sample_diagnosis_mix_df,
            rx_df=sample_rx_df,
            seed=42,
        )

        df = sample_wave_dict["F1"][0].df
        valid_triage = set(sample_diagnosis_mix_df["triage"])
        assert set(df["triage"].unique()).issubset(valid_triage)


class TestMergeWavesToCasualtiesDf:
    """Tests for the merge_waves_to_casualties_df function that combines all waves."""

    def test_merge_multiple_waves(self):
        """Verify that casualties from multiple waves and facilities are merged and sorted."""
        wave_dict = {
            "F1": [
                Wave(wave_id="w1", t_start=0, t_end=60, facility_id="F1", n_casualties=5),
                Wave(wave_id="w2", t_start=100, t_end=160, facility_id="F1", n_casualties=3),
            ],
            "F2": [
                Wave(wave_id="w3", t_start=50, t_end=110, facility_id="F2", n_casualties=4),
            ],
        }
        fill_wave_dataframes(wave_dict, seed=42)

        df = merge_waves_to_casualties_df(wave_dict)

        assert len(df) == 12  # 5 + 3 + 4
        # Should be sorted by t_create
        assert (df["t_create"].diff().dropna() >= 0).all()

    def test_empty_waves(self):
        """Verify that merging waves with zero casualties produces empty DataFrame."""
        wave_dict = {"F1": [Wave(wave_id="w1", t_start=0, t_end=60, facility_id="F1", n_casualties=0)]}
        fill_wave_dataframes(wave_dict, seed=42)

        df = merge_waves_to_casualties_df(wave_dict)

        assert len(df) == 0


class TestGenerateCasualtiesDfFromWaves:
    """Tests for the generate_casualties_df_from_waves orchestrator function."""

    @pytest.fixture
    def sample_diagnosis_mix_df(self):
        """Create a sample diagnosis mix DataFrame for end-to-end testing."""
        return pd.DataFrame({
            "triage": ["T1", "T2"],
            "frequency": [0.5, 0.5],
            "lifetime_distribution": ["uniform", "uniform"],
            "lifetime_low": [10, 20],
            "lifetime_high": [30, 40],
        })

    @pytest.fixture
    def sample_rx_df(self):
        """Create a sample prescriptions DataFrame for end-to-end testing."""
        return pd.DataFrame({
            "label": ["T1", "T2"],
            "frequency": [1.0, 1.0],
            "prescription": [["DCR"], ["PEC"]],
        })

    def test_end_to_end_generation(self, sample_diagnosis_mix_df, sample_rx_df):
        """Verify that the full pipeline produces a valid casualties DataFrame."""
        casualties_df, wave_dict = generate_casualties_df_from_waves(
            facility_ids=["F1", "F2"],
            par_total=1000,
            n_days=2,
            waves_per_day=3,
            wave_duration_min=60,
            rate_per_day=0.1,
            diagnosis_mix_df=sample_diagnosis_mix_df,
            rx_df=sample_rx_df,
            seed=42,
        )

        assert len(casualties_df) > 0
        assert "t_create" in casualties_df.columns
        assert "spawn_facility_id" in casualties_df.columns
        assert "triage" in casualties_df.columns
        assert "prescription" in casualties_df.columns
        assert "lifetime_dist" in casualties_df.columns

    def test_reproducibility(self, sample_diagnosis_mix_df, sample_rx_df):
        """Verify that using the same seed produces identical results."""
        kwargs = {
            "facility_ids": ["F1"],
            "par_total": 1000,
            "n_days": 2,
            "waves_per_day": 3,
            "wave_duration_min": 60,
            "rate_per_day": 0.1,
            "diagnosis_mix_df": sample_diagnosis_mix_df,
            "rx_df": sample_rx_df,
            "seed": 42,
        }

        df1, _ = generate_casualties_df_from_waves(**kwargs)
        df2, _ = generate_casualties_df_from_waves(**kwargs)

        assert len(df1) == len(df2)
        assert (df1["t_create"].values == df2["t_create"].values).all()
