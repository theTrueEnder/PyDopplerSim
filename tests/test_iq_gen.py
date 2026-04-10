"""
Tests for iq_gen module: generate_iq, save_iq_wav, load_iq_wav.
"""

import numpy as np
import pytest
from iq_gen import generate_iq, save_iq_wav, load_iq_wav
import wave


class TestGenerateIQ:
    """Test suite for generate_iq function."""

    def test_output_dict_has_expected_keys(self, scenario_oncoming_cfg):
        """Test that generate_iq returns dict with all expected keys."""
        result = generate_iq(scenario_oncoming_cfg)

        expected_keys = [
            "t",
            "iq",
            "iq_clean",
            "cfg",
            "r",
            "r_dot",
            "r_ddot",
            "delta_f",
            "tx_x",
            "rx_x",
            "tx_y",
            "rx_y",
            "los_angle",
        ]

        for key in expected_keys:
            assert key in result, f"Missing expected key: {key}"

    def test_iq_length_equals_duration_times_fs(self, scenario_oncoming_cfg):
        """Test that IQ length matches duration * fs."""
        cfg = scenario_oncoming_cfg

        result = generate_iq(cfg)

        expected_length = int(cfg.duration * cfg.fs)
        assert len(result["iq"]) == expected_length
        assert len(result["iq_clean"]) == expected_length
        assert len(result["t"]) == expected_length

    def test_signal_power_is_unit_amplitude(self, scenario_oncoming_cfg, rng):
        """Test that clean IQ signal has unit amplitude (power = 1)."""
        cfg = scenario_oncoming_cfg
        cfg.snr_db = 100  # Very high SNR to minimize noise

        result = generate_iq(cfg, path_provider=None)

        # Power of unit amplitude complex signal = |z|^2 = 1
        power_clean = np.mean(np.abs(result["iq_clean"]) ** 2)
        np.testing.assert_allclose(power_clean, 1.0, rtol=1e-10)

    def test_iq_time_array_linear(self, scenario_oncoming_cfg):
        """Test that time array is linearly spaced."""
        result = generate_iq(scenario_oncoming_cfg)

        t = result["t"]
        dt = t[1] - t[0]

        # All intervals should be approximately equal
        for i in range(1, len(t)):
            np.testing.assert_allclose(t[i] - t[i - 1], dt, rtol=1e-10)

    def test_different_scenarios(self, any_scenario_cfg):
        """Test generate_iq works with all scenario configs."""
        result = generate_iq(any_scenario_cfg)

        assert len(result["iq"]) > 0
        assert len(result["t"]) > 0
        assert result["iq"].dtype == np.complex128

    @pytest.mark.parametrize("snr_db", [10.0, 20.0, 30.0])
    def test_different_snr_values(self, scenario_oncoming_cfg, snr_db):
        """Test generate_iq with different SNR values."""
        cfg = scenario_oncoming_cfg
        cfg.snr_db = snr_db

        result = generate_iq(cfg)

        # Noisy IQ should have more power than clean
        power_clean = np.mean(np.abs(result["iq_clean"]) ** 2)
        power_noisy = np.mean(np.abs(result["iq"]) ** 2)

        # Clean should always have power = 1
        np.testing.assert_allclose(power_clean, 1.0, rtol=1e-10)

        # Noisy should have more total power (signal + noise)
        assert power_noisy > power_clean


class TestSaveLoadIQWav:
    """Test suite for save_iq_wav and load_iq_wav functions."""

    def test_save_iq_wav_creates_valid_wav_file(
        self, scenario_oncoming_cfg, temp_wav_dir
    ):
        """Test that save_iq_wav creates a valid WAV file."""
        cfg = scenario_oncoming_cfg
        result = generate_iq(cfg)

        wav_path = temp_wav_dir / "test_signal.wav"

        save_iq_wav(result["iq"], result["t"], wav_path, cfg)

        assert wav_path.exists()

        # Validate WAV file structure
        with wave.open(str(wav_path), "r") as wav:
            assert wav.getnchannels() == 2  # I and Q
            assert wav.getsampwidth() == 2  # 16-bit
            assert wav.getframerate() == cfg.fs

    def test_load_iq_wav_restores_data_within_tolerance(
        self, scenario_oncoming_cfg, temp_wav_dir
    ):
        """Test that load_iq_wav restores data within tolerance < 1e-6."""
        cfg = scenario_oncoming_cfg
        cfg.snr_db = 100  # High SNR for accurate comparison

        result = generate_iq(cfg)
        original_iq = result["iq"]
        original_t = result["t"]

        wav_path = temp_wav_dir / "test_restore.wav"
        save_iq_wav(original_iq, original_t, wav_path, cfg)

        loaded_iq, loaded_t, metadata = load_iq_wav(wav_path)

        # Check time array
        np.testing.assert_allclose(loaded_t, original_t, rtol=1e-10)

        # Check IQ data within tolerance < 1e-6
        np.testing.assert_allclose(loaded_iq, original_iq, atol=1e-6)

    def test_load_iq_wav_metadata(self, scenario_oncoming_cfg, temp_wav_dir):
        """Test that metadata is correctly saved and loaded."""
        cfg = scenario_oncoming_cfg
        result = generate_iq(cfg)

        wav_path = temp_wav_dir / "test_metadata.wav"
        save_iq_wav(result["iq"], result["t"], wav_path, cfg)

        loaded_iq, loaded_t, metadata = load_iq_wav(wav_path)

        # Check that metadata contains expected keys
        assert "fc" in metadata
        assert "fs" in metadata
        assert "duration" in metadata

        # Check metadata values
        np.testing.assert_allclose(metadata["fc"], cfg.fc)
        np.testing.assert_allclose(metadata["fs"], cfg.fs)
        np.testing.assert_allclose(metadata["duration"], cfg.duration)

    @pytest.mark.slow
    def test_roundtrip_with_different_durations(self, temp_wav_dir):
        """Test save/load with different duration values."""
        from config import ScenarioConfig

        durations = [1.0, 5.0, 10.0]

        for duration in durations:
            cfg = ScenarioConfig(
                fc=5.8e9,
                fs=1e6,
                duration=duration,
                tx_x0=100.0,
                tx_y=3.7,
                tx_vx=30.0,
                rx_x0=0.0,
                rx_y=0.0,
                rx_vx=30.0,
                snr_db=30.0,
                f_tone=0.0,
                interp_oversample=8,
            )

            result = generate_iq(cfg)
            original_iq = result["iq"]

            wav_path = temp_wav_dir / f"test_dur_{duration}.wav"
            save_iq_wav(original_iq, result["t"], wav_path, cfg)

            loaded_iq, loaded_t, _ = load_iq_wav(wav_path)

            # Verify length
            assert len(loaded_iq) == len(original_iq)

            # Verify data within tolerance
            np.testing.assert_allclose(loaded_iq, original_iq, atol=1e-6)

    def test_iqmeta_file_created(self, scenario_oncoming_cfg, temp_wav_dir):
        """Test that companion .iqmeta file is created."""
        cfg = scenario_oncoming_cfg
        result = generate_iq(cfg)

        wav_path = temp_wav_dir / "test_metafile.wav"
        save_iq_wav(result["iq"], result["t"], wav_path, cfg)

        meta_path = wav_path.with_suffix(".iqmeta")
        assert meta_path.exists()

    def test_wav_preserves_iq_complex_structure(
        self, scenario_oncoming_cfg, temp_wav_dir
    ):
        """Test that WAV correctly stores I and Q as separate channels."""
        cfg = scenario_oncoming_cfg
        result = generate_iq(cfg)

        wav_path = temp_wav_dir / "test_channels.wav"
        save_iq_wav(result["iq"], result["t"], wav_path, cfg)

        # Read raw WAV to verify channel structure
        with wave.open(str(wav_path), "r") as wav:
            raw = wav.readframes(wav.getnframes())
            data = np.frombuffer(raw, dtype=np.int16)

            # First sample should be I, second should be Q
            i_first = data[0] / 32767.0
            q_first = data[1] / 32767.0

            np.testing.assert_allclose(i_first, result["iq"].real[0], atol=0.001)
            np.testing.assert_allclose(q_first, result["iq"].imag[0], atol=0.001)