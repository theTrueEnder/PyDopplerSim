"""
Tests for estimation module functions.
"""

import numpy as np
import pytest

from estimation import _hann_smooth, estimate_doppler_phase_diff, estimate_doppler_stft


class TestHannSmooth:
    """Test _hann_smooth with known input/output."""

    def test_identity_for_window_1(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = _hann_smooth(x, 1)
        np.testing.assert_allclose(result, x)

    def test_smoothing_basic(self):
        x = np.ones(100)
        result = _hann_smooth(x, 5)
        np.testing.assert_allclose(result, x, atol=1e-10)

    def test_preserves_mean(self):
        x = np.random.randn(1000)
        result = _hann_smooth(x, 51)
        assert np.isclose(result.mean(), x.mean(), atol=1e-10)

    def test_reduces_variance(self):
        x = np.random.randn(1000)
        original_var = x.var()
        result = _hann_smooth(x, 51)
        smoothed_var = result.var()
        assert smoothed_var < original_var

    def test_window_normalization(self):
        x = np.full(100, 5.0)
        result = _hann_smooth(x, 11)
        np.testing.assert_allclose(result, 5.0, atol=1e-10)


class TestPhaseDiffEstimation:
    """Test estimate_doppler_phase_diff with synthetic IQ at known frequency."""

    def test_dc_signal(self):
        fs = 1e6
        t = np.arange(1000) / fs
        iq = np.exp(1j * 2 * np.pi * 0 * t)
        time, delta_f = estimate_doppler_phase_diff(iq, fs, smooth_window=51)
        np.testing.assert_allclose(delta_f, 0.0, atol=10.0)

    def test_known_frequency_positive(self):
        fs = 1e6
        f_signal = 1000.0
        t = np.arange(10000) / fs
        iq = np.exp(1j * 2 * np.pi * f_signal * t)
        time, delta_f = estimate_doppler_phase_diff(iq, fs, smooth_window=101)
        mean_est = np.mean(delta_f[500:-500])
        assert np.isclose(mean_est, f_signal, rtol=0.1)

    def test_known_frequency_negative(self):
        fs = 1e6
        f_signal = -1000.0
        t = np.arange(10000) / fs
        iq = np.exp(1j * 2 * np.pi * f_signal * t)
        time, delta_f = estimate_doppler_phase_diff(iq, fs, smooth_window=101)
        mean_est = np.mean(delta_f[500:-500])
        assert np.isclose(mean_est, f_signal, rtol=0.1)

    def test_output_length(self):
        fs = 1e6
        n_samples = 1000
        iq = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        time, delta_f = estimate_doppler_phase_diff(iq, fs)
        assert len(time) == n_samples - 1
        assert len(delta_f) == n_samples - 1

    def test_time_array(self):
        fs = 1e6
        n_samples = 1000
        iq = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        time, _ = estimate_doppler_phase_diff(iq, fs)
        expected_time = np.arange(n_samples - 1) / fs
        np.testing.assert_allclose(time, expected_time)


class TestSTFTEstimation:
    """Test estimate_doppler_stft with single-tone IQ."""

    def test_single_tone_positive(self):
        fs = 1e6
        f_signal = 500.0
        duration = 0.1
        t = np.arange(int(fs * duration)) / fs
        iq = np.exp(1j * 2 * np.pi * f_signal * t)

        result = estimate_doppler_stft(
            iq, fs, window_dur=0.01, hop_dur=0.005, freq_zoom=2000.0
        )

        mean_peak = np.mean(result["peak_freq"][5:-5])
        assert np.isclose(mean_peak, f_signal, rtol=0.2)

    def test_single_tone_negative(self):
        fs = 1e6
        f_signal = -500.0
        duration = 0.1
        t = np.arange(int(fs * duration)) / fs
        iq = np.exp(1j * 2 * np.pi * f_signal * t)

        result = estimate_doppler_stft(
            iq, fs, window_dur=0.01, hop_dur=0.005, freq_zoom=2000.0
        )

        mean_peak = np.mean(result["peak_freq"][5:-5])
        assert np.isclose(mean_peak, f_signal, rtol=0.2)

    def test_output_dict_keys(self):
        fs = 1e6
        t = np.arange(int(fs * 0.1)) / fs
        iq = np.exp(1j * 2 * np.pi * 100.0 * t)

        result = estimate_doppler_stft(iq, fs)

        assert "t" in result
        assert "freq_axis" in result
        assert "Sxx" in result
        assert "peak_freq" in result
        assert "mask" in result

    def test_spectrogram_shape(self):
        fs = 1e6
        n_samples = 10000
        iq = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)

        window_dur = 0.05
        hop_dur = 0.005
        result = estimate_doppler_stft(iq, fs, window_dur=window_dur, hop_dur=hop_dur)

        n_fft = int(window_dur * fs) * 4
        n_frames = (n_samples - int(window_dur * fs)) // int(hop_dur * fs) + 1

        assert result["Sxx"].shape[0] == n_fft
        assert result["Sxx"].shape[1] == n_frames
        assert len(result["peak_freq"]) == n_frames

    def test_freq_axis_symmetry(self):
        fs = 1e6
        t = np.arange(int(fs * 0.1)) / fs
        iq = np.exp(1j * 2 * np.pi * 0 * t)

        result = estimate_doppler_stft(iq, fs)

        freq_pos = result["freq_axis"][result["freq_axis"] > 0]
        freq_neg = result["freq_axis"][result["freq_axis"] < 0]
        assert len(freq_pos) == len(freq_neg)

    def test_mask_within_zoom(self):
        fs = 1e6
        t = np.arange(int(fs * 0.1)) / fs
        iq = np.exp(1j * 2 * np.pi * 100.0 * t)

        freq_zoom = 500.0
        result = estimate_doppler_stft(iq, fs, freq_zoom=freq_zoom)

        masked_freq = result["freq_axis"][result["mask"]]
        assert np.all(np.abs(masked_freq) <= freq_zoom)


class TestWithFixtures:
    """Tests using conftest fixtures."""

    def test_phase_diff_with_fixed_seed(self, rng):
        fs = 1e6
        f_signal = 200.0
        n_samples = 5000
        t = np.arange(n_samples) / fs
        iq = np.exp(1j * 2 * np.pi * f_signal * t)
        noise = (rng.randn(n_samples) + 1j * rng.randn(n_samples)) * 0.01
        iq_noisy = iq + noise

        _, delta_f = estimate_doppler_phase_diff(iq_noisy, fs, smooth_window=101)
        mean_est = np.mean(delta_f[100:-100])
        assert np.isclose(mean_est, f_signal, rtol=0.3)

    def test_stft_temp_wav_dir(self, temp_wav_dir):
        fs = 1e6
        t = np.arange(int(fs * 0.05)) / fs
        iq = np.exp(1j * 2 * np.pi * 300.0 * t)

        result = estimate_doppler_stft(iq, fs, window_dur=0.01, hop_dur=0.005)
        assert result["t"] is not None
        assert len(result["peak_freq"]) > 0