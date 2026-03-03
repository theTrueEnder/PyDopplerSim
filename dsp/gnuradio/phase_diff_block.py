
"""
dsp/gnuradio/phase_diff_block.py
=================================
GNURadio sync block wrapping dsp/estimators.phase_diff.

Block type: gr.sync_block  (1 output sample per input sample)
Input  port 0: complex float (IQ samples)
Output port 0: float (raw instantaneous frequency [Hz])
Output port 1: float (Hann-smoothed frequency [Hz])

The Hann smoother requires a look-ahead of smooth_window//2 samples, so we
maintain a circular buffer of that length.  This introduces a fixed latency of
smooth_window//2 samples — acceptable for offline/near-realtime use.

Usage in GNURadio Companion
---------------------------
1. Copy this file into your GNURadio out-of-tree module or a directory on
   your GRC Python path (PYTHONPATH).
2. In GRC, add a "Python Block" and set the module/class to this file.
3. Set fc, fs, smooth_window as block parameters.
4. Connect: SDR Source → this block → Time Sink (port 0 = raw, port 1 = smoothed)

Simulation use (no GNURadio installed)
---------------------------------------
Call PhaseDiffBlock.process_batch(iq) which runs the estimator on a full
array — identical to calling estimators.phase_diff() directly.

NOTE: This file imports gnuradio conditionally so it can be imported in
simulation environments where GNURadio is not installed.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dsp.estimators import phase_diff, hann_smooth
from config import SimConfig

try:
    import gnuradio.gr as gr
    _GR_AVAILABLE = True
except ImportError:
    _GR_AVAILABLE = False


# ---------------------------------------------------------------------------
# GNURadio streaming block (only defined if gnuradio is importable)
# ---------------------------------------------------------------------------

if _GR_AVAILABLE:
    class PhaseDiffBlock(gr.sync_block):
        """
        Streaming FM discriminator with Hann smoother.

        Processes IQ samples in GNURadio work() chunks, maintaining a
        ring buffer for the FIR smoother so state is preserved across
        consecutive work() calls.
        """

        def __init__(self, fs: float, smooth_window: int | None = None):
            """
            Parameters
            ----------
            fs            : sample rate [Hz]  — must match your SDR source
            smooth_window : Hann FIR length; None → SimConfig.PD_SMOOTH_WINDOW
            """
            gr.sync_block.__init__(
                self,
                name="Phase Diff Doppler",
                in_sig=[np.complex64],
                out_sig=[np.float32, np.float32],   # port 0: raw, port 1: smoothed
            )
            self.fs            = fs
            self.smooth_window = smooth_window or SimConfig.PD_SMOOTH_WINDOW

            # Ring buffer: store the last (smooth_window - 1) raw samples so
            # the FIR convolution has correct context across work() boundaries.
            self._buf = np.zeros(self.smooth_window - 1, dtype=np.float32)

            # Keep the last IQ sample across work() calls for lag-1 product
            self._prev_sample = np.complex64(1.0 + 0j)

        def work(self, input_items, output_items):
            iq  = input_items[0]
            N   = len(iq)

            # Lag-1 instantaneous frequency
            iq_with_prev = np.concatenate([[self._prev_sample], iq])
            lag1   = iq_with_prev[1:] * np.conj(iq_with_prev[:-1])
            f_raw  = np.angle(lag1).astype(np.float32) / (2 * np.pi) * self.fs

            # Update last sample for next call
            self._prev_sample = iq[-1]

            # Smooth with ring buffer prepended for correct FIR boundary
            padded    = np.concatenate([self._buf, f_raw])
            h         = np.hanning(self.smooth_window).astype(np.float32)
            h        /= h.sum()
            smoothed  = np.convolve(padded, h, mode='valid')

            # Update ring buffer with the last (smooth_window - 1) raw samples
            self._buf[:] = f_raw[-(self.smooth_window - 1):]

            output_items[0][:N] = f_raw
            output_items[1][:N] = smoothed[:N]
            return N


# ---------------------------------------------------------------------------
# Batch-mode wrapper (simulation / testing without GNURadio)
# ---------------------------------------------------------------------------

class PhaseDiffProcessor:
    """
    Stateless batch processor — wraps estimators.phase_diff for use in
    simulation or unit tests.  Identical output to the streaming block
    for inputs longer than smooth_window.
    """

    def __init__(self, fs: float, smooth_window: int | None = None):
        self.fs            = fs
        self.smooth_window = smooth_window or SimConfig.PD_SMOOTH_WINDOW

    def process(self, iq: np.ndarray) -> dict:
        """Run phase_diff on a full IQ array.  Returns same dict as estimators.phase_diff."""
        return phase_diff(iq, self.fs, self.smooth_window)