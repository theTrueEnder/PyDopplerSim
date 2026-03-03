
# =============================================================================
# dsp/gnuradio/kinematics_block.py
# =============================================================================
"""
GNURadio decimating block wrapping dsp/kinematics.recover_kinematics.

Block type: gr.decim_block  (outputs 1 sample per decim_ratio input samples)
Input  port 0: float (smoothed Δf [Hz], from PhaseDiffBlock port 1)
Output port 0: float (ṙ estimate [m/s])
Output port 1: float (r̈ estimate [m/s²])

The decimation ratio is fs / RDOT_DECIM_HZ.  GNURadio requires this to be
an integer — if fs / RDOT_DECIM_HZ is not integer, round and log a warning.

Live integration notes
----------------------
Chain in GNURadio:
  SDR Source (complex)
    → PhaseDiffBlock      (complex → float, float)
    → KinematicsBlock     (float [smoothed Δf] → float, float)
    → QT GUI Number Sink  (display ṙ and r̈ in real time)

You can also branch the smoothed Δf into a QT Time Sink for the Δf plot.
"""

if _GR_AVAILABLE:
    class KinematicsBlock(gr.decim_block):
        """
        Streaming kinematic recovery block.

        Accumulates input samples into a buffer of size decim_ratio,
        averages them (the decimation step), then differentiates and outputs
        ṙ and r̈.  Uses a two-sample ring buffer for finite differencing.
        """

        def __init__(self, fs: float, fc: float, c: float = 299_792_458.0):
            decim_ratio = max(1, round(fs / SimConfig.RDOT_DECIM_HZ))
            if abs(decim_ratio - fs / SimConfig.RDOT_DECIM_HZ) > 0.01:
                import warnings
                warnings.warn(
                    f"fs/RDOT_DECIM_HZ = {fs/SimConfig.RDOT_DECIM_HZ:.2f} is not integer; "
                    f"rounding to {decim_ratio}.  Adjust RDOT_DECIM_HZ if needed."
                )

            gr.decim_block.__init__(
                self,
                name="Kinematics Recovery",
                in_sig=[np.float32],
                out_sig=[np.float32, np.float32],
                decim=decim_ratio,
            )
            self.fc          = fc
            self.c           = c
            self.decim_ratio = decim_ratio
            self._dt_out     = decim_ratio / fs     # output sample interval [s]

            # Ring buffer: last two ṙ values for finite differencing
            self._prev_rdot  = np.nan

        def work(self, input_items, output_items):
            delta_f_buf = input_items[0]            # length = decim_ratio × n_out
            n_out = len(output_items[0])

            for i in range(n_out):
                sl            = slice(i * self.decim_ratio, (i + 1) * self.decim_ratio)
                delta_f_mean  = np.mean(delta_f_buf[sl])
                rdot          = -delta_f_mean * self.c / self.fc

                if np.isfinite(self._prev_rdot):
                    rddot = (rdot - self._prev_rdot) / self._dt_out
                else:
                    rddot = 0.0

                self._prev_rdot = rdot

                output_items[0][i] = np.float32(rdot)
                output_items[1][i] = np.float32(rddot)

            return n_out


class KinematicsProcessor:
    """
    Batch-mode wrapper for simulation / testing without GNURadio.
    Delegates to kinematics.recover_kinematics.
    """

    def __init__(self, fs: float, fc: float, c: float = 299_792_458.0):
        self.fs = fs
        self.fc = fc
        self.c  = c

    def process(self, t: np.ndarray, delta_f_raw: np.ndarray,
                delta_f_smoothed: np.ndarray) -> dict:
        from dsp.kinematics import recover_kinematics
        return recover_kinematics(t, delta_f_raw, delta_f_smoothed, self.fc, self.c)