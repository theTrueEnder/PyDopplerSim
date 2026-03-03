# ── dsp/gnuradio/__init__.py ──────────────────────────────────────────────
# (save as dsp/gnuradio/__init__.py)
#
# Conditionally export GNURadio blocks only if gnuradio is available,
# so importing dsp.gnuradio doesn't crash in pure-sim environments.
try:
    from dsp.gnuradio.phase_diff_block import PhaseDiffBlock
    from dsp.gnuradio.kinematics_block import KinematicsBlock
    __all__ = ["PhaseDiffBlock", "KinematicsBlock"]
except ImportError:
    __all__ = []

# Always export the batch processors (no gnuradio dependency)
from dsp.gnuradio.phase_diff_block import PhaseDiffProcessor
from dsp.gnuradio.kinematics_block import KinematicsProcessor