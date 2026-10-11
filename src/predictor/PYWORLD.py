"""PyWorld pitch tracker wrapper.

The original implementation loaded a compiled WORLD shared library from a
pickled blob ("models/world.bin") through ctypes, which was fragile and
contained a NameError typo. This module now delegates to the maintained
"pyworld" pip package while keeping the same method signatures that
predictor/generator.py expects:

    pw.harvest(x, fs=..., f0_ceil=..., f0_floor=..., frame_period=...) -> (f0, t)
    pw.dio(x, fs=..., ...)                                             -> (f0, t)
    pw.stonemask(x, fs, t, f0)                                         -> f0
"""

import numpy as np
import pyworld


class PYWORLD:
    def harvest(self, x, fs, f0_floor=50, f0_ceil=1100, frame_period=10):
        x = np.ascontiguousarray(x, dtype=np.double)
        f0, t = pyworld.harvest(
            x,
            fs=fs,
            f0_ceil=f0_ceil,
            f0_floor=f0_floor,
            frame_period=frame_period,
        )
        return np.array(f0, dtype=np.float32), np.array(t, dtype=np.float32)

    def dio(self, x, fs, f0_floor=50, f0_ceil=1100, channels_in_octave=2, frame_period=10, speed=1, allowed_range=0.1):
        x = np.ascontiguousarray(x, dtype=np.double)
        f0, t = pyworld.dio(
            x,
            fs=fs,
            f0_ceil=f0_ceil,
            f0_floor=f0_floor,
            frame_period=frame_period,
            speed=speed,
            allowed_range=allowed_range,
        )
        return np.array(f0, dtype=np.float32), np.array(t, dtype=np.float32)

    def stonemask(self, x, fs, tpos, f0):
        x = np.ascontiguousarray(x, dtype=np.double)
        # pyworld requires double precision buffers for every argument.
        f0 = np.ascontiguousarray(f0, dtype=np.double)
        tpos = np.ascontiguousarray(tpos, dtype=np.double)
        # Original signature: stonemask(x, fs, tpos, f0); pip pyworld: stonemask(x, f0, t, fs)
        return np.array(pyworld.stonemask(x, f0, tpos, fs), dtype=np.float32)
