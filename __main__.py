import wave
from math import tau
from fractions import Fraction
from typing import Union

import numpy as np
from matplotlib import pyplot as plt


class MonoWave:
    """
    A class for representing mono WAV files.
    """
    def __init__(self, samples: np.ndarray, framerate) -> None:
        assert samples.dtype.kind == "i"
        assert samples.ndim == 1

        if samples.dtype.byteorder != '=':
            samples = samples.byteswap()
        samples.view(samples.dtype.newbyteorder('='))

        self.samples = samples
        self.framerate = framerate
        self.step = Fraction(1, self.framerate)
        self.duration = samples.size * self.step

    def show(self, title: str='') -> None:
        """
        Plots and shows the samples as a scatter plot.
        """
        plt.figure()
        plt.xlim(0, 256 / self.framerate)
        if title:
            plt.title(title)
        plt.xlabel('time (s)')
        plt.scatter(np.arange(0, self.duration, self.step), self.samples)
        plt.show()

    def save(self, filename) -> None:
        """
        Saves the samples to a WAV file without compressing.
        """
        with wave.open(filename, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(self.samples.dtype.itemsize)
            f.setframerate(self.framerate)
            f.setcomptype("NONE", "not compressed")
            f.writeframes(self.samples)


class SimpleHarmonicWave(MonoWave):
    """
    A class for representing simple harmonic waves.
    """
    def __init__(self, frequency, dtype: np.dtype, framerate, duration, volume=1.0) -> None:
        dtype = np.dtype(dtype)
        if dtype.kind != "i":
            raise ValueError("dtype must be signed integer")

        assert 0.0 <= volume <= 1.0 # TODO: raise an error instead
        amplitude = (2 ** (dtype.itemsize * 8 - 1) - 1) * volume
        samples = np.arange(duration * framerate)
        samples = amplitude * np.sin(tau * frequency * samples / framerate)
        samples = np.round(samples).astype(dtype)

        super().__init__(samples, framerate)


class SquareWave(MonoWave):
    """
    A class for representing square waves.
    """
    def __init__(self, frequency, dtype: np.dtype, framerate, duration, volume=1.0) -> None:
        dtype = np.dtype(dtype)
        if dtype.kind != "i":
            raise ValueError("dtype must be signed integer")

        assert 0.0 <= volume <= 1.0 # TODO: raise an error instead
        amplitude = (2 ** (dtype.itemsize * 8 - 1) - 1) * volume
        samples = np.arange(duration * framerate)
        samples_per_half_period = Fraction(1, frequency) * framerate / 2
        samples = np.where(samples // samples_per_half_period % 2 == 0, amplitude, -amplitude)
        samples = np.round(samples).astype(dtype)

        super().__init__(samples, framerate)


if __name__ == "__main__":
    shwave = SimpleHarmonicWave(frequency=440, dtype=np.int8, framerate=44100, duration=1.0, volume=0.5)
    shwave.show(title="Simple Harmonic Wave")
    shwave.save("simple_harmonic.wav")

    sqwave = SquareWave(frequency=440, dtype=np.int8, framerate=44100, duration=1.0, volume=0.5)
    sqwave.show(title="Square Wave")
    sqwave.save("square.wav")
