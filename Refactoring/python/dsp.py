"""
DSP Functions for audio reactive LED strip
"""
import numpy as np
import config as cfg
import melbank


class ExpFilter:
    # pylint: disable=too-few-public-methods
    """Simple exponential smoothing filter"""
    def __init__(self, val=0.0, alpha_decay=0.5, alpha_rise=0.5):
        """Small rise / decay factors = more smoothing"""
        assert 0.0 < alpha_decay < 1.0, 'Invalid decay smoothing factor'
        assert 0.0 < alpha_rise < 1.0, 'Invalid rise smoothing factor'
        self.alpha_decay = alpha_decay
        self.alpha_rise = alpha_rise
        self.value = val

    def update(self, value):
        """  Update the output of the Filter
        """
        if isinstance(self.value, (list, np.ndarray, tuple)):
            alpha = value - self.value
            alpha[alpha > 0.0] = self.alpha_rise
            alpha[alpha <= 0.0] = self.alpha_decay
        else:
            alpha = self.alpha_rise if value > self.value else self.alpha_decay
        self.value = alpha * value + (1.0 - alpha) * self.value
        return self.value


def rfft(data, window=None):
    """ Fourier Transform for real input
    """
    window = 1.0 if window is None else window(len(data))
    mags = np.abs(np.fft.rfft(data * window))
    freqs = np.fft.rfftfreq(len(data), 1.0 / cfg.MIC_RATE)
    return freqs, mags


def fft(data, window=None):
    """  Fourier Transform
    """
    window = 1.0 if window is None else window(len(data))
    mags = np.fft.fft(data * window)
    freqs = np.fft.fftfreq(len(data), 1.0 / cfg.MIC_RATE)
    return freqs, mags


def create_mel_bank():
    """ Create a bank of mel filter coefficients
    """
    samples = int(cfg.MIC_RATE * cfg.N_ROLLING_HISTORY / (2.0 * cfg.FPS))
    mel_y, _, mel_x = melbank.compute_melmat(num_mel_bands=cfg.N_FFT_BINS,
                                             freq_min=cfg.MIN_FREQUENCY,
                                             freq_max=cfg.MAX_FREQUENCY,
                                             num_fft_bands=samples,
                                             sample_rate=cfg.MIC_RATE)
    return mel_x, mel_y


SAMPLES = None
MEL_X, MEL_Y = create_mel_bank()
