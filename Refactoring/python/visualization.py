"""

Entry point for running the visualisation

"""

from textwrap import dedent
import time
import logging
import os
from getopt import gnu_getopt, GetoptError
import sys

import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

import config
import microphone
import dsp
import led

LOGGER = logging.getLogger(name=__file__)

# The previous time that the frames_per_second() function was called
_time_prev = time.time() * 1000.0

# The low-pass filter used to estimate frames-per-second
_fps = dsp.ExpFilter(val=config.FPS, alpha_decay=0.2, alpha_rise=0.2)

r_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.2, alpha_rise=0.99)

g_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.05, alpha_rise=0.3)

b_filt = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                       alpha_decay=0.1, alpha_rise=0.5)

common_mode = dsp.ExpFilter(np.tile(0.01, config.N_PIXELS // 2),
                            alpha_decay=0.99, alpha_rise=0.01)

p_filt = dsp.ExpFilter(np.tile(1, (3, config.N_PIXELS // 2)),
                       alpha_decay=0.1, alpha_rise=0.99)

p = np.tile(1.0, (3, config.N_PIXELS // 2))

gain = dsp.ExpFilter(np.tile(0.01, config.N_FFT_BINS),
                     alpha_decay=0.001, alpha_rise=0.99)

_prev_spectrum = np.tile(0.01, config.N_PIXELS // 2)


# Number of audio samples to read every time frame
SAMPLES_PER_FRAME = int(config.MIC_RATE / config.FPS)

# Array containing the rolling audio sample window
y_roll = np.random.rand(config.N_ROLLING_HISTORY, SAMPLES_PER_FRAME) / 1e16


# def memoize(function):
#     """Provides a decorator for memoizing functions"""
#     from functools import wraps
#     memo = {}

#     @wraps(function)
#     def wrapper(*args):
#         if args in memo:
#             return memo[args]
#         else:
#             rv = function(*args)
#             memo[args] = rv
#             return rv
#     return wrapper


# @memoize
# def _normalized_linspace(size):
#     return np.linspace(0, 1, size)


def interpolate(data, new_length):
    """Intelligently resizes the array by linearly interpolating the values

    Parameters
    ----------
    data : np.array
        Array that should be resized

    new_length : int
        The length of the new interpolated array

    Returns
    -------
    z : np.array
        New array with length of new_length that contains the interpolated
        values of y.
    """
    if len(data) == new_length:
        return data

    # x_old = _normalized_linspace(len(data))
    # x_new = _normalized_linspace(new_length)

    x_old = np.linspace(0, 1, len(data))
    x_new = np.linspace(0, 1, new_length)

    resized_data = np.interp(x_new, x_old, data)
    return resized_data


def visualize_scroll(data):
    """ Effect that originates in the center and scrolls outwards
    """
    global p
    data = data**2.0
    gain.update(data)
    data /= gain.value
    data *= 255.0
    red = int(np.max(data[:len(data) // 3]))
    green = int(np.max(data[len(data) // 3: 2 * len(data) // 3]))
    blue = int(np.max(data[2 * len(data) // 3:]))
    # Scrolling effect window
    p[:, 1:] = p[:, :-1]
    p *= 0.98
    p = gaussian_filter1d(p, sigma=0.2)
    # Create new color originating at the center
    p[0, 0] = red
    p[1, 0] = green
    p[2, 0] = blue
    # Update the LED strip
    return np.concatenate((p[:, ::-1], p), axis=1)


def visualize_energy(data):
    """ Effect that expands from the center with increasing sound energy
    """
    global p
    data = np.copy(data)
    gain.update(data)
    data /= gain.value
    # Scale by the width of the LED strip
    data *= float((config.N_PIXELS // 2) - 1)
    # Map color channels according to energy in the different freq bands
    scale = 0.9
    red = int(np.mean(y[:len(data) // 3]**scale))
    green = int(np.mean(y[len(data) // 3: 2 * len(data) // 3]**scale))
    blue = int(np.mean(y[2 * len(data) // 3:]**scale))
    # Assign color to different frequency regions
    p[0, :red] = 255.0
    p[0, red:] = 0.0
    p[1, :green] = 255.0
    p[1, green:] = 0.0
    p[2, :blue] = 255.0
    p[2, blue:] = 0.0
    p_filt.update(p)
    p = np.round(p_filt.value)
    # Apply substantial blur to smooth the edges
    p[0, :] = gaussian_filter1d(p[0, :], sigma=4.0)
    p[1, :] = gaussian_filter1d(p[1, :], sigma=4.0)
    p[2, :] = gaussian_filter1d(p[2, :], sigma=4.0)
    # Set the new pixel value
    return np.concatenate((p[:, ::-1], p), axis=1)



def visualize_spectrum(data):
    """ Effect that maps the Mel filterbank frequencies onto the LED strip
    """
    global _prev_spectrum
    data = np.copy(interpolate(data, config.N_PIXELS // 2))
    common_mode.update(data)
    diff = data - _prev_spectrum
    _prev_spectrum = np.copy(data)
    # Color channel mappings
    red = r_filt.update(data - common_mode.value)
    green = np.abs(diff)
    blue = b_filt.update(np.copy(data))
    # Mirror the color channels for symmetric output
    red = np.concatenate((red[::-1], red))
    green = np.concatenate((green[::-1], green))
    blue = np.concatenate((blue[::-1], blue))
    output = np.array([red, green, blue]) * 255
    return output


# fft_plot_filter = dsp.ExpFilter(
#     np.tile(1e-1, config.N_FFT_BINS),
#     alpha_decay=0.5, alpha_rise=0.99)

mel_gain = dsp.ExpFilter(
    np.tile(1e-1, config.N_FFT_BINS),
    alpha_decay=0.01, alpha_rise=0.99)
mel_smoothing = dsp.ExpFilter(
    np.tile(1e-1, config.N_FFT_BINS),
    alpha_decay=0.5, alpha_rise=0.99)
volume = dsp.ExpFilter(
    config.MIN_VOLUME_THRESHOLD,
    alpha_decay=0.02, alpha_rise=0.02)

fft_window = np.hamming(
    int(config.MIC_RATE / config.FPS) * config.N_ROLLING_HISTORY)
prev_fps_update = time.time()


def microphone_update(audio_samples, led_strip):
    global y_roll, prev_rms, prev_exp, prev_fps_update
    # Normalize samples between 0 and 1
    data = audio_samples / 2.0**15
    # Construct a rolling window of audio samples
    y_roll[:-1] = y_roll[1:]
    y_roll[-1, :] = np.copy(data)
    y_data = np.concatenate(y_roll, axis=0).astype(np.float32)

    vol = np.max(np.abs(y_data))
    if vol < config.MIN_VOLUME_THRESHOLD:
        print('No audio input. Volume below threshold. Volume:', vol)
        led.pixels = np.tile(0, (3, config.N_PIXELS))
        led.update()
    else:
        # Transform audio input into the frequency domain
        n_vals = len(y_data)
        n_zeros = 2**int(np.ceil(np.log2(n_vals))) - n_vals
        # Pad with zeros until the next power of two
        y_data *= fft_window
        y_padded = np.pad(y_data, (0, n_zeros), mode='constant')
        YS = np.abs(np.fft.rfft(y_padded)[:n_vals // 2])
        # Construct a Mel filterbank from the FFT data
        mel = np.atleast_2d(YS).T * dsp.MEL_Y.T
        # Scale data to values more suitable for visualization
        # mel = np.sum(mel, axis=0)
        mel = np.sum(mel, axis=0)
        mel = mel**2.0
        # Gain normalization
        mel_gain.update(np.max(gaussian_filter1d(mel, sigma=1.0)))
        mel /= mel_gain.value
        mel = mel_smoothing.update(mel)
        # Map filterbank output onto LED strip
        output = visualization_effect(mel)
        led_strip.pixels = output
        led_strip.update()




def get_args():
    """ Get command line arguments and return the selected visualisation function
    """
    usage = dedent(f"""
    USAGE: ./{os.path.basename(__file__)} [-h] spectrum|energy|scroll

    Generates visual LED display with selected visualisation function
    """)

    try:
        opts, args = gnu_getopt(sys.argv[1:], "h")
        for opt, _ in opts:
            if opt == '-h':
                print(usage)
                sys.exit(0)

    except GetoptError:
        print("ERROR: Invalid argument")
        print(usage)
        sys.exit(1)

    if args:
        viz_func = args[0]
        funcs = {
            'spectrum': visualize_spectrum,
            'scroll': visualize_scroll,
            'energy': visualize_energy}

        if viz_func in funcs:
            return funcs[viz_func]

        print(f'ERROR: Visualisation function {viz_func} not recognised')
        sys.exit(1)

    else:
        print("ERROR: No visualisation function given")
        print(usage)
        sys.exit(1)


if __name__ == '__main__':
    logging.basicConfig(level=logging.ERROR)

    # Visualization effect to display on the LED strip
    visualization_effect = get_args()

    # Initialize LEDs
    led_strip = led.Leds(debug=True)
    led_strip.update()

    # Start listening to live audio stream
    # Audio returned in a queue object
    microphone.start_stream()


