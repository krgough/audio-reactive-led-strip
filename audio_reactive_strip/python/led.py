"""

Execute this file to run a LED strand test
If everything is working, you should see a red, green, and blue pixel scroll
across the LED strip continuously

"""


import numpy as np
import config
from apa102_pi.driver import apa102

STRIP = apa102.APA102(num_led=60, order='rgb')

GAMMA = np.load(config.GAMMA_TABLE_PATH)
"""Gamma lookup table used for nonlinear brightness correction"""

PREV_PIXELS = np.tile(253, (3, config.N_PIXELS))
"""Pixel values that were most recently displayed on the LED strip"""

PIXELS = np.tile(1, (3, config.N_PIXELS))
"""Pixel values for the LED strip"""


def update():
    """Writes new LED values to the Raspberry Pi's LED strip

    Raspberry Pi uses the rpi_ws281x to control the LED strip directly.
    This function updates the LED strip with new values.
    """
    global PIXELS, PREV_PIXELS
    # Truncate values and cast to integer
    PIXELS = np.clip(PIXELS, 0, 255).astype(int)

    # Encode 24-bit LED values in 32 bit integers
    r = np.left_shift(GAMMA[PIXELS][0][:].astype(int), 8)
    g = np.left_shift(GAMMA[PIXELS][1][:].astype(int), 16)
    b = GAMMA[PIXELS][2][:].astype(int)
    rgb = np.bitwise_or(np.bitwise_or(r, g), b)
    # Update the pixels
    for i in range(config.N_PIXELS):
        # Ignore pixels if they haven't changed (saves bandwidth)
        if np.array_equal(GAMMA[PIXELS][:, i], PREV_PIXELS[:, i]):
            continue
        # strip._led_data[i] = rgb[i]
        STRIP.set_pixel_rgb(i, int(rgb[i]))
    PREV_PIXELS = np.copy(GAMMA[PIXELS])
    STRIP.show()


if __name__ == '__main__':
    import time
    # Turn all pixels off
    PIXELS *= 0
    PIXELS[0, 0] = 255  # Set 1st pixel red
    PIXELS[1, 1] = 255  # Set 2nd pixel green
    PIXELS[2, 2] = 255  # Set 3rd pixel blue
    print('Starting LED strand test')
    while True:
        PIXELS = np.roll(PIXELS, shift=1, axis=1)
        update()
        time.sleep(.1)
