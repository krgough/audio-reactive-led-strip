"""

Execute this file to run a LED strand test
If everything is working, you should see a red, green, and blue pixel scroll
across the LED strip continuously

"""


import numpy as np
import config
# from apa102_pi.driver import apa102


GAMMA = np.load(config.GAMMA_TABLE_PATH)
"""Gamma lookup table used for nonlinear brightness correction"""


class DummyStrip():
    """ A stubbed out object to support testing """
    def __init__(self, num_led) -> None:
        self.pixels = [0 for i in range(num_led)]

    def set_pixel_rgb(self, pixel_num, val):
        """ Set the buffer for the given pixel to the given value
        """
        self.pixels[pixel_num] = val

    def show(self):
        """ Mimic the show() function
        """

        def colored(red, green, blue, text):
            return (f"\033[38;2;{red};{green};{blue}m{text} "
                    "\033[38;2;255;255;255m")

        for i in self.pixels:
            num = f"{i:06x}"
            red = int(num[:2], 16)
            green = int(num[2:4], 16)
            blue = int(num[4:], 16)
            print(colored(red, green, blue, '@'), end='')
        print()


class Leds():
    """ Class to manage the operation of the LED strip
        works with APA102 led strips or for testing on pc you
        can use the DummyStrip class.
    """

    def __init__(self, debug=False) -> None:

        # Define the LED strip interface
        if debug:
            self.strip = DummyStrip(num_led=60)
        else:
            self.strip = apa102.APA102(num_led=60, order='rgb')

        # Pixel value buffers for the LED strip
        # Writing to the buffer does not update the strip
        # Call update() to change the LED states.
        self.pixels = np.tile(1, (3, config.N_PIXELS))
        self.prev_pixels = np.tile(253, (3, config.N_PIXELS))

    def clear_pixels(self):
        """ Clear out the pixel buffer
        """
        self.pixels *= 0

    def update(self):
        """Writes new LED values to the Raspberry Pi's LED strip

        Raspberry Pi uses the rpi_ws281x to control the LED strip directly.
        This function updates the LED strip with new values.
        """
        # Truncate values and cast to integer
        self.pixels = np.clip(self.pixels, 0, 255).astype(int)

        # Encode 24-bit LED values in 32 bit integers
        red = np.left_shift(GAMMA[self.pixels][0][:].astype(int), 8)
        green = np.left_shift(GAMMA[self.pixels][1][:].astype(int), 16)
        blue = GAMMA[self.pixels][2][:].astype(int)
        rgb = np.bitwise_or(np.bitwise_or(red, green), blue)
        # Update the pixels
        for i in range(config.N_PIXELS):
            # Ignore pixels if they haven't changed (saves bandwidth)
            if np.array_equal(
                    GAMMA[self.pixels][:, i],
                    self.prev_pixels[:, i]):
                continue
            # strip._led_data[i] = rgb[i]
            self.strip.set_pixel_rgb(i, int(rgb[i]))
        self.prev_pixels = np.copy(GAMMA[self.pixels])
        self.strip.show()


if __name__ == '__main__':
    import time

    # Turn all pixels off
    leds = Leds(debug=True)
    leds.clear_pixels()
    leds.pixels[0, 0] = 255  # Set 1st pixel red
    leds.pixels[1, 1] = 255  # Set 2nd pixel green
    leds.pixels[2, 2] = 255  # Set 3rd pixel blue
    print('Starting LED strand test')
    while True:
        leds.pixels = np.roll(leds.pixels, shift=1, axis=1)
        leds.update()
        time.sleep(.1)
