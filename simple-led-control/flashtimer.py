#!/usr/bin/env python3
import time
from apa102_pi.driver import apa102

"""
This example should show Red, Purple, Blue, Teal, Green, Yellow in sequence.
"""

NUM_LEDS = 60

colors = [
    (255, 0, 0),
    (255, 255, 0),
    (0, 255, 0),
    (0, 255, 255),
    (0, 0, 255),
    (255, 0, 255)
]

i = 128
colors = [
    (i, 0, 0),
    (i, i, 0),
    (0, i, 0),
    (0, i, i),
    (0, 0, i),
    (i, 0, i),
    (i, i, i)
]
on = (255, 255, 255)
off = (0, 0, 0)
col = on
sleeptime = 0

lights = apa102.APA102(num_led=NUM_LEDS, order='rgb')

while True:
    if (col == on): 
        col = off
        sleeptime = 0.9
    else: 
        col = on
        sleeptime = 0.1
    for p in range(NUM_LEDS):
        lights.set_pixel(p, *col)
    lights.show()
    time.sleep(sleeptime)
	
    #for p in range(NUM_LEDS):
    #    lights.set_pixel(p, *colors[0])
    #colors.insert(0, colors.pop(-1))
    #lights.show()
    #time.sleep(0.1)
