#!/usr/bin/env python3
"""
Testing sounddevice
"""

import sounddevice as sd
import numpy as np


DURATION = 20  # seconds
data = []

def callback(indata, frames, time, status, stuff):
    # if status:
    #     print(status)
    for d in indata:
        print(d)


with sd.RawStream(channels=1, dtype='int16', callback=callback):
    sd.sleep(int(DURATION * 1000))


print('all done')
