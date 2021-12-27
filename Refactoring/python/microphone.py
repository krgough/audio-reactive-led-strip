"""

Read audio from the microphone and write that out to a queue

"""

import time
import logging
import queue
import numpy as np
import pyaudio

import config


LOGGER = logging.getLogger(name=__file__)

MIC_DATA = queue.Queue()


def start_stream():
    """ Read audio and send data to the callback function
    """
    audio = pyaudio.PyAudio()
    frames_per_buffer = int(config.MIC_RATE / config.FPS)
    with audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=config.MIC_RATE,
                    input=True,
                    frames_per_buffer=frames_per_buffer) as stream:

        overflows = 0
        prev_ovf_time = time.time()
        while True:
            try:
                data = np.fromstring(
                    stream.read(frames_per_buffer,
                                exception_on_overflow=False),
                    dtype=np.int16)
                data = data.astype(np.float32)
                stream.read(stream.get_read_available(),
                            exception_on_overflow=False)
                LOGGER.debug("Audio Buffer Length = %s", len(data))
                MIC_DATA.put(data)

            except IOError:
                overflows += 1
                if time.time() > prev_ovf_time + 1:
                    prev_ovf_time = time.time()
                    LOGGER.error('Audio buffer has overflowed %s times',
                                 overflows)

    # stream.stop_stream()
    # stream.close()
    # audio.terminate()
