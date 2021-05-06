"""
This module contains some common methods used by other scripts.

GitHub: https://github.com/Mars-Rover-Localization/PyASIFT
"""

# Built-in modules
from contextlib import contextmanager

# Third party modules
import numpy as np
import cv2 as cv


def clock():
    return cv.getTickCount() / cv.getTickFrequency()


@contextmanager
def Timer(msg):
    print(msg)
    start = clock()
    try:
        yield
    finally:
        print("%.4f ms" % ((clock() - start) * 1000))


def log_keypoints(kp_pairs, path: str = '/Users/langzhou/Downloads/keypoints.txt'):
    with open(path, 'w+') as log:
        for kp1, kp2 in kp_pairs:
            log.write(f"{np.int32(kp1.pt)}      {np.int32(kp2.pt)}\n")
    log.close()

    print(f"Keypoints logged at {path}")


def image_resize(src, ratio: float):
    dim = (int(src.shape[-1] * ratio), int(src.shape[0] * ratio))
    return cv.resize(src, dim, interpolation=cv.INTER_AREA)
