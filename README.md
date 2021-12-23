# PyASIFT

![Build Status](https://img.shields.io/github/workflow/status/Mars-Rover-Localization/PyASIFT/CodeQL?style=for-the-badge)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/Mars-Rover-Localization/PyASIFT.svg?logo=lgtm&style=for-the-badge)](https://lgtm.com/projects/g/Mars-Rover-Localization/PyASIFT/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/Mars-Rover-Localization/PyASIFT.svg?logo=lgtm&style=for-the-badge)](https://lgtm.com/projects/g/Mars-Rover-Localization/PyASIFT/alerts/)
![Issues](https://img.shields.io/github/issues/Mars-Rover-Localization/PyASIFT?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/Mars-Rover-Localization/PyASIFT?style=for-the-badge)
![License](https://img.shields.io/github/license/Mars-Rover-Localization/PyASIFT?style=for-the-badge)

ASIFT Python implementation with CUDA support.

***This repo is still in active development***.

### Dependencies:
* [opencv_contrib_python](https://pypi.org/project/opencv-contrib-python/)
* [numpy](https://numpy.org)

### Operating System Environment:
* macOS 11.4 Big Sur
* macOS 12.2 Monterey
* Windows 11 Pro version 21H2

***All OS are installed with Python 3.9, please notice that we haven't done test on other OS platforms.***

### Hardware Requirements:
While the program should run on most modern platforms, considering the time complexity of ASIFT algorithm, we recommend using a 6 core or better CPU for better image matching speed.

Requirements regarding GPU will be added when we complete development of relating modules.

### Usage
The current code reads two images from `sample` folder and save keypoints data in plain text format in the same directory, also display an image visualizing matching result.

Command-line argument support maybe added in near future.

### About GPU Acceleration
The team is currently working out ways to accelerate the program effectively. We may release a CUDA enabled version in early 2022.

**Please notice that** the time bottleneck in image matching is descriptor matching (which GPU acceleration may yield limited performance improvement) , rather than feature extraction. Hence, GPU acceleration won't be the team's primary focus.

### References
Original ASIFT algorithm was put forward by JM Morel, please refer to:

[Morel, Jean-Michel, and Guoshen Yu. "ASIFT: A new framework for fully affine invariant image comparison." SIAM journal on imaging sciences 2.2 (2009): 438-469.](https://epubs.siam.org/doi/abs/10.1137/080732730)

Currently, the project also use code from the Python OpenCV sample of ASIFT, please refer to:

https://github.com/opencv/opencv/blob/master/samples/python/asift.py
