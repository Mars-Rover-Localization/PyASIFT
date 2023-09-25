# PyASIFT

![Build Status](https://img.shields.io/github/actions/workflow/status/Mars-Rover-Localization/PyASIFT/codeql-analysis.yml?branch=main&style=for-the-badge)
![Issues](https://img.shields.io/github/issues/Mars-Rover-Localization/PyASIFT?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/Mars-Rover-Localization/PyASIFT?style=for-the-badge)
![License](https://img.shields.io/github/license/Mars-Rover-Localization/PyASIFT?style=for-the-badge)

Affine-SIFT algorithm Python implementation.

***This repo is no longer in active development***.

### Dependencies:
* [opencv_contrib_python](https://pypi.org/project/opencv-contrib-python/)
* [numpy](https://numpy.org)

### Usage
* CLI: `python asift.py --img1 [PATH_TO_FIRST_IMAGE] --img2 [PATH_TO_SECOND_IMAGE] --detector [DETECTOR_NAME]`
* Invoke the `asift_main()` method from `asift.py`

Example - Using SIFT detector with FLANN algorithm to match two sample images:
```
python asift.py --img1 sample/adam1.png --img2 sample/adam2.png --detector 'sift-flann'
```
Please refer to inline docs for more info.

### Operating System Environment:
* macOS 11.4 Big Sur
* macOS 12.2 Monterey
* Windows 11 Pro version 22H2

***Tested with Python 3.7, 3.9 and 3.10.***

### Hardware Requirements:
While the program should run on most modern platforms, considering the time complexity of ASIFT algorithm, we recommend using a 6 core or better CPU for better image matching speed.

Requirements regarding GPU will be added when we complete development of relating modules.

### About GPU Acceleration
The team is currently working out ways to accelerate the program effectively. We may release a CUDA enabled version in the future.

**Please notice that** the time bottleneck in image matching is descriptor matching (which GPU acceleration may yield limited performance improvement) , rather than feature extraction. Hence, GPU acceleration won't be the team's primary focus.

### References
Original ASIFT algorithm was put forward by JM Morel, please refer to:

[Morel, Jean-Michel, and Guoshen Yu. "ASIFT: A new framework for fully affine invariant image comparison." SIAM journal on imaging sciences 2.2 (2009): 438-469.](https://epubs.siam.org/doi/abs/10.1137/080732730)

The project also use code from the Python OpenCV sample of ASIFT, please refer to:

https://github.com/opencv/opencv/blob/master/samples/python/asift.py

Should you use code from this repo for research purposes, please use the following citation:
```
@article{zhou2022mars,
  title={Mars Rover Localization Based on A2G Obstacle Distribution Pattern Matching},
  author={Zhou, Lang and Zhang, Zhitai and Wang, Hongliang},
  journal={arXiv preprint arXiv:2210.03398},
  year={2022}
}
```
