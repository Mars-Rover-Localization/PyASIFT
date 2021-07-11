"""
This module contains configuration data for ASIFT matching.
For usage of each variable, please refer to in-line notations.
"""


"""
Currently PyASIFT cannot process large size image correctly.
We believe it's a bug in OpenCV's knnmatch algorithm.
While we are actively developing alternative matching algorithms, current input image size is deliberately limited.
After resizing, the program ensures that the width of image will not exceed MAX_SIZE.
Please notice that the keypoints returned and logged will be rescaled to original size.
From our testing, it's recommended that MAX_SIZE be set to 1500-2000.
If PyASIFT throws an error while executing, reduce the MAX_SIZE value may help. 
"""
MAX_SIZE = 1000
