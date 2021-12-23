"""
Affine invariant feature-based image matching.

Based on Affine-SIFT algorithm[1].

The original implementation is based on SIFT, support for other common detectors is also added for testing use. Homography RANSAC is used to reject outliers.

Threading is used for faster affine sampling. Multicore CPUs with Hyper-threading is strongly recommended for better performance.

Copyleft Lang Zhou, zhoulang731@tongji.edu.cn

GitHub: https://github.com/Mars-Rover-Localization/PyASIFT

Created April 2021

Last modified October 2021

[1] http://www.ipol.im/pub/algo/my_affine_sift/
"""

# Built-in modules
from multiprocessing.pool import ThreadPool     # Use multiprocessing to avoid GIL
import sys

# Third party modules, opencv-contrib-python is needed
from cv2 import cv2
import numpy as np

# Local modules
from utilities import Timer, log_keypoints, image_resize
from image_matching import init_feature, filter_matches, draw_match
from config import MAX_SIZE


def affine_skew(tilt, phi, img, mask=None):
    """
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    phi is in degrees

    Ai is an affine transform matrix from skew_img to img
    """
    h, w = img.shape[:2]

    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255

    A = np.float32([[1, 0, 0], [0, 1, 0]])

    # Rotate image
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Tilt image (resizing after rotation)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt

    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)

    Ai = cv2.invertAffineTransform(A)

    return img, mask, Ai


def affine_detect(detector, img, pool=None):
    """
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.

    ThreadPool object may be passed to speedup the computation. Please use multiprocess pool to bypass GIL limitations.
    """
    params = [(1.0, 0.0)]

    # Simulate all possible affine transformations
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))

        if descrs is None:
            descrs = []

        return keypoints, descrs

    keypoints, descrs = [], []

    ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print(f"Affine sampling: {i + 1} / {len(params)}\r", end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()

    return keypoints, np.array(descrs)


def asift_main(image1: str, image2: str, detector_name: str = "sift-flann"):
    """
    Main function of ASIFT Python implementation.

    :param image1: Path for first image
    :param image2: Path for second image
    :param detector_name: (sift|surf|orb|akaze|brisk)[-flann] Detector type to use, default as SIFT. Add '-flann' to use FLANN matching.
    :return: None (Will return coordinate pairs in future)
    """
    # It seems that FLANN has performance issues, may be replaced by CUDA in future

    # Read images
    ori_img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    ori_img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize feature detector and keypoint matcher
    detector, matcher = init_feature(detector_name)

    # Exit when reading empty image
    if ori_img1 is None or ori_img2 is None:
        print("Failed to load images")
        sys.exit(1)

    # Exit when encountering unknown detector parameter
    if detector is None:
        print(f"Unknown detector: {detector_name}")
        sys.exit(1)

    ratio_1 = 1
    ratio_2 = 1

    if ori_img1.shape[0] > MAX_SIZE or ori_img1.shape[1] > MAX_SIZE:
        ratio_1 = MAX_SIZE / ori_img1.shape[1]
        print("Large input detected, image 1 will be resized")
        img1 = image_resize(ori_img1, ratio_1)
    else:
        img1 = ori_img1

    if ori_img2.shape[0] > MAX_SIZE or ori_img2.shape[1] > MAX_SIZE:
        ratio_2 = MAX_SIZE / ori_img2.shape[1]
        print("Large input detected, image 2 will be resized")
        img2 = image_resize(ori_img2, ratio_2)
    else:
        img2 = ori_img2

    print(f"Using {detector_name.upper()} detector...")

    # Profile time consumption of keypoints extraction
    with Timer(f"Extracting {detector_name.upper()} keypoints..."):
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        kp1, desc1 = affine_detect(detector, img1, pool=pool)
        kp2, desc2 = affine_detect(detector, img2, pool=pool)

    print(f"img1 - {len(kp1)} features, img2 - {len(kp2)} features")

    # Profile time consumption of keypoints matching
    with Timer('Matching...'):
        raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)

    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)

    if len(p1) >= 4:
        # TODO: The effect of resizing on homography matrix needs to be investigated.
        # TODO: Investigate function consistency when image aren't resized.
        for index in range(len(p1)):
            pt = p1[index]
            p1[index] = pt / ratio_1

        for index in range(len(p2)):
            pt = p2[index]
            p2[index] = pt / ratio_2

        for index in range(len(kp_pairs)):
            element = kp_pairs[index]
            kp1, kp2 = element

            new_kp1 = cv2.KeyPoint(kp1.pt[0] / ratio_1, kp1.pt[1] / ratio_1, kp1.size)
            new_kp2 = cv2.KeyPoint(kp2.pt[0] / ratio_2, kp2.pt[1] / ratio_2, kp2.size)

            kp_pairs[index] = (new_kp1, new_kp2)

        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
        print(f"{np.sum(status)} / {len(status)}  inliers/matched")
        # do not draw outliers (there will be a lot of them)
        kp_pairs = [kpp for kpp, flag in zip(kp_pairs, status) if flag]
    else:
        H, status = None, None
        print(f"{len(p1)} matches found, not enough for homography estimation")

    # kp_pairs: list[(cv2.KeyPoint, cv2.KeyPoint)]

    draw_match("ASIFT Match Result", ori_img1, ori_img2, kp_pairs, None, H)     # Visualize result
    cv2.waitKey()

    log_keypoints(kp_pairs, "sample/keypoints.txt")     # Save keypoint pairs for further inspection

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    asift_main("sample/IMG_0011.jpeg", "sample/IMG_0011_r.jpeg")
    cv2.destroyAllWindows()
