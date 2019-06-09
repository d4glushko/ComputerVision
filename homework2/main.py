import cv2
import numpy as np
from matplotlib import pyplot as plt

#from opencv_det_and_descr import detect_keypoints_and_calculate_descriptors
from opencv_det_and_descr import detect_keypoints_and_calculate_descriptors


img_in_path = './res/tracking/'


def match_brute_force(descr_arr0, descr_arr1):
    bf = cv2.BFMatcher()
    # return 2 matches for keypoint
    matches = bf.knnMatch(descr_arr0, descr_arr1, k=2)

    matches_arr = []
    for match_a, match_b in matches:
        # mark match good if 2nd match has bigger distance
        # (filtering similar keypoints)
        if match_a.distance < 0.75 * match_b.distance:
            matches_arr.append((
                match_a.queryIdx,
                match_a.trainIdx
            ))

    return matches_arr


for test_name in ['translation_', 'translation_noise_', 'rotation_2_', 'rotation_5_']:
    for frame_idx in range(9):
        # read two frames
        img0_fpath = img_in_path + test_name + str(frame_idx) + '.png'
        img0 = cv2.imread(img0_fpath, cv2.IMREAD_GRAYSCALE)

        img1_fpath = img_in_path + test_name + str(frame_idx+1) + '.png'
        img1 = cv2.imread(img1_fpath, cv2.IMREAD_GRAYSCALE)

        rows = img0.shape[0]
        cols = img0.shape[1]

        # detect keypoints and calculate descriptors
        kp0, descr0 = detect_keypoints_and_calculate_descriptors(img0.copy())
        kp1, descr1 = detect_keypoints_and_calculate_descriptors(img1.copy())

        # match
        match_arr = match_brute_force(descr0, descr1)

        # draw on one image
        img_both = np.zeros((rows, cols*2), np.uint8)
        img_both[:, 0:cols] = img0
        img_both[:, cols:cols*2] = img1
        img_both_bgr = cv2.cvtColor(img_both, cv2.COLOR_GRAY2BGR)

        # keypoints as red circles
        for i in range(len(kp0)):
            kp = kp0[i]
            x = kp[0]
            y = kp[1]
            cv2.circle(img_both_bgr, (x, y), 10, (0, 0, 255))

        for i in range(len(kp1)):
            kp = kp1[i]
            x = kp[0] + cols
            y = kp[1]
            cv2.circle(img_both_bgr, (x, y), 10, (0, 0, 255))

        # matches as green lines
        for pair in match_arr:
            x0 = kp0[pair[0]][0]
            y0 = kp0[pair[0]][1]
            x1 = kp1[pair[1]][0] + cols
            y1 = kp1[pair[1]][1]
            cv2.line(img_both_bgr, (x0, y0), (x1, y1), (0, 255, 0))

        # show image and wait for key press
        cv2.imshow('img_both', img_both_bgr)
        cv2.waitKey(-1)
