import cv2


fast = cv2.FastFeatureDetector_create()
orb = cv2.ORB_create()


# function for keypoints and descriptors calculation
def detect_keypoints_and_calculate_descriptors(img):
    # img - numpy 2d array (grayscale image)
    img_blur = cv2.GaussianBlur(img, (31, 31), 3.0)

    # keypoints
    kp_arr = []

    cv_kp_arr = fast.detect(img_blur, None)

    # descriptors
    cv_kp_arr, cv_descr_arr = orb.compute(img_blur, cv_kp_arr)

    for i in range(len(cv_kp_arr)):
        kp_arr.append((
            int(cv_kp_arr[i].pt[0]),
            int(cv_kp_arr[i].pt[1])
        ))

    # kp_arr is array of 2d coordinates-tuples, example:
    # [(x0, y0), (x1, y1), ...]
    # xN, yN - integers
    #
    # cv_descr_arr is array of descriptors (arrays), example:
    # [[v00, v01, v02, ...], [v10, v11, v12, ...], ...]
    # vNM - floats
    #
    return kp_arr, cv_descr_arr
