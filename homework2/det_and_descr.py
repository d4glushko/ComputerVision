import cv2


def gaussian_blur(img, kernel_sz, sigma):
    return None

def find_keypoints_candidates(img):
    return None

def compute_descriptors(img, kp_arr):
    return None


# function for keypoints and descriptors calculation
def detect_keypoints_and_calculate_descriptors(img):
    # img - numpy 2d array (grayscale image)
    img_blur = gaussian_blur(img, 31, 3.0)

    # keypoints
    kp_arr = find_keypoints_candidates(img_blur)
    # kp_arr is array of 2d coordinates-tuples, example:
    # [(x0, y0), (x1, y1), ...]
    # xN, yN - integers

    # descriptors
    descr_arr = compute_descriptors(img_blur, kp_arr)
    # cv_descr_arr is array of descriptors (arrays), example:
    # [[v00, v01, v02, ...], [v10, v11, v12, ...], ...]
    # vNM - floats

    return kp_arr, descr_arr
