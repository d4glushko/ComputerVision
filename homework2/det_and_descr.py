import cv2
import numpy as np


def gauss_pdf(sigma, x, y):
    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

def gauss_kernel(sigma=1, size=5):
    kernel = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            kernel[i, j] = gauss_pdf(sigma, i - size // 2, j - size // 2)

    return kernel

def convolution(img, kernel):
    grayscale = False
    if img.ndim == 2:
        grayscale = True
        img = img.reshape((img.shape[0], img.shape[1], 1))
    height, width, channels = img.shape
    kernel_h, kernel_w = kernel.shape
    output_w = width - kernel_w + 1
    output_h = height - kernel_h + 1
    result = np.zeros(img.shape)
    
    left_padding_size = (width - output_w) // 2
    top_padding_size = (height - output_h) // 2
    
    new_img = np.zeros((height + top_padding_size * 2, width + left_padding_size * 2, channels))
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                new_img[i + top_padding_size, j + left_padding_size, channel] = img[i,j,channel]
    
    for channel in range(channels):
        for i in range(height):
            for j in range(width):
                element = np.sum(np.multiply(kernel, new_img[i:i+kernel_h, j:j+kernel_w, channel]))
                result[i,j,channel] = element
    
    if grayscale:
        result = result.reshape((result.shape[0], result.shape[1]))
    return result

def gaussian_blur(img, kernel_sz, sigma):
    return convolution(img, gauss_kernel(sigma, kernel_sz))

def bresenham_circle(img, i, j):
    return [
        img[i, j - 3],
        img[i + 1, j - 3],
        img[i + 2, j - 2],
        img[i + 3, j - 1],
        img[i + 3, j],
        img[i + 3, j + 1],
        img[i + 2, j + 2],
        img[i + 1, j + 3],
        img[i, j + 3],
        img[i - 1, j + 3],
        img[i - 2, j + 2],
        img[i - 3, j + 1],
        img[i - 3, j],
        img[i - 3, j - 1],
        img[i - 2, j - 2],
        img[i - 1, j - 3],
    ]

def corner_test(img, i, j, threshold):
    point = img[i, j]
    # Pixels 1,5,9 and 13
    pixels = img[i, j - 3], img[i + 3, j], img[i, j + 3], img[i - 3, j]

    if sum([pixel > point + threshold for pixel in pixels]) >= 3:
        return True, True

    if sum([pixel < point - threshold for pixel in pixels]) >= 3:
        return True, False
    return False, None

def calculate_corner_score(img, i, j, threshold, is_brighter):
    point = img[i, j]
    if is_brighter:
        pixels_score = [
            pixel - point - threshold for pixel in bresenham_circle(img, i, j)
        ]
    else:
        pixels_score = [
            point - threshold - pixel for pixel in bresenham_circle(img, i, j)
        ]

    max_contiguous_pixels = 0
    current_contiguous_pixels = 0
    for pixel in pixels_score * 2:
        current_contiguous_pixels = current_contiguous_pixels + 1 if pixel > 0 else 0
        if current_contiguous_pixels > max_contiguous_pixels:
            max_contiguous_pixels = current_contiguous_pixels

    pixels_score = np.array(pixels_score)
    return (max_contiguous_pixels, np.sum(pixels_score[pixels_score > 0]))

def non_maximum_suppresion(img, distance=5):
    height, width = img.shape
    half_distance = distance // 2
    new_img = np.zeros((height + half_distance * 2, width + half_distance * 2))
    for i in range(height):
        for j in range(width):
            new_img[i + half_distance, j + half_distance] = img[i,j]

    result = []

    for i in range(height):
        for j in range(width):
            maximum = new_img[i : i + distance, j : j + distance].max()
            if maximum and maximum == img[i, j]:
                result.append((j,i))
    return result

def find_keypoints_candidates(img, threshold=10):
    height, width = img.shape
    corners = np.zeros(img.shape)

    for i in range(3, height - 3):
        for j in range(3, width - 3):
            is_candidate, is_brighter = corner_test(img, i, j, threshold)

            if is_candidate:
                max_contiguous_pixels, corner_score = calculate_corner_score(
                    img, i, j, threshold, is_brighter
                )
                if max_contiguous_pixels >= 10:
                    corners[i, j] = corner_score
    suppressed_corners = non_maximum_suppresion(corners)

    return suppressed_corners

def brief(img, x, y, pairs, patch_size):
    if len(pairs) >= patch_size ** 2:
        return None
    height, width = img.shape
    half_patch_size = patch_size // 2

    if (
        x < half_patch_size
        or y < half_patch_size
        or x >= width - half_patch_size
        or y >= height - half_patch_size
    ):
        return None

    p = img[x, y]
    descriptor = np.zeros(int(len(pairs) / 8), dtype=np.uint8)

    for index, (pair_x, pair_y) in enumerate(pairs):
        value = img[x - half_patch_size + pair_x, y - half_patch_size + pair_y]
        if (value > p):
            descriptor[index // 8] += 2 ** (index % 8)

    return descriptor

def compute_descriptors(img, kp_arr):
    patch_size = 51
    descriptor_length = 256

    np.random.seed(0)
    pairs = set()
    while len(pairs) < descriptor_length:
        x, y = np.random.randint(0, patch_size, 2)
        if x == patch_size // 2 and y == patch_size // 2:
            continue
        pairs.add((x, y))

    pairs = list(pairs)

    descriptors = []
    result_kp_arr = []
    for kp in kp_arr:
        descriptor = brief(img, kp[0], kp[1], pairs, patch_size)
        if descriptor is not None:
            descriptors.append(descriptor)
            result_kp_arr.append(kp)
    
    return np.array(result_kp_arr), np.array(descriptors)

# function for keypoints and descriptors calculation
def detect_keypoints_and_calculate_descriptors(img):
    # img - numpy 2d array (grayscale image)
    img_blur = gaussian_blur(img, 31, 3)

    # keypoints
    kp_arr = find_keypoints_candidates(img_blur, 7)
    # kp_arr is array of 2d coordinates-tuples, example:
    # [(x0, y0), (x1, y1), ...]
    # xN, yN - integers

    # descriptors
    kp_arr, descr_arr = compute_descriptors(img_blur, kp_arr)
    # cv_descr_arr is array of descriptors (arrays), example:
    # [[v00, v01, v02, ...], [v10, v11, v12, ...], ...]
    # vNM - floats

    return kp_arr, descr_arr
