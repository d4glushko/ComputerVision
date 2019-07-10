import argparse
import os
import cv2
import numpy as np

from utils import config

DATA_FOLDER = 'data'
RESULT_FOLDER = 'result'
IMAGES_FOLDER = 'img'

def meanshift(images, roi, dataset):
    x, y, width, height = roi
    first_frame, *frames = images

    roi_frame = first_frame[y: y + height, x: x + width]
    cv2.imshow("First Frame", roi_frame)
    hsv_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    hist_size = [180]
    ranges = [0,180]
    mask = cv2.inRange(hsv_roi_frame, np.array(config[dataset]['filter_from']), np.array(config[dataset]['filter_to']))
    roi_hist = cv2.calcHist([hsv_roi_frame], config[dataset]['channels'], mask, hist_size, ranges)
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        scale = 1
        mask = cv2.calcBackProject([hsv], config[dataset]['channels'], roi_hist, ranges, scale)
        iterations = 1000
        step = 1
        term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, step)
        _, track_window = cv2.meanShift(mask, (x, y, width, height), term_criteria)
        x, y, w, h = track_window
        rect_color = (255,0,0)
        rect_size = 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, rect_size)

        cv2.imshow("Mask", mask)
        cv2.imshow("Frame", frame)

        speed = 30
        key = cv2.waitKey(speed)

    cv2.destroyAllWindows()
    record_video(frames, os.path.join(dataset, 'meanshift'))


def record_video(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, layers = frames[0].shape
    video_path = os.path.join(RESULT_FOLDER, path + '.mp4')
    fps = 60
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    print("{}".format(video_path))

    for frame in frames:
        video.write(frame)

    print("done")
    cv2.destroyAllWindows()
    video.release()


def get_images(path):
    path = os.path.join(DATA_FOLDER, path, IMAGES_FOLDER)
    sorted_names = sorted(os.listdir(path), key=lambda name: int(name.split('.')[0]))
    images = [cv2.imread(os.path.join(path, name)) for name in sorted_names]
    return images

def main(args):
    dataset = args.dataset
    roi = args.roi
    if dataset not in config:
        print("{} dataset does not exist".format(dataset))
        return
    if not roi:
        roi = config[dataset]['roi']

    images = get_images(dataset)
    meanshift(images, roi, dataset)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--roi', type=int, nargs=4, help='(x, y, width, height)')

args = parser.parse_args()

main(args)
