import argparse
import os
import cv2
import numpy as np

from utils import config

DATA_FOLDER = 'data'
RESULT_FOLDER = 'result'
IMAGES_FOLDER = 'img'

HIST_RANGES = [0,180]
HIST_SIZE = [180]
SCALE = 1
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
RECT_COLOR = (255,0,0)
RECT_SIZE = 2
SPEED = 30

def get_roi_hist(roi_frame, dataset):
    hsv_roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi_frame, np.array(config[dataset]['filter_from']), np.array(config[dataset]['filter_to']))
    roi_hist = cv2.calcHist([hsv_roi_frame], config[dataset]['channels'], mask, HIST_SIZE, HIST_RANGES)
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist

def meanshift(images, roi, dataset):
    x, y, width, height = roi
    first_frame, *frames = images

    roi_frame = first_frame[y: y + height, x: x + width]
    cv2.imshow("First Frame", roi_frame)
    roi_hist = get_roi_hist(roi_frame, dataset)

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], config[dataset]['channels'], roi_hist, HIST_RANGES, SCALE)
        _, track_window = cv2.meanShift(mask, (x, y, width, height), TERM_CRITERIA)
        x, y, w, h = track_window

        cv2.rectangle(frame, (x, y), (x + w, y + h), RECT_COLOR, RECT_SIZE)
        cv2.imshow("Mask", mask)
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(SPEED)

    cv2.destroyAllWindows()
    record_video(frames, os.path.join(dataset, 'meanshift'))

def camshift(images, roi, dataset):
    x, y, width, height = roi
    first_frame, *frames = images

    roi_frame = first_frame[y: y + height, x: x + width]
    cv2.imshow("First Frame", roi_frame)
    roi_hist = get_roi_hist(roi_frame, dataset)

    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.calcBackProject([hsv], config[dataset]['channels'], roi_hist, HIST_RANGES, SCALE)
        ret, track_window = cv2.CamShift(mask, (x, y, width, height), TERM_CRITERIA)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)

        cv2.polylines(frame, [pts], True, RECT_COLOR, RECT_SIZE)
        cv2.imshow('Mask', mask)
        cv2.imshow('Frame', frame)

        key = cv2.waitKey(SPEED)

    cv2.destroyAllWindows()
    record_video(frames, os.path.join(dataset, 'camshift'))


def record_video(frames, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, layers = frames[0].shape
    video_path = os.path.join(RESULT_FOLDER, path + '.mp4')
    if not os.path.exists(os.path.dirname(video_path)):
        os.makedirs(os.path.dirname(video_path))
    fps = 60
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        video.write(frame)

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
    meanshift(np.copy(images), roi, dataset)
    camshift(np.copy(images), roi, dataset)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--roi', type=int, nargs=4, help='(x, y, width, height)')

args = parser.parse_args()

main(args)
