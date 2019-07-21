import argparse
import os
import cv2
import numpy as np

from utils import config

DATA_FOLDER = 'data'
RESULT_FOLDER = 'result'
IMAGES_FOLDER = 'img'

#LK
WIN_SIZE = (15,15)
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)

POINTS = 100

SPEED = 30

OPACITY = 0.99

def lk(images, dataset, pyramids_max_level=0):
    p0 = np.array(np.random.rand(POINTS,1,2), dtype=np.float32)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = WIN_SIZE,
                    maxLevel = pyramids_max_level,
                    criteria = TERM_CRITERIA)

    # Create some random colors
    color = np.random.randint(0,255,(POINTS,3))

    # Take first frame and find corners in it
    first_frame, *frames = images
    h, w, _ = first_frame.shape
    p0[:,:,0] *= w
    p0[:,:,1] *= h
    old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)

    for frame in frames:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # delete redundant colors
        delete_elements = np.where(st == 0)[0]
        color = np.delete(color, delete_elements, 0)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new, good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        frame = cv2.add(frame,mask)
        cv2.imshow('Frame with {} pyramid levels'.format(pyramids_max_level), frame)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        # Update mask to hide previous points over time using opacity
        np.multiply(mask, OPACITY, out=mask, casting='unsafe')

        key = cv2.waitKey(SPEED)

    cv2.destroyAllWindows()
    record_video(frames, os.path.join(dataset, 'lk-{}'.format(pyramids_max_level)))


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
    np.random.seed(1)
    dataset = args.dataset
    if dataset not in config:
        print("{} dataset does not exist".format(dataset))
        return

    images = get_images(dataset)
    lk(np.copy(images), dataset)
    lk_pyramids_max_level = 10
    lk(np.copy(images), dataset, lk_pyramids_max_level)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)

args = parser.parse_args()

main(args)
