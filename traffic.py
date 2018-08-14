import os
import logging
import logging.handlers
import random
import argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import utils

cv2.ocl.setUseOpenCL(False)
random.seed(123)

from pipeline import PipelineRunner
from contour_detection import ContourDetection
from visualizer import Visualizer
from csv_writer import CsvWriter
from vehicle_counter import VehicleCounter
from stabilize import Stabilizer

from video_writer import VideoWriter


# ============================================================================
IMAGE_DIR = "./out"
SHAPE = (720, 1280)  # HxW
CWD_PATH = os.getcwd()
EXIT_PTS = np.array([
    [[732, 720], [732, 590], [1280, 500], [1280, 720]],
    [[0, 400], [645, 400], [645, 0], [0, 0]]
])
# ============================================================================


def train_bg_subtractor(inst, stream, num=500):
    '''
        BG substractor need process some amount of frames to start giving result
    '''
    print('Training background subtractor')
    i = 0
    while i <= num:
        (grabbed, frame) = stream.read()
        inst.apply(frame, None, 0.001)
        i += 1
    
    return stream


def stabilize_frames(cap, log):
    path = os.path.join(CWD_PATH, 'vids', 'output.mp4')
    stabilizer = Stabilizer(cap, 100, path)

    log.info("Getting transformation matrices...")
    while True:
        ret, current = cap.read()
        if ret:
            log.info("looping...")
            stabilizer.get_transform_matrices(current[:, :, 0])
        else:
            break
    
    cap.release()
    log.info("Stabilizing frames...")
    stabilizer.stabilize_images()


def main():
    log = logging.getLogger('main')

    # creating exit mask from points, where we will be counting our vehicles
    base = np.zeros(SHAPE + (3,), dtype='uint8')
    exit_mask = cv2.fillPoly(base, EXIT_PTS, (255, 255, 255))[:, :, 0]
    stream = None
    # produce a stabilized video
    if args.stabilize_video == 'yes':
        cap = cv2.VideoCapture(args.video_source)
        stabilize_frames(cap, log)
        return
    else:
        stream = cv2.VideoCapture(args.video_source)
        stream.set(cv2.CAP_PROP_FRAME_WIDTH, SHAPE[1])
        stream.set(cv2.CAP_PROP_FRAME_HEIGHT, SHAPE[0])

    writer = VideoWriter('detected.mp4', (SHAPE[1], SHAPE[0]))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, detectShadows=True)
    # skipping 500 frames to train bg subtractor
    train_bg_subtractor(bg_subtractor, stream, num=500)

    pipeline = PipelineRunner(pipeline=[
        ContourDetection(bg_subtractor=bg_subtractor,
                         save_image=False, image_dir=IMAGE_DIR),
        # we use y_weight == 2.0 because traffic are moving vertically on video
        # use x_weight == 2.0 for horizontal.
        # VehicleCounter(exit_masks=[exit_mask], y_weight=2.0),
        VehicleCounter(),
        Visualizer(image_dir=IMAGE_DIR),
        CsvWriter(path='./', name='report.csv')
    ], log_level=logging.DEBUG)

    _frame_number = -1
    frame_number = -1

    while True:
        (grabbed, frame) = stream.read()
    
        if not frame.any():
            log.error("Frame capture failed, stopping...")
            break

        # real frame number
        _frame_number += 1

        # skip every 2nd frame to speed up processing
        if _frame_number % 2 != 0:
            continue

        # frame number that will be passed to pipline
        # this needed to make video from cutted frames
        frame_number += 1

        pipeline.set_context({
            'frame': frame,
            'frame_number': frame_number,
        })
        new_context = pipeline.run()

        cv2.imshow('Video', new_context['frame'])
        writer(new_context['frame'])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ============================================================================

if __name__ == "__main__":
    log = utils.init_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument('-src', '--source', dest='video_source', type=str,
                        default='raw/input_4.mp4', help='Source of the input video file')
    parser.add_argument('-stable', '--stabilize', dest='stabilize_video', type=str,
                        default='no', help='Stabilize video before background subtraction')
    args = parser.parse_args()

    if not os.path.exists(IMAGE_DIR):
        log.debug("Creating image directory `%s`...", IMAGE_DIR)
        os.makedirs(IMAGE_DIR)

    main()
