from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import sys
from collections import defaultdict
from threading import Thread

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue

import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False

        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                if not grabbed:
                    self.stop()
                    return

                # add the frame to the queue
                self.Q.put(frame)

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type=str,
        help='Video file name.'
    )
    parser.add_argument(
        '-out',
        '--output-dir',
        dest='output_dir',
        type=str,
        default='.tmp/',
        help='Output directory'
    )
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=float,
        default=13.057672,
        help='FPS.'
    )
    parser.add_argument(
        '-codec',
        '--codec',
        dest='codec',
        type=str,
        default='XVID',
        help='codec MJPG or XVID'
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TRAIN.WEIGHTS = ''
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg()
    model = infer_engine.initialize_model_from_cfg()
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

    # start the file video stream thread and allow the buffer to
    # start to fill
    logger.info("[INFO] starting video file thread...")

    for file in os.listdir("/samples"):
        if file.endswith(".mp4"):
            logger.info('Processing %s', file)
            fvs = FileVideoStream('/samples/' +file).start()
            time.sleep(1.0)
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*args.codec)
            writer = None

            target_dir = args.output_dir
            ensure_dir(target_dir)

            # loop over frames from the video file stream
            while fvs.more():
                im = fvs.read()
                # check if the writer is None
                if writer is None:
                    # store the image dimensions, initialize the video writer
                    (h, w) = im.shape[:2]
                    writer = cv2.VideoWriter('{}/{}_{}.avi'.format(target_dir, 'out', int(time.time())), fourcc, args.fps,
                                             (w, h), True)

                timers = defaultdict(Timer)
                t = time.time()
                with c2_utils.NamedCudaScope(0):
                    cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                        model, im, None, timers=timers
                    )
                logger.info('Inference time: {:.3f}s'.format(time.time() - t))
                for k, v in timers.items():
                    logger.info(' | {}: {:.3f}s'.format(k, v.average_time))

                im = vis_utils.vis_one_image_opencv(
                    im,
                    cls_boxes,
                    segms=cls_segms,
                    keypoints=cls_keyps,
                    dataset=dummy_coco_dataset,
                    show_class=True,
                    show_box=True,
                    kp_thresh=2,
                    thresh=0.7,
                    bbox_thickness=3
                )

                writer.write(im)
                cv2.imshow('Detectron', im)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # do a bit of cleanup
            cv2.destroyAllWindows()
            writer.release()
            fvs.stop()


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
