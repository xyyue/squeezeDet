# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import time
import sys
import os
import glob

import numpy as np
import tensorflow as tf

from config import *
from train import _draw_box
from nets import *
from utils import util

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")


def video_demo():
  """Detect videos."""

  cap = cv2.VideoCapture(FLAGS.input_path)

  # Define the codec and create VideoWriter object
  # fourcc = cv2.cv.CV_FOURCC(*'XVID')
  # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
  # in_file_name = os.path.split(FLAGS.input_path)[1]
  # out_file_name = os.path.join(FLAGS.out_dir, 'out_'+in_file_name)
  # out = cv2.VideoWriter(out_file_name, fourcc, 30.0, (375,1242), True)
  # out = VideoWriter(out_file_name, frameSize=(1242, 375))
  # out.open()

  with tf.Graph().as_default():
    # Load model
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      times = {}
      count = 0
      while cap.isOpened():
        t_start = time.time()
        count += 1
        out_im_name = os.path.join(FLAGS.out_dir, str(count).zfill(6)+'.jpg')

        # Load images from video and crop
        ret, frame = cap.read()
        if ret==True:
          # crop frames
          frame = frame[500:-205, 239:-439, :]
          im_input = frame.astype(np.float32) - mc.BGR_MEANS
        else:
          break

        t_reshape = time.time()
        times['reshape']= t_reshape - t_start

        # Detect
        # the shape of det_probs is [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES]
        det_boxes, det_probs, det_class = sess.run(
            [model.det_boxes, model.det_probs, model.det_class],
            feed_dict={model.image_input:[im_input], model.keep_prob: 1.0})

        t_detect = time.time()
        times['detect']= t_detect - t_reshape
        
        # Filter
        final_boxes, final_probs, final_class = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])

        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        t_filter = time.time()
        times['filter']= t_filter - t_detect

        # Draw boxes

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }
        _draw_box(
            frame, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr
        )

        t_draw = time.time()
        times['draw']= t_draw - t_filter

        cv2.imwrite(out_im_name, frame)
        # out.write(frame)

        times['total']= time.time() - t_start

        # time_str = ''
        # for t in times:
        #   time_str += '{} time: {:.4f} '.format(t[0], t[1])
        # time_str += '\n'
        time_str = 'Total time: {:.4f}, detection time: {:.4f}, filter time: '\
                   '{:.4f}'. \
            format(times['total'], times['detect'], times['filter'])

        print (time_str)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  # Release everything if job is finished
  cap.release()
  # out.release()
  cv2.destroyAllWindows()


def image_demo():
  """Detect image."""

  with tf.Graph().as_default():
    # Load model
    mc = kitti_squeezeDet_config()
    mc.BATCH_SIZE = 1
    # model parameters will be restored from checkpoint
    mc.LOAD_PRETRAINED_MODEL = False
    model = SqueezeDet(mc, FLAGS.gpu)

    saver = tf.train.Saver(model.model_params)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      saver.restore(sess, FLAGS.checkpoint)

      for f in glob.iglob(FLAGS.input_path):
        im = cv2.imread(f)
        im = im.astype(np.float32, copy=False)
        print("the imsize is: ", len(im))
        print("the imsize is: ", len(im[0]))
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
        print("the resized size is: ", mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT)
        input_image = im - mc.BGR_MEANS

        # Detect
        det_boxes, det_probs, det_class, det_pred_conf, det_pred_class_probs = sess.run(
            [model.det_boxes, model.det_probs, model.det_class, model.pred_conf, model.pred_class_probs],
            feed_dict={model.image_input:[input_image], model.keep_prob: 1.0})
        print("det_pred_conf =========", det_pred_conf, det_pred_conf.shape)
        print("pred_class_probs =========", det_pred_class_probs, det_pred_class_probs.shape)

        # Filter
        print("BEFORE!!!")
        final_boxes, final_probs, final_class, final_order = model.filter_prediction(
            det_boxes[0], det_probs[0], det_class[0])
        #print("AFTER!!!")
        #print("the order is: ", final_order)
        #print("final_probs is: ", final_probs)
        #final_index = final_order[np.argmax(final_probs)]
        #print("the index of max_prob is: ", final_order[np.argmax(final_probs)])
        #
        #print("the other one is: ", det_probs[0][final_order])
        #
        ## to compute iou, we need [cx, cy, width, height]
        #file_name = os.path.split(f)[1]
        #temp_dir  = os.path.split(f)[0]
        #print("temp_dir is: ", temp_dir)
        #label_file_name = os.path.join(temp_dir.replace('pics', 'labels'), file_name.replace('.png', '.txt'))
        #with open(label_file_name) as fl:
        #    line = fl.readline().strip().split(',')
        #    # left, top, right, bottom
        #    n0 = int(line[0])
        #    n1 = int(line[1])
        #    n2 = int(line[2])
        #    n3 = int(line[3])
        #gt_box = [(n0 + n2) / 2, (n1 + n3) / 2, abs(n2 - n0), abs(n1 - n3)]    
        #print('gt_box is: ', gt_box)
        #
        #print("========================================")
        #print("the iou is: ", util.iou(gt_box, det_boxes[0][final_index]))
        #print('The det_class is: ', det_class[0][final_index])
        #print('The det_pred_conf is: ', det_pred_conf[0][final_index])
        #print('The det_pred_class_probs is: ', det_pred_class_probs[0][final_index])
        #print('the det_prob is: ', det_probs[0][final_index])
        #print('the predicted box is: ', det_boxes[0][final_index])
        #print("========================================")

        mc.PLOT_PROB_THRESH = 0.15
        keep_idx    = [idx for idx in range(len(final_probs)) \
                          if final_probs[idx] > mc.PLOT_PROB_THRESH]
        print("mc.PLOT_PROB_THRESH = ", mc.PLOT_PROB_THRESH)
        final_boxes = [final_boxes[idx] for idx in keep_idx]
        final_probs = [final_probs[idx] for idx in keep_idx]
        final_class = [final_class[idx] for idx in keep_idx]

        # TODO(bichen): move this color dict to configuration file
        cls2clr = {
            'car': (255, 191, 0),
            'cyclist': (0, 191, 255),
            'pedestrian':(255, 0, 191)
        }

        # Draw boxes
        _draw_box(
            im, final_boxes,
            [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
                for idx, prob in zip(final_class, final_probs)],
            cdict=cls2clr,
        )
        #_draw_box(im, [gt_box], ["gt_box"])

        file_name = os.path.split(f)[1]
        out_file_name = os.path.join(FLAGS.out_dir, 'out_'+file_name)
        #out_text_file_name = os.path.join(FLAGS.out_dir, 'notfool', file_name.replace('.png', '.txt')) ###
        ## det_class, iou, pred_probs, pred_conf, pred_class_probs
        #with open(out_text_file_name, 'w') as fw:
        #    fw.write(str(det_class[0][final_index]) + ',' + str(util.iou(gt_box, det_boxes[0][final_index])) + ',' + str(det_probs[0][final_index]) +',' + str(det_pred_conf[0][final_index]) + ',' + str(det_pred_class_probs[0][final_index][0]) + ',' + str(det_pred_class_probs[0][final_index][1]) + ',' + str(det_pred_class_probs[0][final_index][2]) + ',' + str((n0 + n2) / 2) + ',' + str((n1 + n3) / 2))
            #fw.write(str(gt_box))
            #fw.write(str(det_boxes[0][final_index]))
        cv2.imwrite(out_file_name, im)
        print ('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
  if not tf.gfile.Exists(FLAGS.out_dir):
    tf.gfile.MakeDirs(FLAGS.out_dir)
  if FLAGS.mode == 'image':
    image_demo()
  else:
    video_demo()

if __name__ == '__main__':
    tf.app.run()
