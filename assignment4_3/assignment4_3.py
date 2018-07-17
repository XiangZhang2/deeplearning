# -*- coding: utf-8 -*-
import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
import cv2



# sess = tf.Session()
# box_scores = tf.random_normal([19, 19, 5, 80], mean=0, stddev=1, seed = 1)
# box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
# box_classes = K.argmax(box_scores, axis=-1)  #返回的是最大值的位置，0-79
# filtering_mask = box_class_scores > 0.5
# e = tf.boolean_mask(box_class_scores, filtering_mask)
# print("e = " + str(e.eval(session=sess)))
# print e.shape
# print box_class_scores.shape
# print sess.run(filtering_mask)

# x = np.array([[1,2,3],[2,4,3],[4,3,1],[6,5,8]]) #shape=[4,3]
# y = np.argmax(x, axis = 1) #[2 1 0 2] shape=(4,)
# z = tf.reduce_max(x, axis = 1) #[3,4,4,8] shape=(4,)
# print(sess.run(z))

a = 0.1 * True
print a
# b = tf.nn.leaky_relu(a)
















