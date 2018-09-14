from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # shape = [row,col,channel,num]
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W): # W: [filter_height, filter_width, in_channels, out_channels]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b

import argparse
import glob
""" 

python  train.py \
--train_images="data/dataset1/images_prepped_train/" \
--train_annotations="data/dataset1/annotations_prepped_train/" \
--val_images="data/dataset1/images_prepped_test/" \
--val_annotations="data/dataset1/annotations_prepped_test/"

python  image_distributer.py --train_images="data1/" --train_annotations="data2/"


python  testnet.py --train_images="data1/"


python  cifartest2.py \
--train_images="data/dataset1/train1/" \
--train_annotations="data/dataset1/train2/" \
--val_images="data/dataset1/test1/" \
--val_annotations="data/dataset1/test2/" 


"""
import random

" 1 set argparse "
parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str  )
#parser.add_argument("--train_annotations", type = str  )
#parser.add_argument("--val_images", type = str , default = "")
#parser.add_argument("--val_annotations", type = str , default = "")
args = parser.parse_args()

" 2 get path "
images_path = args.train_images
#segs_path = args.train_annotations
#val_images_path = args.val_images
#val_segs_path = args.val_annotations
img = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )

random.shuffle(img) # shuffle

#seg  = glob.glob( segs_path + "*.jpg"  ) + glob.glob( segs_path + "*.png"  ) +  glob.glob( segs_path + "*.jpeg"  )
#seg.sort()

import cv2

def getimage(path,height,width): # path isnt array, one path
	img = cv2.imread(path, 1)
	img = cv2.resize(img, ( width , height ))
	return img

def getimageArr(path,height,width): # path is array
    imgArr=[]
    n=len(path)
    for i in range(n):
        img = cv2.imread(path[i], 1)
        img = cv2.resize(img, ( width , height )) # warning !! w,h order
        imgArr.append(img)
	return np.array(imgArr)

b=getimage(img[0],500,350)
b=getimageArr(img,640,480)


b = b.astype('float32') # it is necessary !!

"3 Model"
x=b
conv1 = conv_layer(x,[5,5,3,8])
pool1 = max_pool_2x2(conv1)

conv2 = conv_layer(pool1,[5,5,8,16])
pool2 = max_pool_2x2(conv2)

conv3 = conv_layer(pool2,[5,5,16,32])
pool3 = max_pool_2x2(conv3)

conv4 = conv_layer(pool3,[5,5,32,64])
pool4 = max_pool_2x2(conv4)

conv5 = conv_layer(pool4,[5,5,64,64])
pool5 = max_pool_2x2(conv5)

flat6=tf.reshape(pool5,[-1,20*15*64]) # w,h,c
full6=tf.nn.relu(full_layer(flat6,))


# W_fc1 = tf.Variable(tf.truncated_normal(shape=[20 * 15 * 64, 16], stddev=0.1)) # h,w,c,num of 
# b_fc1 = tf.Variable(tf.constant(0.1, shape=[16]))
# h_conv4_flat = tf.reshape(h_conv4, [-1, 40*30*32])
