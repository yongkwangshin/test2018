import tensorflow as tf
import numpy as np

import argparse
import glob
""" 

python  test_image_on_model.py --train_images="data1/"


"""

" 1 set argparse "
parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str  )
args = parser.parse_args()

" 2 get path "
images_path = args.train_images
img = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
img.sort()	

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
b=getimageArr(img,400,600) # it have to getimageArr !! because input Dimension

b = b.astype('float32') # it is necessary !!


W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(b, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)