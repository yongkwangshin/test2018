import tensorflow as tf
import numpy as np

import argparse
import glob
import cv2

""" 
python  test_getimage.py --train_images="data1/"

"""

" 1 set argparse "
parser = argparse.ArgumentParser()
parser.add_argument("--train_images", type = str  )
args = parser.parse_args()

" 2 get path "
images_path = args.train_images
img = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
img.sort()	


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

print(len(img))
a=getimageArr(img,400,600)
b=getimage(img[1],500,350)

cv2.imshow( 'load image', a[0])
key = cv2.waitKey(0) 
print('next')
cv2.imshow( 'load image', b)
key = cv2.waitKey(0) 