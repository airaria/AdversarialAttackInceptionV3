import numpy as np
from skimage.io import imread, imshow, imsave
import tensorflow as tf
import os

def load_images(input_dir, batch_shape,num=None):
    images = np.zeros(batch_shape)
    idx = 0
    batch_size = batch_shape[0]
    filenames = []
    filepaths = tf.gfile.Glob(os.path.join(input_dir,"*.png"))
    if num is None:
        num = len(filepaths)
    for filepath in filepaths:  #和python 自带的glob功能一样
        image = imread(filepath).astype(np.float)/255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx+=1
        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, images

def show_image(a):
    a = np.uint8((a+1.0)/2.0*255.0)
    imshow(a)

def save_image(fn,a):
    a = np.uint8((a+1.0)/2.0*255.0)
    imsave(fn,a)
