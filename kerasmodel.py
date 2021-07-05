import base64
import tensorflow as tf
import time
from object_detection.utils import label_map_util
import cv2
import numpy as np
#import math
import os
import keras
from keras.preprocessing import image
#from keras.optimizers import SGD
#from keras.models import Sequential
#from keras.layers.core import Dense, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.utils import np_utils
from PIL import Image
from pathlib import Path
#from tensorflow import keras
    
model = tf.saved_model.load('saved_model') # load model
category_index=label_map_util.create_category_index_from_labelmap("label_map.pbtxt",use_display_name=True)

def prediction(data):
    img = data[22:]
    imgPath = CargarImagen(img)
    letterDetected = detectLetter(imgPath)
    print(letterDetected)
    return letterDetected
    
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def detectLetter(imgPath):

    image_np=load_image_into_numpy_array(imgPath)
    img_path = Path(imgPath)
    
    input_tensor=tf.convert_to_tensor(image_np)
    input_tensor=input_tensor[tf.newaxis, ...]
    input_tensor=input_tensor[:, :, :, :3]

    with tf.device("cpu:0"): detections=model(input_tensor)

    num_detections=int(detections.pop('num_detections'))
    detections={key:value[0,:num_detections].numpy()
                   for key,value in detections.items()}
    detections['num_detections']=num_detections
    classes = detections['detection_classes']
    detections['detection_classes']=detections['detection_classes'].astype(np.int64)

    image_np_with_detections=image_np.copy()

    print(category_index.get(classes[0]))
    # im = Image.fromarray(image_np_with_detections)
    # im.save('C:/Users/oscar/Documents/Backup HDD/testImg/'+ img_path.name)
    return category_index.get(classes[0])

def CargarImagen(img):
    imgdata=base64.b64decode(img)
    #fullpath2="C:/Users/edwar/Desktop/24245e7fd492f3df563284726d559538b4b935b2a06a32e547a7f605caaba3ea/a/a_001.jpg"
    with open("C:/Users/oscar/Documents/Backup HDD/imgdelete.png", 'wb') as f:
        f.write(imgdata)
    fullpath2="C:/Users/oscar/Documents/Backup HDD/imgdelete.png"
    # Create a window to display the camera feed
    print("Gesture Detected", fullpath2)
    
    return fullpath2