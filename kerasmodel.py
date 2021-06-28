import base64
import tensorflow as tf
import cv2
import numpy
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
#from tensorflow import keras

classes = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'i': 6,
    'k': 7,
    'l': 8,
    'm': 9,
    'n': 10,
    'o': 11,
    'p': 12,
    'q': 13,
    'r': 14,
    't': 15,
    'u': 16,
    'v': 17,
    'w': 18,
    'x': 19,
    'y': 20,
    
}
    
model = keras.models.load_model('model_tf1.5.model', compile=False) # load model
model._make_predict_function()
graph = tf.get_default_graph()

def prediction(data):
    img = data[22:]
    letter = CargarImagen(img)
    return letter
    
    
def identifyGesture(handTrainImage):

    # convertimos la imagen a la misma resolución que los datos de entrenamiento rellenando para alcanzar una relación de aspecto 1: 1 y luego
    # redimensionando a 400 x 400. Lo mismo se hace con los datos de entrenamiento en preprocess_image.py. La imagen de OpenCV es la primera.
    # convertido a imagen de almohada para hacer esto.
    handTrainImage = cv2.cvtColor(handTrainImage, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(handTrainImage)
    img_w, img_h = img.size
    M = min(img_w, img_h)
    background = Image.new('RGB', (M, M), (0, 0, 0))
    bg_w, bg_h = background.size
    offset = (int((bg_w - img_w) / 2),(int((bg_h - img_h) / 2))) 
    print(offset)
    background.paste(img, offset)
    size = 32,32
    background = background.resize(size, Image.ANTIALIAS)
    #print(background)
    # Guardamos la imagen procesada para verificarla nosotros mismo.
    background.save("img_32.jpeg")
    #get image as numpy array and predict using model
    # obtenemos la imagen como un arreglo por medio de numpy y usamos el modelo para predecir
    #open_cv_image = numpy.array(background)
    #background = open_cv_image.astype('float32')
    #background = background / 255
    #background = background.reshape((1,) + background.shape)
    #background = background.resize(32,32, Image.ANTIALIAS)
    #background.save("img_32.jpeg")

    #background = image.load_img("img_32.jpeg", target_size=(32, 32))
    #background = image.img_to_array(background)
    #img_prueba = image.array_to_img(background)
    #background = numpy.expand_dims(background, axis=0)

    #img_prueba.save("img_prueba.jpeg")
    
    new = numpy.array(background)

    label = predict(new)

    # print predicted class and get the class name (character name) for the given class number and return it
    #Imprimimos la letra que se da por medio de la prediccion.
    #print ('Esta es la letra',predictions)
    #key = (key for key, value in classes.items() if value == predictions[0]).__next__()
    #print(key)
    #print("")
    #print("")

    return label

def predict(data):
   data = numpy.expand_dims(data, axis=0)
   pred = model.predict(data)  
   pred = pred.argmax(axis=1)
   label_pred = pred[0]
   letra_pred = list(classes.keys())[list(classes.values()).index(label_pred)]          
   return letra_pred

def CargarImagen(img):
    imgdata=base64.b64decode(img)
    #fullpath2="C:/Users/edwar/Desktop/24245e7fd492f3df563284726d559538b4b935b2a06a32e547a7f605caaba3ea/a/a_001.jpg"
    with open("C:/Users/Oscar Rodriguez/Documents/Proyecto LSC/Flask_App/imgdelete.png", 'wb') as f:
        f.write(imgdata)
    fullpath2="C:/Users/Oscar Rodriguez/Documents/Proyecto LSC/Flask_App/imgdelete.png"
    # Create a window to display the camera feed



    # Get pointer to video frames from primary device

    min_YCrCb = numpy.array([0,130,72], numpy.uint8)
    max_YCrCb = numpy.array([255,180,130], numpy.uint8)


    # cascade xml file for detecting palm. Haar classifier
    #palm_cascade = cv2.CascadeClassifier('/content/drive/My Drive/ASL-Finger-Spelling-Recognition-master/palm.xml')

    # previous values of cropped variable
    #x_crop_prev, y_crop_prev, w_crop_prev, h_crop_prev = 0, 0, 0, 0

    # previous cropped frame if we need to compare histograms of previous image with this to see the change.
    # Not used but may need later.


    #prevcnt = numpy.array([], dtype=numpy.int32)

    # gesture static increments when gesture doesn't change till it reaches 10 (frames) and then resets to 0.
    # gesture detected is set to 10 when gesture static reaches 10."Gesture Detected is displayed for next
    # 10 frames till gestureDetected decrements to 0.

    # Getting min and max colors for skin
    #min_YCrCb = numpy.array([0,130,103], numpy.uint8)
    #max_YCrCb = numpy.array([255,182,130], numpy.uint8)

    sourceImage = cv2.imread(fullpath2) #Se Carga la imagen
    

    # Convierte la Imagen a escala de grises y se le hace un desenfoque Gausiano
    imageYCrCb = cv2.cvtColor(sourceImage, cv2.COLOR_BGR2YCR_CB)
    imageYCrCb = cv2.GaussianBlur(imageYCrCb, (5, 5), 0)

    # Find region with skin tone in YCrCb image - Busca en la Region de la imagen los tonos que coincidan entre los rangos min y max
    skinRegion = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    # Do contour detection on skin region , Convierte la imagen en contornos los cuales se hacen de forma jerarquica
    contours, hierarchy = cv2.findContours(skinRegion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # sorting contours by area. Largest area first. Se clasifican los contornos encontrados 
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # get largest contour and compare with largest contour from previous frame.
    # set previous contour to this one after comparison.
    cnt = contours[0] # Guarda los contornos y los compara con los contornos anteriores de la imagen que estaba entre los rangos min y max
    #ret = cv2.matchShapes(cnt, prevcnt, 2, 0.0) 
    #prevcnt = contours[0]

    # once we get contour, extract it without background into a new window called handTrainImage
    stencil = numpy.zeros(sourceImage.shape).astype(sourceImage.dtype) # Se obtienen los contornos despues de la comparacion
                                                                        # y se extrae la imagen de ellos sin el fondo.
    color = [255, 255, 255]
    cv2.fillPoly(stencil, [cnt], color)
    handTrainImage = cv2.bitwise_and(sourceImage, stencil)  # Aqui se guarda ya la imagen procesada en escala de grises 
                                                            # por medio de compuertas logicas se extraen los detalles de la imagen


    # if comparison returns a high value (shapes are different), start gestureStatic over. Else increment it.


    # crop coordinates for hand.
    #x_crop, y_crop, w_crop, h_crop = cv2.boundingRect(cnt)

    # place a rectange around the hand.
    #cv2.rectangle(sourceImage, (x_crop, y_crop), (x_crop + w_crop, y_crop + h_crop), (0, 255, 0), 2)

    # create crop image
    #handImage = sourceImage.copy()[max(0, y_crop_prev - 50):y_crop_prev + h_crop_prev + 50,
    #           max(0, x_crop_prev - 50):x_crop_prev + w_crop_prev + 50]

    ImagenProcesada = handTrainImage
    ImagenProcesada = Image.fromarray(ImagenProcesada)
    ImagenProcesada.save("C:/Users/Oscar Rodriguez/Documents/Proyecto LSC/Flask_App/imgdelete2.jpeg")

    # Training image with black background
    #handTrainImage = handTrainImage[max(0, y_crop_prev - 15):y_crop_prev + h_crop_prev + 15,
    #max(0, x_crop_prev - 15):x_crop_prev + w_crop_prev + 15]

    print("Gesture Detected")
    letterDetected = identifyGesture(handTrainImage)  # Se envia la imagen procesada para realizar la prediccion.

    return letterDetected
    
    
    

"""with open("C:/Users/edwar/Desktop/ProyectoV2/Prueba/LSCCompleto/static/Model/ImagenPredicta/imgdelete.png", 'wb') as f:
f.write(imgdata)"""
    






        
        
        
        
