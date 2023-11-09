import pickle 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import imutils
import os
import glob
import cv2
import pickle 
import numpy as np
import cv2



def load_model():
    """
    Este método carga el modelo preentrenado

    :return Modelo entrenado 

    """
    with open("models/knn.model", "rb") as knn:
        model = pickle.load(knn)

    return model

def draw_square(frame, x, y, h, w, color):
    """
    Dibuja el cuadrado sobre las caras y pone el nombre

    :param frame

    :param x

    :param y

    :param h

    :param w

    :param name

    """
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 5)
    return frame

def draw_name(frame, x, y, h, name,color):
    """
    Dibuja el cuadrado sobre las caras y pone el nombre

    :param frame

    :param x

    :param y

    :param h

    :param w

    :param name

    """
    cv2.putText(frame, name, (x, y+h+35), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=color)
    return frame


# Load video
cap = cv2.VideoCapture('/data/videos/videoChorra3.mp4')

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
# eyeClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml.xml')
# smileClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

success = True
c = 0
while success:
    
    success, frame = cap.read()
    
    frame =  imutils.resize(frame, width=640)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    amarillo = (0,255,255)
    azul = (0,255,0)
    verde = (255,0,0)

    
    for (x,y,w,h) in faces:
        
        rostro = auxFrame[y:y+h,x:x+w]

        rostro = cv2.resize(rostro, (32, 32), interpolation=cv2.INTER_AREA).reshape(1, -1)

        frame = draw_square(frame=frame, x=x, y=y, w=w, h=h, color=amarillo)


    cv2.imshow('frame',frame)

    k =  cv2.waitKey(1)
    if k == 27:
        break
