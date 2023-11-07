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

def draw_square(frame, x, y, h, w, name):
    """
    Dibuja el cuadrado sobre las caras y pone el nombre

    :param frame

    :param x

    :param y

    :param h

    :param w

    :param name

    """
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 5)
    cv2.putText(frame, name, (x, y+h+35), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(0,255,255))
    return frame

# Load video
cap = cv2.VideoCapture('./videoChorra3.mp4')

#faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

success = True
c = 0
while success:
    
    success, frame = cap.read()
    
    frame =  imutils.resize(frame, width=640)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    model = load_model()

    for (x,y,w,h) in faces:
        
        rostro = auxFrame[y:y+h,x:x+w]

        rostro = cv2.resize(rostro, (32, 32), interpolation=cv2.INTER_AREA).reshape(1, -1)
        clase = model.predict(rostro)

        if clase[0] == 0: name = 'Alex'
        else: name = "Endika"

        frame = draw_square(frame=frame, x=x, y=y, w=w, h=h, name=name)

        if len(faces) > 1:
            cv2.imwrite(f"./prueba/doble_cara_{c}.jpg", frame)
            c += 1

    cv2.imshow('frame',frame)

    k =  cv2.waitKey(1)
    if k == 27:
        break