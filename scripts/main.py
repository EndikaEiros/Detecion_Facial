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
from PIL import ImageFont, ImageDraw, Image



def load_model(model_path):
    """
    Este método carga el modelo preentrenado

    :return Modelo entrenado 

    """
    with open(model_path, "rb") as knn:
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
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 6)
    #cv2.putText(frame, name, (x, y+h+70), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=4, color=(0,255,0))
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil_img).text((x, y+h), name, font=ImageFont.truetype("Lato-Black.ttf", 100), fill='#ffff00')
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return frame


def bgremove(frame, min_thres=90, min_satur=60, min_brigth=50):
    
    # Valores limite del color de la piel
    min_piel = np.array([min_thres, min_satur, min_brigth])
    max_piel = np.array([255, 255, 255])

    # Se aplica un blurr Gausioano para eliminar ruido y se convierte a HSV
    frameHSV = cv2.GaussianBlur(frame, (7, 7), 0)
    frameHSV = cv2.cvtColor(frameHSV, cv2.COLOR_RGB2HSV)

    # mascara que obtine el color del tono de piel del frame
    skinRegion = cv2.inRange(frameHSV, min_piel, max_piel)
    frame_skin = cv2.bitwise_and(frame, frame, mask=skinRegion)

    return frame_skin

# Load video
cap = cv2.VideoCapture('../data/videos/videoChorra3.mp4')

#faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')

success, frame = cap.read()
c = 0
while success:
    
    frame =  imutils.resize(frame, width=640)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray,1.3,5)

    model = load_model("../models/knn.model")

    for (x,y,w,h) in faces:
        

        #frame_masked = bgremove(auxFrame, 70, 30, 30)[y-10:y + h+10, x-10:x + w+10]
        frame_masked = auxFrame[y:y + h, x:x + w]

        rostro = cv2.resize(frame_masked, (32, 32), interpolation=cv2.INTER_AREA).reshape(1, -1)
        clase = model.predict(rostro)

        if clase[0] == 0: name = 'Alex'
        else: name = "Endika"

        
        frame = draw_square(frame=frame, x=x, y=y, w=w, h=h, name=name)
        
    
    cv2.imshow('frame', frame)

    success, frame = cap.read()

    k =  cv2.waitKey(1)
    if k == 27:
        break

cap.release()
