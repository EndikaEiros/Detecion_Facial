import os
import sys
import time
import pandas as pd
import numpy as np

import cv2
import imutils
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from scripts import Data
from scripts import Model
from scripts import Recognition


headers = ['right_eye', 'left_eye', 'nose_height', 'nose_width', 'right_eyebrow', 'left_eyebrow',
            'mouth', 'chin', 'dist_chin_mouth', 'dist_mouth_nose',
            'pixel_piel_1_B', 'pixel_piel_1_G', 'pixel_piel_1_R',
            'pixel_piel_2_B', 'pixel_piel_2_G', 'pixel_piel_2_R',
            'pixel_piel_3_B', 'pixel_piel_3_G', 'pixel_piel_3_R',
            'pixel_piel_4_B', 'pixel_piel_4_G', 'pixel_piel_4_R', 'Etiqueta']

def version1(realtime=True):
    """
    Esta versión únicamente detecta rostros en tiempo real
    """

    if realtime:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/videos/example.MOV')

    success, frame = cap.read()
    while success:

        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:
            frame = Recognition.draw_square(frame=frame, x=x, y=y, w=w, h=h, name='')

        cv2.imshow('frame', frame)

        success, frame = cap.read()

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()


def version2(model, realtime=True):
    """
    Esta versión detecta e identifica rostros en tiempo real.
    No realiza ningún preprocesado, solo usa las imágenes.
    """

    if realtime:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/test/TEST3.MOV')

    success, frame = cap.read()
    while success:

        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:
            rostro = frame[y - 5:y + h + 5, x - 5:x + w + 5]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            probs = model.predict_proba(rostro.reshape(1, -1))
            if any(prob > 0.999999 for prob in probs[0]): name = Model.predict_one(model, rostro)
            else: name = ['Desconocido']
            frame = Recognition.draw_square(frame=frame, x=x, y=y, w=w, h=h, name=name[0])

        cv2.imshow('frame', frame)

        success, frame = cap.read()

        k = cv2.waitKey(1)

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def version3(model, realtime=True):
    """
    Esta versión detecta e identifica rostros en tiempo real.
    Obtiene las distancias de los landmarks por cada rostro.
    """

    if realtime:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/test_data/videos/TEST3.MOV')

    success, frame = cap.read()
    while success:

        faces, squares = Recognition.get_distances(frame)

        # Por cada cara detectada
        for face, (x, y, h, w) in zip(faces, squares):
            
            df = pd.DataFrame([face], columns=headers[:-1])
            
            probs = model.predict_proba(df[headers[:-1]])
            if any(prob > 0.999999 for prob in probs[0]):
                name = model.predict(df[headers])
            else:
                name = ['Desconocido']

            frame = Recognition.draw_square(frame=frame, x=x, y=y, w=w, h=h, name=name[0])
        
        frame = imutils.resize(frame, width=640)
        cv2.imshow('frame', frame)

        success, frame = cap.read()

        k = cv2.waitKey(1)

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


try:
    task = sys.argv[1]
    version = sys.argv[2]

except:

    task = ''
    version = 'v1'

if task == 'train':

    videos = os.listdir('data/videos')

    # model = MLPClassifier(hidden_layer_sizes=[20, 40], activation='relu', early_stopping=True,
    #     random_state=13, max_iter=1000, solver='adam', verbose=False)
    
    model = LogisticRegression(max_iter=10000)
    data = []

    if version == 'v2':

        for video in videos:
            Data.generate_data_as_images(f'data/videos/{video}', {video.split('.')[0]}, 'data/train_images_data')
        
        images_model = Model.train_model('../data/train_images_data/', model)
        Model.save_model(images_model, 'v2.model')

    elif version == 'v3':

        data = []
        for video in videos:
            landmarks =  Data.generate_data_as_landmarks(f'data/videos/{video}', video.split('.')[0])
            for elem in landmarks:
                data.append(elem)
    
        landmarks_df = pd.DataFrame(data, columns=headers)
        landmarks_df.to_csv(f'train.csv')
        landmarks_model = Model.train_model_csv(landmarks_df, model)
        Model.save_model(landmarks_model, 'v3')

    else:
        print("Especificar versión (v2 / v3)")


elif task == 'test':

    if version == 'v2':
        version2(Model.load_model("models/v2.model"))

    elif version == 'v3':
        version3(Model.load_model("models/v3.model"), realtime=False)

    else:
        version1()

elif task == 'example':

    if version == 'v2':
        version2(Model.load_model("../models/example_v2.model"), realtime=False)

    elif version == 'v3':
        version3(Model.load_model("../models/example_v3.model"), realtime=False)

    else:
        version1(realtime=False)

else:
    print("Especificar tarea (collect / train / test / example)")
