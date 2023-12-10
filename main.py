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
        cap = cv2.VideoCapture('data/videos/example.MOV')

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
        cap = cv2.VideoCapture('data/videos/example.MOV')

    success, frame = cap.read()
    while success:

        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:
            rostro = frame[y - 5:y + h + 5, x - 5:x + w + 5]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            probs = model.predict_proba(rostro.reshape(1, -1))
            if any(prob > 0.999999 for prob in probs[0]):
                name = Model.predict_one(model, rostro)
            else:
                name = ['Desconocido']
            frame = Recognition.draw_square(frame=frame, x=x, y=y, w=w, h=h, name=name[0])

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

    model = MLPClassifier(
        hidden_layer_sizes=[20, 40],
        activation='relu',
        early_stopping=True,
        random_state=13,
        max_iter=1000,
        solver='adam',
        verbose=False
    )

    if version == 'v2':
        Data.generate_data_as_images(f'data/videos/{video}', {video.split('.')[0]}, 'data/train_images_data')
        images_model = Model.train_model('../data/train_images_data/', model)
        Model.save_model(images_model, 'mlp_v2.model')

    elif version == 'v3':
        Data.generate_data_as_landmarks(f'data/videos/{video}', {video.split('.')[0]}, 'data/train_csv_data', False)
        landmarks_model = Model.train_model('../data/train_csv_data/', model)
        Model.save_model(landmarks_model, 'mlp_v3.model')

    else:
        print("Especificar versión (v2 / v3)")

elif task == 'test':

    if version == 'v2':
        version2(Model.load_model("../models/mlp_v2.model"))

    elif version == 'v3':
        version3(Model.load_model("../models/mlp_v3.model"))

    else:
        version1()

elif task == 'example':

    if version == 'v2':
        version2(Model.load_model("../models/example_mlp_v2.model"), realtime=False)

    elif version == 'v3':
        version3(Model.load_model("../models/example_mlp_v3.model"), realtime=False)

    else:
        version1(realtime=False)

else:
    print("Especificar tarea (collect / train / test / example)")
