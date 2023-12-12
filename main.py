import os
import sys
import time
import pandas as pd
import numpy as np
import dlib

import cv2
import imutils
from shutil import rmtree
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
        cap = cv2.VideoCapture('data/test/EXAMPLE.MOV')

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
        cap = cv2.VideoCapture('data/test/EXAMPLE.MOV')

    success, frame = cap.read()
    while success:

        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:
            rostro = frame[y:y+h, x:x+w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            probs = model.predict_proba(rostro.reshape(1, -1))
            if any(prob > 0.8 for prob in probs[0]): name = Model.predict_one(model, rostro)
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

    p = "models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    if realtime:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture('data/test/EXAMPLE.MOV')

    success, frame = cap.read()
    while success:

        frame = imutils.resize(frame, width=640)
        faces, squares = Recognition.get_distances(frame, detector, predictor)

        # Por cada cara detectada
        for face, (x, y, h, w) in zip(faces, squares):
            
            df = pd.DataFrame([face], columns=headers[:-1])
            
            probs = model.predict_proba(df)
            if any(prob > 0.5 for prob in probs[0]):
                name = model.predict(df)
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

    if version == 'v1':

        print("La versión 1 no requiere entrenamiento")
        exit(1)

    videos = os.listdir('data/train')

    if len(videos) == 0:
        print("\nNo hay vídeos para el entrenamiento:\n"
            " - Introducir vídeos en 'data/train'\n"
            " - El nombre de los vídeos deben ser el nombre de la persona que sale en él\n"
            "    (Ejemplo: Pedro.MOV, Maria.mp4, etc.)\n")
        exit(1)

    # model = MLPClassifier(hidden_layer_sizes=[20, 40], activation='relu', early_stopping=True,
    #     random_state=13, max_iter=10000, solver='adam', verbose=False)
    
    model = LogisticRegression(max_iter=10000)

    if version == 'v2':
        
        for video in videos:
            Data.generate_data_as_images(f'data/train/{video}', video.split('.')[0], 'data/train_images')

        images_model = Model.train_model('data/train_images/', model)
        Model.save_model(images_model, 'v2')
        rmtree("data/train_images")


    elif version == 'v3':

        data = []
        for video in videos:
            landmarks =  Data.generate_data_as_landmarks(f'data/train/{video}', video.split('.')[0])
            data += [elem for elem in landmarks]
    
        landmarks_df = pd.DataFrame(data, columns=headers)
        # landmarks_df.to_csv(f'train.csv')
        landmarks_model = Model.train_model_df(landmarks_df, model)
        Model.save_model(landmarks_model, 'v3')


elif task == 'test':

    if version == 'v2':
        try:
            model = Model.load_model("models/v2.model")

        except:
            print("Es necesario entrenar un modelo primero")
            exit(1)

        version2(model, realtime=False)

    elif version == 'v3':
        try:
            model = Model.load_model("models/v3.model")
            
        except:
            print("Es necesario entrenar un modelo primero")
            exit(1)

        version3(Model.load_model("models/v3.model"), realtime=False)

    else:
        version1(realtime=False)

elif task == 'example':

    if version == 'v2':
        version2(Model.load_model("models/example_v2.model"), realtime=False)

    elif version == 'v3':
        version3(Model.load_model("models/example_v3.model"), realtime=False)

    else:
        version1(realtime=False)

else:
    print("Especificar tarea (train / test / example)")
