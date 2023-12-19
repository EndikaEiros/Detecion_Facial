import cv2
import os
import imutils
import numpy as np
import pandas as pd
import dlib
import time
from scripts import Recognition

from time import sleep
from math import sqrt
from imutils import face_utils


def generate_data_as_images(video_path, person_name, images_path):

    images_path += f'/{person_name}'

    # Si el directorio no existe, se crea
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # Cargar video
    cap = cv2.VideoCapture(video_path)

    # Inicializaciones
    image_num = 0

    # Obtener primer frame
    success, frame = cap.read()

    print(f"\n Saving {person_name}'s faces...\n")

    # Por cada frame del video
    while success:

        # Preprocesar frame y obtener caras detectadas
        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:

            image_num += 1

            # Recortar rostro
            rostro = frame[y:y + h, x:x + w]
            try:
                rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            except:
                continue

            # Almacenar rostros, en la carpeta correspondiente
            cv2.imwrite(f'{images_path}/rostro_{str(image_num)}.jpg', rostro)

        # Obtener siguiente frame
        success, frame = cap.read()

    print(f'{image_num} imágenes de {person_name} obtenidas para el entrenamiento')
    cap.release()

def generate_data_as_landmarks(video_path, person_name, headers):

    # Cargar video
    cap = cv2.VideoCapture(video_path)

    # Cargar el modelo de reconocimiento de landmarks
    p = "models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Obtener primer frame
    success, frame = cap.read()

    # Almacenar caras
    data = []

    # Por cada frame del video
    while success:

        # Preprocesar frame y obtener caras detectadas
        frame = imutils.resize(frame, width=640)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            data.append(generate_dist_from_frame(shape, headers) + [person_name])

        # Obtener siguiente frame
        success, frame = cap.read()

    cap.release()

    return data


def generate_dist_from_frame(shape, headers):

    distancias = []

    for header in headers:
        if header != 'Etiqueta':
            desde, hasta = header.split('TO')
            distancias.append(sqrt((shape[int(desde)][0] - shape[int(hasta)][0])**2 + (shape[int(desde)][1] - shape[int(hasta)][1])**2))

    return distancias


