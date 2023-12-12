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

def generate_data_as_landmarks(video_path, person_name):

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

            data.append(generate_dist_from_frame(shape, frame) + [person_name])

        # Obtener siguiente frame
        success, frame = cap.read()

    cap.release()

    return data


def generate_dist_from_frame(shape, frame):

    h = sqrt((shape[0][0] - shape[16][0])**2 + (shape[0][1] - shape[16][1])**2)
    v = sqrt((shape[8][0] - shape[27][0])**2 + (shape[8][1] - shape[27][1])**2)

    distancias = []
    list1 = [36, 42, 27, 31, 17, 22, 48, 6, 8, 51]
    vertical = [28, 9, 52]
    list2 = [39, 45, 33, 35, 21, 26, 54, 10, 57, 33]

    for a, b in zip(list1, list2):

        real_dist = sqrt((shape[a][0] - shape[b][0])**2 + (shape[a][1] - shape[b][1])**2)

        if a in vertical:
            distancias.append(real_dist / v)
        else:
            distancias.append(real_dist / h)

    extra = []

    for n in frame[shape[2][1],shape[19][0]]:
        extra.append(n)

    for n in frame[shape[14][1],shape[24][0]]:
        extra.append(n)

    for n in frame[shape[5][1],shape[8][0]]:
        extra.append(n)

    for n in frame[min(shape[19][1],shape[24][1]),shape[27][0]]:
        extra.append(n)

    return distancias + extra


