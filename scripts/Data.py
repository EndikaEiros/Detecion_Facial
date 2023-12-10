import cv2
import os
import imutils
import numpy as np
import pandas as pd
import Recognition
import dlib
import time

from time import sleep
from math import sqrt
from imutils import face_utils

cabeceras = ['0X', '0Y', '1X', '1Y', '2X', '2Y', '3X', '3Y', '4X', '4Y', '5X', '5Y', '6X', '6Y', '7X', '7Y', '8X',
                 '8Y', '9X', '9Y', '10X', '10Y', '11X', '11Y', '12X', '12Y', '13X', '13Y', '14X', '14Y', '15X', '15Y',
                 '16X', '16Y', '17X', '17Y', '18X', '18Y', '19X', '19Y', '20X', '20Y', '21X', '21Y', '22X', '22Y',
                 '23X', '23Y', '24X', '24Y', '25X', '25Y', '26X', '26Y', '27X', '27Y', '28X', '28Y', '29X', '29Y',
                 '30X', '30Y', '31X', '31Y', '32X', '32Y', '33X', '33Y', '34X', '34Y', '35X', '35Y', '36X', '36Y',
                 '37X', '37Y', '38X', '38Y', '39X', '39Y', '40X', '40Y', '41X', '41Y', '42X', '42Y', '43X', '43Y',
                 '44X', '44Y', '45X', '45Y', '46X', '46Y', '47X', '47Y', '48X', '48Y', '49X', '49Y', '50X', '50Y',
                 '51X', '51Y', '52X', '52Y', '53X', '53Y', '54X', '54Y', '55X', '55Y', '56X', '56Y', '57X', '57Y',
                 '58X', '58Y', '59X', '59Y', '60X', '60Y', '61X', '61Y', '62X', '62Y', '63X', '63Y', '64X', '64Y',
                 '65X', '65Y', '66X', '66Y', '67X', '67Y', '0_TO_00', '1_TO_00', '2_TO_00', '3_TO_00', '4_TO_00',
                 '5_TO_00', '6_TO_00', '7_TO_00', '8_TO_00', '9_TO_00', '10_TO_00', '11_TO_00', '12_TO_00', '13_TO_00',
                 '14_TO_00', '15_TO_00', '16_TO_00', '17_TO_00', '18_TO_00', '19_TO_00', '20_TO_00', '21_TO_00',
                 '22_TO_00', '23_TO_00', '24_TO_00', '25_TO_00', '26_TO_00', '27_TO_00', '28_TO_00', '29_TO_00',
                 '30_TO_00', '31_TO_00', '32_TO_00', '33_TO_00', '34_TO_00', '35_TO_00', '36_TO_00', '37_TO_00',
                 '38_TO_00', '39_TO_00', '40_TO_00', '41_TO_00', '42_TO_00', '43_TO_00', '44_TO_00', '45_TO_00',
                 '46_TO_00', '47_TO_00', '48_TO_00', '49_TO_00', '50_TO_00', '51_TO_00', '52_TO_00', '53_TO_00',
                 '54_TO_00', '55_TO_00', '56_TO_00', '57_TO_00', '58_TO_00', '59_TO_00', '60_TO_00', '61_TO_00',
                 '62_TO_00', '63_TO_00', '64_TO_00', '65_TO_00', '66_TO_00', '67_TO_00', 'Etiqueta']
def generate_data_as_images(video_path, person_name, images_path):

    images_path += f'/{person_name}'

    # Si el directorio no existe, se crea
    if not os.path.exists(images_path):
        print('Carpeta creada: ', images_path)
        os.makedirs(images_path)

    # Cargar video
    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(video_path)

    # Inicializaciones
    image_num = 0

    # Obtener primer frame
    success, frame = cap.read()

    print('\n Saving faces...\n')

    # Por cada frame del video
    while success:

        # Preprocesar frame y obtener caras detectadas
        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:

            image_num += 1

            # Recortar rostro
            rostro = frame[y-5:y + h+5, x-5:x + w+5]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            # Almacenar rostros, en la carpeta correspondiente
            cv2.imwrite(f'{images_path}/rostro_{str(image_num)}.jpg', rostro)

            # Visualiar el video
            # cv2.imshow('frame', frame)
            # cv2.imshow('masked_frame', frame_masked)

        # Obtener siguiente frame
        success, frame = cap.read()

        k = cv2.waitKey(1)
        if k == 27:
            break

    print(f'{image_num} imágenes de {person_name} almacenadas en: {images_path}')
    cap.release()
    cv2.destroyAllWindows()

def generate_data_as_landmarks(video_path, person_name, landmarks_path, render:bool= False):

    # Si el directorio no existe, se crea
    if not os.path.exists(video_path):
        print('Carpeta creada: ', video_path)
        os.makedirs(video_path)

    landmarks_df = pd.DataFrame(columns=cabeceras)

    # Cargar video
    cap = cv2.VideoCapture(video_path)

    # Cargar el modelo de reconocimiento de landmarks
    p = "models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # Obtener primer frame
    success, frame = cap.read()

    print(f'\n Saving faces landmarks from {video_path}\n')
    k = -1
    # Por cada frame del video
    start_time = time.time()
    while success:
        # print('fame')

        face_data = generate_data_from_frame(frame, detector, predictor)
        
        for num,  posiciones , distancias in face_data:
            landmarks_df.loc[len(landmarks_df)] =  posiciones + distancias + [person_name]

        if render:
            # cv2.rectangle(frame, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            # cv2.imshow('frame_landmarks', rostro)
            cv2.waitKey(1)
            k = cv2.waitKey(1)
            
        # Obtener siguiente frame
        success, frame = cap.read()
        
        if k == 27:
            break


    cap.release()
    cv2.destroyAllWindows()
    landmarks_df.to_csv(f'{landmarks_path}/{person_name}.csv')
    print(f'Se han guardado los datos faciales de {person_name} en {landmarks_path}/{person_name}.csv')
    print(f'Ha tardado {time.time()-start_time}s')

    return f'{landmarks_path}/{person_name}.csv'


def calcular_landmarks(shape, height, width):
    """
    Devuelve un array con la distancia desde cada punto hasta el punto 30, que es la punta de la nariz y nuestro centro de cara
    
    """

    distancias = []
    posiciones = []
    centroX, centroY = shape[30]

    for (i, (x, y)) in enumerate(shape):

        posiciones.append((x - centroX))
        posiciones.append((y - centroY))

        # Calcula la distancia entre el punto actual y el centro de la cara (la punta de la nariz) en proporcion al tamaño del frame
        distancias.append(sqrt(pow((x - centroX), 2) + pow((y - centroY), 2))/ sqrt(width**2 + height**2))
    
    return posiciones , distancias


def generate_data_from_frame(frame, detector, predictor):

    height, width, channels = frame.shape

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    faces_data = []
    
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        posiciones , distancias = calcular_landmarks(shape, height, width)
        
        faces_data.append( (str(int(i)),  posiciones , distancias) )

    return faces_data