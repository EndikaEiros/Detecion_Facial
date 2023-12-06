import cv2
import os
import imutils
import numpy as np
import Recognition


def generate_data_as_images(video_path, person_name, images_path):

    images_path += f'/{person_name}'

    # Si el directorio no existe, se crea
    if not os.path.exists(images_path):
        print('Carpeta creada: ', images_path)
        os.makedirs(images_path)

    # Cargar video
    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    cap = cv2.VideoCapture(video_path)

    # Haarcascade face classifiers
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    # face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

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

def generate_data_as_landmarks(video_path, person_name, landsmarks_path):
    pass

