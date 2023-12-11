from math import sin, cos, radians

import cv2
import dlib
import imutils
import numpy as np
import pandas as pd
from math import sqrt

""" Borrar el fondo de la imagen """


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


""" Rota la imagen los grados indicados """


def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result


""" Corrige los puntos del frame rotado al frame sin rotar """


def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1] * 0.35
    y = pos[1] - img.shape[0] * 0.35
    newx = x * cos(radians(angle)) + y * sin(radians(angle)) + img.shape[1] * 0.4
    newy = -x * sin(radians(angle)) + y * cos(radians(angle)) + img.shape[0] * 0.4
    if not -45 < angle < 45: newy - img.shape[0] * 2
    return int(newx), int(newy), pos[2], pos[3]


""" Obtiene todas las caras (también rotadas) de un frame """


def get_faces(frame, angles=None):
    if angles is None or not len(angles):
        angles = [0]

    # Haarcascade face classifiers
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    # face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = list()

    # Por cada ángulo de rotación
    for angle in angles:

        # Rotar frame
        rimg = rotate_image(frame, angle=angle)

        # Detectar rostros
        faces = face_classifier.detectMultiScale(rimg, 1.3, 5)

        for face in faces:
            # Ajustar los puntos del rostro
            detected_faces.append(rotate_point(face, frame, angle=-angle))

    return detected_faces


def get_distances(frame):

    p = "models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    all_faces = []
    all_squares = []

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = imutils.face_utils.shape_to_np(shape)

        h = sqrt((shape[0][0] - shape[16][0])**2 + (shape[0][1] - shape[16][1])**2)
        v = sqrt((shape[8][0] - shape[27][0])**2 + (shape[8][1] - shape[27][1])**2)

        data = []
        list1 = [36, 42, 27, 31, 17, 22, 48, 6, 8, 51]
        vertical = [28, 9, 52]
        list2 = [39, 45, 33, 35, 21, 26, 54, 10, 57, 33]

        # Get distances
        for a, b in zip(list1, list2):

            real_dist = sqrt((shape[a][0] - shape[b][0])**2 + (shape[a][1] - shape[b][1])**2)

            if a in vertical:
                data.append(real_dist / v)
            else:
                data.append(real_dist / h)

        # Get colors

        for n in frame[shape[2][1],shape[19][0]]:
            data.append(n)

        for n in frame[shape[14][1],shape[24][0]]:
            data.append(n)

        for n in frame[shape[5][1],shape[8][0]]:
            data.append(n)

        for n in frame[min(shape[19][1],shape[24][1]),shape[27][0]]:
            data.append(n)

        all_faces.append(data)

        # Get square coords
        all_squares.append([shape[0][0], shape[23][1], shape[16][0]-shape[0][0], shape[8][1]-shape[23][1]])


    return all_faces, all_squares


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
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 6)
    cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_circle(frame, x, y, t=2):
    """
    Dibuja el circulo en el punto indicado. Utlizado para representar los landmarks faciales

    :param frame

    :param x

    :param y

    :param h

    :param w

    :param name

    """
    frame = cv2.circle(frame, (x, y), t, (0, 0, 255), -1)

    return frame
