from math import sin, cos, radians

import cv2
import imutils

# Rota la imagen los grados indicados
def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

# Corrige los puntos del frame rotado al frame sin rotar
def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1] * 0.35
    y = pos[1] - img.shape[0] * 0.35
    newx = x * cos(radians(angle)) + y * sin(radians(angle)) + img.shape[1] * 0.4
    newy = -x * sin(radians(angle)) + y * cos(radians(angle)) + img.shape[0] * 0.4
    if not -45 < angle < 45: newy - img.shape[0] * 2
    return int(newx), int(newy), pos[2], pos[3]


def get_faces(frame, angles):

    if not len(angles): angles = [0]

    # Haarcascade face classifiers
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
    # face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Por cada ángulo de rotación
    for angle in angles:

        # Rotar frame
        rimg = rotate_image(frame, angle=angle)

        # Detectar rostros
        faces = face_classifier.detectMultiScale(rimg, 1.3, 5)

        if len(faces):
            # Ajustar los puntos del rostro
            faces = [rotate_point(faces[-1], frame, angle=-angle)]
            break

    return faces
