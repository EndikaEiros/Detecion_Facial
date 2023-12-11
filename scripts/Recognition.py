import cv2
import dlib
import imutils
import numpy as np

from scripts import Data
from math import sin, cos, radians

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

""" Obtiene los datos de la imagen (distancias y colores) """

def get_distances(frame):

    # Cargar el modelo de reconocimiento de landmarks
    p = "models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    data = []
    all_squares = []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = imutils.face_utils.shape_to_np(shape)

        data.append(Data.generate_dist_from_frame(shape, frame))

        # Get square coords
        all_squares.append([shape[0][0], shape[23][1], shape[16][0]-shape[0][0], shape[8][1]-shape[23][1]])


    return data, all_squares

""" Dibuja el cuadrado sobre las caras y pone el nombre """

def draw_square(frame, x, y, h, w, name):
 
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 6)
    cv2.putText(frame, name, (x, y + h + 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 1, cv2.LINE_AA)

    return frame

""" Dibuja el circulo en el punto indicado. Utlizado para representar los landmarks faciales """

def draw_circle(frame, x, y, t=2):

    frame = cv2.circle(frame, (x, y), t, (0, 0, 255), -1)

    return frame
