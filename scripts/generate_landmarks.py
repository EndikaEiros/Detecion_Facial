import cv2
import numpy as np
import pandas as pd
import dlib
import Recognition

from matplotlib import pyplot as plt
from imutils import face_utils
from math import sqrt, exp


def gen_data(shape):
    """
    Devuelve un array con la distancia desde cada punto hasta el punto 30, que es la punta de la nariz y nuestro centro de cara
    
    """
    distancias = []
    posiciones = []
    centroX, centroY = shape[30]

    # indices_posiciones = []
    # indices_distancias = []

    for (i, (x, y)) in enumerate(shape):

        posiciones.append((x - centroX))
        posiciones.append((y - centroY))

        distancias.append(sqrt(pow(0 - (x - centroX), 2) + pow(0 - (y - centroY), 2)))
    
    datos = posiciones + distancias
    return datos


def obtener_data_landmarks(frame, etiqueta):

    p = "../models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    h, w, c = frame.shape
    image = cv2.resize(frame, (int(w / 4), int(h / 4)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        df.loc[len(df)] = gen_data(shape)

        for (x, y) in shape:
            Recognition.draw_circle(image, x, y)

    cv2.imshow(image)
    cv2.waitKey(1)


    return data

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
                 '62_TO_00', '63_TO_00', '64_TO_00', '65_TO_00', '66_TO_00', '67_TO_00']

    df = pd.DataFrame([datos], columns=cabeceras)
    df.loc[len(df)] = data

    df.to_csv('folder/subfolder/out.csv') 
