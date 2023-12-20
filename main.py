import os
import sys
import cv2
import json
import dlib
import imutils
import pandas as pd

from shutil import rmtree
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from scripts import Data
from scripts import Model
from scripts import Recognition

CORRELATION_THRESHOLD = 0.6

# Load dataframe headers
with open('data/headers.json', 'r', encoding='utf-8') as file:
    headers = json.load(file)["headers"]

""" Ejecutar versión 1 """


def version1(realtime=True):
    """
    Esta versión únicamente detecta rostros en tiempo real
    """

    if realtime:
        cap = cv2.VideoCapture(0)
    else:
        try:
            cap = cv2.VideoCapture('data/test/EXAMPLE.MOV')
        except:
            print(" Error al abrir el vídeo, comprueba que exista data/test/EXAMPLE.MOV")
            exit(1)

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


""" Ejecutar versión 2 """


def version2(model, realtime=True):
    """
    Esta versión detecta e identifica rostros en tiempo real.
    No realiza ningún preprocesado, solo usa las imágenes.
    """

    if realtime:
        cap = cv2.VideoCapture(0)
    else:
        try:
            cap = cv2.VideoCapture('data/test/EXAMPLE.MOV')
        except:
            print(" Error al abrir el vídeo, comprueba que exista data/test/EXAMPLE.MOV")
            exit(1)

    success, frame = cap.read()
    while success:

        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:
            rostro = frame[y:y + h, x:x + w]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            probs = model.predict_proba(rostro.reshape(1, -1))
            if any(prob > 0.95 for prob in probs[0]):
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


""" Ejecutar versión 3 """


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
        try:
            cap = cv2.VideoCapture('data/test/EXAMPLE.MOV')
        except:
            print(" Error al abrir el vídeo, comprueba que exista data/test/EXAMPLE.MOV")
            exit(1)

    success, frame = cap.read()

    f_names = model.feature_names[:-1]

    while success:

        frame = imutils.resize(frame, width=640)
        faces, squares, shapes = Recognition.get_distances(frame, detector, predictor, f_names)

        # Por cada cara detectada
        for face, square, shape in zip(faces, squares, shapes):

            df = pd.DataFrame([face], columns=f_names)

            probs = model.predict_proba(df)

            if any(prob > 0.95 for prob in probs[0]):
                name = model.predict(df)
            else:
                name = ['Desconocido']

            frame = Recognition.draw_landmarks(frame=frame, face_shape=shape, square=square, name=name[0])

        cv2.imshow('frame', frame)

        success, frame = cap.read()

        k = cv2.waitKey(1)

        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


################## PARAMETERS ##################

try:
    task = sys.argv[1]
    version = sys.argv[2]

except:

    task = ''
    version = 'v1'

# ----------------------------------------------#
# ------------------- TRAIN --------------------#
# ----------------------------------------------#

if task == 'train':

    videos = os.listdir('data/train')
    if len(videos) == 0:
        print(" No hay vídeos para el entrenamiento:\n"
              " - Introducir vídeos en 'data/train'\n"
              " - El nombre de los vídeos deben ser el nombre de la persona que sale en él\n"
              "    (Ejemplo: Pedro.MOV, Maria.mp4, etc.)\n")
        exit(1)

    ###### TRAIN v1 ######
    if version == 'v1':
        print(" La versión 1 no requiere entrenamiento")
        exit(1)

    ###### TRAIN v2 ######
    if version == 'v2':

        model = MLPClassifier(hidden_layer_sizes=[20, 40], activation='relu', early_stopping=True,
                              random_state=13, max_iter=10000, solver='adam', verbose=False)

        for video in videos:
            Data.generate_data_as_images(f'data/train/{video}', video.split('.')[0], 'data/train_images')

        print(f" Entrenando MPL...", end="")
        images_model = Model.train_model('data/train_images/', model)
        Model.save_model(images_model, 'v2')
        rmtree("data/train_images")
        print(" ✓")

    ###### TRAIN v3 ######
    elif version == 'v3':

        data = []
        model = LogisticRegression(max_iter=10000)

        ### CALCULAR LANDMARKS ###

        for video in videos:
            print(f" Almacenando cara de {video.split('.')[0]}...", end="")
            landmarks = Data.generate_data_as_landmarks(f'data/train/{video}', video.split('.')[0], headers)
            data += [elem for elem in landmarks]
            print(" ✓")
        landmarks_df = pd.DataFrame(data, columns=headers)
        # landmarks_df.to_csv(f'train.csv') # if want to save dataset

        ### OPTIMIZAR DISTANCIAS ###

        mapeo = {}
        print(f' Optimizando el dataset...', end="")
        for i, nombre in enumerate(landmarks_df['Etiqueta'].unique()):
            mapeo.update({nombre: i})

        train_df_num = landmarks_df.copy()
        train_df_num['Etiqueta'] = landmarks_df['Etiqueta'].map(mapeo)

        corr_df = pd.DataFrame(data=train_df_num.corr()['Etiqueta'])
        corr_df.reset_index(inplace=True)
        corr_df = corr_df.rename({'index': 'Puntos', 'Etiqueta': 'Correlacion'}, axis=1)

        headers_max_corr = list(corr_df.loc[(abs(corr_df['Correlacion']) >= CORRELATION_THRESHOLD)]['Puntos'])
        print(" ✓")
        print(f'--> El número de columnas se ha reducido de {len(headers) - 1} a {len(headers_max_corr) - 1}')

        ### ENTRENAR MODELO ###

        print(f" Entrenando modelo de regresión logística...", end="")
        landmarks_model = Model.train_model_df(landmarks_df[headers_max_corr], model)
        landmarks_model.feature_names = headers_max_corr
        Model.save_model(landmarks_model, 'v3')
        print(" ✓")

# ----------------------------------------------#
# ------------------- TEST ---------------------#
# ----------------------------------------------#

elif task == 'test':

    ###### TEST v2 ######
    if version == 'v2':
        try:
            model = Model.load_model("models/v2.model")

        except:
            print(" Es necesario entrenar un modelo primero")
            exit(1)

        version2(model, realtime=True)

    ###### TEST v3 ######
    elif version == 'v3':

        try:
            model = Model.load_model("models/v3.model")


        except:
            print(" Es necesario entrenar un modelo primero")
            exit(1)

        version3(Model.load_model("models/v3.model"), realtime=True)

    ###### TEST v1 ######
    else:
        version1(realtime=False)

# ------------------------------------------------#
# -------------------- EXAMPLE -------------------#
# ------------------------------------------------#

elif task == 'example':

    ###### EXAMPLE v2 ######
    if version == 'v2':
        version2(Model.load_model("models/example_v2.model"), realtime=False)

    ###### EXAMPLE v3 ######
    elif version == 'v3':
        version3(Model.load_model("models/example_v3.model"), realtime=False)

    ###### EXAMPLE v1 ######
    else:
        version1(realtime=False)

else:
    print("Especificar tarea (train / test / example) y version (v1 / v2 / v3)")
