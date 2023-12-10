import cv2
import imutils
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import Model
import dlib
import Model_landmarks
import Data
import Recognition
from imutils import face_utils
from Recognition import draw_square

def show_example():
    # Load video
    cap = cv2.VideoCapture('../data/videos/TEST3.MOV')

    # # Load classifier model
    # model = Model.load_model("../models/knn.model")
    model = Model.train_model('../data/train_data/', KNeighborsClassifier(n_neighbors=3))

    success, frame = cap.read()
    while success:

        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)

        # Por cada cara detectada
        for (x, y, w, h) in faces:
            rostro = frame[y - 5:y + h + 5, x - 5:x + w + 5]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
            name = Model.predict_one(model, rostro)
            frame = Recognition.draw_square(frame=frame, x=x, y=y, w=w, h=h, name=name[0])

        cv2.imshow('frame', frame)

        success, frame = cap.read()

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()


def show_example_landmarks(video_path, model):
    etiquetas_distancias = ['0_TO_00','1_TO_00','2_TO_00','3_TO_00','4_TO_00','5_TO_00','6_TO_00','7_TO_00','8_TO_00','9_TO_00','10_TO_00','11_TO_00','12_TO_00','13_TO_00','14_TO_00','15_TO_00','16_TO_00','17_TO_00','18_TO_00','19_TO_00','20_TO_00','21_TO_00','22_TO_00','23_TO_00','24_TO_00','25_TO_00','26_TO_00','27_TO_00','28_TO_00','29_TO_00','30_TO_00','31_TO_00','32_TO_00','33_TO_00','34_TO_00','35_TO_00','36_TO_00','37_TO_00','38_TO_00','39_TO_00','40_TO_00','41_TO_00','42_TO_00','43_TO_00','44_TO_00','45_TO_00','46_TO_00','47_TO_00','48_TO_00','49_TO_00','50_TO_00','51_TO_00','52_TO_00','53_TO_00','54_TO_00','55_TO_00','56_TO_00','57_TO_00','58_TO_00','59_TO_00','60_TO_00','61_TO_00','62_TO_00','63_TO_00','64_TO_00','65_TO_00','66_TO_00','67_TO_00']
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
    # Load video
    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture('data/videos/alex_y_endika.MOV')

    p = "models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    success, frame = cap.read()
    while success:

        datos_caras = Data.generate_data_from_frame(frame, detector, predictor)
        
        landmarks_df = pd.DataFrame(columns=cabeceras)
        
        for num,  posiciones , distancias in datos_caras:
            landmarks_df.loc[len(landmarks_df)] =  posiciones + distancias


        nombre = model.predict(landmarks_df[etiquetas_distancias])
        print(nombre)

        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        

        for (i, rect) in enumerate(rects):

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            x0, y0 = shape[0]
            x16, y16 = shape[16]
            x8, y8 = shape[8]
            x23, y23 = shape[23]


            frame = Recognition.draw_square(frame=frame, x=int(x0)- 10, y=int(y23) -10, w=int(0.05*640 + (x16-x0)), h=int(0.05*640 + (y8-y23)), name=nombre[i])

        cv2.imshow('frame', frame)

        success, frame = cap.read()

        k = cv2.waitKey(1)
        if k == 27:
            break

    cap.release()

def show_example_landmarks_v2():
    etiquetas_distancias = ['0_TO_00','1_TO_00','2_TO_00','3_TO_00','4_TO_00','5_TO_00','6_TO_00','7_TO_00','8_TO_00','9_TO_00','10_TO_00','11_TO_00','12_TO_00','13_TO_00','14_TO_00','15_TO_00','16_TO_00','17_TO_00','18_TO_00','19_TO_00','20_TO_00','21_TO_00','22_TO_00','23_TO_00','24_TO_00','25_TO_00','26_TO_00','27_TO_00','28_TO_00','29_TO_00','30_TO_00','31_TO_00','32_TO_00','33_TO_00','34_TO_00','35_TO_00','36_TO_00','37_TO_00','38_TO_00','39_TO_00','40_TO_00','41_TO_00','42_TO_00','43_TO_00','44_TO_00','45_TO_00','46_TO_00','47_TO_00','48_TO_00','49_TO_00','50_TO_00','51_TO_00','52_TO_00','53_TO_00','54_TO_00','55_TO_00','56_TO_00','57_TO_00','58_TO_00','59_TO_00','60_TO_00','61_TO_00','62_TO_00','63_TO_00','64_TO_00','65_TO_00','66_TO_00','67_TO_00']

    # Load video
    video_path = 'data/videos/alex_diego_paula_test.MOV'
    person_name ='endika_test'
    landsmarks_path = 'data/train_data/csv'

    landmarks_df = pd.read_csv('data/train_data/csv/alex_diego_paula_test.csv',sep=',',on_bad_lines='skip', encoding='latin-1', index_col=0)
    # landmarks_df = Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False)
    landmarks_df = landmarks_df[etiquetas_distancias]
    

    cap = cv2.VideoCapture(video_path)
    # cap = cv2.VideoCapture('data/videos/alex_y_endika.MOV')

    model = Model.load_model('models/alex_diego_paula.model')
    # model = Model.load_model('models/knn_classifier_distancias.model')
    nombres = model.predict(landmarks_df)
    
    frame_num = 0

    success, frame = cap.read()
    while success:

        # datos_caras = Data.generate_data_from_frame(frame)
        # print(datos_caras[etiquetas_distancias])

        frame = imutils.resize(frame, width=640)
        faces = Recognition.get_faces(frame)
        
        for (i, (x, y, w, h)) in enumerate(faces):
            rostro = frame[y - 5:y + h + 5, x - 5:x + w + 5]
            rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

            frame = Recognition.draw_square(frame=frame, x=x, y=y, w=w, h=h, name=nombres[frame_num])

        cv2.imshow('frame', frame)

        success, frame = cap.read()

        k = cv2.waitKey(1)
        if k == 27:
            break
        frame_num += 1

    cap.release()

def generate_csv():
    paths = []
    
    inicio = time.time()
    # video_path = 'data/videos/endika.mp4'
    # person_name ='endika'
    # landsmarks_path = 'data/train_data/csv'
    # paths.append(Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False))

    # video_path = 'data/videos/alex.MOV'
    # person_name ='alex'
    # landsmarks_path = 'data/train_data/csv'
    # paths.append(Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False))

    # video_path = 'data/videos/gorka.mp4'
    # person_name ='gorka'
    # landsmarks_path = 'data/train_data/csv'
    # paths.append(Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False))

    # video_path = 'data/videos/junkus.mov'
    # person_name ='junkus'
    # landsmarks_path = 'data/train_data/csv'
    # paths.append(Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False))

    # video_path = 'data/videos/luis.mp4'
    # person_name ='luis'
    # landsmarks_path = 'data/train_data/csv'
    # paths.append(Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False))

    # video_path = 'data/videos/victor.mp4'
    # person_name ='victor'
    # landsmarks_path = 'data/train_data/csv'
    # paths.append(Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False))

    # video_path = 'data/videos/villasante.MOV'
    # person_name ='villasante'
    # landsmarks_path = 'data/train_data/csv'
    # paths.append(Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False))

    # print(f'En total ha tardado {time.time() - inicio}')
    paths.append('data/train_data/csv/alex.csv')
    paths.append('data/train_data/csv/endika.csv')
    paths.append('data/train_data/csv/gorka.csv')
    paths.append('data/train_data/csv/junkus.csv')
    paths.append('data/train_data/csv/luis.csv')
    paths.append('data/train_data/csv/victor.csv')
    paths.append('data/train_data/csv/villasante.csv')


    # paths.append('data/train_data/csv/alex_diego_paula_train.csv')
    
    return paths

def create_model_landmarks():

    paths = generate_csv()

    test_video_path = 'data/videos/endika_test.mp4'

    person_name ='gente'
    landsmarks_path = 'data/test_data/csv'

    model = LogisticRegression(multi_class='multinomial')

    model = Model_landmarks.train_model(paths, model)

    show_example_landmarks(test_video_path, model)

    acc = Model_landmarks.test_model(model, Data.generate_data_as_landmarks(test_video_path, person_name, landsmarks_path, False))
    
    print(f'Accuracy del modelo es {acc}')
    if acc > 0.8:
        Model_landmarks.save_model(model, f'alex_test')


# generate_csv()
create_model_landmarks()