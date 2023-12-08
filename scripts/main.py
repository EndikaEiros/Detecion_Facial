import cv2
import imutils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import Model
import Data
import Recognition

from Recognition import draw_square
def v1():
    # Load video
    cap = cv2.VideoCapture('../data/videos/TEST3.MOV')

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

def v2():
    # Load video
    cap = cv2.VideoCapture('../data/videos/TEST3.MOV')

    # # Load classifier model
    model = Model.load_model("../models/regression_3people.model")
    # model = Model.train_model('../data/train_data/', KNeighborsClassifier(n_neighbors=3))

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
    cv2.destroyAllWindows()


def generate_csv():

    video_path = '../data/videos/ALEX.mp4'
    person_name ='Alex'
    landsmarks_path = 'data/train_csv_data'
    Data.generate_data_as_landmarks(video_path, person_name, landsmarks_path, False)

generate_csv()



# Data.generate_data_as_images('../data/videos/TEST3.MOV', 'Alex', '../data/test_data/3people')
#
# model = Model.train_model('../data/train_data/', KNeighborsClassifier(n_neighbors=3))
# Model.save_model(model, 'knn_3people')

# v2()

# model = Model.load_model("../models/knn_3people.model")
# Model.test_model(model, '../data/test_data/3people')

# model = Model.train_model('../data/train_data/', LogisticRegression(solver='lbfgs', max_iter=1000))
# Model.save_model(model, 'regression_2people')

# model = Model.load_model("../models/regression_3people.model")
# accuracy = Model.test_model(model, '../data/test_data/3people')
# print(accuracy)

# model = MLPClassifier(hidden_layer_sizes=(300, 100), max_iter=1000, solver='lbfgs')
# model = Model.train_model('../data/train_data', model)
# accuracy = Model.test_model(model, '../data/test_data/3people')
# print(f"\tAccuracy: {accuracy}")
