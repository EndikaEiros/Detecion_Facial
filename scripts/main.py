import cv2
import imutils
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import Model
import Data
import Recognition

from Recognition import draw_square

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
