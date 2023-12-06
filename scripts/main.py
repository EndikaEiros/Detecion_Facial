import cv2
import imutils

import Model
import Data
import Recognition

# Load video
cap = cv2.VideoCapture('../data/videos/ALEX.MOV')

# faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

success, frame = cap.read()
c = 0

while success:

    frame = imutils.resize(frame, width=640)

    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    model = Model.load_model("../models/knn.model")

    for (x, y, w, h) in faces:
        # frame_masked = bgremove(auxFrame, 70, 30, 30)[y-10:y + h+10, x-10:x + w+10]
        frame_masked = auxFrame[y:y + h, x:x + w]

        name = Model.predict_one(model, rostro)

        frame = draw_square(frame=frame, x=x, y=y, w=w, h=h, name=name)

    cv2.imshow('frame', frame)

    success, frame = cap.read()

    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
