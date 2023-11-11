import cv2
from detect_rotated_faces import get_faces
import imutils

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture('../data/videos/full_rotation.mp4')

success, frame = cap.read()

while success and cv2.waitKey(1) == -1:

    frame = imutils.resize(frame, width=640)

    original_frame = frame.copy()

    faces1 = get_faces(frame, [0, -65, -32, 32, 65])
    faces2 = get_faces(frame, [])

    for (x, y, w, h) in faces1:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y, w, h) in faces2:
        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("TRUE", frame)
    cv2.imshow("FALSE", original_frame)

    success, frame = cap.read()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
