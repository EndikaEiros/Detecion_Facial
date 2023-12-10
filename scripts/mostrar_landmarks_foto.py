from imutils import face_utils
import dlib
import cv2
 
p = "models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

image = cv2.imread('data/keanu.png')
 
# while True:
    # _, image = cap.read()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
rects = detector(gray, 0)

for (i, rect) in enumerate(rects):

    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

cv2.imshow("Salida", image)
cv2.imwrite('data/keanu_landmarks.png',image)
