from PIL import Image, ImageDraw
import cv2
from Recognition import get_faces
import imutils
import face_recognition
from imutils import face_utils
import dlib


def test_landmarks():

    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file('../data/keanu.png')

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(image)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], width=5)

    # Show the picture
    cv2.imshow('landmarks', pil_image)

def test_rotation():

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

def get_drawed_landmarks_from_image()
         
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
