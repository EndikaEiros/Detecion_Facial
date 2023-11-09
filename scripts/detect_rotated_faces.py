import cv2
import imutils
from math import sin, cos, radians

def rotate_image(image, angle):
    if angle == 0: return image
    height, width = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((width/2, height/2), angle, 0.9)
    result = cv2.warpAffine(image, rot_mat, (width, height), flags=cv2.INTER_LINEAR)
    return result

def rotate_point(pos, img, angle):
    if angle == 0: return pos
    x = pos[0] - img.shape[1]*0.4
    y = pos[1] - img.shape[0]*0.4
    newx = x*cos(radians(angle)) + y*sin(radians(angle)) + img.shape[1]*0.4
    newy = -x*sin(radians(angle)) + y*cos(radians(angle)) + img.shape[0]*0.4
    return int(newx), int(newy), pos[2], pos[3]


cap = cv2.VideoCapture('/data/videos/videoChorra3.mp4')

# faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
# faceClassif_alt = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt.xml')
faceClassif_alt2 = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')


while True:
    
    ret, frame = cap.read()

    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    for angle in [0, -25, 25]:

        rimg = rotate_image(frame, angle=angle)

        # faces = faceClassif.detectMultiScale(gray,1.3,5)
        # faces = faceClassif_alt.detectMultiScale(gray,1.3,5)
        # faces = faceClassif_alt2.detectMultiScale(gray,1.3,5)
        detected_faces = faceClassif_alt2.detectMultiScale(rimg)

        if len(detected_faces):
            detected_faces = [rotate_point(detected_faces[-1], frame, angle=-angle)]
            break

    if ret == False: break
    frame =  imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    
    for (x,y,w,h) in detected_faces:

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        rostro = auxFrame[y:y+h,x:x+w]
        rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) != -1: break

cap.release()
cv2.destroyAllWindows()
