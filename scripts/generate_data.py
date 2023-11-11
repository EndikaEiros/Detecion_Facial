import cv2
import os
import imutils

####################### Parámetros #######################

# Path al vídeo
video_path = '../data/videos/videoChorra3.mp4'

# Nombre de la persona del video
person_name = 'Alex'

# Path donde se guardarán las imágenes
person_path = '../data/train_data/unmasked/' + person_name

##########################################################

# Si el directorio no existe, se crea
if not os.path.exists(person_path):
    print('Carpeta creada: ', person_path)
    os.makedirs(person_path)

# Cargar video
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture(video_path)

# Haarcascade face classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Inicializaciones
image_num = 350

# Obtener primer frame
success, frame = cap.read()

# Por cada frame del video
while success:

    # Preprocesar frame y obtener caras detectadas
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Por cada cara detectada
    for (x, y, w, h) in faces:

        image_num += 1

        # Recortar rostro
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        rostro = auxFrame[y:y + h, x:x + w]
        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)

        # Almacenar rostro
        cv2.imwrite(f'{person_path}/rostro_{str(image_num)}.jpg', rostro)

        cv2.imshow('frame', frame)

    # Obtener siguiente frame
    success, frame = cap.read()

    k = cv2.waitKey(1)
    if k == 27:
        break

print(f'{image_num} imágenes de {person_name} almacenadas en: {person_path}')
cap.release()
cv2.destroyAllWindows()