import cv2
import os
import imutils
import numpy as np

####################### Parámetros #######################

# Path al vídeo
video_path = '../data/videos/videoChorra3.mp4'

# Nombre de la persona del video
person_name = 'Alex'

background_remove = True

# Path donde se guardarán las imágenes
unmasked_path = '../data/train_data/unmasked/' + person_name

# Path donde se guardarán las imágenes con el fondo borrado
masked_path = '../data/train_data/masked/' + person_name

# Path donde se guardarán las imágenes con los "landsmarks" faciales
landsmarks_path = '../data/train_data/landsmarks/' + person_name

##########################################################

# Metodo auxliar para borrar el fondo de la imagen
def bgremove(frame, min_thres=90, min_satur=60, min_brigth=50):
    
    # Valores limite del color de la piel
    min_piel = np.array([min_thres, min_satur, min_brigth])
    max_piel = np.array([255, 255, 255])

    # Se aplica un blurr Gausioano para eliminar ruido y se convierte a HSV
    frameHSV = cv2.GaussianBlur(frame, (7, 7), 0)
    frameHSV = cv2.cvtColor(frameHSV, cv2.COLOR_RGB2HSV)

    # mascara que obtine el color del tono de piel del frame
    skinRegion = cv2.inRange(frameHSV, min_piel, max_piel)
    frame_skin = cv2.bitwise_and(frame, frame, mask=skinRegion)

    return frame_skin

##########################################################

# Si el directorio no existe, se crea
if not os.path.exists(unmasked_path):
    print('Carpeta creada: ', unmasked_path)
    os.makedirs(unmasked_path)

if not os.path.exists(masked_path):
    print('Carpeta creada: ', masked_path)
    os.makedirs(masked_path)

# Cargar video
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture(video_path)

# Haarcascade face classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

# Inicializaciones
image_num = 0

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
        
        if background_remove:
            frame_masked = bgremove(frame)[y:y + h, x:x + w]
            rostro_mask = cv2.resize(frame_masked, (150, 150), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(f'{masked_path}/rostro_{str(image_num)}.jpg', rostro_mask)

        # Almacenar rostros, en la carpeta correspondiente
        cv2.imwrite(f'{unmasked_path}/rostro_{str(image_num)}.jpg', rostro)
        

        # Visualiar el video
        # cv2.imshow('frame', frame)
        # cv2.imshow('masked_frame', frame_masked)

    # Obtener siguiente frame
    success, frame = cap.read()

    k = cv2.waitKey(1)
    if k == 27:
        break

print(f'{image_num} imágenes de {person_name} almacenadas en: {unmasked_path}') 
cap.release()
cv2.destroyAllWindows()