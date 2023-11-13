import os
import pickle

import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

#################### Parámetros ####################

# Path a las imágenes
train_images_path = '../data/train_data/masked/'

# Nombre del modelo a entrenar: (knn, decisiontree)
modelo = 'knn'

# Nombre de como se va a guardar el modelo
nombre_modelo = 'knn_masked'

####################################################


# Almacenar el modelo entrenado
def save_model(model, name):
    with open(f"../models/{name}.model", "wb") as f:
        pickle.dump(model, f)


# Cargar imágenes de entrenamiento
def get_train_images(directory_name):
    # Obtener todos los archivos del directorio
    files = os.listdir(directory_name)
    all_images = list()

    # Por cada fichero/directorio
    for file in files:

        # Obtener su path completo
        file_path = os.path.join(directory_name, file)

        # Si es un directorio se cargan los fichero en su interior
        if os.path.isdir(file_path):
            all_images = all_images + get_train_images(file_path)

        # Si es un fichero, se carga
        else:
            all_images.append(file_path)

    return all_images


# Cargar imágenes para el entrenamiento
imagePaths = get_train_images(train_images_path)

# Inicializaciones
data = []
lables = []

# Por cada imagen
for image in imagePaths:
    # Obtener label
    lable = os.path.split(os.path.split(image)[0])[1]
    lables.append(lable)

    # Obtener imagen
    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    data.append(img)

data = np.array(data)
lables = np.array(lables)

le = LabelEncoder()
lables = le.fit_transform(lables)

myset = set(lables)
print(myset)

dataset_size = data.shape[0]
data = data.reshape(dataset_size, -1)

# Split
# trainX, _, trainY, _ = train_test_split(data, lables, test_size=0, random_state=42)

if modelo == 'decisiontree':
    model = DecisionTreeClassifier(max_depth=200000)

else:
    model = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)

#
# save_model(model, "decisiontree")

# Entrenar modelo
model.fit(data, lables)

# Guardar modelo entrendo
save_model(model, nombre_modelo)

# Probar
image = cv2.imread("../data/train_data/unmasked/Alex/rostro_130.jpg")
image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).reshape(1, -1)

print(model.predict(image))
