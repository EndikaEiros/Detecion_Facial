
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import  train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import os
import glob
import cv2
import pickle 
import numpy as np


def save_model(knn):

    with open("models/decisiontree.model", "wb") as model:
        pickle.dump(knn, model)



# Cargar imágenes
def getListOfFiles(dirName):
    # Cara imagen se anota con el nombre de su carpeta
    
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:

        fullPath = os.path.join(dirName, entry)

        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

imagePaths = getListOfFiles("./caras/")

data = []
lables = []

for image in imagePaths:

    lable = os.path.split(os.path.split(image)[0])[1]
    lables.append(lable)

    img = cv2.imread(image)
    img = cv2.resize(img, (32, 32), interpolation = cv2.INTER_AREA)
    data.append(img)


data = np.array(data)
lables = np.array(lables)

le = LabelEncoder()
lables = le.fit_transform(lables)

myset = set(lables)
print(myset)

dataset_size = data.shape[0]
data = data.reshape(dataset_size,-1)

# print(f"Data shape\t--> {data.shape}")
# print(f"Labels shape\t--> {lables.shape}")
# print(f"Dataset size\t--> {dataset_size}")

# Split
(trainX, testX, trainY, testY) = train_test_split(data, lables, test_size= 0.25, random_state=42)

model = DecisionTreeClassifier(max_depth=200000)
model.fit(trainX, trainY)
save_model(model)

image = cv2.imread("./guess/endika.jpg")
image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).reshape(1, -1)

print(model.predict(image))
# print(classification_report(testY, model.predict(testX), target_names=le.classes_))