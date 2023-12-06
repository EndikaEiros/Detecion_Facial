import os
import pickle
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

""" Save trained model """
def save_model(model, name):
    with open(f"../models/{name}.model", "wb") as f:
        pickle.dump(model, f)


""" Load trained model """
def load_model(model_path):
    with open(model_path, "rb") as knn:
        loaded_model = pickle.load(knn)

    return loaded_model


""" Load data (X = images, y = labels) """
def load_image_data(path):
    X = list()
    y = list()

    for directory in os.listdir(path):

        images_path = os.path.join(path, directory)

        for img in os.listdir(images_path):
            y.append(directory)
            imagen = cv2.imread(os.path.join(images_path, img))
            imagen = imagen.reshape(1, -1)
            X.append(imagen)

    return np.array(X).reshape(np.array(X).shape[0], -1), np.array(y)


""" Model Evaluation (accuracy) """
def test_model(trained_model, test_images_path):

    X_test, y_test = load_image_data(test_images_path)
    pred = trained_model.predict(X_test)

    accuracy = trained_model.score(X_test, y_test)

    # print(confusion_matrix(y_test, pred))
    # print(f"\nAccuracy: {accuracy}")

    return accuracy


""" Predict an image label """
def predict_one(trained_model, image):

    return trained_model.predict(image.reshape(1, -1))


""" Train new model """
def train_model(train_images_path, model):

    X_train, y_train = load_image_data(train_images_path)
    model.fit(X_train, y_train)

    return model


