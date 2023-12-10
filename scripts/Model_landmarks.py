import os
import pickle
import cv2
import numpy as np
import pandas as pd
import Data

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
etiquetas_distancias = ['0_TO_00','1_TO_00','2_TO_00','3_TO_00','4_TO_00','5_TO_00','6_TO_00','7_TO_00','8_TO_00','9_TO_00','10_TO_00','11_TO_00','12_TO_00','13_TO_00','14_TO_00','15_TO_00','16_TO_00','17_TO_00','18_TO_00','19_TO_00','20_TO_00','21_TO_00','22_TO_00','23_TO_00','24_TO_00','25_TO_00','26_TO_00','27_TO_00','28_TO_00','29_TO_00','30_TO_00','31_TO_00','32_TO_00','33_TO_00','34_TO_00','35_TO_00','36_TO_00','37_TO_00','38_TO_00','39_TO_00','40_TO_00','41_TO_00','42_TO_00','43_TO_00','44_TO_00','45_TO_00','46_TO_00','47_TO_00','48_TO_00','49_TO_00','50_TO_00','51_TO_00','52_TO_00','53_TO_00','54_TO_00','55_TO_00','56_TO_00','57_TO_00','58_TO_00','59_TO_00','60_TO_00','61_TO_00','62_TO_00','63_TO_00','64_TO_00','65_TO_00','66_TO_00','67_TO_00']


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
def load_csv_data(paths):

    X = pd.concat([pd.read_csv(csv,sep=',',on_bad_lines='skip', encoding='latin-1', index_col=0) for csv in paths],axis=0)

    y = X['Etiqueta']

    return X.drop(columns=['Etiqueta']), y 


""" Model Evaluation (accuracy) """
def test_model(trained_model, test_csv_path):

    X_test, y_test = load_csv_data(test_csv_path)
    pred = trained_model.predict(X_test)

    accuracy = trained_model.score(X_test[etiquetas_distancias], y_test)

    print(confusion_matrix(y_test, pred))
    print(f"\nAccuracy: {accuracy}")

    return accuracy


""" Predict an image label """
def predict_one(trained_model, landmarks_df):

    return trained_model.predict(landmarks_df[etiquetas_distancias])


""" Train new model """
def train_model(train_csvs_path, model):

    X_train, y_train = load_csv_data(train_csvs_path)
    model.fit(X_train[etiquetas_distancias], y_train)

    return model


    

    