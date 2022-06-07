from cgi import test
from prepare_dataset import X_train, X_test, y_train, y_test, y_test_labels_encoded, y_train_labels_encoded

import os
import cv2
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16

SIZE = 224

def model_VGG16():

    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(SIZE, SIZE, 3))

    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in VGG_model.layers:
        layer.trainable = False

    return(VGG_model)

def features_exraction(model, data):

    feature_extractor=model.predict(data)
    features = feature_extractor.reshape(feature_extractor.shape[0], -1)

    return(features)

def SVM(data, label, test):

    svc_params = {'kernel': ['rbf'], 'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100]}
    SVM_model = GridSearchCV(estimator=SVC(), param_grid=svc_params, cv=3, n_jobs=-1, 
                        scoring='accuracy', verbose=10)
    
    SVM_model.fit(data, label)
    
    #Encode labels as integers
    le = LabelEncoder()
    le.fit(y_test)
    y_test_labels_encoded = le.transform(y_test)
    le.fit(y_train)
    y_train_labels_encoded = le.transform(y_train)

    prediction_svm = SVM_model.predict(test)
    prediction_svm = le.inverse_transform(prediction_svm)

    cm = confusion_matrix(y_test, prediction_svm)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    with open("output.txt", "w") as file1:
        # Writing data to a file
        file1.writelines("Accuracy")
        file1.writelines("\n")
        file1.writelines(str(cm_df.head()))
        file1.writelines("\n")
        file1.writelines(str(classification_report(y_test, prediction_svm)))

    plt.figure(figsize=(9,9))
    sns.set(font_scale=1.5, color_codes=True, palette='deep')
    sns.heatmap(cm_df, annot=True, annot_kws={'size':16}, fmt='d', cmap='Pastel1')
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title('Confusion Matrix of Validation Data',size=15)
    plt.savefig("output.png")

    
    filesvm = "SVM_Model.pkl"
    with open(filesvm, 'wb') as files:
        pickle.dump(SVM_model, files)


    


vgg_model = model_VGG16()
X_for_SVM = features_exraction(vgg_model, X_train)
X_test_feature = features_exraction(vgg_model, X_test)


SVM(data = X_for_SVM , label = y_train_labels_encoded, test = X_test_feature)