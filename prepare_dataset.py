import os
import cv2

from tabulate import tabulate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical

imagePaths = []
image_size = 224  #Resize images

for dirname, _, filenames in os.walk(r'Dataset'):
    for filename in filenames:
        if (filename[-3:] == 'png'):
            imagePaths.append(os.path.join(dirname, filename))

# pre_process images.
def datagen(image_path):
    X = []
    y = []
    for img_path in image_path:
        label = img_path.split(os.path.sep)[-2]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (image_size,image_size))
    

        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    return (X, y)

# generate the training and validation dataset
def train_val_data(img, label):
    X_train, X_test, y_train, y_test = train_test_split(img, label, test_size=0.30, random_state=3)

    #Encode labels as integers
    le = LabelEncoder()
    le.fit(y_test)
    y_test_labels_encoded = le.transform(y_test)
    le.fit(y_train)
    y_train_labels_encoded = le.transform(y_train)

    # Normalize pixel values to between 0 and 1
    X_train, X_test = X_train / 255.0, X_test / 255.0

    y_train_one_hot = np_utils.to_categorical(y_train_labels_encoded)
    y_test_one_hot = np_utils.to_categorical(y_test_labels_encoded)

    return(X_train, X_test, y_train, y_test, y_test_labels_encoded, y_train_labels_encoded)

# visualize dataset  
def view_labels(labels):
    y_df = pd.DataFrame(labels, columns=['Labels'])
    
    with open("data_labels.txt", "w") as file1:
        # Writing data to a file
        file1.writelines(str(y_df['Labels'].value_counts()))
        file1.writelines("\n")
        file1.writelines(str(tabulate(y_df, headers = 'keys', tablefmt = 'psql')))
        
    sns.countplot(y_df['Labels'])
    plt.savefig("Data_labels.png")


X, y = datagen(image_path= imagePaths)

X_train, X_test, y_train, y_test, y_test_labels_encoded, y_train_labels_encoded = train_val_data(img = X, label = y)

view_labels(labels = y)