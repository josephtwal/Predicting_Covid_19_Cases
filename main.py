import tkinter as tk
from tkinter import Label, filedialog, Text
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from model import vgg_model, features_exraction
from prepare_dataset import X_train, X_test, y_train, y_test, y_test_labels_encoded, y_train_labels_encoded

import pickle

from sklearn.preprocessing import LabelEncoder





win = tk.Toplevel()
win.title("Covid-19 Test Application")
img_path = []

def openimage():

    for widget in frame.winfo_children():
        widget.destroy()
    
    img_path.clear()

    filename = filedialog.askopenfilename(initialdir=r"C:\Users\josep\Desktop\Covid_Application", title="Select Image", 
                                            filetypes=(("PNG", "*.png"), ("JPEG", "*.jpeg")))

    img_path.append(filename)
    for img in img_path:
        label2 = tk.Label(frame, text=img, bg="white", fg="#263D42" ).pack()
   
def run():

    #Encode labels as integers
    le = LabelEncoder()
    le.fit(y_test)
    y_test_labels_encoded = le.transform(y_test)
    le.fit(y_train)
    y_train_labels_encoded = le.transform(y_train)

    filesvm = "SVM_Model.pkl"
    with open(filesvm, 'rb') as file:
        final_model = pickle.load(file)
    

    img1= cv2.imread(str(img_path[0]), cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (224,224))/255.0
    
    input_img = np.expand_dims(img1, axis=0) #Expand dims so the input is (num images, x, y, c)
    img_test = features_exraction(vgg_model ,input_img)

    input_img_features=img_test.reshape(img_test.shape[0], -1)

    label3 = tk.Label(frame, text=f"******Checking for covid ******\n", bg="white", fg="#263D42" ).pack()
    pred = final_model.predict(input_img_features)
    pred = le.inverse_transform([pred])

             # loading the image to GUI
    img2 = Image.open(str(img_path[0]))
    img2 = img2.resize((500,500), Image.ANTIALIAS)
    img2 = ImageTk.PhotoImage(img2)
    label1 = tk.Label(frame, image=img2)
    label1.image=img2
    label1.pack()


    label4 = tk.Label(frame, text=f"Predicted output: {pred}", bg="white", fg="#263D42" ).pack()


canvas = tk.Canvas(win, height=900, width=900, bg="#263D42")
canvas.pack()

frame = tk.Frame(win, bg="white")
frame.place(relheight=0.8, relwidth=0.8, relx=0.1, rely=0.1)

label1 = tk.Label(frame, text="Covid-19 Test \n Please Select an Image", bg="white", fg="#263D42" ).pack()

openfile = tk.Button(win, text="Open Image", padx=10,
                         pady=5, fg="white", bg="#263D42", command=openimage).pack()


run_application = tk.Button(win, text="Run", padx=10,
                         pady=5, fg="white", bg="#263D42", command=run).pack()

                                           



win.mainloop()


