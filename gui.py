import tkinter as tk
from tkinter import filedialog
from tkinter import *
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

def DrowsinessDetectionModel(json_file, weights_file):
    with open(json_file, "r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

top = tk.Tk()
top.geometry('800x600')
top.title("DROWSINESS DETECTOR")
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_tree_eyeglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

model = DrowsinessDetectionModel("model_a2.json", "bestModels1.h5")

EYE_STATE = ["Open Eyes", "Closed Eyes"]

def Detect(file_path):
    global label1

    try:
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                roi_gray = gray_image[y:y + h, x:x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)

                for (ex, ey, ew, eh) in eyes:
                    roi_eye = roi_gray[ey:ey + eh, ex:ex + ew]
                    roi_resized = cv2.resize(roi_eye, (64, 64))
                    roi_resized = np.expand_dims(roi_resized, axis=-1)  # Add channel dimension
                    roi_resized = np.expand_dims(roi_resized, axis=0)  # Add batch dimension
                    pred = EYE_STATE[np.argmax(model.predict(roi_resized))]

                print("Predicted Drowsiness is " + pred)
                label1.configure(foreground="#011638", text=pred)
        else:
            label1.configure(foreground="#011638", text="No face detected")
    except Exception as e:
        print(e)
        label1.configure(foreground="#011638", text="Error processing image")

def show_Detect_button(file_path):
    detect_b = Button(top, text="Detect Drowsiness", command=lambda: Detect(file_path), padx=10, pady=5)
    detect_b.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except Exception as e:
        print(e)
        label1.configure(foreground="#011638", text="Error uploading image")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top, text='Drowsiness Detection', pady=20, font=('arial', 25, 'bold'))
heading.configure(background='#CDCDCD', foreground="#364156")
heading.pack()

top.mainloop()