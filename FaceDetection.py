# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
from Tkinter import *
import tkFileDialog


class FaceDetection:  # create window
    inputFile = 0
    scaleFactor = 1.03
    minNeighbors = 5
    entry1 = 0
    entry2 = 0

    def __init__(self):
        global scaleFactor
        global minNeighbors
        global entry1
        global entry2
        window = Tk()
        window.title("Face Detection")
        window.withdraw()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight() - 100
        window.resizable(False, False)
        window.update_idletasks()
        window.deiconify()
        window.withdraw()
        window.geometry('%sx%s+%s+%s' % (window.winfo_width() + 500, window.winfo_height() + 10,
                                         (screen_width - window.winfo_width()) / 2, (screen_height - window.winfo_height()) / 2))
        window.deiconify()

        frame1 = Frame(window)
        frame1.pack()

        label1 = Label(frame1, text="Choose a Photo", font=('Arial', 13))
        label1.grid(row=0)

        button = Button(frame1, text="Open File", command=self.openFile)
        button.grid(row=0, column=1)

        frame2 = Frame(window)
        frame2.pack()

        label2 = Label(frame2, text="scaleFactor", font=('Arial', 13))
        label2.grid(row=0)

        entry1 = Entry(frame2, bd=5)
        entry1.grid(row=0, column=1)

        label3 = Label(frame2, text="minNeighbors", font=('Arial', 13))
        label3.grid(row=1)

        entry2 = Entry(frame2, bd=5)
        entry2.grid(row=1, column=1)

        button = Button(frame2, text="Start", command=self.processImage)
        button.grid(row=2, column=1)

        window.mainloop()

    def openFile(self):  # input file
        global inputFile

        inputFile = tkFileDialog.askopenfilename()

    def processImage(self):  # process image
        global inputFile
        global scaleFactor
        global minNeighbors
        global entry1
        global entry2

        face_cascade = cv2.CascadeClassifier(
            'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

        img = cv2.imread(inputFile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        scaleFactor = eval(entry1.get())
        minNeighbors = eval(entry2.get())

        faces = face_cascade.detectMultiScale(gray, scaleFactor, minNeighbors)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(
                    roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


FaceDetection()
