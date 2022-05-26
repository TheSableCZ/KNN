import math
import tkinter
from tkinter import messagebox
from tkinter import ttk
from tkinter import *

import numpy as np
from PIL import Image, ImageTk
import sys
import os
import cv2
import glob
import tensorflow as tf

from YogiAI.utils.model import predict_with_static_image_for_gui


class Gui:

    def __init__(self):
        self.textOfSaved = None
        self.textOfClass = None
        self.arrayOfPictures = []
        self.range = None
        self.disp_img = None
        self.skipBy = None
        self.Lb1 = None
        self.currnentClass = None
        self.currnentIndexOfClass = None
        self.ws = Tk()
        self.ws.bind("<Key>", self.onKeyPress)
        self.ws.title('PythonGuides')
        self.ws.geometry('1024x720')
        self.ws.config(bg='#4a7a8c')
        self.iterator = 0
        self.skipValue = 0
        self.findPic = False
        self.listOfImages = []
        self.video_capture = cv2.VideoCapture(sys.argv[1])
        self.maxIteratorValue = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self.class_labels = {'Chair Pose': 0,
                             'Child Pose': 1,
                             'Cobra Pose': 2,
                             'Downward-Facing Dog pose': 3,
                             'Four-Limbed Staff Pose': 4,
                             'Happy Baby Pose': 5,
                             'Intense Side Stretch Pose': 6,
                             'Low Lunge pose': 7,
                             'Plank Pose': 8,
                             'Standing Forward Bend': 9,
                             'Warrior I Pose': 10}

        self.classLabelsRev = {0: 'Chair Pose',
                               1: 'Child Pose',
                               2: 'Cobra Pose',
                               3: 'Downward-Facing Dog pose',
                               4: 'Four-Limbed Staff Pose',
                               5: 'Happy Baby Pose',
                               6: 'Intense Side Stretch Pose',
                               7: 'Low Lunge pose',
                               8: 'Plank Pose',
                               9: 'Standing Forward Bend',
                               10: 'Warrior I Pose'}
        isExist = os.path.exists("workingDirectory")
        if not isExist:
            os.mkdir("workingDirectory")
        for i in range(0, 11):
            isExist = os.path.exists("workingDirectory/class" + str(i))
            if not isExist:
                os.mkdir("workingDirectory/class" + str(i))

        self.model = tf.keras.models.load_model("model")

        self.initGuiElements()
        self.ws.mainloop()

    def loopThroughVideo(self):
        while self.iterator < self.maxIteratorValue:
            self.loadListWithFrameAndClass()
            if self.findPic:
                self.findPic = False
                self.listOfImages.append((self.iterator, self.currnentIndexOfClass))
            self.iterator += 20
        self.iterator = 0
        self.maxIteratorValue = len(self.listOfImages)

    def loadListWithFrameAndClass(self):
        self.video_capture.set(1, self.iterator)
        success, frame = self.video_capture.read()
        if frame is None:
            return
        result = predict_with_static_image_for_gui(self.model, self.class_labels, frame)
        resultMax = result.max()
        if resultMax > 0.95:
            self.findPic = True
            self.currnentClass = list(self.class_labels.keys())[np.argmax(result)]
            self.currnentIndexOfClass = self.class_labels[self.currnentClass]
        """
        if success:
            return ImageTk.PhotoImage(image=Image.fromarray(frame[..., ::-1].copy())), frame[..., ::-1].copy()
        return None
        """

    def get_frame_by_index(self):
        self.video_capture.set(1, self.listOfImages[self.iterator][0])
        success, frame = self.video_capture.read()
        self.changeLabelName(self.listOfImages[self.iterator][1])
        self.checkIfSaved()
        self.currnentIndexOfClass = self.listOfImages[self.iterator][1]
        if success:
            return ImageTk.PhotoImage(image=Image.fromarray(frame[..., ::-1].copy())), frame[..., ::-1].copy()
        return None

    def loadClassesToList(self):
        for key, value in self.class_labels.items():
            self.Lb1.insert(value, key)

    def insertInfoOfRange(self):
        self.range.delete('1.0', END)
        self.range.insert(chars=str(self.iterator) + ' / ' + str(self.maxIteratorValue), index=END)

    def deleteImage(self):
        nameOfFile = sys.argv[1] + str(self.listOfImages[self.iterator][0]).zfill(
            math.ceil(math.log10(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))))
        list = glob.glob('workingDirectory/*/' + nameOfFile + '.png')
        for x in list:
            if os.path.exists(x):
                os.remove(x)
        self.checkIfSaved()


    def initGuiElements(self):
        frame = Frame(self.ws)
        frame.pack(side=BOTTOM)

        confirm = ttk.Button(frame, text='Confirm', command=lambda: self.confirm())
        confirm.pack(side=LEFT)

        self.Lb1 = Listbox(frame, selectmode=SINGLE, exportselection=False)
        self.loadClassesToList()
        self.Lb1.pack(side=LEFT)

        self.textOfClass = Label(frame, text="Random class", fg="RED")
        self.textOfClass.pack(side=LEFT)

        self.textOfSaved = Label(frame, text="Random class", fg="RED")
        self.textOfSaved.pack(side=LEFT)

        prevImg = ttk.Button(frame, text='Prev', command=lambda: self.prevImgFunc())
        prevImg.pack(side=LEFT)

        nextImg = ttk.Button(frame, text='Next', command=lambda: self.nextImgFunc())
        nextImg.pack(side=LEFT)

        deleteImage = ttk.Button(frame, text='Delete', command=lambda: self.deleteImage())
        deleteImage.pack(side=LEFT)

        self.range = Text(frame, width=15, height=1)

        self.range.pack(side=LEFT)

        self.disp_img = Label()
        self.disp_img.pack(pady=40)

        self.loopThroughVideo()

        self.iterator = 0
        self.maxIteratorValue = len(self.listOfImages)

        self.range.insert(chars=str(self.iterator) + ' / ' + str(self.maxIteratorValue), index=END)

        img, frame = self.get_frame_by_index()
        self.disp_img.config(image=img)
        self.disp_img.image = img

    def changeLabelName(self, classNumber):
        key = [k for k, v in self.class_labels.items() if v == classNumber]
        self.textOfClass.configure(text=str(key[0]))

    def checkIfSaved(self):
        nameOfFile = sys.argv[1] + str(self.listOfImages[self.iterator][0]).zfill(
            math.ceil(math.log10(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))))
        list = glob.glob('workingDirectory/*/' + nameOfFile + '.png')
        if len(list) > 0:
            self.textOfSaved.configure(text="Saved", fg="GREEN")
        else:
            self.textOfSaved.configure(text="Unsaved", fg="RED")

    def onKeyPress(self, event):
        if event.char == 's':
            self.confirm()
        elif event.char == 'd':
            self.nextImgFunc()
        elif event.char == 'a':
            self.prevImgFunc()

    def showImage(self):
        self.Lb1.selection_clear(0, tkinter.END)
        img, _ = self.get_frame_by_index()
        self.disp_img.config(image=img)
        self.disp_img.image = img

    def prevImgFunc(self):
        if -1 >= self.iterator-1 < self.maxIteratorValue:
            messagebox.showinfo("Information", "Out of bound")
            return None
        self.iterator -= 1
        self.insertInfoOfRange()
        self.showImage()

    def nextImgFunc(self):
        if self.iterator+1 > self.maxIteratorValue:
            messagebox.showinfo("Information", "Out of bound")
            return None
        self.iterator += 1
        self.insertInfoOfRange()
        self.showImage()

    def confirm(self):
        selected = self.Lb1.curselection()
        if len(selected) == 1:
            self.currnentIndexOfClass = selected[0]
        directoryNum = self.currnentIndexOfClass
        _, frame = self.get_frame_by_index()
        nameOfFile = sys.argv[1] + str(self.listOfImages[self.iterator][0]).zfill(math.ceil(math.log10(int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))))
        self.deleteImage()
        image = Image.fromarray(frame)
        image.save('workingDirectory/class' + str(directoryNum) + '/' + nameOfFile + '.png', 'PNG')
        self.nextImgFunc()


if __name__ == "__main__":
    Gui()
