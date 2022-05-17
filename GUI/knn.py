import math
import tkinter
from tkinter import messagebox
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
import sys
import os
import cv2
import glob
import tensorflow as tf

from YogiAI.utils.model import predict_with_static_image


class Gui:

    def __init__(self):
        self.textOfClass = None
        self.arrayOfPictures = []
        self.range = None
        self.disp_img = None
        self.skipBy = None
        self.Lb1 = None
        self.ws = Tk()
        self.ws.bind("<Key>", self.onKeyPress)
        self.ws.title('PythonGuides')
        self.ws.geometry('1024x720')
        self.ws.config(bg='#4a7a8c')
        self.iterator = 1450
        self.skipValue = 0
        self.video_capture = cv2.VideoCapture('../pictures/video.mp4') # todo video z argv[1]
        # todo potrebuju zaplnit list arrayOfPictures dvojcemi (cisloSnimku, classa)
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

    def get_frame_by_index(self):

        self.video_capture.set(1, self.iterator)

        success, frame = self.video_capture.read()

        if success:
            return ImageTk.PhotoImage(image=Image.fromarray(frame)), frame
        return None

    def getSkipbyValue(self):
        return self.skipBy.get()

    # todo def createClasses(self):

    def loadClassesToList(self):
        for key, value in self.class_labels.items():
            self.Lb1.insert(value, key)

    def insertInfoOfRange(self):
        self.range.delete('1.0', END)
        self.range.insert(chars = str(self.iterator) + ' / ' + str(self.maxIteratorValue), index=END)

    def selectFromList(self):
        # todo
        return None

    def activeListBox(self):
        self.Lb1.itemconfig(1, {'bg': 'Blue'})

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

        prevImg = ttk.Button(frame, text='Prev', command=lambda: self.prevImgFunc())
        prevImg.pack(side=LEFT)

        nextImg = ttk.Button(frame, text='Next', command=lambda: self.nextImgFunc())
        nextImg.pack(side=LEFT)

        deleteImage = ttk.Button(frame, text='Undo', command=lambda: self.nextImgFunc())
        deleteImage.pack(side=LEFT)

        self.skipBy = Entry(frame, width=10)
        self.skipBy.insert(END, 10)
        self.skipBy.pack(side=LEFT)

        self.range = Text(frame, width=15, height=1)
        self.range.insert(chars=str(self.iterator) + ' / ' + str(self.maxIteratorValue), index=END)
        self.range.pack(side=LEFT)

        self.disp_img = Label()
        self.disp_img.pack(pady=40)
        img, frame = self.get_frame_by_index()
        result = predict_with_static_image(self.model, self.class_labels, frame)
        # todo tady se pusti KNN a rekne nam cislo classy
        # self.Lb1.activate(1)
        # self.activeListBox()

        self.disp_img.config(image=img)
        self.disp_img.image = img

    def changeLabelName(self, classNumber):
        key = [k for k, v in self.class_labels.items() if v == classNumber]
        self.textOfClass.configure(text=str(key[0]))

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
        if -1 >= self.iterator - int(self.getSkipbyValue()) < self.maxIteratorValue:
            messagebox.showinfo("Information", "Out of bound")
            return None
        self.iterator = self.iterator - int(self.getSkipbyValue())
        self.insertInfoOfRange()
        self.showImage()

    def nextImgFunc(self):
        if -1 >= self.iterator + int(self.getSkipbyValue()) < self.maxIteratorValue:
            messagebox.showinfo("Information","Out of bound")
            return None
        self.iterator = self.iterator + int(self.getSkipbyValue())
        self.insertInfoOfRange()
        self.showImage()

    def confirm(self):
        selected = self.Lb1.curselection()
        if len(selected) == 0:
            return
        className = self.Lb1.get(selected[0])
        classNumber = self.class_labels.get(className)
        image, _ = ImageTk.getimage(self.get_frame_by_index())
        nameOfFile = str(self.iterator).zfill(math.ceil(math.log10((self.maxIteratorValue))))
        list = glob.glob('workingDirectory/*/' + nameOfFile + '.png')
        for x in list:
            if os.path.exists(x):
                os.remove(x)
        # todo pred cislo nejaky nazev argv[1] treba
        image.save('workingDirectory/class' + str(classNumber) + '/' + nameOfFile + '.png', 'PNG')
        self.nextImgFunc()


if __name__ == "__main__":
    Gui()
