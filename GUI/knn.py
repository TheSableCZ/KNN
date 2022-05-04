from tkinter import messagebox
from tkinter import ttk
from tkinter import *
from PIL import Image, ImageTk
import sys
import os
import cv2


class Gui:

    def __init__(self):
        self.range = None
        self.disp_img = None
        self.skipBy = None
        self.Lb1 = None
        self.ws = Tk()
        self.ws.bind("<Key>", self.onKeyPress)
        self.ws.title('PythonGuides')
        self.ws.geometry('1024x720')
        self.ws.config(bg='#4a7a8c')
        self.iterator = 0
        self.skipValue = 0
        self.video_capture = cv2.VideoCapture('../pictures/video.mp4')
        self.maxIteratorValue = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.class_labels = {
            "Warrior_I": 0,
            "Warrior_II": 1,
            "Tree": 2,
            "Triangle": 3,
            "Standing_Splits": 4
        }
        self.initGuiElements()
        self.ws.mainloop()

    def get_frame_by_index(self):
        current_index = self.video_capture.get(1)
        self.video_capture.set(1, self.iterator)

        success, frame = self.video_capture.read()

        if success:
            return ImageTk.PhotoImage(image=Image.fromarray(frame))
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
        return None

    def activeListBox(self):
        self.Lb1.itemconfig(1, {'bg': 'Blue'})

    def initGuiElements(self):
        frame = Frame(self.ws)
        frame.pack(side=BOTTOM)

        confirm = ttk.Button(frame, text='Confirm', command=lambda: self.confirm())
        confirm.pack(side=LEFT)

        self.Lb1 = Listbox(frame, selectmode = SINGLE, exportselection=False )
        self.loadClassesToList()
        self.Lb1.pack(side=LEFT)

        prevImg = ttk.Button(frame, text='Prev', command=lambda: self.prevImgFunc())
        prevImg.pack(side=LEFT)

        nextImg = ttk.Button(frame, text='Next', command=lambda: self.nextImgFunc())
        nextImg.pack(side=LEFT)

        self.skipBy = Entry(frame, width=10)
        self.skipBy.insert(END, 10)
        self.skipBy.pack(side=LEFT)

        self.range = Text(frame, width=15, height=1)
        self.range.insert(chars=str(self.iterator) + ' / ' + str(self.maxIteratorValue), index=END)
        self.range.pack(side=LEFT)

        self.disp_img = Label()
        self.disp_img.pack(pady=20)
        img = self.get_frame_by_index()
        #todo tady se pusti KNN a rekne nam cislo classy
        self.Lb1.activate(1)
        self.activeListBox()

        self.disp_img.config(image=img)
        self.disp_img.image = img

    def onKeyPress(self, event):
        if event.char == 's':
            self.confirm()
        elif event.char == 'd':
            self.nextImgFunc()
        elif event.char == 'a':
            self.prevImgFunc()

    def prevImgFunc(self):
        if -1 >= self.iterator - int(self.getSkipbyValue()) < self.maxIteratorValue:
            messagebox.showinfo("Information","Out of bound")
            return None
        self.iterator = self.iterator - int(self.getSkipbyValue())
        self.insertInfoOfRange()
        img = self.get_frame_by_index()
        self.disp_img.config(image=img)
        self.disp_img.image = img

    def nextImgFunc(self):
        if -1 >= self.iterator + int(self.getSkipbyValue()) < self.maxIteratorValue:
            messagebox.showinfo("Information","Out of bound")
            return None
        self.iterator = self.iterator + int(self.getSkipbyValue())
        self.insertInfoOfRange()
        img = self.get_frame_by_index()
        self.disp_img.config(image=img)
        self.disp_img.image = img

    def confirm(self):
        selected = self.Lb1.curselection()
        className = self.Lb1.get(selected[0])
        classNumber = self.class_labels.get(className)
        image = ImageTk.getimage(self.get_frame_by_index())
        image.save('video_numOfpicture' + str(self.iterator)+ 'from' + str(self.maxIteratorValue) +'_class' +str (classNumber)+ '.png', 'PNG')
        image.close()





if __name__ == "__main__":
    Gui()
