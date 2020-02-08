import os
import pickle
from tkinter import *



if __name__ == '__main__':
    thisDir = os.path.dirname(__file__)
    path = os.path.join(thisDir, 'Result', 'useful_old_result')
    results =  os.listdir(path)

    def CurSelet(evt):
        value = str((listbox.get(ACTIVE)))
        logdir = os.path.join(path, value)
        master.destroy()
        os.system('tensorboard --logdir=' + logdir)
        

    master = Tk()
    textFont = ('Microsoft YaHei UI', 20)
    listbox = Listbox(master, font=textFont)
    listbox.bind('<Double-Button-1>',CurSelet)
    listbox.pack()

    for res in results:
        listbox.insert(END, res)

    mainloop()

