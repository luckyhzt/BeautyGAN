import os
import tkinter as tk
from tkinter import filedialog


def getDir(root):
    logdir = filedialog.askdirectory(parent=root)
    return logdir

root = tk.Tk()
root.withdraw()
logdir = getDir(root)
print('\nLog directory: ', logdir, '\n')
if len(logdir) != 0:
    os.system('tensorboard --logdir=' + logdir + ' --port=8008')