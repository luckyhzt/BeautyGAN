import os
import tkinter as tk
from tkinter import filedialog


def getDir(root):
    logdir = filedialog.askdirectory(parent=root)
    return logdir


def run_tensorboard(event):
    port = entry.get()
    root.quit()
    root.destroy()
    if len(logdir) != 0:
        os.system('python -m tensorboard.main --logdir=' + logdir + ' --port=' + port)


root = tk.Tk()
lbl = tk.Label(root)
lbl.grid(column=0, row=0, columnspan=2)
tk.Label(root, text='Port: ').grid(column=0, row=1)
entry = tk.Entry(root)
entry.grid(column=1, row=1)
logdir = getDir(root)
lbl.config(text = '\nLog directory: ' + logdir)

root.bind('<Return>', run_tensorboard)

root.focus_force()
entry.focus_set()

root.mainloop()