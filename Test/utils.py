import numpy as np
import cv2
import torch

def display_img(x, name):
    img = x.data.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = ( img * 255.0 ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, img)
    #fig, ax = plt.subplots()
    #ax.imshow(img)
    #plt.show()
    cv2.waitKey(0)


def save_img(x, name):
    img = x.data.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = ( img * 255.0 ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)