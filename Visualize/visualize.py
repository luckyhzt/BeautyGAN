import os
import cv2
import torch
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

''' Unused function

def visual_mask(oimg, mask, mimg, save_path):
    filename = os.path.join(save_path, 'mask.png')
    arrayname = os.path.join(save_path, 'mask.npy')

    if mask.size(1) == 1:
        mask = mask.repeat(1, 3, 1, 1)
    all_images = torch.cat([oimg, mask, mimg], 0)
    images_array = all_images.data.cpu().numpy()
    ori_array, images_array = make_grid(images_array, nrow=3, pad=10)

    cv2.imwrite(filename, cv2.cvtColor(images_array, cv2.COLOR_RGB2BGR))
    np.save(arrayname, ori_array)'''


def add_label(imgs, label):
    if label.is_cuda:
        label = label.data.cpu().numpy()
    else:
        label = label.data.numpy()

    imgs = tensor_to_cv(imgs)

    nrow, ncol, chs = imgs.shape
    nlab, nlvl = label.shape
    img_width = int(ncol / nlab)
    
    label = ( np.argmax(label, axis=1) / nlvl ) * 4.0 + 1.0 

    for i in range(nlab):
        text = '{:.2f}'.format(label[i])
        y = nrow - 1
        x = int( (i + 0.5) * img_width )
        cv2.putText(imgs, text, (x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)

    return cv_to_tensor(imgs)


def tensor_to_cv(tensor):
    if tensor.is_cuda:
        mat = tensor.data.cpu().numpy()
    else:
        mat = tensor.data.numpy()

    mat = np.moveaxis(mat, 0, -1)
    mat = (mat * 255).astype(np.uint8)
    mat = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    return mat


def cv_to_tensor(mat):
    mat = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)
    mat = mat.astype(np.float32) / 255.0
    mat = np.moveaxis(mat, -1, 0)

    return torch.from_numpy(mat)


def make_grid(imgs, nrow=1, pad=10):
    nimage, channel, rows, cols = imgs.shape
    ncol = int(np.ceil(nimage / nrow))

    img_array = np.zeros([nrow*rows + (nrow+1)*pad, ncol*cols + (ncol+1)*pad, channel], dtype=np.float32)
    #ori_array = np.zeros([nrow*rows + (nrow+1)*pad, ncol*cols + (ncol+1)*pad, channel], dtype=np.float32)

    i = -1
    for r in range(0, nrow):
        for c in range(0, ncol):
            i += 1
            if i < nimage:
                r_start = pad + r*(rows + pad)
                r_end = r_start + rows
                c_start = pad + c*(cols + pad)
                c_end = c_start + cols
                #ori_img = np.moveaxis(imgs[i, :, :, :], 0, -1)
                #norm_img = cv2.normalize(ori_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                #img_array[r_start:r_end, c_start:c_end, :] = norm_img
                #ori_array[r_start:r_end, c_start:c_end, :] = ori_img
                img_array[r_start:r_end, c_start:c_end, :] = np.moveaxis(imgs[i, :, :, :], 0, -1)
    
    return img_array
