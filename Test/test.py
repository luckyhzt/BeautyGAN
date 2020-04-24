import os
import sys
thisDir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(thisDir, os.pardir)))

import torch
from torch.utils import data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle
from datetime import datetime
from collections import OrderedDict
import PIL
import threading
import cv2

from dataset import FBP_dataset, FBP_dataset_V2, Label_Sampler
import models.Generator as G
import models.Feature_extractor as F
import models.Regressor as R

import models.Loss as L


def load_config():
    thisDir = os.path.dirname(__file__)
    result_path = 'D:/Program/BeautyGAN/Train/Exp_3_cycle/Result'
    experiment = 'Exp_11'
    param = 'generator_300.pth'

    exp_path = os.path.join(result_path, experiment)
    with open(os.path.join(exp_path, 'config.pkl'), 'rb') as readFile:
        config = pickle.load(readFile)
    
    config['exp_path'] = exp_path
    config['param_path'] = os.path.join(exp_path, param)

    return config


def compare():
    # Load trained model
    exps = [11, 13, 14, 15]
    all_images = []

    for e in exps:
        gen_path = 'D:/Program/BeautyGAN/Train/Exp_3_cycle/Result/Exp_' + str(e) + '/generator_300.pth'
        generator = G.Generator(config, None)
        generator.cuda()
        generator.load(gen_path)

        # Dataset
        testset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['test_index'], config['img_size'], config['crop_size'])
        index = 120
        n = 7

        # Test
        x, y = testset[index]
        x = x.unsqueeze(0)
        x = x.cuda()

        gap = 3.6/(n-1)
        score = np.arange(n)
        score = gap * score + 1.2
        score = score.reshape([n, 1])
        y_g = torch.cuda.FloatTensor(score)
        x_g = x.repeat(n, 1, 1, 1)
        x_g = generator(x, y_g)
        output = torch.cat([x, x_g], dim=0)
        all_images.append(output)
    
    all_images = torch.cat(all_images, dim=0)
    gen_img = torchvision.utils.make_grid(all_images / 2.0 + 0.5, padding=10, nrow=8)
    display_img(gen_img, 'generated')

    cv2.waitKey(0)



def display_img(x, name):
    img = x.data.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = ( img * 255.0 ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, img)
    #fig, ax = plt.subplots()
    #ax.imshow(img)
    #plt.show()


def save_img(x, name):
    img = x.data.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = ( img * 255.0 ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)


def average_score(label):
    avg = np.sum(label, axis=1, keepdims=True)
    num = np.count_nonzero(label, axis=1).reshape([-1, 1])
    avg = avg / num

    return avg




if __name__ == '__main__':
    config = load_config()
    compare()
