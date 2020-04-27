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

import Test.utils as utils
from dataset import FBP_dataset, FBP_dataset_V2, Label_Sampler
import models.Generator as G
import models.Feature_extractor as F
import models.Regressor as R

import models.Loss as L


def main():
    trainset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['train_index'], config['img_size'], config['crop_size'])
    testset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['test_index'], config['img_size'], config['crop_size'])
    # Compare results of different experiments:
    compare([11, 13, 15, 23], testset, 110, [1.5, 4.5], 7)
    

def compare(exps, testset, index, score_range, n):
    # Load trained model
    all_images = []

    for e in exps:
        gen_path = 'D:/Program/BeautyGAN/Train/Exp_3_cycle/Result/Exp_' + str(e) + '/generator_300.pth'
        generator = G.Generator(config, None)
        generator.cuda()
        generator.load(gen_path)

        # Test
        x, y = testset[index]
        x = x.unsqueeze(0)
        x = x.cuda()

        gap = (score_range[1] - score_range[0])/(n-1)
        score = np.arange(n)
        score = gap * score + score_range[0]
        score = score.reshape([n, 1])
        y_g = torch.cuda.FloatTensor(score)
        x_g = x.repeat(n, 1, 1, 1)
        x_g = generator(x, y_g)
        output = torch.cat([x, x_g], dim=0)
        all_images.append(output)
    
    all_images = torch.cat(all_images, dim=0)
    gen_img = torchvision.utils.make_grid(all_images / 2.0 + 0.5, padding=10, nrow=8)
    utils.display_img(gen_img, 'generated')


def load_config():
    thisDir = os.path.dirname(__file__)
    result_path = 'D:/Program/BeautyGAN/Train/Exp_3_cycle/Result'
    experiment = 'Exp_11'
    exp_path = os.path.join(result_path, experiment)
    with open(os.path.join(exp_path, 'config.pkl'), 'rb') as readFile:
        config = pickle.load(readFile)

    return config


if __name__ == '__main__':
    config = load_config()
    main()
