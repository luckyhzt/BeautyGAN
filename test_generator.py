import torch
from torch.utils import data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
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


def test():
    # Load trained model
    
    exps = [11, 13, 14, 15]

    for e in exps:
        gen_path = 'D:/Program/BeautyGAN/Train/Exp_3_cycle/Result/Exp_' + str(e) + '/generator_300.pth'
        generator = G.Generator(config, None)
        generator.cuda()
        generator.load(gen_path)

        # Dataset
        testset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['test_index'], config['img_size'], config['crop_size'])
        index = 120
        n = 10

        # Test
        x, y = testset[index]
        x = x.unsqueeze(0)
        x = x.cuda()
        #display_img(x.squeeze(0) / 2.0 + 0.5, 'original')
        save_img(x.squeeze(0) / 2.0 + 0.5, 'C:/Users/Zhitong Huang/iCloudDrive/Documents/Thesis/Images/origin_' + str(index) + '.png')

        gap = 4/(n-1)
        score = np.arange(n)
        score = gap * score + 1.0
        score = score.reshape([n, 1])
        y_g = torch.cuda.FloatTensor(score)
        x = x.repeat(n, 1, 1, 1)
        x_g = generator(x, y_g)
        gen_img = torchvision.utils.make_grid(x_g / 2.0 + 0.5, padding=10, nrow=5)
        #display_img(gen_img, 'generated')
        save_img(gen_img, 'C:/Users/Zhitong Huang/iCloudDrive/Documents/Thesis/Images/' + str(index) + '_' + str(e) + '.png')

        #cv2.waitKey(0)



def load_config():
    config = OrderedDict()

    thisDir = os.path.dirname(__file__)

    # Dataset
    config['SCUT-FBP-V2'] = 'D:/ThesisData/SCUT-FBP5500_v2'
    config['img_size'] = 236
    config['crop_size'] = 224
    config['unlabel_samples'] = 2000
    # Train and test set
    index_file = os.path.join(config['SCUT-FBP-V2'], 'data_index_1800_200.pkl')
    with open(index_file, 'rb') as readFile:
        data_index = pickle.load(readFile)
    config['train_index'] = data_index['train']
    config['test_index'] = data_index['test']
    config['residual_blocks'] = 4

    return config



def display_img(x, name):
    img = x.data.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = ( img * 255.0 ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, img)
    #fig, ax = plt.subplots()
    #ax.imshow(img)
    #plt.show()
    cv2.im


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
    test()