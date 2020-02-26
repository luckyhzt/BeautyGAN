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
    label = np.load(config['label_path'])
    label = average_score(label[config['trainset_index'], :])

    n = 21
    index = 90

    # Load trained model
    gen_path = 'D:/Program/BeautyGAN/Train/Exp_3_cycle/Result/Exp_3/generator_240.pth'
    generator = G.Generator(config, None)
    generator.cuda()
    generator.load(gen_path)

    # Dataset
    #dataset = FBP_dataset_V2(config['uimg_path'], config['unlabel_index'])
    dataset = FBP_dataset('train', config['limg_path'], config['testset_index'], config['label_path'], \
        config['img_size'], config['crop_size'])

    # Test
    x, _ = dataset[index]
    x = x.unsqueeze(0)
    x = x.cuda()
    display_img(x.squeeze(0), 'original')
    for i in range (n):
        gap = 4/(n-1)
        score = np.float32( gap * i + 1.0 )
        times = ((score-gap/2 <= label) & (label < score+gap/2)).sum()
        y = torch.cuda.FloatTensor([[score]])
        x_g = generator(x, y)
        display_img(x_g.squeeze(0), "{:.2f}".format(score) + '  ' +str(times))
    cv2.waitKey(0)



def load_config():
    config = OrderedDict()

    thisDir = os.path.dirname(__file__)

    # Dataset
    config['label_path'] = 'D:/ThesisData/SCUT-FBP/Label/label.npy'
    config['limg_path'] = 'D:/ThesisData/SCUT-FBP/Pad_Square'
    config['SCUT-FBP-V2'] = 'D:/ThesisData/SCUT-FBP5500_v2'
    config['img_size'] = 236
    config['crop_size'] = 224
    config['train_samples'] = 400
    config['test_samples'] = 100
    config['unlabel_samples'] = 2000
    # Train and test set
    num_images = 500
    img_index = np.arange(num_images)
    #np.random.shuffle(img_index)
    config['trainset_index'] = img_index[:config['train_samples']]
    config['testset_index'] = img_index[config['train_samples']:]
    config['unlabel_index'] = np.arange(config['unlabel_samples'])

    return config



def display_img(x, name):
    img = x.data.cpu().numpy()
    img = np.moveaxis(img, 0, -1)
    img = ( (img + 1.0) / 2.0 * 255 ).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(name, img)


def average_score(label):
    avg = np.sum(label, axis=1, keepdims=True)
    num = np.count_nonzero(label, axis=1).reshape([-1, 1])
    avg = avg / num

    return avg




if __name__ == '__main__':
    config = load_config()
    test()