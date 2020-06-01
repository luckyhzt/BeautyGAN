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
    # Load weights of Generator
    generator = G.Generator(config, None)
    generator.cuda()
    generator.load(config['param_path'])
    generator.eval()
    # Load face recognition network
    FRN = F.Feature_extractor(config)
    FRN.cuda()
    FRN.eval()
    # Load face beauty predictor
    FBP = R.Regressor(config, None)
    FBP.cuda()
    FBP.load(config['regressor_path'])
    FBP.eval()
    # Load dataset
    trainset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['train_index'], config['img_size'], config['crop_size'])
    testset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['test_index'], config['img_size'], config['crop_size'])
    sampler = Label_Sampler(config['SCUT-FBP-V2'], config['train_index'])
    # Show result
    show_index = range(200)
    show_score = np.array([1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75])
    save_result(generator, testset, show_index, show_score)
    # Test idenity loss
    #beauty_loss, identity_loss = test_loss(generator, FBP, FRN, trainset, sampler, 20)
    #print(beauty_loss, identity_loss)


def test_loss(generator, FBP, FRN, dataset, sampler, num_samples):
    n = 10
    beauty_loss = 0
    identity_loss = 0

    for i in range(0, n):
        x, y = dataset[i]
        x = x.unsqueeze(0)
        x = x.cuda()
        x_g = x.repeat(num_samples, 1, 1, 1)
        y_g = sampler.sample(num_samples).cuda()
        x_g = generator(x_g, y_g)
        # Calculate beauty loss
        y_g_g = FBP(x_g)
        beauty_loss += L.MSELoss(y_g_g, y_g).data.cpu().numpy()
        # Calculate identity loss
        u_feat = FRN(x, config['feature_layer'])[0].repeat(num_samples, 1, 1, 1)
        g_feat = FRN(x_g, config['feature_layer'])[0]
        identity_loss += L.MSELoss(g_feat, u_feat).data.cpu().numpy()

    return beauty_loss / n, identity_loss / n


def display_result(generator, dataset, index, score):
    result = []
    for i in index:
        x, y = dataset[i]
        print(str(i) + ' : ', str(y[0]))
        x = x.unsqueeze(0)
        x = x.cuda()
        x_g = x.repeat(score.shape[0], 1, 1, 1)
        y_g = torch.cuda.FloatTensor(score.reshape([-1, 1]))
        x_g = generator(x_g, y_g)
        x_g = torch.cat([x, x_g], dim=0)
        result.append(x_g)
    result = torch.cat(result, dim=0)
    result = torchvision.utils.make_grid(result / 2.0 + 0.5, padding=10, nrow=score.shape[0]+1)
    utils.display_img(result, 'generated faces')


def save_result(generator, dataset, index, score):
    for i in index:
        x, y = dataset[i]
        name = '%.4f' % np.round(y[0], 4)
        save_path = os.path.join(config['image_save_path'], name)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        print(str(i) + ' : ', str(y[0]))
        x = x.unsqueeze(0)
        x = x.cuda()
        x_g = x.repeat(score.shape[0], 1, 1, 1)
        y_g = torch.cuda.FloatTensor(score.reshape([-1, 1]))
        x_g = generator(x_g, y_g)
        # Save original image
        save_tensor_as_image(x[0], os.path.join(save_path, 'original.png'))
        # Save generated image
        for k in range(score.shape[0]):
            save_tensor_as_image(x_g[k], os.path.join(save_path, str(score[k]) + '.png'))



def save_tensor_as_image(tensor, path):
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = (tensor + 1) / 2
    img = torchvision.transforms.ToPILImage()(tensor)
    img.save(path)




def load_config():
    thisDir = os.path.dirname(__file__)
    result_path = 'D:/Program/BeautyGAN/Train/Exp_3_cycle/Result'
    experiment = 'Final'
    param = 'generator_300.pth'

    exp_path = os.path.join(result_path, experiment)
    with open(os.path.join(exp_path, 'config.pkl'), 'rb') as readFile:
        config = pickle.load(readFile)
    
    config['exp_path'] = exp_path
    config['param_path'] = os.path.join(exp_path, param)
    config['image_save_path'] = 'Test/Result'

    return config


if __name__ == '__main__':
    config = load_config()
    main()
