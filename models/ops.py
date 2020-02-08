import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math
import numpy as np


def combine_input(inputx, label):
    '''
    Concatenate input (images) with label (can be single number or one-hot label)
    input size: 4-dimensional: B x C x W x H
    label size: 2-dimensional: B x N    or    1-dimensional: B
    '''
    # If one of input use cuda, put both in cuda
    if inputx.is_cuda or label.is_cuda:
        if not label.is_cuda:
            label = label.cuda()
        if not inputx.is_cuda:
            inputx = inputx.cuda()

    # If label is 1-dim, unsqueeze it
    if len(label.size()) == 1:
        label = label.view(-1, 1)

    # Shape
    batch_size = inputx.size(0)
    channel = label.size(1)
    w = inputx.size(2)
    h = inputx.size(3)
    # Expand label to fit inputx
    label = label.view(batch_size, channel, 1, 1)
    label = label.expand(batch_size, channel, w, h)

    # Return concated input
    return torch.cat([inputx, label], dim=1)


def convert_one_hot(label):
    batch_size, num_class = label.size()

    index = torch.argmax(label, dim=1)
    onehot = torch.zeros(label.size(), device=label.device)
    for i in range (batch_size):
        onehot[i, index[i]] = 1.0
    
    return onehot


'''def generate_fake_label(label):
    batch_size, num_class = label.size()

    fake_label = torch.zeros(label.size(), device=label.device)
    index = torch.argmax(label, dim=1)
    index = index.data.cpu().numpy()

    for i in range (batch_size):
        prop = np.array([23, 270,  72,  35])
        prop[int(index[i])] = 0
        prop = prop / np.sum(prop)
        fake_index = np.random.choice(np.arange(0, num_class), p=prop)
        fake_label[i, fake_index] = 1.0
    
    return fake_label'''


def generate_fake_label(label):
    batch_size, num_class = label.size()

    fake_label = torch.rand(label.size(), device=label.device)
    index = torch.argmax(label, dim=1)
    index = index.data.cpu().numpy()

    for i in range (batch_size):
        prop = np.array([34, 205, 166, 69, 25])
        prop[int(index[i])] = 0
        prop = prop / np.sum(prop)
        fake_index = np.random.choice(np.arange(0, num_class), p=prop)
        fake_label[i, fake_index] = 1.1
        # Normalize to 1
        fake_label[i, :] = fake_label[i, :] / torch.sum(fake_label[i, :])
    
    return fake_label


class Img_to_zero_center(object):
    def __int__(self):
        pass
    def __call__(self, t_img):
        '''
        :param img:tensor be 0-1
        :return:
        '''
        t_img=(t_img-0.5)*2
        return t_img


class Reverse_zero_center(object):
    def __init__(self):
        pass
    def __call__(self,t_img):
        t_img=t_img/2+0.5
        return t_img
