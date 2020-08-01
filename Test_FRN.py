import os
import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable, grad
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import threading
import pickle
from collections import OrderedDict
import cv2

import Models.Loss as L
import Models.Feature_extractor as F
from Models import Ops
from Dataset import FBP_dataset_V2, Label_Sampler
import Utils


def main():
    # Create dataset
    trainset = FBP_dataset_V2('train', config['SCUT-FBP-V2'], config['train_index'], config['img_size'], config['crop_size'])
    testset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['test_index'], config['img_size'], config['crop_size'])

    # Get dataset loader
    train_loader = Data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=1, shuffle=True)

    # Lauch Tensorboard
    #t = threading.Thread(target=Utils.launchTensorBoard, args=([config['result_path']]))
    #t.start()

    # Start
    frn = F.Feature_extractor(config)
    frn.eval()
    if config['use_gpu'] and torch.cuda.is_available():
        use_cuda = True
        frn.cuda()

    train_iter = iter(train_loader)
    x, _ = train_iter.next()
    x = x.cuda()
    x = Variable(x, requires_grad=True)
    f = frn(x, config['identity_layer'])[0]
    f = torch.mean( torch.square(f) )
    g = grad(f, x)[0]
    g = torch.abs(g)
    g = torch.sum(g, dim=1)
    g = g.squeeze(0)
    g = g / torch.max(g)

    x = x.squeeze(0)
    x = (x + 1.0) * 0.5
    x = x.data.cpu().numpy()
    g = g.data.cpu().numpy()

    x = np.swapaxes(x, 0, -1)
    x = np.swapaxes(x, 0, 1)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

    cv2.imshow("Face", x)
    cv2.imshow("Mask", g)

    # Apply mask
    index = np.argsort(g, axis=None)
    index = np.unravel_index(index, (224, 224))
    for i in range(3000):
        x[index[0][224*224 - 1 - i], index[1][224*224 - 1 - i], :] = [0.0, 0.0, 1.0]

    cv2.imshow("Combine", x)

    cv2.waitKey(0)



def load_config(root_dir):
    config = OrderedDict()

    config['use_gpu'] = True

    # Hyper-parameters
    config['batch_size'] = 1

    # Pretrained feature extractor and regressor
    feature_networks = ['resnet50_ft_dag', 'senet50_256', 'vgg_face_dag', 'vgg_m_face_bn_dag']
    config['pretrained_path'] = os.path.join(root_dir, 'Pretrained_models')
    config['feature_model'] = feature_networks[3]
    config['identity_layer'] = [5]    # 1 to 5, define which output layer of pretrained model to be used as feature

    # Dataset
    config['SCUT-FBP-V2'] = os.path.join(root_dir, 'SCUT-FBP5500_v2')
    config['img_size'] = 236
    config['crop_size'] = 224
    index_file = os.path.join(config['SCUT-FBP-V2'], 'data_index_1800_200.pkl')
    with open(index_file, 'rb') as readFile:
        data_index = pickle.load(readFile)
    config['train_index'] = data_index['train']
    config['test_index'] = data_index['test']

    print('Configuration loaded and saved.')

    return config



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='D:/ThesisData', type=str)
    args = parser.parse_args()

    # Load config and tensorboard writer
    config = load_config(args.path)

    # Start training
    main()

