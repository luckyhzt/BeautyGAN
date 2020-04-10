import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from datetime import datetime
import os
import argparse
import pickle
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter



def load_config(root_dir):
    config = OrderedDict()

    config['use_gpu'] = True

    # Hyper-parameters
    config['batch_size'] = 16
    config['max_epoch'] = 300
    config['lr'] = 1e-4
    config['lr_decay'] = 0.3
    config['lr_decay_epoch'] = 100
    config['alpha_g'] = 1.0    # weights of generator training against discriminator
    config['alpha_d'] = 1.0    # weights of discriminator training with real samples
    config['identity_loss_weight'] = 0.8
    config['beauty_loss_weight'] = 1.3
    config['consist_loss_weight'] = 0.08
    config['residual_blocks'] = 4

    # Log
    config['log_step'] = 10
    config['image_save_step'] = 100
    config['num_visualize_images'] = 5   # num of generated images to be saved
    # Save checkpoint
    config['cpt_save_epoch'] = 10
    config['cpt_save_min_epoch'] = 200

    # Pretrained feature extractor and regressor
    feature_networks = ['resnet50_ft_dag', 'senet50_256', 'vgg_face_dag', 'vgg_m_face_bn_dag']
    config['pretrained_path'] = os.path.join(root_dir, 'Pretrained_models')
    config['feature_model'] = feature_networks[3]
    config['identity_layer'] = [5]    # 1 to 5, define which output layer of pretrained model to be used as feature
    config['consist_layer'] = [0]
    config['regressor_path'] = os.path.join(config['pretrained_path'], 'pretrained_regressor_V2.pth')

    # Dataset
    config['SCUT-FBP-V2'] = os.path.join(root_dir, 'SCUT-FBP5500_v2')
    config['img_size'] = 236
    config['crop_size'] = 224
    index_file = os.path.join(config['SCUT-FBP-V2'], 'data_index_1800_200.pkl')
    with open(index_file, 'rb') as readFile:
        data_index = pickle.load(readFile)
    config['train_index'] = data_index['train']
    config['test_index'] = data_index['test']

    # Create directory to save result
    thisDir = os.path.dirname(__file__)
    date_time = datetime.now()
    time_str = date_time.strftime("%Y%m%d_%H-%M-%S")
    result_path = os.path.join(thisDir, 'Result')
    if not os.path.isdir(result_path):
        os.mkdir(result_path)
    result_path = os.path.join(result_path, time_str)
    os.mkdir(result_path)
    config['result_path'] = result_path

    # Save config in pkl file
    pk_file = os.path.join(result_path, 'config.pkl')
    with open(pk_file, 'wb') as outFile:
        pickle.dump(config, outFile)

    print('Configuration loaded and saved.')


    # Create Tensorboard to save result
    writer = SummaryWriter(config['result_path'])
    # Write all configs to tensorboard
    content = ''
    for keys,values in config.items():
            content += keys + ': ' + str(values) + '  \n  \n'
    writer.add_text('config', content, 0)

    # Save config in txt file
    txt_file = os.path.join(result_path, 'config.txt')
    with open(txt_file, 'w') as outFile:
        outFile.write(content)

    return config, writer


