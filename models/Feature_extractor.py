import imp
import torch
import os
import torch.nn as nn
import numpy as np
from models.pretrained_models import senet50_256, resnet50_ft_dag, vgg_face_dag, vgg_m_face_bn_dag
import PIL


class Feature_extractor(nn.Module):

    def __init__(self, config):
        model_type = config['feature_model']
        path = config['pretrained_path']

        super(Feature_extractor, self).__init__()
        weights_path = os.path.join(path, model_type+'.pth')

        if model_type == 'resnet50_ft_dag':
            self.model = resnet50_ft_dag.resnet50_ft_dag(weights_path=weights_path)
        elif model_type == 'senet50_256':
            self.model = senet50_256.senet50_256(weights_path=weights_path)
        elif model_type == 'vgg_face_dag':
            self.model = vgg_face_dag.vgg_face_dag(weights_path=weights_path)
        elif model_type == 'vgg_m_face_bn_dag':
            self.model = vgg_m_face_bn_dag.vgg_m_face_bn_dag(config['use_gpu'], weights_path=weights_path)
    

    def forward(self, x, output_layer):
        return self.model(x, output_layer)