import imp
import torch
import os
import torch.nn as nn
import numpy as np
from models.pretrained_models import senet50_256, resnet50_ft_dag, vgg_face_dag, vgg_m_face_bn_dag
import PIL


class Feature_extractor(nn.Module):

    def __init__(self, model_type, path):
        super(Feature_extractor, self).__init__()
        weights_path = os.path.join(path, model_type+'.pth')

        if model_type == 'resnet50_ft_dag':
            self.model = resnet50_ft_dag.resnet50_ft_dag(weights_path=weights_path)
        elif model_type == 'senet50_256':
            self.model = senet50_256.senet50_256(weights_path=weights_path)
        elif model_type == 'vgg_face_dag':
            self.model = vgg_face_dag.vgg_face_dag(weights_path=weights_path)
        elif model_type == 'vgg_m_face_bn_dag':
            self.model = vgg_m_face_bn_dag.vgg_m_face_bn_dag(weights_path=weights_path)

        mean = np.array(self.model.meta['mean'], dtype=np.float32)
        std = np.array(self.model.meta['std'], dtype=np.float32)
        
        self.norm_mean = torch.from_numpy(mean).view(1, -1, 1, 1).cuda()
        self.norm_std = torch.from_numpy(std).view(1, -1, 1, 1).cuda()
    

    def forward(self, x, output_layer):
        y = x / 2.0 + 0.5    # from [-1,1] to [0,1]
        y = y * 255.0   # from [0,1] to [0,255]
        y = (y - self.norm_mean) / self.norm_std    # normalize

        return self.model(y, output_layer)