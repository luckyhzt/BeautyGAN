import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math
import numpy as np
from models.Basic_model import Residual_block


class Regressor(nn.Module):

    def __init__(self, config, writer):
        super(Regressor, self).__init__()

        self.resnet = torchvision.models.resnet18(pretrained=True)

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.norm_mean = torch.from_numpy(mean).view(1, -1, 1, 1).cuda()
        self.norm_std = torch.from_numpy(std).view(1, -1, 1, 1).cuda()

        self.output = nn.Sequential(
            nn.Linear(512, 1),
        )


    def forward(self, x):
        y = x / 2.0 + 0.5    #from[-1, 1] to [0, 1]
        y = (y - self.norm_mean) / self.norm_std    #normalize

        for name, layer in self.resnet.named_children():
            if name != 'fc':
                y = layer(y)
                
        y = y.view(y.size(0), -1)

        return self.output(y)
    

    def save(self, path):
        torch.save(self.state_dict(), path)

    
    def load(self, path):
        self.load_state_dict(torch.load(path))
        
        