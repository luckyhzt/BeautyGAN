import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math
import Models.Ops as ops
import Models.Basic_model as basic
from torch.nn.utils import weight_norm
from inspect import signature



class PatchDiscriminator(nn.Module):
    def __init__(self, config, writer):
        super(PatchDiscriminator, self).__init__()

        # layers
        layers = [
            nn.ReflectionPad2d(2),
            weight_norm( nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=0, bias=True) ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            weight_norm( nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True) ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            weight_norm( nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True) ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            weight_norm( nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True) ),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            weight_norm( nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=True) ),
            nn.Dropout2d(p=0.5, inplace=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            weight_norm( nn.Conv2d(512, 1, kernel_size=3, stride=2, padding=1, bias=True) ),
        ]
        self.layers = nn.Sequential(*layers)

        # Write architecture to tensorboard
        if writer != None:
            title = str(self.__class__.__name__)
            content = ''
            for name, layer in self.named_children():
                content += ' ' + str(name) + ': ' + str(layer) + '\n\n'
            content = content.replace('\n', '  \n')
            writer.add_text(title, content, 0)


    def forward(self, x):

        for name, layer in self.layers.named_children():
            x = layer(x)

        return x.view(x.size(0), -1)



