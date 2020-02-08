import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math
import models.ops as ops
import models.Basic_model as basic
from torch.nn.utils import weight_norm
from inspect import signature


class Discriminator(nn.Module):

    def __init__(self, in_channels, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        use_bias = (norm_layer==nn.InstanceNorm2d)

        model = []

        model += [
            nn.Conv2d(in_channels, ndf, kernel_size=7, stride=2, padding=3, bias=True),
            nn.LeakyReLU(0.2, True),
        ]

        mult = 1
        mult_prev = 1
        for i in range(n_layers):
            mult_prev = mult
            mult = min(2 ** i, 8)
            model += [
                nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]

        mult_prev = mult
        mult = min(2 ** n_layers, 8)
        model += [
            nn.Conv2d(ndf * mult_prev, ndf * mult, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Output 1-channel prediction map
        model += [ 
            nn.Conv2d(ndf * mult, ndf* mult, kernel_size=3, stride=1, padding=1),
            norm_layer(ndf * mult),
            nn.LeakyReLU(0.2, True),
            
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(ndf * mult, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        ]

        self.model = nn.Sequential(*model)
    

    def forward(self, x, y):
        output = self.model( ops.combine_input(x, y / 5.0) )
        return output.squeeze()



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
            for name, layer in self.layers.named_children():
                content += ' ' + str(name) + ': ' + str(layer) + '\n\n'
            content = content.replace('\n', '  \n')
            writer.add_text(title, content, 0)


    def forward(self, x):

        for name, layer in self.layers.named_children():
            x = layer(x)

        return x.view(x.size(0), -1)



