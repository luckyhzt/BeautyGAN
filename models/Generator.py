import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math
from models.Basic_model import Residual_block
import models.Basic_model as basic
import models.ops as ops
from torch.nn.utils import weight_norm
from inspect import signature


'''class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        cond_size = config['num_level']

        self.norm_layer = nn.InstanceNorm2d
        # Decide whether use bias
        if self.norm_layer == nn.BatchNorm2d:
            self.use_bias = False
        else: self.use_bias = True

        # Down-sample layers with condition concat
        layers = [
            nn.InstanceNorm2d(3),
            nn.ReflectionPad2d(3),
            basic.ConcatConv2d(3+cond_size, 64, kernel_size=7, stride=1, padding=0, norm=None, activation=nn.ReLU(inplace=True)),
            basic.ConcatConv2d(64+cond_size, 128, kernel_size=3, stride=2, padding=1, norm=None, activation=nn.ReLU(inplace=True)),
            basic.ConcatConv2d(128+cond_size, 256, kernel_size=3, stride=2, padding=1, norm=None, activation=nn.ReLU(inplace=True)),
            basic.ConcatConv2d(256+cond_size, 512, kernel_size=3, stride=2, padding=1, norm=None, activation=nn.ReLU(inplace=True)),
            #basic.ConcatConv2d(512+cond_size, 512, kernel_size=3, stride=2, padding=1, norm=None, activation=self.relu),
        ]

        # Residual layers
        for _ in range(6):
            layers += [Residual_block(512, self.norm_layer, use_bias=self.use_bias)]

        # Up-sample layers
        layers += [
            #nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1, bias=self.use_bias),
            #self.norm_layer(512),
            #nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=self.use_bias),
            self.norm_layer(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=self.use_bias),
            self.norm_layer(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=self.use_bias),
            self.norm_layer(64),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            weight_norm( nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0, bias=False) ),
            nn.Tanh(),
        ]
        self.layers = nn.Sequential(*layers)

        # Write architecture to tensorboard
        title = str(self.__class__.__name__)
        content = ''
        for name, layer in self.layers.named_children():
            content += ' ' + str(name) + ': ' + str(layer) + '\n\n' 
        content = content.replace('\n', '  \n')
        writer.add_text(title, content, 0)
    

    def forward(self, x, cond):
        for name, layer in self.layers.named_children():
            if str(layer)[0:6] == 'Concat':
                x = layer(x, cond)
            else:
                x = layer(x)

            
        return x'''



class Generator(nn.Module):
    def __init__(self, config, writer):
        super(Generator, self).__init__()

        # Down-sample layers
        down_sample_layers = [
            nn.InstanceNorm2d(3),
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=True),
        ]
        self.down_sample = nn.Sequential(*down_sample_layers)
        
        # Conditional Instance Norm layer
        self.cond_norm = basic.Condition_IN(1, 512)

        # Residual layers
        residual_layers = []
        for _ in range(config['residual_blocks']):
            residual_layers += [Residual_block(512, None, use_bias=True)]
        self.residual =  nn.Sequential(*residual_layers)

        # Up-sample layers
        up_sample_layers = [
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=0, bias=True),
            nn.Tanh(),
        ]
        self.up_sample = nn.Sequential(*up_sample_layers)

        # Write architecture to tensorboard
        if writer != None:
            title = str(self.__class__.__name__)
            content = ''
            for name, layer in self.named_children():
                content += ' ' + str(name) + ': ' + str(layer) + '\n\n' 
            content = content.replace('\n', '  \n')
            writer.add_text(title, content, 0)
    

    def forward(self, x, cond):
        x = self.down_sample(x)
        x = self.cond_norm(x, cond)
        x = self.residual(x)
        x = self.up_sample(x)
        return x

    
    def save(self, path):
        torch.save(self.state_dict(), path)

    
    def load(self, path):
        self.load_state_dict(torch.load(path))