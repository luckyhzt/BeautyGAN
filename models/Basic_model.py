import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math
from torch.nn.utils import weight_norm


class Residual_block(nn.Module):

    def __init__(self, in_channels, norm_layer, use_bias):
        super(Residual_block, self).__init__()

        conv_blocks = []

        conv_blocks += [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            #norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            #norm_layer(in_channels),
        ]
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu_out( x + self.conv_blocks(x) )



class ConcatConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm, activation):
        super(ConcatConv2d, self).__init__()
        if norm == nn.BatchNorm2d:
            self.use_bias = False
        else: self.use_bias = True

        blocks = []

        if norm == weight_norm:
            blocks += [ norm( nn.Conv2d(in_channels, out_channels, \
                kernel_size=kernel_size, stride=stride, padding=padding, bias=self.use_bias) ) ]
        else:
            blocks += [ nn.Conv2d(in_channels, out_channels, \
                kernel_size=kernel_size, stride=stride, padding=padding, bias=self.use_bias) ]
            if norm != None:
                blocks += [ norm(out_channels) ]
        
        if activation != None:
            blocks += [ activation ]

        self.block = nn.Sequential(*blocks)
    

    def forward(self, x, cond):
        cond = cond.view(cond.size(0), cond.size(1), 1, 1)
        cond = cond.expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond], dim=1)

        return self.block(x)



class Concat2d(nn.Module):
    def __init__(self):
        super(Concat2d, self).__init__()
    
    def forward(self, x, cond):
        cond = cond.view(cond.size(0), cond.size(1), 1, 1)
        cond = cond.expand(-1, -1, x.size(2), x.size(3))

        return torch.cat([x, cond], dim=1)



class Condition_IN(nn.Module):
    def __init__(self, cond_size, channels):
        super(Condition_IN, self).__init__()

        self.norm = nn.InstanceNorm2d(channels)

        self.fc = nn.Linear(cond_size, channels)
        self.sigm1 = nn.Sigmoid()
        self.var = nn.Linear(channels, channels)
        self.sigm2 = nn.Sigmoid()
        self.mean = nn.Linear(channels, channels)
    
    def forward(self, x, cond):
        x = self.norm(x)
        cond = self.sigm1( self.fc(cond) )
        var = self.sigm2( self.var(cond) ) * 2
        mean = self.mean(cond)

        var = var.view(var.size(0), var.size(1), 1, 1)
        mean = mean.view(mean.size(0), mean.size(1), 1, 1)

        return x * var + mean



class Interpolate(nn.Module):
    def __init__(self, scale, mode):
        super(Interpolate, self).__init__()
        self.mode = mode
        self.scale = scale
    
    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, mode=self.mode, align_corners=False)
        return x



class resize_layer(nn.Module):
    def __init__(self, channel, size):
        super(resize_layer, self).__init__()
        self.channel = channel
        self.size = size
    
    def forward(self, x):
        return x.view(x.size(0), self.channel, self.size, self.size)



class select_layer(nn.Module):
    def __init__(self, channel, block):
        super(select_layer, self).__init__()
        self.channel = channel
        self.block = block
    
    def forward(self, x, cond):
        batch_size = x.size(0)

        x = torch.split(x, self.channel, dim=1)
        cond = torch.split(cond, 1, dim=1)

        result = 0

        for i in range(self.block):
            result += x[i] * cond[i].view(batch_size, 1, 1, 1)

        return result

