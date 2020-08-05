import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import math


def MSELoss(pred, target):
    loss_fn = nn.MSELoss()
    return loss_fn(pred, target)


def CrossEntropy(pred, target):
    # pred: output after softmax layer
    # target: one-hot label
    e = 1e-6
    loss = - torch.log( torch.sum(pred * target, dim=1) + e )

    return torch.mean(loss)
    

def adversarial_loss(pred, real, loss_fn=nn.MSELoss()):
    # Binary Cross Entropy Loss:
    # l(x, y) = -( y*log(x) + (1-y)*log(1-x) )
    # l(x, 0) = -log(1-x)  => maximize: log(1-x) => minimize: x
    # l(x, 1) = -log(x)    => maximize: log(x)   => maximize: x

    if real:
        target = torch.ones(pred.size(), device=pred.device)
    else:
        target = torch.zeros(pred.size(), device=pred.device)

    return loss_fn(pred, target)
