# Useful function or class

import os
import numpy as np


class Running_vars:
    def __init__(self):
        self.clear()
    

    def add(self, var):
        self.step += 1

        if self.vars == None:
            self.vars = var
        else:
            for i in range(len(self.vars)):
                self.vars[i] += var[i]


    def run_mean(self):
        for i in range(len(self.vars)):
                self.vars[i] /= self.step
        
        return self.vars

    
    def clear(self):
        self.vars = None
        self.step = 0


def launchTensorBoard(logdir):
    os.system('tensorboard --logdir=' + logdir)
    return


def loss_and_corr(pred, target):
        RMSE = np.sqrt(np.square(pred - target).mean())
        MAE = np.absolute(pred - target).mean()
        PC = np.corrcoef(pred, target, rowvar=False)[1, 0]
        return RMSE, MAE, PC


def average_score(label):
    avg = np.sum(label, axis=1, keepdims=True)
    num = np.count_nonzero(label, axis=1).reshape([-1, 1])
    avg = avg / num

    return avg


def label_distribution(label):
    label = label.astype(int)
    images = label.shape[0]
    dist = np.zeros((images, 5))

    for i in range(0, images):
        occur = np.bincount(label[i, :].reshape((-1)), minlength=6)
        occur = occur[1:6]
        dist[i, :] = occur / np.sum(occur)

    return dist


def label_level_balanced(label, num_level):
    ordered = np.sort(label, axis=0)

    gap = label.shape[0] / num_level
    threshold = np.zeros(num_level+1)
    threshold[-1] = 5.0
    for i in range(1, num_level):
        threshold[i] = ordered[int(i*gap), 0]

    level = []
    
    for i in range(label.shape[0]):
        for j in range(threshold.shape[0]):
            if label[i] >= threshold[j]:
                this_level = j
            else: break
        level.append(this_level)
    
    level = np.array(level)
    level = one_hot(level, num_level)
    return level


def label_level(label, num_level):
    level_gap = 4.0 / num_level
    level = np.floor( (label - 1) / level_gap)
    level = one_hot(level, num_class=int(num_level))

    return level


def one_hot(label, num_class):
    label = label.astype(int)
    n = label.shape[0]
    onehot = np.zeros((n, num_class), np.float32)
    
    for i in range(n):
        onehot[i, label[i]] = 1.0

    return onehot
