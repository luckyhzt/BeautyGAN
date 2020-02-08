import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from datetime import datetime
import os
import matplotlib.pyplot as plt
import threading

import models.Discriminator as D
import models.Generator as G
import models.Regressor as R
import models.Loss as L
import models.Feature_extractor as F
from models import ops
from dataset import FBP_dataset, FBP_dataset_V2, Label_Sampler

from config import config, writer



class Trainer:

    def __init__(self, train_loader, test_loader, unlabel_loader, label_sampler):
        # Model
        self.regressor = R.Regressor()
        self.generator = G.Generator()
        # Data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.unlabel_loader = unlabel_loader
        self.label_sampler = label_sampler
        # Use CUDA
        if config['use_gpu'] and torch.cuda.is_available():
            self.use_cuda = True
            # Put model in GPU
            self.generator.cuda()
            self.regressor.cuda()
        else:
            self.use_cuda = False
        
        # Setup pre-trained model
        self.generator.load(config['generator_path'])
        self.generator.eval()

        # Optimizer and learning rate scheduler
        self.optim_R = torch.optim.Adam(self.regressor.parameters(), lr=config['lr'], betas=(0.5,0.99))
        self.scheduler_R = torch.optim.lr_scheduler.StepLR(self.optim_R, step_size=config['lr_decay_epoch'], gamma=config['lr_decay'])


    def train(self):
        # Set network to train mode
        self.regressor.train()

        # Start Training
        step = 0
        run_vars = Running_vars()

        print('\nStart training...\n')
        for e in range(config['max_epoch']):
            steps_per_epoch = min(len(self.train_loader), len(self.unlabel_loader))
            if e % (len(self.train_loader) / steps_per_epoch) == 0:
                train_iter = iter(self.train_loader)
            if e % (len(self.unlabel_loader) / steps_per_epoch) == 0:
                unlabel_iter = iter(self.unlabel_loader)

            for _ in range(steps_per_epoch):
                start = time.time()
                step += 1
                # Get data
                x, t = train_iter.next()
                x_u = unlabel_iter.next()
                t_g = self.label_sampler.sample()
                # Put in GPU
                if self.use_cuda:
                    x = x.cuda(); t = t.cuda()
                    x_u = x_u.cuda(); t_g = t_g.cuda()

                #========== Train ==========
                x_g = self.generator(x_u, t_g).detach_()
                
                # Train regressor
                r_loss_r, r_loss_g, r_loss = self.train_regressor(x, t, x_g, t_g)

                elapsed = time.time() - start

                run_vars.add([r_loss_r, r_loss_g, elapsed])
                
                # Print log
                if step % config['log_step'] == 0:
                    var = run_vars.run_mean()
                    run_vars.clear()
                    for i in range(2):
                        var[i] = torch.sqrt(var[i])
                    print('epoch {} step {},   r: {:.4f}, g: {:.4f} --- {} samples/sec' 
                        .format(e, step, var[0], var[1], int(config['batch_size']/var[2]) ))
                    # Save result
                    writer.add_scalars('Train', {'r': var[0], 'g': var[1]}, step)
                
                # Eval regressor
                if step % config['test_step'] == 0:
                    RMSE, MAE, PC = self.eval_regressor()
                    writer.add_scalars('Test/RMSE', {'train': var[0], 'test': RMSE}, step)
                    writer.add_scalar('Test/MAE', MAE, step)
                    writer.add_scalar('Test/PC', PC, step)
                    # Print memory info
                    print('\nMemory allocated:', torch.cuda.max_memory_cached(0), '\n')
        
            self.scheduler_R.step()

    
    def train_regressor(self, x, t, x_g, t_g):
        self.regressor.train()

        self.optim_R.zero_grad()
        x = torch.cat([x, x_g], dim=0)
        t = torch.cat([t, t_g], dim=0)
        # Forward
        y = self.regressor(x)
        # Loss
        r_loss_r = L.MSELoss(y, t)
        r_loss_g = torch.tensor(0.0)
        r_loss = r_loss_r
        # Back propagate
        r_loss.backward()
        self.optim_R.step()

        return r_loss_r, r_loss_g, r_loss


    def eval_regressor(self):
        # Set regressor to eval mode
        self.regressor.eval()

        dataiter = iter(self.test_loader)

        pred = []
        target = []

        for i in range(len(self.test_loader)):
            x, t = dataiter.next()
            if self.use_cuda:
                x = x.cuda()
            
            y = self.regressor(x).detach_()
            y = y.data.cpu().numpy()
            t = t.data.numpy()

            pred.append(y)
            target.append(t)

        pred = np.concatenate(pred, axis=0)
        target = np.concatenate(target, axis=0)

        RMSE, MAE, PC = self.loss_and_corr(pred, target)

        print('\nEvaluation, RMSE: {:.4f}, MAE: {:.4f}, PC: {:.4f}\n'.format(RMSE, MAE, PC))

        # Set regressor back to train mode
        self.regressor.train()

        return RMSE, MAE, PC


    def loss_and_corr(self, pred, target):
        RMSE = np.sqrt(np.square(pred - target).mean())
        MAE = np.absolute(pred - target).mean()
        PC = np.corrcoef(pred, target, rowvar=False)[1, 0]
        return RMSE, MAE, PC



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



def main():
    # Create dataset
    trainset = FBP_dataset('train', config['limg_path'], config['trainset_index'], config['label_path'])
    testset = FBP_dataset('test', config['limg_path'], config['testset_index'], config['label_path'])
    unlabelset = FBP_dataset_V2(config['uimg_path'], config['unlabel_index'])

    # Get dataset loader
    train_loader = Data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False)
    unlabel_loader = Data.DataLoader(unlabelset, batch_size=int(config['batch_size']/2), shuffle=True)
    # Label sampler for Generator
    sampler = Label_Sampler(config['label_path'], int(config['batch_size']/2))

    # Initialize trainer
    trainer = Trainer(train_loader, test_loader, unlabel_loader, sampler)

    # Lauch Tensorboard
    t = threading.Thread(target=launchTensorBoard, args=([config['result_path']]))
    t.start()

    # Start training
    trainer.train()

    writer.close()


if __name__ == '__main__':
   main()









