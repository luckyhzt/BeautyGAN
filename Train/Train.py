import os
import sys
thisDir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(thisDir, os.pardir)))

import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import threading
import pickle
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

import Models.Discriminator as D
import Models.Generator as G
import Models.Regressor as R
import Models.Loss as L
import Models.Feature_extractor as F
from Models import Ops
from Dataset import FBP_dataset_V2, Label_Sampler
import Utils
from Config import load_config



class Trainer:

    def __init__(self, train_loader_d, train_loader_u, train_loader_c, test_loader, label_sampler):
        # Model
        self.regressor = R.Regressor(config, writer)
        self.generator = G.Generator(config, writer)
        self.discriminator = D.PatchDiscriminator(config, writer)
        self.feature = F.Feature_extractor(config)
        # Data
        self.train_loader_d = train_loader_d
        self.train_loader_u = train_loader_u
        self.train_loader_c = train_loader_c
        self.test_loader = test_loader
        self.label_sampler = label_sampler
        # Use CUDA
        if config['use_gpu'] and torch.cuda.is_available():
            self.use_cuda = True
            # Put model in GPU
            self.generator.cuda()
            self.regressor.cuda()
            self.discriminator.cuda()
            self.feature.cuda()
        else:
            self.use_cuda = False

        # Setup pre-trained model
        self.regressor.load(config['regressor_path'])
        self.regressor.eval()
        self.feature.eval()

        # Optimizer and learning rate scheduler
        self.optim_G = torch.optim.Adam(self.generator.parameters(), lr=config['lr'], betas=(0.5,0.99))
        self.optim_D = torch.optim.Adam(self.discriminator.parameters(), lr=config['lr'], betas=(0.5,0.99))
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(self.optim_G, step_size=config['lr_decay_epoch'], gamma=config['lr_decay'])
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(self.optim_D, step_size=config['lr_decay_epoch'], gamma=config['lr_decay'])


    def train(self):
        # Set network to train mode
        self.generator.train()
        self.discriminator.train()

        # Start Training
        step = 0
        run_vars = Utils.Running_vars()

        print('\nStart training...\n')
        for e in range(config['max_epoch']):
            steps_per_epoch = min(len(self.train_loader_d), len(self.train_loader_u), len(self.train_loader_c))
            if e % (len(self.train_loader_d) / steps_per_epoch) == 0:
                train_iter_d = iter(self.train_loader_d)
            if e % (len(self.train_loader_u) / steps_per_epoch) == 0:
                train_iter_u = iter(self.train_loader_u)
            if e % (len(self.train_loader_c) / steps_per_epoch) == 0:
                train_iter_c = iter(self.train_loader_c)

            for _ in range(steps_per_epoch):
                start = time.time()
                step += 1

                # Get data
                x_d, _ = train_iter_d.next()
                x_u, _ = train_iter_u.next()
                y_g = self.label_sampler.sample(x_u.size(0))
                x_c, y_c = train_iter_c.next()

                # Put in GPU
                if self.use_cuda:
                    x_d = x_d.cuda()
                    x_u = x_u.cuda()
                    y_g = y_g.cuda()
                    x_c = x_c.cuda(); y_c = y_c.cuda()
                
                #========== Train ==========
                x_g = self.generator(x_u, y_g)

                # Train Discriminator
                d_loss_d, d_loss_g, d_loss = self.train_discriminator(x_d, x_g)
                # Train Generator
                g_loss_d, g_loss_id, g_loss_r, g_loss = self.train_generator(x_u, x_g, y_g)
                # Train Generator with Cycle loss
                g_loss_consist = self.train_generator_consist(x_c, y_c)

                elapsed = time.time() - start
                run_vars.add([d_loss_d, d_loss_g, g_loss_d, g_loss_id, g_loss_r, g_loss_consist, elapsed])
                
                # Print log
                if step % config['log_step'] == 0:
                    var = run_vars.run_mean()
                    run_vars.clear()
                    for i in range(6):
                        var[i] = torch.sqrt(var[i])
                    print('epoch {} step {},   d_real: {:.4f}   d_g: {:.4f}-{:.4f}-id: {:.4f}   beauty: {:.4f}   consist: {:.4f} --- {:.2f} samples/sec' 
                        .format(e, step, var[0], var[1], var[2], var[3], var[4], var[5], config['batch_size']/var[6] ))
                    # Save result
                    writer.add_scalar('GAN/real', var[0], step)
                    writer.add_scalars('GAN/gen_vs_disc', {'d': var[1], 'g': var[2]}, step)
                    writer.add_scalars('GAN/identity_loss', {'train': var[3]}, step)
                    writer.add_scalars('GAN/beauty_loss', {'train': var[4]}, step)
                    writer.add_scalar('GAN/consist_loss', var[5], step)
                
                # Save generated images
                if step % config['eval_step'] == 0:
                    # Eval generator
                    gen_imgs = self.eval_generator(x_u[0:1,:,:,:])
                    image_grid = torchvision.utils.make_grid(gen_imgs / 2.0 + 0.5, padding=15, nrow=config['num_visualize_images'])
                    writer.add_image('generated_images', image_grid, step)
                    # Eval identity and beauty loss
                    identity_loss, beauty_loss = self.eval_loss()
                    writer.add_scalars('GAN/identity_loss', {'test': identity_loss}, step)
                    writer.add_scalars('GAN/beauty_loss', {'test': beauty_loss}, step)
                    # Print memory info
                    print('\nMemory allocated:', torch.cuda.max_memory_cached(0), '\n')
                
            if e+1 >= config['cpt_save_min_epoch'] and (e+1 - config['cpt_save_min_epoch']) % config['cpt_save_epoch'] == 0:
                # Save generator
                model_name = 'generator_' + str(e+1) + '.pth'
                self.generator.save( os.path.join(config['result_path'], model_name) )
        
            self.scheduler_D.step()
            self.scheduler_G.step()


    def train_discriminator(self, x_d, x_g):
        self.discriminator.train()

        self.optim_D.zero_grad()
        # Output
        d_output_d = self.discriminator(x_d)               # Real image
        d_output_g = self.discriminator(x_g.detach())      # Fake image
        # Loss
        d_loss_d = L.adversarial_loss(d_output_d, True)
        d_loss_g = L.adversarial_loss(d_output_g, False)
        d_loss = config['alpha_d']*d_loss_d + config['alpha_g']*d_loss_g
        # Back propagation
        d_loss.backward()
        self.optim_D.step()

        return d_loss_d, d_loss_g, d_loss

    
    def train_generator(self, x_u, x_g, y_g):
        self.generator.train()
        self.discriminator.eval()

        self.optim_G.zero_grad()
        # Forward
        g_output_d = self.discriminator(x_g)
        g_feat = self.feature(x_g, config['identity_layer'])[0]
        u_feat = self.feature(x_u, config['identity_layer'])[0]
        g_output_r = self.regressor(x_g)
        # Loss
        g_loss_d = L.adversarial_loss(g_output_d, True)
        g_loss_feat = L.MSELoss(g_feat, u_feat)
        g_loss_r = L.MSELoss(g_output_r, y_g)
        g_loss = config['alpha_g']*g_loss_d + config['identity_loss_weight']*g_loss_feat + config['beauty_loss_weight']*g_loss_r
        # Backward
        g_loss.backward()
        self.optim_G.step()

        self.discriminator.train()

        return g_loss_d, g_loss_feat, g_loss_r, g_loss

    
    def train_generator_consist(self, x_c, y_c):
        self.generator.train()

        self.optim_G.zero_grad()
        # Forward
        x_g = self.generator(x_c, y_c)
        g_feat = self.feature(x_g, config['consist_layer'])[0]
        c_feat = self.feature(x_c, config['consist_layer'])[0]
        # Loss
        g_loss_consist = L.MSELoss(g_feat, c_feat)
        g_loss = config['consist_loss_weight'] * g_loss_consist
        # Backward
        g_loss.backward()
        self.optim_G.step()

        return g_loss_consist


    def eval_generator(self, x_g):
        self.generator.eval()
        # Forward one image into generator for n times with different label
        test_iter = iter(self.test_loader)
        x_t, _ = test_iter.next()
        x_t = x_t.cuda()
        x_g = torch.cat([x_g, x_t], dim=0)

        num_images = x_g.size(0)
        l = np.array([[1.5], [2.5], [3.5], [4.5]], dtype=np.float32)

        label = torch.from_numpy(l)
        if self.use_cuda:
            label = label.cuda()
        
        for i in range(label.size(0)):
            this_label = label[i:i+1, :]
            this_label = this_label.expand(num_images, -1)
            gen_imgs = self.generator(x_g, this_label).detach_()
            if i == 0:
                result = torch.cat([x_g, gen_imgs], dim=0)
            else:
                result = torch.cat([result, gen_imgs], dim=0)

        self.generator.train()

        return result


    def eval_loss(self):
        self.generator.eval()

        test_iter = iter(self.test_loader)
        identity_loss = 0
        beauty_loss = 0

        for i in range(len(self.test_loader)):
            x, _ = test_iter.next()
            x = x.cuda()
            y_g = self.label_sampler.sample(x.size(0)).cuda()
            x_g = self.generator(x, y_g).detach_()
            # Calculate identity loss
            u_feat = self.feature(x, config['identity_layer'])[0].detach_()
            g_feat = self.feature(x_g, config['identity_layer'])[0].detach_()
            identity_loss += L.MSELoss(g_feat, u_feat) * x.size(0)
            # Calculate beauty loss
            y_g_g = self.regressor(x_g).detach_()
            beauty_loss += L.MSELoss(y_g_g, y_g) * x.size(0)

        identity_loss = identity_loss / len(self.test_loader.dataset)
        beauty_loss = beauty_loss / len(self.test_loader.dataset)
        
        self.generator.train()

        return torch.sqrt(identity_loss), torch.sqrt(beauty_loss)

        


def main():
    # Create dataset
    trainset = FBP_dataset_V2('train', config['SCUT-FBP-V2'], config['train_index'], config['img_size'], config['crop_size'])
    testset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['test_index'], config['img_size'], config['crop_size'])

    # Get dataset loader
    train_loader_c = Data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    train_loader_d = Data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    train_loader_u = Data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=config['num_visualize_images']-1, shuffle=True)
    # Label sampler for Generator
    sampler = Label_Sampler(config['SCUT-FBP-V2'], config['train_index'])

    # Initialize trainer
    trainer = Trainer(train_loader_d, train_loader_u, train_loader_c, test_loader, sampler)

    # Lauch Tensorboard
    t = threading.Thread(target=Utils.launchTensorBoard, args=([config['result_path']]))
    t.start()

    # Start training
    trainer.train()

    writer.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='D:/ThesisData', type=str)
    args = parser.parse_args()

    # Load config and tensorboard writer
    config, writer = load_config(args.path)

    # Start training
    main()

