import numpy as np
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from datetime import datetime
import os
import argparse
import matplotlib.pyplot as plt
import threading
import pickle
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter

import models.Discriminator as D
import models.Generator as G
import models.Regressor as R
import models.Loss as L
import models.Feature_extractor as F
from models import ops
from dataset import FBP_dataset, FBP_dataset_V2, AFAD, Label_Sampler
import utils



class Trainer:

    def __init__(self, train_loader_d, train_loader_u, train_loader_c, test_loader, label_sampler):
        # Model
        self.regressor = R.Regressor(config, writer)
        self.generator = G.Generator(config, writer)
        self.discriminator = D.PatchDiscriminator(config, writer)
        self.feature = F.Feature_extractor(config['feature_model'], config['pretrained_path'])
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
        run_vars = utils.Running_vars()

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
                x_c, y_c = train_iter_c.next()
                y_g = self.label_sampler.sample(x_u.size(0))
                # Put in GPU
                if self.use_cuda:
                    x_d = x_d.cuda()
                    x_u = x_u.cuda(); y_g = y_g.cuda()
                    x_c = x_c.cuda(); y_c = y_c.cuda()

                #========== Train ==========
                x_g = self.generator(x_u, y_g)

                # Train Discriminator
                d_loss_d, d_loss_g, d_loss = self.train_discriminator(x_d, x_g)
                # Train Generator
                g_loss_d, g_loss_feat, g_loss_r, g_loss = self.train_generator(x_u, x_g, y_g)
                # Train Generator with Cycle loss
                g_loss_cycle = self.train_generator_cycle(x_c, y_c)

                elapsed = time.time() - start

                run_vars.add([d_loss_d, d_loss_g, g_loss_d, g_loss_feat, g_loss_r, g_loss_cycle, elapsed])
                
                # Print log
                if step % config['log_step'] == 0:
                    var = run_vars.run_mean()
                    run_vars.clear()
                    for i in range(6):
                        var[i] = torch.sqrt(var[i])
                    print('epoch {} step {},   d_real: {:.4f}   d_g: {:.4f}-{:.4f}-f: {:.4f}   g_r: {:.4f}   g_cyc: {:.4f} --- {:.2f} samples/sec' 
                        .format(e, step, var[0], var[1], var[2], var[3], var[4], var[5], config['batch_size']/var[6] ))
                    # Save result
                    writer.add_scalar('GAN/real', var[0], step)
                    writer.add_scalars('GAN/gen_vs_disc', {'d': var[1], 'g': var[2]}, step)
                    writer.add_scalar('GAN/feature_loss', var[3], step)
                    writer.add_scalar('GAN/reg_loss', var[4], step)
                    writer.add_scalar('GAN/cycle_loss', var[5], step)
                
                # Save generated images
                if step % config['image_save_step'] == 0:
                    # Eval generator
                    gen_imgs = self.eval_generator(x_u[0:1,:,:,:])
                    image_grid = torchvision.utils.make_grid(gen_imgs / 2.0 + 0.5, padding=15, nrow=config['num_visualize_images'])
                    writer.add_image('generated_images', image_grid, step)
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
        g_feat = self.feature(x_g, config['feature_layer'])
        u_feat = self.feature(x_u, config['feature_layer'])
        g_output_r = self.regressor(x_g)
        # Loss
        g_loss_d = L.adversarial_loss(g_output_d, True)
        g_loss_feat = L.MSELoss(g_feat, u_feat)
        g_loss_r = L.MSELoss(g_output_r, y_g)
        g_loss = config['alpha_g']*g_loss_d + config['feature_loss_weight']*g_loss_feat + config['reg_loss_weight']*g_loss_r
        # Backward
        g_loss.backward()
        self.optim_G.step()

        self.discriminator.train()

        return g_loss_d, g_loss_feat, g_loss_r, g_loss

    
    def train_generator_cycle(self, x_d, y_d):
        self.generator.train()
        self.discriminator.eval()

        self.optim_G.zero_grad()
        # Forward
        x_g = self.generator(x_d, y_d)
        g_cycle = self.feature(x_g, config['cycle_layer'])
        d_cycle = self.feature(x_d, config['cycle_layer'])
        # Loss
        g_loss_cycle = L.MSELoss(g_cycle, d_cycle)
        g_loss = config['cycle_g_loss'] * g_loss_cycle
        # Backward
        g_loss.backward()
        self.optim_G.step()

        self.discriminator.train()

        return g_loss_cycle


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



def load_config(root_dir):
    config = OrderedDict()

    config['train'] = 'Generator'
    config['use_gpu'] = True

    # Hyper-parameters
    config['batch_size'] = 16
    config['max_epoch'] = 150
    config['lr'] = 1e-4
    config['lr_decay'] = 0.3
    config['lr_decay_epoch'] = 50
    config['alpha_g'] = 0.8    # weights of generator training against discriminator
    config['alpha_d'] = 0.8    # weights of discriminator training with real samples
    config['feature_loss_weight'] = 0.5
    config['reg_loss_weight'] = 1.5
    config['cycle_g_loss'] = 0.05
    # Log
    config['log_step'] = 10
    config['image_save_step'] = 100
    config['num_visualize_images'] = 5   # num of generated images to be saved
    config['cpt_save_epoch'] = 10
    config['cpt_save_min_epoch'] = 100

    # Pretrained feature extractor and regressor
    feature_networks = ['resnet50_ft_dag', 'senet50_256', 'vgg_face_dag', 'vgg_m_face_bn_dag']
    config['pretrained_path'] = os.path.join(root_dir, 'Pretrained_models')
    config['feature_model'] = feature_networks[3]
    config['feature_layer'] = 5    # 1 to 5, define which output layer of pretrained model to be used as feature
    config['cycle_layer'] = 2    # 1 to 5, define which output layer of pretrained model to be used as feature
    config['regressor_path'] = os.path.join(config['pretrained_path'], 'pretrained_regressor_V2.pth')

    # Dataset
    config['SCUT-FBP-V2'] = os.path.join(root_dir, 'SCUT-FBP5500_v2')
    config['AFAD'] = os.path.join(root_dir, 'AFAD_Lite')
    config['img_size'] = 236
    config['crop_size'] = 224
    #config['c_samples'] = 2000
    #config['d_samples'] = 4000
    #config['u_samples'] = 20000
    #config['train_samples'] = 1800
    #config['test_samples'] = 200
    config['sample_range'] = [1.0, 5.0]
    index_file = os.path.join(config['SCUT-FBP-V2'], 'data_index_1800_200.pkl')
    with open(index_file, 'rb') as readFile:
        data_index = pickle.load(readFile)
    config['train_index'] = data_index['train']
    config['test_index'] = data_index['test']
    # Train and test set
    #config['c_index'] = np.arange(config['c_samples'])
    #female_index = np.load( os.path.join(config['AFAD'], 'female_index.npy') )
    #config['d_index'], config['u_index'], config['test_index'], _ = \
    #    np.split(female_index, [config['d_samples'], config['d_samples']+config['u_samples'], \
    #        config['d_samples']+config['u_samples']+config['test_samples']])

    # Create directory to save result
    date_time = datetime.now()
    time_str = date_time.strftime("%Y%m%d_%H-%M-%S")
    result_path = os.path.join(root_dir, 'Result', config['train'])
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    result_path = os.path.join(result_path, time_str)
    os.mkdir(result_path)
    config['result_path'] = result_path

    # Save config in pkl file
    pk_file = os.path.join(result_path, 'config.pkl')
    with open(pk_file, 'wb') as outFile:
        pickle.dump(config, outFile)

    print('Configuration loaded and saved.')


    # Create Tensorboard to save result
    writer = SummaryWriter(config['result_path'])
    # Write all configs to tensorboard
    content = ''
    for keys,values in config.items():
            content += keys + ': ' + str(values) + '  \n  \n'
    writer.add_text('config', content, 0)

    # Save config in txt file
    txt_file = os.path.join(result_path, 'config.txt')
    with open(txt_file, 'w') as outFile:
        outFile.write(content)

    return config, writer



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
    sampler = Label_Sampler(config['sample_range'])

    # Initialize trainer
    trainer = Trainer(train_loader_d, train_loader_u, train_loader_c, test_loader, sampler)

    # Lauch Tensorboard
    t = threading.Thread(target=utils.launchTensorBoard, args=([config['result_path']]))
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

