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

import Models.Regressor as R
import Models.Loss as L
from Models import Ops
from Dataset import FBP_dataset, FBP_dataset_V2
import Utils



class Trainer:

    def __init__(self, train_loader, test_loader):
        # Model
        self.regressor = R.Regressor(config, writer)
        # Data
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Use CUDA
        if config['use_gpu'] and torch.cuda.is_available():
            self.use_cuda = True
            # Put model in GPU
            self.regressor.cuda()
        else:
            self.use_cuda = False

        # Optimizer and learning rate scheduler
        self.optim_R = torch.optim.Adam(self.regressor.parameters(), lr=config['lr'], betas=(0.5,0.99))
        self.scheduler_R = torch.optim.lr_scheduler.StepLR(self.optim_R, step_size=config['lr_decay_epoch'], gamma=config['lr_decay'])


    def train(self):
        # Set network to train mode
        self.regressor.train()

        # Start Training
        step = 0
        run_vars = Utils.Running_vars()

        print('\nStart training...\n')
        for e in range(config['max_epoch']):
            train_iter = iter(self.train_loader)

            for _ in range(len(self.train_loader)):
                start = time.time()
                step += 1
                # Get data
                x, t = train_iter.next()
                # Put in GPU
                if self.use_cuda:
                    x = x.cuda(); t = t.cuda()

                #========== Train ==========
                self.optim_R.zero_grad()
                # Forward
                y = self.regressor(x)
                # Loss
                loss = L.MSELoss(y, t)
                # Backward
                loss.backward()
                self.optim_R.step()

                elapsed = time.time() - start

                run_vars.add([loss, elapsed])
                
                # Print train log
                if step % config['log_step'] == 0:
                    # Print train loss
                    var = run_vars.run_mean()
                    run_vars.clear()
                    var[0] = torch.sqrt(var[0])
                    print('epoch {} step {}, train_RMSE: {:.4f} --- {:.2f} samples/sec' 
                        .format(e, step, var[0], config['batch_size']/var[1] ))

                if step % config['test_step'] == 0:
                    test_RMSE, test_MAE, test_PC = self.eval_regressor()
                    # Save result
                    writer.add_scalars('Regression/RMSE', {'train': var[0], 'test': test_RMSE}, step)
                    writer.add_scalar('Regression/MAE', test_MAE, step)
                    writer.add_scalar('Regression/PC', test_PC, step)
                
            if e+1 >= config['cpt_save_min_epoch'] and (e+1 - config['cpt_save_min_epoch']) % config['cpt_save_epoch'] == 0:
                # Save generator
                model_name = 'regressor_' + str(e+1) + '.pth'
                self.regressor.save( os.path.join(config['result_path'], model_name) )
        
            self.scheduler_R.step()
    

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

        RMSE, MAE, PC = Utils.loss_and_corr(pred, target)

        print('\nEvaluation, RMSE: {:.4f}, MAE: {:.4f}, PC: {:.4f}\n'.format(RMSE, MAE, PC))

        # Set regressor back to train mode
        self.regressor.train()

        return RMSE, MAE, PC




def load_config(root_dir):
    config = OrderedDict()

    config['train'] = 'Regressor'
    config['use_gpu'] = True

    # Hyper-parameters
    config['batch_size'] = 32
    config['max_epoch'] = 80
    config['lr'] = 1e-4
    config['lr_decay'] = 0.1
    config['lr_decay_epoch'] = 50

    # Log
    config['log_step'] = 10
    config['test_step'] = 40
    config['cpt_save_epoch'] = 10
    config['cpt_save_min_epoch'] = 50

    # Dataset
    config['SCUT-FBP-V2'] = os.path.join(root_dir, 'SCUT-FBP5500_v2')
    config['img_size'] = 236
    config['crop_size'] = 224
    config['train_samples'] = 1800
    config['test_samples'] = 200
    # Train and test set
    index_file = os.path.join(config['SCUT-FBP-V2'], 'data_index_1800_200.pkl')
    with open(index_file, 'rb') as readFile:
        data_index = pickle.load(readFile)
    config['trainset_index'] = data_index['train']
    config['testset_index'] = data_index['test']

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
    trainset = FBP_dataset_V2('train', config['SCUT-FBP-V2'], config['trainset_index'], \
        config['img_size'], config['crop_size'])
    testset = FBP_dataset_V2('test', config['SCUT-FBP-V2'], config['testset_index'], \
        config['img_size'], config['crop_size'])

    # Get dataset loader
    train_loader = Data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True)
    test_loader = Data.DataLoader(testset, batch_size=config['batch_size'], shuffle=False)

    # Initialize trainer
    trainer = Trainer(train_loader, test_loader)

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









