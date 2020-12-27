from torch import nn
from torch import optim
import torch
import numpy as np
import os
import scipy.io as sio
import random
import math
from tqdm import tqdm
import utils
import config
from data import NumpyDataset
import torchvision 
from torch.utils.tensorboard import SummaryWriter
from pretrained_net import Network

def main(a):

    # set up model, loss function and optimizer
    model = Network(load_pretrain = 0)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), 
            lr= a['lr'], weight_decay = a['reg1'])

    # define datasets
    ds_path = '/data/llbricks/datasets'
    f2_2 = [] # this is a list of files which contain the training data, made by make_npy.py
    f2_v = []

    # regular, no anechoic regions
    if(0):
        ds_fdr = 'field2/20200103_orig/npy'
        for i in range(1,a['ar']): 
            f2_2 = f2_2 + ['{}/{}/batch_{}_{}.npy'.format(ds_path,ds_fdr,l,i) for l in range(1,480)]
        for i in range(0,5):
            f2_v = f2_v + ['{}/{}/batch_{}_{}.npy'.format(ds_path, ds_fdr,l,i) for l in range(480,501)]

    # some anechoic regions
    if(1):
        ds_fdr = 'field2/20200103_orig_anechoic/npy' 
        for i in range(1,5): 
            f2_2 = f2_2 + ['{}/{}/batch_{}_{}.npy'.format(ds_path,ds_fdr,l,i) for l in range(501,980)]
            f2_v = f2_v + ['{}/{}/batch_{}_{}.npy'.format(ds_path,ds_fdr,l,i) for l in range(980,1001)]

    # many anechoic regions 
    if(1):
        ds_fdr = 'field2/20200501_anechoic/npy'
        for i in range(1,5): 
            f2_2 = f2_2 + ['{}/{}/batch_{}_{}.npy'.format(ds_path,ds_fdr,l,i) for l in range(1001,1480)]
            f2_v = f2_v + ['{}/{}/batch_{}_{}.npy'.format(ds_path,ds_fdr,l,i) for l in range(1480,1501)]

    # initialize dataset loader
    data_train = NumpyDataset(f2_2)
    data_val = NumpyDataset(f2_v)

    # get useful variables 
    val_bs = 4
    a['n_train'] = len(f2_2)
    a['n_val'] = len(f2_v)
    idx = list(range(a['n_train']))
    n_batches_t = math.floor(a['n_train']/a['bs'])
    n_batches_v = math.floor(a['n_val']/val_bs)

    # put it on the gpu 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # check if there's a checkpoint to load
    print(a['ckpt'])
    if a['ckpt'] is not '':
        model.load_state_dict(torch.load(a['ckpt'],
            map_location=torch.device('cuda'))) 

    # train model
    a = utils.make_save_dir(a)
    writer = SummaryWriter(os.path.join(a['model_path'],'logs'))
    utils.save_model_settings(a)

    for epoch in range(a['n_epochs']+1):

        # TRAIN ------------------------
        model.eval()
        random.shuffle(idx)
        loss_t = 0.0
        for batch in range(n_batches_t):

            # get batch
            b_idx = idx[(batch*a['bs']):((batch+1)*a['bs'])]
            batch_data = data_train[b_idx]
            x_batch = torch.Tensor(batch_data['x']).to(device)
            y_batch = torch.Tensor(batch_data['y']).to(device)

            # optimize
            model.train()
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(y_batch,pred)
            loss_t += loss.item()
            loss.backward()
            optimizer.step()
            
            # log tensorboard values 
            lognum = epoch*n_batches_t + batch
            if batch%a['tb_img'] == 0: 
                utils.write_imgs(writer,'train',x_batch,
                        y_batch,pred,lognum,a)

            if batch%a['tb_loss'] == 0: 
                writer.add_scalar('loss_t',loss,lognum)
                utils.write_bm_loss(writer,'bmode_loss_t',
                        y_batch,pred,lognum,a)
                utils.write_phase_loss(writer,'phase_loss_t',
                        y_batch,pred,lognum,a)

        # VAL ------------------------
        model.eval()
        loss_v = 0.0
        with torch.no_grad():
            for batch in range(n_batches_v):

                # get batch
                b_idx = range(batch*val_bs,(batch+1)*val_bs)
                batch_data = data_val[b_idx]
                x_batch = torch.Tensor(batch_data['x']).to(device)
                y_batch = torch.Tensor(batch_data['y']).to(device)

                # compute loss 
                model.eval()
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(y_batch,pred)
                loss_v += loss.item()
                
                # log tensorboard values 
                lognum = epoch*n_batches_t + batch

                utils.write_imgs(writer,'val',x_batch,
                        y_batch,pred,lognum,a)

                writer.add_scalar('loss_v',loss,lognum)
                utils.write_bm_loss(writer,'bmode_loss_v',
                        y_batch, pred,lognum,a)
                utils.write_phase_loss(writer,'phase_loss_v',
                            y_batch,pred,lognum,a)

        # END EPOCH -----------------------_
        print('train loss: {:.3f}'.format(loss_t/n_batches_t),
                'val loss: {:.3f}'.format(loss_v/n_batches_v))

        # save checkpoint
        if epoch%a['ckpt_interval']== 0:
            save_name = os.path.join(a['model_path'],'epoch_{}.pth'.format(epoch))
            torch.save(model.state_dict(), save_name)

    print('Finished')

if __name__ == '__main__':

    a = config.parser()
    a['run_name'] = 'retrain_oldnet'
    a['n_epochs'] = 40
    a['ar'] = 5
    main(a)


