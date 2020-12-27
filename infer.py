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

    # set up model and optimization
    model = Network()
    criterion = nn.L1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), 
            lr= a['lr'], weight_decay = a['reg1'])

    # define datasets
    ds_path = '/home/llbricks/datasets'

#    # 0.5mm anechoic lesion validation set 
#    for sim in range(1,4):
#        f2_v = ['./datasets/lesion0p5mm_val/sim{}_{}dB_0.npy'.format(sim,n) for n in range(-15,11)]

    # delete this part later
    f2_v = []
    if(0): # patch size = 200
        ds_fdr = 'verasonics/20201221_compare_time/npy'
        for l in range(1,8):
            f2_v = f2_v + ['{}/{}/pair{}_{}_0.npy'.format(ds_path,
                ds_fdr,l,i) for i in range(1,3)]

    # initialize dataset loader
    data_val = NumpyDataset(f2_v)

    # get useful variables 
    val_bs = 1
    a['n_val'] = len(f2_v)
    n_batches_v = math.floor(a['n_val']/val_bs)

    # put it on the gpu 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # check if there's a checkpoint to load
    if a['ckpt'] is not '':
        model.load_state_dict(torch.load(a['ckpt'],map_location=torch.device('cuda')))

    # train model
    a = utils.make_save_dir(a)
    writer = SummaryWriter(os.path.join(a['model_path'],'logs'))
    utils.save_model_settings(a,copytrain=0)

    model.eval()
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
            
            # log tensorboard values 
            lognum = n_batches_v + batch

            utils.write_imgs(writer,'val',x_batch,
                    y_batch,pred,lognum,a)
            mat_save_fn = os.path.join(a['model_path'],'output_{}.mat'.format(batch))
            mat_save_dict = {'pred':pred.detach().cpu().numpy(),
                                'x':x_batch.detach().cpu().numpy(),
                                'y':y_batch.detach().cpu().numpy()}
            sio.savemat(mat_save_fn,mat_save_dict)

        # save checkpoint
        save_name = os.path.join(a['model_path'],'infer.pth')
        torch.save(model.state_dict(), save_name)

    print('Finished')

if __name__ == '__main__':

    a = config.parser()
    main(a)



