import os
import sys
sys.path.insert(0,'..') # so that we have access to utils in the main directory of the repo
import utils
import random
import numpy as np
import argparse
import glob
import scipy.io as sio

def make_dataset_batches(data_file,save_file,patch_size,n_patches = 1):

    print(data_file)
    x, y = utils.mat2np(data_file)
    
    for n in range(n_patches):

        x_patched, y_patched = patch(x,y, patch_size = patch_size)

        x_patched = utils.preprocess_data_batch(x_patched)
        y_patched = utils.preprocess_data_batch(y_patched)

        x_patched = x_patched.astype(np.float32)
        y_patched = y_patched.astype(np.float32)

        x_patched = np.transpose(x_patched,(0,3,1,2,4))
        y_patched = np.transpose(y_patched,(0,3,1,2,4))

        patch_filename = save_file + '_' + str(n)
        mpdict = {'x':x_patched,'y':y_patched}
        np.save(patch_filename,mpdict)

def patch(x,y,patch_size):

    batch_size, height, width, n_channels, _ = x.shape

    max_height = height - patch_size
    max_width = width - patch_size

    x_patched = np.zeros((batch_size,patch_size,patch_size,n_channels,2))
    y_patched = np.zeros((batch_size,patch_size,patch_size,n_channels,2))
    for b in range(batch_size):
        h_start = random.randint(0,max_height)
        h_end = h_start + patch_size
        w_start = random.randint(0,max_width)
        w_end = w_start + patch_size
        x_patched[b] = x[b,h_start:h_end,w_start:w_end]
        y_patched[b] = y[b,h_start:h_end,w_start:w_end]

    return x_patched, y_patched

def arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',type= str, 
            default='./datasets/lesion0p5mm_val'
            help='folder containing the data to convert')
    p.add_argument('--patch_size',type= int, default = 200, help='size of patches')
    p.add_argument('--n_patches',type= int, default = 1, help='number of patches per sample')
    return p.parse_args()

if __name__ == '__main__':

    args = arg_parser()

    for data_file in list(glob.glob(os.path.join(args.data_dir,'mat_files','*.mat'))): 
        file_prefix = os.path.basename(data_file)[:-4]
        save_folder = os.path.join(args.data_dir,'npy') 
        if not os.path.exists(save_folder): os.mkdir(save_folder)
        save_file = os.path.join(save_folder,file_prefix)
        make_dataset_batches(data_file,save_file,args.patch_size,args.n_patches)

