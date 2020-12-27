import torch
from torch.utils import data
import numpy as np 
import utils

# Defines a data class to load the npy files generated from make_npy.py to use for pytorch
class NumpyDataset(data.Dataset):

    def __init__(self, npy_list):

        # make sure the files exist
        utils.test_filenames(npy_list)

        # store file list
        self.data_numpy_list = npy_list

    def __getitem__(self, index):

        self.x_list = []
        self.y_list = []
        for ind in index:
            data_slice_file_name = self.data_numpy_list[ind]
            data_i = np.load(data_slice_file_name,allow_pickle = True)
            x_i = data_i.item().get('x')
            y_i = data_i.item().get('y')

            self.x_list.append(x_i[0,:,:,:,:])   
            self.y_list.append(y_i[0,:,:,:,:])   

            x_data = np.asarray(self.x_list)
            y_data = np.asarray(self.y_list)

            self.data = {'x':torch.from_numpy(x_data).float(),
                    'y':torch.from_numpy(y_data).float()}

        return self.data

    def __len__(self):
        return len(self.data_numpy_list)
