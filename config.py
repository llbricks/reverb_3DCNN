"""This file contains the options used by the training procedure. All options must be passed via the command line."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import argparse


def parser():
    """Adds various options to ArgParse (library used to parse command line arguments).

    Returns:
        opt: An ArgParse dictionary containing keys (argument name) and values (argument value).
    """
    p = argparse.ArgumentParser()

    # CPU/GPU and logging settings.
    p.add_argument('--ckpt_interval', default=2, 
            help='Write TensorBoard summaries to disk every X epochs.')
    p.add_argument('--tb_img', default=10, type=int,
            help='Write bmode image TensorBaord summaries of the current batch every X batch.')
    p.add_argument('--dr', default=[-25, 35], 
            help='The dynamic range at which the bmode image will be displayed on tensorboard')
    p.add_argument('--tb_loss', default=5, 
            help='Write loss TensorBaord summary of the current batch every X batch.')

    # dest path
    p.add_argument('--model_dir', help='output data main directory',
	        default ='./logs') # currently saves it into the 'logs' folder inside this repo
    p.add_argument('--run_name', default='', 
            help='name of the run for documentation')

    # General optimization settings.
    p.add_argument('--bs', default=10, type=int, help='Batch size.')
    p.add_argument('--training', default=True, type=bool, 
            help='boolean for if the network is training or not')
    p.add_argument('--reg1', default=1E-3, type=float, 
            help='L1 reg scalar')
    p.add_argument('--do', default=0.1, type=float, 
            help='dropout value')
    p.add_argument('--lr', default=1E-4, type=float, 
            help='Initial learning rate.')
    p.add_argument('--n_epochs', default=10, type=int, 
            help='Max number of training epochs.')

    p.add_argument('--ckpt', default='', type=str,
            help='path to a .pth file to load a pretrained network before training')

    opt = vars(p.parse_args())
    opt['date'] = time.strftime("%Y%m%d")

    return opt
