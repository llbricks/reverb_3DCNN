from __future__ import division
import numpy as np
import os
from shutil import copyfile
import scipy.io as sio
import torch
import torchvision
import h5py 

def make_save_dir(a):

    # get the folder name for this experiment
    a['folder_name'] = '_'.join(filter(None,[a['date'],a['run_name']]))

    # check if this folder exists, if not make it 
    experiment_dir = os.path.join(a['model_dir'],a['folder_name'])
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # make the model name based on the model folders already in that directory
    if len(os.listdir(experiment_dir)) == 0:
        a['model_name'] = 'model_0'
    else:
        models_list = os.listdir(experiment_dir)
        nums = []
        for model in models_list:
            is_a_model = ('model_' in model)
            is_a_folder = (os.path.isdir(os.path.join(experiment_dir,model)))
            if is_a_model and is_a_folder:
                nums.append(int(model[6:]))
        a['model_name'] = 'model_{}'.format(max(nums)+1)
    a['model_path'] = os.path.join(experiment_dir,a['model_name'])
    os.makedirs(a['model_path'])
    print(a['model_path'])

    return a

def test_filenames(filenames):
    for filename in filenames:
        if not os.path.isfile(filename):
            raise Exception(' {} is in the dataset, and does not exist'.format(filename))
        
def save_model_settings(args,copytrain = 1):

    if copytrain:
        copyfile('./train.py', os.path.join(args['model_path'],'train.py'))
    np.save(os.path.join(args['model_path'],'args.npy'),args)
    f = open(os.path.join(args['model_path'],'args.txt'),'w')
    for key in args.keys():
        f.write(key + ' : ' + str(args[key]) +'\n')
    f.close()

def load_data(filename_list,ps):

    # establish shape of data
    data = np.load(filename_list[0],allow_pickle=True)
    x_data = data.item().get('x')
    sz = x_data.shape
    x = np.zeros((len(filename_list),sz[1],ps,ps,sz[4]))
    y = np.zeros((len(filename_list),sz[1],ps,ps,sz[4]))

    # load them and store them in npy array
    for i in range(len(filename_list)):
        data = np.load(filename_list[i],allow_pickle=True)
        x[i,:,:,:,:] = data.item().get('x')[:,:,:ps,:ps,:]
        y[i,:,:,:,:] = data.item().get('y')[:,:,:ps,:ps,:]

    return x,y

def mat2np(dataset_path): 
    ''' Assumes input is a single, complex valued image of size: 
    [n_batch,height, width,n_channels], according to how it's read in matlab''' 

    if dataset_path != '': 

        data = h5py.File(dataset_path,'r') 
        x = np.squeeze(data['x'].value.view(np.complex64)) 
        y = np.squeeze(data['y'].value.view(np.complex64)) 

        print('x shape after import',x.shape) 
        # if it doesn't have the first dim = 1 for batch dimension, add it
        if len(x.shape)<4: 
            x = np.expand_dims(x,3) 
            y = np.expand_dims(y,3) 
        y = np.transpose(y,(3,2,1,0)) 
        x = np.transpose(x,(3,2,1,0)) 

        # convert complex data to real and imaginary 
        last_axis = len(y.shape) 
        x = np.concatenate((np.expand_dims(np.real(x),last_axis) 
            ,np.expand_dims(np.imag(x),last_axis)),axis = last_axis) 
        y = np.concatenate((np.expand_dims(np.real(y),last_axis) 
            ,np.expand_dims(np.imag(y),last_axis)),axis = last_axis) 

        print('x shape after preprocess',y.shape) 
        return x, y  

 
def preprocess_data_batch(data): 
    # expected input size: (batch_size,axial,lateral,n_channels)

    mean = np.mean(data,axis=(1,2,3),keepdims = True)
    std = np.std(data,axis=(1,2,3),keepdims = True)
    data = (data - mean)/std

    return data

def chan2bmode(channel_data,dr=[-80, 20],rescale = True):

    real_data = channel_data[:,:,:,:,0].sum(dim=1)
    imag_data = channel_data[:,:,:,:,1].sum(dim=1)
    abs_data = torch.sqrt(real_data.pow(2)+imag_data.pow(2))
    
    # normalize
    mean_data = torch.mean(torch.mean(abs_data,dim=2,
        keepdim=True),dim=1,keepdim = True) + 1E-16
    norm_data = abs_data/mean_data

    # compress
    log_data = 20*torch.log10(norm_data)

    # clip to dynamic range
    log_data = torch.clamp(log_data,dr[0],dr[1])
    
    # rescale to 1 - 255 for tb functions
    if rescale: 
        min_data = torch.min(log_data)
        max_data = torch.max(log_data)
        log_data = 255*(log_data-min_data)/(max_data-min_data)

    return log_data

def write_imgs(writer,tag,x,y,pred,lognum,args):

    x_l = chan2bmode(x[0,:,:,:,:].unsqueeze(0),args['dr']).byte() 
    y_l = chan2bmode(y[0,:,:,:,:].unsqueeze(0),args['dr']).byte()
    pred_l = chan2bmode(pred[0,:,:,:,:].unsqueeze(0),args['dr']).byte()
    imgs = torch.cat((y_l,x_l,pred_l),0).unsqueeze(1)
    grid = torchvision.utils.make_grid(imgs)
    writer.add_image(tag,grid,lognum)

def write_bm_loss(writer,tag,y,pred,lognum,args):

    y_l = chan2bmode(y[0,:,:,:,:].unsqueeze(0),args['dr'])
    pred_l = chan2bmode(pred[0,:,:,:,:].unsqueeze(0),args['dr'])
    bm_loss = torch.mean(torch.abs(y_l - pred_l))
    writer.add_scalar(tag,bm_loss,lognum)

def write_bm_lesion_loss(writer,tag,y,pred,lognum,args):

    y_l = chan2bmode(y[0,:,50:-50,50:-50,:].unsqueeze(0),args['dr'])
    pred_l = chan2bmode(pred[0,:,50:-50,50:-50,:].unsqueeze(0),args['dr'])
    bm_loss = torch.mean(torch.abs(y_l - pred_l))
    writer.add_scalar(tag,bm_loss,lognum)

def write_phase_loss(writer,tag,y,p,lognum,args):

    y_phase = torch.atan(y[0,:,:,:,1]/y[0,:,:,:,0])
    p_phase = torch.atan(p[0,:,:,:,1]/p[0,:,:,:,0])

    phase_loss = torch.mean(torch.abs(y_phase-p_phase))
    writer.add_scalar(tag,phase_loss,lognum)


