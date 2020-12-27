Dataset Generation: 
This repo contains code which will preprocess the validation set provided with the TMI publication. The dataset can be found under additional media in the publication. You will need to download that dataset and put it in the 'datasets' folder in the root of this repo for the following instructions to work. 

Within the dataset_gen folder, there are two main files used to create the anechoic lesion dataset
make_noise_sweep_dataset.m and make_npy.py
make_noise_sweep_dataset.m will take the output data from the FieldII Pro simulations, add thermal and reverberation noise, and then apply delays to create channel data. 
To beamform, we used the interp2_gpumex function developed by Dongwoon Hyun, presented in the 2018 GTC presentation. We have provided the resulting data from this script in the 'mat_files' folder inside the zipped dataset file provided with the publication. 

make_npy.py takes the output .mat files from the first script, imports it to python and applies the preprocessing needed to use on the network. For the input arguments to the function patch_size and n_patches can be adjusted when creating a training dataset. However, we have left the default value for these arguments to provide centered, 200x200 pixel samples for this validaiton set. This script will create a 'npy' folder in the dataset which can be used on our python scripts for infer.py or train.py

The output of make_npy.py is not provided, and it will be necessary to run this script before trying to infer or train in this repository. 

Inference: 
infer.py is the script you can run to filter the validation set once it has been preprocessed. All results will be sent to the 'logs' folder, with the folder name of the date it was ran on. You can look at the results in two ways: 
1. tensorboard. tensorboard summaries are written for the loss function, the log loss function (detailed in the publication) and the b-mode images of the reference, unfiltered input data and filtered output data. 
2. mat files. mat_files are also written which contain the full channel data of the reference, unfiltered input and filtered output data. 

Training: 
train.py is the script you can run to train the network from scratch. Because of it's size, the training dataset for our publication is not provided. 
