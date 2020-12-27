from torch import nn
import torch
import numpy as np

class Network(nn.Module):
    def __init__(self,load_pretrain = 1):
        super().__init__()
                            
        # define the network convolutional layers
        k_size = (3,3,11)
        pad = (1,1,5)
        self.hidden1 = nn.Conv3d(in_channels=2,out_channels=5,
                kernel_size=k_size,stride=1,padding=pad)
        self.hidden2 = nn.Conv3d(in_channels=5,out_channels=5,
                kernel_size=k_size,stride=1,padding=pad)
        k_size = (3,3,7)
        pad = (1,1,3)
        self.hidden3 = nn.Conv3d(in_channels=5,out_channels=5,
                kernel_size=k_size,stride=1,padding=pad)
        self.hidden4 = nn.Conv3d(in_channels=5,out_channels=5,
                kernel_size=k_size,stride=1,padding=pad)
        k_size = (3,3,3)
        pad = (1,1,1)
        self.hidden5 = nn.Conv3d(in_channels=5,out_channels=5,
                kernel_size=k_size,stride=1,padding=pad)
        self.hidden6 = nn.Conv3d(in_channels=5,out_channels=5,
                kernel_size=k_size,stride=1,padding=pad)
        self.hidden7 = nn.Conv3d(in_channels=5,out_channels=2,
                kernel_size=k_size,stride=1,padding=pad)

        # Define sigmoid activation and softmax output 
        self.relu = nn.ReLU()

        # if we are loading the pretrained weights, load the checkpoint 
        if load_pretrain:
            d = np.load('./pretrained_model_weights.npy', allow_pickle = True)
            kperm = (4,3,0,1,2) 
            k1 = torch.Tensor(np.transpose(d.item().get('k1'),kperm))
            b1 = torch.Tensor(d.item().get('b1'))
            k2 = torch.Tensor(np.transpose(d.item().get('k2'),kperm))
            b2 = torch.Tensor(d.item().get('b2'))
            k3 = torch.Tensor(np.transpose(d.item().get('k3'),kperm))
            b3 = torch.Tensor(d.item().get('b3'))
            k4 = torch.Tensor(np.transpose(d.item().get('k4'),kperm))
            b4 = torch.Tensor(d.item().get('b4'))
            k5 = torch.Tensor(np.transpose(d.item().get('k5'),kperm))
            b5 = torch.Tensor(d.item().get('b5'))
            k6 = torch.Tensor(np.transpose(d.item().get('k6'),kperm))
            b6 = torch.Tensor(d.item().get('b6'))
            k7 = torch.Tensor(np.transpose(d.item().get('k7'),kperm))
            b7 = torch.Tensor(d.item().get('b7'))
            
            self.hidden1.weight = torch.nn.Parameter(k1)
            self.hidden1.bias = torch.nn.Parameter(b1)
            self.hidden2.weight = torch.nn.Parameter(k2)
            self.hidden2.bias = torch.nn.Parameter(b2)
            self.hidden3.weight = torch.nn.Parameter(k3)
            self.hidden3.bias = torch.nn.Parameter(b3)
            self.hidden4.weight = torch.nn.Parameter(k4)
            self.hidden4.bias = torch.nn.Parameter(b4)
            self.hidden5.weight = torch.nn.Parameter(k5)
            self.hidden5.bias = torch.nn.Parameter(b5)
            self.hidden6.weight = torch.nn.Parameter(k6)
            self.hidden6.bias = torch.nn.Parameter(b6)
            self.hidden7.weight = torch.nn.Parameter(k7)
            self.hidden7.bias = torch.nn.Parameter(b7)
            
      
    def forward(self, x):

        # Pass the input tensor through each of our operations
        x = x.permute(0,4,2,3,1)
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.hidden3(x)
        x = self.relu(x)
        x = self.hidden4(x)
        x = self.relu(x)
        x = self.hidden5(x)
        x = self.relu(x)
        x = self.hidden6(x)
        x = self.relu(x)
        x = self.hidden7(x)
        x = x.permute(0,4,2,3,1)
        
        return x

