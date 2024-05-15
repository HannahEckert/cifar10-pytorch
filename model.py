import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,stride=1,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=(3,3),padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1)

        self.fc1 = nn.Linear(in_features=6*6*256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=128)
        self.fc3 = nn.Linear(in_features=128,out_features=64)
        self.fc4 = nn.Linear(in_features=64,out_features=10)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self,x,targets,inj=False):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x.view(-1,6*6*256)
        x = F.relu(self.fc1(x))
        mcbe_train = x.detach().numpy()
        print("shape of mcbe_train",mcbe_train.shape)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        logits = self.fc4(x)
        
        loss = None
        if targets is not None:
            if not inj:
                loss = F.cross_entropy(logits,targets)
            else:
                loss1 = F.cross_entropy(logits,targets)
                max_bias = mcbe.dd_mcbe(W=np.array(self.fc2.weights.detach().numpy())[-1],X_train = mcbe_train, num_estimation_points=10000,dd_method="blowup")
                loss2 = 0.1*np.linalg.norm(np.max(np.array([self.fc2.bias.detach().numpy() - max_bias, np.zeros_like(self.fc2.bias.detach().numpy())]),axis=0))
        return logits,loss
    
    def configure_optimizers(self,config):
        optimizer = optim.Adam(self.parameters(),lr=config.lr,betas=config.betas,weight_decay=config.weight_decay)
        return optimizer