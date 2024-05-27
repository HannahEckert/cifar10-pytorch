import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import ToTensor,Compose,RandomHorizontalFlip, Normalize, ToPILImage, RandomRotation, ColorJitter
from torchvision.utils import make_grid
import mcbe

def extract(filename):
    with open(filename,"rb") as f:
        batch_data = pickle.load(f,encoding="bytes")
    return batch_data

data = [] #Store all batches in a list
for files in os.listdir("cifar-10-batches-py"):
    if "_batch" in files:
        data.append(extract(os.path.join('./cifar-10-batches-py',files)))

class CIFAR10(Dataset):
    
    def __init__(self,root,train=True,transforms=None):
        self.root = root
        self.transforms = transforms
        self.split = train
        
        self.data = []
        self.targets = []
        self.train_data = [file for file in os.listdir(root) if "data_batch" in file]
        self.test_data = [file for file in os.listdir(root) if "test_batch" in file]
                
        data_split = self.train_data if self.split else self.test_data
        
        for files in data_split:
            entry = self.extract(os.path.join(root,files))
            self.data.append(entry["data"])
            self.targets.extend(entry["labels"])
                
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.load_meta()
        
    def extract(self,filename):
        with open(filename,"rb") as f:
            batch_data = pickle.load(f,encoding="latin1")
        return batch_data  
    
    def load_meta(self):
        path = os.path.join(self.root,"batches.meta")
        with open(path,"rb") as infile:
            data = pickle.load(infile,encoding="latin1")
            self.classes = data["label_names"]
            self.classes_to_idx = {_class:i for i,_class in enumerate(self.classes)}
            
    def plot(self,image,target=None):
        if target is not None:
            print(f"Target :{target} class :{self.classes[target]}")
        plt.figure(figsize=(2,2))
        plt.imshow(image.permute(1,2,0))
        plt.show()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        image,target = self.data[idx],self.targets[idx]
        image = Image.fromarray(image)
        
        if self.transforms:
            image = self.transforms(image)
            
        return image,target
    
train_set = CIFAR10(root="./cifar-10-batches-py",train=True,
                    transforms=Compose([
                        ToTensor()]))
test_set = CIFAR10(root="./cifar-10-batches-py",train=False,
                    transforms=Compose([
                        ToTensor()]))

class Maxbias_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,max_bias, bias):
        return 0.1*np.linalg.norm(np.max(np.array([bias - max_bias, np.zeros_like(bias)]),axis=0))
    
class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet,self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=8,stride=1,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=32,kernel_size=(3,3),padding=1,stride=1)
        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1)

        self.fc1 = nn.Linear(in_features=6*6*256,out_features=256)
        self.fc2 = nn.Linear(in_features=256,out_features=512)
        self.fc3 = nn.Linear(in_features=512,out_features=128)
        self.fc4 = nn.Linear(in_features=128,out_features=64)
        self.fc5 = nn.Linear(in_features=64,out_features=10)
        
        self.max_pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self,x,targets,inj=True):
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
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        logits = self.fc5(x)
        
        loss = None
        if targets is not None:
            if not inj:
                loss = F.cross_entropy(logits,targets)
            else:
                loss1 = F.cross_entropy(logits,targets)
                max_bias = mcbe.dd_mcbe(W=np.array(self.fc2.weight.detach().numpy()),X_train = mcbe_train, num_estimation_points=5000,dd_method="blowup")
                loss_fn_maxbias = Maxbias_loss()
                loss2 = loss_fn_maxbias(max_bias,self.fc2.bias.detach().numpy())
                loss = loss1 + loss2
                #print("crossentropy:",loss1,"maxbias:",loss2)
        return logits,loss
    
    def configure_optimizers(self,config):
        optimizer = optim.Adam(self.parameters(),lr=config.lr,betas=config.betas,weight_decay=config.weight_decay)
        return optimizer
    
model = ConvNet()

class TrainingConfig:
    
    lr=3e-4
    betas=(0.9,0.995)
    weight_decay=5e-4
    num_workers=0
    max_epochs=10
    batch_size=64
    ckpt_path="./Model_inj.pt" #Specify a model path here. Ex: "./Model.pt"
    shuffle=True
    pin_memory=True
    verbose=True
    
    def __init__(self,**kwargs):
        for key,value in kwargs.items():
            setattr(self,key,value)

train_config = TrainingConfig()

class Trainer:
    def __init__(self,model,train_dataset,test_dataset,config):
        self.model = model
        self.train_dataset=train_dataset
        self.test_dataset=test_dataset
        self.config = config
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = self.model.to(self.device)
    
    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        torch.save(raw_model.state_dict(),self.config.ckpt_path)
        print("Model Saved!")
        
    def train(self):
        model,config = self.model,self.config
        raw_model = self.model.module if hasattr(self.model,"module") else self.model
        optimizer = raw_model.configure_optimizers(config)
        
        def run_epoch(split):
            is_train = split=="train"
            if is_train:
                model.train()
            else:
                model.eval() #important don't miss this. Since we have used dropout, this is required.
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data,batch_size=config.batch_size,
                                shuffle=config.shuffle,
                                pin_memory=config.pin_memory,
                                num_workers=config.num_workers)
            
            losses = []
            accuracies = []
            correct = 0
            num_samples = 0
            
            pbar = tqdm(enumerate(loader),total=len(loader)) if is_train and config.verbose else enumerate(loader)
            for it,(images,targets) in pbar:
                images = images.to(self.device)
                targets = targets.to(self.device)
                num_samples += targets.size(0)
                
                with torch.set_grad_enabled(is_train):
                    #forward the model
                    logits,loss = model(images,targets)
                    loss = loss.mean()
                    losses.append(loss.item())
                    
                with torch.no_grad():
                    predictions = torch.argmax(logits,dim=1) #softmax gives prob distribution. Find the index of max prob
                    correct+= predictions.eq(targets).sum().item()
                    accuracies.append(correct/num_samples)
                    
                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if config.verbose:
                        pbar.set_description(f"Epoch:{epoch+1} iteration:{it+1} | loss:{np.mean(losses)} accuracy:{np.mean(accuracies)} lr:{config.lr}")
                    
                    self.train_losses.append(np.mean(losses))
                    self.train_accuracies.append(np.mean(accuracies))
            
            if not is_train:
                test_loss = np.mean(losses)
                if config.verbose:
                    print(f"\nEpoch:{epoch+1} | Test Loss:{test_loss} Test Accuracy:{correct/num_samples}\n")
                self.test_losses.append(test_loss)
                self.test_accuracies.append(correct/num_samples)
                return test_loss
                
        best_loss = float('inf')
        test_loss = float('inf')
        
        for epoch in range(config.max_epochs):
            run_epoch('train')
            if self.test_dataset is not None:
                test_loss = run_epoch("test")
                
            good_model = self.test_dataset is not None and test_loss < best_loss
            if config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

for i in range(1):
    Model = ConvNet()
    # model.load_state_dict(torch.load("./Model.pt")) #Uncomment this to load pre-trained weights
    train_set = CIFAR10(root="./cifar-10-batches-py",train=True,
                        transforms=Compose([
                            ToTensor(),
                            RandomHorizontalFlip(),
                            RandomRotation(degrees=10),
                            ColorJitter(brightness=0.5),
                            Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                                    std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                        ]))

    test_set = CIFAR10(root="./cifar-10-batches-py",train=False,
                    transforms=Compose([
                            ToTensor(),
                            Normalize(mean=(0.4913997551666284, 0.48215855929893703, 0.4465309133731618),
                                    std=(0.24703225141799082, 0.24348516474564, 0.26158783926049628))
                        ]))

    train_config = TrainingConfig(max_epochs=50,
                                lr=0.0009446932175584296,
                                weight_decay=0.00011257445443209662,
                                ckpt_path="./models/Final_Model_inj" +str(i) +".pt",
                                batch_size=64,
                                num_workers=0)

    trainer = Trainer(model,train_dataset=train_set,
                    test_dataset=test_set,config=train_config)
    trainer.train()
    # torch.save(Model.state_dict(),"./models/Model300.pt") #Uncomment this if you want to save the model 
    torch.save(trainer.train_losses,"./log_inj2/train_losses" + str(i) +".pt")
    torch.save(trainer.train_accuracies,"./log_inj2/train_accuracies" + str(i) +".pt")
    torch.save(trainer.test_losses,"./log_inj2/test_losses" + str(i) +".pt")
    torch.save(trainer.test_accuracies,"./log_inj2/test_accuracies" + str(i) +".pt")