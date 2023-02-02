import sys 
import os
sys.path.append(os.getcwd())
import pytorch_lightning as pl
import lightly
import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torchvision
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import ParameterGrid
from torchvision import datasets
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss

#%%

#Parser for the gridsearch
parser = ArgumentParser()
parser.add_argument("--param_i", type=int, default=0)
parser.add_argument("--path_to_data", type=str, default="/mnt/beegfs/apuissant/train_label_16")
parser.add_argument("--path_save_model", type=str, default="/mnt/beegfs/apuissant/gridsearch_16")
parser.add_argument("--num_workers", type=int, default=16)


args = parser.parse_args()
dict_args = vars(args)
param = dict_args['param_i']
path_to_data = dict_args['path_to_data']
path_save_model = dict_args['path_save_model']
num_workers = dict_args['num_workers']

input_size = 224 #Images input size

#Create the paramater grid for the optimization search
batch_size = [64, 128, 256]
max_epochs = [50,100,300]
temp = [0.1, 0.5, 0.8]
param_grid = {'batch_size' : batch_size, 'max_epochs' : max_epochs, 'temperature' : temp}
grid = list(ParameterGrid(param_grid))

batch_size = grid[param]["batch_size"]
max_epochs = grid[param]["max_epochs"]
temp = grid[param]["temperature"]

#To use pre determined parameter values
if param == -1 :
  batch_size=256
  max_epochs = 300
  temp = 0.5

n_class = len(os.listdir(path_to_data+"/unlabeled")) #Number of classes (used to balance the training)

print('batch_size = '+str(batch_size)+' max_epochs = '+str(max_epochs)+' temperature = '+str(temp))

#%%


#Augmentation function for simCLR
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=input_size,
    min_scale=0.5,
    vf_prob=0.5,
    rr_prob=0.5,
    gaussian_blur=0,
    cj_prob=0
)

#Training dataset with all images
dataset_unlabeled_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_data+"/unlabeled"
)

#Creation of the train sampler to balance the classes
dataset_unlabeled = datasets.ImageFolder(path_to_data+"/unlabeled")
trainratio = np.bincount(dataset_unlabeled.targets)
classcount = trainratio.tolist()
train_weights = 1./torch.tensor(classcount, dtype=torch.float)
train_sampleweights = train_weights[dataset_unlabeled.targets]
train_sampler = WeightedRandomSampler(weights=train_sampleweights, 
num_samples = len(train_sampleweights))



#dataloaders
dataloader_unlabeled_simclr = torch.utils.data.DataLoader(
    dataset_unlabeled_simclr,
    batch_size=batch_size,
    collate_fn=collate_fn,
    sampler = train_sampler,
    drop_last=True,
    num_workers=num_workers
)


#%% Model creation

class SimCLRModel(pl.LightningModule):
    def __init__(self,t=0.5):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        
        #Loading pretrained resnet
        resnet = torchvision.models.resnet50(pretrained=True)
        
        #Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        #Adding multilinear projection head
        hidden_dim = resnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, 128)

        #On définit la fonction de coût
        self.criterion = NTXentLoss(temperature = t)

    #Generate embeddings for one batch
    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    #Training step
    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    #Optimizers configuration
    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, max_epochs
        )
        return [optim], [scheduler]
    
#%%

torch.cuda.empty_cache()
gpus = 1 if torch.cuda.is_available() else 0

model = SimCLRModel(t=temp)

trainer = pl.Trainer(
    max_epochs=max_epochs, progress_bar_refresh_rate=100, log_every_n_steps=1, gpus=gpus
)
trainer.fit(model, dataloader_unlabeled_simclr)

#Saving trained model
trainer.save_checkpoint(path_save_model+"/simclr_bs_"+str(batch_size)+"_nepochs_"+str(max_epochs)+"_t_"+str(temp)+".ckpt")

