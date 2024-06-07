import sys 
import os
sys.path.append(os.getcwd())
import pytorch_lightning as pl
import lightly
import os
import torch
from torch.utils.data import WeightedRandomSampler
import torch.nn as nn
import torchvision
import numpy as np
from torch.nn import functional as F
from argparse import ArgumentParser
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score,cohen_kappa_score
from torchvision import datasets
import torch.optim as optim
import time
import copy
from sklearn.preprocessing import normalize


#%%

#Parser for the gridsearch
parser = ArgumentParser()
parser.add_argument("--param_i", type=int, default=0)
parser.add_argument("--path_to_data", type=str, default="/mnt/beegfs/apuissant/without_tail_train_dataset")
parser.add_argument("--path_save_model", type=str, default="/mnt/beegfs/apuissant/without_tail_model")
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

model = SimCLRModel()

#Charger le modèle à évaluer
pre = torch.load(path_save_model+"/simclr_bs_"+str(batch_size)+"_nepochs_"+str(max_epochs)+"_t_"+str(temp)+".ckpt")
model.load_state_dict(pre['state_dict'])

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


# We create a torchvision transformation for embedding the dataset after 
# training
test_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((input_size, input_size)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])


#Creation of different datasets for training, evaluation and balancing of classes
dataset_train_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_data+"/train"
)

dataset_unlabeled_simclr = lightly.data.LightlyDataset(
    input_dir=path_to_data+"/unlabeled"
)

dataset_test = lightly.data.LightlyDataset(
    input_dir=path_to_data+"/validation",
    transform=test_transforms
)

#Creation of the train sampler to balance the classes
dataset_unlabeled = datasets.ImageFolder(path_to_data+"/unlabeled")
trainratio = np.bincount(dataset_unlabeled.targets)
classcount = trainratio.tolist()
train_weights = 1./torch.tensor(classcount, dtype=torch.float)
train_sampleweights = train_weights[dataset_unlabeled.targets]
train_sampler = WeightedRandomSampler(weights=train_sampleweights, 
num_samples = len(train_sampleweights))



dataloader_train_simclr = torch.utils.data.DataLoader(
    dataset_train_simclr,
    batch_size=batch_size,
    collate_fn = collate_fn,
    sampler = train_sampler,
    drop_last=True,
    num_workers=num_workers
)

dataloader_unlabeled_simclr = torch.utils.data.DataLoader(
    dataset_unlabeled_simclr,
    batch_size=batch_size,
    collate_fn=collate_fn,
    sampler = train_sampler,
    drop_last=True,
    num_workers=num_workers
)

dataloader_test = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

#%%

torch.cuda.empty_cache()
gpus = 1 if torch.cuda.is_available() else 0

#We load the trained model and freeze the network weights.
embed = model
for param in embed.parameters():
    param.requires_grad = False
    
#We replace the projection head with a classification head
embed.projection_head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=n_class),
        )


#Definition of the cost function and optimizers
n_epochs=50 #Training duration
criterion = F.cross_entropy
optimizer_conv = optim.AdamW(embed.projection_head.parameters(), lr =0.001)
exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_conv, milestones=[int(n_epochs*0.6),
                                                                              int(n_epochs*0.8)],
                                                                              gamma=0.1)
#Calculation of the number of classes and size of datasets
dataset_sizes = {'train' : len(dataset_train_simclr), 'validation' : len(dataset_test) }
print(dataset_sizes)
print(n_class)


def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train' :
              dataloaders = dataloader_train_simclr

              # Iterate over data.
              for inputs, labels, filenames in dataloaders:
                  if type(inputs)==tuple :
                    inputs = inputs[0]
                    # print(inputs[0].shape)
                  inputs = inputs.to(model.device)
                  labels = labels.to(model.device)

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(inputs)
                      _, preds = torch.max(outputs, 1)
                      loss = criterion(outputs, labels)

                      # backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()

                  # statistics
                  running_loss += loss.item() * inputs.size(0)
                  running_corrects += torch.sum(preds == labels.data)

            else :
              dataloaders = dataloader_test

              # Iterate over data.
              for inputs, labels, filenames in dataloaders:
                  inputs = inputs.to(model.device)
                  labels = labels.to(model.device)

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # forward
                  # track history if only in train
                  with torch.set_grad_enabled(phase == 'train'):
                      outputs = model(inputs)
                      _, preds = torch.max(outputs, 1)
                      loss = criterion(outputs, labels)

                      # backward + optimize only if in training phase
                      if phase == 'train':
                          loss.backward()
                          optimizer.step()

                  # statistics
                  running_loss += loss.item() * inputs.size(0)
                  running_corrects += torch.sum(preds == labels.data)

            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


#Entraînement et récupération du meilleur modèle de classification
model_conv = train_model(embed, criterion, optimizer_conv,exp_lr_scheduler, num_epochs=n_epochs)


#Model accuracy calculations
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in dataloader_test:
        images, labels, filenames = data
        # calculate outputs by running images through the network
        images = images.to(model_conv.device)
        labels = labels.to(model_conv.device)
        outputs = model_conv(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct // total} %')


# prepare to count predictions for each class
import os
classes = os.listdir(path_to_data+"/validation")
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in dataloader_test:
        images, labels, filenames = data
        images = images.to(model_conv.device)
        labels = labels.to(model_conv.device)
        outputs = model_conv(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    if total_pred[classname]>0 :
      accuracy = 100 * float(correct_count) / total_pred[classname]
      print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
      
      
      
#Calculation of the f1-stat and the kappa score 
prediction_list = []
labels_list = []

def predict(dataloader):
    with torch.no_grad():
        for data in dataloader:
            images, labels, filenames = data
            images = images.to(model_conv.device)
            labels = labels.to(model_conv.device)
            outputs = model_conv(images)
            _, predictions = torch.max(outputs, 1)
            prediction_list.append(predictions.cpu())
            labels_list.append(labels.cpu())
        return (labels_list, prediction_list)
    
labels_list, prediction_list = predict(dataloader_test)

labels_list, prediction_list = labels_list[0], prediction_list[0]


print('f1 Score :')
print(f1_score(labels_list,prediction_list, average='micro'))
print('kappa :')
print(cohen_kappa_score(labels_list,prediction_list))








