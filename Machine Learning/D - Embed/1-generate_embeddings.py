import pytorch_lightning as pl
import lightly
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.loss import NTXentLoss
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--path_to_data", type=str, default = r"C:\Users\Agathe\Desktop\dataset_test") #path to the segmented images
parser.add_argument("--model_to_use", type=str, default = r"C:\Users\Agathe\Mon Drive\Données\saved_models\ContrastiveLearning\transfer_trained_on_all_classified_16") 
parser.add_argument("--path_save", type=str, default = "C:/Users/Agathe/Desktop") 
parser.add_argument("--param_i", type=int, default=-1)
parser.add_argument("--num_workers", type=int, default=16)

args = parser.parse_args()
dict_args = vars(args)

path_to_data = dict_args['path_to_data'] #Path to all images to generate embeddings for
model_to_use = dict_args['model_to_use'] #Path to trained model to use
path_save = dict_args['path_save'] #Path to save embeddings to 
param = dict_args['param_i']
num_workers = dict_args['num_workers']

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
  
input_size=224

def run():

    
    #Définition du modèle
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
        
        
    torch.cuda.empty_cache()
    model = SimCLRModel(t=temp)
    
    #Chargement du modèle voulu
    pre = torch.load(model_to_use+"/simclr_bs_"+batch_size+"_nepochs_"+max_epochs+"_t_"+temp+".ckpt")
    model.load_state_dict(pre['state_dict'])
    
    
    
    
    def generate_embeddings(model, dataloader_embed):
        """Generates representations for all images in the dataloader with
        the given model
        """
    
        embeddings = []
        labels = []
        filenames = []
        with torch.no_grad():
            for img, label, fnames in tqdm(dataloader_embed):
                
                if type(img)==tuple :
                  img = img[0].to(model.device)
                else :
                  img = img.to(model.device)
                labels.append(label)
                emb = model.backbone(img).flatten(start_dim=1)
                embeddings.append(emb)
                filenames.extend(fnames)
        embeddings = torch.cat(embeddings, 0)
        labels = torch.cat(labels, 0)
    
        embeddings = normalize(embeddings)
        return embeddings, labels, filenames
    
    
    model.eval()
    
    
    #Transformations to be applied to the image to pass them in the model
    embed_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((input_size, input_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=lightly.data.collate.imagenet_normalize['mean'],
            std=lightly.data.collate.imagenet_normalize['std'],
        )
    ])
    
    #Creation of the dataset to be passed and the associated dataloader
    dataset_embed = lightly.data.LightlyDataset(
        input_dir=path_to_data,
        transform= embed_transforms
    )
    
    dataloader_embed = torch.utils.data.DataLoader(
        dataset_embed,
        batch_size=100,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    d = generate_embeddings(model, dataloader_embed)
    
    return d

if __name__ == '__main__':
    d = run()

    embeddings = d[0]
    photo = d[2]

    #PCA on raw coordinates dimensions 2048
    pca = PCA()
    pca.fit(embeddings)
    
    #Selection of the axes bringing more than 0.5% of explained variance in addition
    sel_comp = pca.explained_variance_ratio_-np.concatenate([pca.explained_variance_ratio_[1:],[0]])>0.0005
    
    print(pca.explained_variance_ratio_[sel_comp]*100)
    print(sum(pca.explained_variance_ratio_[sel_comp]*100))
    print(np.cumsum(pca.explained_variance_ratio_[sel_comp]*100))

    #summary figures of the PCA
    plt.figure()
    plt.plot(np.cumsum(pca.explained_variance_ratio_[sel_comp]), 'o-')
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.xticks(ticks=sel_comp,labels=range(1,np.sum(sel_comp)+1))
    
    plt.figure()
    plt.bar(sel_comp,pca.explained_variance_ratio_[sel_comp]*100)
    plt.xlabel('components')
    plt.ylabel('explained variance');
    plt.xticks(ticks=sel_comp,labels=range(1,np.sum(sel_comp)+1))

    pca_embeddings = pca.fit_transform(embeddings)
    pca_embeddings = pca_embeddings[:,sel_comp]

    plt.figure()
    plt.scatter(pca_embeddings[:,0],pca_embeddings[:,1])
    plt.figure()
    plt.scatter(pca_embeddings[:,0],pca_embeddings[:,2])
    plt.figure()
    plt.scatter(pca_embeddings[:,1],pca_embeddings[:,2])
    
    #Sauvegarde des coordonnées brutes et dans l'ACP (attention les noms sont ceux bruts des photos)
    pd.DataFrame(pca_embeddings,index=photo).to_csv(path_save+"/pca_embeddings.csv", sep=";")
    pd.DataFrame(embeddings,index=photo).to_csv(path_save+"/embeddings.csv", sep=";")

    
