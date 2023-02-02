
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input 

# models 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

# for everything else
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import umap
import hdbscan
from sklearn.metrics import silhouette_samples, silhouette_score
import shutil
import matplotlib.cm as cm

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--path", type=str, default = r"C:\Users\Agathe\Desktop\dataset_all") #path to the segmented images
parser.add_argument("--path_clustering", type=str, default = "C:/Users/Agathe/Desktop/clustering") #path to the clustering folder to save clustered images

args = parser.parse_args()
dict_args = vars(args)

path = dict_args['path']
path_clustering = dict_args['path_clustering']

#change the working directory to the path where the images are located
os.chdir(path)

#%%     

#load the model first and pass as an argument
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)



def extract_features(file, model):
    #load the image as a 224x224 array
  img = load_img(file, target_size=(224,224))
  #convert from 'PIL.Image.Image' to numpy array
  img = np.array(img) 
  #reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
  reshaped_img = img.reshape(1,224,224,3) 
  #prepare image for model
  imgx = preprocess_input(reshaped_img)
  #get the feature vector
  features = model.predict(imgx, use_multiprocessing=True)
  return features


data = {}
p = path

#this list holds all the image filename
images = []

# creates a ScandirIterator aliased as files
with os.scandir(p) as files:
  # loops through each file in the directory
    for file in files:
            images.append(file.name)
            
#%%
# loop through each image in the dataset
for image in tqdm.tqdm(images):
        feat = extract_features(image,model)
        data[image] = feat
        
#%%
filenames = images

# get a list of just the features
feat = np.array(list(data.values()))
# reshape 
feat = feat.reshape(-1,4096)


#%% UMAP projection
mapper = umap.UMAP(n_components=3).fit(feat)
umap_results=mapper.embedding_

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(umap_results[:,0],umap_results[:,1],umap_results[:,2])
plt.show()
#%%

stop = 0

while stop==0 :
    
    plt.close('all')
    
    mcs = int(input("Choose min_cluster_size: "))
    cse = float(input("Choose cluster_selection_epsilon: "))

    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs,cluster_selection_epsilon=cse) #Changer les paramètres pour choisir les clusters
    clusterer.fit(umap_results)
    labels = clusterer.labels_
    
    print("Number of clusters: "+str(clusterer.labels_.max()))
    print("Number of outliers: "+str(sum(clusterer.labels_==-1)))
    
    
    #Silhouette score calculation and cluster visualization
    X = umap_results
    
    X = np.array([X[i] for i in range(len(X)) if labels[i]!=-1])
    cluster_labels = np.array([labels[i] for i in range(len(labels)) if labels[i]!=-1])
    
    n_clusters = len(np.unique(cluster_labels))
    
    # Create a subplot with 1 row and 2 columns
    fig.set_size_inches(18, 7)
    ax1 = plt.subplot(121)
    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    
    
    
    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    
    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    
        ith_cluster_silhouette_values.sort()
    
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
    
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )
    
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    
    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
    # 2nd Plot showing the actual clusters formed
    ax2=plt.subplot(122, projection='3d')
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(
        X[:, 0], X[:, 1], X[:,2], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )
    
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")
    
    plt.suptitle(
        "Silhouette analysis with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )
    
    plt.show()
    
    stop = int(input('Choose these values ? (0/1) '))


#%% Create a dictionary with groups
# holds the cluster id and the images { id: [images] }
groups = {}
for file, cluster in zip(filenames,labels):
    if cluster not in groups.keys():
        groups[cluster] = []
        groups[cluster].append(file)
    else:
        groups[cluster].append(file)
        
#%%
#Create folders with grouped images to view and potentially create an evaluation dataset
shutil.rmtree(path_clustering)
os.mkdir(path_clustering)
for i in np.unique(labels) :
    if i!=-1 :
        os.mkdir(path_clustering+"/group_"+str(i))
        for j in range(len(groups[i])) :
            shutil.copy(p+'/'+groups[i][j][:-4]+'.jpg',path_clustering+"/group_"+str(i))
