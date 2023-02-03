# Convergence in sympatric swallowtail butterflies reveal the significance of ecological interactions as driver of trait diversification at global scale

🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋🦋

This repository contains the python code necessary to: 
- Segment butterfly images into 4 wings and remove all other parts
- Cluster the images into phenotypic groups
- Train a SimCLR model on these images
- Optimize the hyperparameters of the model based on classification evaluation
- Generate embeddings from a trained model

And the R code necessary to:
- Visualize the morpho space
- Run a permutation test to assess convergence and divergence in sympatry
- Assess which sex contributes to recent dimorphism evolution

Python packages needed :

*** 
**Segmentation**

```
pixellib
cv2
matplotlib
numpy
PIL
shapely
scipy
shutil
tqdm
skimage
rembg
argparse
```

**Clustering**

```
tensorflow
umap
hdbscan
sklearn
pathlib
```

**Training**

```
lightly
torch
torchvision
```

**Embed**
```
pytorch_lightning
pandas
```

*** 


**Instructions to run the codes are listed below**

## Machine Learning

### A - Segmentation

The segmentation code uses a pretrained segmentation model that uses PixelLib to automatically segment wings for other parts of the butterflies and background. Then there is a refinement step to obtain well segmented wings.

Python codes can be run from command lines. You should change directory to the folder named "A - Segmentation" using the following command : ```cd ./SimCLR_phenotypic_convergence/Machine Learning/A - Segmentation```


For the segmentation, you should prepare one folder named as you wish where various subfolders will be created. The trained segmentation model can be found in supplementary materials and should be put in this folder.

#### 1-segmentation.py

```
python 1-segmentation.py --path ./segmentation --path_jpg ./images_jpg --wing_name ['_Ant_g','_Post_g','_Ant_d','_Post_d'] --save_imwing_raw False --save_as_tif False --dim_images 3
```

Arguments :
- path: path to your segmentation folder
- path_jpg: path to your raw images in jpg format
- wing_name: suffixes used for the 4 wings
- save_imwing_raw: whether you want the raw segmentation output saved
- save_as_tif: whether you want the refined segmentation output as tiff, if not it will be saved as jpg
- dim_images: number of channels of your images (3 for RGB)

A subfolder named "segmented" will be created where segmentation outputs will be stored for your images. Segmentation output for an image named "ABC.jpg" will be stored in the "ABC_output" folder with one mask for each wing and the segmented image.

A subfolder named "rembg_out" will be created with PNG files of your images with the background removed.

#### 2-find_errors.py

This code is optional. It finds the images for which there is potentially errors in the refined segmentation by comparing the raw and refined segmentation. It saves the list of potentially erroneous images in a .nppy file that can be then used to "correct" the segmentation.


```
python 2-find_errors.py --path ./segmentation --path_jpg ./images_jpg --wing_name ['_Ant_g','_Post_g','_Ant_d','_Post_d']
```

#### 3-correct_errors.py

This code is optional. Should be run after find_errors, goes back to erroneous images and replaces their refined segmentation by the raw segmentation.

```
python 3-correct_errors.py --path ./segmentation --path_jpg ./images_jpg --wing_name ['_Ant_g','_Post_g','_Ant_d','_Post_d'] --save_imwing_raw False --save_as_tif False --dim_images 3
```

#### 4-get_imagedata.py

This code copies the segmented images in the subfolders to a folder of your choice.

```
python 4-get_imagedata.py --path ./segmentation --path_save ./all_segmented
```

Argument:
- path_save: path to where you want your dataset saved 


### B - Clustering

```cd ./SimCLR_phenotypic_convergence/Machine Learning/B - Clustering```

#### 1-VGG16_clustering.py

This code generated embeddings from a VGG16 pretrained network, projects them in 3 dimensions using a UMAP projection and clusters using HDBscan clustering.

You should create a folder named as you want to store the results from the clustering.

```
python 1-VGG16_clustering.py --path_images ./all_segmented --path_clustering ./clustering
```
Argument:
- path_images: path to the dataset with segmented images
- path_clustering: path to the designated folder to store the results from the clustering

Running this code, you will be asked to choose two parameters for the HDBscan clustering (see HDBscan documentation for more info).


#### 2-create_dataset.py

This code will create the unlabeled data used for SimCLR training, and the training and validation data used for the evaluation by classification, using the cluster labels as labels.

You should create a folder with whatever name to store the dataset.

```
python 2-create_dataset.py --path_images ./all_segmented --path_dataset ./dataset --path_clustering ./clustering
```

### C - Training

```cd ./SimCLR_phenotypic_convergence/Machine Learning/C - Training```

#### 1-simCLR_training.py

Trains the SimCLR model using the previously built dataset.

This code can be used either to train following a parameter grid search, or can be trained used the predetermined parameter values found in the article.

```
python 1-simCLR_training.py --param_i -1 --path_to_data ./dataset --path_save_model ./saved_models --num_workers 16 
```
Arguments:
- param_i: parameters for the model. If this argument is -1 (the default), the model will be trained using pre determined parameters (batch size=256, max_epochs=300, temperature=0.5). If you wish to change parameters I recommend that you look into the code.
- path_to_data: path to the previously built dataset.
- path_save_model: path to a folder where your trained model will be saved in a .ckpt file

#### 2-evaluation_classification.py

This code allows to estimate accuracy and other evaluation statistics for the classification of your image datasets into the clusters, using only the embeddings from the trained SimCLR model. It gives an idea of the quality of embeddings, if they contain enough information about the images to find the cluster labels.

```
python 2-evaluation_classification.py --param_i -1 --path_to_data ./dataset --path_save_model ./saved_models --num_workers 16 
```

### D - Embed

```cd ./SimCLR_phenotypic_convergence/Machine Learning/D - Embed```

#### 1-generate_embeddings.py

This code generates embeddings for your image dataset using your trained SimCLR model.
PCA embeddings are generated by taking the images scores on a certain numbers of PCA axes - adding PCA axes until the gain of variance does not surpass .5%.

```
python 1-generate_embeddings.py --param_i -1 --path_to_data ./dataset --path_save_model ./saved_model --path_save ./coordinates --num_workers 16 
```
Arguments:
- path_save: path where to save the pca coordinates and embeddings.


## R analysis

You can open the R project and open the files in the "code" folder.
All necessary data and image for vizualisation are provided.

#### Visualize_data.R 
This code plots the phylo morpho space and the scatterplot with butterfly images. It also computes the phylogenetic signal for colour pattern variation.

#### Permutations.R 
Code necessary to run the Lapointe-Garland permuation test to assess convergence and divergence among pairs in sympatry and allopatry. The run time is **very long** so results are saved in a RData file (saved_pairs_F.Rdata and saved_pairs_M.RData)

#### Pairs_analysis.R 
Reproduces the tests and plots from the paper.

#### Dimorphism.R
Computes dimorphism and ratio of raw contrasts for males and females, and analyses the distribution of ratio.

