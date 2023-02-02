# Convergence in sympatric swallowtail butterflies reveal the significance of ecological interactions as driver of trait diversification at global scale


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



