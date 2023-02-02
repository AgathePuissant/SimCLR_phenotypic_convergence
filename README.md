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



