import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import tqdm
from pixellib.instance import custom_segmentation


from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--path_masks", type=str, default = "J:/Segmentation/test_code/segmented") #path to the segmentation folder
parser.add_argument("--path_jpg", type=str, default =r'J:\Data\test_images\\') #path to the jsp image files
parser.add_argument("--wing_name", type=list, default =['_Ant_g','_Post_g','_Ant_d','_Post_d']) #List of names for the 4 wings

args = parser.parse_args()
dict_args = vars(args)

path_masks = dict_args['path_masks']
path_jpg = dict_args['path_jpg']
wing_names = dict_args['wing_name']

model = "mask_rcnn_model.085-0.066738.h5" #Trained segmentation model


#loading the automatic segmentation model with PixelLib
segment_image = custom_segmentation()
segment_image.inferConfig(num_classes= 1, class_names= ["BG","aile"])
segment_image.load_model(path_masks[:-9]+model)

diffpixels = [] #List to store the difference of piwel between raw segmentation and refined segmentation for each image

listdirs = os.listdir(path_masks)
listdirs = [x for x in listdirs if x[-7:]=="_output"] #Keep only segmentation output folder


for d in tqdm.tqdm(listdirs) :
    
    #Loading the whole image mask from the wing masks
    mask_1_segmented = np.array(Image.open(path_masks+'/'+d+'/'+d[:-7]+wing_names[0]+'.jpg'))
    mask_2_segmented = np.array(Image.open(path_masks+'/'+d+'/'+d[:-7]+wing_names[1]+'.jpg'))
    mask_3_segmented = np.array(Image.open(path_masks+'/'+d+'/'+d[:-7]+wing_names[2]+'.jpg'))
    mask_4_segmented = np.array(Image.open(path_masks+'/'+d+'/'+d[:-7]+wing_names[3]+'.jpg'))
    
    shape = mask_1_segmented+mask_2_segmented+mask_3_segmented+mask_4_segmented
    
    os.chdir(path_jpg)
    indexim = d[:-7]+'.jpg'
    
    #Automatic segmentation with PixelLib
    segmask, output = segment_image.segmentImage(path_jpg+indexim)
    
    segmask_all = np.sum(segmask['masks'], axis=2).astype('uint8')
    
    #What are the pixels in common ?
    intersection = cv2.bitwise_and(segmask_all, shape)
    
    #Saving the difference of pixels
    diffpixels.append(np.sum(segmask_all) - np.sum(intersection))

#%%

#Investinagting which images have a large difference of pixels using their distribution and the .99 quantile
plt.hist(diffpixels, log=True)
lim=np.quantile(diffpixels,0.99) #CHANGE THE QUANTILE IF YOU WANT MORE OR LESS CORRECTION
sum(np.array(diffpixels)>lim)        
np.array(listdirs)[np.array(diffpixels)>lim] #this gives a list of the names of inciminated images
            
np.save(path_masks[:-9]+"to_correct.npy",np.array(listdirs)[np.array(diffpixels)>lim]) #Uncomment to save

            