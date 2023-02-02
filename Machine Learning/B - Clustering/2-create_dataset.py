# -*- coding: utf-8 -*-
import os
from pathlib import Path
from PIL import Image
import random
import shutil

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--path_images", type=str, default = r"C:\Users\Agathe\Desktop\dataset_test") #path to the segmented images
parser.add_argument("--path_dataset", type=str, default = "C:/Users/Agathe/Desktop/train_label_test") 
parser.add_argument("--path_clustering", type=str, default = "C:/Users/Agathe/Desktop/clustering") 

args = parser.parse_args()
dict_args = vars(args)

path_images = dict_args['path_images']
path_dataset = dict_args['path_dataset']
path_clusters = dict_args['path_clustering']

im =  os.listdir(path_images)

im = [x[:-4] for x in im]

if not Path(path_dataset+"/train").exists() :
    os.mkdir(path_dataset+"/train")
if not Path(path_dataset+"/validation").exists() :
    os.mkdir(path_dataset+"/validation")
if not Path(path_dataset+"/unlabeled").exists() :
    os.mkdir(path_dataset+"/unlabeled")

path_unlabeled = path_dataset+"/unlabeled" #all unlabeled images for training

def is_float(element: any) -> bool:
    #If you expect None to be passed:
    if element is None: 
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False
    
#To group by clusters
dico = {}
for dirs,_,files in os.walk(path_clusters) :
    if is_float(dirs[-2:].replace("_","")) :
        dico[dirs[-2:].replace("_","")] = [x[:-4] for x in files]
        

    
val=list(dico.values())


for i in range(len(im)) :
    
    #We put 20% of the images in the validation and 80% in the training
    coin = random.random()
    if coin <= 0.2 :
        path = path_dataset +"/validation"
    else :
        path = path_dataset +"/train"
        
        
    im_to_copy = im[i]
    
    if len([g for g in dico.keys() if im[i][:] in val[int(g)]]) > 0 :
        cl=[str(g) for g in dico.keys() if im[i][:] in val[int(g)]]
        
        if not Path(path+"/"+cl[-1]).exists() :
            os.mkdir(path+"/"+cl[-1])
            
        if not Path(path+"/"+cl[-1]+"/"+im_to_copy).exists():
            im_read = Image.open(path_images+"/"+im_to_copy+".jpg")
            im_read.save(path+"/"+cl[-1]+"/"+im_to_copy[:]+".jpg")
    else :
        print("Not found in clustering")
        print(im[i][:])
        cl="other"
        
    if not Path(path_unlabeled+"/"+cl[-1]).exists() :
        os.mkdir(path_unlabeled+"/"+cl[-1])
        
    if not Path(path_unlabeled+"/"+cl[-1]+"/"+im_to_copy).exists():
        im_read = Image.open(path_images+"/"+im_to_copy+".jpg")
        im_read.save(path_unlabeled+"/"+cl[-1]+"/"+im_to_copy[:]+".jpg")